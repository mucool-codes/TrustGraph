"""Explanation Layer 2: why an RSU's TRUST is what it is, from its own features and its neighbours.

GNNExplainer (torch_geometric.explain) is applied to the trust head only - never to the selection rule (L11).
Everything here follows DECISIONS.md D54 (frozen before any fidelity number existed):

  * a UNIT is one (node, feature) pair in the 2-hop receptive field of the explained RSU, using only valid
    feature slots (an RSU row's 6 RSU features, a vehicle row's 3 vehicle features); its importance is its
    GNNExplainer node-mask value;
  * the neighbour clause is CONDITIONAL: it is produced only when the same-segment RSU neighbours carry at
    least `NEIGHBOUR_SHARE_MIN` of the attribution, and the direction of a named neighbour's effect is
    MEASURED by occlusion (its RSU row replaced by the step's median RSU row), not inferred from the mask;
  * the perturbation used by the H4 fidelity test moves one scalar by `DELTA` toward the other end of [0, 1].

It reads the trust head and the step's graph, never the sealed ground truth (L4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

from .explain import Explanation, render_text
from .features import EDGE_COL, RSU_FEATURES, VEHICLE_FEATURES
from .selection import select

# ---- D54 constants (frozen) ------------------------------------------------------------------
EPOCHS = 100
HOPS = 2
DELTA = 0.3
TRUST_EPS = 0.05
NEIGHBOUR_SHARE_MIN = 0.30
NEIGHBOUR_NAMED_MIN = 0.05
NEIGHBOUR_MAX_NAMED = 2
OWN_SECOND_MIN = 0.25
DIRECTION_EPS = 0.01

NUM_RSU_FEATURES = len(RSU_FEATURES)
FEATURE_WORDS = {
    "load": "advertised load",
    "queue_depth": "queue depth",
    "cert_valid": "certificate validity",
    "success_ewma": "promise-keeping record",
    "latency_dev": "latency deviation",
    "uptime_stability": "uptime stability",
    "task_demand": "task demand",
    "speed": "speed",
    "dwell_estimate": "dwell estimate",
}


def feature_name(col: int) -> str:
    return RSU_FEATURES[col] if col < NUM_RSU_FEATURES else VEHICLE_FEATURES[col - NUM_RSU_FEATURES]


@dataclass(frozen=True)
class Unit:
    node: int
    col: int  # column of the node feature matrix
    importance: float
    value: float

    @property
    def feature(self) -> str:
        return feature_name(self.col)

    @property
    def key(self) -> tuple[int, int]:
        return (self.node, self.col)


@dataclass
class Attribution:
    """GNNExplainer's output for one explained RSU, reduced to units."""

    rsu: int
    run_seed: int
    units: list[Unit]  # every unit in the receptive field, importance descending (ties: node, col)
    mass_by_node: dict[int, float]
    total_mass: float

    def top(self, k: int = 1) -> list[Unit]:
        return self.units[:k]

    def low_pool(self) -> list[Unit]:
        """The bottom half by importance (D54 section 2)."""
        return self.units[len(self.units) // 2 + len(self.units) % 2 :]

    def share(self, node: int) -> float:
        return self.mass_by_node.get(node, 0.0) / self.total_mass if self.total_mass > 0 else 0.0


# ------------------------------------------------------------------------------- graph helpers


def receptive_field(edge_index: torch.Tensor, v: int, hops: int = HOPS) -> set[int]:
    seen = {int(v)}
    frontier = {int(v)}
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    adj: dict[int, list[int]] = {}
    for a, b in zip(src, dst):
        adj.setdefault(a, []).append(b)
    for _ in range(hops):
        frontier = {n for u in frontier for n in adj.get(u, ())} - seen
        seen |= frontier
    return seen


def valid_cols(node: int, num_rsus: int) -> range:
    return range(NUM_RSU_FEATURES) if node < num_rsus else range(NUM_RSU_FEATURES, NUM_RSU_FEATURES + len(VEHICLE_FEATURES))


def same_segment_rsu_neighbours(graph: Data, v: int, num_rsus: int) -> list[int]:
    src, dst = graph.edge_index
    same = graph.edge_attr[:, EDGE_COL["same_segment"]] > 0.5
    sel = (src == v) & (dst < num_rsus) & same
    return sorted(set(dst[sel].tolist()))


def active_median_row(graph: Data, num_rsus: int) -> torch.Tensor:
    """Median RSU feature row over the RSUs present at this step."""
    rows = graph.x[:num_rsus, :NUM_RSU_FEATURES]
    active = graph.rsu_active[:num_rsus] if hasattr(graph, "rsu_active") else torch.ones(num_rsus, dtype=torch.bool)
    return rows[active].median(dim=0).values


def flipped(value: float, delta: float = DELTA) -> float:
    return value + delta if value < 0.5 else value - delta


# -------------------------------------------------------------------------------- attribution


def attribute(
    model: torch.nn.Module, graph: Data, v: int, num_rsus: int, run_seed: int, decision_id: int,
    epochs: int = EPOCHS,
) -> Attribution:
    """Run GNNExplainer for RSU `v`'s trust and reduce the mask to units (D54 sections 1-2)."""
    has_edges = any(hasattr(m, "propagate") for m in model.modules())  # D has message passing, C does not
    explainer = Explainer(
        model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object" if has_edges else None,
        model_config=dict(mode="regression", task_level="node", return_type="raw"),
    )
    torch.manual_seed(run_seed * 1_000_003 + decision_id * 101 + v)
    exp = explainer(graph.x, graph.edge_index, edge_attr=graph.edge_attr, index=int(v))
    mask = exp.node_mask.detach()
    field_nodes = receptive_field(graph.edge_index, v)
    units: list[Unit] = []
    mass: dict[int, float] = {}
    for node in sorted(field_nodes):
        for col in valid_cols(node, num_rsus):
            imp = float(mask[node, col])
            units.append(Unit(node, col, imp, float(graph.x[node, col])))
            mass[node] = mass.get(node, 0.0) + imp
    units.sort(key=lambda u: (-u.importance, u.node, u.col))
    return Attribution(rsu=int(v), run_seed=run_seed, units=units, mass_by_node=mass, total_mass=sum(mass.values()))


# ------------------------------------------------------------------------ trust re-evaluation


@torch.no_grad()
def trust_vector(model: torch.nn.Module, graph: Data, num_rsus: int, x: torch.Tensor | None = None) -> np.ndarray:
    out = model(graph.x if x is None else x, graph.edge_index, graph.edge_attr)
    return out[:num_rsus].numpy().astype(np.float64)


def rerun_decision(record: dict, trust: np.ndarray) -> int:
    """The rule re-run on the logged latency/load/weights with a (perturbed) trust vector."""
    cands = [c for c in record["candidates"] if c["dwell_eligible"]]
    w = record["weights"]
    rsus = np.array([c["rsu"] for c in cands])
    d = select(
        timestep=record["step"], vehicle_id=record["vehicle"], candidates=rsus,
        trust=trust[rsus], latency=np.array([c["latency_norm"] for c in cands]),
        load=np.array([c["load"] for c in cands]), alpha=w["alpha"], beta=w["beta"], gamma=w["gamma"],
    )
    return int(d.chosen_rsu)


def perturb_effect(
    model: torch.nn.Module, graph: Data, record: dict, v: int, unit: Unit, num_rsus: int, base_trust: np.ndarray
) -> dict:
    """What moving one unit by DELTA does to v's trust and to the decision (D54 section 3)."""
    x = graph.x.clone()
    x[unit.node, unit.col] = flipped(float(x[unit.node, unit.col]))
    t2 = trust_vector(model, graph, num_rsus, x)
    delta = float(t2[v] - base_trust[v])
    return {
        "trust_delta": delta,
        "trust_changed": abs(delta) >= TRUST_EPS,
        "decision_changed": rerun_decision(record, t2) != rerun_decision(record, base_trust),
    }


# ----------------------------------------------------------------------- the node explanation


@dataclass
class OwnFactor:
    feature: str
    value: float
    importance: float
    share_of_own: float
    effect: float  # trust - trust with this feature set to the step's median RSU value


@dataclass
class NeighbourFactor:
    rsu: int
    share: float
    top_feature: str
    top_value: float
    effect: float  # trust - trust with this neighbour's RSU row set to the median row


@dataclass
class NodeTrustExplanation:
    rsu: int
    trust: float
    own: list[OwnFactor]
    neighbours: list[NeighbourFactor]  # same-segment RSU neighbours named (may be empty)
    own_share: float  # share of the attribution on v's own units
    same_segment_share: float
    neighbour_clause: bool
    same_segment_neighbours: list[int] = field(default_factory=list)


def explain_node(
    model: torch.nn.Module, graph: Data, v: int, num_rsus: int, run_seed: int, decision_id: int,
    attribution: Attribution | None = None,
) -> NodeTrustExplanation:
    attr = attribution or attribute(model, graph, v, num_rsus, run_seed, decision_id)
    base = trust_vector(model, graph, num_rsus)
    median = active_median_row(graph, num_rsus)

    own_units = sorted([u for u in attr.units if u.node == v], key=lambda u: (-u.importance, u.col))
    own_mass = sum(u.importance for u in own_units)
    own: list[OwnFactor] = []
    for i, u in enumerate(own_units[:2]):
        share = u.importance / own_mass if own_mass > 0 else 0.0
        if i == 1 and share < OWN_SECOND_MIN:
            break
        x = graph.x.clone()
        x[v, u.col] = median[u.col]
        eff = float(base[v] - trust_vector(model, graph, num_rsus, x)[v])
        own.append(OwnFactor(u.feature, u.value, u.importance, share, eff))

    nbrs = same_segment_rsu_neighbours(graph, v, num_rsus)
    nshare = sum(attr.share(n) for n in nbrs)
    clause = nshare >= NEIGHBOUR_SHARE_MIN
    named: list[NeighbourFactor] = []
    if clause:
        ranked = sorted((n for n in nbrs if attr.share(n) >= NEIGHBOUR_NAMED_MIN), key=lambda n: (-attr.share(n), n))
        for n in ranked[:NEIGHBOUR_MAX_NAMED]:
            top = max((u for u in attr.units if u.node == n), key=lambda u: (u.importance, -u.col))
            x = graph.x.clone()
            x[n, :NUM_RSU_FEATURES] = median
            eff = float(base[v] - trust_vector(model, graph, num_rsus, x)[v])
            named.append(NeighbourFactor(n, attr.share(n), top.feature, top.value, eff))
    return NodeTrustExplanation(
        rsu=int(v), trust=float(base[v]), own=own, neighbours=named, own_share=attr.share(v),
        same_segment_share=nshare, neighbour_clause=clause and bool(named), same_segment_neighbours=nbrs,
    )


# ---------------------------------------------------------------------------------- rendering


def _effect_words(effect: float) -> str:
    """effect = trust - trust with the typical (median) value: positive means the real value pulls trust up.
    Measured by occlusion, so it is stated as a measured change, not as a reason."""
    if effect >= DIRECTION_EPS:
        return f"raises trust by {effect:.2f} compared with a typical value"
    if effect <= -DIRECTION_EPS:
        return f"lowers trust by {-effect:.2f} compared with a typical value"
    return f"swapping it for a typical value changes trust by less than {DIRECTION_EPS:g}"


def node_clause(n: NodeTrustExplanation) -> str:
    """The Section 3.4 clause for one RSU: its trust, the evidence the attribution rests on, and - only if the
    attribution supports it - the same-segment neighbours."""
    parts = [f"RSU-{n.rsu} trust score {n.trust:.2f}"]
    if n.own:
        own = "; ".join(f"{FEATURE_WORDS[o.feature]} {o.value:.2f} ({_effect_words(o.effect)})" for o in n.own)
        parts.append(f"the attribution rests on its own {own}")
    if n.neighbour_clause:
        nb = "; ".join(
            f"RSU-{b.rsu}'s {FEATURE_WORDS[b.top_feature]} {b.top_value:.2f} ({_effect_words(b.effect)})"
            for b in n.neighbours
        )
        parts.append(
            f"same-backhaul-segment neighbours carry {n.same_segment_share:.0%} of it: {nb}"
        )
    else:
        parts.append(
            f"neighbouring RSUs did not drive this score (same-segment neighbours carry {n.same_segment_share:.0%} "
            f"of the attribution, below the {NEIGHBOUR_SHARE_MIN:.0%} needed to name them)"
        )
    return ", ".join(parts)


def render_merged(layer1: Explanation, layer2: dict[int, NodeTrustExplanation]) -> str:
    """Layer 1 (why this node won) + Layer 2 (why its trust is what it is), in the TECHNICAL_AND_NOVELTY.md
    3.4 structure: 'RSU-X selected over RSU-Y. Primary factors: ...'."""
    e = layer1
    lines = [render_text(e, layer2_note=False), ""]
    if e.runner_up is None:
        lines.append(f"Primary factors: only one eligible RSU. {node_clause(layer2[e.winner])}.")
    elif e.trust_mode == "unused":
        lines.append("Trust did not influence this decision, so no trust attribution is given.")
        return "\n".join(lines)
    else:
        win, lose = layer2[e.winner], layer2[e.runner_up.loser]
        lines.append(
            f"RSU-{e.winner} selected over RSU-{e.runner_up.loser}. Primary factors: {node_clause(win)}; "
            f"{node_clause(lose)}."
        )
    lines.append(
        "(Layer 2: GNNExplainer on the trust head. Own-evidence and neighbour effects are measured by replacing "
        "the evidence with the step's median RSU value; a neighbour is named only when same-segment neighbours "
        f"carry at least {NEIGHBOUR_SHARE_MIN:.0%} of the attribution.)"
    )
    return "\n".join(lines)

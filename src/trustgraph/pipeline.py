"""End-to-end scenario run.

Wires the pieces together: a mobility trace and an observable scenario, both read from
disk (L7, D23) -> dynamic graph sequence (L6) -> untrained trust head (L1) -> analytic
selection rule (L1/L2) -> printed decision sequence.

As of S2 the graph carries real advertised load and real discrepancy features, and the
vehicles that decide at each step are the ones that generated a task in the scenario.
There is still no training and no explanation. Note what the printed decisions are and
are not: each is what the L1 rule *with the untrained trust head* would pick for that
task. The scenario's own outcomes were produced by dispatching with the trust-agnostic
Baseline A rule (DECISIONS.md D33), so a printed decision is not the RSU that task
actually ran on. Closing that loop is the offloading engine's job (S6).
"""

from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .features import RSU_COL
from .model import build_trust_head
from .observed import ObservedScenario
from .scenario import World, build_snapshot_builder, build_world
from .selection import Decision, select
from .trace import Trace

# How many decision rows `format_decisions` prints in full. A scenario produces
# thousands; the exit condition needs the run to be inspectable and byte-reproducible,
# not exhaustively listed.
MAX_PRINTED_DECISIONS = 30


def run_pipeline(
    cfg: Config,
    trace: Trace,
    observed: ObservedScenario,
    world: World | None = None,
) -> list[Decision]:
    """Run the trust head and selection rule over the scenario's graph sequence.

    Deterministic: the model is seeded from the seed chain (DECISIONS.md D17), and the
    trace and scenario are read rather than regenerated.
    """
    world = world or build_world(cfg)
    topology = world.topology
    device = torch.device(cfg.device)

    model = build_trust_head(cfg.model, cfg.seeds.torch_seed("model_init"), cfg.device)
    builder = build_snapshot_builder(cfg, world, trace, observed)

    alpha = float(cfg.selection["alpha"])
    beta = float(cfg.selection["beta"])
    gamma = float(cfg.selection["gamma"])
    num_rsus = topology.num_rsus

    # Tasks are stored in generation order, so each step's offloaders are one slice.
    steps = observed.tasks.step
    bounds = np.searchsorted(steps, np.arange(trace.num_steps + 1))

    decisions: list[Decision] = []

    for data in builder.snapshots():
        t = int(data.timestep)
        offloaders = observed.tasks.vehicle[bounds[t] : bounds[t + 1]]
        if offloaders.size == 0:
            continue
        graph = data.to(device)

        with torch.no_grad():
            trust_all = model(graph.x, graph.edge_index).cpu().numpy()

        # Advertised load is read straight out of the graph's RSU block, so there is
        # no parallel array to drift out of sync with the features.
        load_all = graph.x[:num_rsus, RSU_COL["load"]].cpu().numpy()
        distances = graph.vehicle_rsu_distance.cpu().numpy()
        active = graph.rsu_active.cpu().numpy()
        # Latency is scaled onto the same [0, 1] range as trust and load so the
        # hand-tuned alpha/beta/gamma of L1 stay commensurable.
        latency = np.clip(
            graph.vehicle_rsu_latency_ms.cpu().numpy()
            / world.link_model.latency_norm_ms,
            0.0,
            1.0,
        )

        for vehicle in offloaders:
            in_range = np.flatnonzero(
                (distances[vehicle] <= topology.coverage_radius_m) & active
            )
            if in_range.size == 0:
                continue
            decision = select(
                timestep=t,
                vehicle_id=int(vehicle),
                candidates=in_range,
                trust=trust_all[in_range],
                latency=latency[vehicle][in_range],
                load=load_all[in_range],
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            if decision is not None:
                decisions.append(decision)

    return decisions


def format_decisions(cfg: Config, decisions: list[Decision], trace: Trace) -> str:
    """Render the decision sequence as fixed-width text.

    Floats are printed at fixed precision so two runs can be compared byte-for-byte -
    the reproducibility check Standing Rule 7 requires.
    """
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("TrustGraph - S2 (real advertised/discrepancy features, UNTRAINED model)")
    lines.append("=" * 78)
    lines.append(f"seed                 : {cfg.seed}")
    lines.append(f"device               : {cfg.device}")
    lines.append(f"RSUs                 : {cfg.topology['num_rsus']}")
    lines.append(f"backhaul segments    : {cfg.topology['num_backhaul_segments']}")
    lines.append(f"vehicles             : {trace.num_vehicles}")
    lines.append(
        f"steps                : {trace.num_steps} ({trace.duration_s:.0f} s "
        f"at dt={trace.dt_s:g}s)"
    )
    lines.append(f"mobility source      : {trace.source}")
    lines.append(
        f"degradation          : fraction={cfg.degraded_fraction:g} rho={cfg.rho:g}"
    )
    lines.append(
        "selection weights    : "
        f"alpha={cfg.selection['alpha']} beta={cfg.selection['beta']} "
        f"gamma={cfg.selection['gamma']}"
    )
    lines.append("")
    lines.append(
        "  t  veh  ->  RSU   score    trust    lat     load   "
        "| runner-up  margin  cands"
    )
    lines.append("-" * 78)

    for d in decisions[:MAX_PRINTED_DECISIONS]:
        runner = "-" if d.runner_up is None else f"RSU{d.runner_up:02d}"
        margin = (
            "     -"
            if d.runner_up_score is None
            else f"{d.score - d.runner_up_score:6.3f}"
        )
        lines.append(
            f"{d.timestep:3d} {d.vehicle_id:4d}  ->  RSU{d.chosen_rsu:02d} "
            f"{d.score:8.4f} {d.trust:8.4f} {d.latency:7.4f} {d.load:7.4f} "
            f"|   {runner:>6}  {margin}  {len(d.candidates):3d}"
        )

    if len(decisions) > MAX_PRINTED_DECISIONS:
        lines.append(
            f"... {len(decisions) - MAX_PRINTED_DECISIONS} further decisions not "
            "printed"
        )

    lines.append("-" * 78)
    lines.append(f"total decisions      : {len(decisions)}")
    if decisions:
        chosen = [d.chosen_rsu for d in decisions]
        lines.append(f"distinct RSUs chosen : {len(set(chosen))}")
        lines.append(
            "mean score           : "
            f"{sum(d.score for d in decisions) / len(decisions):.6f}"
        )
        lines.append(
            "mean candidates      : "
            f"{sum(len(d.candidates) for d in decisions) / len(decisions):.4f}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)

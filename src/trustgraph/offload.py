"""The offloading engine: candidate sets, trust sources, closed-loop dispatch, handoff, log.

S6. `simulator.generate_scenario(..., engine=engine)` runs the world with this engine making
the dispatch decisions (DECISIONS.md D51). With no engine the simulator is the open-loop
Baseline A of S2 and nothing here runs.

What happens at each step, in this order (the simulator drives it):

  1. `begin_step(t, ...)`  - the step's graph is built from the live arrays (the same graph the
     trust heads were trained on) and the frozen trust head is run once for every RSU;
  2. `dispatch(...)`       - per offloading vehicle: candidate set (in range and, optionally,
     dwell-feasible), then the L1 rule `alpha*trust - beta*latency - gamma*load`, argmax;
  3. `apply_handoff(...)`  - after execution: what happens if the vehicle has moved out of the
     serving RSU's coverage by the time the result is ready;
  4. `finish_decision(...)` - the outcome is written into the decision record.

Trust is a WEIGHTED TERM of the score. There is no threshold, cutoff or pre-filter on trust
anywhere in this module (L1) - that would turn trust into an admission gate.

This module never imports the sealed ground truth (L4). The one variant that needs it,
B_oracle (D50), receives its `cert_valid_oracle` vector from the simulator, which is world-side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .features import RSU_COL, RSU_FEATURES
from .graph import SnapshotBuilder, dwell_estimate_s
from .model import build_trust_head
from .selection import Decision, select
from .topology import pairwise_distances
from .train import load_checkpoint

VARIANTS: tuple[str, ...] = ("A", "B", "B_oracle", "C", "D", "E")
HANDOFF_MODES: tuple[str, ...] = ("relay", "off")
LOG_SCHEMA_VERSION = 1

# variant -> where its trust term comes from
TRUST_SOURCE = {
    "A": "none",
    "B": "cert_valid_feature",
    "B_oracle": "cert_valid_oracle",
    "C": "model_C",
    "D": "model_D",
    "E": "model_D",  # E = D + explanation; selection is identical to D (D50)
}


@dataclass(frozen=True)
class OffloadSettings:
    """Everything that selects a closed-loop run. A run is reproducible from (config, seed,
    these settings)."""

    variant: str = "A"
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.3
    dwell_filter: bool = True
    handoff: str = "relay"
    checkpoint_dir: str = "checkpoints"
    log_decisions: bool = False
    # Score every RSU at every step and keep the history (for plots); changes no outcome.
    record_trust: bool = False

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; expected one of {VARIANTS}")
        if self.handoff not in HANDOFF_MODES:
            raise ValueError(f"unknown handoff mode {self.handoff!r}; expected {HANDOFF_MODES}")
        if min(self.alpha, self.beta, self.gamma) < 0:
            raise ValueError("selection weights must be non-negative")

    @property
    def effective_alpha(self) -> float:
        """Variant A is the same rule with alpha = 0 (L2)."""
        return 0.0 if self.variant == "A" else self.alpha


def build_offload_settings(
    cfg: Config,
    variant: str | None = None,
    weights: tuple[float, float, float] | None = None,
    log_decisions: bool | None = None,
    **overrides,
) -> OffloadSettings:
    """Settings from the config's `offload` section (if any) and the `selection` weights."""
    section = dict(getattr(cfg, "offload", {}) or {})
    known = {"variant", "dwell_filter", "handoff", "checkpoint_dir"}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"unknown offload config keys: {sorted(unknown)}")
    a, b, g = (
        weights
        if weights is not None
        else (float(cfg.selection["alpha"]), float(cfg.selection["beta"]), float(cfg.selection["gamma"]))
    )
    kwargs = dict(
        variant=str(variant or section.get("variant", "A")),
        alpha=a,
        beta=b,
        gamma=g,
        dwell_filter=bool(section.get("dwell_filter", True)),
        handoff=str(section.get("handoff", "relay")),
        checkpoint_dir=str(section.get("checkpoint_dir", "checkpoints")),
    )
    if log_decisions is not None:
        kwargs["log_decisions"] = log_decisions
    kwargs.update(overrides)
    return OffloadSettings(**kwargs)


@dataclass
class EngineCounters:
    decisions: int = 0
    dwell_excluded: int = 0  # decisions where the filter removed at least one in-range RSU
    dwell_fallback: int = 0
    relayed: int = 0
    handoff_lost: int = 0


@dataclass
class OffloadEngine:
    """Closed-loop dispatch for one (config, seed, variant). Create one per run."""

    cfg: Config
    settings: OffloadSettings
    model: torch.nn.Module | None = field(default=None, repr=False)
    counters: EngineCounters = field(default_factory=EngineCounters)

    def __post_init__(self) -> None:
        self.log: list[dict] = []
        self.trust_history: list[np.ndarray] = []
        self._builder: SnapshotBuilder | None = None
        self._trust = np.zeros(0)
        self._t = -1
        if self.model is None and TRUST_SOURCE[self.settings.variant].startswith("model_"):
            self.model = load_frozen_head(self.cfg, self.settings)

    # ------------------------------------------------------------------ wiring

    def bind(self, world, trace, exec_model, rsu_features, rsu_active, task_demand) -> None:
        """Attach the live arrays the simulator is filling. The snapshot builder reads step t
        of each only after the simulator has written it."""
        self._world = world
        self._trace = trace
        self._exec = exec_model
        self._rsu_features = rsu_features
        self._rsu_active = rsu_active
        self._builder = SnapshotBuilder(
            topology=world.topology,
            trace=trace,
            link_model=world.link_model,
            cfg_graph=self.cfg.graph,
            rsu_features=rsu_features,
            rsu_active=rsu_active,
            vehicle_task_demand=task_demand,
        )
        self._num_rsus = world.topology.num_rsus

    def begin_step(self, t: int, need_trust: bool, oracle_cert_valid: np.ndarray) -> None:
        """Build step t's graph (every step: link_age is stateful) and score every RSU."""
        data = self._builder.build(t)
        self._t = t
        variant = self.settings.variant
        source = TRUST_SOURCE[variant]
        n = self._num_rsus
        if variant == "A" or not (need_trust or self.settings.record_trust):
            self._trust = np.zeros(n)
        elif source == "cert_valid_feature":
            self._trust = self._rsu_features[t, :, RSU_COL["cert_valid"]].astype(np.float64)
        elif source == "cert_valid_oracle":
            self._trust = np.asarray(oracle_cert_valid, dtype=np.float64)
        else:
            with torch.no_grad():
                out = self.model(data.x, data.edge_index, data.edge_attr)
            self._trust = out[:n].numpy().astype(np.float64)
        if self.settings.record_trust:
            self.trust_history.append(self._trust.copy())

    # ----------------------------------------------------------------- dispatch

    def dispatch(
        self,
        *,
        t: int,
        task_idx: int,
        vehicle: int,
        covered_row: np.ndarray,
        dist_row: np.ndarray,
        latency_ms_row: np.ndarray,
        latency_norm_row: np.ndarray,
        adv_load: np.ndarray,
        cycles: float,
        deadline_s: float,
        type_idx: int,
    ) -> tuple[Decision | None, dict | None]:
        """Candidate set and argmax for one task. Returns (decision, log record)."""
        s = self.settings
        topo = self._world.topology
        in_range = np.flatnonzero(covered_row)
        pos = self._trace.positions[t, vehicle]
        vel = self._trace.velocities[t, vehicle]

        dwell = np.full(in_range.size, np.nan)
        adv_s = np.full(in_range.size, np.nan)
        eligible = np.ones(in_range.size, dtype=bool)
        for i, r in enumerate(in_range):
            dwell[i] = dwell_estimate_s(pos, vel, topo.positions[r], topo.coverage_radius_m)
            comm = float(2.0 * latency_ms_row[r] / 1000.0)
            adv_s[i] = float(self._exec.advertised_completion_s(cycles, comm, adv_load[r]))
        fallback = False
        if s.dwell_filter and in_range.size:
            eligible = dwell >= adv_s
            if not eligible.any():
                eligible = np.ones(in_range.size, dtype=bool)
                fallback = True
        candidates = in_range[eligible]

        self.counters.decisions += 1
        self.counters.dwell_fallback += int(fallback)
        self.counters.dwell_excluded += int(bool((~eligible).any()))

        decision = None
        if candidates.size:
            decision = select(
                timestep=t,
                vehicle_id=vehicle,
                candidates=candidates,
                trust=self._trust[candidates],
                latency=latency_norm_row[candidates],
                load=adv_load[candidates],
                alpha=s.effective_alpha,
                beta=s.beta,
                gamma=s.gamma,
            )

        record = None
        if s.log_decisions:
            record = self._record(
                t, task_idx, vehicle, pos, vel, in_range, eligible, dwell, adv_s, dist_row,
                latency_ms_row, latency_norm_row, adv_load, cycles, deadline_s, type_idx,
                decision, fallback,
            )
            self.log.append(record)
        return decision, record

    def _record(
        self, t, task_idx, vehicle, pos, vel, in_range, eligible, dwell, adv_s, dist_row,
        latency_ms_row, latency_norm_row, adv_load, cycles, deadline_s, type_idx, decision, fallback,
    ) -> dict:
        s = self.settings
        alpha = s.effective_alpha
        topo = self._world.topology
        cand = []
        for i, r in enumerate(in_range):
            trust = float(self._trust[r]) if s.variant != "A" else None
            t_term = None if trust is None else alpha * trust
            l_term = -s.beta * float(latency_norm_row[r])
            g_term = -s.gamma * float(adv_load[r])
            cand.append(
                {
                    "rsu": int(r),
                    "segment": int(topo.backhaul_segment_id[r]),
                    "features": {
                        name: float(self._rsu_features[t, r, k]) for k, name in enumerate(RSU_FEATURES)
                    },
                    "distance_m": float(dist_row[r]),
                    "latency_ms": float(latency_ms_row[r]),
                    "latency_norm": float(latency_norm_row[r]),
                    "dwell_estimate_s": None if not np.isfinite(dwell[i]) else float(dwell[i]),
                    "advertised_completion_s": float(adv_s[i]),
                    "dwell_eligible": bool(eligible[i]),
                    "trust": trust,
                    "trust_term": t_term,
                    "latency_term": l_term,
                    "load_term": g_term,
                    "score": float((t_term or 0.0) + l_term + g_term),
                }
            )
        return {
            "schema": LOG_SCHEMA_VERSION,
            "decision_id": int(task_idx),
            "step": int(t),
            "time_s": float(t * self._trace.dt_s),
            "vehicle": int(vehicle),
            "vehicle_position_m": [float(pos[0]), float(pos[1])],
            "vehicle_velocity_mps": [float(vel[0]), float(vel[1])],
            "task_type": int(type_idx),
            "cycles": float(cycles),
            "deadline_s": float(deadline_s),
            "variant": s.variant,
            "trust_source": TRUST_SOURCE[s.variant],
            "weights": {"alpha": alpha, "beta": s.beta, "gamma": s.gamma},
            "dwell_filter": s.dwell_filter,
            "dwell_fallback": bool(fallback),
            "candidates": cand,
            "selected_rsu": None if decision is None else int(decision.chosen_rsu),
            "runner_up_rsu": None if decision is None or decision.runner_up is None else int(decision.runner_up),
            "outcome": None,
        }

    # ------------------------------------------------------------------ handoff

    def apply_handoff(
        self, t: int, vehicle: int, rsu: int, completion_s: float, deadline_s: float
    ) -> tuple[float, bool, bool]:
        """(completion, relayed, lost) after the vehicle's movement (D51 item 4).

        Only tasks that would otherwise meet their deadline are affected; one that is already
        late or dropped stays as it is.
        """
        if self.settings.handoff == "off" or not np.isfinite(completion_s) or completion_s > deadline_s:
            return completion_s, False, False
        topo = self._world.topology
        pos = self._trace.positions[t, vehicle] + self._trace.velocities[t, vehicle] * completion_s
        radius = topo.coverage_radius_m
        d_serving = float(np.linalg.norm(pos - topo.positions[rsu]))
        if d_serving <= radius:
            return completion_s, False, False
        d_all = pairwise_distances(pos[None, :], topo.positions)[0]
        ok = (d_all <= radius) & self._rsu_active[t]
        if not ok.any():
            self.counters.handoff_lost += 1
            return float("inf"), False, True
        other = int(np.argmin(np.where(ok, d_all, np.inf)))
        hop_m = float(np.linalg.norm(topo.positions[other] - topo.positions[rsu]))
        penalty_s = float(self._world.link_model.latency_ms(np.asarray(hop_m), is_backhaul=True)) / 1000.0
        self.counters.relayed += 1
        return completion_s + penalty_s, True, False

    def finish_decision(
        self, record: dict | None, advertised_s: float, observed_s: float, met: bool,
        relayed: bool,
    ) -> None:
        if record is not None:
            record["outcome"] = {
                "advertised_s": float(advertised_s),
                "observed_s": float(observed_s),
                "met_deadline": bool(met),
                "relayed": bool(relayed),
            }


def load_frozen_head(cfg: Config, settings: OffloadSettings) -> torch.nn.Module:
    """The seed's frozen checkpoint for the variant's trust head (C, or D for D/E)."""
    head = "C" if settings.variant == "C" else "D"
    path = Path(settings.checkpoint_dir) / f"s3-seed{cfg.seed}-{head}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"no frozen {head} checkpoint for seed {cfg.seed} at {path}; train it with "
            f"python scripts/s3_train.py --config <cfg> --seed {cfg.seed}"
        )
    model = build_trust_head(cfg.model, 0, cfg.device, head)
    blob = load_checkpoint(path, model)
    if blob["variant"] != head or blob["seed"] != cfg.seed:
        raise ValueError(f"{path} holds variant {blob['variant']} seed {blob['seed']}")
    return model

"""End-to-end scenario run.

Wires the pieces together: a mobility trace read from disk (L7) -> dynamic graph
sequence (L6) -> untrained trust head (L1) -> analytic selection rule (L1/L2) ->
printed decision sequence.

There is still no training, no degradation injection, and no explanation rendering -
those are later sessions. As of S1 the topology, mobility, geometry, and link
features are real; the behavioural features are the placeholders named in `graph.py`.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .features import RSU_COL
from .model import build_trust_head
from .scenario import World, build_snapshot_builder, build_world
from .selection import Decision, select
from .trace import Trace

# How many decision rows `format_decisions` prints in full. A five-minute scenario
# produces hundreds; the exit condition needs the run to be inspectable and
# byte-reproducible, not exhaustively listed.
MAX_PRINTED_DECISIONS = 30


def run_pipeline(cfg: Config, trace: Trace, world: World | None = None) -> list[Decision]:
    """Run the scenario over a trace and return the decision sequence.

    All randomness is drawn from per-purpose generators derived from `cfg.seed`
    (DECISIONS.md D17), and the trace is read rather than regenerated, so two runs
    with the same config and trace produce identical output.
    """
    world = world or build_world(cfg)
    topology = world.topology
    device = torch.device(cfg.device)

    model = build_trust_head(cfg.model, cfg.seeds.torch_seed("model_init"), cfg.device)
    builder = build_snapshot_builder(cfg, world, trace)
    task_rng = cfg.seeds.generator("tasks")

    tasks_per_step = int(cfg.scenario["tasks_per_step"])
    alpha = float(cfg.selection["alpha"])
    beta = float(cfg.selection["beta"])
    gamma = float(cfg.selection["gamma"])
    num_rsus = topology.num_rsus

    decisions: list[Decision] = []

    for data in builder.snapshots():
        t = int(data.timestep)
        graph = data.to(device)

        with torch.no_grad():
            trust_all = model(graph.x, graph.edge_index).cpu().numpy()

        # Advertised load is read straight out of the graph's RSU block, so there is
        # no parallel array to drift out of sync with the features.
        load_all = graph.x[:num_rsus, RSU_COL["load"]].cpu().numpy()
        distances = graph.vehicle_rsu_distance.cpu().numpy()
        # Latency is scaled onto the same [0, 1] range as trust and load so the
        # hand-tuned alpha/beta/gamma of L1 stay commensurable.
        latency = np.clip(
            graph.vehicle_rsu_latency_ms.cpu().numpy()
            / world.link_model.latency_norm_ms,
            0.0,
            1.0,
        )

        # Which vehicles offload this step. Sampling without replacement from a
        # dedicated stream keeps the choice independent of how many draws mobility or
        # feature generation happened to take.
        n_tasks = min(tasks_per_step, trace.num_vehicles)
        offloaders = np.sort(
            task_rng.choice(trace.num_vehicles, size=n_tasks, replace=False)
        )

        for vehicle in offloaders:
            in_range = np.flatnonzero(
                distances[vehicle] <= topology.coverage_radius_m
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
    lines.append("TrustGraph - S1 (real topology and mobility, UNTRAINED model)")
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

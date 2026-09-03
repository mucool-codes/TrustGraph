"""End-to-end walking skeleton.

Wires the pieces together: synthetic mobility (L7) -> dynamic graph (L6) -> untrained
trust head (L1) -> analytic selection rule (L1/L2) -> printed decision sequence.

S0 has no training, no degradation injection, and no explanation rendering. Every
feature value is fake. What is real is the shape of the pipeline and its determinism.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .features import RSU_COL
from .graph import build_snapshot
from .mobility import build_mobility
from .model import build_trust_head
from .selection import Decision, select
from .topology import build_topology


def run_pipeline(cfg: Config) -> list[Decision]:
    """Run the scenario and return the decision sequence.

    All randomness is drawn from per-purpose generators derived from `cfg.seed`
    (DECISIONS.md D17), so two runs with the same config produce identical output.
    """
    seeds = cfg.seeds
    device = torch.device(cfg.device)

    topology = build_topology(cfg.topology, seeds.generator("topology"))
    mobility = build_mobility(
        {**cfg.mobility, "num_vehicles": cfg.mobility["num_vehicles"]},
        seeds.generator("mobility"),
    )
    model = build_trust_head(cfg.model, seeds.torch_seed("model_init"), cfg.device)

    feature_rng = seeds.generator("features")
    task_rng = seeds.generator("tasks")

    num_steps = int(cfg.scenario["num_steps"])
    tasks_per_step = int(cfg.scenario["tasks_per_step"])
    alpha = float(cfg.selection["alpha"])
    beta = float(cfg.selection["beta"])
    gamma = float(cfg.selection["gamma"])

    decisions: list[Decision] = []
    positions = mobility.reset()

    for t in range(num_steps):
        if t > 0:
            positions = mobility.step()

        data = build_snapshot(topology, positions, feature_rng)
        data = data.to(device)

        with torch.no_grad():
            trust_all = model(data.x, data.edge_index).cpu().numpy()

        num_rsus = topology.num_rsus
        # Advertised load is read straight out of the graph's RSU block, so there is
        # no parallel array to drift out of sync with the features.
        load_all = data.x[:num_rsus, RSU_COL["load"]].cpu().numpy()
        distances = data.vehicle_rsu_distance.cpu().numpy()

        # Which vehicles offload this step. Sampling without replacement from a
        # dedicated stream keeps the choice independent of how many draws mobility or
        # feature generation happened to take.
        n_tasks = min(tasks_per_step, mobility.num_vehicles)
        offloaders = np.sort(
            task_rng.choice(mobility.num_vehicles, size=n_tasks, replace=False)
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
                # Latency proxy: normalised distance. A real latency model arrives
                # with the real features.
                latency=distances[vehicle][in_range] / topology.coverage_radius_m,
                load=load_all[in_range],
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            if decision is not None:
                decisions.append(decision)

    return decisions


def format_decisions(cfg: Config, decisions: list[Decision]) -> str:
    """Render the decision sequence as fixed-width text.

    Floats are printed at fixed precision so two runs can be compared byte-for-byte -
    that is the S0 exit condition.
    """
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("TrustGraph - S0 walking skeleton (fake features, UNTRAINED model)")
    lines.append("=" * 78)
    lines.append(f"seed                 : {cfg.seed}")
    lines.append(f"device               : {cfg.device}")
    lines.append(f"RSUs                 : {cfg.topology['num_rsus']}")
    lines.append(f"backhaul segments    : {cfg.topology['num_backhaul_segments']}")
    lines.append(f"vehicles             : {cfg.mobility['num_vehicles']}")
    lines.append(f"steps                : {cfg.scenario['num_steps']}")
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

    for d in decisions:
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

    lines.append("-" * 78)
    lines.append(f"total decisions      : {len(decisions)}")
    if decisions:
        chosen = [d.chosen_rsu for d in decisions]
        lines.append(f"distinct RSUs chosen : {len(set(chosen))}")
        lines.append(
            "mean score           : "
            f"{sum(d.score for d in decisions) / len(decisions):.6f}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)

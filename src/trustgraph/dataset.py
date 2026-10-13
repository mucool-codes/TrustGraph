"""Training data for the trust heads (L3): graphs plus next-task outcome labels.

Everything here is built from the OBSERVABLE scenario file and the mobility trace. This
module is on the training path and must never import the sealed ground truth (L4;
`tests/test_sealing.py` enforces it).

The L3 target. For RSU `v` at step `t`, the label is whether the next task dispatched to
`v` at step `t` or later met its deadline (`TaskStream.met_deadline`). The feature row
at `t` holds only outcomes observed at or before `t * dt` (observed.py), and that
task's outcome becomes known after it is dispatched, so the label is strictly in the
feature row's future. RSUs that are not yet active at `t` (cold start) and steps after
an RSU's last task have no label and are masked out of every loss.

Under Baseline A dispatch (D33, D41 Q2) an RSU receives tasks regardless of its trust,
so a degraded RSU keeps producing labels after onset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .config import Config
from .observed import ObservedScenario, default_observed_path, load_observed
from .scenario import build_snapshot_builder, build_world
from .trace import default_trace_path, load_trace


def next_task_labels(observed: ObservedScenario) -> tuple[np.ndarray, np.ndarray]:
    """(y, valid), both (num_steps, num_rsus).

    `y[t, v]` is 1.0 if the next task dispatched to RSU `v` at step >= `t` met its
    deadline, else 0.0; `valid[t, v]` is False where there is no such task or where `v`
    is not active at `t`.
    """
    tasks = observed.tasks
    num_steps, num_rsus = observed.rsu_active.shape
    y = np.zeros((num_steps, num_rsus), dtype=np.float32)
    valid = np.zeros((num_steps, num_rsus), dtype=bool)
    t_axis = np.arange(num_steps)
    for v in range(num_rsus):
        mine = np.flatnonzero(tasks.rsu == v)  # generation order = step order
        if mine.size == 0:
            continue
        steps_v = tasks.step[mine]
        nxt = np.searchsorted(steps_v, t_axis, side="left")
        has = nxt < mine.size
        y[has, v] = tasks.met_deadline[mine[nxt[has]]].astype(np.float32)
        valid[:, v] = has
    valid &= observed.rsu_active
    return y, valid


@dataclass(frozen=True)
class GraphSet:
    """Cached graphs of one scenario at a stride, with labels.

    Each `Data` carries `x`, `edge_index`, `edge_attr`, and per-node `y` / `mask` of
    length `num_nodes` (RSU rows first, vehicle rows always masked out) so PyG batching
    keeps them aligned with the node ordering. `steps[i]` is the timestep of
    `graphs[i]`.
    """

    graphs: list[Data]
    steps: np.ndarray
    num_rsus: int
    name: str

    def __len__(self) -> int:
        return len(self.graphs)


def build_graphset(
    cfg: Config,
    config_path: str | Path,
    stride: int = 4,
    step_range: tuple[int, int] | None = None,
) -> GraphSet:
    """Load the observable scenario and trace for `cfg` and cache every `stride`-th
    graph. The snapshot builder is stateful (D25), so it is stepped through every
    timestep; only the retained graphs are kept."""
    observed_path = default_observed_path(config_path, cfg)
    observed = load_observed(observed_path)
    trace = load_trace(default_trace_path(config_path, cfg.seed))
    world = build_world(cfg)
    builder = build_snapshot_builder(cfg, world, trace, observed)
    y_all, valid_all = next_task_labels(observed)
    num_rsus = world.topology.num_rsus

    lo, hi = (0, trace.num_steps) if step_range is None else step_range
    graphs: list[Data] = []
    steps: list[int] = []
    for t in range(hi):
        data = builder.build(t)
        if t < lo or t % stride != 0:
            continue
        n = int(data.num_nodes)
        y = torch.zeros(n)
        mask = torch.zeros(n, dtype=torch.bool)
        y[:num_rsus] = torch.from_numpy(y_all[t])
        mask[:num_rsus] = torch.from_numpy(valid_all[t])
        graphs.append(
            Data(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                y=y,
                mask=mask,
                num_nodes=n,
            )
        )
        steps.append(t)
    return GraphSet(
        graphs=graphs,
        steps=np.asarray(steps, dtype=np.int64),
        num_rsus=num_rsus,
        name=observed_path.name.removesuffix(".observed.npz"),
    )

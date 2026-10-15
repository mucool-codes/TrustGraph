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
from .scenario import World, build_snapshot_builder, build_world
from .trace import Trace, default_trace_path, load_trace


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


# S15 (D72): the forward-looking objective. Fixed in D72 before any number existed.
HORIZON_STEPS = 150
HORIZON_THRESHOLD = 0.75
HORIZON_MIN_TASKS = 5
OBJECTIVES = ("next", "horizon")


def horizon_labels(
    observed: ObservedScenario,
    horizon: int = HORIZON_STEPS,
    threshold: float = HORIZON_THRESHOLD,
    min_tasks: int = HORIZON_MIN_TASKS,
) -> tuple[np.ndarray, np.ndarray]:
    """(z, valid), both (num_steps, num_rsus): the D72 horizon objective.

    For RSU `v` at step `t`, f is the fraction of the tasks dispatched to `v` at steps in
    (t, t + horizon] (truncated at the last step) that met their deadline. D72 defines
    y = 1 when f < threshold ("about to fail"); the models' output is trust (high =
    healthy), so the target returned here is z = 1 - y, 1.0 where f >= threshold. BCE of
    trust against z is the same loss as BCE of (1 - trust) against y. `valid` is False
    where `v` has fewer than `min_tasks` tasks in the window or is not active at `t`.

    Built from the observable task stream only (L4); the sealed class is never read.
    """
    tasks = observed.tasks
    num_steps, num_rsus = observed.rsu_active.shape
    z = np.zeros((num_steps, num_rsus), dtype=np.float32)
    valid = np.zeros((num_steps, num_rsus), dtype=bool)
    t_axis = np.arange(num_steps)
    for v in range(num_rsus):
        mine = np.flatnonzero(tasks.rsu == v)  # generation order = step order
        if mine.size == 0:
            continue
        steps_v = tasks.step[mine]
        cum_met = np.concatenate([[0], np.cumsum(tasks.met_deadline[mine].astype(np.int64))])
        lo = np.searchsorted(steps_v, t_axis, side="right")  # first task at step >= t + 1
        hi = np.searchsorted(steps_v, t_axis + horizon, side="right")  # one past step t + horizon
        count = hi - lo
        frac = (cum_met[hi] - cum_met[lo]) / np.maximum(count, 1)
        ok = count >= min_tasks
        z[ok, v] = (frac[ok] >= threshold).astype(np.float32)
        valid[:, v] = ok
    valid &= observed.rsu_active
    return z, valid


def labels_for(observed: ObservedScenario, objective: str = "next") -> tuple[np.ndarray, np.ndarray]:
    """The (target, valid) pair of the named training objective (`next` is L3, `horizon` is D72)."""
    if objective == "next":
        return next_task_labels(observed)
    if objective == "horizon":
        return horizon_labels(observed)
    raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")


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
    objective: str = "next",
) -> GraphSet:
    """Load the observable scenario and trace for `cfg` from disk and cache every
    `stride`-th graph."""
    observed_path = default_observed_path(config_path, cfg)
    observed = load_observed(observed_path)
    trace = load_trace(default_trace_path(config_path, cfg.seed))
    return graphset_from(
        cfg,
        build_world(cfg),
        trace,
        observed,
        stride=stride,
        step_range=step_range,
        name=observed_path.name.removesuffix(".observed.npz"),
        objective=objective,
    )


def graphset_from(
    cfg: Config,
    world: World,
    trace: Trace,
    observed: ObservedScenario,
    stride: int = 4,
    step_range: tuple[int, int] | None = None,
    name: str = "",
    objective: str = "next",
) -> GraphSet:
    """Cache every `stride`-th graph of an in-memory (world, trace, observable) triple.

    The snapshot builder is stateful (D25), so it is stepped through every timestep;
    only the retained graphs are kept.
    """
    builder = build_snapshot_builder(cfg, world, trace, observed)
    y_all, valid_all = labels_for(observed, objective)
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
        name=name,
    )

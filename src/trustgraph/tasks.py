"""The task model: what a vehicle offloads, how heavy it is, and how long it may take.

Three task types - light, medium, heavy - each with a fixed compute-cycle requirement.
A task's deadline is correlated with its type: it is a communication allowance plus a
per-task *slack* multiple of the time the task would take on an idle reference node,
capped at `deadline_cap_s`. So a heavier task gets a proportionally longer deadline,
but not an unlimited one - past the cap, a heavy task has less slack than a light one,
which is the physical situation (a perception result that arrives after the vehicle
has passed the hazard is worthless however expensive it was to compute).

Everything here is *observable*: the vehicle knows the type, the cycle count, and the
deadline of the task it is offloading. Nothing in this module knows anything about
node behaviour, and it must never import the sealed ground truth (L4).

Draws are taken in fixed-size blocks per timestep - one uniform per vehicle for
arrival, type, and slack, whether or not the vehicle offloads - so the task stream for
a seed is identical across degradation settings that differ only in which nodes
misbehave (DECISIONS.md D17).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

TASK_TYPES: tuple[str, ...] = ("light", "medium", "heavy")


@dataclass(frozen=True)
class TaskDraws:
    """One timestep's task arrivals, aligned with vehicle index.

    Attributes:
        offloads: (num_vehicles,) bool - the vehicle generates a task this step.
        type_idx: (num_vehicles,) int index into `TASK_TYPES`. Defined for every
            vehicle so the draw count is fixed; read only where `offloads` is true.
        slack: (num_vehicles,) deadline slack multiple, same convention.
    """

    offloads: np.ndarray
    type_idx: np.ndarray
    slack: np.ndarray


@dataclass(frozen=True)
class TaskModel:
    """Task types, the arrival process, and the deadline rule.

    Defaults describe tasks sized for a ~10 GHz roadside edge server: a light task
    (hazard classification on a cropped frame) is 20 ms of compute on an idle node, a
    medium one (short-horizon path re-plan) 100 ms, a heavy one (cooperative perception
    fusion) 300 ms.
    """

    rate_per_vehicle_hz: float = 0.1
    cycles: tuple[float, ...] = (2.0e8, 1.0e9, 3.0e9)
    shares: tuple[float, ...] = (0.5, 0.35, 0.15)
    # The node speed deadlines are expressed against. A vehicle sets its deadline from
    # the task it has, not from the node it will get, so this is a fixed reference and
    # not any particular RSU's capacity.
    reference_capacity_cycles_per_s: float = 1.0e10
    comm_allowance_s: float = 0.05
    slack_min: float = 2.0
    slack_max: float = 4.0
    deadline_cap_s: float = 1.0

    def __post_init__(self) -> None:
        if len(self.cycles) != len(TASK_TYPES) or len(self.shares) != len(TASK_TYPES):
            raise ValueError(
                f"tasks.cycles and tasks.shares need one entry per type {TASK_TYPES}"
            )
        if any(c <= 0 for c in self.cycles):
            raise ValueError("tasks.cycles must be positive")
        if list(self.cycles) != sorted(self.cycles):
            raise ValueError("tasks.cycles must be non-decreasing light -> heavy")
        if any(s < 0 for s in self.shares) or not np.isclose(sum(self.shares), 1.0):
            raise ValueError("tasks.shares must be non-negative and sum to 1")
        if self.rate_per_vehicle_hz < 0:
            raise ValueError("tasks.rate_per_vehicle_hz must be non-negative")
        if not 0 < self.slack_min <= self.slack_max:
            raise ValueError("require 0 < tasks.slack_min <= tasks.slack_max")
        if self.deadline_cap_s <= self.comm_allowance_s:
            raise ValueError("tasks.deadline_cap_s must exceed comm_allowance_s")

    @property
    def max_cycles(self) -> float:
        return float(max(self.cycles))

    def cycles_of(self, type_idx: np.ndarray) -> np.ndarray:
        return np.asarray(self.cycles, dtype=np.float64)[type_idx]

    def deadline_s(self, type_idx: np.ndarray, slack: np.ndarray) -> np.ndarray:
        """comm_allowance + slack * (idle reference compute time), capped."""
        nominal_s = self.cycles_of(type_idx) / self.reference_capacity_cycles_per_s
        return np.minimum(
            self.deadline_cap_s, self.comm_allowance_s + np.asarray(slack) * nominal_s
        )

    def draw_step(
        self, num_vehicles: int, dt_s: float, rng: np.random.Generator
    ) -> TaskDraws:
        """Draw one timestep of arrivals.

        Arrivals are a Poisson process per vehicle thinned to at most one task per
        step; at the default 0.1 Hz and dt = 1 s the chance of a second arrival in the
        same step is under 0.5%, so the thinning is immaterial.
        """
        u_arrival = rng.random(num_vehicles)
        u_type = rng.random(num_vehicles)
        u_slack = rng.random(num_vehicles)

        p_arrival = 1.0 - np.exp(-self.rate_per_vehicle_hz * dt_s)
        bounds = np.cumsum(self.shares)
        type_idx = np.minimum(
            np.searchsorted(bounds, u_type, side="right"), len(TASK_TYPES) - 1
        )
        return TaskDraws(
            offloads=u_arrival < p_arrival,
            type_idx=type_idx.astype(np.int64),
            slack=self.slack_min + u_slack * (self.slack_max - self.slack_min),
        )


def build_task_model(cfg_tasks: dict | None) -> TaskModel:
    """Construct the task model from the config, falling back to the defaults."""
    if not cfg_tasks:
        return TaskModel()
    known = {f.name for f in fields(TaskModel)}
    unknown = set(cfg_tasks) - known
    if unknown:
        raise ValueError(f"unknown tasks config keys: {sorted(unknown)}")
    kwargs: dict = {}
    for key, value in cfg_tasks.items():
        if key in ("cycles", "shares"):
            kwargs[key] = tuple(float(v) for v in value)
        else:
            kwargs[key] = float(value)
    return TaskModel(**kwargs)

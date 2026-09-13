"""The node execution model: how long a task really takes on a node, and whether it lands.

Given a task assigned to an RSU with some true load and some true condition, this
decides the completion time and the outcome. It is the world, not an observer - but it
takes the node's condition as plain numbers (a capacity factor and a drop probability)
rather than as a behaviour class, so it has no reason to import the sealed ground truth
and does not. The simulator translates the sealed state into those numbers.

The model, deliberately closed-form:

    completion = comm + cycles / (C * capacity_factor * (1 - kappa * load)) * noise

  * `comm` is the round trip over the access link: twice the one-way latency from
    `links.py`. Payload transfer time is not modelled separately.
  * `C` is the node's nominal compute capacity in cycles per second.
  * `load` in [0, 1] is the node's true background utilisation, and `kappa < 1` is how
    much of the capacity that utilisation takes away. A fully loaded node still makes
    progress, just slowly - the shape of an M/M/1 slowdown without its pole at 1.
  * `capacity_factor` is 1 for a healthy node and falls toward `1 - severity` as a
    degraded node's fault ramps in.
  * `noise` is a mean-one lognormal, so a node's realised time scatters around what
    an honest estimate of it would predict.

A degraded node also silently drops a task with probability `drop_prob`, in which case
it never completes and the vehicle sees a timeout.

True load is coverage demand, the same quantity S1 computed in `graph.py`. It does not
respond to the tasks dispatched to the node: at the task rates used here offloaded
compute is a small fraction of each node's capacity, and making load a function of the
dispatch policy would tie the world's behaviour to the selection rule under test
(DECISIONS.md D32).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


def coverage_load(
    in_range_count: np.ndarray, capacity_vehicles: float
) -> tuple[np.ndarray, np.ndarray]:
    """True (load, queue_depth) from coverage demand.

    Carried over from S1 unchanged in meaning: `load` saturates at capacity and
    `queue_depth` is only the excess beyond it, so a fully loaded node with nothing
    queued and one with a backlog stay distinguishable.
    """
    demand = np.asarray(in_range_count, dtype=np.float64) / max(capacity_vehicles, 1e-9)
    return np.clip(demand, 0.0, 1.0), np.clip(demand - 1.0, 0.0, 1.0)


def round_trip_comm_s(one_way_latency_ms: np.ndarray) -> np.ndarray:
    """Uplink plus downlink over the access link, in seconds."""
    return 2.0 * np.asarray(one_way_latency_ms, dtype=np.float64) / 1000.0


@dataclass(frozen=True)
class ExecutionModel:
    """Compute capacity, load sensitivity, and runtime noise for an RSU."""

    capacity_cycles_per_s: float = 1.0e10
    rsu_capacity_vehicles: float = 10.0
    load_sensitivity: float = 0.6
    noise_sigma: float = 0.25

    def __post_init__(self) -> None:
        if self.capacity_cycles_per_s <= 0:
            raise ValueError("execution.capacity_cycles_per_s must be positive")
        if self.rsu_capacity_vehicles <= 0:
            raise ValueError("execution.rsu_capacity_vehicles must be positive")
        if not 0.0 <= self.load_sensitivity < 1.0:
            raise ValueError("execution.load_sensitivity must lie in [0, 1)")
        if self.noise_sigma < 0:
            raise ValueError("execution.noise_sigma must be non-negative")

    def service_time_s(
        self,
        cycles: np.ndarray,
        load: np.ndarray,
        capacity_factor: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        """Deterministic compute time for `cycles` at `load`, before noise."""
        rate = (
            self.capacity_cycles_per_s
            * np.asarray(capacity_factor, dtype=np.float64)
            * (1.0 - self.load_sensitivity * np.asarray(load, dtype=np.float64))
        )
        return np.asarray(cycles, dtype=np.float64) / rate

    def advertised_completion_s(
        self, cycles: np.ndarray, comm_s: np.ndarray, advertised_load: np.ndarray
    ) -> np.ndarray:
        """What a node's advertisement promises for this task.

        A node advertises its load; the expected completion follows from the nominal
        model at that load plus the vehicle's own measured link latency. A degraded
        node does not know it is degraded, so it advertises *nominal* capacity. A
        colluding node advertises a load below its true one. In both cases the promise
        is optimistic, and that optimism is what L8 measures.
        """
        return np.asarray(comm_s, dtype=np.float64) + self.service_time_s(
            cycles, advertised_load
        )

    def true_completion_s(
        self,
        cycles: np.ndarray,
        comm_s: np.ndarray,
        true_load: np.ndarray,
        capacity_factor: np.ndarray,
        drop_prob: np.ndarray,
        noise_z: np.ndarray,
        drop_u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Realised completion time and whether the task was silently dropped.

        `noise_z` is a standard normal and `drop_u` a uniform, both drawn by the
        caller once per task regardless of outcome, so the draw count never depends
        on the node's condition. A dropped task has completion `inf`.
        """
        sigma = self.noise_sigma
        noise = np.exp(sigma * np.asarray(noise_z) - 0.5 * sigma * sigma)
        completion = np.asarray(comm_s, dtype=np.float64) + self.service_time_s(
            cycles, true_load, capacity_factor
        ) * noise
        dropped = np.asarray(drop_u) < np.asarray(drop_prob)
        return np.where(dropped, np.inf, completion), dropped


def build_execution_model(cfg_execution: dict | None) -> ExecutionModel:
    """Construct the execution model from the config, falling back to the defaults."""
    if not cfg_execution:
        return ExecutionModel()
    known = {f.name for f in fields(ExecutionModel)}
    unknown = set(cfg_execution) - known
    if unknown:
        raise ValueError(f"unknown execution config keys: {sorted(unknown)}")
    return ExecutionModel(**{k: float(v) for k, v in cfg_execution.items()})

"""Scenario statistics from a trace and a topology.

These are the numbers that say whether the scenario is a sane vehicular one before any
model is trained on it: is anything in coverage, does anyone hand off, does a vehicle
stay with an RSU long enough for a task to complete. Getting them wrong here would
quietly invalidate everything downstream, so they are computed once, from the trace on
disk, and reported by `scripts/s1_report.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .topology import Topology, pairwise_distances
from .trace import Trace


@dataclass(frozen=True)
class CoverageSeries:
    """Per-timestep coverage, the raw material for every statistic below.

    Attributes:
        serving: (num_steps, num_vehicles) index of each vehicle's nearest in-range
            RSU, or -1 when it has no coverage. Ties break to the lowest index, the
            same rule `graph.SnapshotBuilder` uses.
        in_range_count: (num_steps, num_rsus) vehicles inside each RSU's radius.
        edge_count: (num_steps,) total vehicle-RSU links (undirected).
        candidates: (num_steps, num_vehicles) RSUs in range of each vehicle.
    """

    serving: np.ndarray
    in_range_count: np.ndarray
    edge_count: np.ndarray
    candidates: np.ndarray
    dt_s: float


def coverage_series(trace: Trace, topology: Topology) -> CoverageSeries:
    """Walk the trace and record who could see whom at each timestep."""
    num_steps, num_vehicles = trace.num_steps, trace.num_vehicles
    num_rsus = topology.num_rsus

    serving = np.full((num_steps, num_vehicles), -1, dtype=np.int64)
    in_range_count = np.zeros((num_steps, num_rsus), dtype=np.int64)
    candidates = np.zeros((num_steps, num_vehicles), dtype=np.int64)

    for t in range(num_steps):
        dist = pairwise_distances(trace.positions[t], topology.positions)
        covered = dist <= topology.coverage_radius_m
        in_range_count[t] = covered.sum(axis=0)
        candidates[t] = covered.sum(axis=1)
        masked = np.where(covered, dist, np.inf)
        nearest = np.argmin(masked, axis=1)
        serving[t] = np.where(covered.any(axis=1), nearest, -1)

    return CoverageSeries(
        serving=serving,
        in_range_count=in_range_count,
        edge_count=in_range_count.sum(axis=1),
        candidates=candidates,
        dt_s=trace.dt_s,
    )


def dwell_times_s(series: CoverageSeries) -> np.ndarray:
    """Durations of every completed stay with one serving RSU, in seconds.

    A dwell is a maximal run of consecutive timesteps for which a vehicle's serving
    RSU is unchanged. Runs touching either end of the trace are dropped: they are
    censored - the vehicle was already there when recording began, or still there when
    it stopped - and keeping them would drag the mean down by exactly the amount the
    horizon happens to truncate.
    """
    num_steps, num_vehicles = series.serving.shape
    durations: list[float] = []

    for v in range(num_vehicles):
        column = series.serving[:, v]
        boundaries = np.flatnonzero(np.diff(column) != 0) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [num_steps]])
        for start, end in zip(starts, ends):
            if column[start] < 0:
                continue  # out of coverage entirely
            if start == 0 or end == num_steps:
                continue  # censored by the recording window
            durations.append((end - start) * series.dt_s)

    return np.asarray(durations, dtype=np.float64)


def handoff_counts(series: CoverageSeries) -> tuple[np.ndarray, np.ndarray]:
    """Per-vehicle (handoffs, coverage gaps).

    A handoff is a change of serving RSU between two consecutive timesteps where both
    the old and the new RSU exist. A transition to or from -1 is a coverage gap, not a
    handoff - counting those as handoffs would let a scenario with poor coverage
    masquerade as one with lively mobility.
    """
    previous = series.serving[:-1]
    current = series.serving[1:]
    changed = current != previous
    both_covered = (previous >= 0) & (current >= 0)

    handoffs = (changed & both_covered).sum(axis=0)
    gaps = (changed & ~both_covered).sum(axis=0)
    return handoffs.astype(np.int64), gaps.astype(np.int64)


@dataclass(frozen=True)
class ScenarioStats:
    """The summary the S1 exit condition asks for, plus the context to read it."""

    num_steps: int
    duration_s: float
    num_vehicles: int
    num_rsus: int
    segment_sizes: tuple[int, ...]
    num_swapped_rsus: int
    num_rsu_rsu_edges: int
    mean_vehicles_per_rsu: float
    std_vehicles_per_rsu: float
    mean_dwell_s: float
    median_dwell_s: float
    std_dwell_s: float
    num_dwell_episodes: int
    handoffs_per_vehicle_per_min: float
    total_handoffs: int
    total_coverage_gaps: int
    coverage_fraction: float
    mean_candidates_in_range: float
    mean_speed_mps: float
    std_speed_mps: float
    mean_vehicle_rsu_edges: float


def scenario_stats(trace: Trace, topology: Topology) -> ScenarioStats:
    """Compute every summary statistic for one (trace, topology) pair."""
    series = coverage_series(trace, topology)
    dwell = dwell_times_s(series)
    handoffs, gaps = handoff_counts(series)

    minutes = trace.duration_s / 60.0
    speeds = trace.speeds

    return ScenarioStats(
        num_steps=trace.num_steps,
        duration_s=trace.duration_s,
        num_vehicles=trace.num_vehicles,
        num_rsus=topology.num_rsus,
        segment_sizes=tuple(int(s) for s in topology.segment_sizes()),
        num_swapped_rsus=int(topology.num_swapped_rsus),
        num_rsu_rsu_edges=int(topology.rsu_edges.shape[0]),
        mean_vehicles_per_rsu=float(series.in_range_count.mean()),
        std_vehicles_per_rsu=float(series.in_range_count.std()),
        mean_dwell_s=float(dwell.mean()) if dwell.size else float("nan"),
        median_dwell_s=float(np.median(dwell)) if dwell.size else float("nan"),
        std_dwell_s=float(dwell.std()) if dwell.size else float("nan"),
        num_dwell_episodes=int(dwell.size),
        handoffs_per_vehicle_per_min=(
            float(handoffs.sum()) / trace.num_vehicles / minutes if minutes else 0.0
        ),
        total_handoffs=int(handoffs.sum()),
        total_coverage_gaps=int(gaps.sum()),
        coverage_fraction=float((series.serving >= 0).mean()),
        mean_candidates_in_range=float(series.candidates.mean()),
        mean_speed_mps=float(speeds.mean()),
        std_speed_mps=float(speeds.std()),
        mean_vehicle_rsu_edges=float(series.edge_count.mean()),
    )


def format_stats(stats: ScenarioStats) -> str:
    """Render the summary as fixed-width text, for stdout and for FINDINGS.md."""
    lines = [
        "scenario",
        f"  steps                      : {stats.num_steps} "
        f"({stats.duration_s:.0f} s)",
        f"  vehicles                   : {stats.num_vehicles}",
        f"  RSUs                       : {stats.num_rsus}",
        f"  backhaul segment sizes     : {list(stats.segment_sizes)}",
        f"  RSUs off their geo segment : {stats.num_swapped_rsus} "
        f"(not visible in the topology figure - DECISIONS.md D22)",
        f"  RSU-RSU edges (undirected) : {stats.num_rsu_rsu_edges}",
        "",
        "mobility",
        f"  mean speed                 : {stats.mean_speed_mps:.2f} m/s "
        f"(sd {stats.std_speed_mps:.2f}) = {stats.mean_speed_mps * 3.6:.1f} km/h",
        "",
        "coverage",
        f"  mean vehicles per RSU      : {stats.mean_vehicles_per_rsu:.3f} "
        f"(sd {stats.std_vehicles_per_rsu:.3f})",
        f"  mean vehicle-RSU edges     : {stats.mean_vehicle_rsu_edges:.2f} per step",
        f"  vehicle-timesteps covered  : {stats.coverage_fraction * 100:.1f}%",
        f"  mean RSUs in range         : {stats.mean_candidates_in_range:.3f}",
        "",
        "dwell and handoff",
        f"  mean dwell time            : {stats.mean_dwell_s:.2f} s "
        f"(median {stats.median_dwell_s:.2f}, sd {stats.std_dwell_s:.2f})",
        f"  completed dwell episodes   : {stats.num_dwell_episodes}",
        f"  handoffs per veh per min   : "
        f"{stats.handoffs_per_vehicle_per_min:.3f}",
        f"  total handoffs             : {stats.total_handoffs}",
        f"  total coverage gaps        : {stats.total_coverage_gaps}",
    ]
    return "\n".join(lines)

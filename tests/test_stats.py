"""Dwell and handoff statistics.

These are the numbers the S1 exit condition reports and that later sessions will quote
when arguing the scenario is a plausible vehicular one. They are asserted against
hand-built serving-RSU sequences, where the right answer is countable by eye, rather
than only against a generated trace.
"""

from __future__ import annotations

import numpy as np

from trustgraph.stats import (
    CoverageSeries,
    coverage_series,
    dwell_times_s,
    format_stats,
    handoff_counts,
    scenario_stats,
)


def _series(serving: list[list[int]], dt_s: float = 1.0) -> CoverageSeries:
    """Build a CoverageSeries from an explicit (num_steps, num_vehicles) table."""
    array = np.asarray(serving, dtype=np.int64)
    return CoverageSeries(
        serving=array,
        in_range_count=np.zeros((array.shape[0], 1), dtype=np.int64),
        edge_count=np.zeros(array.shape[0], dtype=np.int64),
        candidates=np.zeros_like(array),
        dt_s=dt_s,
    )


# ---------------------------------------------------------------------- dwell time


def test_dwell_drops_censored_episodes():
    """First and last runs are truncated by the window, so they are not measurements.

    Here vehicle 0 is at RSU 1 for two steps (censored: starts at t=0), RSU 2 for
    three, then RSU 3 to the end (censored). Only the middle stay is a dwell.
    """
    series = _series([[1], [1], [2], [2], [2], [3], [3]])
    assert dwell_times_s(series).tolist() == [3.0]


def test_dwell_scales_with_dt():
    series = _series([[1], [2], [2], [3]], dt_s=0.5)
    assert dwell_times_s(series).tolist() == [1.0]


def test_dwell_ignores_out_of_coverage_runs():
    """Time spent with no serving RSU is not dwell anywhere."""
    series = _series([[1], [-1], [-1], [2], [2], [3]])
    assert dwell_times_s(series).tolist() == [2.0]


def test_dwell_is_empty_when_nothing_completes():
    assert dwell_times_s(_series([[1], [1], [1]])).size == 0


# ------------------------------------------------------------------------- handoff


def test_handoff_counts_rsu_to_rsu_changes_only():
    """A transition through no-coverage is a gap, not a handoff."""
    series = _series([[1], [1], [2], [-1], [3]])
    handoffs, gaps = handoff_counts(series)
    assert handoffs.tolist() == [1]  # 1 -> 2
    assert gaps.tolist() == [2]  # 2 -> -1 and -1 -> 3


def test_handoff_counts_per_vehicle():
    series = _series([[1, 5], [2, 5], [3, 5]])
    handoffs, gaps = handoff_counts(series)
    assert handoffs.tolist() == [2, 0]
    assert gaps.tolist() == [0, 0]


def test_no_handoff_when_serving_rsu_never_changes():
    handoffs, gaps = handoff_counts(_series([[4], [4], [4], [4]]))
    assert handoffs.tolist() == [0] and gaps.tolist() == [0]


# ----------------------------------------------------------- against a real trace


def test_coverage_series_matches_the_geometry(world, trace):
    from trustgraph.topology import pairwise_distances

    series = coverage_series(trace, world.topology)
    topo = world.topology

    for t in (0, trace.num_steps // 2, trace.num_steps - 1):
        dist = pairwise_distances(trace.positions[t], topo.positions)
        covered = dist <= topo.coverage_radius_m
        assert np.array_equal(series.in_range_count[t], covered.sum(axis=0))
        assert np.array_equal(series.candidates[t], covered.sum(axis=1))
        assert series.edge_count[t] == covered.sum()


def test_scenario_stats_are_internally_consistent(world, trace):
    stats = scenario_stats(trace, world.topology)
    assert stats.num_steps == trace.num_steps
    assert stats.num_vehicles == trace.num_vehicles
    assert sum(stats.segment_sizes) == world.topology.num_rsus
    assert 0.0 <= stats.coverage_fraction <= 1.0
    # Every vehicle-RSU link is one vehicle seeing one RSU, counted from both sides.
    assert stats.mean_vehicles_per_rsu * stats.num_rsus == (
        stats.mean_candidates_in_range * stats.num_vehicles
    )
    assert format_stats(stats)


def test_operating_scenario_is_a_plausible_vehicular_one(demo_cfg):
    """The demo scenario must keep vehicles covered and handing off.

    Guards the calibration recorded in FINDINGS.md F4: without redundant coverage the
    selection rule of L1 has a single candidate and measures nothing (F3).
    """
    from trustgraph.scenario import build_world, generate_trace

    world = build_world(demo_cfg)
    stats = scenario_stats(generate_trace(demo_cfg, world), world.topology)

    assert stats.coverage_fraction > 0.98
    assert stats.mean_candidates_in_range > 1.8
    assert 0.5 < stats.handoffs_per_vehicle_per_min < 6.0
    assert 5.0 < stats.mean_dwell_s < 120.0

"""Mobility source, trace round-trip, and the on-disk contract.

L7 puts synthetic mobility on the critical path and SUMO off it. What the rest of the
project depends on is not the particular motion model but the contract: a trace is a
pure function of (config, seed), it is written once, and every consumer reads it back.
"""

from __future__ import annotations

import numpy as np
import pytest

from trustgraph.mobility import MobilitySource, build_mobility
from trustgraph.roads import build_road_network
from trustgraph.scenario import generate_trace
from trustgraph.trace import TRACE_FORMAT_VERSION, default_trace_path, load_trace


# --------------------------------------------------------------------- determinism


def test_trace_is_a_function_of_config_and_seed(cfg, world):
    """Standing Rule 7, at the level the whole pipeline inherits it from."""
    first = generate_trace(cfg, world)
    second = generate_trace(cfg, world)
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.velocities, second.velocities)


def test_trace_differs_across_seeds(cfg, world):
    """Guards against the determinism test passing on constant output."""
    other = type(cfg)(**{**cfg.__dict__, "seed": cfg.seed + 1})
    assert not np.array_equal(
        generate_trace(cfg, world).positions, generate_trace(other).positions
    )


def test_mobility_stream_is_independent_of_other_draws(cfg, world):
    """D17: consuming the topology stream must not shift the mobility stream."""
    expected = generate_trace(cfg, world).positions
    seeds = cfg.seeds
    seeds.generator("topology").random(1000)
    seeds.generator("features").random(37)
    assert np.array_equal(generate_trace(cfg, world).positions, expected)


# ----------------------------------------------------------------------- the trace


def test_trace_shapes_and_metadata(cfg, trace):
    steps = cfg.scenario["num_steps"]
    vehicles = cfg.mobility["num_vehicles"]
    assert trace.positions.shape == (steps, vehicles, 2)
    assert trace.velocities.shape == (steps, vehicles, 2)
    assert trace.num_steps == steps and trace.num_vehicles == vehicles
    assert trace.dt_s == cfg.mobility["dt_s"]
    assert trace.duration_s == steps * cfg.mobility["dt_s"]
    assert trace.seed == cfg.seed
    assert trace.source == "synthetic_road"


def test_trace_round_trips_through_disk(trace, tmp_path):
    path = trace.save(tmp_path / "t.npz")
    reloaded = load_trace(path)
    assert np.array_equal(reloaded.positions, trace.positions)
    assert np.array_equal(reloaded.velocities, trace.velocities)
    assert reloaded.dt_s == trace.dt_s
    assert reloaded.seed == trace.seed
    assert reloaded.source == trace.source


def test_missing_trace_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_trace.py"):
        load_trace(tmp_path / "absent.npz")


def test_trace_format_version_mismatch_is_refused(trace, tmp_path):
    path = trace.save(tmp_path / "t.npz")
    with np.load(path) as blob:
        fields = {k: blob[k] for k in blob.files}
    fields["format_version"] = np.int64(TRACE_FORMAT_VERSION + 1)
    np.savez_compressed(path, **fields)
    with pytest.raises(ValueError, match="format version"):
        load_trace(path)


def test_default_trace_path_separates_seeds():
    a = default_trace_path("configs/demo.yaml", 1)
    b = default_trace_path("configs/demo.yaml", 2)
    assert a != b and a.parent.name == "traces" and a.suffix == ".npz"


# ---------------------------------------------------------------------- the motion


def test_vehicles_stay_on_the_road_network(cfg, world, trace):
    """Vehicles interpolate between adjacent intersections, so they are on a road."""
    road = world.road
    positions = trace.positions.reshape(-1, 2)
    # Distance from each point to the nearest road segment, computed as the
    # perpendicular distance to the segment clamped to its endpoints.
    best = np.full(positions.shape[0], np.inf)
    for a, b in road.segments():
        pa, pb = road.intersections[a], road.intersections[b]
        d = pb - pa
        t = np.clip(((positions - pa) @ d) / (d @ d), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(positions - (pa + t[:, None] * d), axis=1))
    assert best.max() < 1e-6


def test_vehicles_stay_inside_the_region(world, trace):
    assert trace.positions.min() >= -1e-9
    assert trace.positions.max() <= world.road.extent_m + 1e-9


def test_speeds_respect_the_configured_band(cfg, trace):
    """Speed is realised displacement, so it is bounded by the configured maximum.

    It can be below the minimum - a vehicle held at a signalised junction has speed
    zero - but never above the maximum.
    """
    speeds = trace.speeds
    assert speeds.max() <= cfg.mobility["speed_max_mps"] + 1e-6
    assert speeds.min() >= 0.0


def test_speed_distribution_has_spread(demo_cfg):
    """A single shared speed would make dwell time a constant instead of a spread."""
    trace = generate_trace(demo_cfg)
    per_vehicle_mean = trace.speeds.mean(axis=0)
    assert per_vehicle_mean.std() > 0.5


def test_velocity_matches_the_recorded_positions(trace):
    """Velocity is the realised displacement, not an internal desired speed."""
    steps = np.diff(trace.positions, axis=0) / trace.dt_s
    assert np.allclose(trace.velocities[1:], steps)


def test_vehicles_actually_travel(demo_cfg):
    """A trace where nothing moves would satisfy every other assertion here."""
    trace = generate_trace(demo_cfg)
    displacement = np.linalg.norm(
        trace.positions[-1] - trace.positions[0], axis=1
    )
    assert displacement.mean() > 100.0


# ------------------------------------------------------------------ the L7 contract


def test_synthetic_road_is_a_mobility_source(cfg, world):
    source = build_mobility(cfg.mobility, world.road, np.random.default_rng(0))
    assert isinstance(source, MobilitySource)
    assert source.num_vehicles == cfg.mobility["num_vehicles"]
    assert source.name == "synthetic_road"


def test_sumo_is_not_imported(cfg, world):
    """L7: SUMO is off the critical path and must not be a dependency."""
    from pathlib import Path

    import trustgraph.mobility as mob

    source = Path(mob.__file__).read_text(encoding="utf-8").lower()
    assert "import traci" not in source
    assert "import sumolib" not in source
    with pytest.raises(ValueError, match="unknown mobility source"):
        build_mobility(
            {**cfg.mobility, "source": "sumo"}, world.road, np.random.default_rng(0)
        )

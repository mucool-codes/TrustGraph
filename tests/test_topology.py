"""Road layout, RSU placement, backhaul segments, and RSU-RSU edges.

The properties asserted here are the ones L5 and L6 rest on. If segments stop being
spatially coherent, or RSU-RSU edges stop appearing, H2 loses the structural signal it
is supposed to find - and it would fail for a reason that has nothing to do with
message passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from trustgraph.config import load_config
from trustgraph.roads import build_road_network
from trustgraph.scenario import build_world
from trustgraph.topology import build_topology, pairwise_distances

from conftest import DEMO_CONFIG


# ----------------------------------------------------------------------- the roads


def test_road_grid_shape(cfg):
    road = build_road_network(cfg.road)
    bx, by = cfg.road["blocks_x"], cfg.road["blocks_y"]
    assert road.num_intersections == (bx + 1) * (by + 1)
    # Each interior intersection has 4 arms, each edge 3, each corner 2.
    degrees = sorted(len(n) for n in road.neighbours)
    assert degrees[0] == 2 and degrees[-1] == 4
    # Undirected segments: horizontal plus vertical.
    assert len(road.segments()) == bx * (by + 1) + by * (bx + 1)


def test_road_intersections_span_the_region(cfg):
    road = build_road_network(cfg.road)
    assert road.intersections.min() == 0.0
    assert road.intersections.max() == pytest.approx(cfg.road["extent_m"])


def test_road_layout_is_deterministic_and_rng_free(cfg):
    """The layout takes no generator: two builds are identical by construction."""
    a = build_road_network(cfg.road)
    b = build_road_network(cfg.road)
    assert np.array_equal(a.intersections, b.intersections)
    assert a.neighbours == b.neighbours


def test_road_rejects_degenerate_config(cfg):
    with pytest.raises(ValueError, match="blocks_x"):
        build_road_network({**cfg.road, "blocks_x": 0})
    with pytest.raises(ValueError, match="extent_m"):
        build_road_network({**cfg.road, "extent_m": 0.0})


# ------------------------------------------------------------------- RSU placement


def test_rsus_are_placed_on_the_road_network(cfg, world):
    """Every RSU sits within the placement jitter of an intersection or midpoint."""
    road = world.road
    sites = np.vstack([road.intersections, road.segment_midpoints()])
    nearest = pairwise_distances(world.topology.positions, sites).min(axis=1)
    tolerance = cfg.topology["placement_jitter_m"] * np.sqrt(2) + 1e-6
    assert np.all(nearest <= tolerance)


def test_placement_is_deterministic_for_a_seed(cfg):
    a = build_world(cfg).topology
    b = build_world(cfg).topology
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.backhaul_segment_id, b.backhaul_segment_id)


def test_explicit_positions_are_used_verbatim(cfg):
    """A fixed deployment can be pinned exactly through the config."""
    wanted = [[0.0, 0.0], [400.0, 0.0], [0.0, 400.0], [400.0, 400.0]]
    topology = build_topology(
        {**cfg.topology, "positions": wanted, "placement_jitter_m": 50.0},
        build_road_network(cfg.road),
        np.random.default_rng(0),
    )
    assert np.array_equal(topology.positions, np.asarray(wanted))


def test_explicit_positions_must_match_num_rsus(cfg):
    with pytest.raises(ValueError, match="num_rsus"):
        build_topology(
            {**cfg.topology, "positions": [[0.0, 0.0]]},
            build_road_network(cfg.road),
            np.random.default_rng(0),
        )


def test_too_many_rsus_for_the_layout_is_an_error(cfg):
    road = build_road_network(cfg.road)
    too_many = len(road.intersections) + len(road.segments()) + 1
    with pytest.raises(ValueError, match="candidate sites"):
        build_topology(
            {**cfg.topology, "num_rsus": too_many}, road, np.random.default_rng(0)
        )


def test_placement_covers_the_road(demo_cfg):
    """The operating scenario must actually cover the roads it is placed on.

    A scenario where vehicles routinely have no RSU in range measures coverage, not
    trust - the failure mode recorded in FINDINGS.md F3.
    """
    world = build_world(demo_cfg)
    road = world.road
    samples = []
    for a, b in road.segments():
        pa, pb = road.intersections[a], road.intersections[b]
        samples.append(pa + np.linspace(0, 1, 25)[:, None] * (pb - pa))
    samples = np.vstack(samples)

    reach = pairwise_distances(samples, world.topology.positions)
    covered = (reach <= world.topology.coverage_radius_m).sum(axis=1)
    assert (covered >= 1).mean() > 0.99, "road coverage has holes"
    assert (covered >= 2).mean() > 0.50, "no redundant coverage: decisions have one candidate"


# --------------------------------------------------------------- backhaul segments


def test_every_rsu_has_a_segment_and_no_segment_is_empty(cfg, world):
    """L5: every RSU carries a backhaul_segment_id at topology build time."""
    ids = world.topology.backhaul_segment_id
    assert ids.shape == (world.topology.num_rsus,)
    assert set(ids.tolist()) == set(range(cfg.topology["num_backhaul_segments"]))


def test_segments_respect_the_minimum_size(demo_cfg):
    world = build_world(demo_cfg)
    assert world.topology.segment_sizes().min() >= demo_cfg.topology["min_segment_size"]


def test_segments_are_spatially_coherent(demo_cfg):
    """L5 only helps if a segment is a place, not a scattering of RSUs.

    Measured as: the mean distance between two RSUs on the same segment is markedly
    smaller than between two on different segments. A random assignment would make
    the two roughly equal.
    """
    topo = build_world(demo_cfg).topology
    dist = pairwise_distances(topo.positions, topo.positions)
    i, j = np.triu_indices(topo.num_rsus, k=1)
    same = topo.backhaul_segment_id[i] == topo.backhaul_segment_id[j]
    assert dist[i, j][same].mean() < 0.6 * dist[i, j][~same].mean()


def test_segment_swap_noise_changes_assignments(demo_cfg):
    """The swap probability must actually do something, or same_segment is geometry.

    Compared across several seeds because a single seed can draw no swaps at all.
    """
    without = {
        s: build_world(
            type(demo_cfg)(
                **{
                    **demo_cfg.__dict__,
                    "seed": s,
                    "topology": {**demo_cfg.topology, "segment_swap_prob": 0.0},
                }
            )
        ).topology.backhaul_segment_id
        for s in range(1, 7)
    }
    with_swaps = {
        s: build_world(
            type(demo_cfg)(**{**demo_cfg.__dict__, "seed": s})
        ).topology.backhaul_segment_id
        for s in range(1, 7)
    }
    differing = sum(
        1 for s in without if not np.array_equal(without[s], with_swaps[s])
    )
    assert differing >= 2, "segment_swap_prob had almost no effect"


# ------------------------------------------------------------------ RSU-RSU edges


def test_rsu_edges_match_the_coordination_radius(cfg, world):
    """L6: exactly the pairs within rsu_link_radius_m, each listed once as (i<j)."""
    topo = world.topology
    dist = pairwise_distances(topo.positions, topo.positions)
    i, j = np.triu_indices(topo.num_rsus, k=1)
    expected = {
        (int(a), int(b))
        for a, b, d in zip(i, j, dist[i, j])
        if d <= topo.rsu_link_radius_m
    }
    assert {(int(a), int(b)) for a, b in topo.rsu_edges} == expected
    assert np.all(topo.rsu_edges[:, 0] < topo.rsu_edges[:, 1])


def test_rsu_graph_is_connected(demo_cfg):
    """Segment evidence can only propagate along a connected RSU graph (L6)."""
    topo = build_world(demo_cfg).topology
    adjacency = {i: set() for i in range(topo.num_rsus)}
    for i, j in topo.rsu_edges:
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))

    seen, stack = set(), [0]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency[node] - seen)
    assert len(seen) == topo.num_rsus


def test_each_segment_is_internally_connected(demo_cfg):
    """A segment whose RSUs cannot reach each other cannot share evidence.

    This is the precondition for H2: correlated failure is injected per segment (L5),
    so the GNN can only exploit it if same-segment RSUs are within k hops of one
    another along same_segment edges.
    """
    topo = build_world(demo_cfg).topology
    adjacency = {i: set() for i in range(topo.num_rsus)}
    for i, j in topo.rsu_edges:
        if topo.backhaul_segment_id[i] == topo.backhaul_segment_id[j]:
            adjacency[int(i)].add(int(j))
            adjacency[int(j)].add(int(i))

    for segment in range(topo.num_segments):
        members = set(np.flatnonzero(topo.backhaul_segment_id == segment).tolist())
        seen, stack = set(), [min(members)]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend((adjacency[node] & members) - seen)
        assert seen == members, f"segment {segment} is internally disconnected"


def test_both_same_segment_and_cross_segment_edges_exist(demo_cfg):
    """same_segment must vary, or it carries no information at all."""
    topo = build_world(demo_cfg).topology
    same = topo.backhaul_segment_id[topo.rsu_edges[:, 0]] == (
        topo.backhaul_segment_id[topo.rsu_edges[:, 1]]
    )
    assert 0 < same.sum() < same.size


def test_config_rejects_more_segments_than_rsus():
    cfg = load_config(DEMO_CONFIG)
    from trustgraph.config import _validate

    with pytest.raises(ValueError, match="exceeds num_rsus"):
        _validate(
            type(cfg)(
                **{
                    **cfg.__dict__,
                    "topology": {**cfg.topology, "num_backhaul_segments": 999},
                }
            )
        )

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


def _geometric_topology(cfg, seed: int):
    """The pure-geometry assignment for a seed: both swap mechanisms disabled.

    The Lloyd step consumes no randomness and the jitter draw is unchanged, so this
    isolates the swaps exactly rather than approximating them with re-derived
    centroids (which a swapped RSU would itself have pulled).
    """
    return build_world(
        type(cfg)(
            **{
                **cfg.__dict__,
                "seed": seed,
                "topology": {
                    **cfg.topology,
                    "segment_swap_prob": 0.0,
                    "min_swap_fraction": 0.0,
                },
            }
        )
    ).topology


def test_swap_count_meets_the_floor_on_every_seed(demo_cfg):
    """The off-geometry property must hold on every evaluation seed, not on average.

    A binomial draw at swap_prob=0.10 over 20 RSUs yields zero often enough to matter
    - seed 4 produced none before the floor existed (FINDINGS.md F6) - which would
    have made the property silently false for one seed of a multi-seed evaluation.
    """
    floor = int(np.ceil(demo_cfg.topology["min_swap_fraction"] * demo_cfg.topology["num_rsus"]))
    assert floor >= 3

    for seed in range(1, 6):
        topo = build_world(type(demo_cfg)(**{**demo_cfg.__dict__, "seed": seed})).topology
        geometric = _geometric_topology(demo_cfg, seed)
        assert np.array_equal(topo.positions, geometric.positions)

        actual = int(
            (topo.backhaul_segment_id != geometric.backhaul_segment_id).sum()
        )
        assert actual == topo.num_swapped_rsus, "reported swap count disagrees"
        assert actual >= floor, f"seed {seed} has {actual} swaps, floor is {floor}"


def test_swap_count_is_reported_on_the_topology(demo_cfg):
    topo = build_world(demo_cfg).topology
    geometric = _geometric_topology(demo_cfg, demo_cfg.seed)
    assert topo.num_swapped_rsus == int(
        (topo.backhaul_segment_id != geometric.backhaul_segment_id).sum()
    )
    assert geometric.num_swapped_rsus == 0


def test_forced_swaps_take_boundary_rsus_first(demo_cfg):
    """A forced swap should move the site least deep inside its region.

    Otherwise the floor would distort the layout more than it needs to, moving an RSU
    from the middle of one segment into a segment it is nowhere near.
    """
    cfg = type(demo_cfg)(
        **{
            **demo_cfg.__dict__,
            "topology": {**demo_cfg.topology, "segment_swap_prob": 0.0},
        }
    )
    topo = build_world(cfg).topology
    geometric = _geometric_topology(demo_cfg, demo_cfg.seed)
    swapped = np.flatnonzero(topo.backhaul_segment_id != geometric.backhaul_segment_id)
    assert swapped.size == topo.num_swapped_rsus >= 3

    centroids = np.stack(
        [
            topo.positions[geometric.backhaul_segment_id == s].mean(axis=0)
            for s in range(topo.num_segments)
        ]
    )
    distance = pairwise_distances(topo.positions, centroids)
    rows = np.arange(topo.num_rsus)
    own = distance[rows, geometric.backhaul_segment_id]
    across = distance.copy()
    across[rows, geometric.backhaul_segment_id] = np.inf
    margin = across.min(axis=1) - own

    # Swapped RSUs sit nearer a boundary than the ones left alone. Asserted on the
    # means rather than as a strict total order, because a boundary RSU is skipped
    # when re-homing it would split its source segment or land it in a segment it has
    # no link to - so a slightly deeper RSU can legitimately be taken instead.
    untouched = np.setdiff1d(rows, swapped)
    assert margin[swapped].mean() < margin[untouched].mean()
    assert margin[swapped].max() < np.median(margin)


def test_swap_floor_respects_the_minimum_segment_size(demo_cfg):
    """The floor is best-effort: L5's "several RSUs each" is not traded away for it."""
    cfg = type(demo_cfg)(
        **{
            **demo_cfg.__dict__,
            "topology": {
                **demo_cfg.topology,
                "min_swap_fraction": 0.9,  # unsatisfiable
                "min_segment_size": 4,
            },
        }
    )
    topo = build_world(cfg).topology
    assert topo.segment_sizes().min() >= 4
    assert set(topo.backhaul_segment_id.tolist()) == set(range(topo.num_segments))


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


def _assert_segments_internally_connected(topo, label: str) -> None:
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
        assert seen == members, f"{label}: segment {segment} is internally disconnected"


def test_each_segment_is_internally_connected(demo_cfg):
    """A segment whose RSUs cannot reach each other cannot share evidence.

    This is the precondition for H2: correlated failure is injected per segment (L5),
    so the GNN can only exploit it if same-segment RSUs are within k hops of one
    another along same_segment edges.

    Checked across seeds, not only the default one. An earlier version of this test
    checked seed 20260903 alone, which happened to be one of the two seeds where the
    property held by luck while four others were silently split (FINDINGS.md F6).
    """
    for seed in [demo_cfg.seed] + list(range(1, 6)):
        topo = build_world(
            type(demo_cfg)(**{**demo_cfg.__dict__, "seed": seed})
        ).topology
        _assert_segments_internally_connected(topo, f"seed {seed}")


def test_swapped_rsus_have_a_link_into_their_new_segment(demo_cfg):
    """A site cannot be homed to a backhaul link it has no physical path to.

    This is what keeps segments connected when a swap happens: without it the swapped
    RSU joins its new segment as an isolated node.
    """
    for seed in [demo_cfg.seed] + list(range(1, 6)):
        topo = build_world(
            type(demo_cfg)(**{**demo_cfg.__dict__, "seed": seed})
        ).topology
        geometric = _geometric_topology(demo_cfg, seed)
        swapped = np.flatnonzero(
            topo.backhaul_segment_id != geometric.backhaul_segment_id
        )
        assert swapped.size >= 3

        adjacency = {i: set() for i in range(topo.num_rsus)}
        for i, j in topo.rsu_edges:
            adjacency[int(i)].add(int(j))
            adjacency[int(j)].add(int(i))

        for r in swapped:
            neighbours = {
                int(topo.backhaul_segment_id[n]) for n in adjacency[int(r)]
            }
            assert int(topo.backhaul_segment_id[r]) in neighbours, (
                f"seed {seed}: RSU {r} was homed to segment "
                f"{topo.backhaul_segment_id[r]} with no link into it"
            )


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

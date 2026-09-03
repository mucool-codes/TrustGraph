"""Graph construction: feature layout, edge semantics, and link-age tracking.

The feature *names and order* are fixed once (D15) and consumed by the model and, in a
later session, by explanation attribution. A mismatch between producer and consumer
should be a shape error here rather than a quietly wrong attribution there.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from trustgraph.features import (
    EDGE_COL,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    RSU_COL,
    RSU_FEATURES,
    VEHICLE_FEATURES,
)
from trustgraph.graph import (
    PLACEHOLDER_CERT_VALID,
    PLACEHOLDER_LATENCY_DEV,
    PLACEHOLDER_SUCCESS_EWMA,
    PLACEHOLDER_UPTIME_STABILITY,
    dwell_estimate_s,
)
from trustgraph.links import LinkModel
from trustgraph.scenario import build_snapshot_builder


@pytest.fixture
def builder(cfg, world, trace):
    return build_snapshot_builder(cfg, world, trace)


# --------------------------------------------------------------------- the layout


def test_feature_layout_matches_spec():
    """Feature names and widths are the ones PROJECT_SPEC.md section 5 fixes."""
    assert RSU_FEATURES == (
        "load",
        "queue_depth",
        "cert_valid",
        "success_ewma",
        "latency_dev",
        "uptime_stability",
    )
    assert VEHICLE_FEATURES == ("task_demand", "speed", "dwell_estimate")
    assert NODE_FEATURE_DIM == len(RSU_FEATURES) + len(VEHICLE_FEATURES) == 9
    assert EDGE_FEATURE_DIM == 4
    assert len(NODE_FEATURE_NAMES) == NODE_FEATURE_DIM
    # same_segment is the structural handle for correlated failure (L5/L6).
    assert "same_segment" in EDGE_COL


def test_snapshot_shapes(cfg, world, trace, builder):
    data = builder.build(0)
    n_rsu = world.topology.num_rsus
    n_veh = trace.num_vehicles

    assert data.x.shape == (n_rsu + n_veh, NODE_FEATURE_DIM)
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape == (data.edge_index.shape[1], EDGE_FEATURE_DIM)
    assert data.is_rsu.sum().item() == n_rsu
    # RSUs come first, then vehicles - relied on by the selection path.
    assert bool(data.is_rsu[0]) and not bool(data.is_rsu[-1])
    assert data.timestep == 0


def test_node_blocks_are_zero_padded(world, builder):
    """RSU rows zero the vehicle block and vice versa (DECISIONS.md D16)."""
    data = builder.build(0)
    n_rsu = world.topology.num_rsus
    n_rsu_feat = len(RSU_FEATURES)
    assert torch.all(data.x[:n_rsu, n_rsu_feat:] == 0)
    assert torch.all(data.x[n_rsu:, :n_rsu_feat] == 0)


def test_graph_is_homogeneous_not_hetero(builder):
    """L6: single edge type, one edge_index - explicitly not HeteroData."""
    data = builder.build(0)
    assert isinstance(data, Data)
    assert type(data).__name__ != "HeteroData"


def test_all_features_are_in_unit_range(builder):
    """Everything the model reads is scaled to [0, 1]; nothing leaks raw metres."""
    for data in builder.snapshots(5):
        assert torch.all(data.x >= 0.0) and torch.all(data.x <= 1.0)
        assert torch.all(data.edge_attr >= 0.0) and torch.all(data.edge_attr <= 1.0)


# -------------------------------------------------------------------- the features


def test_behavioural_features_are_the_declared_placeholders(world, builder):
    """S1 leaves these constant. When S2/S3 make them real, this test must change.

    Constants rather than random values: a result that depends on a placeholder is
    then obviously degenerate instead of plausibly noisy.
    """
    data = builder.build(0)
    rsu = data.x[: world.topology.num_rsus]
    assert torch.all(rsu[:, RSU_COL["success_ewma"]] == PLACEHOLDER_SUCCESS_EWMA)
    assert torch.all(rsu[:, RSU_COL["latency_dev"]] == PLACEHOLDER_LATENCY_DEV)
    assert torch.all(
        rsu[:, RSU_COL["uptime_stability"]] == PLACEHOLDER_UPTIME_STABILITY
    )
    assert torch.all(rsu[:, RSU_COL["cert_valid"]] == PLACEHOLDER_CERT_VALID)


def test_load_tracks_coverage_demand(cfg, world, builder):
    """load is the RSU's in-range vehicle count against its capacity."""
    from trustgraph.topology import pairwise_distances

    data = builder.build(0)
    topo = world.topology
    covered = (
        pairwise_distances(builder.trace.positions[0], topo.positions)
        <= topo.coverage_radius_m
    )
    expected = np.clip(
        covered.sum(axis=0) / cfg.graph["rsu_capacity_vehicles"], 0.0, 1.0
    )
    assert np.allclose(
        data.x[: topo.num_rsus, RSU_COL["load"]].numpy(), expected, atol=1e-6
    )


def test_queue_depth_is_the_excess_beyond_capacity(world, builder):
    """queue_depth must be zero wherever load has not saturated."""
    data = builder.build(0)
    rsu = data.x[: world.topology.num_rsus]
    unsaturated = rsu[:, RSU_COL["load"]] < 1.0
    assert torch.all(rsu[unsaturated, RSU_COL["queue_depth"]] == 0.0)


def test_dwell_estimate_geometry():
    """The straight-line crossing time, checked against a case done by hand."""
    # At the centre of a 100 m circle, moving at 10 m/s: 10 s to the edge.
    assert dwell_estimate_s(
        np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([0.0, 0.0]), 100.0
    ) == pytest.approx(10.0)
    # 50 m in front of the centre, heading straight out: 5 s of the remaining 50 m.
    assert dwell_estimate_s(
        np.array([50.0, 0.0]), np.array([10.0, 0.0]), np.array([0.0, 0.0]), 100.0
    ) == pytest.approx(5.0)
    # Already outside.
    assert dwell_estimate_s(
        np.array([150.0, 0.0]), np.array([10.0, 0.0]), np.array([0.0, 0.0]), 100.0
    ) == 0.0
    # Stationary: never leaves.
    assert dwell_estimate_s(
        np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), 100.0
    ) == float("inf")


def test_serving_rsu_is_the_nearest_in_range(world, builder):
    from trustgraph.topology import pairwise_distances

    data = builder.build(0)
    topo = world.topology
    dist = pairwise_distances(builder.trace.positions[0], topo.positions)
    for v, served in enumerate(data.serving_rsu.numpy()):
        in_range = np.flatnonzero(dist[v] <= topo.coverage_radius_m)
        if in_range.size == 0:
            assert served == -1
        else:
            assert served == in_range[np.argmin(dist[v][in_range])]


# ----------------------------------------------------------------------- the edges


def test_same_segment_edge_feature_is_correct(world, builder):
    """L6: same_segment is true exactly for RSU-RSU pairs sharing a segment."""
    data = builder.build(0)
    topo = world.topology
    src, dst = data.edge_index
    flag = data.edge_attr[:, EDGE_COL["same_segment"]]

    for e in range(data.edge_index.shape[1]):
        i, j = int(src[e]), int(dst[e])
        if i < topo.num_rsus and j < topo.num_rsus:
            shared = topo.backhaul_segment_id[i] == topo.backhaul_segment_id[j]
            assert bool(flag[e]) == bool(shared)
        else:
            # A vehicle sits on no backhaul segment.
            assert flag[e] == 0.0


def test_rsu_rsu_edges_exist(world, builder):
    """L6: RSU-RSU edges are required - without them nothing propagates."""
    data = builder.build(0)
    src, dst = data.edge_index
    n_rsu = world.topology.num_rsus
    assert ((src < n_rsu) & (dst < n_rsu)).sum().item() > 0


def test_edges_are_stored_in_both_directions(builder):
    """SAGEConv aggregates over in-edges, so an undirected graph needs both."""
    data = builder.build(0)
    pairs = {(int(a), int(b)) for a, b in data.edge_index.t().tolist()}
    assert all((b, a) in pairs for a, b in pairs)


def test_vehicle_rsu_edges_respect_the_coverage_radius(world, builder):
    from trustgraph.topology import pairwise_distances

    data = builder.build(0)
    topo = world.topology
    dist = pairwise_distances(builder.trace.positions[0], topo.positions)
    expected = int((dist <= topo.coverage_radius_m).sum())

    src, dst = data.edge_index
    n_rsu = topo.num_rsus
    veh_rsu = ((src >= n_rsu) & (dst < n_rsu)).sum().item()
    assert veh_rsu == expected


def test_latency_and_signal_move_against_each_other(world, builder):
    """Within one link type, latency rises exactly as signal falls (links.py).

    Split by type deliberately: access and backhaul sit in different latency bands,
    so a weak backhaul link can still beat a strong access link and the relationship
    does not hold across the mixed edge set.
    """
    data = builder.build(0)
    src, dst = data.edge_index
    backhaul = ((src < world.topology.num_rsus) & (dst < world.topology.num_rsus)).numpy()
    latency = data.edge_attr[:, EDGE_COL["link_latency"]].numpy()
    signal = data.edge_attr[:, EDGE_COL["signal_strength"]].numpy()

    for mask, label in ((backhaul, "backhaul"), (~backhaul, "access")):
        assert mask.sum() > 2, f"no {label} edges to test"
        assert np.corrcoef(latency[mask], signal[mask])[0, 1] < -0.9


def test_signal_strength_falls_with_distance():
    """The path-loss model itself: further is weaker, and it saturates in [0, 1]."""
    model = LinkModel()
    signal = model.signal_strength(np.array([1.0, 50.0, 200.0, 400.0, 650.0]))
    assert np.all(np.diff(signal) < 0)
    assert signal[0] == pytest.approx(1.0)  # near field, clipped at the cap
    assert np.all((signal >= 0.0) & (signal <= 1.0))
    # Far past the sensitivity floor everything pins to zero, as it should.
    assert model.signal_strength(np.array([5000.0]))[0] == 0.0
    # A backhaul hop is cheaper than an access hop at the same distance.
    assert model.latency_ms(np.array([300.0]), True) < model.latency_ms(
        np.array([300.0]), False
    )


def test_signal_resolves_across_the_whole_coverage_radius(demo_cfg):
    """The sensitivity floor must sit below the RSSI at the cell edge.

    Otherwise every link near `coverage_radius_m` saturates at signal 0 and maximum
    latency, and the two edge features stop distinguishing a boundary link from one
    well past it.
    """
    from trustgraph.links import build_link_model

    model = build_link_model(demo_cfg.link)
    edge = model.signal_strength(
        np.array([float(demo_cfg.topology["coverage_radius_m"])])
    )[0]
    assert 0.02 < edge < 0.5


# -------------------------------------------------------------------- the link age


def test_link_age_grows_while_a_link_persists(cfg, world, trace, builder):
    """link_age is history, not geometry: a surviving link must age."""
    ages: dict[tuple[int, int], list[float]] = {}
    for data in builder.snapshots(6):
        column = data.edge_attr[:, EDGE_COL["link_age"]]
        for e, (a, b) in enumerate(data.edge_index.t().tolist()):
            ages.setdefault((a, b), []).append(float(column[e]))

    persisted = [v for v in ages.values() if len(v) >= 3]
    assert persisted, "no link survived three timesteps; the test proves nothing"
    assert any(v[-1] > v[0] for v in persisted)


def test_link_age_starts_at_zero_for_a_new_link(builder):
    """Every link present at t=0 has just been observed for the first time."""
    data = builder.build(0)
    assert torch.all(data.edge_attr[:, EDGE_COL["link_age"]] == 0.0)


def test_link_age_resets_when_a_link_breaks_and_reforms(world):
    """A re-formed link is genuinely new and carries the extra uncertainty."""
    import copy

    from trustgraph.graph import SnapshotBuilder
    from trustgraph.trace import Trace

    # One vehicle, one RSU: in range, out of range, then back in range.
    topo = world.topology
    centre = topo.positions[0]
    far = centre + np.array([topo.coverage_radius_m * 5, 0.0])
    positions = np.array([[centre], [centre], [far], [centre]], dtype=np.float64)
    trace = Trace(
        positions=positions,
        velocities=np.zeros_like(positions),
        dt_s=1.0,
        seed=0,
        source="test",
    )
    builder = SnapshotBuilder(
        topo, trace, LinkModel(), {"link_age_norm_s": 10.0}, np.random.default_rng(0)
    )

    def age_to_rsu0(data):
        src, dst = data.edge_index
        mask = (src == topo.num_rsus) & (dst == 0)
        return data.edge_attr[mask, EDGE_COL["link_age"]]

    snapshots = list(builder.snapshots())
    assert float(age_to_rsu0(snapshots[0])) == 0.0
    assert float(age_to_rsu0(snapshots[1])) > 0.0  # link has aged one step
    assert age_to_rsu0(snapshots[2]).numel() == 0  # out of range: no link
    assert float(age_to_rsu0(snapshots[3])) == 0.0  # re-formed: age reset

    del copy


# ---------------------------------------------------------------- sequential access


def test_snapshots_must_be_walked_in_order(builder):
    """link_age depends on history, so a random-access build is refused loudly."""
    builder.build(0)
    builder.build(1)
    with pytest.raises(ValueError, match="in order"):
        builder.build(5)


def test_reset_rewinds_the_builder(builder):
    builder.build(0)
    builder.build(1)
    builder.reset()
    data = builder.build(0)
    assert torch.all(data.edge_attr[:, EDGE_COL["link_age"]] == 0.0)


def test_snapshot_sequence_is_reproducible(cfg, world, trace):
    """The same (config, seed, trace) yields identical graph tensors."""
    a = list(build_snapshot_builder(cfg, world, trace).snapshots())
    b = list(build_snapshot_builder(cfg, world, trace).snapshots())
    assert len(a) == len(b) == trace.num_steps
    for da, db in zip(a, b):
        assert torch.equal(da.x, db.x)
        assert torch.equal(da.edge_index, db.edge_index)
        assert torch.equal(da.edge_attr, db.edge_attr)


def test_graph_changes_over_time(demo_cfg):
    """A static graph would make the whole dynamic-topology premise vacuous."""
    from trustgraph.scenario import build_world, generate_trace

    world = build_world(demo_cfg)
    trace = generate_trace(demo_cfg, world)
    builder = build_snapshot_builder(demo_cfg, world, trace)
    counts = {int(d.edge_index.shape[1]) for d in builder.snapshots(60)}
    assert len(counts) > 1

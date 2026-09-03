"""Smoke test: the whole pipeline on a tiny graph.

Asserts output shape and determinism across two runs with the same seed - the S0 exit
condition. Also pins the parts of the locked decisions that are cheap to check
structurally, so a later session that breaks one gets a failing test rather than a
quietly wrong result.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import SeedChain, load_config  # noqa: E402
from trustgraph.features import (  # noqa: E402
    EDGE_COL,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    RSU_FEATURES,
    VEHICLE_FEATURES,
)
from trustgraph.graph import build_snapshot  # noqa: E402
from trustgraph.mobility import build_mobility  # noqa: E402
from trustgraph.model import build_trust_head  # noqa: E402
from trustgraph.pipeline import format_decisions, run_pipeline  # noqa: E402
from trustgraph.selection import score_candidates, select  # noqa: E402
from trustgraph.topology import build_topology  # noqa: E402

SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke.yaml"
DEMO_CONFIG = REPO_ROOT / "configs" / "demo.yaml"


@pytest.fixture
def cfg():
    return load_config(SMOKE_CONFIG)


# --------------------------------------------------------------------------- shapes


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


def test_snapshot_shapes(cfg):
    seeds = cfg.seeds
    topology = build_topology(cfg.topology, seeds.generator("topology"))
    mobility = build_mobility(cfg.mobility, seeds.generator("mobility"))
    positions = mobility.reset()

    data = build_snapshot(topology, positions, seeds.generator("features"))

    n_rsu = cfg.topology["num_rsus"]
    n_veh = cfg.mobility["num_vehicles"]
    assert data.x.shape == (n_rsu + n_veh, NODE_FEATURE_DIM)
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape == (data.edge_index.shape[1], EDGE_FEATURE_DIM)
    assert data.is_rsu.sum().item() == n_rsu
    # RSUs come first, then vehicles - relied on by the selection path.
    assert bool(data.is_rsu[0]) and not bool(data.is_rsu[-1])


def test_node_blocks_are_zero_padded(cfg):
    """RSU rows zero the vehicle block and vice versa (DECISIONS.md D16)."""
    seeds = cfg.seeds
    topology = build_topology(cfg.topology, seeds.generator("topology"))
    mobility = build_mobility(cfg.mobility, seeds.generator("mobility"))
    data = build_snapshot(topology, mobility.reset(), seeds.generator("features"))

    n_rsu = topology.num_rsus
    n_rsu_feat = len(RSU_FEATURES)
    assert torch.all(data.x[:n_rsu, n_rsu_feat:] == 0)
    assert torch.all(data.x[n_rsu:, :n_rsu_feat] == 0)


def test_trust_head_output_shape_and_range(cfg):
    seeds = cfg.seeds
    topology = build_topology(cfg.topology, seeds.generator("topology"))
    mobility = build_mobility(cfg.mobility, seeds.generator("mobility"))
    data = build_snapshot(topology, mobility.reset(), seeds.generator("features"))
    model = build_trust_head(cfg.model, seeds.torch_seed("model_init"), cfg.device)

    with torch.no_grad():
        trust = model(data.x, data.edge_index)

    # L1: a single scalar per node, nothing else.
    assert trust.shape == (data.num_nodes,)
    assert torch.all(trust >= 0.0) and torch.all(trust <= 1.0)


# ---------------------------------------------------------------------- determinism


def test_pipeline_deterministic_same_seed(cfg):
    """Two runs, same seed, identical decisions - the S0 exit condition."""
    first = run_pipeline(cfg)
    second = run_pipeline(cfg)

    assert len(first) > 0, "scenario produced no decisions; the test proves nothing"
    assert first == second
    assert format_decisions(cfg, first) == format_decisions(cfg, second)


def test_pipeline_differs_across_seeds(cfg):
    """Guards against the determinism test passing on constant output."""
    other = type(cfg)(**{**cfg.__dict__, "seed": cfg.seed + 1})
    assert run_pipeline(cfg) != run_pipeline(other)


def test_seed_chain_is_order_independent():
    """Streams do not shift when an unrelated purpose draws (DECISIONS.md D17)."""
    a = SeedChain(1234)
    expected = a.generator("mobility").random(5)

    b = SeedChain(1234)
    b.generator("topology").random(100)  # unrelated draws
    b.generator("features").random(7)
    assert np.allclose(b.generator("mobility").random(5), expected)


def test_seed_chain_streams_are_distinct():
    chain = SeedChain(99)
    assert not np.allclose(
        chain.generator("topology").random(8), chain.generator("mobility").random(8)
    )


def test_run_py_byte_identical_across_processes():
    """The literal exit condition: run it twice, compare stdout byte-for-byte.

    Uses a subprocess so this also catches anything that depends on interpreter state
    surviving between runs (a global RNG, a module-level cache).
    """
    cmd = [sys.executable, "run.py", "--config", str(DEMO_CONFIG)]
    first = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    second = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert first.stdout == second.stdout
    assert "total decisions" in first.stdout


# ----------------------------------------------------------------- locked decisions


def test_selection_rule_is_the_locked_formula():
    """L1: score(v) = alpha*trust - beta*latency - gamma*load."""
    trust = np.array([0.9, 0.2])
    latency = np.array([0.5, 0.1])
    load = np.array([0.4, 0.7])
    got = score_candidates(trust, latency, load, alpha=1.0, beta=0.5, gamma=0.3)
    want = 1.0 * trust - 0.5 * latency - 0.3 * load
    assert np.allclose(got, want)


def test_baseline_a_ignores_trust():
    """L2: Baseline A is the same rule with alpha = 0."""
    latency = np.array([0.5, 0.1])
    load = np.array([0.4, 0.7])
    low = score_candidates(np.array([0.0, 0.0]), latency, load, 0.0, 0.5, 0.3)
    high = score_candidates(np.array([1.0, 1.0]), latency, load, 0.0, 0.5, 0.3)
    assert np.allclose(low, high)


def test_selection_terms_sum_to_score():
    """L11 Layer 1 is exact: the three terms reconstruct the score identically."""
    d = select(
        timestep=0,
        vehicle_id=0,
        candidates=np.array([0, 1, 2]),
        trust=np.array([0.9, 0.5, 0.7]),
        latency=np.array([0.2, 0.1, 0.9]),
        load=np.array([0.3, 0.8, 0.1]),
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
    )
    assert d is not None
    assert d.trust_term + d.latency_term + d.load_term == pytest.approx(d.score)
    assert d.runner_up_score is not None and d.score >= d.runner_up_score


def test_selection_ties_break_deterministically():
    """Identical scores must resolve to the lowest RSU index, not to argmax order."""
    args = dict(
        timestep=0,
        vehicle_id=0,
        candidates=np.array([5, 2, 9]),
        trust=np.array([0.5, 0.5, 0.5]),
        latency=np.array([0.1, 0.1, 0.1]),
        load=np.array([0.2, 0.2, 0.2]),
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
    )
    assert select(**args).chosen_rsu == 2


def test_no_candidates_returns_none():
    assert (
        select(
            timestep=0,
            vehicle_id=0,
            candidates=np.array([], dtype=int),
            trust=np.array([]),
            latency=np.array([]),
            load=np.array([]),
            alpha=1.0,
            beta=0.5,
            gamma=0.3,
        )
        is None
    )


def test_backhaul_segments_assigned_to_every_rsu(cfg):
    """L5: every RSU carries a backhaul_segment_id at topology build time."""
    topology = build_topology(cfg.topology, cfg.seeds.generator("topology"))
    ids = topology.backhaul_segment_id
    assert ids.shape == (topology.num_rsus,)
    assert set(ids.tolist()) == set(range(cfg.topology["num_backhaul_segments"]))


def test_same_segment_edge_feature_is_correct(cfg):
    """L6: same_segment is true exactly for RSU-RSU pairs sharing a segment."""
    seeds = cfg.seeds
    topology = build_topology(cfg.topology, seeds.generator("topology"))
    mobility = build_mobility(cfg.mobility, seeds.generator("mobility"))
    data = build_snapshot(topology, mobility.reset(), seeds.generator("features"))

    n_rsu = topology.num_rsus
    src, dst = data.edge_index
    flag = data.edge_attr[:, EDGE_COL["same_segment"]]

    for e in range(data.edge_index.shape[1]):
        i, j = int(src[e]), int(dst[e])
        if i < n_rsu and j < n_rsu:
            shared = (
                topology.backhaul_segment_id[i] == topology.backhaul_segment_id[j]
            )
            assert bool(flag[e]) == bool(shared)
        else:
            # A vehicle sits on no backhaul segment.
            assert flag[e] == 0.0


def test_graph_is_homogeneous_not_hetero(cfg):
    """L6: single edge type, one edge_index - explicitly not HeteroData."""
    from torch_geometric.data import Data

    seeds = cfg.seeds
    topology = build_topology(cfg.topology, seeds.generator("topology"))
    mobility = build_mobility(cfg.mobility, seeds.generator("mobility"))
    data = build_snapshot(topology, mobility.reset(), seeds.generator("features"))

    assert isinstance(data, Data)
    assert type(data).__name__ != "HeteroData"


def test_rsu_rsu_edges_exist(cfg):
    """L6: RSU-RSU edges are required - without them nothing propagates."""
    seeds = cfg.seeds
    topology = build_topology(cfg.topology, seeds.generator("topology"))
    mobility = build_mobility(cfg.mobility, seeds.generator("mobility"))
    data = build_snapshot(topology, mobility.reset(), seeds.generator("features"))

    n_rsu = topology.num_rsus
    src, dst = data.edge_index
    rsu_rsu = ((src < n_rsu) & (dst < n_rsu)).sum().item()
    assert rsu_rsu > 0


def test_scale_cap_enforced(cfg):
    """L9: the config loader refuses to exceed 30 RSUs / 100 vehicles."""
    over = {**cfg.topology, "num_rsus": 31}
    with pytest.raises(ValueError, match="L9 cap"):
        from trustgraph.config import _validate

        _validate(type(cfg)(**{**cfg.__dict__, "topology": over}))


def test_sumo_is_not_imported():
    """L7: SUMO is off the critical path and must not be a dependency."""
    import trustgraph.mobility as mob

    source = Path(mob.__file__).read_text(encoding="utf-8").lower()
    assert "import traci" not in source
    assert "import sumolib" not in source
    with pytest.raises(ValueError, match="unknown mobility source"):
        build_mobility({"source": "sumo", "num_vehicles": 1}, np.random.default_rng(0))

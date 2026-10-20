"""S17 (D81): the spreading-fault and cold-start (restart) scenario modes.

Unit tests of the injector (cheap), then scenario-level tests on the operating config (the
session-scoped demo trace is shared): L = 0 reproduces the shared-onset scenario exactly (F11 hash),
two generations from one (config, seed, draw) are identical, and a stored hash pins one spreading and
one restart scenario so the generator cannot drift after the pre-registration.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import load_config  # noqa: E402
from trustgraph.observed import load_observed  # noqa: E402
from trustgraph.scenario import build_snapshot_builder, build_world  # noqa: E402
from trustgraph.sealed.ground_truth import load_ground_truth  # noqa: E402
from trustgraph.sealed.injector import (  # noqa: E402
    INJECTION_PURPOSES,
    degraded_count,
    hop_distances,
    inject,
    restart_plan,
    RestartConfig,
)
from trustgraph.simulator import generate_scenario  # noqa: E402

CONFIGS = REPO_ROOT / "configs"
F11_HASH = "f3f247ac6c9becb1d2ddf859bf0b635a3c0d4778076ff028e46cfc0c97130de7"  # FINDINGS F11, rho 1.0, draw 0
# Stored at the S17 build (D81 section 11), config <x>.yaml, seed 20260903, rho 1.0, draw 0. Graph-sequence hash
# of the observable file after a disk round trip, then a hash of the sealed per-RSU arrays and task outcomes.
STORED = {
    "spread_L80": ("3d2fd91a4bf6b46078902fc5341d4f84828d34333e312a0037d28b928ee54a02", "fdc188f1cbd85ba002e01995ffb11189ec1aea5979bf3d319ee555dc5ba287c4"),
    "restart_f20": ("889d749943f2ad7ff046fe2e9bab9a83ebebee0219b710c495eb473807f8a382", "b57d5f78df7805f0b1f60b86690f54add1d2b8cbb00c732edbc26fee80e9038a"),
}

DEG = {"fraction": 0.2, "rho": 1.0, "onset_step": 400, "ramp_steps": 300}
SEG = np.repeat(np.arange(4), 5)
# a path 0-1-2-...-19 plus a chord, so hops are easy to state
PATH_EDGES = np.array([[i, i + 1] for i in range(19)], dtype=np.int64)


def _rngs(seed, draw=0):
    return {p: np.random.default_rng([seed, k, draw]) for k, p in enumerate(INJECTION_PURPOSES)}


# ------------------------------------------------------------------ hop distances and the spread rule


def test_hop_distances_on_a_path_and_unreachable_is_minus_one():
    hops = hop_distances(PATH_EDGES, 20, 0)
    assert hops.tolist() == list(range(20))
    assert hop_distances(PATH_EDGES[:5], 20, 0)[10] == -1  # nodes beyond the cut


@pytest.mark.parametrize("lag", [0, 20, 40, 80])
def test_spread_onset_is_lag_times_hop_from_a_degraded_seed(lag):
    gt = inject(SEG, {**DEG, "spread_lag_steps": lag}, None, None, _rngs(1), rsu_edges=PATH_EDGES)
    hop = gt.spread_hop
    members = np.flatnonzero(gt.degraded)
    assert hop is not None and (hop[~gt.degraded] == -1).all()
    assert (hop[members] >= 0).all() and (hop[members] == 0).sum() == 1  # exactly one seed, and it is degraded
    assert (gt.onset_per_rsu()[members] == 400 + lag * hop[members]).all()


def test_spread_changes_only_the_onsets_never_the_degraded_set():
    base = inject(SEG, DEG, None, None, _rngs(2))
    for lag in (0, 20, 40, 80):
        gt = inject(SEG, {**DEG, "spread_lag_steps": lag}, None, None, _rngs(2), rsu_edges=PATH_EDGES)
        assert np.array_equal(gt.degraded, base.degraded)
        assert np.array_equal(gt.behavior_class, base.behavior_class)
    # the seed RSU (and so the hops) does not depend on the lag either
    hops = [inject(SEG, {**DEG, "spread_lag_steps": lag}, None, None, _rngs(2), rsu_edges=PATH_EDGES).spread_hop for lag in (0, 40)]
    assert np.array_equal(hops[0], hops[1])


def test_spread_lag_zero_is_the_shared_onset():
    gt = inject(SEG, {**DEG, "spread_lag_steps": 0}, None, None, _rngs(3), rsu_edges=PATH_EDGES)
    base = inject(SEG, DEG, None, None, _rngs(3))
    assert np.array_equal(gt.onset_per_rsu(), base.onset_per_rsu())
    for step in (0, 399, 400, 401, 550, 700, 1199):
        assert np.array_equal(gt.degradation_level(step), base.degradation_level(step))
        assert np.array_equal(gt.capacity_factor(step), base.capacity_factor(step))


def test_fault_reaches_each_rsu_at_the_configured_lag():
    gt = inject(SEG, {**DEG, "spread_lag_steps": 40}, None, None, _rngs(4), rsu_edges=PATH_EDGES)
    for v in np.flatnonzero(gt.degraded):
        onset = 400 + 40 * int(gt.spread_hop[v])
        assert gt.degradation_level(onset)[v] == 0.0
        assert gt.degradation_level(onset + 1)[v] > 0.0
        assert gt.degradation_level(onset + 300)[v] == 1.0


def test_spread_is_off_by_default_and_refuses_bad_use():
    gt = inject(SEG, DEG, None, None, _rngs(1))
    assert gt.onset_steps is None and gt.spread_hop is None and gt.restart_steps is None
    with pytest.raises(ValueError):
        inject(SEG, {**DEG, "spread_lag_steps": -1}, None, None, _rngs(1), rsu_edges=PATH_EDGES)
    with pytest.raises(ValueError):  # needs the topology
        inject(SEG, {**DEG, "spread_lag_steps": 20}, None, None, _rngs(1))
    with pytest.raises(ValueError):  # separate from the staggered scenario
        inject(SEG, {**DEG, "spread_lag_steps": 20, "onset_spread_steps": 300}, None, None, _rngs(1), rsu_edges=PATH_EDGES)
    with pytest.raises(ValueError):  # a degraded RSU the seed cannot reach
        inject(SEG, {**DEG, "spread_lag_steps": 20}, None, None, _rngs(1), rsu_edges=PATH_EDGES[:3])


# ------------------------------------------------------------------ the restart rule

RESTART = {"fraction": 0.2, "step": 400, "rho": 1.0, "fail_fraction": 0.5, "fail_delay_steps": 20}
NO_DEG = {"fraction": 0.0, "rho": 1.0, "onset_step": 400, "ramp_steps": 100}


@pytest.mark.parametrize("f,n_restart,n_fail", [(0.10, 2, 1), (0.20, 4, 2), (0.40, 8, 4)])
def test_restart_sizes_and_the_failing_subset(f, n_restart, n_fail):
    gt = inject(SEG, NO_DEG, None, None, _rngs(5), cfg_restart={**RESTART, "fraction": f})
    restarted = gt.restart_steps >= 0
    assert restarted.sum() == n_restart == degraded_count(20, f)
    assert (gt.restart_steps[restarted] == 400).all() and (gt.restart_steps[~restarted] == -1).all()
    assert gt.degraded.sum() == n_fail
    assert (restarted | ~gt.degraded).all()  # failing implies restarted
    assert (gt.onset_per_rsu()[gt.degraded] == 420).all()
    assert gt.ramp_steps == 100
    assert gt.degradation_level(420)[gt.degraded].max() == 0.0
    assert gt.degradation_level(520)[gt.degraded].min() == 1.0


def test_restart_is_deterministic_and_clusters_by_segment_at_rho_one():
    a = inject(SEG, NO_DEG, None, None, _rngs(6), cfg_restart=RESTART)
    b = inject(SEG, NO_DEG, None, None, _rngs(6), cfg_restart=RESTART)
    assert np.array_equal(a.restart_steps, b.restart_steps) and np.array_equal(a.behavior_class, b.behavior_class)
    restarted, failing = restart_plan(SEG, RestartConfig(**RESTART), np.random.default_rng(1))
    assert np.unique(SEG[restarted]).size == 1  # 4 of a 5-RSU segment: one segment holds the whole restart set
    assert (restarted | ~failing).all()


def test_restart_refuses_to_mix_with_other_scenarios():
    with pytest.raises(ValueError):
        inject(SEG, DEG, None, None, _rngs(1), cfg_restart=RESTART)  # degradation.fraction > 0
    with pytest.raises(ValueError):
        inject(SEG, NO_DEG, None, {"num_nodes": 1, "join_step": 5}, _rngs(1), cfg_restart=RESTART)
    with pytest.raises(ValueError):
        inject(SEG, NO_DEG, None, None, _rngs(1), cfg_restart={**RESTART, "step": 0})


def test_colluders_are_never_failing_restarted_rsus():
    gt = inject(SEG, NO_DEG, {"num_groups": 1, "group_size": 3}, None, _rngs(7), cfg_restart=RESTART)
    assert not (gt.colluding & gt.degraded).any()


# ------------------------------------------------------------------ configs


def _load(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("lag", [0, 20, 40, 80])
def test_spread_configs_differ_from_demo_only_by_the_one_key(lag):
    demo, cfg = _load("demo"), _load(f"spread_L{lag}")
    assert cfg["degradation"].pop("spread_lag_steps") == lag
    assert cfg == demo


@pytest.mark.parametrize("pct,frac", [(10, 0.10), (20, 0.20), (40, 0.40)])
def test_restart_configs_differ_from_demo_only_as_registered(pct, frac):
    demo, cfg = _load("demo"), _load(f"restart_f{pct}")
    assert cfg.pop("restart") == {"fraction": frac, "step": 400, "rho": 1.0, "fail_fraction": 0.5, "fail_delay_steps": 20}
    assert cfg["degradation"].pop("fraction") == 0.0 and cfg["degradation"].pop("ramp_steps") == 100
    del demo["degradation"]["fraction"], demo["degradation"]["ramp_steps"]
    assert cfg == demo


@pytest.mark.parametrize("tag,rate", [("r10", 0.10), ("r04", 0.04), ("r02", 0.02)])
def test_sparse_configs_differ_from_demo_only_as_registered(tag, rate):
    """D89 section 1: 30 RSUs, 50 vehicles and the task rate; nothing else."""
    demo, cfg = _load("demo"), _load(f"sparse_{tag}")
    assert cfg["topology"].pop("num_rsus") == 30 and cfg["mobility"].pop("num_vehicles") == 50
    assert cfg["tasks"].pop("rate_per_vehicle_hz") == rate
    del demo["topology"]["num_rsus"], demo["mobility"]["num_vehicles"], demo["tasks"]["rate_per_vehicle_hz"]
    assert cfg == demo


# ------------------------------------------------------------------ scenario level (operating config)


def _scenario(demo_bundle, name, tmp_path):
    cfg0, world, trace = demo_bundle
    cfg = load_config(CONFIGS / f"{name}.yaml")
    sc = generate_scenario(cfg, world, trace)
    sc.observed.save(tmp_path / f"{name}.observed.npz")
    observed = load_observed(tmp_path / f"{name}.observed.npz")
    return cfg, world, trace, sc, observed


def graph_hash(cfg, world, trace, observed) -> str:
    digest = hashlib.sha256()
    for data in build_snapshot_builder(cfg, world, trace, observed).snapshots():
        digest.update(data.x.numpy().tobytes())
        digest.update(data.edge_index.numpy().tobytes())
        digest.update(data.edge_attr.numpy().tobytes())
    return digest.hexdigest()


def sealed_hash(sc) -> str:
    gt, digest = sc.ground_truth, hashlib.sha256()
    for arr in (gt.behavior_class, gt.onset_per_rsu(), gt.join_step, gt.true_load, gt.task_true_completion_s, gt.task_dropped):
        digest.update(np.ascontiguousarray(arr).tobytes())
    for arr in (gt.spread_hop, gt.restart_steps):
        digest.update(b"-" if arr is None else np.ascontiguousarray(arr).tobytes())
    t = sc.observed.tasks
    for arr in (t.rsu, t.met_deadline, t.observed_s):
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def test_lag_zero_reproduces_the_shared_onset_scenario_exactly(demo_bundle, tmp_path):
    """D81 section 11: spread_lag_steps = 0 is the control - the F11 hash, and every array of the plain run."""
    cfg, world, trace, sc, observed = _scenario(demo_bundle, "spread_L0", tmp_path)
    assert cfg.degradation["spread_lag_steps"] == 0
    assert graph_hash(cfg, world, trace, observed) == F11_HASH
    cfg0, world0, trace0 = demo_bundle
    plain = generate_scenario(cfg0, world0, trace0)
    assert sealed_hash_equal_ignoring_spread(sc, plain)
    assert np.array_equal(sc.observed.rsu_features, plain.observed.rsu_features)
    assert np.array_equal(sc.observed.tasks.met_deadline, plain.observed.tasks.met_deadline)


def sealed_hash_equal_ignoring_spread(a, b) -> bool:
    ga, gb = a.ground_truth, b.ground_truth
    return (
        np.array_equal(ga.behavior_class, gb.behavior_class)
        and np.array_equal(ga.onset_per_rsu(), gb.onset_per_rsu())
        and np.array_equal(ga.true_load, gb.true_load)
        and np.array_equal(ga.task_true_completion_s, gb.task_true_completion_s)
        and np.array_equal(ga.task_dropped, gb.task_dropped)
    )


def test_two_generations_from_one_config_and_seed_are_identical(demo_bundle, tmp_path):
    for name in ("spread_L80", "restart_f20"):
        *_, sc1, obs1 = _scenario(demo_bundle, name, tmp_path)
        *_, sc2, obs2 = _scenario(demo_bundle, name, tmp_path)
        assert sealed_hash(sc1) == sealed_hash(sc2)
        assert np.array_equal(obs1.rsu_features, obs2.rsu_features)
        assert np.array_equal(obs1.tasks.rsu, obs2.tasks.rsu)


@pytest.mark.parametrize("name", sorted(STORED))
def test_stored_hash_pins_the_generator(demo_bundle, tmp_path, name):
    cfg, world, trace, sc, observed = _scenario(demo_bundle, name, tmp_path)
    graph, sealed = STORED[name]
    assert graph_hash(cfg, world, trace, observed) == graph
    assert sealed_hash(sc) == sealed


def test_spread_scenario_timeline_follows_the_lag(demo_bundle, tmp_path):
    """The sealed state of an L = 80 scenario: the fault is injected at 400 + 80 x hop, and the realised
    capacity loss of every degraded RSU is zero before its own onset and positive after."""
    cfg, world, trace, sc, observed = _scenario(demo_bundle, "spread_L80", tmp_path)
    gt = sc.ground_truth
    members = np.flatnonzero(gt.degraded)
    assert gt.spread_hop[members].min() == 0 and gt.spread_hop[members].max() >= 1
    for v in members:
        onset = 400 + 80 * int(gt.spread_hop[v])
        assert gt.onset_per_rsu()[v] == onset
        assert gt.degradation_level(onset)[v] == 0.0 and gt.degradation_level(onset + 30)[v] > 0.0
    path = tmp_path / "x.SEALED.npz"
    gt.save(path)
    back = load_ground_truth(path)
    assert np.array_equal(back.spread_hop, gt.spread_hop) and np.array_equal(back.onset_steps, gt.onset_steps)


def test_restart_resets_the_tracker_at_the_restart_step_and_keeps_the_rsu_in_service(demo_bundle, tmp_path):
    cfg, world, trace, sc, observed = _scenario(demo_bundle, "restart_f20", tmp_path)
    gt = sc.ground_truth
    restarted = np.flatnonzero(gt.restart_steps == 400)
    assert restarted.size == 4
    from trustgraph.features import RSU_COL

    feats = observed.rsu_features
    assert observed.rsu_active[:, restarted].all()  # restarted RSUs never leave service
    # neutral state at the restart step, whatever the node had accumulated before it
    assert (feats[400, restarted, RSU_COL["success_ewma"]] == 1.0).all()
    assert (feats[400, restarted, RSU_COL["latency_dev"]] == 0.0).all()
    # not all RSUs sit at 1.0 just before it: the check above is not vacuous
    assert (feats[399, :, RSU_COL["success_ewma"]] < 1.0).any()

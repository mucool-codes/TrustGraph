"""S2: the task model, the execution model, the injector, and outcome tracking.

These are the properties H2 rests on. If onset is instantaneous, the graph has nothing
to anticipate; if rho does not actually move degradation onto segments, the correlation
sweep measures nothing; if a feature can see an outcome before it happened, every
detection-latency number is optimistic. Each of those would fail silently in a result,
so each is asserted here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trustgraph.config import with_overrides
from trustgraph.execution import ExecutionModel, coverage_load
from trustgraph.observed import load_observed
from trustgraph.scenario import build_world
from trustgraph.sealed.ground_truth import (
    COMPROMISED,
    DEGRADED,
    RELIABLE,
    load_ground_truth,
    sealed_path_for,
)
from trustgraph.sealed.injector import degraded_count, inject, select_degraded
from trustgraph.simulator import check_trace_matches, generate_scenario
from trustgraph.tasks import TASK_TYPES, TaskModel, build_task_model
from trustgraph.trace import Trace
from trustgraph.tracking import (
    NEUTRAL_LATENCY_DEV,
    NEUTRAL_SUCCESS_EWMA,
    DiscrepancyTracker,
    TrackingConfig,
    promise_kept,
)


def _rngs(seed: int) -> dict[str, np.random.Generator]:
    return {
        name: np.random.default_rng([seed, k])
        for k, name in enumerate(("degradation", "collusion", "cold_start"))
    }


# ---------------------------------------------------------------------- task model


def test_deadlines_grow_with_task_type_but_are_capped():
    model = TaskModel()
    types = np.arange(len(TASK_TYPES))
    deadlines = model.deadline_s(types, np.full(3, 3.0))
    assert np.all(np.diff(deadlines) > 0), "heavier tasks must get longer deadlines"

    generous = model.deadline_s(types, np.full(3, 1000.0))
    assert np.all(generous <= model.deadline_cap_s), "deadlines must not be unlimited"

    # Past the cap, a heavy task has less slack per unit of compute than a light one.
    nominal = model.cycles_of(types) / model.reference_capacity_cycles_per_s
    slack = (model.deadline_s(types, np.full(3, 4.0)) - model.comm_allowance_s) / nominal
    assert slack[-1] < slack[0]


def test_task_draws_are_fixed_size_and_follow_the_shares():
    model = TaskModel(rate_per_vehicle_hz=5.0)
    draws = model.draw_step(200_000, 1.0, np.random.default_rng(0))
    assert draws.offloads.shape == draws.type_idx.shape == draws.slack.shape == (200_000,)
    freq = np.bincount(draws.type_idx, minlength=3) / draws.type_idx.size
    assert np.allclose(freq, model.shares, atol=0.01)
    assert draws.slack.min() >= model.slack_min and draws.slack.max() <= model.slack_max


def test_task_model_rejects_bad_config():
    with pytest.raises(ValueError, match="sum to 1"):
        TaskModel(shares=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="non-decreasing"):
        TaskModel(cycles=(3e9, 1e9, 2e8))
    with pytest.raises(ValueError, match="unknown tasks"):
        build_task_model({"deadline": 1.0})


# ----------------------------------------------------------------- execution model


def test_coverage_load_is_the_s1_definition():
    load, queue = coverage_load(np.array([0, 5, 10, 15, 30]), 10.0)
    assert np.allclose(load, [0.0, 0.5, 1.0, 1.0, 1.0])
    assert np.allclose(queue, [0.0, 0.0, 0.0, 0.5, 1.0])


def test_service_time_rises_with_load_and_with_degradation():
    model = ExecutionModel()
    by_load = model.service_time_s(1e9, np.array([0.0, 0.5, 1.0]))
    assert np.all(np.diff(by_load) > 0)
    by_capacity = model.service_time_s(1e9, 0.5, np.array([1.0, 0.7, 0.4]))
    assert np.all(np.diff(by_capacity) > 0)


def test_honest_advertisement_is_the_noise_free_truth():
    """With no degradation, no lie and no noise, the promise is exactly the outcome."""
    model = ExecutionModel()
    sigma = model.noise_sigma
    advertised = model.advertised_completion_s(1e9, 0.02, 0.6)
    true, dropped = model.true_completion_s(
        1e9, 0.02, 0.6, 1.0, 0.0, noise_z=sigma / 2, drop_u=0.5
    )
    assert float(true) == pytest.approx(float(advertised))
    assert not bool(dropped)


def test_runtime_noise_is_mean_one():
    model = ExecutionModel()
    z = np.random.default_rng(1).standard_normal(400_000)
    t, _ = model.true_completion_s(1e9, 0.0, 0.0, 1.0, 0.0, z, np.ones_like(z))
    assert t.mean() / float(model.service_time_s(1e9, 0.0)) == pytest.approx(1.0, abs=0.005)


def test_dropped_task_never_completes():
    t, dropped = ExecutionModel().true_completion_s(1e9, 0.02, 0.5, 0.5, 1.0, 0.0, 0.0)
    assert bool(dropped) and np.isinf(t)


# ------------------------------------------------------------------------ tracking


def _tracker() -> DiscrepancyTracker:
    return DiscrepancyTracker(3, TrackingConfig(ewma_weight=0.5, latency_dev_cap=2.0))


def test_on_time_as_promised_leaves_features_neutral():
    tr = _tracker()
    tr.observe(0, advertised_s=0.2, deadline_s=0.3, observed_s=0.2, met_deadline=True)
    assert tr.success_ewma[0] == NEUTRAL_SUCCESS_EWMA
    assert tr.latency_dev[0] == NEUTRAL_LATENCY_DEV


def test_broken_promise_lowers_success_ewma():
    tr = _tracker()
    tr.observe(0, advertised_s=0.2, deadline_s=0.3, observed_s=0.3, met_deadline=False)
    assert tr.success_ewma[0] == pytest.approx(0.5)
    assert tr.latency_dev[0] > 0.0


def test_honest_busy_node_is_not_penalised_for_a_miss_it_advertised():
    """D8: busy-and-honest is the selection rule's problem, not trust's."""
    assert promise_kept(advertised_s=0.5, deadline_s=0.3, met_deadline=False)
    tr = _tracker()
    tr.observe(0, advertised_s=0.5, deadline_s=0.3, observed_s=0.3, met_deadline=False)
    assert tr.success_ewma[0] == NEUTRAL_SUCCESS_EWMA
    assert tr.latency_dev[0] == NEUTRAL_LATENCY_DEV


def test_features_are_discrepancy_not_the_advertised_value():
    """L8: two nodes advertising very different things but delivering exactly what they
    advertised look identical; the same observed time against different promises does
    not."""
    tr = _tracker()
    tr.observe(0, advertised_s=0.05, deadline_s=0.9, observed_s=0.05, met_deadline=True)
    tr.observe(1, advertised_s=0.80, deadline_s=0.9, observed_s=0.80, met_deadline=True)
    assert tr.success_ewma[0] == tr.success_ewma[1]
    assert tr.latency_dev[0] == tr.latency_dev[1]

    tr = _tracker()
    tr.observe(0, advertised_s=0.2, deadline_s=0.9, observed_s=0.4, met_deadline=True)
    tr.observe(1, advertised_s=0.4, deadline_s=0.9, observed_s=0.4, met_deadline=True)
    assert tr.latency_dev[0] > tr.latency_dev[1]


def test_latency_dev_is_bounded():
    tr = _tracker()
    for _ in range(20):
        tr.observe(0, advertised_s=0.01, deadline_s=1.0, observed_s=1.0, met_deadline=False)
    assert 0.0 <= tr.latency_dev[0] <= 1.0


def test_reset_forgets_history():
    tr = _tracker()
    tr.observe(2, advertised_s=0.1, deadline_s=0.3, observed_s=0.3, met_deadline=False)
    tr.reset(2)
    assert tr.success_ewma[2] == NEUTRAL_SUCCESS_EWMA
    assert tr.num_observations[2] == 0


# ------------------------------------------------------------------------ injector


@pytest.fixture(scope="module")
def demo_segments():
    from conftest import DEMO_CONFIG
    from trustgraph.config import load_config

    return build_world(load_config(DEMO_CONFIG)).topology.backhaul_segment_id


def test_degraded_count_matches_each_sweep_fraction(demo_segments):
    n = demo_segments.size
    assert [degraded_count(n, f) for f in (0.05, 0.10, 0.20, 0.30)] == [1, 2, 4, 6]
    for f, k in ((0.05, 1), (0.10, 2), (0.20, 4), (0.30, 6)):
        gt = inject(demo_segments, {"fraction": f, "rho": 0.5}, {}, {}, _rngs(3))
        assert int(gt.degraded.sum()) == k


def test_rho_one_degrades_whole_segments(demo_segments):
    """At rho = 1 at most one segment is partially degraded; the rest are all-or-none."""
    sizes = np.bincount(demo_segments)
    for k in (4, 6, 10):
        for seed in range(100):
            mask, correlated = select_degraded(
                demo_segments, k, 1.0, np.random.default_rng(seed)
            )
            counts = np.bincount(demo_segments[mask], minlength=sizes.size)
            partial = (counts > 0) & (counts < sizes)
            assert partial.sum() <= 1, f"k={k} seed={seed}: {counts} of {sizes}"
            assert correlated == k - 1


def test_rho_zero_is_uniform_independent_selection(demo_segments):
    n, k, draws = demo_segments.size, 4, 4000
    hits = np.zeros(n)
    concentration = []
    for seed in range(draws):
        mask, correlated = select_degraded(demo_segments, k, 0.0, np.random.default_rng(seed))
        assert correlated == 0
        hits += mask
        counts = np.bincount(demo_segments[mask])
        concentration.append((counts * (counts - 1) / 2).sum() / (k * (k - 1) / 2))

    assert np.allclose(hits / draws, k / n, atol=0.035)
    sizes = np.bincount(demo_segments)
    uniform = (sizes * (sizes - 1) / 2).sum() / (n * (n - 1) / 2)
    assert np.mean(concentration) == pytest.approx(uniform, abs=0.02)


def test_segment_concentration_rises_monotonically_with_rho(demo_segments):
    means = []
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        values = []
        for seed in range(1500):
            mask, _ = select_degraded(demo_segments, 4, rho, np.random.default_rng(seed))
            counts = np.bincount(demo_segments[mask])
            values.append((counts * (counts - 1) / 2).sum() / 6.0)
        means.append(np.mean(values))
    assert np.all(np.diff(means) > 0), means


def test_degraded_sets_are_nested_across_fractions(demo_segments):
    """Same seed and rho: the 10% set sits inside the 20% set inside the 30% set."""
    for rho in (0.0, 0.5, 1.0):
        sets = [
            select_degraded(demo_segments, k, rho, np.random.default_rng(11))[0]
            for k in (2, 4, 6)
        ]
        assert np.all(sets[0] <= sets[1]) and np.all(sets[1] <= sets[2])


def test_injector_draw_count_does_not_depend_on_rho_or_fraction(demo_segments):
    a = np.random.default_rng(5)
    select_degraded(demo_segments, 1, 0.0, a)
    b = np.random.default_rng(5)
    select_degraded(demo_segments, 6, 1.0, b)
    assert a.random() == b.random()


def test_behaviour_classes_colluders_and_cold_start(demo_segments):
    gt = inject(
        demo_segments,
        {"fraction": 0.2, "rho": 1.0},
        {"num_groups": 2, "group_size": 2, "under_report": 0.4},
        {"num_nodes": 3, "join_step": 50},
        _rngs(8),
    )
    assert int((gt.behavior_class == DEGRADED).sum()) == 4
    assert int((gt.behavior_class == COMPROMISED).sum()) == 4
    assert not np.any(gt.degraded & gt.colluding), "a node is degraded or colluding, not both"
    assert np.array_equal(gt.behavior_class == COMPROMISED, gt.colluding)
    assert set(np.bincount(gt.collusion_group[gt.colluding]).tolist()) == {2}
    assert int(gt.cold_start.sum()) == 3
    assert not np.any(gt.active(49)[gt.cold_start]) and np.all(gt.active(50))
    assert int((gt.behavior_class == RELIABLE).sum()) == 12


def test_degradation_onset_is_gradual(demo_segments):
    gt = inject(
        demo_segments,
        {"fraction": 0.2, "rho": 1.0, "onset_step": 100, "ramp_steps": 50, "severity": 0.6},
        {},
        {},
        _rngs(1),
    )
    d = gt.degraded
    assert np.all(gt.degradation_level(99)[d] == 0.0)
    assert np.all(gt.degradation_level(100)[d] == 0.0)
    assert np.allclose(gt.degradation_level(101)[d], 1 / 50)
    assert np.allclose(gt.degradation_level(125)[d], 0.5)
    assert np.all(gt.degradation_level(150)[d] == 1.0)
    assert np.all(gt.degradation_level(1000)[~d] == 0.0)
    levels = np.array([gt.capacity_factor(s)[d].mean() for s in range(90, 160)])
    assert np.all(np.diff(levels) <= 0) and np.sum(np.diff(levels) < 0) == 50


def test_injector_rejects_bad_config(demo_segments):
    with pytest.raises(ValueError, match="rho"):
        inject(demo_segments, {"rho": 1.5}, {}, {}, _rngs(0))
    with pytest.raises(ValueError, match="ramp_steps"):
        inject(demo_segments, {"ramp_steps": 0}, {}, {}, _rngs(0))
    with pytest.raises(ValueError, match="unknown degradation"):
        inject(demo_segments, {"correlation": 0.5}, {}, {}, _rngs(0))
    with pytest.raises(ValueError, match="collusion needs"):
        inject(demo_segments, {"fraction": 0.9}, {"num_groups": 2, "group_size": 3}, {}, _rngs(0))


# ----------------------------------------------------------------------- simulator


def test_scenario_is_reproducible(cfg, world, trace):
    a = generate_scenario(cfg, world, trace)
    b = generate_scenario(cfg, world, trace)
    assert np.array_equal(a.observed.rsu_features, b.observed.rsu_features)
    for name in a.observed.tasks.__dataclass_fields__:
        assert np.array_equal(
            getattr(a.observed.tasks, name), getattr(b.observed.tasks, name), equal_nan=True
        )
    assert np.array_equal(a.ground_truth.behavior_class, b.ground_truth.behavior_class)


def test_features_are_an_exact_replay_of_outcomes_already_observed(scenario, trace):
    """Causality, checked by oracle: re-derive every feature row from the task stream
    using only outcomes observed at or before that step. Because every outcome is
    observed strictly after its task was dispatched, row t can never contain a task
    from step t or later."""
    tasks = scenario.observed.tasks
    d = tasks.dispatched
    assert np.all(tasks.observed_at_s[d] > tasks.step[d] * trace.dt_s)

    tracker = DiscrepancyTracker(scenario.observed.num_rsus, TrackingConfig())
    order = np.flatnonzero(d)[np.lexsort((np.flatnonzero(d), tasks.observed_at_s[d]))]
    cursor = 0
    for t in range(trace.num_steps):
        while cursor < order.size and tasks.observed_at_s[order[cursor]] <= t * trace.dt_s:
            i = order[cursor]
            tracker.observe(
                int(tasks.rsu[i]), tasks.advertised_s[i], tasks.deadline_s[i],
                tasks.observed_s[i], bool(tasks.met_deadline[i]),
            )
            cursor += 1
        assert np.allclose(scenario.observed.feature("success_ewma")[t], tracker.success_ewma, atol=1e-6)
        assert np.allclose(scenario.observed.feature("latency_dev")[t], tracker.latency_dev, atol=1e-6)


def test_future_steps_cannot_change_past_features(demo_bundle):
    cfg, world, trace = demo_bundle
    cut = 600
    short_cfg = replace(cfg, scenario={**cfg.scenario, "num_steps": cut})
    short_trace = Trace(
        positions=trace.positions[:cut], velocities=trace.velocities[:cut],
        dt_s=trace.dt_s, seed=trace.seed, source=trace.source,
    )
    full = generate_scenario(cfg, world, trace).observed
    short = generate_scenario(short_cfg, world, short_trace).observed
    assert np.array_equal(full.rsu_features[:cut], short.rsu_features)


def test_advertised_load_is_true_load_except_for_colluders(scenario):
    obs, gt = scenario.observed, scenario.ground_truth
    advertised = obs.feature("load")
    true = np.where(obs.rsu_active, gt.true_load, 0.0)
    honest = ~gt.colluding
    assert gt.colluding.any(), "smoke config has no colluder; the test proves nothing"
    assert np.allclose(advertised[:, honest], true[:, honest], atol=1e-6)
    assert np.allclose(
        advertised[:, gt.colluding], (1 - gt.under_report) * true[:, gt.colluding], atol=1e-6
    )


def test_absent_rsus_never_receive_tasks(scenario):
    tasks, gt = scenario.observed.tasks, scenario.ground_truth
    d = tasks.dispatched
    assert np.all(gt.join_step[tasks.rsu[d]] <= tasks.step[d])


def test_outcomes_agree_with_sealed_completion_times(scenario):
    tasks, gt = scenario.observed.tasks, scenario.ground_truth
    d = tasks.dispatched
    true = gt.task_true_completion_s[d]
    assert np.array_equal(tasks.met_deadline[d], true <= tasks.deadline_s[d])
    assert np.allclose(tasks.observed_s[d], np.minimum(true, tasks.deadline_s[d]))
    assert not np.any(tasks.met_deadline[d][gt.task_dropped[d]])


def test_task_stream_is_paired_across_degradation_settings(demo_bundle):
    """Same seed: identical arrivals, deadlines and - with no colluders - dispatch,
    whatever rho and fraction are. Only outcomes differ, so sweep points are paired."""
    cfg, world, trace = demo_bundle
    a = generate_scenario(with_overrides(cfg, fraction=0.0), world, trace).observed.tasks
    b = generate_scenario(with_overrides(cfg, fraction=0.3, rho=0.0), world, trace).observed.tasks
    for name in ("step", "vehicle", "task_type", "deadline_s", "rsu", "advertised_s"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert not np.array_equal(a.met_deadline, b.met_deadline)


def test_observable_file_carries_no_ground_truth(tmp_path, scenario):
    path = scenario.observed.save(tmp_path / "s.observed.npz")
    sealed = scenario.ground_truth.save(sealed_path_for(path))
    assert sealed.name == "s.SEALED.npz"

    with np.load(path) as blob:
        keys = set(blob.files)
    for forbidden in ("behavior_class", "true_load", "collusion_group", "join_step",
                      "task_dropped", "task_true_completion_s", "rho", "degraded_fraction"):
        assert not any(forbidden in k for k in keys), forbidden

    back = load_observed(path)
    assert np.array_equal(back.rsu_features, scenario.observed.rsu_features)
    assert np.array_equal(back.tasks.met_deadline, scenario.observed.tasks.met_deadline)
    gt = load_ground_truth(sealed)
    assert np.array_equal(gt.behavior_class, scenario.ground_truth.behavior_class)

    # A sealed field smuggled into the observable file is refused on load.
    with np.load(path) as blob:
        contents = {k: blob[k] for k in blob.files}
    np.savez(tmp_path / "leaky.observed.npz", behavior_class=gt.behavior_class, **contents)
    with pytest.raises(ValueError, match="non-observable"):
        load_observed(tmp_path / "leaky.observed.npz")


def test_stale_trace_is_refused(cfg, trace):
    short = Trace(
        positions=trace.positions[:5], velocities=trace.velocities[:5],
        dt_s=trace.dt_s, seed=trace.seed, source=trace.source,
    )
    with pytest.raises(ValueError, match="does not match"):
        check_trace_matches(cfg, short)


def test_unknown_dispatch_policy_is_refused(cfg, world, trace):
    with pytest.raises(ValueError, match="dispatch.policy"):
        generate_scenario(replace(cfg, dispatch={"policy": "trust_aware"}), world, trace)


# ---------------------------------------------------------- calibration regressions


def test_baseline_success_is_high_but_not_perfect(demo_bundle):
    """The calibration target: 90-97% with no degradation (FINDINGS.md F7)."""
    cfg, world, trace = demo_bundle
    tasks = generate_scenario(with_overrides(cfg, fraction=0.0), world, trace).observed.tasks
    assert 0.90 <= tasks.success_rate() <= 0.97


def test_degradation_drops_success_meaningfully(demo_bundle):
    cfg, world, trace = demo_bundle
    baseline = generate_scenario(with_overrides(cfg, fraction=0.0), world, trace)
    for rho in (0.0, 1.0):
        s = generate_scenario(with_overrides(cfg, fraction=0.2, rho=rho), world, trace)
        tasks, gt = s.observed.tasks, s.ground_truth
        post = tasks.step >= gt.onset_step + gt.ramp_steps
        d = tasks.dispatched
        on_degraded = np.zeros(len(tasks), dtype=bool)
        on_degraded[d] = gt.degraded[tasks.rsu[d]]
        assert tasks.success_rate(post & on_degraded) < 0.5
        assert tasks.success_rate(post) < baseline.observed.tasks.success_rate(post) - 0.05


def test_own_success_ewma_does_not_collapse_at_onset(demo_bundle):
    """The CRITICAL S2 property. If a degraded node's own evidence collapsed the moment
    it degraded, neighbourhood evidence could never lead it and the GNN could not beat
    the MLP by construction. The decline must be spread across the ramp."""
    cfg, world, trace = demo_bundle
    s = generate_scenario(with_overrides(cfg, fraction=0.2, rho=1.0), world, trace)
    gt, ewma = s.ground_truth, s.observed.feature("success_ewma")
    onset, ramp = gt.onset_step, gt.ramp_steps
    degraded = ewma[:, gt.degraded].mean(axis=1)

    before = degraded[onset - 100 : onset].mean()
    plateau = degraded[onset + ramp : onset + ramp + 100].mean()
    assert before - plateau > 0.4, "degradation must be visible in the feature"
    assert degraded[onset + 10] > before - 0.1, "no collapse in the first 10 s"
    assert degraded[onset + ramp // 4] > (before + plateau) / 2, "still above midpoint a quarter in"

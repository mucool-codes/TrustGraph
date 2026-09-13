"""S2b: cold start as two separable conditions, and injection draws per seed.

The control and test conditions are only separable if the placement rule is exactly
what it claims: a control node never shares a segment with degradation, a test node is
always one of a degraded segment's degraded RSUs. Draws are only a cheap evaluation axis
if they vary which nodes misbehave without varying the traffic.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import DEMO_CONFIG
from trustgraph.config import load_config, with_overrides
from trustgraph.observed import scenario_stem
from trustgraph.scenario import build_world
from trustgraph.sealed.ground_truth import DEGRADED, RELIABLE
from trustgraph.sealed.injector import (
    INJECTION_PURPOSES,
    inject,
    injection_stream_name,
)
from trustgraph.simulator import generate_scenario


def _rngs(seed: int) -> dict[str, np.random.Generator]:
    return {
        name: np.random.default_rng([seed, k]) for k, name in enumerate(INJECTION_PURPOSES)
    }


@pytest.fixture(scope="module")
def demo_segments():
    return build_world(load_config(DEMO_CONFIG)).topology.backhaul_segment_id


def _cold(placement: str, num_nodes: int = 2) -> dict:
    return {"num_nodes": num_nodes, "join_step": 600, "placement": placement}


# ----------------------------------------------------------------------- placement


def test_uniform_placement_is_the_pre_s2b_rule(demo_segments):
    """Explicit `uniform` and an absent key give the same joiners as before S2b."""
    for seed in range(20):
        a = inject(demo_segments, {"fraction": 0.2, "rho": 0.5}, {}, {"num_nodes": 3, "join_step": 50}, _rngs(seed))
        b = inject(demo_segments, {"fraction": 0.2, "rho": 0.5}, {}, {**_cold("uniform", 3), "join_step": 50}, _rngs(seed))
        assert np.array_equal(a.join_step, b.join_step)
        expected = np.random.default_rng([seed, 2]).permutation(demo_segments.size)[:3]
        assert set(np.flatnonzero(a.join_step > 0).tolist()) == set(expected.tolist())


def test_control_nodes_join_segments_with_no_degradation(demo_segments):
    for rho in (0.0, 0.5, 1.0):
        for seed in range(100):
            try:
                gt = inject(
                    demo_segments,
                    {"fraction": 0.2, "rho": rho},
                    {"num_groups": 1, "group_size": 2},
                    _cold("healthy_segment"),
                    _rngs(seed),
                )
            except ValueError:
                continue  # infeasible draw - measured in FINDINGS.md F12
            cold = gt.cold_start
            touched = np.isin(demo_segments, demo_segments[gt.degraded])
            assert cold.sum() == 2
            assert not np.any(cold & touched), "control node on a degraded segment"
            assert not np.any(cold & gt.colluding), "control node is a colluder"
            assert np.all(gt.behavior_class[cold] == RELIABLE)


def test_test_nodes_are_degraded_rsus_of_degraded_segments(demo_segments):
    for rho in (0.0, 0.5, 1.0):
        for seed in range(100):
            gt = inject(demo_segments, {"fraction": 0.2, "rho": rho}, {}, _cold("degraded_segment"), _rngs(seed))
            cold = gt.cold_start
            assert cold.sum() == 2
            assert np.all(gt.behavior_class[cold] == DEGRADED)
            assert np.all(gt.segment_degraded_count()[demo_segments[cold]] >= 1)


def test_infeasible_conditions_are_refused_loudly(demo_segments):
    with pytest.raises(ValueError, match="infeasible"):
        inject(demo_segments, {"fraction": 0.0}, {}, _cold("degraded_segment"), _rngs(0))
    with pytest.raises(ValueError, match="infeasible"):
        inject(demo_segments, {"fraction": 0.9, "rho": 0.0}, {}, _cold("healthy_segment"), _rngs(0))
    with pytest.raises(ValueError, match="infeasible"):
        inject(demo_segments, {"fraction": 0.05}, {}, _cold("degraded_segment", 2), _rngs(0))


def test_unknown_placement_is_rejected(demo_segments):
    with pytest.raises(ValueError, match="placement"):
        inject(demo_segments, {}, {}, _cold("near_degraded"), _rngs(0))


# --------------------------------------------------------------------------- draws


def test_draw_zero_is_the_bare_stream_and_draws_are_distinct():
    assert injection_stream_name("degradation", 0) == "degradation"
    names = {injection_stream_name("degradation", d) for d in range(5)}
    assert len(names) == 5
    with pytest.raises(ValueError):
        injection_stream_name("degradation", -1)


def test_draws_vary_the_injection_but_not_the_traffic(demo_bundle):
    cfg, world, trace = demo_bundle
    base = generate_scenario(with_overrides(cfg, fraction=0.2, rho=0.0), world, trace)
    explicit0 = generate_scenario(with_overrides(cfg, fraction=0.2, rho=0.0, draw=0), world, trace)
    assert np.array_equal(base.ground_truth.behavior_class, explicit0.ground_truth.behavior_class)

    sets = {tuple(np.flatnonzero(base.ground_truth.degraded))}
    for draw in range(1, 6):
        s = generate_scenario(with_overrides(cfg, fraction=0.2, rho=0.0, draw=draw), world, trace)
        sets.add(tuple(np.flatnonzero(s.ground_truth.degraded)))
        for name in ("step", "vehicle", "task_type", "deadline_s", "rsu"):
            assert np.array_equal(getattr(s.observed.tasks, name), getattr(base.observed.tasks, name)), name
    assert len(sets) > 1, "draws did not change the degraded set"


def test_both_cold_start_conditions_run_end_to_end(demo_bundle):
    cfg, world, trace = demo_bundle
    for placement, cls in (("healthy_segment", RELIABLE), ("degraded_segment", DEGRADED)):
        c = with_overrides(cfg, fraction=0.2, rho=1.0, cold_start_nodes=1, cold_start_placement=placement)
        s = generate_scenario(c, world, trace)
        gt, obs = s.ground_truth, s.observed
        r = int(np.flatnonzero(gt.cold_start)[0])
        join = int(gt.join_step[r])
        assert gt.behavior_class[r] == cls
        assert not obs.rsu_active[:join, r].any() and obs.rsu_active[join:, r].all()
        tasks = obs.tasks
        assert not np.any(tasks.rsu[tasks.step < join] == r)
        assert np.any(tasks.rsu[tasks.step >= join] == r), "joined node never served a task"


def test_scenario_stem_names_draw_and_condition(demo_cfg):
    assert scenario_stem("configs/demo.yaml", demo_cfg) == "demo-seed20260903-rho1.00-frac0.20"
    c = with_overrides(demo_cfg, draw=7, cold_start_nodes=1, cold_start_placement="healthy_segment")
    assert scenario_stem("configs/demo.yaml", c) == (
        "demo-seed20260903-rho1.00-frac0.20-draw7-cold1-healthy_segment"
    )


def test_config_rejects_a_negative_draw(demo_cfg):
    from trustgraph.config import _validate

    with pytest.raises(ValueError, match="injection_draw"):
        _validate(with_overrides(demo_cfg, draw=-1))

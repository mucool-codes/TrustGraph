"""S6: the offloading engine - closed-loop equivalence, candidate set, handoff, log, variants."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from trustgraph.execution import build_execution_model
from trustgraph.model import build_trust_head
from trustgraph.offload import (
    TRUST_SOURCE,
    VARIANTS,
    OffloadEngine,
    OffloadSettings,
    build_offload_settings,
)
from trustgraph.selection import select
from trustgraph.simulator import generate_scenario, run_closed_loop


def _settings(variant, **kw):
    base = dict(variant=variant, alpha=1.0, beta=0.5, gamma=0.3, log_decisions=True)
    base.update(kw)
    return OffloadSettings(**base)


def _same_tasks(a, b):
    for name in a.__dataclass_fields__:
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name


# ------------------------------------------------------------------ equivalence


def test_closed_loop_A_without_dwell_or_handoff_is_the_open_loop_scenario(cfg, world, trace):
    open_ = generate_scenario(cfg, world, trace)
    closed = run_closed_loop(cfg, world, trace, _settings("A", dwell_filter=False, handoff="off"))
    _same_tasks(open_.observed.tasks, closed.observed.tasks)
    assert np.array_equal(open_.observed.rsu_features, closed.observed.rsu_features)
    assert np.array_equal(open_.observed.vehicle_task_demand, closed.observed.vehicle_task_demand)
    assert closed.observed.dispatch_policy == "closed_loop:A"
    assert open_.observed.dispatch_policy == "baseline_a"


def test_B_makes_the_same_decisions_as_A_because_cert_valid_is_constant(cfg, world, trace):
    a = run_closed_loop(cfg, world, trace, _settings("A"))
    b = run_closed_loop(cfg, world, trace, _settings("B"))
    _same_tasks(a.observed.tasks, b.observed.tasks)


def test_closed_loop_is_deterministic(cfg, world, trace):
    one = run_closed_loop(cfg, world, trace, _settings("B_oracle"))
    two = run_closed_loop(cfg, world, trace, _settings("B_oracle"))
    assert json.dumps(one.extras["decision_log"]) == json.dumps(two.extras["decision_log"])


# --------------------------------------------------------------------- variants


def test_variants_and_trust_sources_cover_the_d50_list():
    assert set(VARIANTS) == {"A", "B", "B_oracle", "C", "D", "E"}
    assert TRUST_SOURCE["E"] == TRUST_SOURCE["D"] == "model_D"
    with pytest.raises(ValueError):
        OffloadSettings(variant="F")
    with pytest.raises(ValueError):
        OffloadSettings(handoff="sometimes")


def test_settings_come_from_the_config_selection_weights(cfg):
    s = build_offload_settings(cfg, variant="C")
    assert (s.alpha, s.beta, s.gamma) == (1.0, 0.5, 0.3)
    assert build_offload_settings(cfg, variant="A").effective_alpha == 0.0


@pytest.mark.parametrize("variant", ["C", "D", "E"])
def test_model_variants_run_with_an_untrained_head_and_trust_is_a_probability(
    cfg, world, trace, variant
):
    head = "C" if variant == "C" else "D"
    model = build_trust_head(cfg.model, 0, "cpu", head).eval()
    out = run_closed_loop(cfg, world, trace, _settings(variant), model=model)
    log = out.extras["decision_log"]
    assert log
    for rec in log:
        for c in rec["candidates"]:
            assert 0.0 <= c["trust"] <= 1.0
            assert rec["trust_source"] == TRUST_SOURCE[variant]


def test_B_oracle_trust_is_zero_exactly_from_the_fault_step(cfg, world, trace):
    out = run_closed_loop(cfg, world, trace, _settings("B_oracle"))
    gt = out.ground_truth
    start = np.full(gt.num_rsus, 10**9)
    start[gt.degraded] = gt.onset_step
    start[gt.colluding] = 0
    seen_faulty = seen_healthy = False
    for rec in out.extras["decision_log"]:
        for c in rec["candidates"]:
            expect = 0.0 if rec["step"] >= start[c["rsu"]] else 1.0
            assert c["trust"] == expect
            seen_faulty |= expect == 0.0
            seen_healthy |= expect == 1.0
    assert seen_faulty and seen_healthy


# ------------------------------------------------- trust is a term, not a gate (L1)


def test_a_zero_trust_candidate_still_wins_when_latency_and_load_outweigh_it():
    cands = np.array([3, 7])
    trust = np.array([1.0, 0.0])
    latency = np.array([0.9, 0.0])
    load = np.array([0.9, 0.0])
    # alpha*1 - 0.5*0.9 - 0.3*0.9 = alpha - 0.72  vs  0
    low = select(0, 0, cands, trust, latency, load, alpha=0.5, beta=0.5, gamma=0.3)
    high = select(0, 0, cands, trust, latency, load, alpha=2.0, beta=0.5, gamma=0.3)
    assert low.chosen_rsu == 7  # zero-trust node chosen: no cutoff, no pre-filter
    assert high.chosen_rsu == 3


# ------------------------------------------------------------------ decision log


def test_log_records_every_candidate_with_terms_that_sum_to_the_score(cfg, world, trace):
    out = run_closed_loop(cfg, world, trace, _settings("B_oracle"))
    log = out.extras["decision_log"]
    tasks = out.observed.tasks
    assert [r["decision_id"] for r in log] == [
        i for i in range(len(tasks)) if tasks.num_candidates[i] > 0
    ]
    for rec in log:
        json.dumps(rec)  # serialisable
        assert rec["schema"] == 1 and rec["outcome"] is not None
        eligible = [c for c in rec["candidates"] if c["dwell_eligible"]]
        for c in rec["candidates"]:
            assert c["score"] == pytest.approx(c["trust_term"] + c["latency_term"] + c["load_term"])
            assert set(c["features"]) == {
                "load", "queue_depth", "cert_valid", "success_ewma", "latency_dev", "uptime_stability"
            }
        best = max(eligible, key=lambda c: (c["score"], -c["rsu"]))
        assert rec["selected_rsu"] == best["rsu"]
        assert rec["selected_rsu"] == tasks.rsu[rec["decision_id"]]
        assert rec["outcome"]["met_deadline"] == bool(tasks.met_deadline[rec["decision_id"]])


def test_the_observable_log_carries_no_sealed_field(cfg, world, trace):
    out = run_closed_loop(cfg, world, trace, _settings("B_oracle"))
    text = json.dumps(out.extras["decision_log"]).lower()
    for forbidden in ("behavior", "degraded", "compromised", "dropped", "true_", "handoff_lost"):
        assert forbidden not in text


def test_variant_A_logs_no_trust(cfg, world, trace):
    out = run_closed_loop(cfg, world, trace, _settings("A"))
    for rec in out.extras["decision_log"]:
        assert rec["weights"]["alpha"] == 0.0
        assert all(c["trust"] is None for c in rec["candidates"])


# ------------------------------------------------- candidate set and handoff, by hand


def _engine(positions, velocities, rsu_xy, radius=400.0, **kw):
    """An engine wired to hand-made geometry, bypassing the snapshot builder."""
    eng = OffloadEngine(cfg=SimpleNamespace(), settings=_settings("A", log_decisions=False, **kw))
    rsu_xy = np.asarray(rsu_xy, dtype=float)
    topo = SimpleNamespace(
        positions=rsu_xy,
        coverage_radius_m=radius,
        backhaul_segment_id=np.zeros(len(rsu_xy), dtype=int),
    )
    link = SimpleNamespace(latency_ms=lambda d, is_backhaul: 2.0 + 0.0 * np.asarray(d, dtype=float))
    eng._world = SimpleNamespace(topology=topo, link_model=link)
    eng._trace = SimpleNamespace(
        positions=np.asarray(positions, dtype=float)[None], velocities=np.asarray(velocities, dtype=float)[None], dt_s=1.0
    )
    eng._exec = build_execution_model({})
    eng._rsu_active = np.ones((1, len(rsu_xy)), dtype=bool)
    eng._trust = np.zeros(len(rsu_xy))
    eng._rsu_features = np.zeros((1, len(rsu_xy), 6))
    return eng


def test_handoff_unchanged_when_the_vehicle_is_still_in_the_serving_rsus_coverage():
    eng = _engine([[0.0, 0.0]], [[10.0, 0.0]], [[0.0, 0.0], [600.0, 0.0]])
    assert eng.apply_handoff(0, 0, 0, 0.2, 0.5) == (0.2, False, False)


def test_handoff_relays_over_the_backhaul_when_another_rsu_covers_the_vehicle():
    # The vehicle at x=395 m moving 20 m/s is at 399+ m... push it clearly out: x=395, 0.5 s -> 405 m.
    eng = _engine([[395.0, 0.0]], [[20.0, 0.0]], [[0.0, 0.0], [700.0, 0.0]])
    completion, relayed, lost = eng.apply_handoff(0, 0, 0, 0.5, 1.0)
    assert (relayed, lost) == (True, False)
    assert completion == pytest.approx(0.5 + 0.002)  # one-way backhaul latency of 2 ms
    assert eng.counters.relayed == 1


def test_handoff_loses_the_result_when_no_rsu_covers_the_vehicle():
    eng = _engine([[395.0, 0.0]], [[20.0, 0.0]], [[0.0, 0.0], [2000.0, 0.0]])
    completion, relayed, lost = eng.apply_handoff(0, 0, 0, 0.5, 1.0)
    assert np.isinf(completion) and (relayed, lost) == (False, True)
    assert eng.counters.handoff_lost == 1


def test_handoff_off_and_already_late_tasks_are_left_alone():
    off = _engine([[395.0, 0.0]], [[20.0, 0.0]], [[0.0, 0.0]], handoff="off")
    assert off.apply_handoff(0, 0, 0, 0.5, 1.0) == (0.5, False, False)
    on = _engine([[395.0, 0.0]], [[20.0, 0.0]], [[0.0, 0.0]])
    assert on.apply_handoff(0, 0, 0, 1.5, 1.0) == (1.5, False, False)
    assert on.apply_handoff(0, 0, 0, float("inf"), 1.0)[0] == float("inf")


def _dispatch(eng, covered):
    n = len(covered)
    return eng.dispatch(
        t=0, task_idx=0, vehicle=0, covered_row=np.asarray(covered), dist_row=np.full(n, 100.0),
        latency_ms_row=np.full(n, 4.0), latency_norm_row=np.full(n, 0.16), adv_load=np.zeros(n),
        cycles=1.0e9, deadline_s=0.5, type_idx=1,
    )


def test_dwell_filter_drops_an_rsu_the_vehicle_is_about_to_leave():
    # Vehicle at x=390 heading +x at 20 m/s: 0.5 s of dwell in RSU 0 (needs ~0.1+ s compute), but RSU 1 is
    # centred on the vehicle's side with a long dwell. Make RSU 0 infeasible with a heavy task by speed.
    eng = _engine([[398.0, 0.0]], [[20.0, 0.0]], [[0.0, 0.0], [300.0, 0.0]])
    decision, _ = _dispatch(eng, [True, True])
    assert decision.chosen_rsu == 1 and decision.candidates == (1,)
    assert eng.counters.dwell_excluded == 1 and eng.counters.dwell_fallback == 0


def test_dwell_filter_falls_back_to_every_in_range_rsu_when_none_is_feasible():
    eng = _engine([[399.5, 0.0]], [[20.0, 0.0]], [[0.0, 0.0]])
    decision, _ = _dispatch(eng, [True])
    assert decision.chosen_rsu == 0
    assert eng.counters.dwell_fallback == 1


def test_dwell_filter_off_keeps_every_in_range_rsu():
    eng = _engine([[398.0, 0.0]], [[20.0, 0.0]], [[0.0, 0.0], [300.0, 0.0]], dwell_filter=False)
    decision, _ = _dispatch(eng, [True, True])
    assert set(decision.candidates) == {0, 1}

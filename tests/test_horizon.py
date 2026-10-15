"""S15 (D72): the horizon objective's labels, checked against a hand computation and a brute-force loop."""

from __future__ import annotations

import numpy as np
import pytest

from trustgraph.dataset import (
    HORIZON_MIN_TASKS,
    HORIZON_STEPS,
    HORIZON_THRESHOLD,
    horizon_labels,
    labels_for,
    next_task_labels,
)
from trustgraph.observed import ObservedScenario, TaskStream


def _stream(step, rsu, met) -> TaskStream:
    n = len(step)
    z = np.zeros(n)
    return TaskStream(
        step=np.asarray(step, np.int64), vehicle=np.zeros(n, np.int64), task_type=np.zeros(n, np.int64),
        cycles=z, deadline_s=z, rsu=np.asarray(rsu, np.int64), num_candidates=np.ones(n, np.int64),
        advertised_s=z, observed_s=z, met_deadline=np.asarray(met, bool), observed_at_s=z,
    )


def _obs(tasks, num_steps, active=None) -> ObservedScenario:
    num_rsus = 2
    act = np.ones((num_steps, num_rsus), bool) if active is None else active
    return ObservedScenario(
        rsu_features=np.zeros((num_steps, num_rsus, 6), np.float32), rsu_active=act,
        vehicle_task_demand=np.zeros((num_steps, 1), np.float32), tasks=tasks, seed=0, dispatch_policy="test",
    )


def test_d72_constants_are_the_preregistered_ones():
    assert (HORIZON_STEPS, HORIZON_THRESHOLD, HORIZON_MIN_TASKS) == (150, 0.75, 5)


def test_hand_computed_window_is_open_closed_and_excludes_thin_windows():
    # RSU 0: tasks at steps 10..19 (10 tasks); the first 3 fail. RSU 1: 4 tasks only.
    step = list(range(10, 20)) + [30, 31, 32, 33]
    rsu = [0] * 10 + [1] * 4
    met = [False] * 3 + [True] * 7 + [True] * 4
    obs = _obs(_stream(step, rsu, met), 400)
    z, valid = horizon_labels(obs, horizon=150, threshold=0.75, min_tasks=5)
    # t = 9: window (9, 159] holds all 10 tasks, 7/10 = 0.7 < 0.75 -> y = 1 -> z = 0
    assert valid[9, 0] and z[9, 0] == 0.0
    # t = 10: window (10, 160] excludes the task at step 10 (open at t): 9 tasks, 2 failed, 7/9 = 0.778 -> healthy
    assert valid[10, 0] and z[10, 0] == 1.0
    # t = 12: tasks at 13..19: 7 tasks, all met -> healthy
    assert valid[12, 0] and z[12, 0] == 1.0
    # t = 14: tasks at 15..19: exactly 5 tasks -> included; t = 15: 16..19 = 4 tasks -> excluded
    assert valid[14, 0] and not valid[15, 0]
    # the window closes at t + H: at t = 170 the task at step 19 is out (19 <= 170) and none is ahead
    assert not valid[170, 0]
    # t = 169: (169, 319] -> none; t = 0: window (0, 150] has all 10
    assert valid[0, 0]
    # RSU 1 never reaches 5 tasks in any window
    assert not valid[:, 1].any()


def test_inactive_steps_are_masked():
    step = list(range(10, 20))
    obs = _obs(_stream(step, [0] * 10, [True] * 10), 100, active=np.concatenate([np.zeros((5, 2), bool), np.ones((95, 2), bool)]))
    _, valid = horizon_labels(obs)
    assert not valid[:5, 0].any() and valid[5, 0]


def test_matches_a_brute_force_loop_on_a_random_stream():
    rng = np.random.default_rng(0)
    n, steps = 600, 500
    st = np.sort(rng.integers(0, steps, n))
    rsu = rng.integers(-1, 2, n)  # -1 = undispatched, must never count
    met = rng.random(n) < 0.6
    obs = _obs(_stream(st, rsu, met), steps)
    z, valid = horizon_labels(obs)
    for v in (0, 1):
        for t in range(0, steps, 3):
            sel = (rsu == v) & (st > t) & (st <= t + 150)
            ok = sel.sum() >= 5
            assert valid[t, v] == ok
            if ok:
                assert z[t, v] == float(met[sel].mean() >= 0.75)


def test_next_objective_unchanged_and_dispatch(observed):
    y0, v0 = next_task_labels(observed)
    y1, v1 = labels_for(observed, "next")
    assert np.array_equal(y0, y1) and np.array_equal(v0, v1)
    zh, vh = labels_for(observed, "horizon")
    assert zh.shape == y0.shape and vh.shape == v0.shape
    with pytest.raises(ValueError):
        labels_for(observed, "bogus")

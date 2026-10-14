"""S9: the sweep metrics, the D56 label rule, the grid definition and resumability (hand-made data only)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from trustgraph import sweep_metrics as M  # noqa: E402
import s9_sweep as S  # noqa: E402

STEPS = 1200


def _stream(share_pre: float, share_post, switch: int = 500, per_step: int = 6):
    """Tasks with `per_step` dispatches each step; a fraction go to RSU 0 (the degraded one), the rest to RSU 1."""
    step, rsu = [], []
    for t in range(STEPS):
        share = share_pre if t < switch else share_post
        for k in range(per_step):
            step.append(t)
            rsu.append(0 if (k + t) % 8 < share * 8 else 1)
    return np.array(step), np.array(rsu)


def test_avoidance_found_after_the_share_drops():
    step, rsu = _stream(0.25, 0.0, switch=500)
    deg = np.array([True, False])
    total, d = M.dispatch_series(step, rsu, deg, STEPS)
    av = M.avoidance(total, d, 400)
    assert 0.2 < av["s0"] < 0.3
    assert not av["never_avoided"]
    assert 430 <= av["t_avoid"] <= 500  # the window [t, t+60) is mostly post-drop from ~ t = 470
    assert av["n_av"] < av["n_win"]
    assert av["n_win"] == int(d[400:].sum())
    assert av["n_av"] == int(d[400:av["t_avoid"]].sum())


def test_never_avoided_counts_every_post_onset_dispatch():
    step, rsu = _stream(0.25, 0.25)
    deg = np.array([True, False])
    total, d = M.dispatch_series(step, rsu, deg, STEPS)
    av = M.avoidance(total, d, 400)
    assert av["never_avoided"] and av["t_avoid"] == STEPS
    assert av["n_av"] == av["n_win"]


def test_a_unsent_task_is_not_a_dispatch():
    step = np.array([5, 5, 6]); rsu = np.array([0, -1, 1])
    total, d = M.dispatch_series(step, rsu, np.array([True, False]), 10)
    assert total.tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0] and d[5] == 1 and d.sum() == 1


def test_detection_latency_and_censoring():
    steps, n = 1200, 4
    trust = np.ones((steps, n))
    trust[450:, 1] = 0.0        # degraded RSU 1 collapses at 450
    active = np.ones((steps, n), bool)
    degraded = np.array([False, True, True, False])  # RSU 2 never drops
    r = M.detection(trust, active, degraded, 400)
    assert r["tau"] == 1.0
    assert r["detected_fraction"] == 0.5
    assert r["latency"] == pytest.approx((50 + 399) / 2)
    # no trust signal (Variant A): nothing can be below tau = 0
    r0 = M.detection(np.zeros((steps, n)), active, degraded, 400)
    assert r0["detected_fraction"] == 0.0 and r0["latency"] == 399


def test_detection_needs_ten_consecutive_steps():
    steps = 1200
    trust = np.ones((steps, 1)); trust[500:509, 0] = 0.0  # nine steps only
    r = M.detection(trust, np.ones((steps, 1), bool), np.array([True]), 400)
    assert r["detected_fraction"] == 0.0
    trust[500:510, 0] = 0.0  # ten steps
    r = M.detection(trust, np.ones((steps, 1), bool), np.array([True]), 400)
    assert r["detected_fraction"] == 1.0 and r["latency"] == 100


@pytest.mark.parametrize(
    "mean, lo, hi, floor, bound, expected",
    [
        (0.03, 0.01, 0.05, 0.01, 0.025, "WIN"),
        (0.03, -0.01, 0.07, 0.01, 0.025, "NOISY"),
        (0.005, -0.004, 0.014, 0.01, 0.025, "NULL"),
        (-0.02, -0.03, -0.01, 0.01, 0.025, "NEGATIVE"),
        (-0.005, -0.03, 0.02, 0.01, 0.025, "NULL"),
        (0.008, 0.002, 0.014, 0.01, 0.025, "NULL"),  # significant but under the floor, bound ruled out
        (0.0, -0.05, 0.05, 0.01, 0.025, "NOISY"),
    ],
)
def test_label_rule(mean, lo, hi, floor, bound, expected):
    assert M.label(mean, lo, hi, floor, bound) == expected


def test_t_interval_needs_five_seeds_and_uses_the_df4_value():
    with pytest.raises(ValueError):
        M.t_interval(np.ones(4))
    m, sd, lo, hi = M.t_interval(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert m == 3.0 and sd == pytest.approx(np.sqrt(2.5))
    assert hi - m == pytest.approx(2.776 * np.sqrt(2.5) / np.sqrt(5))


def test_h1_win_is_downgraded_when_the_fixed_window_guard_fails():
    rng = np.random.default_rng(0)
    ref = rng.uniform(90, 110, (5, 50))
    x = ref * 0.5
    win_guard = ref * 1.0  # the guard shows no reduction at all
    primary, guard = M.h1_effect(x, ref, win_guard, ref)
    assert primary.label == "NOISY"
    primary, guard = M.h1_effect(x, ref, x, ref)
    assert primary.label == "WIN" and guard.label == "WIN"


def test_h1_zero_reference_is_an_error():
    with pytest.raises(ValueError):
        M.relative_reduction(np.ones((5, 3)), np.zeros((5, 3)))


# ------------------------------------------------------------------ grid and resumability

def test_the_production_grid_is_two_1d_sweeps_with_one_shared_point():
    sw = S.load_sweep("configs/sweep.yaml")
    pts = S.points(sw)
    assert len(pts) == 8  # 5 + 4 - 1, not a cross product (L12)
    assert S.shared_point(sw) == (1.0, 0.2)
    assert sorted(pts[(1.0, 0.2)]) == ["fraction", "rho"]
    assert sw["seeds"] == [1, 2, 3, 4, 5] and sw["draws"] == list(range(100, 150))
    assert len(S.all_tasks(sw)) == 8 * 5 * 50


def test_validation_rejects_the_tuning_seed_and_training_draws():
    sw = S.load_sweep("configs/sweep.yaml")
    bad = copy.deepcopy(sw); bad["seeds"] = [1, 2, 3, 4, 100]
    with pytest.raises(ValueError):
        S.validate(bad)
    bad = copy.deepcopy(sw); bad["draws"] = [99] + sw["draws"][1:]
    with pytest.raises(ValueError):
        S.validate(bad)


def test_a_shared_point_that_disagrees_is_caught():
    sw = S.load_sweep("configs/sweep.yaml")
    bad = copy.deepcopy(sw); bad["fraction_sweep"]["rho"] = 0.9  # the fraction sweep no longer passes through the rho sweep
    with pytest.raises(AssertionError):
        S.assert_shared_point(bad)


def test_resume_skips_completed_tasks_and_leaves_no_temp_files(tmp_path):
    sw = S.load_sweep("configs/sweep_smoke.yaml")
    sw = copy.deepcopy(sw); sw["out_dir"] = str(tmp_path)
    S.REPO_ROOT, old = tmp_path, S.REPO_ROOT  # task paths resolve under REPO_ROOT / out_dir
    try:
        sw["out_dir"] = "out"
        todo, total = S.pending(sw)
        assert len(todo) == total == 8 * 5 * 2
        done = todo[:7]
        for t in done:
            S.write_atomic(S.task_path(sw, *t), "{}")
        todo2, _ = S.pending(sw)
        assert len(todo2) == total - 7 and not set(done) & set(todo2)
        assert not list((tmp_path / "out").rglob("*.tmp*"))
    finally:
        S.REPO_ROOT = old


def test_config_hash_changes_with_the_grid_but_not_the_worker_count():
    sw = S.load_sweep("configs/sweep.yaml")
    h = S.config_hash(sw)
    w2 = copy.deepcopy(sw); w2["workers"] = 3
    assert S.config_hash(w2) == h
    d2 = copy.deepcopy(sw); d2["draws"] = sw["draws"][:-1]
    assert S.config_hash(d2) != h

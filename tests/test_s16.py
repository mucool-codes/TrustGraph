"""S16 (D76): the detection rule, the false-alarm rule and the decision labels, checked on hand-built series."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import s16_confirm as S16  # noqa: E402

N = S16.LAST - S16.FIRST  # 1100 stored steps, index 0 = step 100


def _col(low_steps=()):
    c = np.ones(N, np.float32)
    for a, b in low_steps:
        c[a - 100:b - 100] = 0.0
    return c


def test_tau_is_first_percentile_of_pre_onset_rows_only():
    t1 = np.ones((N, 2))
    t1[300:, :] = -5.0  # post-onset rows must not enter tau
    t1[:300, 0] = np.linspace(0, 1, 300)
    tau = S16.pre_onset_tau([t1])
    assert tau == np.percentile(np.concatenate([np.linspace(0, 1, 300), np.ones(300)]), 1.0)


def test_false_alarm_needs_ten_consecutive_strictly_below():
    assert S16.false_alarm(_col([(500, 509)]), 0.5) == (False, False, False)  # 9 steps
    assert S16.false_alarm(_col([(500, 510)]), 0.5) == (True, False, True)  # 10 steps, post-onset
    c = _col()
    c[400 - 100:410 - 100] = 0.5  # equal to tau is not below
    assert S16.false_alarm(c, 0.5) == (False, False, False)


def test_false_alarm_split_pre_and_post():
    assert S16.false_alarm(_col([(100, 110)]), 0.5) == (True, True, False)  # first possible run
    assert S16.false_alarm(_col([(395, 405)]), 0.5) == (True, True, False)  # starts before 400
    assert S16.false_alarm(_col([(396, 405)]), 0.5) == (False, False, False)  # 9 steps across 400
    assert S16.false_alarm(_col([(1190, 1200)]), 0.5) == (True, False, True)  # last possible run


def test_run_starts_are_steps():
    assert S16.run_starts(_col([(700, 711)]), 0.5).tolist() == [700, 701]


def test_latency_labels():
    assert S16.label_latency(-25, -40, -10) == "EARLIER"
    assert S16.label_latency(-19, -30, -8) == "NOISY"  # |mean| below 20 and interval not inside +-20
    assert S16.label_latency(25, 10, 40) == "LATER"
    assert S16.label_latency(-5, -15, 5) == "NULL"
    assert S16.label_latency(-5, -25, 15) == "NOISY"


def test_false_alarm_label_and_overall():
    assert S16.label_false_alarm(0.05) == "ACCEPTABLE"
    assert S16.label_false_alarm(0.0501) == "NOT ACCEPTABLE"
    assert S16.overall("EARLIER", "ACCEPTABLE") == "OBJECTIVE HELPS"
    assert S16.overall("EARLIER", "NOT ACCEPTABLE") == "NOT CONFIRMED"
    assert S16.overall("NULL", "ACCEPTABLE") == "NOT CONFIRMED"


def test_draws_are_fresh_and_fixed():
    assert S16.EVAL_DRAWS == tuple(range(350, 400))
    assert S16.RHOS == (0.0, 1.0) and S16.FRACTION == 0.20 and S16.HOLD == 10

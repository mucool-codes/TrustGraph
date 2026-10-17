"""S18 (D81, D83): the verdict labels, the window AUC and the populations, checked on hand-built data (no generated artefacts)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import s18_run as S18  # noqa: E402


def test_latency_label_order_and_floors():
    assert S18.label_latency(-25, -40, -10) == "WIN"
    assert S18.label_latency(-19.9, -40, -1) != "WIN"  # floor 20
    assert S18.label_latency(25, 10, 40) == "NEGATIVE"
    assert S18.label_latency(5, -10, 15) == "NULL"
    assert S18.label_latency(-25, -60, 5) == "NOISY"  # interval reaches 0
    assert S18.label_latency(0, -20, 5) == "NOISY"  # lo = -20 is not > -20


def test_false_alarm_tolerance_and_strict_reading():
    assert S18.label_fa(0.05) == "ACCEPTABLE" and S18.label_fa(0.0501) == "NOT ACCEPTABLE"
    assert S18.fa_strict(0.0) and not S18.fa_strict(0.001)


def test_success_label_is_d81_section_9():
    assert S18.label_success(-0.02, -0.03, -0.005) == "NEGATIVE"
    assert S18.label_success(0.02, 0.005, 0.03) == "WIN"
    assert S18.label_success(0.0, -0.01, 0.02) == "NULL"  # hi < 0.025
    assert S18.label_success(0.0, -0.03, 0.03) == "NOISY"
    assert S18.label_success(0.009, 0.001, 0.02) == "NULL"  # below the 0.01 floor


@pytest.mark.parametrize("labels,expected", [
    (["WIN", "WIN", "NULL"], "SUPPORTED"), (["WIN", "WIN", "WIN"], "SUPPORTED"), (["NEGATIVE", "NEGATIVE", "WIN"], "CONTRADICTED"),
    (["WIN", "NULL", "NOISY"], "NOT SEPARATED"), (["WIN", "NEGATIVE", "NULL"], "NOT SEPARATED"), (["NULL", "NULL", "NULL"], "NOT SEPARATED"),
])
def test_sb_overall(labels, expected):
    assert S18.sb_overall(labels) == expected


def test_sa_needs_both_lags_and_both_labels():
    ok = ("WIN", "ACCEPTABLE")
    assert S18.sa_overall({"L40": ok, "L80": ok}) == "SUPPORTED"
    assert S18.sa_overall({"L40": ok, "L80": ("WIN", "NOT ACCEPTABLE")}) == "NOT SUPPORTED"
    assert S18.sa_overall({"L40": ok, "L80": ("NULL", "ACCEPTABLE")}) == "NOT SUPPORTED"


def test_sc_control():
    assert S18.sc_overall("NULL") == "HOLDS"
    assert S18.sc_overall("WIN").startswith("VIOLATED") and S18.sc_overall("NEGATIVE").startswith("VIOLATED")
    assert S18.sc_overall("NOISY").startswith("INCONCLUSIVE")


def test_window_auc_perfect_and_chance():
    n = S18.LAST - S18.FIRST
    t = np.ones((n, 5), np.float32)
    t[200:300, 0] = 0.1  # RSU 0 low in [300, 400)
    start = np.full(5, 300)
    assert S18.window_auc(t, [0], [1, 2, 3, 4], start) == 1.0
    assert S18.window_auc(np.ones((n, 5), np.float32), [0], [1, 2, 3, 4], start) == 0.5  # ties count half
    t2 = np.ones((n, 5), np.float32)
    t2[200:300, 0] = 2.0  # faulty RSU looks MORE trusted
    assert S18.window_auc(t2, [0], [1, 2, 3, 4], start) == 0.0


def test_window_uses_each_rsus_own_start():
    n = S18.LAST - S18.FIRST
    t = np.ones((n, 3), np.float32)
    t[400:500, 0] = 0.0  # low in [500, 600) only
    assert S18.window_auc(t, [0], [1, 2], np.array([500, 0, 0])) == 1.0
    assert S18.window_auc(t, [0], [1, 2], np.array([300, 0, 0])) == 0.5


def test_registry_matches_d81():
    assert tuple(S18.CONDS) == ("L0", "L20", "L40", "L80", "f10", "f20", "f40")
    assert S18.EVAL_DRAWS[0] == 400 and S18.EVAL_DRAWS[-1] == 449 and len(S18.EVAL_DRAWS) == 50
    train = set(S18.TRAIN_DRAWS) | {S18.VAL_DRAW}
    assert all(600 <= d <= 649 for d in train) and not train & set(S18.EVAL_DRAWS) and S18.HARNESS_DRAW >= 900
    assert (S18.LAT_FLOOR, S18.FA_BOUND, S18.SUCC_FLOOR, S18.SUCC_BOUND) == (20.0, 0.05, 0.01, 0.025)


def test_weights_load_and_retrained_heads_keep_primary_weights():
    w = S18.load_weights()
    assert w["C_r"] == w["C"] and w["D_r"] == w["D"] and w["A"][0] == 0.0
    assert set(w) >= set(S18.CL_NAMES["primary"]) | set(S18.CL_NAMES["secondary"])

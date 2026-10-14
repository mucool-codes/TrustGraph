"""S9 per-run metrics and the paired statistics of the sweep (DECISIONS.md D56).

Pure functions of arrays: nothing here imports the sealed ground truth (L4). The caller (the
evaluation-side runner) passes the degraded mask in, so the metric definitions can be tested on
hand-made data and the analysis can run on result files alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T_CRIT_DF4 = 2.776  # Student-t, 95% two-sided, 5 seeds (D43)

# D56 section 2 constants.
PRE_WINDOW = (100, 400)
AVOID_WINDOW = 60
AVOID_HOLD = 30
AVOID_RATIO = 0.75
DETECT_HOLD = 10
TRUST_PERCENTILE = 1.0

# D56 section 4: floor f and ruling-out bound b per hypothesis.
FLOORS = {"H3": (0.01, 0.025), "H1": (0.10, 0.25)}


# ------------------------------------------------------------------ H1: dispatches to degraded RSUs

def dispatch_series(
    task_step: np.ndarray, task_rsu: np.ndarray, degraded: np.ndarray, num_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step count of dispatched tasks and of those dispatched to a degraded RSU."""
    sent = task_rsu >= 0
    total = np.bincount(task_step[sent], minlength=num_steps).astype(np.int64)
    to_deg = np.zeros(task_rsu.shape, dtype=bool)
    to_deg[sent] = degraded[task_rsu[sent]]
    deg = np.bincount(task_step[to_deg], minlength=num_steps).astype(np.int64)
    return total, deg


def avoidance(total: np.ndarray, deg: np.ndarray, onset: int) -> dict:
    """D56 section 2: n_av, t_avoid, s0, n_win.

    t_avoid = first t >= onset with r(t') <= AVOID_RATIO * s0 for every t' in [t, t+AVOID_HOLD], where r is
    the share of dispatches to degraded RSUs over the AVOID_WINDOW steps starting at t (0 if none) and s0 is
    the same share over PRE_WINDOW. No such t: t_avoid = num_steps (never avoided).
    """
    n = total.size
    c_tot = np.concatenate([[0], np.cumsum(total)])
    c_deg = np.concatenate([[0], np.cumsum(deg)])
    lo, hi = PRE_WINDOW
    pre_tot = c_tot[hi] - c_tot[lo]
    s0 = float((c_deg[hi] - c_deg[lo]) / pre_tot) if pre_tot > 0 else 0.0

    last_t = n - AVOID_WINDOW  # last window start with a full window
    ts = np.arange(onset, last_t + 1)
    w_tot = c_tot[ts + AVOID_WINDOW] - c_tot[ts]
    w_deg = c_deg[ts + AVOID_WINDOW] - c_deg[ts]
    r = np.where(w_tot > 0, w_deg / np.maximum(w_tot, 1), 0.0)
    bad = np.concatenate([[0], np.cumsum(r > AVOID_RATIO * s0 + 1e-12)])  # prefix count of failing t'
    t_avoid = n
    for i in range(ts.size - AVOID_HOLD):
        if bad[i + AVOID_HOLD + 1] - bad[i] == 0:  # t' in [t, t + hold] all pass
            t_avoid = int(ts[i])
            break
    never = t_avoid == n
    return {
        "s0": s0,
        "t_avoid": t_avoid,
        "never_avoided": bool(never),
        "n_av": int(c_deg[t_avoid] - c_deg[onset]),
        "n_win": int(c_deg[n] - c_deg[onset]),
    }


# ------------------------------------------------------------------ H1: detection latency

def detection(
    trust: np.ndarray, active: np.ndarray, degraded: np.ndarray, onset: int
) -> dict:
    """D56 section 2 (D43's rule on the closed-loop trust): latency of the degraded RSUs.

    tau = the 1st percentile of trust over active RSU-steps in PRE_WINDOW; an RSU is detected at the first
    t >= onset from which its trust stays strictly below tau for DETECT_HOLD consecutive steps. Latency =
    t - onset; an RSU never detected is censored at 399 (D43's 799 minus the onset).
    """
    steps = trust.shape[0]
    lo, hi = PRE_WINDOW
    pool = trust[lo:hi][active[lo:hi]]
    tau = float(np.percentile(pool, TRUST_PERCENTILE)) if pool.size else 0.0
    horizon = 800  # D43: detection is sought in [onset, onset + 400); censor at onset + 399
    censor = horizon - onset - 1
    lat, detected = [], []
    for v in np.flatnonzero(degraded):
        below = trust[:, v] < tau
        run = np.convolve(below.astype(np.int64), np.ones(DETECT_HOLD, dtype=np.int64), mode="valid")
        # run[t] == DETECT_HOLD  <=>  steps t..t+hold-1 all below tau
        hit = np.flatnonzero(run[onset:horizon - DETECT_HOLD + 1] == DETECT_HOLD)
        if hit.size:
            lat.append(float(hit[0])); detected.append(1.0)
        else:
            lat.append(float(censor)); detected.append(0.0)
    if not lat:
        return {"tau": tau, "latency": float("nan"), "detected_fraction": float("nan")}
    return {"tau": tau, "latency": float(np.mean(lat)), "detected_fraction": float(np.mean(detected))}


# ------------------------------------------------------------------ statistics

def seed_means(values: np.ndarray) -> np.ndarray:
    """(seeds, draws) -> (seeds,): the mean over draws, ignoring NaN (D43: the seed is the unit)."""
    return np.nanmean(values, axis=1)


def t_interval(per_seed: np.ndarray) -> tuple[float, float, float, float]:
    """mean, sd (ddof=1), lo, hi: Student-t 95% over the per-seed values (df = n - 1, n = 5 -> 2.776)."""
    v = np.asarray(per_seed, float)
    if v.size != 5:
        raise ValueError(f"D56 fixes five seeds (t = {T_CRIT_DF4}); got {v.size}")
    m, sd = float(v.mean()), float(v.std(ddof=1))
    h = T_CRIT_DF4 * sd / np.sqrt(v.size)
    return m, sd, m - h, m + h


def label(mean: float, lo: float, hi: float, floor: float, bound: float) -> str:
    """D56 section 4. Check order: NEGATIVE, WIN, NULL, NOISY."""
    if hi < 0 and mean <= -floor:
        return "NEGATIVE"
    if lo > 0 and mean >= floor:
        return "WIN"
    if hi < bound:
        return "NULL"
    return "NOISY"


@dataclass(frozen=True)
class Effect:
    mean: float
    sd: float
    lo: float
    hi: float
    label: str


def h3_effect(succ_x: np.ndarray, succ_ref: np.ndarray) -> Effect:
    """(seeds, draws) success rates -> absolute difference X - ref, seed means, D56 labels."""
    m, sd, lo, hi = t_interval(seed_means(succ_x - succ_ref))
    return Effect(m, sd, lo, hi, label(m, lo, hi, *FLOORS["H3"]))


def relative_reduction(count_x: np.ndarray, count_ref: np.ndarray) -> np.ndarray:
    """Per-seed 1 - mean_draws(x) / mean_draws(ref). A reference with a zero seed mean is an error."""
    mx, mr = np.nanmean(count_x, axis=1), np.nanmean(count_ref, axis=1)
    if (mr <= 0).any():
        raise ValueError("reference count is zero for a seed; the relative reduction is undefined")
    return 1.0 - mx / mr


def h1_effect(n_av_x, n_av_ref, n_win_x, n_win_ref) -> tuple[Effect, Effect]:
    """(primary on n_av, guard on n_win). The primary's WIN is downgraded to NOISY unless the guard's lo > 0."""
    m, sd, lo, hi = t_interval(relative_reduction(n_av_x, n_av_ref))
    g = t_interval(relative_reduction(n_win_x, n_win_ref))
    lab = label(m, lo, hi, *FLOORS["H1"])
    if lab == "WIN" and not g[2] > 0:
        lab = "NOISY"
    guard = Effect(*g, label(g[0], g[2], g[3], *FLOORS["H1"]))
    return Effect(m, sd, lo, hi, lab), guard

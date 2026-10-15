"""S16: confirmatory test of CLAIMS A30, C-h (horizon objective) versus C (next-task objective) (DECISIONS.md D76, frozen).

    python scripts/s16_confirm.py gen    [--seeds ...]              # staggered scenarios for draws 350-399 (observable + SEALED)
    python scripts/s16_confirm.py check                             # harness check on an S15 draw that is not a result
    python scripts/s16_confirm.py trust  [--seeds ...]              # frozen C-h and C: open-loop trust, stride 1, steps 100-1199
    python scripts/s16_confirm.py eval   [--seeds ...] [--verdict]  # CO-PRIMARY 1 (latency, all degraded) and 2 (false alarms)
    python scripts/s16_confirm.py closed [--seeds ...]              # SECONDARY: closed-loop A, C, C-h at rho 1.0
    python scripts/s16_confirm.py report                            # SECONDARY tables

No training and no tuning: C = checkpoints/s13 (S13), C-h = checkpoints/s15 (S15); weights from weights_staggered.yaml (A, C)
and weights_horizon.yaml (C-h). Scenario generation and the closed-loop runner are the S15 ones, unchanged.
EVALUATION-SIDE: reads the sealed ground truth (onsets, classes) to score; allowlisted in tests/test_sealing.py.
The first result each command writes is dated in results/s16/first_number.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import s13_stag as S13  # noqa: E402  (detect_latency, t_interval: the D65/D66 rule, reused unchanged)
import s15_horizon as S15  # noqa: E402  (scenario generation, model loading, closed-loop runner, reused unchanged)

OUT = REPO_ROOT / "results" / "s16"
CONFIG = S15.CONFIG  # configs/staggered.yaml
SEEDS = (1, 2, 3, 4, 5)
FRACTION = 0.20
EVAL_DRAWS = tuple(range(350, 400))
RHOS = (0.0, 1.0)
HOLD = S13.HOLD  # 10
FIRST, LAST = 100, 1200  # stored steps [100, 1200)
PRE_END = 400  # tau window [100, 400); every onset is >= 400
LAT_FLOOR = 20.0  # D76 section 4
FA_BOUND = 0.05  # D76 section 5
# stored name -> S15 model name (S15.MODELS: C_h = checkpoints/s15 C, C_old = checkpoints/s13 C)
STORED = {"C_h": "C_h", "C": "C_old"}
NAMES = ("C_h", "C")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def note_first_number(what: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "first_number.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    if what not in cur:
        cur[what] = utc_now()
        p.write_text(json.dumps(cur, indent=1))


# ------------------------------------------------------------------ the D76 rule (pure functions; tested in tests/test_s16.py)

def pre_onset_tau(trust_per_draw) -> float:
    """D76 section 3: 1st percentile of trust over all RSU-steps at steps 100-399 (stored rows 0-299), pooled over the draws."""
    return float(np.percentile(np.concatenate([t[0:PRE_END - FIRST].ravel() for t in trust_per_draw]), 1.0))


def run_starts(trust_col: np.ndarray, tau: float) -> np.ndarray:
    """Steps t at which a crossing starts: trust strictly below tau at t..t+HOLD-1, the run inside the stored range."""
    below = (trust_col < tau).astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(below)])
    ok = (cs[HOLD:] - cs[:-HOLD]) == HOLD
    return np.flatnonzero(ok) + FIRST


def false_alarm(trust_col: np.ndarray, tau: float) -> tuple[bool, bool, bool]:
    """D76 section 5 for one healthy RSU: (any crossing in [100, 1200); one starting in [100, 400); only crossings inside [400, 1200))."""
    s = run_starts(trust_col, tau)
    anyc = s.size > 0
    pre = bool(anyc and s[0] < PRE_END)
    return anyc, pre, anyc and not pre


def label_latency(mean: float, lo: float, hi: float) -> str:
    """D76 section 4. Interval on C-h - C in steps; negative = C-h earlier. Order EARLIER, LATER, NULL, NOISY."""
    if hi < 0 and mean <= -LAT_FLOOR:
        return "EARLIER"
    if lo > 0 and mean >= LAT_FLOOR:
        return "LATER"
    if lo > -LAT_FLOOR and hi < LAT_FLOOR:
        return "NULL"
    return "NOISY"


def label_false_alarm(hi: float) -> str:
    """D76 section 5. Interval on C-h - C false-alarm rate; positive = C-h raises more."""
    return "ACCEPTABLE" if hi <= FA_BOUND else "NOT ACCEPTABLE"


def overall(l1: str, l2: str) -> str:
    """D76 section 6."""
    return "OBJECTIVE HELPS" if (l1 == "EARLIER" and l2 == "ACCEPTABLE") else "NOT CONFIRMED"


# ------------------------------------------------------------------ scenarios and open-loop trust

def cmd_gen(seeds, workers: int) -> int:
    jobs = [(s, rho, d) for s in seeds for rho in RHOS for d in EVAL_DRAWS]
    print(f"[gen] {len(jobs)} scenarios (draws {EVAL_DRAWS[0]}-{EVAL_DRAWS[-1]})", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=S15._single_thread) as pool:
        for job, st in pool.imap_unordered(S15._gen_one, jobs, chunksize=2):
            n += 1
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0


def trust_path(seed, rho, draw) -> Path:
    return OUT / "trust" / f"seed{seed}-rho{rho:g}-draw{draw}.npz"


def compute_trust(seed, rho, draw) -> dict:
    from trustgraph.config import load_config, with_overrides
    from trustgraph.dataset import graphset_from
    from trustgraph.observed import default_observed_path, load_observed
    from trustgraph.scenario import build_world
    from trustgraph.trace import default_trace_path, load_trace
    from trustgraph.train import predict_trust

    W = S15._W
    base = W.get("base") or W.setdefault("base", load_config(CONFIG))
    key = ("s16", seed)
    if key not in W:
        W[key] = (build_world(with_overrides(base, seed=seed)), load_trace(default_trace_path(CONFIG, seed)),
                  S15.load_models(base, seed, names=tuple(STORED.values())))
    world, trace, models = W[key]
    cfg = with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=draw)
    gs = graphset_from(cfg, world, trace, load_observed(default_observed_path(CONFIG, cfg)), stride=1, step_range=(FIRST, LAST))
    assert gs.steps[0] == FIRST and gs.steps[-1] == LAST - 1
    return {n: predict_trust(models[STORED[n]], gs).astype(np.float32) for n in NAMES}


def _trust_one(job):
    seed, rho, draw = job
    out = trust_path(seed, rho, draw)
    if out.exists():
        return job, "cached"
    tr = compute_trust(seed, rho, draw)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **tr)
    tmp.replace(out)
    return job, "ok"


def cmd_trust(seeds, workers: int) -> int:
    jobs = [(s, rho, d) for s in seeds for rho in RHOS for d in EVAL_DRAWS]
    print(f"[trust] {len(jobs)} scenarios", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=S15._single_thread) as pool:
        for job, st in pool.imap_unordered(_trust_one, jobs, chunksize=2):
            n += 1
            if n == 1:
                note_first_number("evaluation_first_trust")
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ harness check (an S15 draw; not a result)

def cmd_check() -> int:
    """Seed 1, rho 1.0, draw 300 (S15): the S16 open-loop path must reproduce the stored S15 C-h and C trust arrays exactly,
    and the closed-loop path the stored S15 C-h closed-loop result."""
    S15._single_thread()
    tr = compute_trust(1, 1.0, 300)
    with np.load(REPO_ROOT / "results/s15/trust/seed1-rho1-draw300.npz") as z:
        ok_h = bool(np.array_equal(tr["C_h"], z["C_h"]))
        ok_c = bool(np.array_equal(tr["C"], z["C_old"]))
    w = load_weights()
    _, r = S15._cl_one(("C_h", w["C_h"], 1, 1.0, FRACTION, 300))
    ref = (REPO_ROOT / "results/s15/closed/C_h-seed1-rho1-draw300.json").read_text()
    ok_cl = json.dumps(r) == json.dumps(json.loads(ref))
    print(f"  open loop: C-h trust equals stored S15 array: {ok_h}; C trust equals stored S15 C(S13) array: {ok_c}")
    print(f"  closed loop: C-h result equals stored S15 result: {ok_cl}")
    return 0 if (ok_h and ok_c and ok_cl) else 1


# ------------------------------------------------------------------ CO-PRIMARIES (D76 sections 3-6)

def evaluate(seeds):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.observed import default_observed_path, load_observed
    from trustgraph.sealed.ground_truth import load_ground_truth, sealed_path_for

    base = load_config(CONFIG)
    res = {}
    for seed in seeds:
        for rho in RHOS:
            tr, gts, inactive = {}, {}, []
            for d in EVAL_DRAWS:
                with np.load(trust_path(seed, rho, d)) as z:
                    tr[d] = {m: z[m] for m in NAMES}
                p = default_observed_path(CONFIG, with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=d))
                gts[d] = load_ground_truth(sealed_path_for(p))
                inactive.append(float(1.0 - load_observed(p).rsu_active[FIRST:PRE_END].mean()))
            tau = {m: pre_onset_tau([tr[d][m] for d in EVAL_DRAWS]) for m in NAMES}
            rec = {m: {k: [] for k in ("lat", "det", "fa", "fa_pre", "fa_post")} for m in NAMES}
            n_deg, n_healthy = [], []
            for d in EVAL_DRAWS:
                gt = gts[d]
                deg = np.flatnonzero(gt.degraded)
                healthy = np.flatnonzero(~gt.degraded)
                on = gt.onset_per_rsu()
                n_deg.append(len(deg))
                n_healthy.append(len(healthy))
                for m in NAMES:
                    t = tr[d][m]
                    if len(deg):
                        lats = [S13.detect_latency(t[:, v], tau[m], int(on[v])) for v in deg]
                        rec[m]["lat"].append(float(np.mean([S13.SEARCH - 1 if l is None else l for l in lats])))
                        rec[m]["det"].append(float(np.mean([l is not None for l in lats])))
                    else:
                        rec[m]["lat"].append(float("nan"))
                        rec[m]["det"].append(float("nan"))
                    fa = np.array([false_alarm(t[:, v], tau[m]) for v in healthy], float).reshape(-1, 3)
                    for i, k in enumerate(("fa", "fa_pre", "fa_post")):
                        rec[m][k].append(float(fa[:, i].mean()) if len(healthy) else float("nan"))
            res[(seed, rho)] = {"tau": tau, "n_deg": n_deg, "n_healthy": n_healthy, "inactive_pre": float(np.mean(inactive)),
                                **{m: {k: np.array(v) for k, v in rec[m].items()} for m in NAMES}}
    return res


def cmd_eval(seeds, verdict: bool) -> int:
    n = len(seeds)
    t_ = S13.T_CRIT[n]
    L = ["=" * 118, f"S16 CO-PRIMARIES: staggered onset (S = 300), open-loop frozen C-h (horizon objective, S15) and C (next-task, S13), "
         f"draws 350-399, fraction 0.20, seeds {list(seeds)}" + (" WITH D76 LABELS" if verdict else " (RAW: no labels)"), "=" * 118]
    res = evaluate(seeds)
    summary = {}
    for rho in (1.0, 0.0):
        L.append(f"\nrho {rho:g}  ({'TEST' if rho == 1.0 else 'CONTROL'})")
        L.append("  D42 accounting per seed: draws attempted / failed / aggregated (draws with a degraded RSU) / degraded RSUs / healthy RSUs")
        per = {k: [] for k in ("C_h_lat", "C_lat", "dlat", "C_h_det", "C_det", "C_h_fa", "C_fa", "dfa",
                               "C_h_fa_pre", "C_fa_pre", "C_h_fa_post", "C_fa_post")}
        for seed in seeds:
            r = res[(seed, rho)]
            ok = ~np.isnan(r["C"]["lat"])
            L.append(f"    seed {seed}: attempted {len(EVAL_DRAWS)}  failed 0  aggregated {int(ok.sum())}  degraded RSUs {int(sum(r['n_deg']))}  "
                     f"healthy RSUs {int(sum(r['n_healthy']))}  tau C-h {r['tau']['C_h']:.4f} C {r['tau']['C']:.4f}  "
                     f"inactive share of RSU-steps in [100, 400) {r['inactive_pre']:.4f}")
            for m in NAMES:
                per[f"{m}_lat"].append(r[m]["lat"][ok].mean())
                per[f"{m}_det"].append(r[m]["det"][ok].mean())
                for k in ("fa", "fa_pre", "fa_post"):
                    per[f"{m}_{k}"].append(r[m][k].mean())
            per["dlat"].append((r["C_h"]["lat"][ok] - r["C"]["lat"][ok]).mean())
            per["dfa"].append((r["C_h"]["fa"] - r["C"]["fa"]).mean())
        s = {k: S13.t_interval(v) for k, v in per.items()}
        l1 = label_latency(s["dlat"][0], s["dlat"][2], s["dlat"][3])
        l2 = label_false_alarm(s["dfa"][3])
        summary[rho] = (l1, l2, overall(l1, l2))
        L.append(f"  detection latency, ALL degraded RSUs (steps after own onset, censored 399): C-h {s['C_h_lat'][0]:.1f} +/- {s['C_h_lat'][1]:.1f}   "
                 f"C {s['C_lat'][0]:.1f} +/- {s['C_lat'][1]:.1f}")
        L.append(f"  detected fraction: C-h {s['C_h_det'][0]:.3f}  C {s['C_det'][0]:.3f}")
        L.append(f"  CO-PRIMARY 1  C-h - C latency: {s['dlat'][0]:+.1f} (sd {s['dlat'][1]:.1f}) 95% [{s['dlat'][2]:+.1f}, {s['dlat'][3]:+.1f}]"
                 + (f"   {l1}" if verdict else ""))
        L.append(f"     per seed: {[round(float(x), 1) for x in per['dlat']]}")
        L.append(f"  false-alarm rate, healthy RSUs, any crossing in [100, 1200): C-h {s['C_h_fa'][0]:.4f} +/- {s['C_h_fa'][1]:.4f}   "
                 f"C {s['C_fa'][0]:.4f} +/- {s['C_fa'][1]:.4f}")
        L.append(f"     (descriptive split) first crossing starting in [100, 400): C-h {s['C_h_fa_pre'][0]:.4f}  C {s['C_fa_pre'][0]:.4f};   "
                 f"only inside [400, 1200): C-h {s['C_h_fa_post'][0]:.4f}  C {s['C_fa_post'][0]:.4f}")
        L.append(f"  CO-PRIMARY 2  C-h - C false-alarm rate: {s['dfa'][0]:+.4f} (sd {s['dfa'][1]:.4f}) 95% [{s['dfa'][2]:+.4f}, {s['dfa'][3]:+.4f}]"
                 + (f"   {l2}" if verdict else ""))
        L.append(f"     per seed: {[round(float(x), 4) for x in per['dfa']]}")
    if verdict:
        L += ["\n" + "=" * 118, f"D76 VERDICT ({n} seeds, t = {t_})"]
        for rho in (1.0, 0.0):
            l1, l2, ov = summary[rho]
            L.append(f"  rho {rho:g} ({'test' if rho == 1.0 else 'control'}): co-primary 1 {l1}, co-primary 2 {l2} -> {ov}")
        L.append(f"  OVERALL (rho 1.0): {summary[1.0][2]}")
    text = "\n".join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"report-eval-{'verdict' if verdict else 'raw'}-{n}seeds.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


# ------------------------------------------------------------------ SECONDARY: closed loop (D76 section 7)

def load_weights() -> dict:
    st = yaml.safe_load((REPO_ROOT / "configs" / "weights_staggered.yaml").read_text())["weights"]
    hz = yaml.safe_load((REPO_ROOT / "configs" / "weights_horizon.yaml").read_text())["weights"]
    t = lambda d: (d["alpha"], d["beta"], d["gamma"])  # noqa: E731
    return {"A": t(st["A"]), "C": t(st["C"]), "C_h": t(hz["C_h"])}


# closed-loop name here -> S15 runner name (A, C_h, C_old)
CL_RUNNER = {"A": "A", "C": "C_old", "C_h": "C_h"}
CL_V = ("A", "C", "C_h")


def cl_path(name, seed, rho, draw) -> Path:
    return OUT / "closed" / f"{name}-seed{seed}-rho{rho:g}-draw{draw}.json"


def _cl_job(job):
    name, w, seed, rho, frac, draw = job
    _, r = S15._cl_one((CL_RUNNER[name], w, seed, rho, frac, draw))
    return job, r


def cmd_closed(seeds, workers: int) -> int:
    w = load_weights()
    todo = [(v, w[v], s, 1.0, FRACTION, d) for s in seeds for d in EVAL_DRAWS for v in CL_V if not cl_path(v, s, 1.0, d).exists()]
    print(f"[closed] {len(todo)} runs to do", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=S15._single_thread) as pool:
        for job, r in pool.imap_unordered(_cl_job, todo, chunksize=2):
            name, _, s, rho, _, d = job
            p = cl_path(name, s, rho, d)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(r))
            n += 1
            if n == 1:
                note_first_number("closed_loop_evaluation_first_result")
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    return 0


def cmd_report(seeds) -> int:
    w = load_weights()
    L = ["=" * 118, f"S16 SECONDARY: closed-loop post-onset success, staggered, rho 1.0, fraction 0.20, draws 350-399, seeds {list(seeds)} "
         "(RAW, descriptive: no labels)", "=" * 118]
    data = {v: {k: np.array([[json.loads(cl_path(v, s, 1.0, d).read_text())[k] for d in EVAL_DRAWS] for s in seeds], float)
                for k in ("success", "n_av", "n_win", "never_avoided")} for v in CL_V}
    L.append(f"weights (alpha, beta, gamma): A {w['A']}  C {w['C']}  C-h {w['C_h']}  (no new tuning)")
    L.append(f"D42: attempted {len(seeds) * len(EVAL_DRAWS)} per model  failed 0  aggregated {data['A']['success'].size}")
    for v in CL_V:
        sm = data[v]["success"].mean(axis=1)
        L.append(f"  {v:4s} success {sm.mean():.4f} +/- {sm.std(ddof=1):.4f}")
    for a, b in (("C_h", "C"), ("C", "A"), ("C_h", "A")):
        m, sd, lo, hi = S13.t_interval((data[a]["success"] - data[b]["success"]).mean(axis=1))
        L.append(f"  {a} - {b}: {m:+.4f} (sd {sd:.4f}) 95% [{lo:+.4f}, {hi:+.4f}]")
    L.append("  n_av / n_win / never-avoided (mean +/- sd over seeds of per-seed means), always together (D60 Q4):")
    for v in CL_V:
        f = lambda k: (data[v][k].mean(axis=1).mean(), data[v][k].mean(axis=1).std(ddof=1))  # noqa: E731
        L.append(f"    {v:4s} n_av {f('n_av')[0]:7.1f} +/- {f('n_av')[1]:6.1f}   n_win {f('n_win')[0]:7.1f} +/- {f('n_win')[1]:6.1f}   "
                 f"never-avoided {f('never_avoided')[0]:.3f} +/- {f('never_avoided')[1]:.3f}")
    for k in ("n_av", "n_win"):
        red = (data["C"][k].mean(axis=1) - data["C_h"][k].mean(axis=1)) / data["A"][k].mean(axis=1)
        m, sd, lo, hi = S13.t_interval(red)
        L.append(f"  D57-normalised (n_C - n_C-h)/n_A on {k}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (positive = C-h sends fewer to degraded RSUs; descriptive)")
    text = "\n".join(L)
    (OUT / "report-closed-raw.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("gen", "check", "trust", "eval", "closed", "report"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    return {"gen": lambda: cmd_gen(a.seeds, a.workers), "check": cmd_check, "trust": lambda: cmd_trust(a.seeds, a.workers),
            "eval": lambda: cmd_eval(a.seeds, a.verdict), "closed": lambda: cmd_closed(a.seeds, a.workers),
            "report": lambda: cmd_report(a.seeds)}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())

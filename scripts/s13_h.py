"""S13 Experiment 1: the heuristic trust baseline H against the learned head C (DECISIONS.md D65, frozen).

    python scripts/s13_h.py tune [--write]     # D52 grid on seed 100, draws 0-2; --write stores configs/weights_h.yaml
    python scripts/s13_h.py run                # H (+ a recompute of C) on the 4 fraction-sweep points, resumable
    python scripts/s13_h.py report [--verdict] # raw tables, or tables with the D65 labels and verdict

H's trust is the RSU's own success_ewma (no learning). H is paired per draw with the stored S9 C and A results
(results/s9/raw). Variant C is recomputed in every task and must equal the stored values bit-for-bit (wall time
excepted): the check that nothing in the existing variants changed.

EVALUATION-SIDE (the runner scores dispatches against the sealed degraded set); allowlisted in tests/test_sealing.py.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import s9_sweep as S  # noqa: E402
from trustgraph import sweep_metrics as M  # noqa: E402

H_CFG = REPO_ROOT / "configs" / "h.yaml"
_G: dict = {}

# D65 section 1.4
H3_FLOOR, H3_BOUND = 0.01, 0.025


def load_h() -> dict:
    g = yaml.safe_load(H_CFG.read_text(encoding="utf-8"))
    g["_sweep"] = S.load_sweep(g["sweep_config"])
    return g


def out_dir(g) -> Path:
    return REPO_ROOT / g["out_dir"]


def eval_points(g) -> list[tuple[float, float]]:
    return [(float(p["rho"]), float(p["fraction"])) for p in g["points"]]


def raw_path(g, rho, frac, seed, draw) -> Path:
    return out_dir(g) / "raw" / f"rho{rho:.2f}-frac{frac:.2f}" / f"seed{seed}-draw{draw}.json"


def weights_path(g) -> Path:
    return REPO_ROOT / g["weights_file"]


def h_hash(g) -> str:
    core = {k: v for k, v in g.items() if not k.startswith("_")}
    h = hashlib.sha256(json.dumps(core, sort_keys=True).encode())
    h.update(S.config_hash(g["_sweep"]).encode())
    h.update(weights_path(g).read_bytes())
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def note_first_number(g, what: str) -> None:
    """Dated record of the first number each stage produced (D65 'first-number timestamps')."""
    p = out_dir(g) / "first_number.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    if what not in cur:
        cur[what] = utc_now()
        S.write_atomic(p, json.dumps(cur, indent=1))


def grid(g):
    t = g["tuning"]
    return list(itertools.product(t["alphas"], t["betas"], t["gammas"]))


# ------------------------------------------------------------------ workers

def _init(g):
    import torch

    torch.set_num_threads(1)
    S._init(g["_sweep"])
    _G["g"] = g


def _sw_with_weights(sw, wfile):
    d = dict(sw)
    d["weights_file"] = wfile
    return d


def _tune_work(task):
    (alpha, beta, gamma), draw = task
    g = _G["g"]
    t = g["tuning"]
    from trustgraph.config import with_overrides
    from trustgraph.offload import OffloadSettings
    from trustgraph.simulator import run_closed_loop

    st = S._seed_state(int(t["seed"]))
    cfg = with_overrides(st["base"], seed=int(t["seed"]), rho=float(t["rho"]), fraction=float(t["fraction"]), draw=draw)
    settings = OffloadSettings(variant="H", alpha=alpha, beta=beta, gamma=gamma)
    sc = run_closed_loop(cfg, st["world"], st["trace"], settings, model=None)
    tasks = sc.observed.tasks
    return task, {"success": tasks.success_rate(tasks.step >= 400), "success_all": tasks.success_rate()}


def _eval_work(task):
    rho, frac, seed, draw = task
    g, sw = _G["g"], _G["g"]["_sweep"]
    try:
        st = S._seed_state(seed)
        hm, _, _ = S.run_one_variant(_sw_with_weights(sw, g["weights_file"]), st["base"], st["world"], st["trace"], st["models"],
                                     "H", seed, rho, frac, draw)
        cm, _, _ = S.run_one_variant(sw, st["base"], st["world"], st["trace"], st["models"], "C", seed, rho, frac, draw)
        stored = json.loads(S.task_path(sw, rho, frac, seed, draw).read_text(encoding="utf-8"))["variants"]["C"]
        res = {"status": "ok", "rho": rho, "fraction": frac, "seed": seed, "draw": draw, "h": hm,
               "c_recompute_equals_s9": bool(S._same(stored, cm)), "finished_utc": utc_now()}
    except Exception:
        res = {"status": "failed", "rho": rho, "fraction": frac, "seed": seed, "draw": draw, "error": traceback.format_exc()}
    return task, res


# ------------------------------------------------------------------ tune

def cmd_tune(write: bool) -> int:
    g = load_h()
    t = g["tuning"]
    tasks = [(w, d) for w in grid(g) for d in t["draws"]]
    d = out_dir(g)
    d.mkdir(parents=True, exist_ok=True)
    print(f"[tune] {len(tasks)} runs on tuning seed {t['seed']} (rho {t['rho']}, fraction {t['fraction']}, draws {t['draws']})", flush=True)
    res = {}
    with mp.get_context("spawn").Pool(int(g["_sweep"]["workers"]), initializer=_init, initargs=(g,)) as pool:
        for task, r in pool.imap_unordered(_tune_work, tasks, chunksize=3):
            res[task] = r
            if len(res) == 1:
                note_first_number(g, "exp1_tuning_first_result")
    rows = []
    for w in grid(g):
        per = [res[(w, dr)]["success"] for dr in t["draws"]]
        rows.append({"alpha": w[0], "beta": w[1], "gamma": w[2], "objective": float(np.mean(per)), "per_draw": per,
                     "success_all": float(np.mean([res[(w, dr)]["success_all"] for dr in t["draws"]]))})
    best = max(range(len(rows)), key=lambda i: (rows[i]["objective"], -i))  # first maximum in grid order
    print("H: objective = mean post-onset deadline success over draws", t["draws"])
    print("  alpha  beta  gamma   objective   per-draw")
    for i, r in enumerate(rows):
        mark = "  <== chosen" if i == best else ""
        print(f"  {r['alpha']:5.2f} {r['beta']:5.2f} {r['gamma']:5.2f}   {r['objective']:.4f}    {[round(x, 4) for x in r['per_draw']]}{mark}")
    b = rows[best]
    print(f"chosen (alpha, beta, gamma) = ({b['alpha']}, {b['beta']}, {b['gamma']}), objective {b['objective']:.4f}, per-draw {[round(x, 4) for x in b['per_draw']]}")
    (d / "tuning.json").write_text(json.dumps({"rows": rows, "chosen": b}, indent=1), encoding="utf-8")
    if write:
        doc = {"comment": "Variant H (D65). Tuned on seed 100, rho 1.0, fraction 0.2, draws 0-2 by the D52 procedure.",
               "tuning_seed": int(t["seed"]), "weights": {"H": {k: b[k] for k in ("alpha", "beta", "gamma")}}}
        weights_path(g).write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(f"wrote {weights_path(g)}")
    return 0


# ------------------------------------------------------------------ run

def cmd_run() -> int:
    g = load_h()
    if not weights_path(g).exists():
        raise SystemExit(f"{weights_path(g)} missing: run `tune --write` and record the grid in FINDINGS.md first")
    sw = g["_sweep"]
    d = out_dir(g)
    d.mkdir(parents=True, exist_ok=True)
    man = d / "manifest.json"
    h = h_hash(g)
    if man.exists() and json.loads(man.read_text())["config_hash"] != h:
        raise SystemExit("manifest from a different config; refusing to resume")
    S.write_atomic(man, json.dumps({"config_hash": h, "points": [list(p) for p in eval_points(g)]}))
    tasks = [(r, f, s, dr) for (r, f) in eval_points(g) for s in sw["seeds"] for dr in sw["draws"]]
    todo = [t for t in tasks if not raw_path(g, *t).exists()]
    print(f"[run] {len(tasks)} tasks, {len(tasks) - len(todo)} done, {len(todo)} to run", flush=True)
    t0, n, bad = time.time(), 0, 0
    with mp.get_context("spawn").Pool(int(sw["workers"]), initializer=_init, initargs=(g,)) as pool:
        for task, res in pool.imap_unordered(_eval_work, todo, chunksize=1):
            n += 1
            if res["status"] == "ok":
                S.write_atomic(raw_path(g, *task), json.dumps(res))
                note_first_number(g, "exp1_evaluation_first_result")
            else:
                bad += 1
                S.write_atomic(out_dir(g) / "failed" / f"{task[0]}-{task[1]}-{task[2]}-{task[3]}.json", json.dumps(res))
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s, {bad} failed", flush=True)
    left = [t for t in tasks if not raw_path(g, *t).exists()]
    print(f"[run] finished; {len(left)} tasks without a result")
    return 0 if not left else 2


# ------------------------------------------------------------------ report

KEYS = ("succ", "sall", "nav", "nwin", "never", "lat", "det")


def load_arrays(g):
    sw = g["_sweep"]
    seeds, draws = sw["seeds"], sw["draws"]
    out = {}
    for (rho, frac) in eval_points(g):
        shape = (len(seeds), len(draws))
        a = {f"{v}_{k}": np.full(shape, np.nan) for v in "ach" for k in KEYS}  # a = A (stored), c = C (stored), h = H
        ok = np.zeros(shape, bool)
        c_mismatch = 0
        failed = 0
        for i, s in enumerate(seeds):
            for j, dr in enumerate(draws):
                p = raw_path(g, rho, frac, s, dr)
                if not p.exists():
                    failed += 1
                    continue
                r = json.loads(p.read_text(encoding="utf-8"))
                st = json.loads(S.task_path(sw, rho, frac, s, dr).read_text(encoding="utf-8"))["variants"]
                ok[i, j] = True
                c_mismatch += int(not r["c_recompute_equals_s9"])
                for v, m in (("a", st["A"]), ("c", st["C"]), ("h", r["h"])):
                    a[f"{v}_succ"][i, j] = m["success_post_onset"]
                    a[f"{v}_sall"][i, j] = m["success_all"]
                    a[f"{v}_nav"][i, j] = m["n_av"]
                    a[f"{v}_nwin"][i, j] = m["n_win"]
                    a[f"{v}_never"][i, j] = float(m["never_avoided"])
                    a[f"{v}_lat"][i, j] = m["latency"]
                    a[f"{v}_det"][i, j] = m["detected_fraction"]
        out[(rho, frac)] = {"a": a, "ok": ok, "c_mismatch": c_mismatch, "failed": failed}
    return out


def label_d65(mean: float, lo: float, hi: float) -> str:
    """D65 section 1.4. Check order NEGATIVE, WIN, NULL, NOISY; NULL needs the whole interval inside +-0.025."""
    if hi < 0 and mean <= -H3_FLOOR:
        return "NEGATIVE"
    if lo > 0 and mean >= H3_FLOOR:
        return "WIN"
    if lo > -H3_BOUND and hi < H3_BOUND:
        return "NULL"
    return "NOISY"


def overall(labels: list[str]) -> str:
    wins, negs = labels.count("WIN"), labels.count("NEGATIVE")
    if wins >= 2 and negs >= 2:
        return "MIXED"
    if wins >= 2 and negs < 2:
        return "LEARNING HELPS"
    if negs >= 2 and wins < 2:
        return "HEURISTIC BETTER"
    return "NOT SEPARATED"


def ms(arr, nd=4):
    sm = M.seed_means(arr)
    return f"{np.mean(sm):.{nd}f} +/- {np.std(sm, ddof=1):.{nd}f}"


def diff(x, y):
    """Per-seed mean of the paired per-draw difference x - y, and its t interval (mean, sd, lo, hi)."""
    return M.t_interval(M.seed_means(x - y))


def cmd_report(verdict: bool) -> int:
    g = load_h()
    sw = g["_sweep"]
    data = load_arrays(g)
    w = yaml.safe_load(weights_path(g).read_text())["weights"]["H"]
    L = ["=" * 118,
         "S13 EXPERIMENT 1: HEURISTIC TRUST BASELINE H vs LEARNED HEAD C" + (" WITH D65 LABELS AND VERDICT" if verdict else " (RAW: no labels, no verdict)"),
         f"H weights (tuned on seed 100): alpha {w['alpha']}, beta {w['beta']}, gamma {w['gamma']}; C weights: D52 (1, 0.25, 0.6). "
         f"seeds {sw['seeds']}, draws {sw['draws'][0]}-{sw['draws'][-1]}",
         "mean +/- sd across the 5 seed means; closed loop; H paired per draw with the stored S9 C and A", "=" * 118,
         "\nD42-style accounting (draws attempted / failed / aggregated) and the C-recompute check"]
    for pt, d in data.items():
        att = len(sw["seeds"]) * len(sw["draws"])
        L.append(f"  rho {pt[0]:.2f} frac {pt[1]:.2f}: attempted {att}  failed {d['failed']}  aggregated {int(d['ok'].sum())}   "
                 f"Variant C recompute differs from stored S9 on {d['c_mismatch']} draws")
    labs = {}
    for pt, d in data.items():
        a = d["a"]
        L.append("\n" + "-" * 118)
        L.append(f"rho {pt[0]:.2f}, fraction {pt[1]:.2f}" + ("   [shared operating point]" if abs(pt[1] - 0.2) < 1e-9 else "")
                 + ("   [K = 1: no correlated failure]" if abs(pt[1] - 0.05) < 1e-9 else ""))
        L.append(f"  {'metric':46s}{'A':>22s}{'C (learned)':>22s}{'H (heuristic)':>22s}")
        rows = (("succ", "H3 success post-onset", 4), ("nav", "n_av: dispatches to degraded, onset->avoidance", 1),
                ("nwin", "n_win: dispatches to degraded, [400,1200)", 1), ("never", "never-avoided rate (D56 rule)", 3),
                ("lat", "detection latency (steps, censored 399)", 1), ("det", "detected fraction of degraded RSUs", 3))
        for key, lab, nd in rows:
            cells = [ms(a[f"{v}_{key}"], nd) if not (key in ("lat", "det") and v == "a") else "n/a" for v in "ach"]
            L.append(f"  {lab:46s}{cells[0]:>22s}{cells[1]:>22s}{cells[2]:>22s}")
        m, sd, lo, hi = diff(a["c_succ"], a["h_succ"])
        lab = label_d65(m, lo, hi)
        labs[pt] = lab
        L.append(f"  PRIMARY  C - H  success: {m:+.4f} (sd {sd:.4f}) 95% [{lo:+.4f}, {hi:+.4f}]" + (f"   {lab}" if verdict else ""))
        mh, _, loh, hih = diff(a["h_succ"], a["a_succ"])
        mc, _, loc, hic = diff(a["c_succ"], a["a_succ"])
        L.append(f"  (descr.) H - A success: {mh:+.4f} [{loh:+.4f}, {hih:+.4f}]   C - A success: {mc:+.4f} [{loc:+.4f}, {hic:+.4f}]")
        ml, _, lol, hil = diff(a["c_lat"], a["h_lat"])
        L.append(f"  (descr.) detection latency C - H: {ml:+.1f} steps [{lol:+.1f}, {hil:+.1f}]   (negative = C detects earlier)")
        nw = M.t_interval(M.relative_reduction(a["c_nwin"], a["h_nwin"], a["a_nwin"]))
        na = M.t_interval(M.relative_reduction(a["c_nav"], a["h_nav"], a["a_nav"]))
        L.append(f"  (descr.) (n_H - n_C)/n_A, positive = C sends fewer: n_win {nw[0]:+.3f} [{nw[2]:+.3f}, {nw[3]:+.3f}]   "
                 f"n_av {na[0]:+.3f} [{na[2]:+.3f}, {na[3]:+.3f}]")
    if verdict:
        order = list(data)
        v = overall([labs[pt] for pt in order])
        L += ["\n" + "=" * 118, "D65 VERDICT (Experiment 1)",
              "  labels (WIN = learned C beats heuristic H; NEGATIVE = H beats C) by fraction: "
              + ", ".join(f"{pt[1]:.2f}: {labs[pt]}" for pt in order),
              f"  WIN at {sum(labs[p] == 'WIN' for p in order)} of 4, NEGATIVE at {sum(labs[p] == 'NEGATIVE' for p in order)} of 4  ->  {v}",
              f"  label at the shared point (fraction 0.20): {labs[(1.0, 0.2)]}"]
    text = "\n".join(L)
    (out_dir(g) / ("report-verdict.txt" if verdict else "report-raw.txt")).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("tune", "run", "report"))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    if a.cmd == "tune":
        return cmd_tune(a.write)
    if a.cmd == "run":
        return cmd_run()
    return cmd_report(a.verdict)


if __name__ == "__main__":
    raise SystemExit(main())

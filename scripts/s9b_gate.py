"""S9b: the Gate-C baseline, tuned on the tuning seed and evaluated on the S9 grid (DECISIONS.md D59, frozen).

    python scripts/s9b_gate.py tune [--write]     # tau grid on seed 100, draws 0-2; --write stores the chosen tau in configs/gate.yaml
    python scripts/s9b_gate.py run                # Gate-C (+ a recompute of C) on the 4 fraction-sweep points, resumable
    python scripts/s9b_gate.py report [--verdict] # raw tables, or tables with the D59 labels and verdict
    python scripts/s9b_gate.py all                # tune --write is NOT implied: tune first, record F28, then run and report

Gate-C is paired per draw with the stored S9 Variant C result (results/s9/raw). Variant C is also recomputed in every task
and must equal the stored values bit-for-bit (wall time excepted): the check that the existing variants did not change.

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
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import s9_sweep as S  # noqa: E402
from trustgraph import sweep_metrics as M  # noqa: E402

GATE_CFG = REPO_ROOT / "configs" / "gate.yaml"
_G: dict = {}


def load_gate() -> dict:
    g = yaml.safe_load(GATE_CFG.read_text(encoding="utf-8"))
    g["_sweep"] = S.load_sweep(g["sweep_config"])
    return g


def out_dir(g) -> Path:
    return REPO_ROOT / g["out_dir"]


def eval_points(g) -> list[tuple[float, float]]:
    return [(float(p["rho"]), float(p["fraction"])) for p in g["points"]]


def raw_path(g, rho, frac, seed, draw) -> Path:
    return out_dir(g) / "raw" / f"rho{rho:.2f}-frac{frac:.2f}" / f"seed{seed}-draw{draw}.json"


def gate_hash(g) -> str:
    core = {k: v for k, v in g.items() if not k.startswith("_")}
    h = hashlib.sha256(json.dumps(core, sort_keys=True).encode())
    h.update(S.config_hash(g["_sweep"]).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ workers

def _init(g):
    import torch

    torch.set_num_threads(1)
    S._init(g["_sweep"])
    _G["g"] = g


def _tune_work(task):
    tau, draw = task
    g, sw = _G["g"], _G["g"]["_sweep"]
    t = g["tuning"]
    st = S._seed_state(int(t["seed"]))
    m, _, _ = S.run_one_variant(sw, st["base"], st["world"], st["trace"], st["models"], "Gate_C", int(t["seed"]),
                                float(t["rho"]), float(t["fraction"]), draw, trust_gate_tau=tau)
    return task, {"success": m["success_post_onset"], "gate": m["gate"]}


def _eval_work(task):
    rho, frac, seed, draw = task
    g, sw = _G["g"], _G["g"]["_sweep"]
    try:
        st = S._seed_state(seed)
        gm, _, _ = S.run_one_variant(sw, st["base"], st["world"], st["trace"], st["models"], "Gate_C", seed, rho, frac, draw,
                                     trust_gate_tau=float(g["tau"]))
        cm, _, _ = S.run_one_variant(sw, st["base"], st["world"], st["trace"], st["models"], "C", seed, rho, frac, draw)
        stored = json.loads(S.task_path(sw, rho, frac, seed, draw).read_text(encoding="utf-8"))["variants"]["C"]
        res = {"status": "ok", "rho": rho, "fraction": frac, "seed": seed, "draw": draw, "tau": float(g["tau"]),
               "gate": gm, "c_recompute_equals_s9": bool(S._same(stored, cm))}
    except Exception:
        res = {"status": "failed", "rho": rho, "fraction": frac, "seed": seed, "draw": draw, "error": traceback.format_exc()}
    return task, res


# ------------------------------------------------------------------ tune

def cmd_tune(write: bool) -> int:
    g = load_gate()
    t = g["tuning"]
    tasks = [(tau, d) for tau in g["tau_grid"] for d in t["draws"]]
    print(f"[tune] {len(tasks)} runs on tuning seed {t['seed']} (rho {t['rho']}, fraction {t['fraction']}, draws {t['draws']})", flush=True)
    res = {}
    with mp.get_context("spawn").Pool(int(g["_sweep"]["workers"]), initializer=_init, initargs=(g,)) as pool:
        for task, r in pool.imap_unordered(_tune_work, tasks, chunksize=1):
            res[task] = r
    rows = []
    for tau in g["tau_grid"]:
        per = [res[(tau, d)]["success"] for d in t["draws"]]
        fb = sum(res[(tau, d)]["gate"]["fallback"] for d in t["draws"])
        ds = sum(res[(tau, d)]["gate"]["dispatched"] for d in t["draws"])
        ex = sum(res[(tau, d)]["gate"]["excluded"] for d in t["draws"])
        rows.append({"tau": tau, "objective": float(np.mean(per)), "per_draw": per, "fallback_rate": fb / ds, "excluded_rate": ex / ds})
    best = max(range(len(rows)), key=lambda i: (rows[i]["objective"], -i))  # first maximum in ascending tau
    print("tau    objective (mean post-onset success over draws 0,1,2)   per-draw                      fallback rate  gate-removed rate")
    for r in rows:
        print(f"{r['tau']:.1f}    {r['objective']:.4f}   {np.round(r['per_draw'], 4).tolist()}   {r['fallback_rate']:.4f}   {r['excluded_rate']:.4f}")
    print(f"chosen tau = {rows[best]['tau']} (objective {rows[best]['objective']:.4f})")
    d = out_dir(g); d.mkdir(parents=True, exist_ok=True)
    (d / "tuning.json").write_text(json.dumps({"rows": rows, "chosen": rows[best]}, indent=1), encoding="utf-8")
    if write:
        text = GATE_CFG.read_text(encoding="utf-8")
        text = text.replace("tau: null", f"tau: {rows[best]['tau']}")
        GATE_CFG.write_text(text, encoding="utf-8")
        print(f"wrote tau = {rows[best]['tau']} to {GATE_CFG}")
    return 0


# ------------------------------------------------------------------ run

def cmd_run() -> int:
    g = load_gate()
    if g["tau"] is None:
        raise SystemExit("configs/gate.yaml has tau: null; run `tune --write` and record F28 first")
    sw = g["_sweep"]
    d = out_dir(g); d.mkdir(parents=True, exist_ok=True)
    man = d / "manifest.json"
    h = gate_hash(g)
    if man.exists() and json.loads(man.read_text())["config_hash"] != h:
        raise SystemExit("manifest from a different config; refusing to resume")
    S.write_atomic(man, json.dumps({"config_hash": h, "tau": g["tau"], "points": [list(p) for p in eval_points(g)]}))
    tasks = [(r, f, s, dr) for (r, f) in eval_points(g) for s in sw["seeds"] for dr in sw["draws"]]
    todo = [t for t in tasks if not raw_path(g, *t).exists()]
    print(f"[run] {len(tasks)} tasks, {len(tasks) - len(todo)} done, {len(todo)} to run, tau {g['tau']}", flush=True)
    t0, n, bad = time.time(), 0, 0
    with mp.get_context("spawn").Pool(int(sw["workers"]), initializer=_init, initargs=(g,)) as pool:
        for task, res in pool.imap_unordered(_eval_work, todo, chunksize=1):
            n += 1
            if res["status"] == "ok":
                S.write_atomic(raw_path(g, *task), json.dumps(res))
            else:
                bad += 1
                S.write_atomic(out_dir(g) / "failed" / f"{task[0]}-{task[1]}-{task[2]}-{task[3]}.json", json.dumps(res))
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s, {bad} failed", flush=True)
    left = [t for t in tasks if not raw_path(g, *t).exists()]
    print(f"[run] finished; {len(left)} tasks without a result")
    return 0 if not left else 2


# ------------------------------------------------------------------ report

def load_arrays(g):
    sw = g["_sweep"]
    seeds, draws = sw["seeds"], sw["draws"]
    out = {}
    for (rho, frac) in eval_points(g):
        shape = (len(seeds), len(draws))
        a = {k: np.full(shape, np.nan) for k in ("g_succ", "c_succ", "a_succ", "g_nwin", "c_nwin", "a_nwin", "g_nav", "c_nav", "a_nav",
                                                 "fb_all", "fb_post", "excl")}
        ok = np.zeros(shape, bool); c_mismatch = 0; failed = 0
        for i, s in enumerate(seeds):
            for j, dr in enumerate(draws):
                p = raw_path(g, rho, frac, s, dr)
                if not p.exists():
                    failed += 1
                    continue
                r = json.loads(p.read_text(encoding="utf-8"))
                st = json.loads(S.task_path(sw, rho, frac, s, dr).read_text(encoding="utf-8"))["variants"]
                gm = r["gate"]
                ok[i, j] = True
                c_mismatch += int(not r["c_recompute_equals_s9"])
                a["g_succ"][i, j], a["c_succ"][i, j], a["a_succ"][i, j] = gm["success_post_onset"], st["C"]["success_post_onset"], st["A"]["success_post_onset"]
                a["g_nwin"][i, j], a["c_nwin"][i, j], a["a_nwin"][i, j] = gm["n_win"], st["C"]["n_win"], st["A"]["n_win"]
                a["g_nav"][i, j], a["c_nav"][i, j], a["a_nav"][i, j] = gm["n_av"], st["C"]["n_av"], st["A"]["n_av"]
                gi = gm["gate"]
                a["fb_all"][i, j] = gi["fallback"] / max(gi["dispatched"], 1)
                a["fb_post"][i, j] = gi["fallback_post_onset"] / max(gm["post_dispatched"], 1)
                a["excl"][i, j] = gi["excluded"] / max(gi["dispatched"], 1)
        out[(rho, frac)] = {"a": a, "ok": ok, "c_mismatch": c_mismatch, "failed": failed}
    return out


def ms(arr, nd=4):
    sm = M.seed_means(arr)
    return f"{np.mean(sm):.{nd}f} +/- {np.std(sm, ddof=1):.{nd}f}"


def cmd_report(verdict: bool) -> int:
    g = load_gate()
    sw = g["_sweep"]
    data = load_arrays(g)
    L = []
    L.append("=" * 112)
    L.append(f"S9b GATE-C" + (" WITH D59 LABELS AND VERDICT" if verdict else " (RAW: no labels, no verdict)")
             + f"   tau = {g['tau']}   seeds {sw['seeds']}  draws {sw['draws'][0]}-{sw['draws'][-1]}  weights: C's (D52)")
    L.append("mean +/- sd across the 5 seed means; closed loop; Gate-C paired per draw with the stored S9 Variant C; A from S9")
    L.append("=" * 112)
    L.append("\nD42-style accounting (draws attempted / failed / aggregated) and the C-recompute check")
    for pt, d in data.items():
        att = len(sw["seeds"]) * len(sw["draws"])
        L.append(f"  rho {pt[0]:.2f} frac {pt[1]:.2f}: attempted {att}  failed {d['failed']}  aggregated {int(d['ok'].sum())}   "
                 f"Variant C recompute differs from stored S9 on {d['c_mismatch']} draws")
    effects = {}
    for pt, d in data.items():
        a = d["a"]
        L.append("\n" + "-" * 112)
        L.append(f"rho {pt[0]:.2f}, fraction {pt[1]:.2f}" + ("   [shared operating point]" if abs(pt[1] - 0.2) < 1e-9 else "")
                 + ("   [K = 1: no correlated failure]" if abs(pt[1] - 0.05) < 1e-9 else ""))
        L.append(f"  {'metric':44s}{'A':>22s}{'C (weighted)':>22s}{'Gate-C':>22s}")
        for key, lab, nd in (("succ", "H3 success post-onset", 4), ("nav", "dispatches to degraded, onset->avoidance", 1), ("nwin", "n_win: dispatches to degraded, [400,1200)", 1)):
            L.append(f"  {lab:44s}{ms(a['a_' + key], nd):>22s}{ms(a['c_' + key], nd):>22s}{ms(a['g_' + key], nd):>22s}")
        L.append(f"  empty-candidate-set (fallback) rate, all steps : {ms(a['fb_all'], 5)}   steps >= 400 : {ms(a['fb_post'], 5)}")
        L.append(f"  decisions where the gate removed >= 1 candidate: {ms(a['excl'], 4)}  (includes the fallback decisions)")
        e = M.h3_effect(a["c_succ"], a["g_succ"])     # C - Gate-C
        eg = M.h3_effect(a["g_succ"], a["a_succ"])    # Gate-C - A
        nw = M.t_interval(M.relative_reduction(a["c_nwin"], a["g_nwin"], a["a_nwin"]))
        na = M.t_interval(M.relative_reduction(a["c_nav"], a["g_nav"], a["a_nav"]))
        lab = f"   {e.label}" if verdict else ""
        L.append(f"  PAIRED  C - Gate-C  success: {e.mean:+.4f} (sd {e.sd:.4f}) 95% [{e.lo:+.4f}, {e.hi:+.4f}]{lab}")
        L.append(f"          Gate-C - A  success: {eg.mean:+.4f} (sd {eg.sd:.4f}) 95% [{eg.lo:+.4f}, {eg.hi:+.4f}]  (descriptive)")
        L.append(f"          (Gate-C - C) n_win reduction in units of A: {nw[0]:+.3f} [{nw[2]:+.3f}, {nw[3]:+.3f}]   n_av: {na[0]:+.3f} [{na[2]:+.3f}, {na[3]:+.3f}]"
                 "   (positive = C sends fewer; descriptive)")
        effects[pt] = e
    if verdict:
        labs = {pt: effects[pt].label for pt in effects}
        wins = sum(l == "WIN" for l in labs.values()); negs = sum(l == "NEGATIVE" for l in labs.values())
        if wins >= 2 and negs >= 2:
            v = "MIXED"
        elif wins >= 2:
            v = "WEIGHTED BEATS GATE"
        elif negs >= 2:
            v = "GATE BEATS WEIGHTED"
        else:
            v = "NOT SEPARATED"
        L.append("\n" + "=" * 112)
        L.append("D59 VERDICT")
        L.append(f"  labels (WIN = weighted C beats Gate-C; NEGATIVE = gate beats weighted) by fraction: "
                 + ", ".join(f"{pt[1]:.2f}: {l}" for pt, l in labs.items()))
        L.append(f"  WIN at {wins} of 4, NEGATIVE at {negs} of 4  ->  {v}")
        L.append(f"  label at the shared point (fraction 0.20): {labs[(1.0, 0.2)]}")
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

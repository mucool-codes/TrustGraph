"""S14: variant D2 ("C plus neighbours") against the frozen C (DECISIONS.md D68, frozen).

    python scripts/s14_d2.py gen   [--seeds ...]            # scenarios for draws 250-299, both regimes (observable + SEALED)
    python scripts/s14_d2.py trust [--seeds ...]            # frozen C and D2 trust, stride 1, steps 100-1199, stored
    python scripts/s14_d2.py eval  [--seeds ...] [--verdict]   # PRIMARY (staggered latency) and SECONDARY (a) (shared AUC)
    python scripts/s14_d2.py tune  [--write]                # D52 grids for D2 on seed 100, one per regime
    python scripts/s14_d2.py closed [--seeds ...]           # SECONDARY (b): closed-loop runs A, C, D2 at the shared point
    python scripts/s14_d2.py report [--verdict]             # SECONDARY (b) tables
    python scripts/s14_d2.py check                          # harness checks on draws that are not results (C vs stored S9 / S13)

EVALUATION-SIDE: reads the sealed ground truth (onsets, classes) to score; allowlisted in tests/test_sealing.py.
The first result each command writes is dated in results/s14/first_number.json.
"""

from __future__ import annotations

import argparse
import itertools
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

import s13_stag as S13  # noqa: E402  (latency, late-onset, AUC and interval helpers are reused unchanged)

OUT = REPO_ROOT / "results" / "s14"
SEEDS = (1, 2, 3, 4, 5)
TUNING_SEED = 100
FRACTION = 0.20
EVAL_DRAWS = tuple(range(250, 300))
RHOS = (0.0, 1.0)
REGIMES = {
    # name: (config, directory of C's frozen checkpoints, weights file for A and C)
    "shared": ("configs/demo.yaml", "checkpoints", "configs/weights.yaml"),
    "stag": ("configs/staggered.yaml", "checkpoints/s13", "configs/weights_staggered.yaml"),
}
D2_WEIGHTS_FILE = REPO_ROOT / "configs" / "weights_d2.yaml"
AUC_WINDOW = (450, 650)
AUC_FLOOR, AUC_BOUND = 0.02, 0.05       # D43 section 4
SUCC_FLOOR, SUCC_BOUND = 0.01, 0.025    # D56 H3
ALPHAS, BETAS, GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0), (0.25, 0.5, 1.0), (0.15, 0.3, 0.6)
TUNE_DRAWS = (0, 1, 2)
_W: dict = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def note_first_number(what: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "first_number.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    if what not in cur:
        cur[what] = utc_now()
        p.write_text(json.dumps(cur, indent=1))


def ckpt_path(regime: str, variant: str, seed: int) -> Path:
    """C: existing frozen checkpoints (seeds 1-5, and 100); D2: checkpoints/s14/<regime>. Seeds > 5 are the single
    pre-declared NOISY extension (D68 section 4) and live in checkpoints/s14/ext/<regime>."""
    if seed > 5 and seed != TUNING_SEED:
        return REPO_ROOT / "checkpoints" / "s14" / "ext" / regime / f"s3-seed{seed}-{variant}.pt"
    d = REPO_ROOT / "checkpoints" / "s14" / regime if variant == "D2" else REPO_ROOT / REGIMES[regime][1]
    return d / f"s3-seed{seed}-{variant}.pt"


def load_models(base, regime: str, seed: int, variants=("C", "D2")):
    from trustgraph.model import build_trust_head
    from trustgraph.train import load_checkpoint

    out = {}
    for v in variants:
        m = build_trust_head(base.model, 0, base.device, v)
        blob = load_checkpoint(ckpt_path(regime, v, seed), m)
        assert blob["variant"] == v and blob["seed"] == seed
        m.eval()
        out[v] = m
    return out


def t_interval(per_seed):
    return S13.t_interval(per_seed)


# ------------------------------------------------------------------ scenario generation (draws 250-299)

def _single_thread():
    import torch

    torch.set_num_threads(1)


def _gen_one(job):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.observed import default_observed_path
    from trustgraph.scenario import build_world
    from trustgraph.sealed.ground_truth import sealed_path_for
    from trustgraph.simulator import generate_scenario
    from trustgraph.trace import default_trace_path, load_trace

    regime, seed, rho, draw = job
    config = REGIMES[regime][0]
    base = _W.get(("base", config)) or _W.setdefault(("base", config), load_config(config))
    cfg = with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=draw)
    out = default_observed_path(config, cfg)
    if out.exists() and sealed_path_for(out).exists():
        return job, "exists"
    key = ("world", config, seed)
    if key not in _W:
        _W[key] = (build_world(cfg), load_trace(default_trace_path(config, seed)))
    world, trace = _W[key]
    sc = generate_scenario(cfg, world, trace)
    sc.observed.save(out)
    sc.ground_truth.save(sealed_path_for(out))
    return job, "written"


def cmd_gen(seeds, workers: int, regimes) -> int:
    jobs = [(r, s, rho, d) for r in regimes for s in seeds for rho in RHOS for d in EVAL_DRAWS]
    print(f"[gen] {len(jobs)} scenarios", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for job, st in pool.imap_unordered(_gen_one, jobs, chunksize=2):
            n += 1
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ open-loop trust of the frozen models

def trust_path(regime, seed, rho, draw) -> Path:
    return OUT / regime / "trust" / f"seed{seed}-rho{rho:g}-draw{draw}.npz"


def _trust_one(job):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.dataset import graphset_from
    from trustgraph.observed import default_observed_path, load_observed
    from trustgraph.scenario import build_world
    from trustgraph.trace import default_trace_path, load_trace
    from trustgraph.train import predict_trust

    regime, seed, rho, draw = job
    out = trust_path(regime, seed, rho, draw)
    if out.exists():
        return job, "cached"
    config = REGIMES[regime][0]
    base = _W.get(("base", config)) or _W.setdefault(("base", config), load_config(config))
    key = ("st", regime, seed)
    if key not in _W:
        _W[key] = (build_world(with_overrides(base, seed=seed)), load_trace(default_trace_path(config, seed)),
                   load_models(base, regime, seed))
    world, trace, models = _W[key]
    cfg = with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=draw)
    observed = load_observed(default_observed_path(config, cfg))
    gs = graphset_from(cfg, world, trace, observed, stride=1, step_range=(100, 1200))
    assert gs.steps[0] == 100 and gs.steps[-1] == 1199
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, C=predict_trust(models["C"], gs).astype(np.float32),
                        D2=predict_trust(models["D2"], gs).astype(np.float32))
    tmp.replace(out)
    return job, "ok"


def cmd_trust(seeds, workers: int, regimes) -> int:
    jobs = [(r, s, rho, d) for r in regimes for s in seeds for rho in RHOS for d in EVAL_DRAWS]
    print(f"[trust] {len(jobs)} scenarios", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for job, st in pool.imap_unordered(_trust_one, jobs, chunksize=2):
            n += 1
            if n == 1:
                note_first_number("evaluation_first_trust")
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ PRIMARY: staggered latency, D2 minus C (D68 section 4)

def evaluate_stag(seeds):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.observed import default_observed_path
    from trustgraph.scenario import build_world
    from trustgraph.sealed.ground_truth import load_ground_truth, sealed_path_for

    config = REGIMES["stag"][0]
    base = load_config(config)
    res = {}
    for seed in seeds:
        nb = S13.same_segment_neighbours(build_world(with_overrides(base, seed=seed)).topology)
        for rho in RHOS:
            tr, gts = {}, {}
            for d in EVAL_DRAWS:
                with np.load(trust_path("stag", seed, rho, d)) as z:
                    tr[d] = {"C": z["C"], "D2": z["D2"]}
                cfg = with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=d)
                gts[d] = load_ground_truth(sealed_path_for(default_observed_path(config, cfg)))
            tau = {m: float(np.percentile(np.concatenate([tr[d][m][0:300].ravel() for d in EVAL_DRAWS]), 1.0)) for m in ("C", "D2")}
            rec = {m: {"lat": [], "det": []} for m in ("C", "D2")}
            n_late = []
            for d in EVAL_DRAWS:
                gt = gts[d]
                late = S13.late_onset_rsus(gt, nb)
                n_late.append(len(late))
                on = gt.onset_per_rsu()
                for m in ("C", "D2"):
                    if late:
                        lats = [S13.detect_latency(tr[d][m][:, v], tau[m], int(on[v])) for v in late]
                        rec[m]["lat"].append(float(np.mean([S13.SEARCH - 1 if l is None else l for l in lats])))
                        rec[m]["det"].append(float(np.mean([l is not None for l in lats])))
                    else:
                        rec[m]["lat"].append(float("nan"))
                        rec[m]["det"].append(float("nan"))
            res[(seed, rho)] = {"tau": tau, "n_late": n_late, **{m: {k: np.array(v) for k, v in rec[m].items()} for m in rec}}
    return res


def adv_reading(l1, adv1, adv0, l0):
    """D68 section 4 reading of the rho 0 control."""
    if l1 != "WIN":
        return f"rho 1.0 label {l1}: no structural reading"
    if adv0 >= adv1:
        return "WIN at rho 1.0 but the rho 0 advantage is as large or larger: the advantage is NOT structural"
    if adv0 > 0:
        return "WIN at rho 1.0 and a smaller D2 advantage at rho 0: both stated"
    return "WIN at rho 1.0 and no D2 advantage at rho 0: structural in the D65 sense (larger when failures are concentrated)"


def cmd_eval(seeds, verdict: bool) -> int:
    n = len(seeds)
    t = S13.T_CRIT[n]
    L = ["=" * 118, f"S14 PRIMARY: STAGGERED ONSET (S = 300), open-loop frozen C and D2-stagger, draws 250-299, fraction 0.20, seeds {list(seeds)}"
         + (" WITH D68 LABELS" if verdict else " (RAW: no labels)"), "=" * 118]
    res = evaluate_stag(seeds)
    summary = {}
    for rho in RHOS:
        L.append(f"\nrho {rho:g}  ({'TEST' if rho == 1.0 else 'CONTROL'})")
        L.append("  D42 accounting per seed: draws attempted / failed / aggregated (draws with a late-onset RSU) / late-onset RSUs")
        per = {k: [] for k in ("C_lat", "D2_lat", "dlat", "C_det", "D2_det")}
        for seed in seeds:
            r = res[(seed, rho)]
            ok = ~np.isnan(r["C"]["lat"])
            L.append(f"    seed {seed}: attempted {len(EVAL_DRAWS)}  failed 0  aggregated {int(ok.sum())}  late-onset RSUs {int(sum(r['n_late']))}  "
                     f"tau C {r['tau']['C']:.4f} D2 {r['tau']['D2']:.4f}")
            per["C_lat"].append(r["C"]["lat"][ok].mean())
            per["D2_lat"].append(r["D2"]["lat"][ok].mean())
            per["dlat"].append((r["D2"]["lat"][ok] - r["C"]["lat"][ok]).mean())
            per["C_det"].append(r["C"]["det"][ok].mean())
            per["D2_det"].append(r["D2"]["det"][ok].mean())
        s = {k: t_interval(v) for k, v in per.items()}
        lab = S13.label_latency(s["dlat"][0], s["dlat"][2], s["dlat"][3])
        summary[rho] = {"per": per, "s": s, "label": lab}
        L.append(f"  detection latency (steps after own onset, late-onset RSUs, censored 399): C {s['C_lat'][0]:.1f} +/- {s['C_lat'][1]:.1f}   "
                 f"D2 {s['D2_lat'][0]:.1f} +/- {s['D2_lat'][1]:.1f}   detected fraction C {s['C_det'][0]:.3f} D2 {s['D2_det'][0]:.3f}")
        L.append(f"  PRIMARY  D2 - C latency: {s['dlat'][0]:+.1f} (sd {s['dlat'][1]:.1f}) 95% [{s['dlat'][2]:+.1f}, {s['dlat'][3]:+.1f}]"
                 + (f"   {lab}" if verdict else ""))
        L.append(f"  per-seed D2 - C latency: {[round(float(x), 1) for x in per['dlat']]}")
    diff = [a - b for a, b in zip(summary[1.0]["per"]["dlat"], summary[0.0]["per"]["dlat"])]
    m, sd, lo, hi = t_interval(diff)
    L.append(f"\n(description) (D2 - C latency at rho 1.0) minus (at rho 0), per-seed paired: {m:+.1f} (sd {sd:.1f}) 95% [{lo:+.1f}, {hi:+.1f}]")
    if verdict:
        a1, a0 = summary[1.0], summary[0.0]
        reading = adv_reading(a1["label"], -a1["s"]["dlat"][0], -a0["s"]["dlat"][0], a0["label"])
        L += ["\n" + "=" * 118, f"D68 VERDICT (PRIMARY, {n} seeds, t = {t})",
              f"  rho 1.0 (test): {a1['label']}   rho 0 (control): {a0['label']}   -> {reading}"]
    if max(seeds) <= 5:  # the seeds 6-10 extension (D68 section 4) is for the primary only
        L += eval_shared_auc(seeds, verdict)
    text = "\n".join(L)
    tag = f"{n}seeds" if n != 5 else "5seeds"
    (OUT / f"report-eval-{'verdict' if verdict else 'raw'}-{tag}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def label_auc(mean, lo, hi):
    """D43 section 4 thresholds on the AUC difference D2 - C (W1 only). Check order NEGATIVE, WIN, NULL, NOISY."""
    if hi < 0 and mean <= -AUC_FLOOR:
        return "NEGATIVE"
    if lo > 0 and mean >= AUC_FLOOR:
        return "WIN"
    if hi < AUC_BOUND:
        return "NULL"
    return "NOISY"


def eval_shared_auc(seeds, verdict: bool):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.diagnostic import degraded_vs_reliable_auc
    from trustgraph.observed import default_observed_path

    config = REGIMES["shared"][0]
    base = load_config(config)
    steps = np.arange(100, 1200)
    L = ["\n" + "=" * 118, "S14 SECONDARY (a): SHARED ONSET, diagnostic AUC of 1 - trust over [450, 650), degraded vs reliable (D43), draws 250-299, fraction 0.20, "
         f"seeds {list(seeds)}" + (" WITH D43 THRESHOLDS" if verdict else " (RAW: no labels)"), "=" * 118]
    for rho in RHOS:
        per = {"C": [], "D2": [], "d": []}
        L.append(f"\nrho {rho:g}   D42: per seed attempted {len(EVAL_DRAWS)} failed 0 aggregated (draws with both classes present) shown")
        for seed in seeds:
            a = {"C": [], "D2": []}
            for d in EVAL_DRAWS:
                cfg = with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=d)
                p = default_observed_path(config, cfg)
                with np.load(trust_path("shared", seed, rho, d)) as z:
                    for m in a:
                        a[m].append(degraded_vs_reliable_auc(z[m], steps, p, AUC_WINDOW))
            a = {m: np.array(v) for m, v in a.items()}
            ok = ~np.isnan(a["C"]) & ~np.isnan(a["D2"])
            L.append(f"    seed {seed}: attempted {len(EVAL_DRAWS)}  failed 0  aggregated {int(ok.sum())}")
            per["C"].append(a["C"][ok].mean())
            per["D2"].append(a["D2"][ok].mean())
            per["d"].append((a["D2"][ok] - a["C"][ok]).mean())
        s = {k: t_interval(v) for k, v in per.items()}
        lab = label_auc(s["d"][0], s["d"][2], s["d"][3])
        L.append(f"  AUC  C {s['C'][0]:.4f} +/- {s['C'][1]:.4f}   D2 {s['D2'][0]:.4f} +/- {s['D2'][1]:.4f}")
        L.append(f"  D2 - C: {s['d'][0]:+.4f} (sd {s['d'][1]:.4f}) 95% [{s['d'][2]:+.4f}, {s['d'][3]:+.4f}]" + (f"   {lab}" if verdict else ""))
        L.append(f"  per-seed D2 - C: {[round(float(x), 4) for x in per['d']]}")
    return L


# ------------------------------------------------------------------ closed loop (D68 section 5)

def weight_grid():
    return list(itertools.product(ALPHAS, BETAS, GAMMAS))


def _cl_state(regime: str, seed: int):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.scenario import build_world
    from trustgraph.trace import default_trace_path, load_trace

    config = REGIMES[regime][0]
    key = ("cl", regime, seed)
    if key not in _W:
        base = _W.get(("base", config)) or _W.setdefault(("base", config), load_config(config))
        _W[key] = (build_world(with_overrides(base, seed=seed)), load_trace(default_trace_path(config, seed)),
                   load_models(base, regime, seed))
    return _W[("base", config)], _W[key]


def _cl_metrics(sc) -> dict:
    from trustgraph import sweep_metrics as M

    tasks, gt = sc.observed.tasks, sc.ground_truth
    num_steps = sc.observed.num_steps
    onset = int(gt.onset_step)
    total, deg = M.dispatch_series(tasks.step, tasks.rsu, gt.degraded, num_steps)
    av = M.avoidance(total, deg, onset)
    post = tasks.step >= onset
    det = M.detection(sc.extras["trust_history"], sc.observed.rsu_active, gt.degraded, onset)
    return {"success": float(tasks.success_rate(post)), "success_all": float(tasks.success_rate()),
            "n_av": av["n_av"], "n_win": av["n_win"], "t_avoid": av["t_avoid"], "never_avoided": av["never_avoided"],
            "s0": av["s0"], "latency": det["latency"], "detected_fraction": det["detected_fraction"]}


def _cl_one(task):
    from trustgraph.config import with_overrides
    from trustgraph.offload import build_offload_settings
    from trustgraph.simulator import run_closed_loop

    regime, variant, (alpha, beta, gamma), seed, rho, fraction, draw = task
    base, (world, trace, models) = _cl_state(regime, seed)
    cfg = with_overrides(base, seed=seed, rho=rho, fraction=fraction, draw=draw)
    settings = build_offload_settings(cfg, variant=variant, weights=(alpha, beta, gamma), record_trust=True)
    sc = run_closed_loop(cfg, world, trace, settings, model=models.get(variant))
    return task, _cl_metrics(sc)


def cl_path(regime, variant, seed, rho, draw) -> Path:
    return OUT / regime / "closed" / f"{variant}-seed{seed}-rho{rho:g}-draw{draw}.json"


def load_ac_weights(regime):
    w = yaml.safe_load((REPO_ROOT / REGIMES[regime][2]).read_text())["weights"]
    return {v: (w[v]["alpha"], w[v]["beta"], w[v]["gamma"]) for v in ("A", "C")}


def cmd_tune(write: bool, workers: int) -> int:
    tasks = [(r, "D2", w, TUNING_SEED, 1.0, FRACTION, d) for r in REGIMES for w in weight_grid() for d in TUNE_DRAWS]
    print(f"[tune] {len(tasks)} runs on tuning seed {TUNING_SEED} (D2, both regimes)", flush=True)
    res = {}
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for task, r in pool.imap_unordered(_cl_one, tasks, chunksize=3):
            res[task] = r
            if len(res) == 1:
                note_first_number("closed_loop_tuning_first_result")
    chosen, table = {}, {}
    for regime in REGIMES:
        rows = []
        for w in weight_grid():
            per = [res[(regime, "D2", w, TUNING_SEED, 1.0, FRACTION, d)]["success"] for d in TUNE_DRAWS]
            rows.append({"alpha": w[0], "beta": w[1], "gamma": w[2], "objective": float(np.mean(per)), "per_draw": per})
        best = max(range(len(rows)), key=lambda i: (rows[i]["objective"], -i))
        table[regime], chosen[regime] = rows, rows[best]
        print(f"\n=== D2-{regime}: objective = mean post-onset deadline success over draws {TUNE_DRAWS} (seed {TUNING_SEED})")
        print("  alpha  beta  gamma   objective   per-draw")
        for i, r in enumerate(rows):
            print(f"  {r['alpha']:5.2f} {r['beta']:5.2f} {r['gamma']:5.2f}   {r['objective']:.4f}    {[round(x, 4) for x in r['per_draw']]}{'  <== chosen' if i == best else ''}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tuning-grid.json").write_text(json.dumps({"grid": table, "chosen": chosen}, indent=1))
    if write:
        doc = {"comment": "S14 (D68 section 5b): D2 tuned on seed 100, rho 1.0, fraction 0.2, draws 0-2 by the D52 procedure, one set per regime.",
               "tuning_seed": TUNING_SEED,
               "weights": {r: {"D2": {k: chosen[r][k] for k in ("alpha", "beta", "gamma")}} for r in REGIMES}}
        D2_WEIGHTS_FILE.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(f"\nwrote {D2_WEIGHTS_FILE}")
    return 0


def cmd_closed(seeds, workers: int, regimes) -> int:
    d2w = yaml.safe_load(D2_WEIGHTS_FILE.read_text())["weights"]
    todo = []
    for r in regimes:
        w = {**load_ac_weights(r), "D2": tuple(d2w[r]["D2"][k] for k in ("alpha", "beta", "gamma"))}
        todo += [(r, v, w[v], s, 1.0, FRACTION, d) for s in seeds for d in EVAL_DRAWS for v in ("A", "C", "D2")
                 if not cl_path(r, v, s, 1.0, d).exists()]
    print(f"[closed] {len(todo)} runs to do", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for task, r in pool.imap_unordered(_cl_one, todo, chunksize=2):
            reg, v, _, s, rho, _, d = task
            p = cl_path(reg, v, s, rho, d)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(r))
            n += 1
            if n == 1:
                note_first_number("closed_loop_evaluation_first_result")
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    return 0


def label_succ(mean, lo, hi):
    """D56 H3 thresholds on D2 - C. Order NEGATIVE, WIN, NULL, NOISY."""
    if hi < 0 and mean <= -SUCC_FLOOR:
        return "NEGATIVE"
    if lo > 0 and mean >= SUCC_FLOOR:
        return "WIN"
    if hi < SUCC_BOUND:
        return "NULL"
    return "NOISY"


def cmd_report(seeds, verdict: bool) -> int:
    d2w = yaml.safe_load(D2_WEIGHTS_FILE.read_text())["weights"]
    L = ["=" * 118, f"S14 SECONDARY (b): closed-loop post-onset success at the shared point (rho 1.0, fraction 0.20), draws 250-299, seeds {list(seeds)}"
         + (" WITH D56 H3 LABELS" if verdict else " (RAW: no labels)"), "=" * 118]
    for regime in REGIMES:
        V = ("A", "C", "D2")
        data = {v: {k: np.array([[json.loads(cl_path(regime, v, s, 1.0, d).read_text())[k] for d in EVAL_DRAWS] for s in seeds], float)
                    for k in ("success", "n_av", "n_win", "never_avoided", "latency", "detected_fraction")} for v in V}
        w = {**load_ac_weights(regime), "D2": tuple(d2w[regime]["D2"][k] for k in ("alpha", "beta", "gamma"))}
        L.append(f"\n--- regime {regime}: weights (alpha, beta, gamma) A {w['A']}  C {w['C']}  D2 {w['D2']}")
        L.append(f"    D42: attempted {len(seeds) * len(EVAL_DRAWS)}  failed 0  aggregated {data['A']['success'].size}")
        for v in V:
            sm = data[v]["success"].mean(axis=1)
            L.append(f"  {v:3s} success {sm.mean():.4f} +/- {sm.std(ddof=1):.4f}")
        for a, b in (("D2", "C"), ("C", "A"), ("D2", "A")):
            m, sd, lo, hi = t_interval((data[a]["success"] - data[b]["success"]).mean(axis=1))
            L.append(f"  {a} - {b}: {m:+.4f} (sd {sd:.4f}) 95% [{lo:+.4f}, {hi:+.4f}]" + (f"   {label_succ(m, lo, hi)}" if verdict and (a, b) == ("D2", "C") else ""))
        L.append("  n_av / n_win / never-avoided (mean +/- sd over seeds of per-seed means), always together (D60 Q4):")
        for v in V:
            f = lambda k: (data[v][k].mean(axis=1).mean(), data[v][k].mean(axis=1).std(ddof=1))  # noqa: E731
            L.append(f"    {v:3s} n_av {f('n_av')[0]:7.1f} +/- {f('n_av')[1]:6.1f}   n_win {f('n_win')[0]:7.1f} +/- {f('n_win')[1]:6.1f}   "
                     f"never-avoided {f('never_avoided')[0]:.3f} +/- {f('never_avoided')[1]:.3f}")
        for k in ("n_av", "n_win"):
            red = (data["C"][k].mean(axis=1) - data["D2"][k].mean(axis=1)) / data["A"][k].mean(axis=1)
            m, sd, lo, hi = t_interval(red)
            L.append(f"  D57-normalised (n_C - n_D2)/n_A on {k}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (positive = D2 sends fewer to degraded RSUs; descriptive)")
        for v in ("C", "D2"):
            L.append(f"  detection latency (descriptive) {v}: {np.nanmean(data[v]['latency'], axis=1).mean():.1f} steps, detected fraction {np.nanmean(data[v]['detected_fraction'], axis=1).mean():.3f}")
    text = "\n".join(L)
    (OUT / f"report-closed-{'verdict' if verdict else 'raw'}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


# ------------------------------------------------------------------ harness checks (draws that are not results)

def cmd_check() -> int:
    """C closed loop on a shared-regime draw already in S9 must equal the stored S9 C bit-for-bit; same for S13 (staggered)."""
    _single_thread()
    ok = True
    for regime, stored, seed, draw in (("shared", REPO_ROOT / "results/s9/raw/rho1.00-frac0.20/seed1-draw100.json", 1, 100),
                                        ("stag", None, 1, 200)):
        w = load_ac_weights(regime)["C"]
        task, r = _cl_one((regime, "C", w, seed, 1.0, FRACTION, draw))
        if stored is not None:
            ref = json.loads(stored.read_text())["variants"]["C"]
            same = (ref["success_post_onset"] == r["success"] and ref["n_av"] == r["n_av"] and ref["n_win"] == r["n_win"]
                    and ref["never_avoided"] == r["never_avoided"] and ref["t_avoid"] == r["t_avoid"])
        else:
            ref = json.loads((REPO_ROOT / "results/s13/stag/closed/C-seed1-rho1-draw200.json").read_text())
            same = ref["success"] == r["success"]
        print(f"  {regime}: C closed loop draw {draw} seed {seed} equals the stored result: {same}  (success {r['success']:.6f})")
        ok &= same
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("gen", "trust", "eval", "tune", "closed", "report", "check"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    if a.cmd == "gen":
        return cmd_gen(a.seeds, a.workers, a.regimes)
    if a.cmd == "trust":
        return cmd_trust(a.seeds, a.workers, a.regimes)
    if a.cmd == "eval":
        return cmd_eval(a.seeds, a.verdict)
    if a.cmd == "tune":
        return cmd_tune(a.write, a.workers)
    if a.cmd == "closed":
        return cmd_closed(a.seeds, a.workers, a.regimes)
    if a.cmd == "report":
        return cmd_report(a.seeds, a.verdict)
    if a.cmd == "check":
        return cmd_check()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

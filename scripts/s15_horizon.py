"""S15: the horizon objective, C-h versus D2-h (DECISIONS.md D72, frozen).

    python scripts/s15_horizon.py gen   [--seeds ...]         # staggered scenarios for draws 300-349 (observable + SEALED)
    python scripts/s15_horizon.py counts                      # data counts of the two objectives on the training scenarios (descriptive)
    python scripts/s15_horizon.py tune  [--write]             # D52 grids for C-h and D2-h on seed 100 (staggered closed loop)
    python scripts/s15_horizon.py trust [--seeds ...]         # frozen C-h, D2-h and the original staggered C: trust, stride 1, steps 100-1199
    python scripts/s15_horizon.py eval  [--seeds ...] [--verdict]   # PRIMARY (D2-h - C-h latency) and SECONDARY (b) (C-h - C)
    python scripts/s15_horizon.py closed [--seeds ...]        # SECONDARY (a): closed-loop A, C-h, D2-h at rho 1.0
    python scripts/s15_horizon.py report                      # SECONDARY (a) tables
    python scripts/s15_horizon.py check                       # harness check on a draw that is not a result (original C vs stored S13)

EVALUATION-SIDE: reads the sealed ground truth (onsets, classes) to score; allowlisted in tests/test_sealing.py.
The first result each command writes is dated in results/s15/first_number.json.
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

import s13_stag as S13  # noqa: E402  (latency, late-onset and interval helpers are reused unchanged)

OUT = REPO_ROOT / "results" / "s15"
CONFIG = "configs/staggered.yaml"
SEEDS = (1, 2, 3, 4, 5)
TUNING_SEED = 100
FRACTION = 0.20
EVAL_DRAWS = tuple(range(300, 350))
RHOS = (0.0, 1.0)
WEIGHTS_FILE = REPO_ROOT / "configs" / "weights_horizon.yaml"
ALPHAS, BETAS, GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0), (0.25, 0.5, 1.0), (0.15, 0.3, 0.6)
TUNE_DRAWS = (0, 1, 2)
# model name -> (architecture variant, checkpoint directory)
MODELS = {"C_h": ("C", "checkpoints/s15"), "D2_h": ("D2", "checkpoints/s15"), "C_old": ("C", "checkpoints/s13")}
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


def t_interval(per_seed):
    return S13.t_interval(per_seed)


def _single_thread():
    import torch

    torch.set_num_threads(1)


def load_models(base, seed: int, names=("C_h", "D2_h", "C_old")):
    from trustgraph.model import build_trust_head
    from trustgraph.train import load_checkpoint

    out = {}
    for n in names:
        variant, d = MODELS[n]
        m = build_trust_head(base.model, 0, base.device, variant)
        blob = load_checkpoint(REPO_ROOT / d / f"s3-seed{seed}-{variant}.pt", m)
        assert blob["variant"] == variant and blob["seed"] == seed
        m.eval()
        out[n] = m
    return out


# ------------------------------------------------------------------ scenario generation (draws 300-349)

def _gen_one(job):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.observed import default_observed_path
    from trustgraph.scenario import build_world
    from trustgraph.sealed.ground_truth import sealed_path_for
    from trustgraph.simulator import generate_scenario
    from trustgraph.trace import default_trace_path, load_trace

    seed, rho, draw = job
    base = _W.get("base") or _W.setdefault("base", load_config(CONFIG))
    cfg = with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=draw)
    out = default_observed_path(CONFIG, cfg)
    if out.exists() and sealed_path_for(out).exists():
        return job, "exists"
    key = ("world", seed)
    if key not in _W:
        _W[key] = (build_world(cfg), load_trace(default_trace_path(CONFIG, seed)))
    world, trace = _W[key]
    sc = generate_scenario(cfg, world, trace)
    sc.observed.save(out)
    sc.ground_truth.save(sealed_path_for(out))
    return job, "written"


def cmd_gen(seeds, workers: int) -> int:
    jobs = [(s, rho, d) for s in seeds for rho in RHOS for d in EVAL_DRAWS]
    print(f"[gen] {len(jobs)} scenarios", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for job, st in pool.imap_unordered(_gen_one, jobs, chunksize=2):
            n += 1
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ data counts (descriptive)

def cmd_counts(seeds) -> int:
    """Share of active RSU-steps that carry a label under each objective, and the share with y = 1 (z = 0) among labelled ones,
    over the S3 training scenarios of each seed (staggered regime). Descriptive; nothing is tuned on it (D72 section 2)."""
    from trustgraph.config import load_config, with_overrides
    from trustgraph.dataset import labels_for
    from trustgraph.observed import default_observed_path, load_observed

    base = load_config(CONFIG)
    L = ["seed   active RSU-steps   next: labelled share   next: share failing (y=0)   horizon: labelled share   horizon: share y=1 (f < 0.75)"]
    for seed in seeds:
        acc = {k: [0, 0, 0, 0, 0] for k in ("next", "horizon")}
        for rho in S13.TRAIN_RHO:
            for frac in S13.TRAIN_FRAC:
                for d in S13.TRAIN_DRAWS:
                    obs = load_observed(default_observed_path(CONFIG, with_overrides(base, seed=seed, rho=rho, fraction=frac, draw=d)))
                    active = int(obs.rsu_active.sum())
                    for k in acc:
                        z, v = labels_for(obs, k)
                        acc[k][0] += active
                        acc[k][1] += int(v.sum())
                        acc[k][2] += int((z[v] == 0).sum())
        a, b = acc["next"], acc["horizon"]
        L.append(f"{seed:4d} {a[0]:18d} {a[1] / a[0]:22.3f} {a[2] / a[1]:27.3f} {b[1] / b[0]:25.3f} {b[2] / b[1]:36.3f}")
    text = "\n".join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "counts.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


# ------------------------------------------------------------------ open-loop trust of the frozen models

def trust_path(seed, rho, draw) -> Path:
    return OUT / "trust" / f"seed{seed}-rho{rho:g}-draw{draw}.npz"


def _trust_one(job):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.dataset import graphset_from
    from trustgraph.observed import default_observed_path, load_observed
    from trustgraph.scenario import build_world
    from trustgraph.trace import default_trace_path, load_trace
    from trustgraph.train import predict_trust

    seed, rho, draw = job
    out = trust_path(seed, rho, draw)
    if out.exists():
        return job, "cached"
    base = _W.get("base") or _W.setdefault("base", load_config(CONFIG))
    key = ("st", seed)
    if key not in _W:
        _W[key] = (build_world(with_overrides(base, seed=seed)), load_trace(default_trace_path(CONFIG, seed)), load_models(base, seed))
    world, trace, models = _W[key]
    cfg = with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=draw)
    observed = load_observed(default_observed_path(CONFIG, cfg))
    gs = graphset_from(cfg, world, trace, observed, stride=1, step_range=(100, 1200))
    assert gs.steps[0] == 100 and gs.steps[-1] == 1199
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **{n: predict_trust(m, gs).astype(np.float32) for n, m in models.items()})
    tmp.replace(out)
    return job, "ok"


def cmd_trust(seeds, workers: int) -> int:
    jobs = [(s, rho, d) for s in seeds for rho in RHOS for d in EVAL_DRAWS]
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


# ------------------------------------------------------------------ PRIMARY and SECONDARY (b): latency of late-onset RSUs (D72 sections 4, 5b)

NAMES = ("C_h", "D2_h", "C_old")


def evaluate(seeds):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.observed import default_observed_path
    from trustgraph.scenario import build_world
    from trustgraph.sealed.ground_truth import load_ground_truth, sealed_path_for

    base = load_config(CONFIG)
    res = {}
    for seed in seeds:
        nb = S13.same_segment_neighbours(build_world(with_overrides(base, seed=seed)).topology)
        for rho in RHOS:
            tr, gts = {}, {}
            for d in EVAL_DRAWS:
                with np.load(trust_path(seed, rho, d)) as z:
                    tr[d] = {m: z[m] for m in NAMES}
                cfg = with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=d)
                gts[d] = load_ground_truth(sealed_path_for(default_observed_path(CONFIG, cfg)))
            tau = {m: float(np.percentile(np.concatenate([tr[d][m][0:300].ravel() for d in EVAL_DRAWS]), 1.0)) for m in NAMES}
            rec = {m: {"lat": [], "det": []} for m in NAMES}
            n_late = []
            for d in EVAL_DRAWS:
                gt = gts[d]
                late = S13.late_onset_rsus(gt, nb)
                n_late.append(len(late))
                on = gt.onset_per_rsu()
                for m in NAMES:
                    if late:
                        lats = [S13.detect_latency(tr[d][m][:, v], tau[m], int(on[v])) for v in late]
                        rec[m]["lat"].append(float(np.mean([S13.SEARCH - 1 if l is None else l for l in lats])))
                        rec[m]["det"].append(float(np.mean([l is not None for l in lats])))
                    else:
                        rec[m]["lat"].append(float("nan"))
                        rec[m]["det"].append(float("nan"))
            res[(seed, rho)] = {"tau": tau, "n_late": n_late, **{m: {k: np.array(v) for k, v in rec[m].items()} for m in rec}}
    return res


def adv_reading(l1, adv1, adv0):
    """D72 section 4 reading of the rho 0 control."""
    if l1 != "WIN":
        return f"rho 1.0 label {l1}: no structural reading"
    if adv0 >= adv1:
        return "WIN at rho 1.0 but the rho 0 advantage is as large or larger: the effect is NOT structural"
    if adv0 > 0:
        return "WIN at rho 1.0 and a smaller D2-h advantage at rho 0: both stated"
    return "WIN at rho 1.0 and no D2-h advantage at rho 0: structural in the D65 sense (larger when failures are concentrated)"


def cmd_eval(seeds, verdict: bool) -> int:
    n = len(seeds)
    t = S13.T_CRIT[n]
    L = ["=" * 118, f"S15 PRIMARY: STAGGERED ONSET (S = 300), open-loop frozen C-h and D2-h (horizon objective), draws 300-349, fraction 0.20, seeds {list(seeds)}"
         + (" WITH D72 LABELS" if verdict else " (RAW: no labels)"), "=" * 118]
    res = evaluate(seeds)
    summary = {}
    for rho in RHOS:
        L.append(f"\nrho {rho:g}  ({'TEST' if rho == 1.0 else 'CONTROL'})")
        L.append("  D42 accounting per seed: draws attempted / failed / aggregated (draws with a late-onset RSU) / late-onset RSUs")
        per = {k: [] for k in ("C_h_lat", "D2_h_lat", "C_old_lat", "dlat", "dlat_h_old", "C_h_det", "D2_h_det", "C_old_det")}
        for seed in seeds:
            r = res[(seed, rho)]
            ok = ~np.isnan(r["C_h"]["lat"])
            L.append(f"    seed {seed}: attempted {len(EVAL_DRAWS)}  failed 0  aggregated {int(ok.sum())}  late-onset RSUs {int(sum(r['n_late']))}  "
                     f"tau C-h {r['tau']['C_h']:.4f} D2-h {r['tau']['D2_h']:.4f} C(S13) {r['tau']['C_old']:.4f}")
            per["C_h_lat"].append(r["C_h"]["lat"][ok].mean())
            per["D2_h_lat"].append(r["D2_h"]["lat"][ok].mean())
            per["C_old_lat"].append(r["C_old"]["lat"][ok].mean())
            per["dlat"].append((r["D2_h"]["lat"][ok] - r["C_h"]["lat"][ok]).mean())
            per["dlat_h_old"].append((r["C_h"]["lat"][ok] - r["C_old"]["lat"][ok]).mean())
            per["C_h_det"].append(r["C_h"]["det"][ok].mean())
            per["D2_h_det"].append(r["D2_h"]["det"][ok].mean())
            per["C_old_det"].append(r["C_old"]["det"][ok].mean())
        s = {k: t_interval(v) for k, v in per.items()}
        lab = S13.label_latency(s["dlat"][0], s["dlat"][2], s["dlat"][3])
        summary[rho] = {"per": per, "s": s, "label": lab}
        L.append(f"  detection latency (steps after own onset, late-onset RSUs, censored 399): C-h {s['C_h_lat'][0]:.1f} +/- {s['C_h_lat'][1]:.1f}   "
                 f"D2-h {s['D2_h_lat'][0]:.1f} +/- {s['D2_h_lat'][1]:.1f}   C(S13, next-task objective) {s['C_old_lat'][0]:.1f} +/- {s['C_old_lat'][1]:.1f}")
        L.append(f"  detected fraction: C-h {s['C_h_det'][0]:.3f}  D2-h {s['D2_h_det'][0]:.3f}  C(S13) {s['C_old_det'][0]:.3f}")
        L.append(f"  PRIMARY  D2-h - C-h latency: {s['dlat'][0]:+.1f} (sd {s['dlat'][1]:.1f}) 95% [{s['dlat'][2]:+.1f}, {s['dlat'][3]:+.1f}]"
                 + (f"   {lab}" if verdict else ""))
        L.append(f"  per-seed D2-h - C-h latency: {[round(float(x), 1) for x in per['dlat']]}")
        L.append(f"  SECONDARY (b) (descriptive, no label)  C-h - C(S13) latency: {s['dlat_h_old'][0]:+.1f} (sd {s['dlat_h_old'][1]:.1f}) "
                 f"95% [{s['dlat_h_old'][2]:+.1f}, {s['dlat_h_old'][3]:+.1f}]   per seed {[round(float(x), 1) for x in per['dlat_h_old']]}")
    diff = [a - b for a, b in zip(summary[1.0]["per"]["dlat"], summary[0.0]["per"]["dlat"])]
    m, sd, lo, hi = t_interval(diff)
    L.append(f"\n(description) (D2-h - C-h latency at rho 1.0) minus (at rho 0), per-seed paired: {m:+.1f} (sd {sd:.1f}) 95% [{lo:+.1f}, {hi:+.1f}]")
    if verdict:
        a1, a0 = summary[1.0], summary[0.0]
        reading = adv_reading(a1["label"], -a1["s"]["dlat"][0], -a0["s"]["dlat"][0])
        L += ["\n" + "=" * 118, f"D72 VERDICT (PRIMARY, {n} seeds, t = {t})",
              f"  rho 1.0 (test): {a1['label']}   rho 0 (control, same rule): {a0['label']}   -> {reading}"]
    text = "\n".join(L)
    tag = f"{n}seeds"
    (OUT / f"report-eval-{'verdict' if verdict else 'raw'}-{tag}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


# ------------------------------------------------------------------ closed loop (D72 section 5a)

def weight_grid():
    return list(itertools.product(ALPHAS, BETAS, GAMMAS))


def _cl_state(seed: int):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.scenario import build_world
    from trustgraph.trace import default_trace_path, load_trace

    key = ("cl", seed)
    if key not in _W:
        base = _W.get("base") or _W.setdefault("base", load_config(CONFIG))
        _W[key] = (build_world(with_overrides(base, seed=seed)), load_trace(default_trace_path(CONFIG, seed)),
                   load_models(base, seed, names=("C_h", "D2_h", "C_old")))
    return _W["base"], _W[key]


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
    """task = (name, weights, seed, rho, fraction, draw); name is A, C_h, D2_h or C_old."""
    from trustgraph.config import with_overrides
    from trustgraph.offload import build_offload_settings
    from trustgraph.simulator import run_closed_loop

    name, (alpha, beta, gamma), seed, rho, fraction, draw = task
    base, (world, trace, models) = _cl_state(seed)
    cfg = with_overrides(base, seed=seed, rho=rho, fraction=fraction, draw=draw)
    variant = "A" if name == "A" else MODELS[name][0]
    settings = build_offload_settings(cfg, variant=variant, weights=(alpha, beta, gamma), record_trust=True)
    sc = run_closed_loop(cfg, world, trace, settings, model=models.get(name))
    return task, _cl_metrics(sc)


def cl_path(name, seed, rho, draw) -> Path:
    return OUT / "closed" / f"{name}-seed{seed}-rho{rho:g}-draw{draw}.json"


def load_weights() -> dict:
    """A from weights_staggered.yaml (unchanged); C_h and D2_h from the new weights_horizon.yaml."""
    a = yaml.safe_load((REPO_ROOT / "configs" / "weights_staggered.yaml").read_text())["weights"]["A"]
    w = {"A": (a["alpha"], a["beta"], a["gamma"])}
    h = yaml.safe_load(WEIGHTS_FILE.read_text())["weights"]
    for n in ("C_h", "D2_h"):
        w[n] = tuple(h[n][k] for k in ("alpha", "beta", "gamma"))
    return w


def cmd_tune(write: bool, workers: int) -> int:
    names = ("C_h", "D2_h")
    tasks = [(n, w, TUNING_SEED, 1.0, FRACTION, d) for n in names for w in weight_grid() for d in TUNE_DRAWS]
    print(f"[tune] {len(tasks)} runs on tuning seed {TUNING_SEED} (C-h and D2-h, staggered)", flush=True)
    res = {}
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for task, r in pool.imap_unordered(_cl_one, tasks, chunksize=3):
            res[task] = r
            if len(res) == 1:
                note_first_number("closed_loop_tuning_first_result")
    chosen, table = {}, {}
    for n in names:
        rows = []
        for w in weight_grid():
            per = [res[(n, w, TUNING_SEED, 1.0, FRACTION, d)]["success"] for d in TUNE_DRAWS]
            rows.append({"alpha": w[0], "beta": w[1], "gamma": w[2], "objective": float(np.mean(per)), "per_draw": per})
        best = max(range(len(rows)), key=lambda i: (rows[i]["objective"], -i))
        table[n], chosen[n] = rows, rows[best]
        print(f"\n=== {n}: objective = mean post-onset deadline success over draws {TUNE_DRAWS} (seed {TUNING_SEED})")
        print("  alpha  beta  gamma   objective   per-draw")
        for i, r in enumerate(rows):
            print(f"  {r['alpha']:5.2f} {r['beta']:5.2f} {r['gamma']:5.2f}   {r['objective']:.4f}    {[round(x, 4) for x in r['per_draw']]}{'  <== chosen' if i == best else ''}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tuning-grid.json").write_text(json.dumps({"grid": table, "chosen": chosen}, indent=1))
    if write:
        doc = {"comment": "S15 (D72 section 5a): C-h and D2-h tuned on seed 100, rho 1.0, fraction 0.2, draws 0-2, staggered, by the D52 procedure.",
               "tuning_seed": TUNING_SEED,
               "weights": {n: {k: chosen[n][k] for k in ("alpha", "beta", "gamma")} for n in names}}
        WEIGHTS_FILE.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(f"\nwrote {WEIGHTS_FILE}")
    return 0


def cmd_closed(seeds, workers: int) -> int:
    w = load_weights()
    todo = [(v, w[v], s, 1.0, FRACTION, d) for s in seeds for d in EVAL_DRAWS for v in ("A", "C_h", "D2_h")
            if not cl_path(v, s, 1.0, d).exists()]
    print(f"[closed] {len(todo)} runs to do", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for task, r in pool.imap_unordered(_cl_one, todo, chunksize=2):
            name, _, s, rho, _, d = task
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
    V = ("A", "C_h", "D2_h")
    L = ["=" * 118, f"S15 SECONDARY (a): closed-loop post-onset success, staggered, rho 1.0, fraction 0.20, draws 300-349, seeds {list(seeds)} (RAW, descriptive: no labels)", "=" * 118]
    data = {v: {k: np.array([[json.loads(cl_path(v, s, 1.0, d).read_text())[k] for d in EVAL_DRAWS] for s in seeds], float)
                for k in ("success", "n_av", "n_win", "never_avoided", "latency", "detected_fraction")} for v in V}
    L.append(f"weights (alpha, beta, gamma): A {w['A']}  C-h {w['C_h']}  D2-h {w['D2_h']}")
    L.append(f"D42: attempted {len(seeds) * len(EVAL_DRAWS)}  failed 0  aggregated {data['A']['success'].size}")
    for v in V:
        sm = data[v]["success"].mean(axis=1)
        L.append(f"  {v:5s} success {sm.mean():.4f} +/- {sm.std(ddof=1):.4f}")
    for a, b in (("D2_h", "C_h"), ("C_h", "A"), ("D2_h", "A")):
        m, sd, lo, hi = t_interval((data[a]["success"] - data[b]["success"]).mean(axis=1))
        L.append(f"  {a} - {b}: {m:+.4f} (sd {sd:.4f}) 95% [{lo:+.4f}, {hi:+.4f}]")
    L.append("  n_av / n_win / never-avoided (mean +/- sd over seeds of per-seed means), always together (D60 Q4):")
    for v in V:
        f = lambda k: (data[v][k].mean(axis=1).mean(), data[v][k].mean(axis=1).std(ddof=1))  # noqa: E731
        L.append(f"    {v:5s} n_av {f('n_av')[0]:7.1f} +/- {f('n_av')[1]:6.1f}   n_win {f('n_win')[0]:7.1f} +/- {f('n_win')[1]:6.1f}   "
                 f"never-avoided {f('never_avoided')[0]:.3f} +/- {f('never_avoided')[1]:.3f}")
    for k in ("n_av", "n_win"):
        red = (data["C_h"][k].mean(axis=1) - data["D2_h"][k].mean(axis=1)) / data["A"][k].mean(axis=1)
        m, sd, lo, hi = t_interval(red)
        L.append(f"  D57-normalised (n_C-h - n_D2-h)/n_A on {k}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (positive = D2-h sends fewer to degraded RSUs; descriptive)")
    for v in ("C_h", "D2_h"):
        L.append(f"  detection latency (closed loop, D56 rule on each run's own trust, descriptive) {v}: {np.nanmean(data[v]['latency'], axis=1).mean():.1f} steps, "
                 f"detected fraction {np.nanmean(data[v]['detected_fraction'], axis=1).mean():.3f}")
    text = "\n".join(L)
    (OUT / "report-closed-raw.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


# ------------------------------------------------------------------ harness check (a draw that is not a result)

def cmd_check() -> int:
    """The original staggered C, run through this script's closed-loop path on draw 200 (an S13 draw), must equal the stored S13 result,
    and its open-loop trust must equal the stored S13 trust array: the harness is the S13 one."""
    from trustgraph.config import load_config, with_overrides
    from trustgraph.dataset import graphset_from
    from trustgraph.observed import default_observed_path, load_observed
    from trustgraph.train import predict_trust

    _single_thread()
    a = yaml.safe_load((REPO_ROOT / "configs" / "weights_staggered.yaml").read_text())["weights"]["C"]
    task, r = _cl_one(("C_old", (a["alpha"], a["beta"], a["gamma"]), 1, 1.0, FRACTION, 200))
    ref = json.loads((REPO_ROOT / "results/s13/stag/closed/C-seed1-rho1-draw200.json").read_text())
    same_cl = ref["success"] == r["success"]
    print(f"  closed loop: original C, seed 1, draw 200 equals the stored S13 result: {same_cl} (success {r['success']:.6f})")
    base, (world, trace, models) = _cl_state(1)
    cfg = with_overrides(base, seed=1, rho=1.0, fraction=FRACTION, draw=200)
    gs = graphset_from(cfg, world, trace, load_observed(default_observed_path(CONFIG, cfg)), stride=1, step_range=(100, 1200))
    tr = predict_trust(models["C_old"], gs).astype(np.float32)
    with np.load(REPO_ROOT / "results/s13/stag/trust/seed1-rho1-draw200.npz") as z:
        stored = z["C"]
    same_ol = bool(np.array_equal(tr, stored))
    print(f"  open loop: original C trust, seed 1, rho 1, draw 200 equals the stored S13 array: {same_ol}")
    return 0 if (same_cl and same_ol) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("gen", "counts", "trust", "eval", "tune", "closed", "report", "check"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    if a.cmd == "gen":
        return cmd_gen(a.seeds, a.workers)
    if a.cmd == "counts":
        return cmd_counts(a.seeds)
    if a.cmd == "trust":
        return cmd_trust(a.seeds, a.workers)
    if a.cmd == "eval":
        return cmd_eval(a.seeds, a.verdict)
    if a.cmd == "tune":
        return cmd_tune(a.write, a.workers)
    if a.cmd == "closed":
        return cmd_closed(a.seeds, a.workers)
    if a.cmd == "report":
        return cmd_report(a.seeds)
    if a.cmd == "check":
        return cmd_check()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

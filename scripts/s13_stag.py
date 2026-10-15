"""S13 Experiment 2: staggered onset (DECISIONS.md D65, frozen).

    python scripts/s13_stag.py gen mech|train|eval [--workers N]   # write staggered scenarios (observable + SEALED) to scenarios/
    python scripts/s13_stag.py mech                                 # mechanism check FIRST (D65 2.3); gate: rho 1.0 lead interval > 0
    python scripts/s13_stag.py eval [--seeds ...]                   # frozen C/D on draws 200-249: latency of late-onset RSUs, diagnostic AUC
    python scripts/s13_stag.py tune [--write]                       # D52 grids for A, C, D under staggered onset on seed 100
    python scripts/s13_stag.py closed                               # closed-loop success with each model's own tuned weights
    python scripts/s13_stag.py report [--verdict]

EVALUATION-SIDE: reads the sealed ground truth (onsets, classes) to score; allowlisted in tests/test_sealing.py.
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

CONFIG = "configs/staggered.yaml"
OUT = REPO_ROOT / "results" / "s13" / "stag"
CKPT_DIR = "checkpoints/s13"
SEEDS = (1, 2, 3, 4, 5)
TUNING_SEED = 100
FRACTION = 0.20
MECH_DRAWS = tuple(range(50, 80))
EVAL_DRAWS = tuple(range(200, 250))
RHOS = (0.0, 1.0)
# S3 protocol (scripts/s3_train.py): train / validation / held-out scenarios, all draw indices < 100
TRAIN_RHO, TRAIN_FRAC, TRAIN_DRAWS = (0.0, 0.25, 0.5, 0.75, 1.0), (0.1, 0.2, 0.3), (0, 1)
VAL_RHO, VAL_DRAW = (0.0, 0.5, 1.0), 40
HELD_OUT = (1.0, 0.2, 90)

T_CRIT = {5: 2.776, 10: 2.262}  # Student-t 95% two-sided, df = n - 1
SEARCH = 400   # D65 2.3 / 2.5: detection and movement are sought in [onset, onset + 400)
HOLD = 10
LAT_FLOOR = 20.0  # D65 2.5: +-20 steps


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
    v = np.asarray(per_seed, float)
    n = v.size
    m, sd = float(v.mean()), float(v.std(ddof=1))
    h = T_CRIT[n] * sd / np.sqrt(n)
    return m, sd, m - h, m + h


# ------------------------------------------------------------------ scenario generation

def scenario_jobs(which: str, seeds) -> list[tuple]:
    jobs = []
    for s in seeds:
        if which == "mech":
            jobs += [(s, rho, FRACTION, d) for rho in RHOS for d in MECH_DRAWS]
        elif which == "eval":
            jobs += [(s, rho, FRACTION, d) for rho in RHOS for d in EVAL_DRAWS]
        elif which == "train":
            jobs += [(s, r, f, d) for r in TRAIN_RHO for f in TRAIN_FRAC for d in TRAIN_DRAWS]
            jobs += [(s, r, f, VAL_DRAW) for r in VAL_RHO for f in TRAIN_FRAC]
            jobs += [(s, *HELD_OUT)]
        else:
            raise SystemExit(f"unknown scenario set {which!r}")
    return jobs


_W: dict = {}


def _gen_init():
    import torch

    torch.set_num_threads(1)


def _gen_one(job):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.observed import default_observed_path
    from trustgraph.scenario import build_world
    from trustgraph.sealed.ground_truth import sealed_path_for
    from trustgraph.simulator import generate_scenario
    from trustgraph.trace import default_trace_path, load_trace

    seed, rho, frac, draw = job
    base = _W.get("base") or _W.setdefault("base", load_config(CONFIG))
    cfg = with_overrides(base, seed=seed, rho=rho, fraction=frac, draw=draw)
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


def cmd_gen(which: str, seeds, workers: int) -> int:
    jobs = scenario_jobs(which, seeds)
    print(f"[gen {which}] {len(jobs)} scenarios", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_gen_init) as pool:
        for job, status in pool.imap_unordered(_gen_one, jobs, chunksize=2):
            n += 1
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ mechanism check (D65 2.3)

def first_run(below: np.ndarray, start: int, end: int):
    """First t in [start, end) with below[t:t+HOLD] all true, or None."""
    cs = np.concatenate([[0], np.cumsum(below.astype(np.int64))])
    for t in range(start, end):
        if t + HOLD <= below.size and cs[t + HOLD] - cs[t] == HOLD:
            return t
    return None


def move_pctile(series, start):
    return first_run(series < np.percentile(series[100:400], 1), start, start + SEARCH)


def move_drop(series, start):
    return first_run(series < series[100:400].mean() - 0.10, start, start + SEARCH)


def same_segment_neighbours(topo):
    seg = topo.backhaul_segment_id
    nb = [[] for _ in range(topo.num_rsus)]
    for i, j in topo.rsu_edges:
        if seg[i] == seg[j]:
            nb[int(i)].append(int(j))
            nb[int(j)].append(int(i))
    return nb


def cmd_mech() -> int:
    from trustgraph.config import load_config, with_overrides
    from trustgraph.observed import default_observed_path, load_observed
    from trustgraph.scenario import build_world
    from trustgraph.sealed.ground_truth import DEGRADED, load_ground_truth, sealed_path_for

    base = load_config(CONFIG)
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    for rule, mover in (("primary", move_pctile), ("secondary", move_drop)):
        for rho in RHOS:
            per = {k: [] for k in ("lead", "gap", "cens_v", "cens_u", "n")}
            for seed in SEEDS:
                topo = build_world(with_overrides(base, seed=seed)).topology
                nb = same_segment_neighbours(topo)
                leads, gaps, cv, cu = [], [], [], []
                for d in MECH_DRAWS:
                    cfg = with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=d)
                    p = default_observed_path(CONFIG, cfg)
                    ewma = load_observed(p).feature("success_ewma")
                    gt = load_ground_truth(sealed_path_for(p))
                    on = gt.onset_per_rsu()
                    deg = set(np.flatnonzero(gt.behavior_class == DEGRADED).tolist())
                    t_of = {w: mover(ewma[:, w], int(on[w])) for w in deg}
                    for v in deg:
                        earlier = [u for u in nb[v] if u in deg and on[u] < on[v]]
                        if not earlier:
                            continue
                        tv = t_of[v] if t_of[v] is not None else int(on[v]) + SEARCH - 1
                        tus = [t_of[u] if t_of[u] is not None else int(on[u]) + SEARCH - 1 for u in earlier]
                        leads.append(np.mean([tv - tu for tu in tus]))
                        gaps.append(np.mean([on[v] - on[u] for u in earlier]))
                        cv.append(t_of[v] is None)
                        cu += [t_of[u] is None for u in earlier]
                        if rule == "primary" and "first" not in result:
                            result["first"] = True
                            note_first_number("exp2_mechanism_check_first_result")
                per["lead"].append(float(np.mean(leads)))
                per["gap"].append(float(np.mean(gaps)))
                per["cens_v"].append(float(np.mean(cv)))
                per["cens_u"].append(float(np.mean(cu)))
                per["n"].append(len(leads))
            result[f"{rule}_rho{rho:g}"] = {
                "lead_per_seed": per["lead"], "lead": t_interval(per["lead"]), "onset_gap": t_interval(per["gap"])[0],
                "censored_late": float(np.mean(per["cens_v"])), "censored_earlier": float(np.mean(per["cens_u"])),
                "n_late_per_seed": per["n"],
            }
    gate = result["primary_rho1"]["lead"][2] > 0
    result["gate_pass"] = bool(gate)
    (OUT / "mech.json").write_text(json.dumps(result, indent=1))
    print("MECHANISM CHECK (D65 2.3): lead = mean over earlier-onset degraded same-segment neighbours of (t_v - t_u), steps;")
    print("positive = neighbour evidence first; late-onset RSU v; draws 50-79; per-seed mean, 95% t over 5 seeds")
    for k, r in result.items():
        if not isinstance(r, dict):
            continue
        m, sd, lo, hi = r["lead"]
        print(f"  {k:16s} lead {m:+7.1f} (sd {sd:5.1f}) [{lo:+7.1f}, {hi:+7.1f}]  mean onset gap {r['onset_gap']:6.1f}  "
              f"censored late {r['censored_late']:.3f} earlier {r['censored_earlier']:.3f}  n late per seed {r['n_late_per_seed']}")
    print(f"GATE (primary rule, rho 1.0, lo > 0): {'PASS' if gate else 'FAIL'}")
    return 0


# ------------------------------------------------------------------ frozen models, open-loop trust (D65 2.5)

def load_models(base, seed: int):
    from trustgraph.model import build_trust_head
    from trustgraph.train import load_checkpoint

    out = {}
    for v in ("C", "D"):
        m = build_trust_head(base.model, 0, base.device, v)
        blob = load_checkpoint(REPO_ROOT / CKPT_DIR / f"s3-seed{seed}-{v}.pt", m)
        assert blob["variant"] == v and blob["seed"] == seed
        m.eval()
        out[v] = m
    return out


def trust_path(seed, rho, draw) -> Path:
    return OUT / "trust" / f"seed{seed}-rho{rho:g}-draw{draw}.npz"


def _trust_init():
    import torch

    torch.set_num_threads(1)


def _trust_one(job):
    """Frozen C and D trust for steps 100..1199 of one evaluation scenario (stride 1)."""
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
        cfg0 = with_overrides(base, seed=seed)
        _W[key] = (build_world(cfg0), load_trace(default_trace_path(CONFIG, seed)), load_models(base, seed))
    world, trace, models = _W[key]
    cfg = with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=draw)
    observed = load_observed(default_observed_path(CONFIG, cfg))
    gs = graphset_from(cfg, world, trace, observed, stride=1, step_range=(100, 1200))
    assert gs.steps[0] == 100 and gs.steps[-1] == 1199
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, C=predict_trust(models["C"], gs).astype(np.float32), D=predict_trust(models["D"], gs).astype(np.float32))
    tmp.replace(out)
    return job, "ok"


def cmd_trust(seeds, workers: int) -> int:
    jobs = [(s, rho, d) for s in seeds for rho in RHOS for d in EVAL_DRAWS]
    print(f"[trust] {len(jobs)} scenarios", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_trust_init) as pool:
        for job, st in pool.imap_unordered(_trust_one, jobs, chunksize=2):
            n += 1
            if n == 1:
                note_first_number("exp2_evaluation_first_trust")
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0


def late_onset_rsus(gt, nb) -> list[int]:
    on = gt.onset_per_rsu()
    deg = set(np.flatnonzero(gt.degraded).tolist())
    return [v for v in sorted(deg) if any(u in deg and on[u] < on[v] for u in nb[v])]


def detect_latency(trust_col: np.ndarray, tau: float, onset: int):
    """Latency (steps after the RSU's own onset) or None. trust_col is indexed from step 100."""
    below = trust_col < tau
    cs = np.concatenate([[0], np.cumsum(below.astype(np.int64))])
    for t in range(onset, onset + SEARCH - HOLD + 1):  # the whole run lies inside [onset, onset + 400)
        a, b = t - 100, t - 100 + HOLD
        if cs[b] - cs[a] == HOLD:
            return t - onset
    return None


def window_auc(trust: np.ndarray, gt) -> float:
    """D65 2.5 secondary (a): AUC of 1 - mean trust over [first onset on the RSU's segment, + 250)."""
    from trustgraph.diagnostic import auc

    on = gt.onset_per_rsu()
    deg = gt.degraded
    seg = gt.backhaul_segment_id
    global_first = int(on[deg].min())
    scores, labels = [], []
    for r in range(gt.num_rsus):
        same = deg & (seg == seg[r])
        start = int(on[same].min()) if same.any() else global_first
        scores.append(1.0 - float(trust[start - 100: start - 100 + 250, r].mean()))
        labels.append(int(deg[r]))
    return float(auc(np.asarray(scores), np.asarray(labels)))


def evaluate_open_loop(seeds):
    """{(seed, rho): per-draw arrays} from the stored trust files; tau per (model, seed, rho) pooled over the 50 draws."""
    from trustgraph.config import load_config, with_overrides
    from trustgraph.observed import default_observed_path
    from trustgraph.scenario import build_world
    from trustgraph.sealed.ground_truth import load_ground_truth, sealed_path_for

    base = load_config(CONFIG)
    res = {}
    for seed in seeds:
        nb = same_segment_neighbours(build_world(with_overrides(base, seed=seed)).topology)
        for rho in RHOS:
            tr = {}
            gts = {}
            for d in EVAL_DRAWS:
                with np.load(trust_path(seed, rho, d)) as z:
                    tr[d] = {"C": z["C"], "D": z["D"]}
                cfg = with_overrides(base, seed=seed, rho=rho, fraction=FRACTION, draw=d)
                gts[d] = load_ground_truth(sealed_path_for(default_observed_path(CONFIG, cfg)))
            # tau: 1st percentile of the model's trust over all RSU-steps in [100, 400) (index 0..299), pooled over the draws
            tau = {m: float(np.percentile(np.concatenate([tr[d][m][0:300].ravel() for d in EVAL_DRAWS]), 1.0)) for m in ("C", "D")}
            rec = {m: {"lat": [], "det": [], "auc": []} for m in ("C", "D")}
            n_late = []
            for d in EVAL_DRAWS:
                gt = gts[d]
                late = late_onset_rsus(gt, nb)
                n_late.append(len(late))
                on = gt.onset_per_rsu()
                for m in ("C", "D"):
                    if late:
                        lats = [detect_latency(tr[d][m][:, v], tau[m], int(on[v])) for v in late]
                        rec[m]["lat"].append(float(np.mean([SEARCH - 1 if l is None else l for l in lats])))
                        rec[m]["det"].append(float(np.mean([l is not None for l in lats])))
                    else:
                        rec[m]["lat"].append(float("nan"))
                        rec[m]["det"].append(float("nan"))
                    rec[m]["auc"].append(window_auc(tr[d][m], gt))
            res[(seed, rho)] = {"tau": tau, "n_late": n_late, **{m: {k: np.array(v) for k, v in rec[m].items()} for m in rec}}
    return res


def label_latency(mean: float, lo: float, hi: float) -> str:
    """D65 2.5. Interval on D - C in steps; negative = D detects earlier. Order WIN, NEGATIVE, NULL, NOISY."""
    if hi < 0 and mean <= -LAT_FLOOR:
        return "WIN"
    if lo > 0 and mean >= LAT_FLOOR:
        return "NEGATIVE"
    if lo > -LAT_FLOOR and hi < LAT_FLOOR:
        return "NULL"
    return "NOISY"


def cmd_eval(seeds, verdict: bool) -> int:
    res = evaluate_open_loop(seeds)
    t = T_CRIT[len(seeds)]
    L = ["=" * 118, "S13 EXPERIMENT 2: STAGGERED ONSET (S = 300), open-loop frozen C and D, draws 200-249, fraction 0.20, seeds " + str(list(seeds))
         + (" WITH D65 LABELS" if verdict else " (RAW: no labels)"), "=" * 118]
    summary = {}
    for rho in RHOS:
        L.append(f"\nrho {rho:g}  ({'TEST' if rho == 1.0 else 'CONTROL'})")
        L.append("  D42 accounting per seed: draws attempted / aggregated for the latency primary (draws with a late-onset RSU) / late-onset RSUs")
        per = {"C_lat": [], "D_lat": [], "dlat": [], "C_det": [], "D_det": [], "C_auc": [], "D_auc": [], "dauc": []}
        for seed in seeds:
            r = res[(seed, rho)]
            ok = ~np.isnan(r["C"]["lat"])
            L.append(f"    seed {seed}: attempted {len(EVAL_DRAWS)}  failed 0  aggregated {int(ok.sum())}  late-onset RSUs {int(sum(r['n_late']))}  "
                     f"tau C {r['tau']['C']:.4f} D {r['tau']['D']:.4f}")
            per["C_lat"].append(r["C"]["lat"][ok].mean())
            per["D_lat"].append(r["D"]["lat"][ok].mean())
            per["dlat"].append((r["D"]["lat"][ok] - r["C"]["lat"][ok]).mean())
            per["C_det"].append(r["C"]["det"][ok].mean())
            per["D_det"].append(r["D"]["det"][ok].mean())
            per["C_auc"].append(r["C"]["auc"].mean())
            per["D_auc"].append(r["D"]["auc"].mean())
            per["dauc"].append((r["D"]["auc"] - r["C"]["auc"]).mean())
        s = {k: t_interval(v) for k, v in per.items()}
        lab = label_latency(s["dlat"][0], s["dlat"][2], s["dlat"][3])
        summary[rho] = {"per": per, "s": s, "label": lab}
        L.append(f"  detection latency (steps after own onset, late-onset RSUs, censored 399): C {s['C_lat'][0]:.1f} +/- {s['C_lat'][1]:.1f}   "
                 f"D {s['D_lat'][0]:.1f} +/- {s['D_lat'][1]:.1f}   detected fraction C {s['C_det'][0]:.3f} D {s['D_det'][0]:.3f}")
        L.append(f"  PRIMARY  D - C latency: {s['dlat'][0]:+.1f} (sd {s['dlat'][1]:.1f}) 95% [{s['dlat'][2]:+.1f}, {s['dlat'][3]:+.1f}]"
                 + (f"   {lab}" if verdict else ""))
        L.append(f"  per-seed D - C latency: {[round(float(x), 1) for x in per['dlat']]}")
        L.append(f"  (secondary) window AUC: C {s['C_auc'][0]:.4f} D {s['D_auc'][0]:.4f}   D - C {s['dauc'][0]:+.4f} (sd {s['dauc'][1]:.4f}) "
                 f"95% [{s['dauc'][2]:+.4f}, {s['dauc'][3]:+.4f}]")
    diff = [a - b for a, b in zip(summary[1.0]["per"]["dlat"], summary[0.0]["per"]["dlat"])]
    m, sd, lo, hi = t_interval(diff)
    L.append(f"\n(description) (D - C latency at rho 1.0) minus (at rho 0), per-seed paired: {m:+.1f} (sd {sd:.1f}) 95% [{lo:+.1f}, {hi:+.1f}]")
    if verdict:
        a1, a0 = summary[1.0], summary[0.0]
        l1, l0 = a1["label"], a0["label"]
        adv1, adv0 = -a1["s"]["dlat"][0], -a0["s"]["dlat"][0]
        if l1 == "WIN":
            if l0 != "WIN":
                reading = "WIN at rho 1.0 and not at rho 0: structural by the D65 reading"
            elif adv0 >= adv1:
                reading = "WIN at rho 1.0 but the rho 0 advantage is as large: NOT structural"
            else:
                reading = "WIN at rho 1.0 and also at rho 0 with a smaller advantage: both stated"
        else:
            reading = f"rho 1.0 label {l1}: no structural reading"
        L += ["\n" + "=" * 118, f"D65 VERDICT (Experiment 2, {len(seeds)} seeds, t = {t})",
              f"  rho 1.0 (test): {l1}   rho 0 (control): {l0}   -> {reading}"]
    text = "\n".join(L)
    (OUT / ("report-eval-verdict.txt" if verdict else "report-eval-raw.txt")).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


# ------------------------------------------------------------------ closed loop with each model's own tuned weights (D65 2.5 secondary b)

WEIGHTS_FILE = REPO_ROOT / "configs" / "weights_staggered.yaml"
ALPHAS, BETAS, GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0), (0.25, 0.5, 1.0), (0.15, 0.3, 0.6)
TUNE_DRAWS = (0, 1, 2)
CL_VARIANTS = ("A", "C", "D")


def weight_grid(variant: str):
    return list(itertools.product((0.0,) if variant == "A" else ALPHAS, BETAS, GAMMAS))


def _cl_init():
    import torch

    torch.set_num_threads(1)


def _cl_state(seed: int):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.scenario import build_world
    from trustgraph.trace import default_trace_path, load_trace

    key = ("cl", seed)
    if key not in _W:
        base = _W.get("base") or _W.setdefault("base", load_config(CONFIG))
        _W[key] = (build_world(with_overrides(base, seed=seed)), load_trace(default_trace_path(CONFIG, seed)), load_models(base, seed))
    return _W["base"], _W[key]


def _cl_one(task):
    from trustgraph.config import with_overrides
    from trustgraph.offload import OffloadSettings
    from trustgraph.simulator import run_closed_loop

    variant, (alpha, beta, gamma), seed, rho, fraction, draw = task
    base, (world, trace, models) = _cl_state(seed)
    cfg = with_overrides(base, seed=seed, rho=rho, fraction=fraction, draw=draw)
    sc = run_closed_loop(cfg, world, trace, OffloadSettings(variant=variant, alpha=alpha, beta=beta, gamma=gamma, checkpoint_dir=CKPT_DIR),
                         model=models.get(variant))
    tk = sc.observed.tasks
    return task, {"success": float(tk.success_rate(tk.step >= 400)), "success_all": float(tk.success_rate())}


def cmd_tune(write: bool, workers: int) -> int:
    tasks = [(v, w, TUNING_SEED, 1.0, FRACTION, d) for v in CL_VARIANTS for w in weight_grid(v) for d in TUNE_DRAWS]
    print(f"[tune] {len(tasks)} runs on tuning seed {TUNING_SEED}, staggered onset", flush=True)
    res = {}
    with mp.get_context("spawn").Pool(workers, initializer=_cl_init) as pool:
        for task, r in pool.imap_unordered(_cl_one, tasks, chunksize=3):
            res[task] = r
            if len(res) == 1:
                note_first_number("exp2_closed_loop_tuning_first_result")
    chosen, table = {}, {}
    for v in CL_VARIANTS:
        rows = []
        for w in weight_grid(v):
            per = [res[(v, w, TUNING_SEED, 1.0, FRACTION, d)]["success"] for d in TUNE_DRAWS]
            rows.append({"alpha": w[0], "beta": w[1], "gamma": w[2], "objective": float(np.mean(per)), "per_draw": per})
        best = max(range(len(rows)), key=lambda i: (rows[i]["objective"], -i))
        table[v], chosen[v] = rows, rows[best]
        print(f"\n=== {v}: objective = mean post-onset deadline success over draws {TUNE_DRAWS} (staggered onset, seed {TUNING_SEED})")
        print("  alpha  beta  gamma   objective   per-draw")
        for i, r in enumerate(rows):
            print(f"  {r['alpha']:5.2f} {r['beta']:5.2f} {r['gamma']:5.2f}   {r['objective']:.4f}    {[round(x, 4) for x in r['per_draw']]}{'  <== chosen' if i == best else ''}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tuning-grid.json").write_text(json.dumps({"grid": table, "chosen": chosen}, indent=1))
    if write:
        doc = {"comment": "Staggered onset (D65 2.5): tuned on seed 100, rho 1.0, fraction 0.2, draws 0-2 by the D52 procedure on the seed-100 staggered models.",
               "tuning_seed": TUNING_SEED, "weights": {v: {k: chosen[v][k] for k in ("alpha", "beta", "gamma")} for v in CL_VARIANTS}}
        WEIGHTS_FILE.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(f"\nwrote {WEIGHTS_FILE}")
    return 0


def cl_path(variant, seed, rho, draw) -> Path:
    return OUT / "closed" / f"{variant}-seed{seed}-rho{rho:g}-draw{draw}.json"


def cmd_closed(seeds, workers: int) -> int:
    w = yaml.safe_load(WEIGHTS_FILE.read_text())["weights"]
    todo = [(v, (w[v]["alpha"], w[v]["beta"], w[v]["gamma"]), s, rho, FRACTION, d) for s in seeds for rho in RHOS for d in EVAL_DRAWS for v in CL_VARIANTS
            if not cl_path(v, s, rho, d).exists()]
    print(f"[closed] {len(todo)} runs to do", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_cl_init) as pool:
        for task, r in pool.imap_unordered(_cl_one, todo, chunksize=2):
            v, _, s, rho, _, d = task
            p = cl_path(v, s, rho, d)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(r))
            n += 1
            if n == 1:
                note_first_number("exp2_closed_loop_evaluation_first_result")
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    return 0


def cmd_closed_report(seeds) -> int:
    w = yaml.safe_load(WEIGHTS_FILE.read_text())["weights"]
    L = ["=" * 110, f"S13 EXPERIMENT 2 SECONDARY (b): closed-loop post-onset success, staggered onset, each model's own weights {w}, draws 200-249, seeds {list(seeds)}", "=" * 110]
    for rho in RHOS:
        succ = {v: np.array([[json.loads(cl_path(v, s, rho, d).read_text())["success"] for d in EVAL_DRAWS] for s in seeds]) for v in CL_VARIANTS}
        L.append(f"\nrho {rho:g}  attempted {len(seeds) * len(EVAL_DRAWS)} failed 0 aggregated {succ['A'].size}")
        for v in CL_VARIANTS:
            sm = succ[v].mean(axis=1)
            L.append(f"  {v}: {sm.mean():.4f} +/- {sm.std(ddof=1):.4f}")
        for a, b in (("D", "C"), ("C", "A"), ("D", "A")):
            m, sd, lo, hi = t_interval((succ[a] - succ[b]).mean(axis=1))
            L.append(f"  {a} - {b}: {m:+.4f} (sd {sd:.4f}) 95% [{lo:+.4f}, {hi:+.4f}]   (descriptive, no label)")
    text = "\n".join(L)
    (OUT / "report-closed.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("gen", "mech", "trust", "eval", "tune", "closed", "report"))
    ap.add_argument("which", nargs="?")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    if a.cmd == "gen":
        return cmd_gen(a.which, a.seeds, a.workers)
    if a.cmd == "mech":
        return cmd_mech()
    if a.cmd == "trust":
        return cmd_trust(a.seeds, a.workers)
    if a.cmd == "eval":
        return cmd_eval(a.seeds, a.verdict)
    if a.cmd == "tune":
        return cmd_tune(a.write, a.workers)
    if a.cmd == "closed":
        return cmd_closed(a.seeds, a.workers)
    if a.cmd == "report":
        return cmd_closed_report(a.seeds)
    raise SystemExit(f"{a.cmd}: not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())

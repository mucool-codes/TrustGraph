"""S24 phase 2: D against C in the sparse-evidence regime, trained in-regime (DECISIONS.md D90, under D89).

    python scripts/s24_run.py gen   --set train|eval|harness   # sparse_r02 scenarios: S3 composition (< 100), eval draws 710-759, harness 900
    python scripts/s24_run.py train [--seeds ...]              # C, D, D2 per seed on the sparse_r02 training draws -> checkpoints/s24
    python scripts/s24_run.py check                            # harness check on draw 900 (not a result)
    python scripts/s24_run.py trust                            # open-loop trust of C, D, D2 and M (success_ewma), stride 1, steps 100-1199
    python scripts/s24_run.py eval  [--verdict]                # P1 trust AUC [450, 650), P2 latency + false alarms (D76 rule)
    python scripts/s24_run.py closed                           # closed loop A, M, C, D, D2, B_oracle
    python scripts/s24_run.py report                           # closed-loop success (descriptive) and n_av / n_win / never-avoided

EVALUATION-SIDE: reads the sealed ground truth to score; allowlisted in tests/test_sealing.py. The training path reads
observable scenario files only (train.py / dataset.py cannot import ground truth). The first result each command writes
is dated in results/s24/first_number.json.
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

import s13_stag as S13  # noqa: E402  (detect_latency, t_interval, SEARCH: the D65 rule, unchanged)
import s16_confirm as S16  # noqa: E402  (pre_onset_tau, false_alarm: the D76 rule, unchanged)

CONFIG = "configs/sparse_r02.yaml"  # D89 section 4: the gated regime
OUT = REPO_ROOT / "results" / "s24"
CKPT = REPO_ROOT / "checkpoints" / "s24"
SEEDS = (1, 2, 3, 4, 5)
RHO, FRACTION = 1.0, 0.20
EVAL_DRAWS = tuple(range(710, 760))  # D90: fresh; 700-709 were the gate's
HARNESS_DRAW = 900
# the S3 composition (scripts/s3_make_scenarios.sh, D43): all draws < 100
TRAIN_RHO, TRAIN_FRAC, TRAIN_DRAWS = (0.0, 0.25, 0.5, 0.75, 1.0), (0.1, 0.2, 0.3), (0, 1)
VAL_RHO, VAL_DRAW = (0.0, 0.5, 1.0), 40
FIRST, LAST = S16.FIRST, S16.LAST  # stored steps [100, 1200)
AUC_WINDOW = (450, 650)  # D43 primary window
AUC_FLOOR, AUC_NULL = 0.02, 0.05  # D43
LAT_FLOOR, FA_BOUND = 20.0, 0.05  # D76 / D81 section 7
MODELS = ("C", "D", "D2")
PARAMS = {"C": 1313, "D": 1614, "D2": 1570}  # the main-study counts (F14, F42)
OPEN_NAMES = ("C", "D", "D2", "M")
CL_NAMES = ("A", "M", "C", "D", "D2", "B_oracle")
CL_VARIANT = {"A": "A", "M": "H", "C": "C", "D": "D", "D2": "D2", "B_oracle": "B_oracle"}
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


def _single_thread():
    import torch

    torch.set_num_threads(1)


# ------------------------------------------------------------------ labels (D90 sections 3-4)

def label_auc(mean: float, lo: float, hi: float) -> str:
    """D43 form. D minus C trust AUC; positive = D separates better. Order WIN, NEGATIVE, NULL, NOISY."""
    if lo > 0 and mean >= AUC_FLOOR:
        return "WIN"
    if hi < 0 and mean <= -AUC_FLOOR:
        return "NEGATIVE"
    if hi < AUC_NULL:
        return "NULL"
    return "NOISY"


def label_latency(mean: float, lo: float, hi: float) -> str:
    """D81 section 7 form. D minus C steps; negative = D earlier. Order WIN, NEGATIVE, NULL, NOISY."""
    if hi < 0 and mean <= -LAT_FLOOR:
        return "WIN"
    if lo > 0 and mean >= LAT_FLOOR:
        return "NEGATIVE"
    if lo > -LAT_FLOOR and hi < LAT_FLOOR:
        return "NULL"
    return "NOISY"


def label_fa(hi: float) -> str:
    return "ACCEPTABLE" if hi <= FA_BOUND else "NOT ACCEPTABLE"


def overall(auc_label: str, lat_label: str, fa_label: str) -> str:
    """D90 section 5."""
    if auc_label == "WIN" and lat_label != "NEGATIVE":
        return "SUPPORTED"
    if auc_label == "NEGATIVE":
        return "CONTRADICTED"
    if auc_label == "NULL":
        return "NOT SEPARATED"
    return "INCONCLUSIVE"


# ------------------------------------------------------------------ scenarios

def scenario_jobs(which: str, seeds) -> list[tuple]:
    jobs = []
    for s in seeds:
        if which == "train":
            jobs += [(s, r, f, d) for r in TRAIN_RHO for f in TRAIN_FRAC for d in TRAIN_DRAWS]
            jobs += [(s, r, f, VAL_DRAW) for r in VAL_RHO for f in TRAIN_FRAC]
        elif which == "eval":
            jobs += [(s, RHO, FRACTION, d) for d in EVAL_DRAWS]
        elif which == "harness":
            jobs += [(s, RHO, FRACTION, HARNESS_DRAW)]
    return jobs


def _base():
    from trustgraph.config import load_config

    return load_config(CONFIG)


def _world(seed: int):
    from trustgraph.config import with_overrides
    from trustgraph.scenario import build_world
    from trustgraph.trace import default_trace_path, load_trace

    key = ("world", seed)
    if key not in _W:
        base = _base()
        _W[key] = (base, build_world(with_overrides(base, seed=seed)), load_trace(default_trace_path(CONFIG, seed)))
    return _W[key]


def obs_path(seed, rho, frac, draw) -> Path:
    from trustgraph.config import with_overrides
    from trustgraph.observed import default_observed_path

    return default_observed_path(CONFIG, with_overrides(_base(), seed=seed, rho=rho, fraction=frac, draw=draw))


def _gen_one(job):
    from trustgraph.config import with_overrides
    from trustgraph.sealed.ground_truth import sealed_path_for
    from trustgraph.simulator import generate_scenario

    seed, rho, frac, draw = job
    base, world, trace = _world(seed)
    out = obs_path(seed, rho, frac, draw)
    if out.exists() and sealed_path_for(out).exists():
        return job, "exists"
    sc = generate_scenario(with_overrides(base, seed=seed, rho=rho, fraction=frac, draw=draw), world, trace)
    out.parent.mkdir(parents=True, exist_ok=True)
    sc.observed.save(out)
    sc.ground_truth.save(sealed_path_for(out))
    return job, "written"


def cmd_gen(which: str, seeds, workers: int) -> int:
    jobs = scenario_jobs(which, seeds)
    print(f"[gen {which}] {len(jobs)} scenarios", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for _ in pool.imap_unordered(_gen_one, jobs, chunksize=2):
            n += 1
            if n % 25 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ training (D90 section 2)

def _train_seed(job):
    seed, threads = job
    import torch

    torch.set_num_threads(threads)
    from trustgraph.config import SeedChain, with_overrides
    from trustgraph.dataset import build_graphset
    from trustgraph.model import build_trust_head, num_parameters
    from trustgraph.train import TrainSettings, load_checkpoint, save_checkpoint, train

    settings = TrainSettings()  # the D45 settings, unchanged
    base = _base()
    t0 = time.time()
    tr, va = [], []
    for r in TRAIN_RHO:
        for f in TRAIN_FRAC:
            for d in TRAIN_DRAWS:
                tr += build_graphset(with_overrides(base, seed=seed, rho=r, fraction=f, draw=d), CONFIG, 4).graphs
    for r in VAL_RHO:
        for f in TRAIN_FRAC:
            va += build_graphset(with_overrides(base, seed=seed, rho=r, fraction=f, draw=VAL_DRAW), CONFIG, 4).graphs
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / f"train-seed{seed}.log", "w", encoding="utf-8")

    def say(s):
        print(s, file=log, flush=True)

    say(f"seed {seed}: train graphs {len(tr)} val graphs {len(va)} ({time.time() - t0:.0f}s) settings {settings}")
    rep = {"seed": seed, "settings": settings.__dict__, "train_graphs": len(tr), "val_graphs": len(va), "variants": {}}
    ref_dir = {"C": REPO_ROOT / "checkpoints", "D": REPO_ROOT / "checkpoints", "D2": REPO_ROOT / "checkpoints" / "s14" / "shared"}
    for variant in MODELS:
        out = CKPT / f"s3-seed{seed}-{variant}.pt"
        seeds = SeedChain(seed)
        model = build_trust_head(base.model, seeds.torch_seed("model_init"), base.device, variant)
        n_par = num_parameters(model)
        assert n_par == PARAMS[variant], "D90 section 2: identical parameter counts to the main study"
        ref_ck = ref_dir[variant] / f"s3-seed{seed}-{variant}.pt"
        if ref_ck.exists():  # and against the main-study checkpoint itself where it is on disk
            ref = build_trust_head(base.model, 0, base.device, variant)
            load_checkpoint(ref_ck, ref)
            assert n_par == num_parameters(ref)
        t1 = time.time()
        res = train(model, tr, va, settings, seeds, log=say)
        save_checkpoint(out, model, variant, seed, settings, res)
        rep["variants"][variant] = {"parameters": n_par, "best_epoch": res.best_epoch, "stopped_epoch": res.stopped_epoch,
                                    "converged_by": res.converged_by, "initial_val_bce": res.initial_val_bce,
                                    "best_val_bce": min(h["val_bce"] for h in res.history), "seconds": round(time.time() - t1)}
        say(f"{variant}: {rep['variants'][variant]}")
    (OUT / f"train-seed{seed}.json").write_text(json.dumps(rep, indent=1))
    return seed, rep


def cmd_train(seeds, workers: int, threads: int) -> int:
    CKPT.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(min(workers, len(seeds))) as pool:
        for seed, rep in pool.imap_unordered(_train_seed, [(s, threads) for s in seeds]):
            print(f"seed {seed}: " + "; ".join(
                f"{v} params {r['parameters']} best {r['best_epoch']} stopped {r['stopped_epoch']} ({r['converged_by']}) "
                f"val BCE {r['initial_val_bce']:.4f} -> {r['best_val_bce']:.4f} {r['seconds']}s" for v, r in rep["variants"].items()),
                flush=True)
    return 0


def _models(seed: int):
    from trustgraph.model import build_trust_head
    from trustgraph.train import load_checkpoint

    key = ("models", seed)
    if key not in _W:
        base = _base()
        ms = {}
        for v in MODELS:
            m = build_trust_head(base.model, 0, base.device, v)
            blob = load_checkpoint(CKPT / f"s3-seed{seed}-{v}.pt", m)
            assert blob["variant"] == v and blob["seed"] == seed
            m.eval()
            ms[v] = m
        _W[key] = ms
    return _W[key]


# ------------------------------------------------------------------ open-loop trust

def trust_path(seed, draw) -> Path:
    return OUT / "trust" / f"seed{seed}-draw{draw}.npz"


def compute_trust(seed, draw) -> dict:
    from trustgraph.config import with_overrides
    from trustgraph.dataset import graphset_from
    from trustgraph.features import RSU_COL
    from trustgraph.observed import load_observed
    from trustgraph.train import predict_trust

    base, world, trace = _world(seed)
    cfg = with_overrides(base, seed=seed, rho=RHO, fraction=FRACTION, draw=draw)
    observed = load_observed(obs_path(seed, RHO, FRACTION, draw))
    gs = graphset_from(cfg, world, trace, observed, stride=1, step_range=(FIRST, LAST))
    assert gs.steps[0] == FIRST and gs.steps[-1] == LAST - 1
    out = {n: predict_trust(m, gs).astype(np.float32) for n, m in _models(seed).items()}
    out["M"] = observed.rsu_features[FIRST:LAST, :, RSU_COL["success_ewma"]].astype(np.float32)
    return out


def _trust_one(job):
    seed, draw = job
    out = trust_path(seed, draw)
    if out.exists():
        return job, "cached"
    tr = compute_trust(seed, draw)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **tr)
    tmp.replace(out)
    return job, "ok"


def cmd_trust(seeds, workers: int) -> int:
    jobs = [(s, d) for s in seeds for d in EVAL_DRAWS]
    print(f"[trust] {len(jobs)} scenarios", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for _ in pool.imap_unordered(_trust_one, jobs, chunksize=2):
            n += 1
            if n == 1:
                note_first_number("evaluation_first_trust")
            if n % 25 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ open-loop evaluation (D90 sections 3-4)

def evaluate(seeds) -> dict:
    from trustgraph.diagnostic import auc
    from trustgraph.sealed.ground_truth import load_ground_truth, sealed_path_for

    res = {}
    lo, hi = AUC_WINDOW[0] - FIRST, AUC_WINDOW[1] - FIRST
    for seed in seeds:
        tr, gts = {}, {}
        for d in EVAL_DRAWS:
            with np.load(trust_path(seed, d)) as z:
                tr[d] = {m: z[m] for m in OPEN_NAMES}
            gts[d] = load_ground_truth(sealed_path_for(obs_path(seed, RHO, FRACTION, d)))
        tau = {m: S16.pre_onset_tau([tr[d][m] for d in EVAL_DRAWS]) for m in OPEN_NAMES}
        rec = {m: {k: [] for k in ("auc", "lat", "det", "fa")} for m in OPEN_NAMES}
        for d in EVAL_DRAWS:
            gt = gts[d]
            deg, healthy = np.flatnonzero(gt.degraded), np.flatnonzero(~gt.degraded)
            on = gt.onset_per_rsu()
            labels = gt.degraded.astype(int)
            for m in OPEN_NAMES:
                t = tr[d][m]
                rec[m]["auc"].append(auc(1.0 - t[lo:hi].mean(axis=0), labels))
                lats = [S13.detect_latency(t[:, v], tau[m], int(on[v])) for v in deg]
                rec[m]["lat"].append(float(np.mean([S13.SEARCH - 1 if x is None else x for x in lats])))
                rec[m]["det"].append(float(np.mean([x is not None for x in lats])))
                rec[m]["fa"].append(float(np.mean([S16.false_alarm(t[:, v], tau[m])[0] for v in healthy])))
        res[seed] = {"tau": tau, **{m: {k: np.array(v, float) for k, v in rec[m].items()} for m in OPEN_NAMES}}
    return res


def interval(v):
    return S13.t_interval(np.asarray(v, float))


def cmd_eval(seeds, verdict: bool) -> int:
    res = evaluate(seeds)
    note_first_number("evaluation_first_label" if verdict else "evaluation_raw_table")
    L = ["=" * 116, f"S24 OPEN LOOP: in-regime C, D, D2 and M (success_ewma), {CONFIG}, rho {RHO}, fraction {FRACTION}, "
         f"draws {EVAL_DRAWS[0]}-{EVAL_DRAWS[-1]}, seeds {list(seeds)}", "=" * 116]
    L.append("per model, mean over seeds of per-seed means (95% t over seeds):")
    for m in OPEN_NAMES:
        a, lt, dt_, fa = (interval([res[s][m][k].mean() for s in seeds]) for k in ("auc", "lat", "det", "fa"))
        L.append(f"  {m:<3} AUC[450,650) {a[0]:.4f} [{a[2]:.4f}, {a[3]:.4f}]   latency {lt[0]:6.1f} [{lt[2]:6.1f}, {lt[3]:6.1f}]   "
                 f"detected {dt_[0]:.3f}   healthy false-alarm rate {fa[0]:.3f} [{fa[2]:.3f}, {fa[3]:.3f}]")
    L.append("  tau per seed: " + "; ".join(f"s{s} " + " ".join(f"{m} {res[s]['tau'][m]:.4f}" for m in OPEN_NAMES) for s in seeds))
    L.append("paired differences (per draw, per-seed mean, 95% t over seeds):")
    out = {}
    for a, b in (("D", "C"), ("D2", "C"), ("M", "C"), ("D", "D2")):
        row = []
        for k in ("auc", "lat", "fa"):
            per = [float(np.nanmean(res[s][a][k] - res[s][b][k])) for s in seeds]
            m, sd, lo, hi = interval(per)
            out[(a, b, k)] = (m, lo, hi, per)
            row.append(f"{k} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" if k != "lat" else f"lat {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]")
        L.append(f"  {a} - {b}: " + "   ".join(row))
        L.append(f"      per-seed auc {' '.join(f'{x:+.4f}' for x in out[(a, b, 'auc')][3])}   "
                 f"lat {' '.join(f'{x:+.1f}' for x in out[(a, b, 'lat')][3])}")
    if verdict:
        L.append("-" * 116)
        L.append("VERDICT (D90 sections 3-5): primary P1 = D - C trust AUC; P2 = D - C latency with the false-alarm condition")
        for a in ("D", "D2"):
            m, lo, hi, _ = out[(a, "C", "auc")]
            p1 = label_auc(m, lo, hi)
            lm, llo, lhi, _ = out[(a, "C", "lat")]
            p2 = label_latency(lm, llo, lhi)
            fm, flo, fhi, _ = out[(a, "C", "fa")]
            fl = label_fa(fhi)
            tag = "PRIMARY" if a == "D" else "secondary (labels printed as description)"
            L.append(f"  {a} vs C [{tag}]: P1 AUC {p1}; P2 latency {p2}, false alarms {fl}  ->  {overall(p1, p2, fl)}")
    text = "\n".join(L)
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("report-open-verdict.txt" if verdict else "report-open-raw.txt")).write_text(text + "\n", encoding="utf-8")
    return 0


# ------------------------------------------------------------------ closed loop (D90 section 6, descriptive)

def load_weights() -> dict:
    w = yaml.safe_load((REPO_ROOT / "configs" / "weights.yaml").read_text())["weights"]
    h = yaml.safe_load((REPO_ROOT / "configs" / "weights_h.yaml").read_text())["weights"]["H"]
    d2 = yaml.safe_load((REPO_ROOT / "configs" / "weights_d2.yaml").read_text())["weights"]["shared"]["D2"]
    t = lambda d: (d["alpha"], d["beta"], d["gamma"])  # noqa: E731
    return {"A": t(w["A"]), "C": t(w["C"]), "D": t(w["D"]), "B_oracle": t(w["B_oracle"]), "M": t(h), "D2": t(d2)}


def cl_path(name, seed, draw) -> Path:
    return OUT / "closed" / f"{name}-seed{seed}-draw{draw}.json"


def _cl_job(job):
    from trustgraph import sweep_metrics as SM
    from trustgraph.config import with_overrides
    from trustgraph.offload import build_offload_settings
    from trustgraph.simulator import run_closed_loop

    name, seed, draw = job
    base, world, trace = _world(seed)
    cfg = with_overrides(base, seed=seed, rho=RHO, fraction=FRACTION, draw=draw)
    settings = build_offload_settings(cfg, variant=CL_VARIANT[name], weights=load_weights()[name], record_trust=False)
    model = _models(seed)[name] if name in MODELS else None
    sc = run_closed_loop(cfg, world, trace, settings, model=model)
    tasks, gt, n_steps = sc.observed.tasks, sc.ground_truth, sc.observed.num_steps
    total, deg = SM.dispatch_series(tasks.step, tasks.rsu, gt.degraded, n_steps)
    av = SM.avoidance(total, deg, int(gt.onset_step))
    post = tasks.step >= int(gt.onset_step)
    return job, {"success_post": float(tasks.success_rate(post)), "success_all": float(tasks.success_rate()),
                 "n_av": av["n_av"], "n_win": av["n_win"], "never_avoided": bool(av["never_avoided"])}


def cmd_closed(seeds, workers: int) -> int:
    todo = [(v, s, d) for s in seeds for d in EVAL_DRAWS for v in CL_NAMES if not cl_path(v, s, d).exists()]
    print(f"[closed] {len(todo)} runs to do", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for job, r in pool.imap_unordered(_cl_job, todo, chunksize=2):
            p = cl_path(*job)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(r))
            n += 1
            if n == 1:
                note_first_number("closed_loop_first_result")
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    return 0


def cmd_report(seeds) -> int:
    get = lambda v, s, k: np.array([json.loads(cl_path(v, s, d).read_text())[k] for d in EVAL_DRAWS], float)  # noqa: E731
    L = ["=" * 100, f"S24 CLOSED LOOP (descriptive, D90 section 6): {CONFIG}, draws {EVAL_DRAWS[0]}-{EVAL_DRAWS[-1]}, seeds {list(seeds)}",
         "weights: the existing tuned files, unchanged (no tuning in this regime)", "=" * 100]
    for v in CL_NAMES:
        sp = interval([get(v, s, "success_post").mean() for s in seeds])
        nav = np.mean([get(v, s, "n_av").mean() for s in seeds])
        nwin = np.mean([get(v, s, "n_win").mean() for s in seeds])
        nev = np.mean([get(v, s, "never_avoided").mean() for s in seeds])
        L.append(f"  {v:<9} post-onset success {sp[0]:.4f} [{sp[2]:.4f}, {sp[3]:.4f}]   n_av {nav:7.1f}  n_win {nwin:7.1f}  never-avoided {nev:.3f}")
    for a, b in (("D", "C"), ("D2", "C"), ("M", "C"), ("C", "A"), ("D", "A")):
        m, sd, lo, hi = interval([(get(a, s, "success_post") - get(b, s, "success_post")).mean() for s in seeds])
        L.append(f"  {a} - {b} post-onset success {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    text = "\n".join(L)
    print(text)
    (OUT / "report-closed.txt").write_text(text + "\n", encoding="utf-8")
    return 0


# ------------------------------------------------------------------ harness check (draw 900; not a result)

def cmd_check() -> int:
    from trustgraph.observed import load_observed
    from trustgraph.simulator import generate_scenario
    from trustgraph.config import with_overrides

    _single_thread()
    _gen_one((1, RHO, FRACTION, HARNESS_DRAW))
    a = load_observed(obs_path(1, RHO, FRACTION, HARNESS_DRAW))
    base, world, trace = _world(1)
    b = generate_scenario(with_overrides(base, seed=1, rho=RHO, fraction=FRACTION, draw=HARNESS_DRAW), world, trace).observed
    same = all(np.array_equal(getattr(a.tasks, f), getattr(b.tasks, f)) for f in ("step", "rsu", "met_deadline"))
    print(f"  regenerated harness scenario equals the stored one: {same}")
    t1 = compute_trust(1, HARNESS_DRAW)
    _W.clear()
    t2 = compute_trust(1, HARNESS_DRAW)
    st = all(np.array_equal(t1[k], t2[k]) for k in t1)
    print(f"  open-loop trust (C, D, D2, M) repeats exactly: {st}")
    _, r1 = _cl_job(("D", 1, HARNESS_DRAW))
    _W.clear()
    _, r2 = _cl_job(("D", 1, HARNESS_DRAW))
    print(f"  closed-loop D run repeats exactly: {r1 == r2}")
    return 0 if (same and st and r1 == r2) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("gen", "train", "check", "trust", "eval", "closed", "report"))
    ap.add_argument("--set", choices=("train", "eval", "harness"), default="eval")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per training process")
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    if a.cmd == "gen":
        return cmd_gen(a.set, a.seeds, a.workers)
    if a.cmd == "train":
        return cmd_train(a.seeds, a.workers, a.threads)
    if a.cmd == "check":
        return cmd_check()
    if a.cmd == "trust":
        return cmd_trust(a.seeds, a.workers)
    if a.cmd == "eval":
        return cmd_eval(a.seeds, a.verdict)
    if a.cmd == "closed":
        return cmd_closed(a.seeds, a.workers)
    return cmd_report(a.seeds)


if __name__ == "__main__":
    raise SystemExit(main())

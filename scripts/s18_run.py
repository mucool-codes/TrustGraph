"""S18: the registered S17 experiment, run (DECISIONS.md D81, amended by D83; both frozen).

    python scripts/s18_run.py gen   --set eval|train      # scenarios: eval draws 400-449, training draws 600-603 and validation draw 640
    python scripts/s18_run.py train [--seeds ...]         # SECONDARY ARM: C and D retrained per seed on draws 600-640 (checkpoints/s18)
    python scripts/s18_run.py check                       # harness check on draw 900 (not a result)
    python scripts/s18_run.py trust  --arm primary|secondary
    python scripts/s18_run.py closed --arm primary|secondary
    python scripts/s18_run.py eval   --arm primary|secondary [--verdict]   # open loop: latency, false alarms, AUC
    python scripts/s18_run.py report --arm primary|secondary [--verdict]   # closed loop: success, n_av / n_win / never-avoided

PRIMARY ARM = the frozen checkpoints of D81 section 4 (C, D: checkpoints/s3-seed{n}; D2: checkpoints/s14/shared; weights unchanged);
it alone decides Sa, Sb and Sc. SECONDARY ARM (D83) = C and D retrained on the spread/restart training draws, same objective,
settings and parameter counts, same weights as the primary C and D (no tuning); descriptive only.
EVALUATION-SIDE: reads the sealed ground truth to score; allowlisted in tests/test_sealing.py. The training path itself reads
observable scenario files only. The first result each command writes is dated in results/s18/first_number.json.
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

import s13_stag as S13  # noqa: E402  (detect_latency, t_interval: the D65/D66 rule, unchanged)
import s16_confirm as S16  # noqa: E402  (pre_onset_tau, false_alarm: the D76 rule, unchanged)

OUT = REPO_ROOT / "results" / "s18"
SEEDS = (1, 2, 3, 4, 5)
EVAL_DRAWS = tuple(range(400, 450))
TRAIN_DRAWS = (600, 601, 602, 603)
VAL_DRAW = 640
HARNESS_DRAW = 900
FIRST, LAST = S16.FIRST, S16.LAST  # stored steps [100, 1200)
WINDOW_N = 100  # D84: the "first N steps" of the trust AUC
LAT_FLOOR = 20.0  # D81 section 7
FA_BOUND = 0.05  # D76 / D81 section 7
SUCC_FLOOR, SUCC_BOUND = 0.01, 0.025  # D81 section 9 (the D56 H3 floor and bound)

# condition -> (config, mode, parameter)
CONDS = {
    "L0": ("configs/spread_L0.yaml", "spread", 0), "L20": ("configs/spread_L20.yaml", "spread", 20),
    "L40": ("configs/spread_L40.yaml", "spread", 40), "L80": ("configs/spread_L80.yaml", "spread", 80),
    "f10": ("configs/restart_f10.yaml", "restart", 0.10), "f20": ("configs/restart_f20.yaml", "restart", 0.20),
    "f40": ("configs/restart_f40.yaml", "restart", 0.40),
}
SPREAD = ("L0", "L20", "L40", "L80")
RESTART = ("f10", "f20", "f40")
# trust model name -> (architecture, checkpoint dir); M (the success_ewma monitor) has no model
ARMS = {
    "primary": {"C": ("C", "checkpoints"), "D": ("D", "checkpoints"), "D2": ("D2", "checkpoints/s14/shared")},
    "secondary": {"C_r": ("C", "checkpoints/s18"), "D_r": ("D", "checkpoints/s18")},
}
OPEN_NAMES = {"primary": ("C", "D", "D2", "M"), "secondary": ("C_r", "D_r")}
# closed-loop variants per arm; name -> offload variant
CL_VARIANT = {"A": "A", "M": "H", "C": "C", "D": "D", "D2": "D2", "B_oracle": "B_oracle", "C_r": "C", "D_r": "D"}
CL_NAMES = {"primary": ("A", "M", "C", "D", "D2", "B_oracle"), "secondary": ("C_r", "D_r")}
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


# ------------------------------------------------------------------ pure functions (tested in tests/test_s18.py)

def label_latency(mean: float, lo: float, hi: float) -> str:
    """D81 section 7. Interval on D - C in steps; negative = D earlier. Order WIN, NEGATIVE, NULL, NOISY."""
    if hi < 0 and mean <= -LAT_FLOOR:
        return "WIN"
    if lo > 0 and mean >= LAT_FLOOR:
        return "NEGATIVE"
    if lo > -LAT_FLOOR and hi < LAT_FLOOR:
        return "NULL"
    return "NOISY"


def label_fa(hi: float) -> str:
    """D81 section 7 / D76: ACCEPTABLE iff hi <= 0.05 (decisive)."""
    return "ACCEPTABLE" if hi <= FA_BOUND else "NOT ACCEPTABLE"


def fa_strict(hi: float) -> bool:
    """D83 item 3: the strict reading (hi <= 0), printed beside the decisive one."""
    return bool(hi <= 0.0)


def label_success(mean: float, lo: float, hi: float) -> str:
    """D81 section 9 (the D56 H3 floor 0.01 and bound 0.025). Order NEGATIVE, WIN, NULL, NOISY."""
    from trustgraph.sweep_metrics import label

    return label(mean, lo, hi, SUCC_FLOOR, SUCC_BOUND)


def sa_holds_at(lat_label: str, fa_label: str) -> bool:
    return lat_label == "WIN" and fa_label == "ACCEPTABLE"


def sa_overall(per_l: dict) -> str:
    """D81 section 7: Sa holds iff it holds at both L = 40 and L = 80; otherwise NOT SUPPORTED."""
    return "SUPPORTED" if all(sa_holds_at(*per_l[l]) for l in ("L40", "L80")) else "NOT SUPPORTED"


def sc_overall(l0_label: str) -> str:
    """D81 section 8: the control holds iff the L = 0 latency label is NULL."""
    return {"NULL": "HOLDS", "WIN": "VIOLATED (control separated: D earlier)", "NEGATIVE": "VIOLATED (control separated: D later)"}.get(l0_label, "INCONCLUSIVE (NOISY)")


def sb_overall(labels: list[str]) -> str:
    """D81 section 9."""
    win, neg = labels.count("WIN"), labels.count("NEGATIVE")
    if win >= 2 and neg >= 2:
        return "MIXED"
    if win >= 2 and neg < 2:
        return "SUPPORTED"
    if neg >= 2 and win < 2:
        return "CONTRADICTED"
    return "NOT SEPARATED"


def window_auc(trust: np.ndarray, pos_ids, neg_ids, start_of, n: int = WINDOW_N) -> float:
    """D84: AUC (positives = the faulty RSUs) of 1 - mean trust over [start, start + n); one AUC per positive RSU against the
    negatives over the same window, then the mean over the positives. `trust` is (1100, rsus), row 0 = step 100."""
    from trustgraph.diagnostic import auc

    neg_ids = np.asarray(list(neg_ids), int)
    vals = []
    for v in pos_ids:
        s = int(start_of[v]) - FIRST
        score = 1.0 - trust[s:s + n].mean(axis=0)
        vals.append(auc(np.concatenate([[score[v]], score[neg_ids]]), np.concatenate([[1], np.zeros(neg_ids.size, int)])))
    return float(np.nanmean(vals)) if vals else float("nan")


def populations(gt, mode: str) -> dict:
    """The sealed RSU sets of a scenario: `pop` (latency population), `healthy`, `restarted_healthy`, `start` (per-RSU onset)."""
    deg = np.flatnonzero(gt.degraded)
    on = gt.onset_per_rsu()
    if mode == "spread":
        pop = deg[gt.spread_hop[deg] >= 1]
        rh = np.array([], int)
    else:
        pop = deg
        rh = np.flatnonzero((gt.restart_steps >= 0) & ~gt.degraded)
    return {"pop": pop, "healthy": np.flatnonzero(~gt.degraded), "restarted_healthy": rh, "start": on, "faulty": deg}


# ------------------------------------------------------------------ paths and world state

def obs_path(cond: str, seed: int, draw: int):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.observed import default_observed_path

    cfgp = CONDS[cond][0]
    return default_observed_path(cfgp, with_overrides(load_config(cfgp), seed=seed, draw=draw))


def _cond_state(cond: str, seed: int, arm: str | None):
    """(base cfg, world, trace, models) cached per process."""
    from trustgraph.config import load_config, with_overrides
    from trustgraph.model import build_trust_head
    from trustgraph.scenario import build_world
    from trustgraph.trace import default_trace_path, load_trace
    from trustgraph.train import load_checkpoint

    key = (cond, seed, arm)
    if key not in _W:
        cfgp = CONDS[cond][0]
        base = load_config(cfgp)
        models = {}
        for name, (variant, d) in (ARMS[arm].items() if arm else ()):
            m = build_trust_head(base.model, 0, base.device, variant)
            blob = load_checkpoint(REPO_ROOT / d / f"s3-seed{seed}-{variant}.pt", m)
            assert blob["variant"] == variant and blob["seed"] == seed
            m.eval()
            models[name] = m
        _W[key] = (base, build_world(with_overrides(base, seed=seed)), load_trace(default_trace_path(cfgp, seed)), models)
    return _W[key]


def _single_thread():
    import torch

    torch.set_num_threads(1)


# ------------------------------------------------------------------ scenario generation (observable + SEALED)

def _gen_one(job):
    from trustgraph.config import with_overrides
    from trustgraph.observed import default_observed_path
    from trustgraph.sealed.ground_truth import sealed_path_for
    from trustgraph.simulator import generate_scenario

    cond, seed, draw = job
    base, world, trace, _ = _cond_state(cond, seed, None)
    cfg = with_overrides(base, seed=seed, draw=draw)
    out = default_observed_path(CONDS[cond][0], cfg)
    if out.exists() and sealed_path_for(out).exists():
        return job, "exists"
    sc = generate_scenario(cfg, world, trace)
    sc.observed.save(out)
    sc.ground_truth.save(sealed_path_for(out))
    return job, "written"


def cmd_gen(which: str, seeds, workers: int) -> int:
    draws = EVAL_DRAWS if which == "eval" else (*TRAIN_DRAWS, VAL_DRAW)
    jobs = [(c, s, d) for s in seeds for c in CONDS for d in draws]
    print(f"[gen {which}] {len(jobs)} scenarios", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for job, st in pool.imap_unordered(_gen_one, jobs, chunksize=4):
            n += 1
            if n % 200 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ SECONDARY ARM: training (D83 item 1)

def _train_seed(seed: int):
    import torch

    torch.set_num_threads(3)
    from trustgraph.config import SeedChain, load_config, with_overrides
    from trustgraph.dataset import build_graphset
    from trustgraph.model import build_trust_head, num_parameters
    from trustgraph.train import TrainSettings, load_checkpoint, save_checkpoint, train

    settings = TrainSettings()  # the D45 settings, unchanged
    t0 = time.time()
    tr, va = [], []
    for cond, (cfgp, _, _) in CONDS.items():
        base = load_config(cfgp)
        for d in TRAIN_DRAWS:
            tr += build_graphset(with_overrides(base, seed=seed, draw=d), cfgp, 4).graphs
        va += build_graphset(with_overrides(base, seed=seed, draw=VAL_DRAW), cfgp, 4).graphs
    base = load_config(CONDS["L80"][0])
    log = open(OUT / f"train-seed{seed}.log", "w", encoding="utf-8")

    def say(s):
        print(s, file=log, flush=True)

    say(f"seed {seed}: train graphs {len(tr)} val graphs {len(va)} ({time.time() - t0:.0f}s) settings {settings}")
    rep = {"seed": seed, "settings": settings.__dict__, "train_graphs": len(tr), "val_graphs": len(va), "variants": {}}
    for variant in ("C", "D"):
        seeds = SeedChain(seed)
        model = build_trust_head(base.model, seeds.torch_seed("model_init"), base.device, variant)
        n_par = num_parameters(model)
        ref = build_trust_head(base.model, 0, base.device, variant)
        load_checkpoint(REPO_ROOT / "checkpoints" / f"s3-seed{seed}-{variant}.pt", ref)
        assert n_par == num_parameters(ref), "parameter count must equal the frozen checkpoint's (D83 item 1)"
        res = train(model, tr, va, settings, seeds, log=say)
        save_checkpoint(REPO_ROOT / "checkpoints" / "s18" / f"s3-seed{seed}-{variant}.pt", model, variant, seed, settings, res)
        rep["variants"][variant] = {"parameters": n_par, "best_epoch": res.best_epoch, "stopped_epoch": res.stopped_epoch,
                                    "converged_by": res.converged_by, "initial_val_bce": res.initial_val_bce,
                                    "best_val_bce": min(h["val_bce"] for h in res.history)}
        say(f"{variant}: {rep['variants'][variant]}")
    (OUT / f"train-seed{seed}.json").write_text(json.dumps(rep, indent=1))
    return seed, rep


def cmd_train(seeds, workers: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "checkpoints" / "s18").mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(min(workers, len(seeds))) as pool:
        for seed, rep in pool.imap_unordered(_train_seed, list(seeds)):
            print(f"seed {seed}: " + "; ".join(f"{v} params {r['parameters']} best epoch {r['best_epoch']} stopped {r['stopped_epoch']} "
                                                f"({r['converged_by']}) val BCE {r['initial_val_bce']:.4f} -> {r['best_val_bce']:.4f}"
                                                for v, r in rep["variants"].items()), flush=True)
    return 0


# ------------------------------------------------------------------ open-loop trust

def trust_path(arm, cond, seed, draw) -> Path:
    return OUT / f"trust-{arm}" / f"{cond}-seed{seed}-draw{draw}.npz"


def compute_trust(arm, cond, seed, draw) -> dict:
    from trustgraph.config import with_overrides
    from trustgraph.dataset import graphset_from
    from trustgraph.features import RSU_COL
    from trustgraph.observed import default_observed_path, load_observed
    from trustgraph.train import predict_trust

    base, world, trace, models = _cond_state(cond, seed, arm)
    cfg = with_overrides(base, seed=seed, draw=draw)
    observed = load_observed(default_observed_path(CONDS[cond][0], cfg))
    gs = graphset_from(cfg, world, trace, observed, stride=1, step_range=(FIRST, LAST))
    assert gs.steps[0] == FIRST and gs.steps[-1] == LAST - 1
    out = {n: predict_trust(m, gs).astype(np.float32) for n, m in models.items()}
    if arm == "primary":
        out["M"] = observed.rsu_features[FIRST:LAST, :, RSU_COL["success_ewma"]].astype(np.float32)
    return out


def _trust_one(job):
    arm, cond, seed, draw = job
    out = trust_path(arm, cond, seed, draw)
    if out.exists():
        return job, "cached"
    tr = compute_trust(arm, cond, seed, draw)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **tr)
    tmp.replace(out)
    return job, "ok"


def cmd_trust(arm, seeds, workers: int) -> int:
    jobs = [(arm, c, s, d) for s in seeds for c in CONDS for d in EVAL_DRAWS]
    print(f"[trust {arm}] {len(jobs)} scenarios", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for job, st in pool.imap_unordered(_trust_one, jobs, chunksize=4):
            n += 1
            if n == 1:
                note_first_number(f"evaluation_first_trust_{arm}")
            if n % 200 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ closed loop

def load_weights() -> dict:
    w = yaml.safe_load((REPO_ROOT / "configs" / "weights.yaml").read_text())["weights"]
    h = yaml.safe_load((REPO_ROOT / "configs" / "weights_h.yaml").read_text())["weights"]["H"]
    d2 = yaml.safe_load((REPO_ROOT / "configs" / "weights_d2.yaml").read_text())["weights"]["shared"]["D2"]
    t = lambda d: (d["alpha"], d["beta"], d["gamma"])  # noqa: E731
    return {"A": t(w["A"]), "C": t(w["C"]), "D": t(w["D"]), "B_oracle": t(w["B_oracle"]), "M": t(h), "D2": t(d2),
            "C_r": t(w["C"]), "D_r": t(w["D"])}  # the retrained heads keep the primary C and D weights: no tuning (D83)


def cl_path(name, cond, seed, draw) -> Path:
    return OUT / "closed" / f"{name}-{cond}-seed{seed}-draw{draw}.json"


def closed_metrics(sc, mode: str) -> dict:
    from trustgraph import sweep_metrics as M

    tasks, gt, n_steps = sc.observed.tasks, sc.ground_truth, sc.observed.num_steps
    pops = populations(gt, mode)
    step = tasks.step
    win = lambda a, b: (step >= a) & (step < b)  # noqa: E731
    total, deg = M.dispatch_series(step, tasks.rsu, gt.degraded, n_steps)
    av = M.avoidance(total, deg, int(gt.onset_step))  # D56 run-level (the registered secondary (b)); onset = the scenario's first onset
    never_rsu, av_rsu = 0, 0
    for v in pops["faulty"]:  # S18 prompt wording, D84: each faulty RSU on its own onset
        only = np.zeros(gt.num_rsus, bool)
        only[v] = True
        tot_v, deg_v = M.dispatch_series(step, tasks.rsu, only, n_steps)
        a = M.avoidance(tot_v, deg_v, int(pops["start"][v]))
        never_rsu += int(a["never_avoided"])
        av_rsu += a["n_av"]
    return {"success_400_500": float(tasks.success_rate(win(400, 500))), "success_500_1200": float(tasks.success_rate(win(500, 1200))),
            "success_400_1200": float(tasks.success_rate(win(400, 1200))), "n_disp_400_500": int(((tasks.rsu >= 0) & win(400, 500)).sum()),
            "n_av": av["n_av"], "n_win": av["n_win"], "never_avoided": bool(av["never_avoided"]), "t_avoid": av["t_avoid"],
            "faulty": int(pops["faulty"].size), "faulty_never_avoided": never_rsu, "faulty_n_av_sum": int(av_rsu)}


def _cl_job(job):
    from trustgraph.config import with_overrides
    from trustgraph.offload import build_offload_settings
    from trustgraph.simulator import run_closed_loop

    arm, name, cond, seed, draw = job
    base, world, trace, models = _cond_state(cond, seed, arm if name in ARMS[arm] else None)
    cfg = with_overrides(base, seed=seed, draw=draw)
    w = load_weights()[name]
    settings = build_offload_settings(cfg, variant=CL_VARIANT[name], weights=w, record_trust=False)
    sc = run_closed_loop(cfg, world, trace, settings, model=models.get(name))
    return job, closed_metrics(sc, CONDS[cond][1])


def cmd_closed(arm, seeds, workers: int) -> int:
    todo = [(arm, v, c, s, d) for s in seeds for c in CONDS for d in EVAL_DRAWS for v in CL_NAMES[arm] if not cl_path(v, c, s, d).exists()]
    print(f"[closed {arm}] {len(todo)} runs to do", flush=True)
    t0, n = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_single_thread) as pool:
        for job, r in pool.imap_unordered(_cl_job, todo, chunksize=4):
            _, v, c, s, d = job
            p = cl_path(v, c, s, d)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(r))
            n += 1
            if n == 1:
                note_first_number(f"closed_loop_first_result_{arm}")
            if n % 500 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ harness check (draw 900; not a result)

def cmd_check() -> int:
    """Determinism and plumbing on harness draw 900, seed 1, never a result: two generations of one scenario are identical;
    the trust path of a frozen model is identical on a repeat; one closed-loop run repeats exactly; the L = 0 scenario's
    observable stream equals the plain shared-onset scenario of the same seed and draw."""
    from trustgraph.config import load_config, with_overrides
    from trustgraph.observed import load_observed
    from trustgraph.sealed.ground_truth import load_ground_truth, sealed_path_for
    from trustgraph.simulator import generate_scenario

    S16.S15._single_thread()
    ok = True
    for cond in ("L80", "f20"):
        _gen_one((cond, 1, HARNESS_DRAW))
        p = obs_path(cond, 1, HARNESS_DRAW)
        a = load_observed(p)
        base, world, trace, _ = _cond_state(cond, 1, None)
        b = generate_scenario(with_overrides(base, seed=1, draw=HARNESS_DRAW), world, trace).observed
        same = all(np.array_equal(getattr(a.tasks, f), getattr(b.tasks, f)) for f in ("step", "rsu", "met_deadline")) and np.array_equal(a.rsu_features, b.rsu_features)
        print(f"  {cond}: regenerated scenario equals the stored one: {same}")
        ok &= same
    t1 = compute_trust("primary", "L80", 1, HARNESS_DRAW)
    _W.clear()
    t2 = compute_trust("primary", "L80", 1, HARNESS_DRAW)
    same = all(np.array_equal(t1[k], t2[k]) for k in t1)
    print(f"  open-loop trust (C, D, D2, M) repeats exactly: {same}")
    ok &= same
    _, r1 = _cl_job(("primary", "D", "L80", 1, HARNESS_DRAW))
    _W.clear()
    _, r2 = _cl_job(("primary", "D", "L80", 1, HARNESS_DRAW))
    print(f"  closed-loop D run repeats exactly: {r1 == r2}")
    ok &= r1 == r2
    gt = load_ground_truth(sealed_path_for(obs_path("L80", 1, HARNESS_DRAW)))
    ok &= bool(gt.spread_hop is not None)
    print(f"  harness draw {HARNESS_DRAW}: seed RSU hop 0 present: {bool((gt.spread_hop == 0).sum() == 1)}")
    return 0 if ok else 1


# ------------------------------------------------------------------ open-loop evaluation: latency, false alarms, window AUC

def evaluate(arm: str, seeds):
    from trustgraph.sealed.ground_truth import load_ground_truth, sealed_path_for

    names = OPEN_NAMES[arm]
    res = {}
    for cond in CONDS:
        mode = CONDS[cond][1]
        for seed in seeds:
            tr, gts = {}, {}
            for d in EVAL_DRAWS:
                with np.load(trust_path(arm, cond, seed, d)) as z:
                    tr[d] = {m: z[m] for m in names}
                gts[d] = load_ground_truth(sealed_path_for(obs_path(cond, seed, d)))
            tau = {m: S16.pre_onset_tau([tr[d][m] for d in EVAL_DRAWS]) for m in names}
            keys = ("lat", "det", "fa", "fa_pre", "fa_post", "fa_rh", "fa_nrh", "auc_all", "auc_rh")
            rec = {m: {k: [] for k in keys} for m in names}
            counts = {"n_pop": [], "n_healthy": [], "n_rh": [], "hops": []}
            for d in EVAL_DRAWS:
                P = populations(gts[d], mode)
                counts["n_pop"].append(int(P["pop"].size))
                counts["n_healthy"].append(int(P["healthy"].size))
                counts["n_rh"].append(int(P["restarted_healthy"].size))
                if mode == "spread":
                    counts["hops"].append([int(h) for h in gts[d].spread_hop[gts[d].degraded]])
                else:  # restarted-but-healthy RSUs that share a backhaul segment with a failing RSU (D81 10e)
                    seg = gts[d].backhaul_segment_id
                    fail_seg = set(seg[P["faulty"]].tolist())
                    counts["hops"].append(int(sum(int(seg[v]) in fail_seg for v in P["restarted_healthy"])))
                neg_all = np.setdiff1d(np.arange(gts[d].num_rsus), P["faulty"])
                for m in names:
                    t = tr[d][m]
                    r = rec[m]
                    if P["pop"].size:
                        lats = [S13.detect_latency(t[:, v], tau[m], int(P["start"][v])) for v in P["pop"]]
                        r["lat"].append(float(np.mean([S13.SEARCH - 1 if l is None else l for l in lats])))
                        r["det"].append(float(np.mean([l is not None for l in lats])))
                    else:
                        r["lat"].append(float("nan"))
                        r["det"].append(float("nan"))
                    fa = np.array([S16.false_alarm(t[:, v], tau[m]) for v in P["healthy"]], float).reshape(-1, 3)
                    for i, k in enumerate(("fa", "fa_pre", "fa_post")):
                        r[k].append(float(fa[:, i].mean()) if P["healthy"].size else float("nan"))
                    rh = np.isin(P["healthy"], P["restarted_healthy"])
                    r["fa_rh"].append(float(fa[rh, 0].mean()) if rh.any() else float("nan"))
                    r["fa_nrh"].append(float(fa[~rh, 0].mean()) if (~rh).any() else float("nan"))
                    if mode == "spread":
                        r["auc_all"].append(window_auc(t, P["pop"], P["healthy"], P["start"]))
                        r["auc_rh"].append(float("nan"))
                    else:
                        r["auc_all"].append(window_auc(t, P["pop"], neg_all, np.full(gts[d].num_rsus, 400)))
                        r["auc_rh"].append(window_auc(t, P["pop"], P["restarted_healthy"], np.full(gts[d].num_rsus, 400)))
            res[(cond, seed)] = {"tau": tau, "counts": counts, **{m: {k: np.array(v) for k, v in rec[m].items()} for m in names}}
    return res


def _seedmean(res, cond, seeds, m, k, ok_from=None):
    """Per-seed mean over the draws that have a population (the D42 'aggregated' draws)."""
    out = []
    for s in seeds:
        r = res[(cond, s)]
        ok = ~np.isnan(r[ok_from or m]["lat"]) if ok_from is not False else np.ones(len(EVAL_DRAWS), bool)
        out.append(np.nanmean(r[m][k][ok]) if np.isfinite(r[m][k][ok]).any() else float("nan"))
    return np.array(out)


def _paired(res, cond, seeds, a, b, k):
    """Per-seed mean over draws of (a - b)[k], on draws with a population for the latency, all draws otherwise."""
    out = []
    for s in seeds:
        r = res[(cond, s)]
        ok = ~np.isnan(r[a]["lat"]) if k in ("lat", "det") else np.ones(len(EVAL_DRAWS), bool)
        x = (r[a][k] - r[b][k])[ok]
        out.append(np.nanmean(x) if np.isfinite(x).any() else float("nan"))
    return np.array(out)


def interval(v):
    return S13.t_interval(np.asarray(v, float))


def cmd_eval(arm: str, seeds, verdict: bool) -> int:
    names = OPEN_NAMES[arm]
    ref, tgt = ("C", "D") if arm == "primary" else ("C_r", "D_r")
    res = evaluate(arm, seeds)
    L = ["=" * 124, f"S18 OPEN LOOP ({arm.upper()} ARM): frozen heads {'(C, D, D2, M = success_ewma)' if arm == 'primary' else '(C_r, D_r retrained on draws 600-640, D83)'}, "
         f"draws 400-449, seeds {list(seeds)}, shared D76 rule" + (" WITH D81 LABELS" if verdict else " (RAW: no labels)"), "=" * 124]
    summary = {}
    for cond in CONDS:
        mode = CONDS[cond][1]
        L.append(f"\n--- condition {cond} ({'spreading fault, L = ' + str(CONDS[cond][2]) if mode == 'spread' else 'cold start, f = ' + str(CONDS[cond][2])}) ---")
        L.append("  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau")
        for s in seeds:
            r, c = res[(cond, s)], res[(cond, s)]["counts"]
            L.append(f"    seed {s}: attempted {len(EVAL_DRAWS)} failed 0 aggregated {int((np.array(c['n_pop']) > 0).sum())} | pop {sum(c['n_pop'])} healthy {sum(c['n_healthy'])} "
                     f"restarted-healthy {sum(c['n_rh'])} | tau " + " ".join(f"{m} {r['tau'][m]:.4f}" for m in names))
        if mode == "spread":
            hist = np.bincount(np.concatenate([np.concatenate([h for h in res[(cond, s)]['counts']['hops']]) for s in seeds]).astype(int))
            L.append(f"  realised hop histogram of the degraded RSUs (hop 0, 1, 2, ...): {hist.tolist()}")
        else:
            L.append(f"  restarted-but-healthy RSUs sharing a backhaul segment with a failing RSU (sum over draws, per seed): "
                     f"{[int(sum(res[(cond, s)]['counts']['hops'])) for s in seeds]}")
        pop_txt = "degraded RSUs with hop >= 1" if mode == "spread" else "failing RSUs, from onset 420"
        L.append(f"  per model ({pop_txt}); mean +/- sd over seeds of per-seed draw means")
        for m in names:
            lat, det, fa = _seedmean(res, cond, seeds, m, "lat", m), _seedmean(res, cond, seeds, m, "det", m), _seedmean(res, cond, seeds, m, "fa", False)
            line = (f"    {m:4s} latency {lat.mean():6.1f} +/- {lat.std(ddof=1):5.1f}  detected {det.mean():.3f}  | false-alarm rate {fa.mean():.4f} +/- {fa.std(ddof=1):.4f} "
                    f"(starting pre-onset {_seedmean(res, cond, seeds, m, 'fa_pre', False).mean():.4f}, only after {_seedmean(res, cond, seeds, m, 'fa_post', False).mean():.4f})")
            if mode == "restart":
                line += (f"  restarted-healthy {_seedmean(res, cond, seeds, m, 'fa_rh', False).mean():.4f}  not restarted {_seedmean(res, cond, seeds, m, 'fa_nrh', False).mean():.4f}")
            L.append(line)
            auc = _seedmean(res, cond, seeds, m, "auc_all", False)
            L.append(f"         trust AUC, first {WINDOW_N} steps after {'own onset (vs healthy)' if mode == 'spread' else 'restart [400, 500) (failing vs all non-failing)'}: {np.nanmean(auc):.4f} +/- {np.nanstd(auc, ddof=1):.4f}"
                     + (f";  failing vs restarted-healthy only: {np.nanmean(_seedmean(res, cond, seeds, m, 'auc_rh', False)):.4f}" if mode == "restart" else ""))
        pairs = [(tgt, ref)] + ([("D2", "C"), ("M", "C")] if arm == "primary" else [])
        for a, b in pairs:
            dl, df = _paired(res, cond, seeds, a, b, "lat"), _paired(res, cond, seeds, a, b, "fa")
            sl, sf = interval(dl), interval(df)
            tag = lambda x: f"   {x}" if (verdict and (a, b) == (tgt, ref)) else ""  # noqa: E731
            lab_l, lab_f = label_latency(sl[0], sl[2], sl[3]), label_fa(sf[3])
            L.append(f"  {a} - {b} latency: {sl[0]:+.1f} (sd {sl[1]:.1f}) 95% [{sl[2]:+.1f}, {sl[3]:+.1f}]" + tag(lab_l) + f"   per seed {[round(float(x), 1) for x in dl]}")
            L.append(f"  {a} - {b} false-alarm rate: {sf[0]:+.4f} (sd {sf[1]:.4f}) 95% [{sf[2]:+.4f}, {sf[3]:+.4f}]" + tag(lab_f)
                     + (f"   strict (hi <= 0): {fa_strict(sf[3])}" if verdict and (a, b) == (tgt, ref) else "") + f"   per seed {[round(float(x), 4) for x in df]}")
            if (a, b) == (tgt, ref):
                summary[cond] = {"lat": sl, "fa": sf, "lat_label": lab_l, "fa_label": lab_f, "strict": fa_strict(sf[3]), "per_seed_lat": [float(x) for x in dl]}
                da = _paired(res, cond, seeds, a, b, "auc_all")
                sa = interval(da)
                L.append(f"  {a} - {b} trust AUC (first {WINDOW_N} steps): {sa[0]:+.4f} (sd {sa[1]:.4f}) 95% [{sa[2]:+.4f}, {sa[3]:+.4f}]  (descriptive)")
        if mode == "spread" and cond != "L0":  # D81 section 8: change of the gap relative to L = 0, same draws (paired)
            ch = []
            for s in seeds:
                r0, r1 = res[("L0", s)], res[(cond, s)]
                ok = ~np.isnan(r0[ref]["lat"]) & ~np.isnan(r1[ref]["lat"])
                g0, g1 = (r0[tgt]["lat"] - r0[ref]["lat"])[ok], (r1[tgt]["lat"] - r1[ref]["lat"])[ok]
                ch.append(float(np.mean(g1 - g0)))
            sc_ = interval(ch)
            summary[cond]["vs_L0"] = sc_
            L.append(f"  change of the {tgt} - {ref} latency gap from L = 0 to this L (same draws): {sc_[0]:+.1f} 95% [{sc_[2]:+.1f}, {sc_[3]:+.1f}]  (descriptive)")
    if verdict and arm == "primary":
        L += ["\n" + "=" * 124, "D81 OPEN-LOOP VERDICTS (primary arm; Sb is in the closed-loop report)"]
        per_l = {l: (summary[l]["lat_label"], summary[l]["fa_label"]) for l in SPREAD}
        for l in SPREAD:
            L.append(f"  L = {CONDS[l][2]:2d}: latency {per_l[l][0]:8s} false alarms {per_l[l][1]:15s} (strict hi <= 0: {summary[l]['strict']})" + ("   [descriptive, not in the verdict]" if l == "L20" else "")
                     + ("   [control]" if l == "L0" else ""))
        L.append(f"  Sa (spreading fault; holds iff WIN and ACCEPTABLE at both L = 40 and L = 80): {sa_overall(per_l)}")
        L.append(f"  Sc (control; holds iff the L = 0 latency label is NULL): {sc_overall(per_l['L0'][0])}")
        adv0 = -summary["L0"]["lat"][0]
        for l in ("L40", "L80"):
            adv = -summary[l]["lat"][0]
            if per_l[l][0] == "WIN":
                L.append(f"  attribution at {l}: advantage {adv:.1f} steps vs L = 0 advantage {adv0:.1f} -> " + ("attributed to the lag" if adv > adv0 else "NOT DUE TO THE LAG"))
    text = "\n".join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"report-open-{arm}-{'verdict' if verdict else 'raw'}.txt").write_text(text + "\n", encoding="utf-8")
    (OUT / f"summary-open-{arm}.json").write_text(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in summary.items()}, indent=1, default=float))
    print(text)
    return 0


# ------------------------------------------------------------------ closed-loop report

def load_closed(arm, seeds):
    out = {}
    for v in CL_NAMES[arm]:
        for c in CONDS:
            out[(v, c)] = {k: np.array([[json.loads(cl_path(v, c, s, d).read_text())[k] for d in EVAL_DRAWS] for s in seeds], float)
                           for k in ("success_400_500", "success_500_1200", "success_400_1200", "n_av", "n_win", "never_avoided",
                                     "faulty", "faulty_never_avoided", "n_disp_400_500")}
    return out


def cmd_report(arm: str, seeds, verdict: bool) -> int:
    w = load_weights()
    data = load_closed(arm, seeds)
    names = CL_NAMES[arm]
    ref, tgt = ("C", "D") if arm == "primary" else ("C_r", "D_r")
    L = ["=" * 124, f"S18 CLOSED LOOP ({arm.upper()} ARM): dwell filter and handoff on, draws 400-449, seeds {list(seeds)}"
         + (" WITH D81 LABELS" if verdict else " (RAW: no labels)"), "=" * 124,
         "weights (alpha, beta, gamma): " + "  ".join(f"{n} {w[n]}" for n in names) + "  (existing tuned files; no new tuning)"]
    sb_labels, sb_sum = [], {}
    for cond in CONDS:
        mode = CONDS[cond][1]
        key = "success_400_1200" if mode == "spread" else "success_400_500"
        L.append(f"\n--- condition {cond} ({'spreading fault, L = ' + str(CONDS[cond][2]) if mode == 'spread' else 'cold start, f = ' + str(CONDS[cond][2])}) ---")
        L.append(f"  D42: attempted {len(seeds) * len(EVAL_DRAWS)} per model, failed 0, aggregated {np.isfinite(data[(names[0], cond)][key]).sum()}")
        for v in names:
            d = data[(v, cond)]
            sm = np.nanmean(d[key], axis=1)
            line = f"  {v:8s} success {'[400,1200)' if mode == 'spread' else '[400,500)'} {sm.mean():.4f} +/- {sm.std(ddof=1):.4f}"
            if mode == "restart":
                s2 = np.nanmean(d["success_500_1200"], axis=1)
                line += f"   after the window [500,1200) {s2.mean():.4f} +/- {s2.std(ddof=1):.4f}"
            f = lambda k: (d[k].mean(axis=1).mean(), d[k].mean(axis=1).std(ddof=1))  # noqa: E731
            line += (f"  | n_av {f('n_av')[0]:7.1f} +/- {f('n_av')[1]:6.1f} n_win {f('n_win')[0]:7.1f} +/- {f('n_win')[1]:6.1f} "
                     f"run-level never-avoided {f('never_avoided')[0]:.3f} +/- {f('never_avoided')[1]:.3f}"
                     f" | faulty RSUs never avoided {(d['faulty_never_avoided'].sum() / d['faulty'].sum()):.3f} ({int(d['faulty_never_avoided'].sum())}/{int(d['faulty'].sum())})")
            L.append(line)
        pairs = [(tgt, ref)] + ([("D2", "C"), ("M", "C"), ("C", "A"), ("D", "A")] if arm == "primary" else [("D_r", "D"), ("C_r", "C")])
        for a, b in pairs:
            x = np.nanmean(data[(a, cond)][key] - data[(b, cond)][key], axis=1)
            m, sd, lo, hi = interval(x)
            lab = label_success(m, lo, hi) if (verdict and (a, b) == (tgt, ref) and mode == "restart") else ""
            L.append(f"  {a} - {b}: {m:+.4f} (sd {sd:.4f}) 95% [{lo:+.4f}, {hi:+.4f}]" + (f"   {lab}" if lab else "") + f"   per seed {[round(float(v), 4) for v in x]}")
            if (a, b) == (tgt, ref):
                sb_sum[cond] = {"succ": (m, sd, lo, hi), "label": label_success(m, lo, hi)}
                if mode == "restart":
                    sb_labels.append(label_success(m, lo, hi))
                    x2 = np.nanmean(data[(a, cond)]["success_500_1200"] - data[(b, cond)]["success_500_1200"], axis=1)
                    m2, sd2, lo2, hi2 = interval(x2)
                    L.append(f"  {a} - {b} after the window [500,1200): {m2:+.4f} 95% [{lo2:+.4f}, {hi2:+.4f}]  (descriptive)")
        if arm == "primary":
            for k in ("n_av", "n_win"):
                red = (data[("C", cond)][k].mean(axis=1) - data[("D", cond)][k].mean(axis=1)) / data[("A", cond)][k].mean(axis=1)
                m, sd, lo, hi = interval(red)
                L.append(f"  D57-normalised (n_C - n_D)/n_A on {k}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (positive = D sends fewer to the faulty RSUs; descriptive)")
    if verdict and arm == "primary":
        L += ["\n" + "=" * 124, "D81 SECTION 9 VERDICT (Sb, cold start; closed-loop success [400,500), D - C)"]
        for c, lab in zip(RESTART, sb_labels):
            L.append(f"  f = {CONDS[c][2]:.2f}: {lab}")
        L.append(f"  Sb: {sb_overall(sb_labels)}  (WIN at {sb_labels.count('WIN')} of 3, NEGATIVE at {sb_labels.count('NEGATIVE')} of 3)")
    text = "\n".join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"report-closed-{arm}-{'verdict' if verdict else 'raw'}.txt").write_text(text + "\n", encoding="utf-8")
    (OUT / f"summary-closed-{arm}.json").write_text(json.dumps(sb_sum, indent=1, default=float))
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("gen", "train", "check", "trust", "closed", "eval", "report"))
    ap.add_argument("--arm", choices=("primary", "secondary"), default="primary")
    ap.add_argument("--set", choices=("eval", "train"), default="eval")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    return {"gen": lambda: cmd_gen(a.set, a.seeds, a.workers), "train": lambda: cmd_train(a.seeds, a.workers), "check": cmd_check,
            "trust": lambda: cmd_trust(a.arm, a.seeds, a.workers), "closed": lambda: cmd_closed(a.arm, a.seeds, a.workers),
            "eval": lambda: cmd_eval(a.arm, a.seeds, a.verdict), "report": lambda: cmd_report(a.arm, a.seeds, a.verdict)}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())

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


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("gen", "mech", "eval", "tune", "closed", "report"))
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
    raise SystemExit(f"{a.cmd}: not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())

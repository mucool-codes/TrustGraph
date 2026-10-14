"""S6 Task 7: tune alpha/beta/gamma on the tuning seed, same procedure for every variant (D50).

    python scripts/s6_tune.py --config configs/demo.yaml --workers 12

Procedure (fixed in DECISIONS.md D50 before this ran): seed 100, rho 1.0, fraction 0.20, injection
draws {0, 1, 2}, closed loop with the dwell filter and handoff on; objective = mean over the draws of
deadline success of dispatched tasks over steps [400, 1200). Full factorial grids: alpha {.25,.5,1,2,4} x
beta {.25,.5,1} x gamma {.15,.3,.6} for C, D, B_oracle; beta x gamma at alpha = 0 for A. B takes A's
weights (its trust term is the constant 1). Ties go to the first grid point in ascending
(alpha, beta, gamma) order.

Writes results/s6/tuning-grid.json (every grid point, raw) and, with --write-weights,
configs/weights.yaml. Raw numbers go into FINDINGS.md before the weights are used anywhere.

EVALUATION-SIDE (reads ground truth for the objective's onset step only); allowlisted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TUNING_SEED = 100
RHO, FRACTION = 1.0, 0.20
DRAWS = (0, 1, 2)
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)
BETAS = (0.25, 0.5, 1.0)
GAMMAS = (0.15, 0.3, 0.6)
VARIANTS_TUNED = ("A", "B_oracle", "C", "D")
ONSET = 400

_STATE: dict = {}


def grid(variant: str):
    alphas = (0.0,) if variant == "A" else ALPHAS
    return list(itertools.product(alphas, BETAS, GAMMAS))


def _init(config_path: str):
    import torch

    torch.set_num_threads(1)
    from trustgraph.config import load_config, with_overrides
    from trustgraph.model import build_trust_head
    from trustgraph.scenario import build_world
    from trustgraph.train import load_checkpoint
    from trustgraph.trace import default_trace_path, load_trace

    base = load_config(config_path)
    cfg = with_overrides(base, seed=TUNING_SEED)
    models = {}
    for v in ("C", "D"):
        m = build_trust_head(base.model, 0, base.device, v)
        load_checkpoint(REPO_ROOT / "checkpoints" / f"s3-seed{TUNING_SEED}-{v}.pt", m)
        models[v] = m
    _STATE.update(
        base=base, world=build_world(cfg), trace=load_trace(default_trace_path(config_path, TUNING_SEED)),
        models=models,
    )


def _run(task):
    from trustgraph.config import with_overrides
    from trustgraph.offload import OffloadSettings
    from trustgraph.simulator import run_closed_loop

    variant, (alpha, beta, gamma), draw = task
    cfg = with_overrides(_STATE["base"], seed=TUNING_SEED, rho=RHO, fraction=FRACTION, draw=draw)
    settings = OffloadSettings(variant=variant, alpha=alpha, beta=beta, gamma=gamma)
    model = _STATE["models"].get(variant)  # only C and D have a head
    sc = run_closed_loop(cfg, _STATE["world"], _STATE["trace"], settings, model=model)
    tasks = sc.observed.tasks
    post = tasks.step >= ONSET
    return task, {
        "success_post_onset": tasks.success_rate(post),
        "success_all": tasks.success_rate(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--write-weights", action="store_true")
    args = ap.parse_args()

    tasks = [(v, g, d) for v in VARIANTS_TUNED for g in grid(v) for d in DRAWS]
    print(f"{len(tasks)} closed-loop runs on tuning seed {TUNING_SEED}", flush=True)
    t0 = time.time()
    results = {}
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.config,)) as pool:
        for i, (task, r) in enumerate(pool.imap_unordered(_run, tasks, chunksize=3), 1):
            results[task] = r
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)

    table, chosen = {}, {}
    for v in VARIANTS_TUNED:
        rows = []
        for g in grid(v):
            per = [results[(v, g, d)]["success_post_onset"] for d in DRAWS]
            rows.append({"alpha": g[0], "beta": g[1], "gamma": g[2], "objective": float(np.mean(per)), "per_draw": per,
                         "success_all": float(np.mean([results[(v, g, d)]["success_all"] for d in DRAWS]))})
        table[v] = rows
        best = max(range(len(rows)), key=lambda i: (rows[i]["objective"], -i))  # first max in grid order
        chosen[v] = rows[best]
    out = REPO_ROOT / "results" / "s6"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tuning-grid.json").write_text(json.dumps({"grid": table, "chosen": chosen}, indent=1), encoding="utf-8")

    for v in VARIANTS_TUNED:
        print(f"\n=== {v}: objective = mean post-onset deadline success over draws {DRAWS}")
        print("  alpha  beta  gamma   objective   per-draw")
        for r in table[v]:
            mark = "  <== chosen" if r is chosen[v] else ""
            print(f"  {r['alpha']:5.2f} {r['beta']:5.2f} {r['gamma']:5.2f}   {r['objective']:.4f}    "
                  f"{[round(x, 4) for x in r['per_draw']]}{mark}")
    if args.write_weights:
        weights = {v: {k: chosen[v][k] for k in ("alpha", "beta", "gamma")} for v in VARIANTS_TUNED}
        doc = {
            "comment": "Tuned on seed 100, rho 1.0, fraction 0.2, draws 0-2 (DECISIONS.md D50). E uses D's; B uses A's.",
            "tuning_seed": TUNING_SEED,
            "weights": weights,
        }
        (REPO_ROOT / "configs" / "weights.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print("wrote configs/weights.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

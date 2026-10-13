"""S4: evaluate the FROZEN per-seed C and D checkpoints on the D43 conditions.

    python scripts/s4_eval.py --config configs/demo.yaml --seeds 1 2 3 4 5 --workers 12

No training happens here. For every (seed, condition, rho, injection draw 100-149) it
generates the scenario (cached under scenarios/s4/), builds the graph sequence from the
observable file, runs both frozen models, and writes one small raw file per scenario to
results/s4/raw/. Aggregation, intervals and the verdict are s4_analyse.py's job.

Conditions (DECISIONS.md D43 section 1), all at degraded fraction 0.20 (K = 4):
    sweep      rho in {0, .25, .5, .75, 1}; no collusion, no cold start
    collusion  rho in {0, 1}; 1 group of 3 colluders, under_report 0.5 (config values)
    cold       rho in {0, 1} x placement {healthy_segment = control, degraded_segment =
               test}; 1 cold-start RSU joining at step 600

EVALUATION-SIDE: holds model outputs and ground truth together, so it is allowlisted in
tests/test_sealing.py. It only reads ground truth to store the class labels next to the
trust scores; no model ever sees them.

A cold-start draw the injector cannot place is recorded as `.infeasible` and counted by
the analysis (D42) - never skipped silently.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RHOS_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)
RHOS_BINARY = (0.0, 1.0)
PLACEMENTS = ("healthy_segment", "degraded_segment")
DRAWS = tuple(range(100, 150))
FRACTION = 0.20
TRUST_STEPS = (100, 810)  # stored per-step trust: pre-onset, ramp, plateau-start + 10 for the run rule
PLATEAU = (1100, 1200)

OUT_DIR = REPO_ROOT / "results" / "s4" / "raw"
SCEN_DIR = REPO_ROOT / "scenarios" / "s4"


def task_key(seed: int, kind: str, rho: float, placement: str, draw: int) -> str:
    tag = f"-{placement}" if kind == "cold" else ""
    return f"seed{seed}-{kind}-rho{rho:.2f}{tag}-draw{draw}"


def all_tasks(seeds):
    tasks = []
    for seed in seeds:
        for rho in RHOS_SWEEP:
            for d in DRAWS:
                tasks.append((seed, "sweep", rho, "none", d))
        for rho in RHOS_BINARY:
            for d in DRAWS:
                tasks.append((seed, "collusion", rho, "none", d))
            for placement in PLACEMENTS:
                for d in DRAWS:
                    tasks.append((seed, "cold", rho, placement, d))
    return tasks


# ------------------------------------------------------------------------- worker state
_STATE: dict = {}


def _init(config_path: str, seeds: list[int]) -> None:
    import torch

    torch.set_num_threads(1)
    from trustgraph.config import load_config, with_overrides
    from trustgraph.model import build_trust_head
    from trustgraph.scenario import build_world
    from trustgraph.train import load_checkpoint
    from trustgraph.trace import default_trace_path, load_trace

    base = load_config(config_path)
    _STATE["config_path"] = config_path
    _STATE["base"] = base
    _STATE["seeds"] = {}
    for seed in seeds:
        cfg = with_overrides(base, seed=seed)
        models = {}
        for variant in ("C", "D"):
            ck = REPO_ROOT / "checkpoints" / f"s3-seed{seed}-{variant}.pt"
            if not ck.exists():
                models[variant] = None
                continue
            model = build_trust_head(base.model, 0, base.device, variant)
            blob = load_checkpoint(ck, model)
            assert blob["variant"] == variant and blob["seed"] == seed
            models[variant] = model
        _STATE["seeds"][seed] = {
            "world": build_world(cfg),
            "trace": load_trace(default_trace_path(config_path, seed)),
            "models": models,
        }


def _run(task):
    from trustgraph.config import with_overrides
    from trustgraph.dataset import graphset_from
    from trustgraph.observed import load_observed, scenario_stem
    from trustgraph.sealed.ground_truth import load_ground_truth, sealed_path_for
    from trustgraph.simulator import generate_scenario
    from trustgraph.dataset import GraphSet
    from trustgraph.train import predict_trust

    seed, kind, rho, placement, draw = task
    key = task_key(seed, kind, rho, placement, draw)
    out = OUT_DIR / f"{key}.npz"
    bad = OUT_DIR / f"{key}.infeasible"
    if out.exists() or bad.exists():
        return key, "cached"

    st = _STATE["seeds"][seed]
    if st["models"]["C"] is None or st["models"]["D"] is None:
        return key, "no-checkpoint"

    base = _STATE["base"]
    cold = kind == "cold"
    cfg = with_overrides(
        base,
        seed=seed,
        rho=rho,
        fraction=FRACTION,
        draw=draw,
        cold_start_nodes=1 if cold else None,
        cold_start_placement=placement if cold else None,
    )
    if kind == "collusion":
        cfg = replace(cfg, collusion={**cfg.collusion, "num_groups": 1})

    stem = scenario_stem(_STATE["config_path"], cfg) + ("-coll1" if kind == "collusion" else "")
    obs_path = SCEN_DIR / f"{stem}.observed.npz"
    sealed_path = sealed_path_for(obs_path)
    if not (obs_path.exists() and sealed_path.exists()):
        try:
            scenario = generate_scenario(cfg, st["world"], st["trace"])
        except ValueError as exc:
            if "infeasible" not in str(exc):
                raise
            bad.write_text(str(exc), encoding="utf-8")
            return key, "infeasible"
        SCEN_DIR.mkdir(parents=True, exist_ok=True)
        scenario.observed.save(obs_path)
        scenario.ground_truth.save(sealed_path)

    observed = load_observed(obs_path)  # graphs are built from what is on disk (D37)
    gt = load_ground_truth(sealed_path)
    gs = graphset_from(cfg, st["world"], st["trace"], observed, stride=1, step_range=(100, 1200))
    steps = gs.steps
    keep = ((steps >= TRUST_STEPS[0]) & (steps < TRUST_STEPS[1])) | (
        (steps >= PLATEAU[0]) & (steps < PLATEAU[1])
    )
    sub = GraphSet(
        graphs=[g for g, k in zip(gs.graphs, keep) if k],
        steps=steps[keep],
        num_rsus=gs.num_rsus,
        name=gs.name,
    )
    rec: dict = {}
    for variant in ("C", "D"):
        trust = predict_trust(st["models"][variant], sub).astype(np.float32)
        in_main = (sub.steps >= TRUST_STEPS[0]) & (sub.steps < TRUST_STEPS[1])
        in_plat = (sub.steps >= PLATEAU[0]) & (sub.steps < PLATEAU[1])
        rec[f"trust_{variant}"] = trust[in_main]
        rec[f"plateau_{variant}"] = trust[in_plat].mean(axis=0)
    assert np.array_equal(sub.steps[(sub.steps < TRUST_STEPS[1])], np.arange(*TRUST_STEPS))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        seed=seed,
        kind=kind,
        rho_nominal=rho,
        placement=placement,
        draw=draw,
        steps=np.arange(*TRUST_STEPS),
        behavior_class=gt.behavior_class,
        backhaul_segment_id=gt.backhaul_segment_id,
        join_step=gt.join_step,
        num_correlated_picks=gt.num_correlated_picks,
        pair_concentration=gt.degraded_pair_concentration(),
        rsu_active_final=observed.rsu_active[-1],
        **rec,
    )
    return key, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = all_tasks(args.seeds)
    print(f"{len(tasks)} scenario evaluations, seeds {args.seeds}, {args.workers} workers", flush=True)
    t0 = time.time()
    counts: dict[str, int] = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.config, args.seeds)) as pool:
        for i, (key, status) in enumerate(pool.imap_unordered(_run, tasks, chunksize=4), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  {counts}  {time.time() - t0:.0f}s", flush=True)
    print(f"done {counts}  {time.time() - t0:.0f}s")
    return 0 if "no-checkpoint" not in counts else 2


if __name__ == "__main__":
    raise SystemExit(main())

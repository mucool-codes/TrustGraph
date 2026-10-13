"""S3: train Variants C and D on identical data and objective; report one held-out scenario.

    python scripts/s3_train.py --config configs/demo.yaml --seed 1

Prerequisites (scenarios are generated once and read from disk, D23):
    python scripts/generate_trace.py --config configs/demo.yaml --seed 1
    bash scripts/s3_make_scenarios.sh configs/demo.yaml 1

This script is EVALUATION-SIDE: it wires the sealed-label diagnostic (`diagnostic.py`)
into the trainer, so it is allowlisted in tests/test_sealing.py. The training path
itself (`train.py`, `dataset.py`, `model.py`) cannot import ground truth. The diagnostic
is printed each epoch and never touches loss, early stopping, or selection.

Data (all Baseline A dispatch, D33/D41 Q2; injection draws per D39, all < 100 per D43):
    train       rho {0, .25, .5, .75, 1} x fraction {.1, .2, .3} x draws {0, 1}   (30)
    validation  rho {0, .5, 1} x fraction {.1, .2, .3}, draw 40                     (9)
    held-out    rho 1.0, fraction 0.2, draw 90 - one scenario, reported only
Validation draws pick the weights and stop training; held-out never does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import SeedChain, load_config, with_overrides  # noqa: E402
from trustgraph.dataset import build_graphset, next_task_labels  # noqa: E402
from trustgraph.diagnostic import (  # noqa: E402
    PLATEAU_WINDOW,
    PRIMARY_WINDOW,
    degraded_vs_reliable_auc,
    make_epoch_diagnostic,
)
from trustgraph.model import build_trust_head, num_parameters  # noqa: E402
from trustgraph.observed import default_observed_path, load_observed  # noqa: E402
from trustgraph.train import (  # noqa: E402
    TrainSettings,
    calibration_table,
    edge_probe,
    mean_bce,
    predict_trust,
    save_checkpoint,
    train,
)

TRAIN_RHO = (0.0, 0.25, 0.5, 0.75, 1.0)
TRAIN_FRACTION = (0.1, 0.2, 0.3)
TRAIN_DRAWS = (0, 1)
VAL_RHO = (0.0, 0.5, 1.0)
VAL_DRAW = 40
DIAG_CELL = (1.0, 0.2, VAL_DRAW)  # per-epoch diagnostic scenario (a validation one)
HELD_OUT = (1.0, 0.2, 90)
STRIDE = 4


def scenario_cfg(base, seed, rho, frac, draw):
    return with_overrides(base, seed=seed, rho=rho, fraction=frac, draw=draw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variants", nargs="+", default=["C", "D"])
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    base = load_config(args.config)
    seed = args.seed
    settings = TrainSettings() if args.max_epochs is None else TrainSettings(max_epochs=args.max_epochs)
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(exist_ok=True)

    t0 = time.time()
    print(f"building graph sets (stride {STRIDE}) ...", flush=True)
    train_graphs = []
    for rho in TRAIN_RHO:
        for frac in TRAIN_FRACTION:
            for draw in TRAIN_DRAWS:
                gs = build_graphset(scenario_cfg(base, seed, rho, frac, draw), args.config, STRIDE)
                train_graphs += gs.graphs
    val_sets = {}
    for rho in VAL_RHO:
        for frac in TRAIN_FRACTION:
            cfg_v = scenario_cfg(base, seed, rho, frac, VAL_DRAW)
            val_sets[(rho, frac)] = (build_graphset(cfg_v, args.config, STRIDE), cfg_v)
    val_graphs = [g for gs, _ in val_sets.values() for g in gs.graphs]
    diag_set, diag_cfg = val_sets[(DIAG_CELL[0], DIAG_CELL[1])]
    diag_path = default_observed_path(args.config, diag_cfg)

    held_cfg = scenario_cfg(base, seed, *HELD_OUT)
    held_set = build_graphset(held_cfg, args.config, stride=1)
    held_path = default_observed_path(args.config, held_cfg)
    held_obs = load_observed(held_path)
    y_all, valid_all = next_task_labels(held_obs)
    print(
        f"  train graphs {len(train_graphs)}  val graphs {len(val_graphs)}  "
        f"held-out graphs {len(held_set)}   ({time.time() - t0:.0f}s)",
        flush=True,
    )

    report = {"seed": seed, "settings": settings.__dict__, "variants": {}}
    for variant in args.variants:
        print("=" * 78)
        print(f"VARIANT {variant}   seed {seed}   settings {settings}")
        print("=" * 78)
        seeds = SeedChain(seed)
        model = build_trust_head(base.model, seeds.torch_seed("model_init"), base.device, variant)
        n_params = num_parameters(model)
        print(f"  parameters: {n_params}")

        # --- diagnostic: visibly separate path; printing only (L4) ---------------
        diagnostic = make_epoch_diagnostic(
            diag_set, diag_path, PRIMARY_WINDOW, label="validation rho=1.0 frac=0.2"
        )
        result = train(
            model, train_graphs, val_graphs, settings, seeds, epoch_callback=diagnostic
        )
        ckpt = save_checkpoint(
            REPO_ROOT / "checkpoints" / f"s3-seed{seed}-{variant}.pt",
            model, variant, seed, settings, result,
        )
        print(
            f"  stopped at epoch {result.stopped_epoch} ({result.converged_by}); "
            f"best epoch {result.best_epoch}; checkpoint {ckpt.name}"
        )

        # --- held-out report ----------------------------------------------------
        trust = predict_trust(model, held_set)
        steps = held_set.steps
        keep = valid_all[steps]
        held_bce = mean_bce(model, held_set.graphs)
        auc_primary = degraded_vs_reliable_auc(trust, steps, held_path, PRIMARY_WINDOW)
        auc_plateau = degraded_vs_reliable_auc(trust, steps, held_path, PLATEAU_WINDOW)
        calib = calibration_table(trust, y_all[steps], keep)
        entry = {
            "parameters": n_params,
            "initial_val_bce": result.initial_val_bce,
            "best_epoch": result.best_epoch,
            "stopped_epoch": result.stopped_epoch,
            "converged_by": result.converged_by,
            "history": result.history,
            "held_out_bce": held_bce,
            "held_out_auc_primary_450_650": auc_primary,
            "held_out_auc_plateau_1100_1200": auc_plateau,
            "calibration": calib,
            "best_val_bce": min(h["val_bce"] for h in result.history),
        }
        if variant == "D":
            entry["edge_probe"] = edge_probe(model, val_graphs, base.topology["num_rsus"])
        report["variants"][variant] = entry

        print(f"\n  HELD-OUT scenario (rho={HELD_OUT[0]}, fraction={HELD_OUT[1]}, draw {HELD_OUT[2]})")
        print(f"    next-task BCE            : {held_bce:.5f}")
        print(f"    diagnostic AUC [450,650) : {auc_primary:.4f}")
        print(f"    diagnostic AUC [1100,1200): {auc_plateau:.4f}")
        print("    calibration (predicted trust bin -> observed next-task success)")
        print("      bin          n    mean_pred  observed")
        for r in calib:
            if r["n"]:
                print(f"      {r['lo']:.1f}-{r['hi']:.1f}  {r['n']:7d}   {r['mean_pred']:.4f}    {r['observed']:.4f}")
        if variant == "D":
            print(f"    edge probe (validation)  : {entry['edge_probe']}")

    if len(report["variants"]) == 2:
        pc, pd = (report["variants"][v]["parameters"] for v in ("C", "D"))
        report["parameter_ratio_D_over_C"] = pd / pc
        print(f"\nparameters: C {pc}, D {pd}, D/C = {pd / pc:.2f}x")

    path = out_dir / f"s3-seed{seed}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}   total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

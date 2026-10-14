"""S6: run ONE closed-loop scenario and write it, its sealed side files and its decision log.

    python scripts/s6_run.py --config configs/demo.yaml --seed 100 --rho 1.0 --fraction 0.2 \
        --draw 7 --variant D --weights-file configs/weights.yaml --log

Writes under scenarios/s6/ (gitignored):
    <stem>.observed.npz      what a deployed system saw (the closed-loop task stream and features)
    <stem>.SEALED.npz        ground truth, as for every scenario (L4)
    <stem>.SEALED-handoff.npz  per-task `handoff_lost` (a vehicle cannot tell it from a drop)
    <stem>.decisions.jsonl   the observable decision log (D51 item 6), with --log

EVALUATION-SIDE (writes the sealed files, reads ground truth for the summary); allowlisted in
tests/test_sealing.py. Reading `--weights-file`: `configs/weights.yaml` (D50 item 4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import load_config, with_overrides  # noqa: E402
from trustgraph.observed import scenario_stem  # noqa: E402
from trustgraph.offload import VARIANTS, build_offload_settings  # noqa: E402
from trustgraph.scenario import build_world  # noqa: E402
from trustgraph.sealed.ground_truth import DEGRADED, sealed_path_for  # noqa: E402
from trustgraph.simulator import run_closed_loop  # noqa: E402
from trustgraph.trace import default_trace_path, load_trace  # noqa: E402

OUT_DIR = REPO_ROOT / "scenarios" / "s6"
# E selects exactly as D, and B has no weights of its own (its trust term is constant): D50.
WEIGHT_ALIAS = {"E": "D", "B": "A"}


def load_weights(path: str | Path, variant: str) -> tuple[float, float, float]:
    table = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["weights"]
    w = table[WEIGHT_ALIAS.get(variant, variant)]
    return float(w["alpha"]), float(w["beta"]), float(w["gamma"])


def summarise(scenario) -> dict:
    """Post-onset task success and where post-onset tasks went (needs the sealed classes)."""
    tasks, gt = scenario.observed.tasks, scenario.ground_truth
    post = tasks.step >= gt.onset_step
    dispatched = tasks.dispatched
    to_degraded = np.zeros(len(tasks), dtype=bool)
    to_degraded[dispatched] = gt.behavior_class[tasks.rsu[dispatched]] == DEGRADED
    n_post = int((post & dispatched).sum())
    return {
        "tasks": len(tasks),
        "success_all": tasks.success_rate(),
        "success_post_onset": tasks.success_rate(post),
        "post_onset_tasks": n_post,
        "post_onset_to_degraded": int((post & to_degraded).sum()),
        "post_onset_share_to_degraded": float((post & to_degraded).sum() / max(n_post, 1)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--draw", type=int, default=None)
    ap.add_argument("--variant", default="A", choices=VARIANTS)
    ap.add_argument("--weights", type=float, nargs=3, metavar=("ALPHA", "BETA", "GAMMA"), default=None)
    ap.add_argument("--weights-file", default=None)
    ap.add_argument("--no-dwell", action="store_true")
    ap.add_argument("--handoff", default=None, choices=("relay", "off"))
    ap.add_argument("--log", action="store_true", help="write the decision log")
    args = ap.parse_args()

    cfg = with_overrides(
        load_config(args.config), seed=args.seed, rho=args.rho, fraction=args.fraction, draw=args.draw
    )
    weights = tuple(args.weights) if args.weights else (
        load_weights(args.weights_file, args.variant) if args.weights_file else None
    )
    overrides = {}
    if args.no_dwell:
        overrides["dwell_filter"] = False
    if args.handoff:
        overrides["handoff"] = args.handoff
    settings = build_offload_settings(
        cfg, variant=args.variant, weights=weights, log_decisions=args.log, **overrides
    )

    world = build_world(cfg)
    trace = load_trace(default_trace_path(args.config, cfg.seed))
    scenario = run_closed_loop(cfg, world, trace, settings)

    stem = f"{scenario_stem(args.config, cfg)}-cl-{args.variant}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obs_path = OUT_DIR / f"{stem}.observed.npz"
    scenario.observed.save(obs_path)
    scenario.ground_truth.save(sealed_path_for(obs_path))
    np.savez_compressed(
        OUT_DIR / f"{stem}.SEALED-handoff.npz", handoff_lost=scenario.extras["handoff_lost"]
    )
    if args.log:
        with (OUT_DIR / f"{stem}.decisions.jsonl").open("w", encoding="utf-8") as fh:
            for rec in scenario.extras["decision_log"]:
                fh.write(json.dumps(rec) + "\n")

    s = summarise(scenario)
    print(f"{stem}  weights (alpha, beta, gamma) = {settings.effective_alpha}, {settings.beta}, {settings.gamma}")
    print(f"  {s}")
    print(f"  engine counters: {scenario.extras['counters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

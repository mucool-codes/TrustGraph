"""Generate a scenario - task stream, observable features, sealed ground truth - to disk.

    python scripts/generate_trace.py    --config configs/demo.yaml
    python scripts/generate_scenario.py --config configs/demo.yaml [--rho R] [--fraction F]

Reads the mobility trace from disk (it never simulates motion itself), runs the
simulator, and writes two files side by side:

    scenarios/<stem>.observed.npz   what a deployed system could see; training reads this
    scenarios/<stem>.SEALED.npz     the ground truth; evaluation only (L4)

It then rebuilds the full graph sequence from the observable file alone and prints its
content hash, which is the reproducibility check: the same (config, seed, rho,
fraction) must give the same hash on every run.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import load_config, with_overrides  # noqa: E402
from trustgraph.observed import default_observed_path, load_observed  # noqa: E402
from trustgraph.scenario import build_snapshot_builder, build_world  # noqa: E402
from trustgraph.sealed.ground_truth import BEHAVIOR_CLASSES, sealed_path_for  # noqa: E402
from trustgraph.simulator import generate_scenario  # noqa: E402
from trustgraph.tasks import TASK_TYPES  # noqa: E402
from trustgraph.trace import default_trace_path, load_trace  # noqa: E402


def graph_sequence_digest(cfg, world, trace, observed) -> str:
    builder = build_snapshot_builder(cfg, world, trace, observed)
    digest = hashlib.sha256()
    for data in builder.snapshots():
        digest.update(data.x.numpy().tobytes())
        digest.update(data.edge_index.numpy().tobytes())
        digest.update(data.edge_attr.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="write a scenario to disk")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--trace", default=None)
    parser.add_argument("--out", default=None, help="observable .npz path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = with_overrides(
        load_config(args.config), seed=args.seed, rho=args.rho, fraction=args.fraction
    )
    trace_path = (
        Path(args.trace) if args.trace else default_trace_path(args.config, cfg.seed)
    )
    out = (
        Path(args.out)
        if args.out
        else default_observed_path(args.config, cfg.seed, cfg.rho, cfg.degraded_fraction)
    )
    sealed = sealed_path_for(out)
    if out.exists() and sealed.exists() and not args.force:
        print(f"scenario already exists: {out}  (pass --force to regenerate)")
        return 0

    trace = load_trace(trace_path)
    world = build_world(cfg)
    scenario = generate_scenario(cfg, world, trace)
    scenario.observed.save(out)
    scenario.ground_truth.save(sealed)

    # Round-trip through disk: the graph sequence is built from what was written.
    observed = load_observed(out)
    gt = scenario.ground_truth
    tasks = observed.tasks

    print(f"wrote {out}")
    print(f"wrote {sealed}   (SEALED - evaluation only)")
    print(f"  seed={cfg.seed}  rho={cfg.rho:g}  fraction={cfg.degraded_fraction:g}")
    print(f"  dispatch policy        : {observed.dispatch_policy}")
    print(f"  tasks generated        : {len(tasks)} ({int(tasks.dispatched.sum())} dispatched)")
    print(f"  deadline success       : {tasks.success_rate():.4f}")
    for i, name in enumerate(TASK_TYPES):
        print(f"    {name:<6}               : {tasks.success_rate(tasks.task_type == i):.4f}")
    counts = {BEHAVIOR_CLASSES[c]: int((gt.behavior_class == c).sum()) for c in range(3)}
    print(f"  behaviour classes      : {counts}")
    print(f"  degraded per segment   : {gt.segment_degraded_count().tolist()}")
    print(f"  correlated picks       : {gt.num_correlated_picks}")
    print(f"  graph sequence sha256  : {graph_sequence_digest(cfg, world, trace, observed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

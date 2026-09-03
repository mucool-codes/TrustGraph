"""Generate a mobility trace and write it to disk.

    python scripts/generate_trace.py --config configs/demo.yaml

This is the only place vehicle motion is simulated. Everything downstream reads the
resulting .npz back (see `trustgraph/trace.py`), so one realisation of the traffic is
shared by graph construction, statistics, plots, and later training and evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import load_config  # noqa: E402
from trustgraph.scenario import build_world, generate_trace  # noqa: E402
from trustgraph.trace import default_trace_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="write a mobility trace to disk")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument(
        "--seed", type=int, default=None, help="override the config's seed"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output .npz path (default: traces/<config stem>-seed<seed>.npz)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing trace instead of leaving it alone",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "seed": args.seed})

    out = Path(args.out) if args.out else default_trace_path(args.config, cfg.seed)
    if out.exists() and not args.force:
        # A trace is a pure function of (config, seed), so an existing one is already
        # correct - but silently reusing it would hide a config edit that should have
        # produced a different trace. Say so and make the overwrite explicit.
        print(f"trace already exists: {out}  (pass --force to regenerate)")
        return 0

    world = build_world(cfg)
    trace = generate_trace(cfg, world)
    trace.save(out)

    print(f"wrote {out}")
    print(
        f"  {trace.num_steps} steps x {trace.num_vehicles} vehicles "
        f"at dt={trace.dt_s:g}s  ({trace.duration_s:.0f} s of traffic)"
    )
    print(f"  source={trace.source}  seed={trace.seed}")
    print(
        f"  mean speed {trace.speeds.mean():.2f} m/s "
        f"({trace.speeds.mean() * 3.6:.1f} km/h)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

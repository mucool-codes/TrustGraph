"""Entry point: python run.py --config configs/demo.yaml

Prints the offloading decision sequence for the configured scenario. Two runs with the
same config produce byte-identical output (Standing Rule 7).

Requires a mobility trace on disk - generate one first with:

    python scripts/generate_trace.py --config configs/demo.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trustgraph.config import load_config  # noqa: E402
from trustgraph.pipeline import format_decisions, run_pipeline  # noqa: E402
from trustgraph.scenario import build_world  # noqa: E402
from trustgraph.trace import default_trace_path, load_trace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="TrustGraph offloading simulation")
    parser.add_argument(
        "--config", required=True, help="path to a YAML config, e.g. configs/demo.yaml"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the config's seed (everything else is unchanged)",
    )
    parser.add_argument(
        "--trace",
        default=None,
        help="path to the mobility trace .npz "
        "(default: traces/<config stem>-seed<seed>.npz)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = type(cfg)(**{**cfg.__dict__, "seed": args.seed})

    trace_path = (
        Path(args.trace) if args.trace else default_trace_path(args.config, cfg.seed)
    )
    trace = load_trace(trace_path)
    world = build_world(cfg)

    decisions = run_pipeline(cfg, trace, world)
    print(format_decisions(cfg, decisions, trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

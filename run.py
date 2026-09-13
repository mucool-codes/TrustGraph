"""Entry point: python run.py --config configs/demo.yaml

Prints the offloading decision sequence for the configured scenario. Two runs with the
same config produce byte-identical output (Standing Rule 7).

Requires a mobility trace and a generated scenario on disk:

    python scripts/generate_trace.py    --config configs/demo.yaml
    python scripts/generate_scenario.py --config configs/demo.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trustgraph.config import load_config, with_overrides  # noqa: E402
from trustgraph.observed import default_observed_path, load_observed  # noqa: E402
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
    parser.add_argument("--rho", type=float, default=None, help="override degradation.rho")
    parser.add_argument(
        "--fraction", type=float, default=None, help="override degradation.fraction"
    )
    parser.add_argument("--draw", type=int, default=None, help="injection draw (D39)")
    parser.add_argument("--cold-start-nodes", type=int, default=None)
    parser.add_argument(
        "--cold-start-placement",
        default=None,
        choices=("uniform", "healthy_segment", "degraded_segment"),
    )
    parser.add_argument(
        "--trace",
        default=None,
        help="path to the mobility trace .npz "
        "(default: traces/<config stem>-seed<seed>.npz)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="path to the observable scenario .npz (default: scenarios/<stem>...)",
    )
    args = parser.parse_args()

    cfg = with_overrides(
        load_config(args.config),
        seed=args.seed,
        rho=args.rho,
        fraction=args.fraction,
        draw=args.draw,
        cold_start_nodes=args.cold_start_nodes,
        cold_start_placement=args.cold_start_placement,
    )

    trace_path = (
        Path(args.trace) if args.trace else default_trace_path(args.config, cfg.seed)
    )
    trace = load_trace(trace_path)
    scenario_path = (
        Path(args.scenario)
        if args.scenario
        else default_observed_path(args.config, cfg)
    )
    observed = load_observed(scenario_path)
    world = build_world(cfg)

    decisions = run_pipeline(cfg, trace, observed, world)
    print(format_decisions(cfg, decisions, trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

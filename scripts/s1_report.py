"""The S1 exit-condition report: summary statistics and the two sanity plots.

    python scripts/generate_trace.py --config configs/demo.yaml
    python scripts/s1_report.py       --config configs/demo.yaml

Reads the trace from disk, rebuilds the world from (config, seed), walks the whole
graph sequence, prints the statistics, and writes the two figures under `figures/`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import load_config  # noqa: E402
from trustgraph.scenario import build_snapshot_builder, build_world  # noqa: E402
from trustgraph.stats import (  # noqa: E402
    coverage_series,
    format_stats,
    scenario_stats,
)
from trustgraph.trace import default_trace_path, load_trace  # noqa: E402
from trustgraph.viz import plot_coverage_over_time, plot_topology  # noqa: E402


def graph_sequence_digest(cfg, world, trace) -> tuple[str, dict]:
    """Walk every snapshot, returning a content hash and a few shape statistics.

    The hash is the reproducibility check the exit condition needs: the same (config,
    seed) must yield byte-identical graph tensors, and comparing one digest is a far
    stronger check than eyeballing a plot.
    """
    builder = build_snapshot_builder(cfg, world, trace)
    digest = hashlib.sha256()
    edge_counts: list[int] = []
    same_segment_edges = 0

    for data in builder.snapshots():
        digest.update(data.x.numpy().tobytes())
        digest.update(data.edge_index.numpy().tobytes())
        digest.update(data.edge_attr.numpy().tobytes())
        edge_counts.append(int(data.edge_index.shape[1]))
        same_segment_edges += int(data.edge_attr[:, 3].sum().item())

    counts = np.asarray(edge_counts)
    return digest.hexdigest(), {
        "snapshots": int(counts.size),
        "mean_directed_edges": float(counts.mean()),
        "min_directed_edges": int(counts.min()),
        "max_directed_edges": int(counts.max()),
        "mean_same_segment_edges": same_segment_edges / max(counts.size, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="S1 statistics and sanity plots")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument(
        "--seed", type=int, default=None, help="override the config's seed"
    )
    parser.add_argument("--trace", default=None, help="path to the trace .npz")
    parser.add_argument(
        "--figures", default="figures", help="directory for the two plots"
    )
    parser.add_argument(
        "--snapshot-timestep",
        type=int,
        default=0,
        help="timestep drawn in the topology figure",
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

    series = coverage_series(trace, world.topology)
    stats = scenario_stats(trace, world.topology)
    digest, graph_info = graph_sequence_digest(cfg, world, trace)

    figures = Path(args.figures)
    topology_png = plot_topology(
        world.topology, trace, figures / "s1_topology.png", args.snapshot_timestep
    )
    coverage_png = plot_coverage_over_time(
        series, trace, figures / "s1_coverage_over_time.png"
    )

    print("=" * 78)
    print("TrustGraph S1 - topology, mobility, and graph sequence")
    print("=" * 78)
    print(f"config                       : {args.config}")
    print(f"seed                         : {cfg.seed}")
    print(f"trace                        : {trace_path}")
    print("")
    print(format_stats(stats))
    print("")
    print("graph sequence")
    print(f"  snapshots                  : {graph_info['snapshots']}")
    print(
        f"  directed edges per snapshot: {graph_info['mean_directed_edges']:.1f} "
        f"(min {graph_info['min_directed_edges']}, "
        f"max {graph_info['max_directed_edges']})"
    )
    print(
        "  same_segment edges         : "
        f"{graph_info['mean_same_segment_edges']:.1f} per snapshot"
    )
    print(f"  sequence sha256            : {digest}")
    print("")
    print("figures")
    print(f"  {topology_png}")
    print(f"  {coverage_png}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

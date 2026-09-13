"""Cold-start control vs test conditions across injection draws (S2b, DECISIONS.md D38).

    python scripts/s2b_cold_start_conditions.py --config configs/demo.yaml

Injector only - no simulation. For each rho and each placement, over seeds x draws:

  * feasible     - fraction of draws where the condition can be placed at all;
  * class        - behaviour class of the placed node (must be pure per condition);
  * degraded nbrs- degraded RSUs among the placed node's same_segment coordination
                   neighbours, i.e. how much segment evidence message passing could
                   carry to the new node (0 for a control node by construction);
  * any same-seg - fraction of placed nodes with at least one same_segment neighbour
                   at all (without one, nothing propagates along a same_segment edge);
  * distinct     - distinct placed RSUs per seed across its draws, the analogue of
                   F10's quantisation for this condition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import SeedChain, load_config, with_overrides  # noqa: E402
from trustgraph.scenario import build_world  # noqa: E402
from trustgraph.sealed.ground_truth import BEHAVIOR_CLASSES  # noqa: E402
from trustgraph.sealed.injector import (  # noqa: E402
    INJECTION_PURPOSES,
    inject,
    injection_stream_name,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--rho", default="0,1")
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--num-nodes", type=int, default=1)
    args = parser.parse_args()

    base = load_config(args.config)
    print(f"config {args.config}   seeds 1..{args.seeds}   draws 0..{args.draws - 1}   "
          f"fraction {args.fraction:g}   cold-start nodes {args.num_nodes}")

    for rho in (float(r) for r in args.rho.split(",")):
        print("")
        print(f"=== rho {rho:g}")
        print("  placement          feasible   classes                         "
              "degraded nbrs (mean)   any same-seg   distinct/seed")
        for placement in ("healthy_segment", "degraded_segment"):
            feasible = 0
            total = 0
            classes: dict[str, int] = {}
            nbrs: list[int] = []
            any_same: list[bool] = []
            distinct: list[int] = []
            for seed in range(1, args.seeds + 1):
                topo = build_world(with_overrides(base, seed=seed)).topology
                seg = topo.backhaul_segment_id
                adjacency = [set() for _ in range(topo.num_rsus)]
                for i, j in topo.rsu_edges:
                    if seg[i] == seg[j]:
                        adjacency[int(i)].add(int(j))
                        adjacency[int(j)].add(int(i))
                chain = SeedChain(seed)
                placed: set[int] = set()
                for draw in range(args.draws):
                    total += 1
                    try:
                        gt = inject(
                            seg,
                            {**base.degradation, "fraction": args.fraction, "rho": rho},
                            {},
                            {"num_nodes": args.num_nodes, "join_step": 600, "placement": placement},
                            rngs={p: chain.generator(injection_stream_name(p, draw)) for p in INJECTION_PURPOSES},
                        )
                    except ValueError:
                        continue
                    feasible += 1
                    for r in np.flatnonzero(gt.cold_start):
                        name = BEHAVIOR_CLASSES[int(gt.behavior_class[r])]
                        classes[name] = classes.get(name, 0) + 1
                        nbrs.append(sum(int(gt.degraded[n]) for n in adjacency[r]))
                        any_same.append(bool(adjacency[r]))
                        placed.add(int(r))
                distinct.append(len(placed))
            print(f"  {placement:<17}  {feasible / total:8.3f}   {str(classes):<30}  "
                  f"{np.mean(nbrs) if nbrs else float('nan'):20.3f}   "
                  f"{np.mean(any_same) if any_same else float('nan'):12.3f}   {distinct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

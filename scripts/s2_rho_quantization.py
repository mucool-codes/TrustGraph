"""How finely does rho actually resolve at each degraded fraction? (S2 task item 6)

    python scripts/s2_rho_quantization.py --config configs/demo.yaml

Runs the injector only - no simulation - over many seeds (each seed re-draws the
topology and the degraded set, as the L12 protocol does) and reports, per degraded
fraction and per rho:

  * K, the number of degraded RSUs;
  * E[concentration] - the mean fraction of degraded-node pairs sharing a backhaul
    segment, the realised correlation;
  * the distribution of degraded-set *shapes*: how many degraded RSUs fell in each
    touched segment, sorted. At small K there are only a handful of shapes, and a
    single seed can only ever produce one of them - that is the quantisation;
  * the concentration each of the evaluation seeds (1..5) actually realises.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import SeedChain, load_config, with_overrides  # noqa: E402
from trustgraph.scenario import build_world  # noqa: E402
from trustgraph.sealed.injector import degraded_count, inject  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--rho", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--fractions", default="0.05,0.10,0.20,0.30")
    args = parser.parse_args()

    base = load_config(args.config)
    rhos = [float(r) for r in args.rho.split(",")]
    fractions = [float(f) for f in args.fractions.split(",")]
    seeds = list(range(1, args.seeds + 1))

    segments = {s: build_world(with_overrides(base, seed=s)).topology.backhaul_segment_id for s in seeds}
    n = next(iter(segments.values())).size

    uniform = []
    for seg in segments.values():
        sizes = np.bincount(seg)
        uniform.append((sizes * (sizes - 1) / 2).sum() / (n * (n - 1) / 2))

    print(f"config {args.config}   RSUs {n}   seeds 1..{args.seeds}")
    print(f"concentration of two independently chosen RSUs (segment geometry only): {np.mean(uniform):.3f}")

    for fraction in fractions:
        k = degraded_count(n, fraction)
        print("")
        print(f"=== fraction {fraction:.2f}  ->  K = {k} degraded RSUs")
        all_shapes: set[tuple[int, ...]] = set()
        rows = []
        for rho in rhos:
            shapes: Counter = Counter()
            conc = []
            per_eval = []
            for s in seeds:
                chain = SeedChain(s)
                gt = inject(
                    segments[s],
                    {**base.degradation, "fraction": fraction, "rho": rho},
                    {},
                    {},
                    rngs={p: chain.generator(p) for p in ("degradation", "collusion", "cold_start")},
                )
                counts = gt.segment_degraded_count()
                shape = tuple(sorted((int(c) for c in counts if c > 0), reverse=True))
                shapes[shape] += 1
                c = gt.degraded_pair_concentration()
                conc.append(c)
                if s <= args.eval_seeds:
                    per_eval.append(c)
            all_shapes |= set(shapes)
            rows.append((rho, conc, shapes, per_eval))

        ordered = sorted(all_shapes, reverse=True)
        header = "  rho   E[conc]  distinct  " + "  ".join(f"{str(list(sh)):>12}" for sh in ordered)
        print(header)
        for rho, conc, shapes, per_eval in rows:
            mean = float("nan") if k < 2 else float(np.mean(conc))
            dist = "  ".join(f"{shapes.get(sh, 0) / len(seeds):12.3f}" for sh in ordered)
            print(f"  {rho:4.2f}  {mean:7.3f}  {len(shapes):8d}  {dist}")
        print(f"  evaluation seeds 1..{args.eval_seeds}: realised concentration per rho")
        for rho, _, _, per_eval in rows:
            vals = " ".join("  nan" if np.isnan(v) else f"{v:5.2f}" for v in per_eval)
            print(f"    rho {rho:4.2f}: {vals}   distinct {len({round(v, 6) for v in per_eval if not np.isnan(v)})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

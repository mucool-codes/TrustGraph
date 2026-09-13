"""The S2 exit-condition report: success_ewma over time, degraded vs healthy, at rho = 0
and at high rho.

    python scripts/generate_trace.py --config configs/demo.yaml
    python scripts/s2_report.py      --config configs/demo.yaml

Reads the trace from disk, generates both scenarios in process (the generator is a pure
function of config, seed and trace), prints the numbers behind the figure, and writes
`figures/s2_success_ewma.png`.

This is evaluation code: it unseals the ground truth to split degraded from healthy RSUs,
which is exactly what PROJECT_SPEC.md 4.4 allows evaluation - and only evaluation - to do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import load_config, with_overrides  # noqa: E402
from trustgraph.scenario import build_world  # noqa: E402
from trustgraph.simulator import generate_scenario  # noqa: E402
from trustgraph.trace import default_trace_path, load_trace  # noqa: E402
from trustgraph.viz import EwmaPanel, plot_success_ewma_over_time  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="S2 success_ewma figure and numbers")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--high-rho", type=float, default=1.0)
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()

    cfg = with_overrides(load_config(args.config), seed=args.seed)
    fraction = cfg.degraded_fraction if args.fraction is None else args.fraction
    trace = load_trace(default_trace_path(args.config, cfg.seed))
    world = build_world(cfg)
    sizes = world.topology.segment_sizes()

    print("=" * 78)
    print("TrustGraph S2 - success_ewma over time, degraded vs healthy")
    print("=" * 78)
    print(f"config {args.config}   seed {cfg.seed}   fraction {fraction:g}   "
          f"segment sizes {sizes.tolist()}")

    baseline = generate_scenario(with_overrides(cfg, fraction=0.0), world, trace)
    print(f"no-degradation deadline success: {baseline.observed.tasks.success_rate():.4f}")

    panels = []
    for rho in (0.0, args.high_rho):
        s = generate_scenario(with_overrides(cfg, fraction=fraction, rho=rho), world, trace)
        obs, gt = s.observed, s.ground_truth
        ewma = obs.feature("success_ewma")
        dev = obs.feature("latency_dev")
        onset, ramp = gt.onset_step, gt.ramp_steps
        degraded = gt.degraded
        touched = np.isin(gt.backhaul_segment_id, gt.backhaul_segment_id[degraded])
        mates = touched & ~degraded
        others = ~touched
        fault = np.array(
            [gt.degradation_level(t)[degraded].mean() for t in range(trace.num_steps)]
        )

        tasks = obs.tasks
        d = tasks.dispatched
        post = tasks.step >= onset + ramp
        on_degraded = np.zeros(len(tasks), dtype=bool)
        on_degraded[d] = degraded[tasks.rsu[d]]

        print("")
        print(f"--- rho = {rho:g}")
        print(f"  degraded RSUs          : {np.flatnonzero(degraded).tolist()}  "
              f"(segments {gt.backhaul_segment_id[degraded].tolist()})")
        print(f"  degraded per segment   : {gt.segment_degraded_count().tolist()} of {sizes.tolist()}")
        print(f"  correlated picks       : {gt.num_correlated_picks}   "
              f"pair concentration {gt.degraded_pair_concentration():.3f}")
        print(f"  success overall        : {tasks.success_rate():.4f}   "
              f"post-ramp {tasks.success_rate(post):.4f}   "
              f"post-ramp on degraded {tasks.success_rate(post & on_degraded):.4f}")
        print("  mean success_ewma        step   degraded   healthy-same-seg   healthy-other")
        checkpoints = (
            ("pre-onset", onset - 1),
            ("onset + 10", onset + 10),
            ("ramp 1/4", onset + ramp // 4),
            ("ramp 1/2", onset + ramp // 2),
            ("ramp end", onset + ramp),
            ("end", trace.num_steps - 1),
        )
        for label, t in checkpoints:
            mate = f"{ewma[t, mates].mean():.3f}" if mates.any() else "  -  "
            print(f"    {label:<12}          {t:5d}      {ewma[t, degraded].mean():.3f}"
                  f"              {mate}           {ewma[t, others].mean():.3f}")
        print(f"  latency_dev at end     : degraded {dev[-1, degraded].mean():.3f}   "
              f"healthy {dev[-1, ~degraded].mean():.3f}")

        panels.append(
            EwmaPanel(
                title=(
                    f"rho = {rho:g}   ({int(degraded.sum())} degraded; per segment "
                    f"{gt.segment_degraded_count().tolist()} of {sizes.tolist()})"
                ),
                success_ewma=ewma,
                degraded=degraded,
                segment_id=gt.backhaul_segment_id,
                fault_progress=fault,
                onset_step=onset,
                ramp_steps=ramp,
                dt_s=trace.dt_s,
            )
        )

    path = plot_success_ewma_over_time(panels, Path(args.figures) / "s2_success_ewma.png")
    print("")
    print(f"figure: {path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

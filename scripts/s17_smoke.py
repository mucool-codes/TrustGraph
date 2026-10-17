"""S17 smoke run: the fault timeline of one spreading-fault and one cold-start scenario (DECISIONS.md D81, exit condition).

    python scripts/s17_smoke.py [--seed 1] [--draw 900] [--lag 80] [--fraction 0.20]

Generates, in memory (nothing is written), one scenario from configs/spread_L<lag>.yaml and one from
configs/restart_f<pct>.yaml for the given seed and a harness draw (>= 900: D81 section 5, not a result) and prints
the sealed timeline of each: per failing RSU its hop, configured onset, the first step its fault is nonzero, the
step it reaches full severity, and the first task silently dropped on it. For the restart it also prints the
observable tracker state around the restart step. No trust head, no variant, no dispatch comparison.
EVALUATION-SIDE: reads the sealed ground truth; allowlisted in tests/test_sealing.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import load_config, with_overrides  # noqa: E402
from trustgraph.features import RSU_COL  # noqa: E402
from trustgraph.scenario import build_world, generate_trace  # noqa: E402
from trustgraph.simulator import generate_scenario  # noqa: E402


def _first(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else -1


def level_matrix(gt, num_steps: int) -> np.ndarray:
    return np.stack([gt.degradation_level(t) for t in range(num_steps)])


def first_drop_step(sc, rsu: int) -> int:
    """Step of the first task silently dropped on `rsu`, -1 if none."""
    hit = np.flatnonzero((sc.observed.tasks.rsu == rsu) & sc.ground_truth.task_dropped)
    return int(sc.observed.tasks.step[hit[0]]) if hit.size else -1


def build(config: str, seed: int, draw: int):
    cfg = with_overrides(load_config(REPO_ROOT / "configs" / config), seed=seed, draw=draw)
    world = build_world(cfg)
    trace = generate_trace(cfg, world)
    return cfg, world, generate_scenario(cfg, world, trace)


def spread_timeline(seed: int, draw: int, lag: int) -> None:
    cfg, world, sc = build(f"spread_L{lag}.yaml", seed, draw)
    gt, n_steps = sc.ground_truth, sc.observed.num_steps
    lvl = level_matrix(gt, n_steps)
    members = np.flatnonzero(gt.degraded)
    members = members[np.lexsort((members, gt.spread_hop[members]))]
    print(f"== spreading fault: configs/spread_L{lag}.yaml  seed {seed}  draw {draw}  rho {gt.rho:g}  fraction {gt.degraded_fraction:g}")
    print(f"   segments of the degraded RSUs: {sorted(set(gt.backhaul_segment_id[members].tolist()))};  lag = {lag} steps per hop;  onset_step = {gt.onset_step}")
    print("   rsu  segment  hop  configured_onset  first_nonzero  full_severity  first_dropped_task  lag_from_seed_(first_nonzero)")
    seed_first = None
    ok = True
    for v in members:
        hop = int(gt.spread_hop[v])
        onset = int(gt.onset_per_rsu()[v])
        nz, full = _first(lvl[:, v] > 0), _first(lvl[:, v] >= 1.0)
        if seed_first is None:
            seed_first = nz
        ok &= onset == gt.onset_step + lag * hop and nz == onset + 1
        print(f"   {int(v):3d}  {int(gt.backhaul_segment_id[v]):7d}  {hop:3d}  {onset:16d}  {nz:13d}  {full:13d}  {first_drop_step(sc, int(v)):18d}  {nz - seed_first:+d}")
    print(f"   check: configured onset = onset_step + lag x hop, and the fault turns on one step later, for every RSU: {ok}")
    healthy = ~gt.degraded
    print(f"   healthy RSUs never faulted: {bool((lvl[:, healthy] == 0).all())}")


def restart_timeline(seed: int, draw: int, fraction: float) -> None:
    pct = int(round(fraction * 100))
    cfg, world, sc = build(f"restart_f{pct}.yaml", seed, draw)
    gt, obs = sc.ground_truth, sc.observed
    lvl = level_matrix(gt, obs.num_steps)
    restarted = np.flatnonzero(gt.restart_steps >= 0)
    ewma = obs.rsu_features[:, :, RSU_COL["success_ewma"]]
    t_r = int(gt.restart_steps[restarted[0]])
    print(f"== cold start: configs/restart_f{pct}.yaml  seed {seed}  draw {draw}  restart step {t_r}  restarted {restarted.size}  failing {int(gt.degraded.sum())}")
    print("   rsu  segment  failing  success_ewma@restart-1  @restart  @restart+1  first_nonzero  full_severity  in_service_throughout")
    for v in restarted:
        failing = bool(gt.degraded[v])
        print(
            f"   {int(v):3d}  {int(gt.backhaul_segment_id[v]):7d}  {str(failing):>7}  {ewma[t_r - 1, v]:22.4f}  {ewma[t_r, v]:8.4f}  {ewma[t_r + 1, v]:10.4f}"
            f"  {_first(lvl[:, v] > 0) if failing else -1:13d}  {_first(lvl[:, v] >= 1.0) if failing else -1:13d}  {bool(obs.rsu_active[:, v].all())}"
        )
    others = np.setdiff1d(np.arange(gt.num_rsus), restarted)
    print(f"   RSUs not restarted: success_ewma at the restart step is untouched (min {ewma[t_r, others].min():.4f}, max {ewma[t_r, others].max():.4f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--draw", type=int, default=900, help="a harness draw >= 900 (D81 section 5); not a result")
    ap.add_argument("--lag", type=int, default=80)
    ap.add_argument("--fraction", type=float, default=0.20)
    args = ap.parse_args()
    if args.draw < 900:
        raise SystemExit("the smoke run uses harness draws >= 900 only (D81 section 5)")
    spread_timeline(args.seed, args.draw, args.lag)
    print()
    restart_timeline(args.seed, args.draw, args.fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

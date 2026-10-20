"""S24 phase 1: the mechanism gate of D89. No model, no trust head, no comparison.

Does an RSU's same-segment neighbourhood, pooled by a mean, show a fault before the RSU's
own `success_ewma` does, when each RSU sees few tasks? D89 sections 2-4, unchanged.

    python scripts/s24_gate.py generate     # scenarios for the 4 configs x seeds 1-5 x draws 700-709
    python scripts/s24_gate.py gate         # the D89 gate table and the PASS/FAIL line

EVALUATION-SIDE (reads the sealed class to pick the degraded RSUs); allowlisted in tests/test_sealing.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import load_config, with_overrides  # noqa: E402
from trustgraph.observed import load_observed, scenario_stem  # noqa: E402
from trustgraph.scenario import build_world  # noqa: E402
from trustgraph.sealed.ground_truth import load_ground_truth, sealed_path_for  # noqa: E402
from trustgraph.simulator import generate_scenario  # noqa: E402
from trustgraph.trace import default_trace_path, load_trace  # noqa: E402

CONFIGS = ("configs/sparse_r02.yaml", "configs/sparse_r04.yaml", "configs/sparse_r10.yaml", "configs/demo.yaml")
GATED = "configs/sparse_r02.yaml"  # D89 section 4: the gate is judged here only
SEEDS = (1, 2, 3, 4, 5)
DRAWS = tuple(range(700, 710))  # D89 section 2: gate only, never a phase-2 evaluation
RHO, FRACTION = 1.0, 0.20
OUT = REPO_ROOT / "scenarios" / "s24"
DEGRADED = 1
ONSET, END, RUN, BASE = 400, 799, 10, slice(100, 400)
T_CRIT_DF4 = 2.776
FLOOR = 20.0


def first_move(series: np.ndarray) -> int | None:
    """The F18 primary rule: first t in [400, 799] below the series' own 1st percentile of [100, 400) for 10 steps."""
    below = series < np.percentile(series[BASE], 1)
    cs = np.concatenate([[0], np.cumsum(below.astype(int))])
    for t in range(ONSET, END + 1):
        if cs[t + RUN] - cs[t] == RUN:
            return t
    return None


def scenario_path(config: str, cfg) -> Path:
    return OUT / f"{scenario_stem(config, cfg)}.observed.npz"


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        base = load_config(config)
        for seed in SEEDS:
            trace_path = default_trace_path(config, seed)
            if not trace_path.exists():
                raise SystemExit(f"missing trace {trace_path}: python scripts/generate_trace.py --config {config} --seed {seed}")
            trace = load_trace(trace_path)
            for draw in DRAWS:
                cfg = with_overrides(base, seed=seed, rho=RHO, fraction=FRACTION, draw=draw)
                out = scenario_path(config, cfg)
                if out.exists() and sealed_path_for(out).exists():
                    continue
                sc = generate_scenario(cfg, build_world(cfg), trace)
                sc.observed.save(out)
                sc.ground_truth.save(sealed_path_for(out))
            print(f"generated {config} seed {seed}", flush=True)


def same_segment_neighbours(topo) -> list[list[int]]:
    seg = topo.backhaul_segment_id
    nbrs: list[list[int]] = [[] for _ in range(topo.num_rsus)]
    for i, j in topo.rsu_edges:
        if seg[i] == seg[j]:
            nbrs[int(i)].append(int(j))
            nbrs[int(j)].append(int(i))
    return nbrs


def tci(values) -> tuple[float, float, float, float]:
    v = np.asarray(values, float)
    m, sd = float(v.mean()), float(v.std(ddof=1))
    h = T_CRIT_DF4 * sd / np.sqrt(v.size)
    return m, sd, m - h, m + h


def gate_for(config: str) -> dict:
    base = load_config(config)
    per_seed = {k: [] for k in ("lead", "own", "pool", "cens_own", "cens_pool", "n")}
    tasks_per_rsu = []
    for seed in SEEDS:
        topo = build_world(with_overrides(base, seed=seed)).topology
        nbrs = same_segment_neighbours(topo)
        lead, own, pool, c_own, c_pool = [], [], [], [], []
        for draw in DRAWS:
            cfg = with_overrides(base, seed=seed, rho=RHO, fraction=FRACTION, draw=draw)
            path = scenario_path(config, cfg)
            obs = load_observed(path)
            ewma = obs.feature("success_ewma")
            tasks_per_rsu.append(int(obs.tasks.dispatched.sum()) / (topo.num_rsus * ewma.shape[0]))
            gt = load_ground_truth(sealed_path_for(path))
            for v in np.flatnonzero(gt.behavior_class == DEGRADED):
                if not nbrs[v]:
                    continue
                t_own = first_move(ewma[:, v])
                t_pool = first_move(ewma[:, nbrs[v]].mean(axis=1))
                o = END if t_own is None else t_own
                p = END if t_pool is None else t_pool
                lead.append(o - p)
                own.append(o - ONSET)
                pool.append(p - ONSET)
                c_own.append(t_own is None)
                c_pool.append(t_pool is None)
        per_seed["lead"].append(np.mean(lead))
        per_seed["own"].append(np.mean(own))
        per_seed["pool"].append(np.mean(pool))
        per_seed["cens_own"].append(np.mean(c_own))
        per_seed["cens_pool"].append(np.mean(c_pool))
        per_seed["n"].append(len(lead))
    return {"per_seed": per_seed, "tasks_per_rsu_step": float(np.mean(tasks_per_rsu))}


def gate() -> int:
    print(f"S24 gate (D89), first number written {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
    print("lead = t_own - t_pool, steps (positive = pooled same-segment neighbours move first); 95% t over 5 seeds")
    print("config               tasks/RSU/s   n(v)   own t-onset   pool t-onset   lead mean [95% CI]          "
          "per-seed lead                          censored own / pool")
    verdict = None
    for config in CONFIGS:
        r = gate_for(config)
        ps = r["per_seed"]
        m, sd, lo, hi = tci(ps["lead"])
        print(f"{Path(config).stem:<20} {r['tasks_per_rsu_step']:10.4f}   {sum(ps['n']):5d}   "
              f"{np.mean(ps['own']):9.1f}     {np.mean(ps['pool']):9.1f}      {m:+7.1f} [{lo:+7.1f}, {hi:+7.1f}]   "
              f"{' '.join(f'{x:+6.1f}' for x in ps['lead'])}   "
              f"{np.mean(ps['cens_own']):.3f} / {np.mean(ps['cens_pool']):.3f}")
        if config == GATED:
            verdict = "PASS" if (m >= FLOOR and lo > 0) else "FAIL"
            gated = (m, lo)
    print(f"\nD89 gate at sparse_r02: mean lead {gated[0]:+.1f} (needs >= {FLOOR:.0f}), lower bound {gated[1]:+.1f} "
          f"(needs > 0) -> {verdict}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("generate", "gate"))
    a = ap.parse_args()
    raise SystemExit(generate() if a.which == "generate" else gate())

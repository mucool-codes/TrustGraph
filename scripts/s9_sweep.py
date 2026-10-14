"""S9: the sweep runner (DECISIONS.md D56). One config, unattended, resumable, one results table.

    python scripts/s9_sweep.py all    --sweep configs/sweep.yaml     # run -> checks -> overhead -> table
    python scripts/s9_sweep.py run    --sweep configs/sweep.yaml     # the grid only (resumable)
    python scripts/s9_sweep.py checks --sweep configs/sweep.yaml     # shared-point + E==D assertions
    python scripts/s9_sweep.py overhead --sweep configs/sweep.yaml   # H4 fidelity sample + H5 timing (serial)
    python scripts/s9_sweep.py status --sweep configs/sweep.yaml     # done / total

The grid: two 1-D sweeps (rho at the fixed fraction, fraction at the fixed rho), the shared operating point
run once. A task is one (rho, fraction, seed, draw); it runs all variants on that paired draw and writes ONE
file, atomically (temp file, then rename). A restart skips the files that exist, so a crash at hour 3 loses
at most the tasks in flight. A stored config hash must match to resume. A failed task is written under
failed/ and retried on resume; whatever is still there at the end is reported (D42), never dropped.

EVALUATION-SIDE: reads the sealed ground truth to score dispatches against the true degraded set. No model
sees it. Allowlisted in tests/test_sealing.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from trustgraph import sweep_metrics as M  # noqa: E402

TUNING_SEED = 100
ONSET_DEFAULT = 400

_STATE: dict = {}


# ------------------------------------------------------------------ config and grid

def load_sweep(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    sw = yaml.safe_load(p.read_text(encoding="utf-8"))
    sw["_path"] = str(p)
    validate(sw)
    return sw


def validate(sw: dict) -> None:
    if TUNING_SEED in sw["seeds"]:
        raise ValueError(f"tuning seed {TUNING_SEED} is excluded from evaluation (D50/D52)")
    if len(sw["seeds"]) != 5:
        raise ValueError("D56 fixes five seeds (the t interval uses df = 4)")
    low = [d for d in sw["draws"] if d < sw["min_eval_draw"]]
    if low:
        raise ValueError(f"draws {low[:5]}... are below min_eval_draw {sw['min_eval_draw']}")
    if len(set(sw["draws"])) != len(sw["draws"]):
        raise ValueError("duplicate draws")
    if 7 in sw["draws"] and TUNING_SEED in sw["seeds"]:
        raise ValueError("the demo draw is excluded")
    assert_shared_point(sw)


def points(sw: dict) -> dict:
    """{(rho, fraction): [sweep names]} in a stable order; the shared point appears once with both names."""
    out: dict = {}
    for rho in sw["rho_sweep"]["rho"]:
        out.setdefault((float(rho), float(sw["rho_sweep"]["fraction"])), []).append("rho")
    for fr in sw["fraction_sweep"]["fraction"]:
        out.setdefault((float(sw["fraction_sweep"]["rho"]), float(fr)), []).append("fraction")
    return out


def shared_point(sw: dict) -> tuple[float, float]:
    return float(sw["fraction_sweep"]["rho"]), float(sw["rho_sweep"]["fraction"])


def assert_shared_point(sw: dict) -> None:
    """D56 section 1(a): both sweep definitions resolve to the same (rho, fraction) key, and it is in both."""
    sp = shared_point(sw)
    in_rho = (sp[0], sp[1]) in {(float(r), float(sw["rho_sweep"]["fraction"])) for r in sw["rho_sweep"]["rho"]}
    in_frac = (sp[0], sp[1]) in {(float(sw["fraction_sweep"]["rho"]), float(f)) for f in sw["fraction_sweep"]["fraction"]}
    if not (in_rho and in_frac):
        raise AssertionError(f"shared operating point {sp} is not on both sweeps")
    names = points(sw)[sp]
    if sorted(names) != ["fraction", "rho"]:
        raise AssertionError(f"shared point {sp} resolves to {names}, expected one key carrying both sweeps")


def config_hash(sw: dict) -> str:
    core = {k: v for k, v in sw.items() if k not in ("workers", "_path")}
    h = hashlib.sha256(json.dumps(core, sort_keys=True).encode())
    for rel in (sw["base_config"], sw["weights_file"]):
        h.update((REPO_ROOT / rel).read_bytes())
    return h.hexdigest()


def out_dir(sw: dict) -> Path:
    return REPO_ROOT / sw["out_dir"]


def task_path(sw: dict, rho: float, fraction: float, seed: int, draw: int, failed: bool = False) -> Path:
    sub = "failed" if failed else "raw"
    return out_dir(sw) / sub / f"rho{rho:.2f}-frac{fraction:.2f}" / f"seed{seed}-draw{draw}.json"


def all_tasks(sw: dict) -> list[tuple]:
    return [
        (rho, fr, seed, draw)
        for (rho, fr) in points(sw)
        for seed in sw["seeds"]
        for draw in sw["draws"]
    ]


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ one task

def _init(sw: dict) -> None:
    import torch

    torch.set_num_threads(1)
    _STATE["sw"] = sw
    _STATE["cache"] = {}


def _seed_state(seed: int) -> dict:
    """World, trace and frozen heads for a seed, built once per worker."""
    cache = _STATE["cache"]
    if seed not in cache:
        from s8_common import load_head  # noqa: F401  (imports the helpers' path setup)
        from trustgraph.config import load_config, with_overrides
        from trustgraph.scenario import build_world
        from trustgraph.trace import default_trace_path, load_trace

        sw = _STATE["sw"]
        base = load_config(REPO_ROOT / sw["base_config"])
        cfg = with_overrides(base, seed=seed)
        cache[seed] = {
            "base": base,
            "world": build_world(cfg),
            "trace": load_trace(default_trace_path(str(REPO_ROOT / sw["base_config"]), seed)),
            "models": {h: load_head(cfg, h) for h in ("C", "D")},
        }
    return cache[seed]


def run_one_variant(sw: dict, base, world, trace, models, variant: str, seed, rho, fraction, draw) -> dict:
    """One closed-loop run and its D56 metrics. Also returns the task arrays for stream comparisons."""
    from s6_run import load_weights
    from trustgraph.config import with_overrides
    from trustgraph.offload import build_offload_settings
    from trustgraph.simulator import run_closed_loop

    cfg = with_overrides(base, seed=seed, rho=rho, fraction=fraction, draw=draw)
    weights = load_weights(REPO_ROOT / sw["weights_file"], variant)
    settings = build_offload_settings(cfg, variant=variant, weights=weights, record_trust=True)
    model = models.get("C" if variant == "C" else "D") if variant in ("C", "D", "E") else None
    t0 = time.perf_counter()
    sc = run_closed_loop(cfg, world, trace, settings, model=model)
    wall = time.perf_counter() - t0

    tasks, gt = sc.observed.tasks, sc.ground_truth
    num_steps = sc.observed.num_steps
    onset = int(gt.onset_step)
    degraded = gt.degraded
    total, deg = M.dispatch_series(tasks.step, tasks.rsu, degraded, num_steps)
    av = M.avoidance(total, deg, onset)
    post = tasks.step >= onset
    trust = sc.extras["trust_history"]
    det = M.detection(trust, sc.observed.rsu_active, degraded, onset)
    c = sc.extras["counters"]
    metrics = {
        "success_post_onset": tasks.success_rate(post),
        "success_all": tasks.success_rate(),
        "post_dispatched": int((post & tasks.dispatched).sum()),
        "n_av": av["n_av"],
        "n_win": av["n_win"],
        "t_avoid": av["t_avoid"],
        "never_avoided": av["never_avoided"],
        "s0": av["s0"],
        "latency": det["latency"],
        "detected_fraction": det["detected_fraction"],
        "tau": det["tau"],
        "counters": {"decisions": c.decisions, "dwell_excluded": c.dwell_excluded,
                     "dwell_fallback": c.dwell_fallback, "relayed": c.relayed, "handoff_lost": c.handoff_lost},
        "wall_s": wall,
    }
    return metrics, {"rsu": tasks.rsu, "met": tasks.met_deadline}, gt


def run_task(task: tuple) -> tuple[tuple, dict]:
    rho, fraction, seed, draw = task
    sw = _STATE["sw"]
    try:
        st = _seed_state(seed)
        per, streams, gt = {}, {}, None
        for v in sw["variants"]:
            m, s, gt = run_one_variant(sw, st["base"], st["world"], st["trace"], st["models"], v, seed, rho, fraction, draw)
            per[v], streams[v] = m, s
        eq = lambda a, b: bool(np.array_equal(streams[a]["rsu"], streams[b]["rsu"])  # noqa: E731
                               and np.array_equal(streams[a]["met"], streams[b]["met"]))
        posts = {per[v]["post_dispatched"] for v in per}
        res = {
            "status": "ok", "rho": rho, "fraction": fraction, "seed": seed, "draw": draw,
            "k_degraded": int(gt.degraded.sum()),
            "pair_concentration": float(gt.degraded_pair_concentration()),
            "num_correlated_picks": int(gt.num_correlated_picks),
            "variants": per,
            "b_equals_a": eq("A", "B") if "A" in streams and "B" in streams else None,
            "post_dispatched_equal": len(posts) == 1,
        }
    except Exception:  # recorded, not dropped (D42); retried on resume
        res = {"status": "failed", "rho": rho, "fraction": fraction, "seed": seed, "draw": draw,
               "error": traceback.format_exc()}
    return task, res


# ------------------------------------------------------------------ run / status

def prepare(sw: dict, fresh: bool) -> Path:
    d = out_dir(sw)
    d.mkdir(parents=True, exist_ok=True)
    man = d / "manifest.json"
    h = config_hash(sw)
    if man.exists() and not fresh:
        old = json.loads(man.read_text(encoding="utf-8"))
        if old["config_hash"] != h:
            raise SystemExit(f"{man} was written by a different config (hash {old['config_hash'][:12]} vs {h[:12]}); "
                             f"refusing to resume. Use a new out_dir or --fresh to start over.")
    else:
        write_atomic(man, json.dumps({"config_hash": h, "sweep": {k: v for k, v in sw.items() if k != "_path"},
                                      "points": [list(p) for p in points(sw)]}, indent=1))
    return d


def pending(sw: dict) -> tuple[list, int]:
    tasks = all_tasks(sw)
    todo = [t for t in tasks if not task_path(sw, *t).exists()]
    return todo, len(tasks)


def cmd_status(sw: dict) -> int:
    todo, total = pending(sw)
    failed = list((out_dir(sw) / "failed").rglob("*.json")) if (out_dir(sw) / "failed").exists() else []
    print(f"{total - len(todo)}/{total} tasks done ({len(todo)} pending, {len(failed)} failed files)")
    return 0


def cmd_run(sw: dict, fresh: bool) -> int:
    prepare(sw, fresh)
    todo, total = pending(sw)
    print(f"[run] {total} tasks ({len(points(sw))} points x {len(sw['seeds'])} seeds x {len(sw['draws'])} draws), "
          f"{total - len(todo)} already done, {len(todo)} to run, {sw['workers']} workers", flush=True)
    if not todo:
        return 0
    t0, done, failed = time.time(), 0, 0
    with mp.get_context("spawn").Pool(int(sw["workers"]), initializer=_init, initargs=(sw,)) as pool:
        for task, res in pool.imap_unordered(run_task, todo, chunksize=1):
            if res["status"] == "ok":
                write_atomic(task_path(sw, *task), json.dumps(res))
                fp = task_path(sw, *task, failed=True)
                if fp.exists():
                    fp.unlink()
            else:
                failed += 1
                write_atomic(task_path(sw, *task, failed=True), json.dumps(res))
            done += 1
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"  {done}/{len(todo)} this session, {failed} failed, {el:.0f}s, "
                      f"eta {el / done * (len(todo) - done):.0f}s", flush=True)
    todo, _ = pending(sw)
    print(f"[run] finished; {len(todo)} tasks still without a result file (failed: reported in the table, D42)", flush=True)
    return 0 if not todo else 2


# ------------------------------------------------------------------ checks (D56 section 1)

def cmd_checks(sw: dict) -> int:
    """(b) the shared cell recomputed serially equals the stored cell bit-for-bit; E's stream equals D's."""
    import torch

    torch.set_num_threads(1)
    _init(sw)
    sp = shared_point(sw)
    draw = sw["draws"][0]
    report = {"shared_point": list(sp), "draw": draw, "cells": [], "e_equals_d": []}
    bad = 0
    for seed in sw["seeds"]:
        p = task_path(sw, sp[0], sp[1], seed, draw)
        if not p.exists():
            raise SystemExit(f"{p} missing: run the grid first")
        stored = json.loads(p.read_text(encoding="utf-8"))
        _, res = run_task((sp[0], sp[1], seed, draw))
        same = res["status"] == "ok" and all(
            _same(stored["variants"][v], res["variants"][v]) for v in stored["variants"]
        )
        report["cells"].append({"seed": seed, "identical": bool(same)})
        bad += int(not same)
        st = _seed_state(seed)
        _, sd, _ = run_one_variant(sw, st["base"], st["world"], st["trace"], st["models"], "D", seed, sp[0], sp[1], draw)
        _, se, _ = run_one_variant(sw, st["base"], st["world"], st["trace"], st["models"], "E", seed, sp[0], sp[1], draw)
        e_eq = bool(np.array_equal(sd["rsu"], se["rsu"]) and np.array_equal(sd["met"], se["met"]))
        report["e_equals_d"].append({"seed": seed, "identical": e_eq})
        bad += int(not e_eq)
        print(f"  seed {seed}: shared cell recompute identical = {same}; E task stream == D's = {e_eq}", flush=True)
    write_atomic(out_dir(sw) / "checks.json", json.dumps(report, indent=1))
    print(f"[checks] {'ALL PASS' if not bad else str(bad) + ' FAILURES'}")
    return 1 if bad else 0


def _same(a: dict, b: dict) -> bool:
    """Metrics equal bit-for-bit, except the wall time."""
    for k in a:
        if k == "wall_s":
            continue
        x, y = a[k], b[k]
        if isinstance(x, float) and np.isnan(x) and isinstance(y, float) and np.isnan(y):
            continue
        if x != y:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("all", "run", "checks", "overhead", "status"))
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--fresh", action="store_true", help="ignore a manifest from a different config")
    a = ap.parse_args()
    sw = load_sweep(a.sweep)
    import torch

    default_threads = int(torch.get_num_threads())  # recorded before anything lowers it (H5 reports both)
    if a.cmd == "status":
        return cmd_status(sw)
    if a.cmd in ("all", "run"):
        rc = cmd_run(sw, a.fresh)
        if rc and a.cmd == "all":
            print("[all] grid incomplete; not continuing. Re-run to resume.")
            return rc
    if a.cmd in ("all", "checks"):
        rc = cmd_checks(sw)
        if rc and a.cmd == "all":
            return rc
    if a.cmd in ("all", "overhead"):
        import s9_overhead

        s9_overhead.run(sw, default_threads)
    if a.cmd == "all":
        import s9_analyse

        s9_analyse.main(["--sweep", a.sweep])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

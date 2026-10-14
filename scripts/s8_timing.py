"""S8 / H5: per-decision timing of the four components, separately (DECISIONS.md D54 section 8).

    python scripts/s8_timing.py

On the same decision sample as the fidelity test (seeds 1-5, 40 decisions each, per head). Per decision:
  (a) trust inference  - one forward pass of the head over the step's graph
  (b) selection        - `select` over the decision's eligible candidates
  (c) Layer 1          - explain_decision + render_text
  (d) Layer 2          - GNNExplainer for the winner and the runner-up, plus the clause construction and occlusions
Run alone (no other job), after a warm-up, one decision at a time, at the default torch thread count and at 1 thread.
EVALUATION-SIDE. Writes results/s8/timing.json and prints the table.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s8_common as C  # noqa: E402
from trustgraph.explain import explain_decision, render_text  # noqa: E402
from trustgraph.explain_layer2 import explain_node, trust_vector  # noqa: E402
from trustgraph.selection import select  # noqa: E402

BUDGET_MS = 100.0  # D54 section 8: the stated assumption


def hardware() -> dict:
    info = {"platform": platform.platform(), "logical_cpus": os.cpu_count(), "torch": torch.__version__,
            "default_torch_threads": torch.get_num_threads(), "device": "cpu"}
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1 | ForEach-Object { $_.Name + ' | ' + "
             "$_.NumberOfCores + ' cores' }); [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"],
            capture_output=True, text=True, timeout=30,
        ).stdout.split("\n")
        info["cpu"] = out[0].strip()
        info["ram_gb"] = out[1].strip()
    except Exception as exc:  # best effort
        info["cpu"] = f"unavailable ({exc})"
    return info


def time_ms(fn):
    t0 = time.perf_counter()
    result = fn()
    return (time.perf_counter() - t0) * 1000.0, result


def stats(v):
    v = np.asarray(v)
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()), "p95": float(np.percentile(v, 95))}


def measure(variant: str, threads: int) -> dict:
    torch.set_num_threads(threads)
    comp = {"trust_inference": [], "selection": [], "layer1": [], "layer2_per_decision": [], "layer2_per_explanation": []}
    for seed in C.SEEDS:
        cfg, world, trace, scenario, model = C.run_variant(variant, seed)
        n = world.topology.num_rsus
        sample = C.sample_decisions(scenario.extras["decision_log"], seed)
        graphs = C.step_graphs(cfg, world, trace, scenario.observed, [r["step"] for r in sample])
        for i, rec in enumerate(sample):
            g = graphs[rec["step"]]
            e = explain_decision(rec)
            ids = [e.winner, e.runner_up.loser]
            cands = [c for c in rec["candidates"] if c["dwell_eligible"]]
            rsus = np.array([c["rsu"] for c in cands])
            lat = np.array([c["latency_norm"] for c in cands])
            load = np.array([c["load"] for c in cands])
            w = rec["weights"]

            def trust():
                return trust_vector(model, g, n)

            def sel(tv):
                return select(rec["step"], rec["vehicle"], rsus, tv[rsus], lat, load, w["alpha"], w["beta"], w["gamma"])

            def l1():
                return render_text(explain_decision(rec))

            def l2():
                each = []
                for v in ids:
                    t, _ = time_ms(lambda v=v: explain_node(model, g, v, n, 0, rec["decision_id"]))
                    each.append(t)
                return each

            if seed == C.SEEDS[0] and i == 0:  # warm-up, discarded
                for _ in range(2):
                    sel(trust()); l1(); l2()
            t_a, tv = time_ms(trust)
            t_b, _ = time_ms(lambda: sel(tv))
            t_c, _ = time_ms(l1)
            t_d, each = time_ms(l2)
            comp["trust_inference"].append(t_a); comp["selection"].append(t_b); comp["layer1"].append(t_c)
            comp["layer2_per_decision"].append(t_d); comp["layer2_per_explanation"] += each
        print(f"  {variant} threads={threads} seed {seed} done", flush=True)
    return {k: stats(v) for k, v in comp.items()}


def main() -> int:
    hw = hardware()
    print("hardware:", hw)
    out = {"hardware": hw, "budget_ms": BUDGET_MS, "results": {}}
    default_threads = torch.get_num_threads()
    for variant in ("D", "C"):
        for threads in (default_threads, 1):
            out["results"][f"{variant}_threads{threads}"] = measure(variant, threads)
    (C.REPO_ROOT / "results" / "s8").mkdir(parents=True, exist_ok=True)
    (C.REPO_ROOT / "results" / "s8" / "timing.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nper-decision times in ms, 200 decisions per head (Layer 2 = winner + runner-up)")
    print("head / torch threads      trust inference        selection          Layer 1          Layer 2 / decision   Layer 2 / explanation")
    for key, r in out["results"].items():
        cells = [f"{r[c]['median']:8.3f} / {r[c]['mean']:8.3f} / {r[c]['p95']:8.3f}" for c in
                 ("trust_inference", "selection", "layer1", "layer2_per_decision", "layer2_per_explanation")]
        print(f"{key:22s} " + "   ".join(cells))
    print("(each cell: median / mean / p95)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""S9 serial stage: H4 fidelity replication and H5 per-component timing (DECISIONS.md D56 section 7).

Called by `s9_sweep.py overhead` after the grid, with nothing else running. Variant E (= D's dispatch
plus Layer 1 and Layer 2), seeds 1-5, the shared operating point, injection draw `overhead_draw` (101;
D54 used 100), 40 decisions per seed from the D54 sampler, winner and runner-up. The D54 procedure is
applied unchanged: one GNNExplainer run (run seed 0), the top unit against one random low-pool unit, both
moved by 0.3, floors 0.20 (trust change) and 0.05 (decision change), no stability repeat.

Timing per decision, four components separately, at the default torch thread count and at 1 thread.

EVALUATION-SIDE (the sealed class of the explained RSU is stored beside each row, for reporting only).
Allowlisted in tests/test_sealing.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s8_common as C  # noqa: E402
import s8_fidelity as F  # noqa: E402
import s8_timing as T  # noqa: E402
from trustgraph.explain import explain_decision, render_text  # noqa: E402
from trustgraph.explain_layer2 import (  # noqa: E402
    attribute,
    explain_node,
    perturb_effect,
    rerun_decision,
    trust_vector,
)
from trustgraph.selection import select  # noqa: E402
from trustgraph.sealed.ground_truth import DEGRADED  # noqa: E402


def collect(sw: dict) -> list[dict]:
    """The sample and the H4 rows for all seeds (single thread: results do not depend on the thread count)."""
    rho, frac = float(sw["fraction_sweep"]["rho"]), float(sw["rho_sweep"]["fraction"])
    rows, checks = [], {"max_trust_diff": 0.0, "decision_mismatch": 0, "decisions": 0}
    samples = {}
    for seed in sw["seeds"]:
        cfg, world, trace, scenario, model = C.run_variant("E", seed, rho=rho, fraction=frac, draw=int(sw["overhead_draw"]))
        n = world.topology.num_rsus
        sample = C.sample_decisions(scenario.extras["decision_log"], seed)
        graphs = C.step_graphs(cfg, world, trace, scenario.observed, [r["step"] for r in sample])
        classes = scenario.ground_truth.behavior_class
        rng_low = np.random.default_rng(C.LOW_RNG + seed)
        for rec in sample:
            g = graphs[rec["step"]]
            base = trust_vector(model, g, n)
            for c in rec["candidates"]:
                if c["trust"] is not None:
                    checks["max_trust_diff"] = max(checks["max_trust_diff"], abs(base[c["rsu"]] - c["trust"]))
            checks["decisions"] += 1
            checks["decision_mismatch"] += int(rerun_decision(rec, base) != rec["selected_rsu"])
            ranked = sorted((c for c in rec["candidates"] if c["dwell_eligible"]), key=lambda c: (-c["score"], c["rsu"]))
            for role, v in (("winner", ranked[0]["rsu"]), ("runner_up", ranked[1]["rsu"])):
                a0 = attribute(model, g, v, n, 0, rec["decision_id"])
                top = a0.top(1)[0]
                pool = a0.low_pool()
                low = pool[int(rng_low.integers(len(pool)))]
                et = perturb_effect(model, g, rec, v, top, n, base)
                el = perturb_effect(model, g, rec, v, low, n, base)
                rows.append({
                    "seed": seed, "decision_id": rec["decision_id"], "step": rec["step"], "role": role, "rsu": v,
                    "degraded": bool(classes[v] == DEGRADED), "trust": float(base[v]),
                    "top": [top.node, top.col], "top_is_own": top.node == v, "low_is_own": low.node == v,
                    "top_trust_changed": et["trust_changed"], "top_decision_changed": et["decision_changed"],
                    "top_trust_delta": et["trust_delta"],
                    "low_trust_changed": el["trust_changed"], "low_decision_changed": el["decision_changed"],
                    "low_trust_delta": el["trust_delta"],
                })
        samples[seed] = len(sample)
        print(f"  [overhead/H4] seed {seed}: {len(sample)} decisions, {len(rows)} explanations so far", flush=True)
    return rows, checks, samples


def timing(sw: dict, threads: int) -> dict:
    """Per-decision component times on the same sample (the S8 H5 procedure)."""
    rho, frac = float(sw["fraction_sweep"]["rho"]), float(sw["rho_sweep"]["fraction"])
    torch.set_num_threads(threads)
    comp = {"trust_inference": [], "selection": [], "layer1": [], "layer2_per_decision": [], "layer2_per_explanation": []}
    warm = False
    for seed in sw["seeds"]:
        cfg, world, trace, scenario, model = C.run_variant("E", seed, rho=rho, fraction=frac, draw=int(sw["overhead_draw"]))
        n = world.topology.num_rsus
        sample = C.sample_decisions(scenario.extras["decision_log"], seed)
        graphs = C.step_graphs(cfg, world, trace, scenario.observed, [r["step"] for r in sample])
        for rec in sample:
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
                    t, _ = T.time_ms(lambda v=v: explain_node(model, g, v, n, 0, rec["decision_id"]))
                    each.append(t)
                return each

            if not warm:  # warm-up, discarded
                for _ in range(2):
                    sel(trust()); l1(); l2()
                warm = True
            t_a, tv = T.time_ms(trust)
            t_b, _ = T.time_ms(lambda: sel(tv))
            t_c, _ = T.time_ms(l1)
            t_d, each = T.time_ms(l2)
            comp["trust_inference"].append(t_a); comp["selection"].append(t_b); comp["layer1"].append(t_c)
            comp["layer2_per_decision"].append(t_d); comp["layer2_per_explanation"] += each
        print(f"  [overhead/H5] threads={threads} seed {seed} done", flush=True)
    return {k: T.stats(v) for k, v in comp.items()}


def h4_summary(rows: list[dict], seeds) -> dict:
    out = {}
    rt = F.rates(rows, seeds=tuple(seeds))
    for kind in ("trust", "decision"):
        tops, lows, gaps = ([x[i] for x in rt[kind]] for i in range(3))
        ct, cl, cg = F.tci(tops), F.tci(lows), F.tci(gaps)
        out[kind] = {"per_seed": rt[kind], "top": ct, "low": cl, "gap": cg, "floor": F.FLOORS[kind]}
    return out


def run(sw: dict, default_threads: int) -> None:
    out = Path(sw["out_dir"])
    out = out if out.is_absolute() else C.REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows, checks, samples = collect(sw)
    h4 = h4_summary(rows, sw["seeds"])
    hw = T.hardware()
    hw["default_torch_threads"] = default_threads
    res = {}
    for threads in sorted({default_threads, 1}):
        res[f"threads{threads}"] = timing(sw, threads)
    (out / "overhead.json").write_text(json.dumps({
        "draw": int(sw["overhead_draw"]), "variant": "E", "hardware": hw, "budget_ms": T.BUDGET_MS, "samples": samples,
        "checks": checks, "h4": h4, "rows": rows, "timing": res,
    }, indent=1), encoding="utf-8")
    print(f"[overhead] wrote {out / 'overhead.json'}")

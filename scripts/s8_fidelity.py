"""S8: the H4 fidelity test, the stability check and the neighbour-clause rate (DECISIONS.md D54).

    python scripts/s8_fidelity.py collect            # runs everything, writes results/s8/raw-<head>.json
    python scripts/s8_fidelity.py report             # prints the RAW tables (no verdicts)
    python scripts/s8_fidelity.py report --verdict   # also applies the D54 criterion

Heads C and D, evaluation seeds 1-5, rho 1.0, fraction 0.2, draw 100, closed loop with the D52 weights.
For each sampled decision the winner's and the runner-up's trust are explained with GNNExplainer (run_seed 0
is the primary; run_seeds 1, 2 are the stability repeats); the top-importance unit and one random low-
importance unit are each perturbed by 0.3 on the same graph. EVALUATION-SIDE: the sealed class of the explained
RSU is recorded next to the result for reporting only; the explanation never sees it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s8_common as C  # noqa: E402
from trustgraph.explain_layer2 import (  # noqa: E402
    TRUST_EPS,
    attribute,
    explain_node,
    perturb_effect,
    rerun_decision,
    trust_vector,
)
from trustgraph.sealed.ground_truth import DEGRADED  # noqa: E402

OUT = C.REPO_ROOT / "results" / "s8"
T_CRIT = 2.776  # df = 4
FLOORS = {"trust": 0.20, "decision": 0.05}


def collect(head: str, variant: str) -> dict:
    rows, checks = [], {"max_trust_diff": 0.0, "decision_mismatch": 0, "decisions": 0}
    for seed in C.SEEDS:
        t0 = time.time()
        cfg, world, trace, scenario, model = C.run_variant(variant, seed)
        num_rsus = world.topology.num_rsus
        sample = C.sample_decisions(scenario.extras["decision_log"], seed)
        graphs = C.step_graphs(cfg, world, trace, scenario.observed, [r["step"] for r in sample])
        classes = scenario.ground_truth.behavior_class
        rng_low = np.random.default_rng(C.LOW_RNG + seed)
        for rec in sample:
            g = graphs[rec["step"]]
            base = trust_vector(model, g, num_rsus)
            for c in rec["candidates"]:  # the rebuilt graph must reproduce what the engine scored
                if c["trust"] is not None:
                    checks["max_trust_diff"] = max(checks["max_trust_diff"], abs(base[c["rsu"]] - c["trust"]))
            checks["decisions"] += 1
            checks["decision_mismatch"] += int(rerun_decision(rec, base) != rec["selected_rsu"])
            ranked = sorted((c for c in rec["candidates"] if c["dwell_eligible"]), key=lambda c: (-c["score"], c["rsu"]))
            for role, v in (("winner", ranked[0]["rsu"]), ("runner_up", ranked[1]["rsu"])):
                a0 = attribute(model, g, v, num_rsus, 0, rec["decision_id"])
                a1 = attribute(model, g, v, num_rsus, 1, rec["decision_id"])
                a2 = attribute(model, g, v, num_rsus, 2, rec["decision_id"])
                top = a0.top(1)[0]
                pool = a0.low_pool()
                low = pool[int(rng_low.integers(len(pool)))]
                et = perturb_effect(model, g, rec, v, top, num_rsus, base)
                el = perturb_effect(model, g, rec, v, low, num_rsus, base)
                ne = explain_node(model, g, v, num_rsus, 0, rec["decision_id"], attribution=a0)
                rows.append({
                    "seed": seed, "decision_id": rec["decision_id"], "step": rec["step"], "role": role, "rsu": v,
                    "degraded": bool(classes[v] == DEGRADED), "trust": float(base[v]),
                    "n_units": len(a0.units), "top": [top.node, top.col], "top_importance": top.importance,
                    "top_is_own": top.node == v, "low": [low.node, low.col], "low_importance": low.importance,
                    "low_is_own": low.node == v,
                    "top_trust_delta": et["trust_delta"], "top_trust_changed": et["trust_changed"],
                    "top_decision_changed": et["decision_changed"],
                    "low_trust_delta": el["trust_delta"], "low_trust_changed": el["trust_changed"],
                    "low_decision_changed": el["decision_changed"],
                    "stab_top": [[a.top(1)[0].node, a.top(1)[0].col] for a in (a0, a1, a2)],
                    "stab_top3": [[[u.node, u.col] for u in a.top(3)] for a in (a0, a1, a2)],
                    "same_segment_share": ne.same_segment_share, "own_share": ne.own_share,
                    "n_same_segment_neighbours": len(ne.same_segment_neighbours),
                    "neighbour_clause": ne.neighbour_clause,
                })
        print(f"  {head} seed {seed}: {len(sample)} decisions, {len(rows)} explanations so far  ({time.time() - t0:.0f}s)", flush=True)
    return {"head": head, "variant": variant, "rows": rows, "checks": checks}


def rates(rows, seeds=C.SEEDS):
    out = {}
    for kind in ("trust", "decision"):
        per_seed = []
        for s in seeds:
            r = [x for x in rows if x["seed"] == s]
            top = float(np.mean([x[f"top_{kind}_changed" if kind == "trust" else "top_decision_changed"] for x in r]))
            low = float(np.mean([x[f"low_{kind}_changed" if kind == "trust" else "low_decision_changed"] for x in r]))
            per_seed.append((top, low, top - low))
        out[kind] = per_seed
    return out


def tci(v):
    v = np.asarray(v, float)
    m, sd = float(v.mean()), float(v.std(ddof=1))
    h = T_CRIT * sd / np.sqrt(v.size)
    return m, sd, m - h, m + h


def pair_agree(tops):
    a = [tops[i] == tops[j] for i in range(3) for j in range(i + 1, 3)]
    return float(np.mean(a))


def jaccard(sets):
    js = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = {tuple(x) for x in sets[i]}, {tuple(x) for x in sets[j]}
            js.append(len(a & b) / len(a | b))
    return float(np.mean(js))


def report(verdict: bool):
    for head in ("D", "C"):
        d = json.loads((OUT / f"raw-{head}.json").read_text())
        rows, chk = d["rows"], d["checks"]
        print("=" * 100)
        print(f"HEAD {head} (Variant {d['variant']}) - {len(rows)} explanations "
              f"({sum(r['role'] == 'winner' for r in rows)} winners, {sum(r['role'] == 'runner_up' for r in rows)} runner-ups), "
              f"{chk['decisions']} decisions")
        print(f"  graph rebuild check: max |rebuilt trust - logged trust| = {chk['max_trust_diff']:.2e}; "
              f"decisions whose re-run differs from the logged selection: {chk['decision_mismatch']}")
        rt = rates(rows)
        for kind in ("trust", "decision"):
            print(f"  H4 fidelity, {kind}-change (perturb one unit by 0.3; trust change >= {TRUST_EPS}):" if kind == "trust"
                  else "  H4 fidelity, decision-change (selected RSU differs after perturbing):")
            print("    seed   top rate  random-low rate   gap")
            for s, (t, l, g) in zip(C.SEEDS, rt[kind]):
                print(f"    {s}      {t:.3f}      {l:.3f}            {g:+.3f}")
            tops, lows, gaps = ([x[i] for x in rt[kind]] for i in range(3))
            ct, cl, cg = tci(tops), tci(lows), tci(gaps)
            print(f"    mean   {ct[0]:.3f} (sd {ct[1]:.3f})   {cl[0]:.3f} (sd {cl[1]:.3f})   gap {cg[0]:+.3f} (sd {cg[1]:.3f}) "
                  f"95% CI [{cg[2]:+.3f}, {cg[3]:+.3f}]")
            if verdict:
                if cg[2] > 0 and cg[0] >= FLOORS[kind]:
                    v = "FAITHFUL"
                elif cg[2] > 0:
                    v = "WEAK"
                else:
                    v = "NOT SUPPORTED"
                print(f"    D54 verdict ({kind}, floor {FLOORS[kind]}): {v}")
        pooled_top = np.mean([r["top_trust_changed"] for r in rows]); pooled_low = np.mean([r["low_trust_changed"] for r in rows])
        print(f"  pooled over explanations: trust-change top {pooled_top:.3f} low {pooled_low:.3f}; decision-change top "
              f"{np.mean([r['top_decision_changed'] for r in rows]):.3f} low {np.mean([r['low_decision_changed'] for r in rows]):.3f}")
        print(f"  top unit is the RSU's own feature: {np.mean([r['top_is_own'] for r in rows]):.3f}; low unit own: "
              f"{np.mean([r['low_is_own'] for r in rows]):.3f}; mean |trust delta| top {np.mean([abs(r['top_trust_delta']) for r in rows]):.3f} "
              f"low {np.mean([abs(r['low_trust_delta']) for r in rows]):.3f}")
        all3 = np.mean([len({tuple(x) for x in r["stab_top"]}) == 1 for r in rows])
        pair = np.mean([pair_agree([tuple(x) for x in r["stab_top"]]) for r in rows])
        node_all = np.mean([len({x[0] for x in r["stab_top"]}) == 1 for r in rows])
        jac = np.mean([jaccard(r["stab_top3"]) for r in rows])
        print(f"  stability over run seeds 0,1,2: top unit identical in all three {all3:.3f}; mean pairwise top-unit agreement "
              f"{pair:.3f}; top node identical in all three {node_all:.3f}; mean pairwise Jaccard of top-3 {jac:.3f}")
        if verdict:
            print(f"    D54: masks are {'STABLE' if all3 >= 0.80 else 'UNSTABLE'} (threshold 0.80 on the all-three top-unit agreement)")
        cl_rate = np.mean([r["neighbour_clause"] for r in rows])
        print(f"  neighbour clause included (same-segment share >= 0.30 and a named neighbour): {cl_rate:.3f} overall; "
              f"mean same-segment share {np.mean([r['same_segment_share'] for r in rows]):.3f}, "
              f"mean own share {np.mean([r['own_share'] for r in rows]):.3f}")
        for label, sel in (("explained RSU degraded", True), ("explained RSU not degraded", False)):
            sub = [r for r in rows if r["degraded"] == sel]
            if sub:
                print(f"    {label}: n={len(sub)}  clause rate {np.mean([r['neighbour_clause'] for r in sub]):.3f}  "
                      f"mean same-segment share {np.mean([r['same_segment_share'] for r in sub]):.3f}  "
                      f"with >=1 same-segment RSU neighbour {np.mean([r['n_same_segment_neighbours'] > 0 for r in sub]):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("collect", "report"))
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "collect":
        torch.set_num_threads(4)
        for head, variant in (("C", "C"), ("D", "D")):
            d = collect(head, variant)
            (OUT / f"raw-{head}.json").write_text(json.dumps(d), encoding="utf-8")
            print(f"wrote raw-{head}.json ({len(d['rows'])} rows)")
    else:
        report(a.verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""S8 exit condition: one full merged explanation for a real segment-correlated degradation decision.

    python scripts/s8_example.py

Selection rule (D54 section 9), fixed before looking: on the demo scenario (seed 100, rho 1.0, fraction 0.2,
injection draw 7, closed loop, Variant D), the FIRST decision at step >= 700 with at least two eligible candidates
whose runner-up is a degraded RSU that shares its backhaul segment with at least one other degraded RSU, and in which
trust is the decisive term of Layer 1. The sealed class is used only to choose the decision, never by the explanation.
The neighbour clause is shown if and only if the attribution supports it (D54 section 7). Writes results/s8/example.txt.
EVALUATION-SIDE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s8_common as C  # noqa: E402
from trustgraph.explain import explain_decision  # noqa: E402
from trustgraph.explain_layer2 import explain_node, render_merged  # noqa: E402
from trustgraph.sealed.ground_truth import DEGRADED  # noqa: E402


def main() -> int:
    cfg, world, trace, scenario, model = C.run_variant("D", 100, draw=7)
    n = world.topology.num_rsus
    gt = scenario.ground_truth
    deg = gt.behavior_class == DEGRADED
    seg = gt.backhaul_segment_id
    qualifying = []
    for rec in scenario.extras["decision_log"]:
        if rec["step"] < 700 or rec["selected_rsu"] is None:
            continue
        if sum(c["dwell_eligible"] for c in rec["candidates"]) < 2:
            continue
        e = explain_decision(rec)
        r = e.runner_up.loser
        if deg[r] and int((deg & (seg == seg[r])).sum()) >= 2 and e.runner_up.decisive == "trust":
            qualifying.append((rec, e))
    rec, e = qualifying[0]
    graph = C.step_graphs(cfg, world, trace, scenario.observed, [rec["step"]])[rec["step"]]
    layer2 = {v: explain_node(model, graph, v, n, 0, rec["decision_id"]) for v in (e.winner, e.runner_up.loser)}
    r = e.runner_up.loser
    head = (
        f"##### Merged explanation - first qualifying decision (D54 section 9): decision {rec['decision_id']}, seed 100, draw 7, "
        f"Variant D; {len(qualifying)} qualifying decisions at step >= 700\n"
        f"(runner-up RSU-{r} is degraded, on backhaul segment {seg[r]} with degraded RSUs "
        f"{[int(i) for i in np.flatnonzero(deg & (seg == seg[r]))]}; sealed class used only to choose the decision)\n"
    )
    text = head + "\n" + render_merged(e, layer2)
    for v, ne in layer2.items():
        text += (f"\n[Layer 2 numbers, RSU-{v}: trust {ne.trust:.3f}, own share {ne.own_share:.3f}, same-segment RSU neighbours "
                 f"{ne.same_segment_neighbours}, same-segment share {ne.same_segment_share:.3f}, clause {ne.neighbour_clause}]")
    out = C.REPO_ROOT / "results" / "s8"
    out.mkdir(parents=True, exist_ok=True)
    (out / "example.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

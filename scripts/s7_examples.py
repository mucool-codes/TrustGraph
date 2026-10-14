"""S7 exit condition: real Layer 1 explanations from the closed-loop demo run (seed 100, draw 7).

    python scripts/s7_examples.py [--log-dir scenarios/s6]

Selection rule, fixed before any explanation was read: in each category, the FIRST logged decision at
step >= 700 (the plateau, when trust has moved), so nothing is picked for how it reads.

    D / trust   runner-up lost on trust          (decisive term = trust)
    D / latency runner-up lost on latency        (decisive term = latency) while the WINNER had lower trust
    C / load    winner chosen despite lower trust, load decisive
    A / any     Variant A: trust did not influence the decision

Writes results/s7/examples.txt. Reads only the observable decision logs (no ground truth).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.explain import explain_decision, iter_log, render_text  # noqa: E402

STEM = "demo-seed100-rho1.00-frac0.20-draw7-cl-{}.decisions.jsonl"
MIN_STEP = 700


def first(log: Path, predicate):
    for rec in iter_log(log):
        if rec["step"] < MIN_STEP or rec["selected_rsu"] is None:
            continue
        e = explain_decision(rec)
        if e.runner_up is not None and predicate(e):
            return e
    raise SystemExit(f"no decision in {log.name} matches")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="scenarios/s6")
    args = ap.parse_args()
    d = Path(args.log_dir)
    picks = [
        ("D - runner-up lost on trust", "D",
         lambda e: e.runner_up.kind == "decisive" and e.runner_up.decisive == "trust"),
        ("D - runner-up lost on latency; the winner had LOWER trust", "D",
         lambda e: e.runner_up.kind == "decisive" and e.runner_up.decisive == "latency"
         and "trust" in e.runner_up.outweighed),
        ("C - winner chosen despite lower trust; load decisive", "C",
         lambda e: e.runner_up.kind == "decisive" and e.runner_up.decisive == "load"
         and "trust" in e.runner_up.outweighed),
        ("A - trust unused", "A", lambda e: e.trust_mode == "unused"),
    ]
    blocks = []
    for title, variant, pred in picks:
        e = first(d / STEM.format(variant), pred)
        blocks.append(f"##### Example: {title}  (decision {e.decision_id}, Variant {variant})\n\n{render_text(e)}\n")
    text = "\n".join(blocks)
    out = REPO_ROOT / "results" / "s7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "examples.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

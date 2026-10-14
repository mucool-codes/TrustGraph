"""Entry point: python explain.py --decision <id> [--log <decisions.jsonl>]

Prints the Layer 1 explanation of one logged offloading decision: why the winner won and why the
runner-up lost, as an exact decomposition of the selection score (DECISIONS.md D53). The decision
log is the S6 closed-loop output, e.g.

    python scripts/s6_run.py --config configs/demo.yaml --seed 100 --rho 1.0 --fraction 0.2 --draw 7 \
        --variant D --weights-file configs/weights.yaml --log
    python explain.py --decision 4120 --log scenarios/s6/demo-seed100-rho1.00-frac0.20-draw7-cl-D.decisions.jsonl

    python explain.py --survey --log <file>      # which decisions are decided by which term

Layer 1 reads only the log: no config, no weights file, no model, no ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trustgraph.explain import (  # noqa: E402
    ExplanationError,
    explain_decision,
    find_decision,
    render_text,
    survey,
)


def default_log() -> Path:
    found = sorted(Path("scenarios/s6").glob("*.decisions.jsonl"))
    if len(found) == 1:
        return found[0]
    raise SystemExit(
        "pass --log <decisions.jsonl>; "
        + ("no decision log found under scenarios/s6/" if not found else "several logs under scenarios/s6/: "
           + ", ".join(p.name for p in found))
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Explain a logged offloading decision (Layer 1).")
    ap.add_argument("--decision", type=int, help="decision id (the task index in the log)")
    ap.add_argument("--log", default=None, help="decision log (jsonl); default: the only one under scenarios/s6/")
    ap.add_argument("--json", action="store_true", help="print the decomposition as JSON instead of text")
    ap.add_argument("--survey", action="store_true", help="count decisions by deciding term and list examples")
    args = ap.parse_args()
    if args.decision is None and not args.survey:
        ap.error("give --decision <id> or --survey")
    log = Path(args.log) if args.log else default_log()

    try:
        if args.survey:
            s = survey(log)
            print(f"{log.name}: {s.total} decisions, {s.no_runner_up} with a single eligible RSU, "
                  f"{s.trust_unused} with trust unused (alpha = 0)")
            for kind in sorted(s.by_kind):
                items = s.by_kind[kind]
                top = sorted(items, key=lambda t: -t[1])[:3]
                print(f"  {kind:18s} {len(items):6d}   largest margins: "
                      + ", ".join(f"#{i} ({m:.3f})" for i, m in top))
            top = sorted(s.lower_trust_winner, key=lambda t: -t[1])[:3]
            print(f"  winner has LOWER trust than the runner-up: {len(s.lower_trust_winner)}   "
                  + ", ".join(f"#{i} ({m:.3f})" for i, m in top))
            return 0
        explanation = explain_decision(find_decision(log, args.decision))
    except (KeyError, ExplanationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(explanation.to_dict(), indent=1) if args.json else render_text(explanation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

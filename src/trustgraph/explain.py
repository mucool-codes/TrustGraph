"""Explanation Layer 1: an EXACT, contrastive decomposition of the linear selection score (L11).

The selection rule is `score(v) = alpha*trust_v - beta*latency_v - gamma*load_v` (L1), so a decision's
winner-versus-runner-up margin is an identity in three signed differences. This module reads one record of
the S6 decision log and turns that identity into an operator-readable statement of why the winner won and
why the runner-up lost (DECISIONS.md D53).

Rules this module keeps:

  * Everything comes from the logged record - the weights, each candidate's trust, latency and load. Nothing
    is read from the config, so a later change to the weights cannot alter an old explanation.
  * The decomposition is verified, not trusted: every term is recomputed from the logged inputs and must equal
    the logged term and the logged score exactly, and the selected node must be the rule's argmax over the
    dwell-eligible candidates. Otherwise `ExplanationError`.
  * It explains the SCORE. Why trust has the value it has is Layer 2's question (S8); the neighbourhood clause
    is not produced here, and the logged own-node record is shown as context with no causal claim.
  * Variants without a trust story say so: alpha = 0 (Variant A) - trust did not influence the decision; a
    constant trust term (Variant B) - trust did not discriminate.

No ground truth is read or imported (L4): the log is observable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

SUPPORTED_SCHEMA = 2
TERMS: tuple[str, ...] = ("trust", "latency", "load")
# A term decides the margin if it is positive and at least this share of it (D53 item 4).
DOMINANCE_SHARE = 0.6
TASK_NAMES = ("light", "medium", "heavy")
TIE_EPS = 1e-12


class ExplanationError(ValueError):
    """The record does not decompose exactly, or is not a log record this module can read."""


@dataclass(frozen=True)
class CandidateTerms:
    rsu: int
    segment: int
    eligible: bool
    trust: float | None
    latency_norm: float
    latency_ms: float
    load: float
    trust_term: float  # alpha * trust, or 0.0 when trust is unused
    latency_term: float  # -beta * latency_norm
    load_term: float  # -gamma * load
    score: float
    success_ewma: float
    latency_dev: float
    dwell_estimate_s: float | None
    advertised_completion_s: float

    def term(self, name: str) -> float:
        return {"trust": self.trust_term, "latency": self.latency_term, "load": self.load_term}[name]


@dataclass(frozen=True)
class Contrast:
    """Winner versus one loser: `margin = d_trust + d_latency + d_load`, exactly."""

    loser: int
    margin: float
    d_trust: float
    d_latency: float
    d_load: float
    kind: str  # "decisive" | "mixed" | "tie"
    decisive: str | None  # the deciding term when kind == "decisive"
    leading: tuple[str, ...]  # terms with a positive difference, largest first
    outweighed: tuple[str, ...]  # terms that favoured the loser

    def diff(self, name: str) -> float:
        return {"trust": self.d_trust, "latency": self.d_latency, "load": self.d_load}[name]

    def share(self, name: str) -> float | None:
        return None if self.margin <= TIE_EPS else self.diff(name) / self.margin


@dataclass(frozen=True)
class Explanation:
    decision_id: int
    step: int
    time_s: float
    vehicle: int
    task_type: int
    deadline_s: float
    variant: str
    alpha: float
    beta: float
    gamma: float
    trust_mode: str  # "unused" (alpha = 0) | "constant" (same for every contender) | "active"
    winner: int
    candidates: tuple[CandidateTerms, ...]  # every in-range RSU, in logged order
    contrasts: tuple[Contrast, ...]  # winner vs each eligible loser, best loser first
    excluded: tuple[tuple[int, str], ...]  # (rsu, reason): in range but dwell-excluded
    dwell_fallback: bool
    outcome: dict | None

    @property
    def runner_up(self) -> Contrast | None:
        return self.contrasts[0] if self.contrasts else None

    def candidate(self, rsu: int) -> CandidateTerms:
        return next(c for c in self.candidates if c.rsu == rsu)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- decomposition


def _terms(record: dict, cand: dict) -> CandidateTerms:
    w = record["weights"]
    alpha, beta, gamma = float(w["alpha"]), float(w["beta"]), float(w["gamma"])
    trust = cand["trust"]
    t_term = alpha * trust if trust is not None else 0.0
    l_term = -beta * cand["latency_norm"]
    g_term = -gamma * cand["load"]
    score = t_term + l_term + g_term  # the order the engine sums in
    logged = (cand["trust_term"] if cand["trust_term"] is not None else 0.0, cand["latency_term"], cand["load_term"])
    if (t_term, l_term, g_term) != logged:
        raise ExplanationError(
            f"decision {record['decision_id']} RSU {cand['rsu']}: terms recomputed from the logged inputs "
            f"{(t_term, l_term, g_term)} differ from the logged terms {logged}"
        )
    if score != cand["score"]:
        raise ExplanationError(
            f"decision {record['decision_id']} RSU {cand['rsu']}: recomputed score {score!r} != logged {cand['score']!r}"
        )
    f = cand["features"]
    return CandidateTerms(
        rsu=int(cand["rsu"]),
        segment=int(cand["segment"]),
        eligible=bool(cand["dwell_eligible"]),
        trust=None if trust is None else float(trust),
        latency_norm=float(cand["latency_norm"]),
        latency_ms=float(cand["latency_ms"]),
        load=float(cand["load"]),
        trust_term=t_term,
        latency_term=l_term,
        load_term=g_term,
        score=score,
        success_ewma=float(f["success_ewma"]),
        latency_dev=float(f["latency_dev"]),
        dwell_estimate_s=cand["dwell_estimate_s"],
        advertised_completion_s=float(cand["advertised_completion_s"]),
    )


def contrast(winner: CandidateTerms, loser: CandidateTerms) -> Contrast:
    d = {n: winner.term(n) - loser.term(n) for n in TERMS}
    margin = winner.score - loser.score
    leading = tuple(sorted((n for n in TERMS if d[n] > TIE_EPS), key=lambda n: -d[n]))
    outweighed = tuple(sorted((n for n in TERMS if d[n] < -TIE_EPS), key=lambda n: d[n]))
    if margin <= TIE_EPS:
        kind, decisive = "tie", None
    elif leading and d[leading[0]] / margin >= DOMINANCE_SHARE:
        kind, decisive = "decisive", leading[0]
    else:
        kind, decisive = "mixed", None
    return Contrast(
        loser=loser.rsu, margin=margin, d_trust=d["trust"], d_latency=d["latency"], d_load=d["load"],
        kind=kind, decisive=decisive, leading=leading, outweighed=outweighed,
    )


def explain_decision(record: dict) -> Explanation:
    """Decompose one logged decision. Raises `ExplanationError` if it does not verify exactly."""
    if record.get("schema") != SUPPORTED_SCHEMA:
        raise ExplanationError(
            f"decision log schema {record.get('schema')!r} is not supported (need {SUPPORTED_SCHEMA}); "
            "regenerate the log with the current engine"
        )
    if record.get("outcome") is None:
        raise ExplanationError(f"decision {record['decision_id']} has no outcome recorded")
    if record["selected_rsu"] is None:
        raise ExplanationError(f"decision {record['decision_id']} selected no RSU")

    cands = tuple(_terms(record, c) for c in record["candidates"])
    eligible = [c for c in cands if c.eligible]
    # The rule: argmax score, ties to the lowest RSU index (selection.select).
    ranked = sorted(eligible, key=lambda c: (-c.score, c.rsu))
    if not ranked or ranked[0].rsu != record["selected_rsu"]:
        raise ExplanationError(
            f"decision {record['decision_id']}: logged selection RSU {record['selected_rsu']} is not the "
            f"rule's argmax over the eligible candidates"
        )
    runner = record["runner_up_rsu"]
    if (runner is None) != (len(ranked) == 1) or (runner is not None and ranked[1].rsu != runner):
        raise ExplanationError(f"decision {record['decision_id']}: logged runner-up does not match the ranking")

    winner = ranked[0]
    alpha = float(record["weights"]["alpha"])
    if alpha == 0.0:
        mode = "unused"
    elif len({c.trust_term for c in eligible}) == 1:
        mode = "constant"
    else:
        mode = "active"

    excluded = tuple(
        (c.rsu, f"the vehicle was expected to leave its coverage in {c.dwell_estimate_s:.2f} s, before the "
                f"result was due ({c.advertised_completion_s:.2f} s)")
        for c in cands if not c.eligible
    )
    w = record["weights"]
    return Explanation(
        decision_id=int(record["decision_id"]), step=int(record["step"]), time_s=float(record["time_s"]),
        vehicle=int(record["vehicle"]), task_type=int(record["task_type"]), deadline_s=float(record["deadline_s"]),
        variant=str(record["variant"]), alpha=alpha, beta=float(w["beta"]), gamma=float(w["gamma"]),
        trust_mode=mode, winner=winner.rsu, candidates=cands,
        contrasts=tuple(contrast(winner, c) for c in ranked[1:]),
        excluded=excluded, dwell_fallback=bool(record["dwell_fallback"]), outcome=record["outcome"],
    )


# ------------------------------------------------------------------------------ rendering


def _rsu(n: int) -> str:
    return f"RSU-{n}"


def _ahead(name: str, w: CandidateTerms, l: CandidateTerms) -> str:
    """How the winner stood against the loser on one term, naming both."""
    W, L = _rsu(w.rsu), _rsu(l.rsu)
    if name == "trust":
        return f"{W}'s trust was {w.trust:.2f} against {L}'s {l.trust:.2f}"
    if name == "latency":
        return f"{W}'s link latency was {w.latency_ms:.1f} ms against {L}'s {l.latency_ms:.1f} ms"
    return f"{W}'s advertised load was {w.load:.2f} against {L}'s {l.load:.2f}"


def _behind(name: str, w: CandidateTerms, l: CandidateTerms) -> str:
    W, L = _rsu(w.rsu), _rsu(l.rsu)
    if name == "trust":
        return f"{L} had higher trust ({l.trust:.2f} against {w.trust:.2f} for {W})"
    if name == "latency":
        return f"{L} had lower link latency ({l.latency_ms:.1f} ms against {w.latency_ms:.1f} ms for {W})"
    return f"{L} had lower advertised load ({l.load:.2f} against {w.load:.2f} for {W})"


def _worth(c: Contrast, name: str) -> str:
    share = c.share(name)
    if share <= 1.0:
        return f"worth {c.diff(name):+.3f} of the {c.margin:.3f} score margin ({share:.0%})"
    return (f"worth {c.diff(name):+.3f}, more than the whole {c.margin:.3f} score margin "
            f"(the terms below offset part of it)")


def contrast_sentence(e: Explanation, c: Contrast) -> str:
    w, l = e.candidate(e.winner), e.candidate(c.loser)
    if c.kind == "tie":
        return (f"{_rsu(e.winner)} and {_rsu(c.loser)} scored the same ({w.score:.3f}); the tie goes to the "
                f"lower RSU index.")
    parts: list[str] = []
    if c.kind == "decisive":
        n = c.decisive
        label = "advertised load" if n == "load" else n
        parts.append(f"{_rsu(c.loser)} lost on {label}: {_ahead(n, w, l)}, {_worth(c, n)}.")
        rest = [t for t in TERMS if t != n and t not in c.outweighed and abs(c.diff(t)) > TIE_EPS]
        if rest:
            parts.append("The rest of the margin came from " + ", ".join(f"{t} ({c.diff(t):+.3f})" for t in rest) + ".")
    else:
        top = c.leading[:2]
        parts.append(
            f"No single term decided it: {_rsu(e.winner)} led by {c.margin:.3f} overall, from "
            + " and ".join(f"{t} ({c.diff(t):+.3f}, {c.share(t):.0%})" for t in top)
            + f" ({'; '.join(_ahead(t, w, l) for t in top)})."
        )
    for n in c.outweighed:
        parts.append(f"{_behind(n, w, l)}, worth {-c.diff(n):.3f} to {_rsu(c.loser)}, but that was outweighed.")
    return " ".join(parts)


def trust_note(e: Explanation) -> str:
    if e.trust_mode == "unused":
        return (f"Trust did not influence this decision: Variant {e.variant} scores on latency and load only "
                f"(alpha = 0).")
    if e.trust_mode == "constant":
        return ("Trust was the same for every candidate, so it did not discriminate between them "
                f"(Variant {e.variant}).")
    return ""


def render_text(e: Explanation) -> str:
    w = e.candidate(e.winner)
    task = TASK_NAMES[e.task_type] if 0 <= e.task_type < len(TASK_NAMES) else f"type {e.task_type}"
    out = [
        f"Decision {e.decision_id}: step {e.step} (t = {e.time_s:.0f} s), vehicle {e.vehicle}, {task} task, "
        f"deadline {e.deadline_s:.2f} s - Variant {e.variant}",
    ]
    if e.runner_up is None:
        out.append(f"{_rsu(e.winner)} was selected: it was the only eligible RSU, so there was no alternative to contrast.")
    else:
        out.append(f"{_rsu(e.winner)} was selected over {_rsu(e.runner_up.loser)} "
                   f"(score {w.score:.3f} vs {e.candidate(e.runner_up.loser).score:.3f}, margin {e.runner_up.margin:.3f}).")
        out.append("Why: " + contrast_sentence(e, e.runner_up))
    note = trust_note(e)
    if note:
        out.append(note)
    out.append("")
    if e.alpha == 0.0:
        out.append(f"Score = - {e.beta:g} * latency - {e.gamma:g} * load   (weights as logged; alpha = 0, trust unused)")
    else:
        out.append(f"Score = {e.alpha:g} * trust - {e.beta:g} * latency - {e.gamma:g} * load   (weights as logged)")
    out.append("   RSU      trust  alpha*trust  latency  -beta*lat   load  -gamma*load     score")
    for c in sorted(e.candidates, key=lambda c: (-c.score, c.rsu)):
        mark = "*" if c.rsu == e.winner else (" " if c.eligible else "x")
        tr = "  n/a" if c.trust is None else f"{c.trust:5.3f}"
        out.append(
            f"  {mark} {c.rsu:3d}   {tr}   {c.trust_term:+9.4f}   {c.latency_norm:6.3f}   {c.latency_term:+8.4f}  "
            f"{c.load:5.3f}   {c.load_term:+9.4f}  {c.score:+8.4f}"
        )
    out.append("  (* selected, x excluded by the dwell filter)")
    others = e.contrasts[1:]
    if others:
        out.append("")
        out.append("Other candidates:")
        for c in others:
            out.append("  " + contrast_sentence(e, c))
    for rsu, why in e.excluded:
        out.append(f"{_rsu(rsu)} was in range but excluded: {why}.")
    if e.dwell_fallback:
        out.append("No in-range RSU passed the dwell filter, so every in-range RSU was kept as a candidate.")
    out.append("")
    ctx = [f"{_rsu(c.rsu)}: promises kept {c.success_ewma:.2f}, latency deviation {c.latency_dev:.2f}"
           for c in sorted(e.candidates, key=lambda c: (c.rsu != e.winner, c.rsu))]
    out.append("Logged behaviour record (context, not an attribution): " + "; ".join(ctx) + ".")
    out.append("Why a node's trust has its value is not explained here (Layer 2).")
    if e.outcome:
        o = e.outcome
        verdict = "met" if o["met_deadline"] else "missed"
        out.append(f"Outcome: deadline {verdict} (observed {o['observed_s']:.3f} s, advertised {o['advertised_s']:.3f} s"
                   f"{', result relayed after a handoff' if o['relayed'] else ''}).")
    return "\n".join(out)


# ----------------------------------------------------------------------------- log access


def iter_log(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def find_decision(path: str | Path, decision_id: int) -> dict:
    for rec in iter_log(path):
        if rec["decision_id"] == decision_id:
            return rec
    raise KeyError(
        f"decision {decision_id} is not in {path}: only tasks with at least one in-range RSU are logged"
    )


@dataclass
class Survey:
    total: int = 0
    no_runner_up: int = 0
    by_kind: dict = field(default_factory=dict)  # "decisive:latency" | "mixed" | "tie" -> [ids]
    lower_trust_winner: list = field(default_factory=list)
    trust_unused: int = 0


def survey(path: str | Path) -> Survey:
    s = Survey()
    for rec in iter_log(path):
        if rec["selected_rsu"] is None:
            continue
        e = explain_decision(rec)
        s.total += 1
        s.trust_unused += int(e.trust_mode == "unused")
        if e.runner_up is None:
            s.no_runner_up += 1
            continue
        c = e.runner_up
        s.by_kind.setdefault(f"decisive:{c.decisive}" if c.kind == "decisive" else c.kind, []).append(
            (e.decision_id, c.margin)
        )
        if e.trust_mode == "active" and "trust" in c.outweighed:
            s.lower_trust_winner.append((e.decision_id, c.margin))
    return s

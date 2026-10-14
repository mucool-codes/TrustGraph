"""S7: Layer 1 - the exact, contrastive decomposition of the selection score."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from trustgraph.explain import (
    DOMINANCE_SHARE,
    ExplanationError,
    explain_decision,
    iter_log,
    render_text,
    survey,
)
from trustgraph.model import build_trust_head
from trustgraph.offload import OffloadSettings
from trustgraph.simulator import run_closed_loop


def _log(cfg, world, trace, variant, weights=(1.0, 0.5, 0.3)):
    head = {"C": "C", "D": "D", "E": "D"}.get(variant)
    model = build_trust_head(cfg.model, 0, "cpu", head).eval() if head else None
    s = OffloadSettings(variant=variant, alpha=weights[0], beta=weights[1], gamma=weights[2], log_decisions=True)
    return run_closed_loop(cfg, world, trace, s, model=model).extras["decision_log"]


def _record(alpha, beta, gamma, specs, variant="D", selected=None, runner=None, excluded=()):
    """A hand-made schema-2 record. specs: (rsu, trust, latency_norm, load). Terms are built the way the
    engine builds them, so the record is internally exact unless a test tampers with it."""
    cands = []
    for rsu, trust, lat, load in specs:
        t = alpha * trust if trust is not None else None
        l, g = -beta * lat, -gamma * load
        cands.append(
            {
                "rsu": rsu, "segment": 0,
                "features": {"load": load, "queue_depth": 0.0, "cert_valid": 1.0, "success_ewma": 0.9,
                             "latency_dev": 0.1, "uptime_stability": 1.0},
                "distance_m": 100.0, "latency_ms": 25.0 * lat, "latency_norm": lat,
                "dwell_estimate_s": 5.0 if rsu not in excluded else 0.01,
                "advertised_completion_s": 0.1, "dwell_eligible": rsu not in excluded,
                "trust": trust, "trust_term": t, "latency_term": l, "load": load, "load_term": g,
                "score": float((t or 0.0) + l + g),
            }
        )
    ok = sorted([c for c in cands if c["dwell_eligible"]], key=lambda c: (-c["score"], c["rsu"]))
    return {
        "schema": 2, "decision_id": 7, "step": 10, "time_s": 10.0, "vehicle": 3, "task_type": 1, "cycles": 1e9,
        "deadline_s": 0.3, "variant": variant, "trust_source": "x",
        "weights": {"alpha": alpha, "beta": beta, "gamma": gamma}, "dwell_filter": True,
        "dwell_fallback": False, "candidates": cands,
        "selected_rsu": ok[0]["rsu"] if selected is None else selected,
        "runner_up_rsu": (ok[1]["rsu"] if len(ok) > 1 else None) if runner is None else runner,
        "outcome": {"advertised_s": 0.1, "observed_s": 0.12, "met_deadline": True, "relayed": False},
    }


# ----------------------------------------------- exactness on real logged decisions


@pytest.mark.parametrize("variant", ["A", "B", "B_oracle", "C", "D", "E"])
def test_every_logged_decision_decomposes_exactly(cfg, world, trace, variant):
    log = _log(cfg, world, trace, variant)
    assert log
    for rec in log:
        if rec["selected_rsu"] is None:
            continue
        e = explain_decision(rec)  # raises unless the recomputed terms/score equal the logged ones exactly
        for c in e.candidates:
            assert c.trust_term + c.latency_term + c.load_term == c.score  # exact, not approx
        for k in rec["candidates"]:
            c = e.candidate(k["rsu"])
            assert (c.trust_term, c.latency_term, c.load_term) == (
                k["trust_term"] or 0.0, k["latency_term"], k["load_term"])
        w = e.candidate(e.winner)
        for con in e.contrasts:
            l = e.candidate(con.loser)
            assert w.score - l.score == pytest.approx(con.d_trust + con.d_latency + con.d_load, abs=1e-12)
            assert con.margin >= 0.0
        assert e.runner_up is None or e.runner_up.loser == rec["runner_up_rsu"]


@pytest.mark.parametrize("variant", ["C", "D"])
def test_a_c_head_and_a_d_head_decision_render_with_their_logged_weights(cfg, world, trace, variant):
    log = _log(cfg, world, trace, variant, weights=(1.7, 0.4, 0.2))
    rec = next(r for r in log if len(r["candidates"]) > 1)
    e = explain_decision(rec)
    assert (e.alpha, e.beta, e.gamma) == (1.7, 0.4, 0.2)
    text = render_text(e)
    assert f"Variant {variant}" in text and "1.7 * trust - 0.4 * latency - 0.2 * load" in text
    assert "Layer 2" in text and "neighbour" not in text.lower()  # no neighbourhood clause in Layer 1


def test_the_weights_come_from_the_log_not_from_the_config(cfg):
    rec = _record(7.0, 0.1, 0.9, [(1, 0.5, 0.2, 0.3), (2, 0.4, 0.1, 0.2)])
    assert cfg.selection["alpha"] != 7.0
    e = explain_decision(rec)
    assert (e.alpha, e.beta, e.gamma) == (7.0, 0.1, 0.9)
    assert e.candidate(1).trust_term == 7.0 * 0.5


# ------------------------------------------------------ the check is a real check


def test_a_tampered_term_score_or_selection_is_refused():
    rec = _record(1.0, 0.5, 0.3, [(1, 0.9, 0.2, 0.3), (2, 0.4, 0.1, 0.2)])
    explain_decision(rec)

    bad = copy.deepcopy(rec)
    bad["candidates"][0]["latency_term"] += 1e-9
    with pytest.raises(ExplanationError, match="differ from the logged terms"):
        explain_decision(bad)

    bad = copy.deepcopy(rec)
    bad["candidates"][1]["score"] += 1e-9
    with pytest.raises(ExplanationError, match="score"):
        explain_decision(bad)

    bad = copy.deepcopy(rec)
    bad["weights"]["beta"] = 0.6  # a config change must not silently re-explain an old record
    with pytest.raises(ExplanationError):
        explain_decision(bad)

    bad = copy.deepcopy(rec)
    bad["selected_rsu"], bad["runner_up_rsu"] = 2, 1
    with pytest.raises(ExplanationError, match="argmax"):
        explain_decision(bad)

    bad = copy.deepcopy(rec)
    bad["schema"] = 1
    with pytest.raises(ExplanationError, match="schema"):
        explain_decision(bad)


# --------------------------------------------------------- contrastive wording


def test_trust_decisive_runner_up_lost_on_trust():
    rec = _record(2.0, 0.25, 0.6, [(1, 0.95, 0.5, 0.5), (2, 0.30, 0.5, 0.5)])
    e = explain_decision(rec)
    c = e.runner_up
    assert (c.kind, c.decisive, c.loser) == ("decisive", "trust", 2)
    assert c.share("trust") == pytest.approx(1.0)
    text = render_text(e)
    assert "RSU-2 lost on trust" in text and "RSU-1 was selected over RSU-2" in text


def test_winner_with_lower_trust_is_explained_by_latency_not_by_trust():
    # RSU-2 has the higher trust but a far worse link: it must lose, and the text must say so.
    rec = _record(1.0, 0.5, 0.3, [(1, 0.80, 0.1, 0.3), (2, 0.95, 0.9, 0.3)])
    e = explain_decision(rec)
    assert e.winner == 1
    c = e.runner_up
    assert c.kind == "decisive" and c.decisive == "latency" and c.outweighed == ("trust",)
    text = render_text(e)
    assert "RSU-2 lost on latency" in text
    assert "RSU-2 had higher trust (0.95 against 0.80 for RSU-1)" in text and "outweighed" in text


def test_winner_with_lower_trust_decided_by_load():
    rec = _record(1.0, 0.5, 0.3, [(1, 0.70, 0.3, 0.1), (2, 0.90, 0.3, 0.9)])
    c = explain_decision(rec).runner_up
    assert c.decisive == "load" and "trust" in c.outweighed


def test_no_single_term_dominates_is_reported_as_mixed_with_the_two_leading_terms():
    # latency and load each contribute about half of the margin; trust is equal.
    rec = _record(1.0, 0.5, 0.5, [(1, 0.8, 0.2, 0.2), (2, 0.8, 0.6, 0.6)])
    e = explain_decision(rec)
    c = e.runner_up
    assert c.kind == "mixed" and c.decisive is None
    assert max(abs(c.share(t)) for t in ("latency", "load")) < DOMINANCE_SHARE
    assert set(c.leading) == {"latency", "load"}
    assert "No single term decided it" in render_text(e)


def test_dominance_threshold_is_the_pre_declared_share():
    assert DOMINANCE_SHARE == 0.6  # D53; changing it is a new decision, not a tweak
    # share exactly 0.7 of the margin -> decisive; 0.5 -> mixed
    dec = explain_decision(_record(1.0, 1.0, 1.0, [(1, 0.0, 0.0, 0.0), (2, 0.0, 0.7, 0.3)])).runner_up
    assert dec.kind == "decisive" and dec.decisive == "latency"
    mix = explain_decision(_record(1.0, 1.0, 1.0, [(1, 0.0, 0.0, 0.0), (2, 0.0, 0.5, 0.5)])).runner_up
    assert mix.kind == "mixed"


def test_a_tie_goes_to_the_lower_rsu_index_and_says_so():
    e = explain_decision(_record(1.0, 0.5, 0.3, [(4, 0.7, 0.2, 0.2), (3, 0.7, 0.2, 0.2)]))
    assert e.winner == 3 and e.runner_up.kind == "tie"
    assert "tie goes to the lower RSU index" in render_text(e)


def test_single_eligible_rsu_has_no_runner_up():
    e = explain_decision(_record(1.0, 0.5, 0.3, [(1, 0.7, 0.2, 0.2), (2, 0.9, 0.1, 0.1)], excluded=(2,)))
    assert e.winner == 1 and e.runner_up is None and e.contrasts == ()
    text = render_text(e)
    assert "only eligible RSU" in text and "RSU-2 was in range but excluded" in text


# ---------------------------------------------------------- variants without trust


def test_variant_A_says_trust_did_not_influence_the_decision(cfg, world, trace):
    log = _log(cfg, world, trace, "A")
    seen = 0
    for rec in log:
        if rec["selected_rsu"] is None or len(rec["candidates"]) < 2:
            continue
        e = explain_decision(rec)
        text = render_text(e)
        assert e.trust_mode == "unused" and e.alpha == 0.0
        assert "Trust did not influence this decision" in text
        assert "lost on trust" not in text and "higher trust" not in text
        assert all(c.decisive != "trust" and "trust" not in c.leading for c in e.contrasts)
        seen += 1
    assert seen


def test_constant_trust_variant_B_says_trust_did_not_discriminate(cfg, world, trace):
    for rec in _log(cfg, world, trace, "B"):
        if rec["selected_rsu"] is None or len(rec["candidates"]) < 2:
            continue
        e = explain_decision(rec)
        assert e.trust_mode == "constant"
        assert "did not discriminate" in render_text(e)
        assert all(c.d_trust == 0.0 for c in e.contrasts)
        return
    pytest.fail("no multi-candidate decision in the smoke run")


# ------------------------------------------------- the log, the survey and the CLI


def test_explanations_carry_no_sealed_field(cfg, world, trace):
    rec = next(r for r in _log(cfg, world, trace, "B_oracle") if r["selected_rsu"] is not None)
    blob = json.dumps(explain_decision(rec).to_dict()).lower()
    for forbidden in ("behavior", "compromised", "dropped", "true_", "handoff_lost"):
        assert forbidden not in blob


def test_survey_counts_every_decision_and_flags_lower_trust_winners(cfg, world, trace, tmp_path):
    log = _log(cfg, world, trace, "C")
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in log) + "\n", encoding="utf-8")
    s = survey(path)
    assert s.total == sum(r["selected_rsu"] is not None for r in log)
    assert s.total == s.no_runner_up + sum(len(v) for v in s.by_kind.values())
    assert len(list(iter_log(path))) == len(log)


def test_cli_prints_an_explanation_and_fails_cleanly_on_an_unknown_id(cfg, world, trace, tmp_path):
    log = _log(cfg, world, trace, "D")
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in log) + "\n", encoding="utf-8")
    rec = next(r for r in log if len(r["candidates"]) > 1 and r["selected_rsu"] is not None)

    ok = subprocess.run(
        [sys.executable, str(REPO_ROOT / "explain.py"), "--decision", str(rec["decision_id"]), "--log", str(path)],
        capture_output=True, text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert f"Decision {rec['decision_id']}:" in ok.stdout and "was selected" in ok.stdout

    js = subprocess.run(
        [sys.executable, str(REPO_ROOT / "explain.py"), "--decision", str(rec["decision_id"]), "--log", str(path), "--json"],
        capture_output=True, text=True,
    )
    assert json.loads(js.stdout)["winner"] == rec["selected_rsu"]

    missing = subprocess.run(
        [sys.executable, str(REPO_ROOT / "explain.py"), "--decision", "999999", "--log", str(path)],
        capture_output=True, text=True,
    )
    assert missing.returncode == 1 and "not in" in missing.stderr

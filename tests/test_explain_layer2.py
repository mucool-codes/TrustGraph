"""S8: Layer 2 - GNNExplainer attribution on the trust head, perturbation, the conditional neighbour clause."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
from trustgraph import explain_layer2 as L2
from trustgraph.explain import explain_decision
from trustgraph.features import EDGE_COL, RSU_FEATURES, VEHICLE_FEATURES
from trustgraph.model import build_trust_head
from trustgraph.offload import OffloadSettings
from trustgraph.scenario import build_snapshot_builder
from trustgraph.simulator import run_closed_loop
from trustgraph.train import TrainResult, TrainSettings, save_checkpoint


def _closed(cfg, world, trace, variant="D", init_seed=0):
    head = "C" if variant == "C" else "D"
    model = build_trust_head(cfg.model, init_seed, "cpu", head).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    s = OffloadSettings(variant=variant, alpha=1.0, beta=0.5, gamma=0.3, log_decisions=True)
    sc = run_closed_loop(cfg, world, trace, s, model=model)
    builder = build_snapshot_builder(cfg, world, trace, sc.observed)
    graphs = {t: builder.build(t) for t in range(trace.num_steps)}
    return sc, model, graphs


def _two_candidate_record(sc):
    for rec in sc.extras["decision_log"]:
        if rec["selected_rsu"] is not None and sum(c["dwell_eligible"] for c in rec["candidates"]) >= 2:
            return rec
    pytest.skip("no two-candidate decision in the smoke run")


def test_d54_constants_are_frozen():
    assert (L2.EPOCHS, L2.HOPS, L2.DELTA, L2.TRUST_EPS) == (100, 2, 0.3, 0.05)
    assert (L2.NEIGHBOUR_SHARE_MIN, L2.NEIGHBOUR_NAMED_MIN, L2.NEIGHBOUR_MAX_NAMED) == (0.30, 0.05, 2)
    assert (L2.OWN_SECOND_MIN, L2.DIRECTION_EPS) == (0.25, 0.01)


def test_flip_moves_a_scalar_by_delta_toward_the_other_end():
    assert L2.flipped(0.2) == pytest.approx(0.5)
    assert L2.flipped(0.9) == pytest.approx(0.6)
    assert L2.flipped(0.5) == pytest.approx(0.2)


@pytest.mark.parametrize("variant", ["C", "D"])
def test_rebuilt_graph_reproduces_the_logged_trust_and_the_logged_decision(cfg, world, trace, variant):
    sc, model, graphs = _closed(cfg, world, trace, variant)
    n = world.topology.num_rsus
    seen = 0
    for rec in sc.extras["decision_log"]:
        if rec["selected_rsu"] is None:
            continue
        base = L2.trust_vector(model, graphs[rec["step"]], n)
        for c in rec["candidates"]:
            assert base[c["rsu"]] == pytest.approx(c["trust"], abs=1e-9)
        assert L2.rerun_decision(rec, base) == rec["selected_rsu"]
        seen += 1
    assert seen


def test_units_use_valid_slots_inside_the_receptive_field_and_rank_by_importance(cfg, world, trace):
    sc, model, graphs = _closed(cfg, world, trace)
    rec = _two_candidate_record(sc)
    g, n, v = graphs[rec["step"]], world.topology.num_rsus, rec["selected_rsu"]
    a = L2.attribute(model, g, v, n, 0, rec["decision_id"])
    field = L2.receptive_field(g.edge_index, v)
    assert v in field and all(u.node in field for u in a.units)
    for u in a.units:
        assert u.col in L2.valid_cols(u.node, n)  # zero-padded slots are never units
        assert u.importance >= 0.0
    assert len({u.key for u in a.units}) == len(a.units)
    imps = [u.importance for u in a.units]
    assert imps == sorted(imps, reverse=True)
    assert a.top(1)[0] is a.units[0]
    pool = a.low_pool()
    assert a.units[0] not in pool and len(pool) == len(a.units) // 2
    assert a.total_mass == pytest.approx(sum(imps))
    assert sum(a.share(node) for node in a.mass_by_node) == pytest.approx(1.0)


def test_attribution_is_deterministic_given_its_seed(cfg, world, trace):
    sc, model, graphs = _closed(cfg, world, trace)
    rec = _two_candidate_record(sc)
    g, n, v = graphs[rec["step"]], world.topology.num_rsus, rec["selected_rsu"]
    a = L2.attribute(model, g, v, n, 0, rec["decision_id"])
    b = L2.attribute(model, g, v, n, 0, rec["decision_id"])
    assert [(u.key, u.importance) for u in a.units] == [(u.key, u.importance) for u in b.units]


def test_perturbation_effect_matches_a_manual_recomputation(cfg, world, trace):
    sc, model, graphs = _closed(cfg, world, trace)
    rec = _two_candidate_record(sc)
    g, n, v = graphs[rec["step"]], world.topology.num_rsus, rec["selected_rsu"]
    base = L2.trust_vector(model, g, n)
    unit = L2.attribute(model, g, v, n, 0, rec["decision_id"]).top(1)[0]
    eff = L2.perturb_effect(model, g, rec, v, unit, n, base)
    x = g.x.clone()
    x[unit.node, unit.col] = L2.flipped(float(g.x[unit.node, unit.col]))
    assert eff["trust_delta"] == pytest.approx(float(L2.trust_vector(model, g, n, x)[v] - base[v]))
    assert eff["trust_changed"] == (abs(eff["trust_delta"]) >= L2.TRUST_EPS)
    assert isinstance(eff["decision_changed"], bool)
    # perturbing an input the model cannot read changes nothing: a vehicle node's RSU-block slot is not a unit,
    # but a node outside the receptive field never moves v's trust
    far = [i for i in range(g.num_nodes) if i not in L2.receptive_field(g.edge_index, v)]
    if far:
        x = g.x.clone()
        x[far[0]] = 0.9
        assert L2.trust_vector(model, g, n, x)[v] == pytest.approx(base[v], abs=1e-9)


def test_node_local_head_has_no_neighbour_story(cfg, world, trace):
    sc, model, graphs = _closed(cfg, world, trace, "C")
    rec = _two_candidate_record(sc)
    g, n, v = graphs[rec["step"]], world.topology.num_rsus, rec["selected_rsu"]
    a = L2.attribute(model, g, v, n, 0, rec["decision_id"])
    assert a.share(v) == pytest.approx(1.0)  # C reads only its own row
    ne = L2.explain_node(model, g, v, n, 0, rec["decision_id"], attribution=a)
    assert not ne.neighbour_clause and ne.same_segment_share == 0.0 and not ne.neighbours


def _graph_with_same_segment_neighbour(sc, graphs, n):
    for rec in sc.extras["decision_log"]:
        if rec["selected_rsu"] is None:
            continue
        g = graphs[rec["step"]]
        for v in range(n):
            nb = L2.same_segment_rsu_neighbours(g, v, n)
            if nb:
                return rec, g, v, nb
    pytest.skip("no same-segment RSU pair in the smoke graph")


def _synthetic(v, nb, own_mass, nb_mass, g, n):
    """An Attribution with chosen node masses: own row and the neighbour's row get the stated mass."""
    units, mass = [], {}
    for node, m in ((v, own_mass), (nb, nb_mass)):
        cols = list(L2.valid_cols(node, n))
        for k, col in enumerate(cols):
            units.append(L2.Unit(node, col, m / len(cols) * (1.0 + 0.01 * (len(cols) - k)), float(g.x[node, col])))
        mass[node] = sum(u.importance for u in units if u.node == node)
    units.sort(key=lambda u: (-u.importance, u.node, u.col))
    return L2.Attribution(rsu=v, run_seed=0, units=units, mass_by_node=mass, total_mass=sum(mass.values()))


def test_neighbour_clause_is_included_only_when_the_attribution_supports_it(cfg, world, trace):
    sc, model, graphs = _closed(cfg, world, trace)
    n = world.topology.num_rsus
    rec, g, v, nbrs = _graph_with_same_segment_neighbour(sc, graphs, n)
    nb = nbrs[0]
    strong = L2.explain_node(model, g, v, n, 0, rec["decision_id"], _synthetic(v, nb, 0.5, 0.5, g, n))
    assert strong.same_segment_share == pytest.approx(0.5, abs=0.01) and strong.neighbour_clause
    assert strong.neighbours and strong.neighbours[0].rsu == nb and len(strong.neighbours) <= L2.NEIGHBOUR_MAX_NAMED
    weak = L2.explain_node(model, g, v, n, 0, rec["decision_id"], _synthetic(v, nb, 0.9, 0.1, g, n))
    assert weak.same_segment_share < L2.NEIGHBOUR_SHARE_MIN and not weak.neighbour_clause and not weak.neighbours
    just_over = L2.explain_node(model, g, v, n, 0, rec["decision_id"], _synthetic(v, nb, 0.69, 0.31, g, n))
    assert just_over.neighbour_clause


def test_merged_text_names_neighbours_only_when_supported_and_says_otherwise(cfg, world, trace):
    sc, model, graphs = _closed(cfg, world, trace)
    n = world.topology.num_rsus
    rec = _two_candidate_record(sc)
    e = explain_decision(rec)
    g = graphs[rec["step"]]
    layer2 = {v: L2.explain_node(model, g, v, n, 0, rec["decision_id"]) for v in (e.winner, e.runner_up.loser)}
    text = L2.render_merged(e, layer2)
    assert f"RSU-{e.winner} selected over RSU-{e.runner_up.loser}. Primary factors:" in text
    assert "Why a node's trust has its value is not explained here" not in text  # Layer 2 answers it
    for v, ne in layer2.items():
        if ne.neighbour_clause:
            assert "same-backhaul-segment neighbours carry" in text
        else:
            assert "did not drive this score" in text
    assert "swapping it for a typical value" in text or "compared with a typical value" in text


def test_merged_text_for_variant_A_gives_no_trust_attribution(cfg, world, trace):
    s = OffloadSettings(variant="A", alpha=1.0, beta=0.5, gamma=0.3, log_decisions=True)
    rec = _two_candidate_record(run_closed_loop(cfg, world, trace, s))
    text = L2.render_merged(explain_decision(rec), {})
    assert "Trust did not influence this decision, so no trust attribution is given." in text
    assert "GNNExplainer" not in text


def test_cli_layer_2_explains_a_logged_decision_and_refuses_the_wrong_checkpoint(cfg, world, trace, tmp_path):
    import json

    sc, model, _ = _closed(cfg, world, trace)
    rec = _two_candidate_record(sc)
    log = tmp_path / "run.decisions.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in sc.extras["decision_log"]) + "\n", encoding="utf-8")
    sc.observed.save(tmp_path / "run.observed.npz")
    trace.save(tmp_path / "trace.npz")
    ck = tmp_path / "ck"
    save_checkpoint(ck / f"s3-seed{cfg.seed}-D.pt", model, "D", cfg.seed, TrainSettings(), TrainResult())
    base = [sys.executable, str(REPO_ROOT / "explain.py"), "--decision", str(rec["decision_id"]), "--log", str(log),
            "--layer", "2", "--config", str(REPO_ROOT / "configs" / "smoke.yaml"), "--seed", str(cfg.seed),
            "--trace", str(tmp_path / "trace.npz")]
    ok = subprocess.run(base + ["--checkpoint-dir", str(ck)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert "Primary factors:" in ok.stdout and "GNNExplainer" in ok.stdout

    other = build_trust_head(cfg.model, 123, "cpu", "D").eval()
    save_checkpoint(tmp_path / "ck2" / f"s3-seed{cfg.seed}-D.pt", other, "D", cfg.seed, TrainSettings(), TrainResult())
    bad = subprocess.run(base + ["--checkpoint-dir", str(tmp_path / "ck2")], capture_output=True, text=True)
    assert bad.returncode == 1 and "wrong --seed" in bad.stderr

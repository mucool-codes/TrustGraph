"""Smoke test: the whole pipeline on a tiny graph, plus the locked decisions.

Asserts end-to-end determinism from (config, seed, trace) and pins the parts of the
locked decisions that are cheap to check structurally, so a later session that breaks
one gets a failing test rather than a quietly wrong result.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch

from trustgraph.config import SeedChain, _validate
from trustgraph.model import build_trust_head
from trustgraph.pipeline import format_decisions, run_pipeline
from trustgraph.scenario import build_snapshot_builder, build_world, generate_trace
from trustgraph.selection import score_candidates, select

from conftest import DEMO_CONFIG, REPO_ROOT


# --------------------------------------------------------------------------- model


def test_trust_head_output_shape_and_range(cfg, world, trace):
    data = build_snapshot_builder(cfg, world, trace).build(0)
    model = build_trust_head(cfg.model, cfg.seeds.torch_seed("model_init"), cfg.device)

    with torch.no_grad():
        trust = model(data.x, data.edge_index)

    # L1: a single scalar per node, nothing else.
    assert trust.shape == (data.num_nodes,)
    assert torch.all(trust >= 0.0) and torch.all(trust <= 1.0)


# ---------------------------------------------------------------------- determinism


def test_pipeline_deterministic_same_seed(cfg, world, trace):
    """Two runs, same seed and trace, identical decisions."""
    first = run_pipeline(cfg, trace, world)
    second = run_pipeline(cfg, trace, world)

    assert len(first) > 0, "scenario produced no decisions; the test proves nothing"
    assert first == second
    assert format_decisions(cfg, first, trace) == format_decisions(cfg, second, trace)


def test_pipeline_differs_across_seeds(cfg):
    """Guards against the determinism test passing on constant output."""
    other = type(cfg)(**{**cfg.__dict__, "seed": cfg.seed + 1})
    assert run_pipeline(cfg, generate_trace(cfg)) != run_pipeline(
        other, generate_trace(other)
    )


def test_seed_chain_is_order_independent():
    """Streams do not shift when an unrelated purpose draws (DECISIONS.md D17)."""
    a = SeedChain(1234)
    expected = a.generator("mobility").random(5)

    b = SeedChain(1234)
    b.generator("topology").random(100)  # unrelated draws
    b.generator("features").random(7)
    assert np.allclose(b.generator("mobility").random(5), expected)


def test_seed_chain_streams_are_distinct():
    chain = SeedChain(99)
    assert not np.allclose(
        chain.generator("topology").random(8), chain.generator("mobility").random(8)
    )


def test_run_py_byte_identical_across_processes(tmp_path):
    """Run it twice against one trace on disk, compare stdout byte-for-byte.

    Uses a subprocess so this also catches anything that depends on interpreter state
    surviving between runs (a global RNG, a module-level cache).
    """
    trace_path = tmp_path / "demo.npz"
    generate = [
        sys.executable,
        "scripts/generate_trace.py",
        "--config",
        str(DEMO_CONFIG),
        "--out",
        str(trace_path),
    ]
    subprocess.run(generate, cwd=REPO_ROOT, capture_output=True, text=True, check=True)

    cmd = [
        sys.executable,
        "run.py",
        "--config",
        str(DEMO_CONFIG),
        "--trace",
        str(trace_path),
    ]
    first = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    second = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert first.stdout == second.stdout
    assert "total decisions" in first.stdout


def test_run_py_refuses_to_invent_a_trace(tmp_path):
    """L7 / trace.py: the pipeline reads motion from disk, it never regenerates it."""
    result = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--config",
            str(DEMO_CONFIG),
            "--trace",
            str(tmp_path / "absent.npz"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "generate_trace.py" in result.stderr


# ----------------------------------------------------------------- locked decisions


def test_selection_rule_is_the_locked_formula():
    """L1: score(v) = alpha*trust - beta*latency - gamma*load."""
    trust = np.array([0.9, 0.2])
    latency = np.array([0.5, 0.1])
    load = np.array([0.4, 0.7])
    got = score_candidates(trust, latency, load, alpha=1.0, beta=0.5, gamma=0.3)
    want = 1.0 * trust - 0.5 * latency - 0.3 * load
    assert np.allclose(got, want)


def test_baseline_a_ignores_trust():
    """L2: Baseline A is the same rule with alpha = 0."""
    latency = np.array([0.5, 0.1])
    load = np.array([0.4, 0.7])
    low = score_candidates(np.array([0.0, 0.0]), latency, load, 0.0, 0.5, 0.3)
    high = score_candidates(np.array([1.0, 1.0]), latency, load, 0.0, 0.5, 0.3)
    assert np.allclose(low, high)


def test_selection_terms_sum_to_score():
    """L11 Layer 1 is exact: the three terms reconstruct the score identically."""
    d = select(
        timestep=0,
        vehicle_id=0,
        candidates=np.array([0, 1, 2]),
        trust=np.array([0.9, 0.5, 0.7]),
        latency=np.array([0.2, 0.1, 0.9]),
        load=np.array([0.3, 0.8, 0.1]),
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
    )
    assert d is not None
    assert d.trust_term + d.latency_term + d.load_term == pytest.approx(d.score)
    assert d.runner_up_score is not None and d.score >= d.runner_up_score


def test_selection_ties_break_deterministically():
    """Identical scores must resolve to the lowest RSU index, not to argmax order."""
    args = dict(
        timestep=0,
        vehicle_id=0,
        candidates=np.array([5, 2, 9]),
        trust=np.array([0.5, 0.5, 0.5]),
        latency=np.array([0.1, 0.1, 0.1]),
        load=np.array([0.2, 0.2, 0.2]),
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
    )
    assert select(**args).chosen_rsu == 2


def test_no_candidates_returns_none():
    """A vehicle with no RSU in range is a real condition under mobility, not a bug."""
    assert (
        select(
            timestep=0,
            vehicle_id=0,
            candidates=np.array([], dtype=int),
            trust=np.array([]),
            latency=np.array([]),
            load=np.array([]),
            alpha=1.0,
            beta=0.5,
            gamma=0.3,
        )
        is None
    )


def test_decisions_have_a_real_choice_to_make(demo_cfg):
    """The selection rule must usually see more than one candidate.

    Otherwise every decision is forced and an evaluation of the trust term measures
    coverage rather than trust - the failure recorded in FINDINGS.md F3.
    """
    world = build_world(demo_cfg)
    decisions = run_pipeline(demo_cfg, generate_trace(demo_cfg, world), world)
    contested = [d for d in decisions if len(d.candidates) > 1]
    assert len(contested) / len(decisions) > 0.9


def test_scale_cap_enforced(cfg):
    """L9: the config loader refuses to exceed 30 RSUs / 100 vehicles."""
    with pytest.raises(ValueError, match="L9 cap"):
        _validate(
            type(cfg)(**{**cfg.__dict__, "topology": {**cfg.topology, "num_rsus": 31}})
        )
    with pytest.raises(ValueError, match="L9 cap"):
        _validate(
            type(cfg)(
                **{**cfg.__dict__, "mobility": {**cfg.mobility, "num_vehicles": 101}}
            )
        )


def test_demo_config_is_within_the_locked_scale(demo_cfg):
    """L9: the operating scenario itself sits in the 15-30 / 50-100 band."""
    assert 15 <= demo_cfg.topology["num_rsus"] <= 30
    assert 50 <= demo_cfg.mobility["num_vehicles"] <= 100

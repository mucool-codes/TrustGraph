"""Shared by the S8 scripts: the D54 sample, closed-loop runs, and rebuilt step graphs.

EVALUATION-SIDE (the closed-loop run reads/produces sealed ground truth for the explained RSU's class);
allowlisted in tests/test_sealing.py. Nothing here is imported by the explanation modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from s6_run import load_weights  # noqa: E402
from trustgraph.config import load_config, with_overrides  # noqa: E402
from trustgraph.model import build_trust_head  # noqa: E402
from trustgraph.offload import build_offload_settings  # noqa: E402
from trustgraph.scenario import build_snapshot_builder, build_world  # noqa: E402
from trustgraph.simulator import run_closed_loop  # noqa: E402
from trustgraph.trace import default_trace_path, load_trace  # noqa: E402
from trustgraph.train import load_checkpoint  # noqa: E402

CONFIG = "configs/demo.yaml"
WEIGHTS_FILE = "configs/weights.yaml"
SEEDS = (1, 2, 3, 4, 5)
RHO, FRACTION, DRAW = 1.0, 0.20, 100
DECISIONS_PER_SEED = 40
MIN_STEP = 400
SAMPLE_RNG = 20261014  # + seed
LOW_RNG = 20261015  # + seed


def load_head(cfg, head: str) -> torch.nn.Module:
    m = build_trust_head(cfg.model, 0, cfg.device, head)
    load_checkpoint(REPO_ROOT / "checkpoints" / f"s3-seed{cfg.seed}-{head}.pt", m)
    m.eval()
    return m


def run_variant(variant: str, seed: int, rho=RHO, fraction=FRACTION, draw=DRAW, config=CONFIG):
    """(cfg, world, trace, scenario, model): one closed-loop run with the decision log kept in memory."""
    cfg = with_overrides(load_config(config), seed=seed, rho=rho, fraction=fraction, draw=draw)
    world = build_world(cfg)
    trace = load_trace(default_trace_path(config, seed))
    head = "C" if variant == "C" else "D"
    model = load_head(cfg, head)
    settings = build_offload_settings(
        cfg, variant=variant, weights=load_weights(WEIGHTS_FILE, variant), log_decisions=True
    )
    scenario = run_closed_loop(cfg, world, trace, settings, model=model)
    return cfg, world, trace, scenario, model


def sample_decisions(records: list[dict], seed: int) -> list[dict]:
    """D54 section 4: step >= 400, at least two dwell-eligible candidates; 40 drawn without replacement."""
    ok = sorted(
        (r for r in records
         if r["selected_rsu"] is not None and r["step"] >= MIN_STEP
         and sum(c["dwell_eligible"] for c in r["candidates"]) >= 2),
        key=lambda r: r["decision_id"],
    )
    rng = np.random.default_rng(SAMPLE_RNG + seed)
    pick = rng.choice(len(ok), size=DECISIONS_PER_SEED, replace=False)
    return [ok[i] for i in sorted(pick)]


def step_graphs(cfg, world, trace, observed, steps) -> dict:
    """Rebuild the graphs of the requested steps from the closed-loop observable scenario (the graph the
    engine scored at that step: same features, same task demand, same link history)."""
    builder = build_snapshot_builder(cfg, world, trace, observed)
    want = set(int(s) for s in steps)
    out = {}
    for t in range(max(want) + 1):
        data = builder.build(t)
        if t in want:
            out[t] = data
    return out

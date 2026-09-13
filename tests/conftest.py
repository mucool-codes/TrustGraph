"""Shared fixtures.

The smoke config is deliberately tiny (4 RSUs, 5 vehicles) so a failing assertion
points at a graph small enough to work through by hand. It switches every S2 mechanism
on - one degraded segment, one colluder, one cold-start RSU.

The demo config is the operating scenario. Its 1200-step trace takes a few seconds to
generate, so it is built once per test session and shared.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import load_config  # noqa: E402
from trustgraph.scenario import build_world, generate_trace  # noqa: E402
from trustgraph.simulator import generate_scenario  # noqa: E402

SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke.yaml"
DEMO_CONFIG = REPO_ROOT / "configs" / "demo.yaml"


@pytest.fixture
def cfg():
    return load_config(SMOKE_CONFIG)


@pytest.fixture
def demo_cfg():
    return load_config(DEMO_CONFIG)


@pytest.fixture
def world(cfg):
    return build_world(cfg)


@pytest.fixture
def trace(cfg, world):
    return generate_trace(cfg, world)


@pytest.fixture
def scenario(cfg, world, trace):
    return generate_scenario(cfg, world, trace)


@pytest.fixture
def observed(scenario):
    return scenario.observed


@pytest.fixture(scope="session")
def demo_bundle():
    """(config, world, trace) for the operating scenario, built once per session."""
    cfg = load_config(DEMO_CONFIG)
    world = build_world(cfg)
    return cfg, world, generate_trace(cfg, world)

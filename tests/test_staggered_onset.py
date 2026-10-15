"""S13 (D65): the onset_spread_steps option. With the default 0 nothing changes (F11 hash); with S > 0 every
degraded RSU has its own onset in [onset_step, onset_step + S] and its own ramp."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trustgraph.config import load_config  # noqa: E402
from trustgraph.observed import load_observed  # noqa: E402
from trustgraph.scenario import build_snapshot_builder, build_world  # noqa: E402
from trustgraph.sealed.ground_truth import load_ground_truth  # noqa: E402
from trustgraph.sealed.injector import inject, injection_stream_name  # noqa: E402
from trustgraph.simulator import generate_scenario  # noqa: E402

F11_HASH = "f3f247ac6c9becb1d2ddf859bf0b635a3c0d4778076ff028e46cfc0c97130de7"  # FINDINGS F11, rho 1.0
DEG = {"fraction": 0.2, "rho": 1.0, "onset_step": 400, "ramp_steps": 300}


def _rngs(seed, draw=0):
    from trustgraph.sealed.injector import INJECTION_PURPOSES

    return {p: np.random.default_rng([seed, k, draw]) for k, p in enumerate(INJECTION_PURPOSES)}


SEG = np.repeat(np.arange(4), 5)


def test_default_spread_is_zero_and_gives_the_shared_onset():
    gt = inject(SEG, DEG, None, None, _rngs(1))
    assert gt.onset_steps is None
    assert (gt.onset_per_rsu() == 400).all()


def test_spread_draws_each_degraded_rsu_an_onset_in_range_and_leaves_others_alone():
    gt = inject(SEG, {**DEG, "onset_spread_steps": 300}, None, None, _rngs(1))
    on = gt.onset_per_rsu()
    assert (on[gt.degraded] >= 400).all() and (on[gt.degraded] <= 700).all()
    assert len(set(on[gt.degraded].tolist())) > 1
    assert (on[~gt.degraded] == 400).all()
    # which RSUs degrade does not depend on the spread
    base = inject(SEG, DEG, None, None, _rngs(1))
    assert np.array_equal(base.degraded, gt.degraded)


def test_each_rsu_ramps_from_its_own_onset():
    gt = inject(SEG, {**DEG, "onset_spread_steps": 300}, None, None, _rngs(2))
    on = gt.onset_per_rsu()
    for v in np.flatnonzero(gt.degraded):
        assert gt.degradation_level(int(on[v]))[v] == 0.0
        assert gt.degradation_level(int(on[v]) + 150)[v] == pytest.approx(0.5)
        assert gt.degradation_level(int(on[v]) + 300)[v] == 1.0
    assert (gt.degradation_level(1199)[~gt.degraded] == 0.0).all()


def test_negative_spread_is_rejected():
    with pytest.raises(ValueError):
        inject(SEG, {**DEG, "onset_spread_steps": -1}, None, None, _rngs(1))


def test_onset_stream_is_per_draw():
    assert injection_stream_name("onset", 0) == "onset"
    assert injection_stream_name("onset", 7) == "onset#draw7"


def test_staggered_config_differs_from_demo_only_by_the_one_key():
    demo = yaml.safe_load((REPO_ROOT / "configs" / "demo.yaml").read_text(encoding="utf-8"))
    stag = yaml.safe_load((REPO_ROOT / "configs" / "staggered.yaml").read_text(encoding="utf-8"))
    assert stag["degradation"].pop("onset_spread_steps") == 300
    assert stag == demo


def test_default_spread_reproduces_the_f11_graph_sequence_hash(demo_bundle, tmp_path):
    """F11/F21: the shared-onset scenario (configs/demo.yaml, seed 20260903, rho 1.0, draw 0) still hashes to f3f247ac...
    Built from the observable file after a disk round trip, as scripts/generate_scenario.py does."""
    cfg, world, trace = demo_bundle
    assert cfg.degradation.get("rho") == 1.0 and "onset_spread_steps" not in cfg.degradation
    sc = generate_scenario(cfg, world, trace)
    sc.observed.save(tmp_path / "x.observed.npz")
    observed = load_observed(tmp_path / "x.observed.npz")
    digest = hashlib.sha256()
    for data in build_snapshot_builder(cfg, world, trace, observed).snapshots():
        digest.update(data.x.numpy().tobytes())
        digest.update(data.edge_index.numpy().tobytes())
        digest.update(data.edge_attr.numpy().tobytes())
    assert digest.hexdigest() == F11_HASH


def test_sealed_file_round_trips_the_per_rsu_onsets_and_omits_them_when_unused(tmp_path):
    gt = inject(SEG, {**DEG, "onset_spread_steps": 300}, None, None, _rngs(3))
    z = np.zeros((2, 20), dtype=np.float32)
    full = gt.with_realised(z, z, np.zeros(1), np.zeros(1, dtype=bool))
    full.save(tmp_path / "a.SEALED.npz")
    back = load_ground_truth(tmp_path / "a.SEALED.npz")
    assert np.array_equal(back.onset_steps, gt.onset_steps)
    plain = inject(SEG, DEG, None, None, _rngs(3)).with_realised(z, z, np.zeros(1), np.zeros(1, dtype=bool))
    plain.save(tmp_path / "b.SEALED.npz")
    assert "onset_steps" not in np.load(tmp_path / "b.SEALED.npz").files
    assert load_ground_truth(tmp_path / "b.SEALED.npz").onset_steps is None

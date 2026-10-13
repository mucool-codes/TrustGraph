"""S3: the two trust heads, the L3 label, and the trainer.

The properties that make the C-vs-D ablation meaningful are tested directly:
C reads only its own RSU feature row; D's output depends on `same_segment`; the label
is the NEXT task's outcome and strictly in the future; training is deterministic and
the diagnostic callback cannot influence it.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from trustgraph.config import SeedChain
from trustgraph.dataset import graphset_from, next_task_labels
from trustgraph.diagnostic import auc
from trustgraph.features import EDGE_COL, NODE_FEATURE_DIM, RSU_FEATURES
from trustgraph.model import NodeLocalTrustHead, TrustHead, build_trust_head, num_parameters
from trustgraph.train import TrainSettings, calibration_table, edge_probe, train

from conftest import DEMO_CONFIG


def _models(cfg):
    seed = cfg.seeds.torch_seed("model_init")
    return (
        build_trust_head(cfg.model, seed, "cpu", "C"),
        build_trust_head(cfg.model, seed, "cpu", "D"),
    )


def _graph(cfg, world, trace, observed, t=6):
    from trustgraph.scenario import build_snapshot_builder

    builder = build_snapshot_builder(cfg, world, trace, observed)
    for step in range(t):  # link_age is history-dependent (D25)
        builder.build(step)
    return builder.build(t)


# ------------------------------------------------------------------------- models


def test_c_reads_only_its_own_rsu_feature_row(cfg, world, trace, observed):
    c, _ = _models(cfg)
    c.eval()
    data = _graph(cfg, world, trace, observed)
    base = c(data.x, data.edge_index, data.edge_attr)
    n_rsu = len(RSU_FEATURES)

    # Other nodes' features, the vehicle block, and every edge are invisible to C.
    x = data.x.clone()
    x[1:, :] = torch.rand_like(x[1:, :])
    x[0, n_rsu:] = 7.0
    changed = c(x, data.edge_index.flip(0), torch.rand_like(data.edge_attr))
    assert torch.equal(changed[0], base[0])
    # ...but its own RSU row is not.
    x2 = data.x.clone()
    x2[0, RSU_FEATURES.index("success_ewma")] += 0.5
    assert not torch.equal(c(x2, data.edge_index, data.edge_attr)[0], base[0])


def test_d_depends_on_same_segment_and_on_neighbours(cfg, world, trace, observed):
    _, d = _models(cfg)
    d.eval()
    data = _graph(cfg, world, trace, observed)
    base = d(data.x, data.edge_index, data.edge_attr)

    flipped = data.edge_attr.clone()
    flipped[:, EDGE_COL["same_segment"]] = 1.0 - flipped[:, EDGE_COL["same_segment"]]
    assert not torch.allclose(d(data.x, data.edge_index, flipped), base)

    # An RSU's score moves when a neighbour's features move (message passing).
    src, dst = data.edge_index
    target = int(dst[0])
    neighbour = int(src[dst == target][0])
    x = data.x.clone()
    x[neighbour] = torch.rand_like(x[neighbour])
    assert d(x, data.edge_index, data.edge_attr)[target] != base[target]


def test_c_and_d_share_width_depth_head_and_output_range(cfg, world, trace, observed):
    c, d = _models(cfg)
    assert isinstance(c, NodeLocalTrustHead) and isinstance(d, TrustHead)
    assert c.head.in_features == d.head.in_features == int(cfg.model["hidden_dim"])
    assert c.fc2.out_features == d.conv2.nn.out_features
    data = _graph(cfg, world, trace, observed)
    for m in (c, d):
        out = m.eval()(data.x, data.edge_index, data.edge_attr)
        assert out.shape == (data.num_nodes,)
        assert torch.all((out >= 0) & (out <= 1))
    assert num_parameters(d) > num_parameters(c)  # the edge projections; reported in F13


def test_d_rejects_other_depths_and_unknown_variant(cfg):
    with pytest.raises(ValueError):
        TrustHead(num_layers=3)
    with pytest.raises(ValueError):
        build_trust_head(cfg.model, 0, "cpu", "E")


def test_build_returns_train_mode_and_inference_mode_is_available(cfg):
    c, d = _models(cfg)
    assert c.training and d.training
    assert not d.eval().training


# ------------------------------------------------------------------------- labels


def test_next_task_label_is_the_next_dispatched_tasks_outcome(observed):
    y, valid = next_task_labels(observed)
    tasks = observed.tasks
    for v in range(observed.num_rsus):
        mine = np.flatnonzero(tasks.rsu == v)
        for t in range(0, observed.num_steps, 7):
            later = mine[tasks.step[mine] >= t]
            if later.size == 0 or not observed.rsu_active[t, v]:
                assert not valid[t, v]
            else:
                assert valid[t, v]
                assert y[t, v] == float(tasks.met_deadline[later[0]])


def test_label_is_strictly_after_the_features_that_predict_it(observed):
    """The row at step t holds only outcomes observed by t*dt; the labelled task is
    dispatched at step >= t and observed later, so its outcome is not in the row."""
    tasks = observed.tasks
    dispatched = tasks.dispatched
    assert np.all(tasks.observed_at_s[dispatched] > tasks.step[dispatched] * 1.0 - 1e-9)


# ------------------------------------------------------------------------ trainer


def _tiny_sets(cfg, world, trace, observed):
    gs = graphset_from(cfg, world, trace, observed, stride=2)
    half = len(gs) // 2
    return gs.graphs[:half], gs.graphs[half:], gs


def test_training_is_deterministic_and_reduces_loss(cfg, world, trace, observed):
    tr, va, _ = _tiny_sets(cfg, world, trace, observed)
    settings = TrainSettings(max_epochs=3, batch_graphs=8)
    finals = []
    for _ in range(2):
        seeds = SeedChain(cfg.seed)
        _, d = (build_trust_head(cfg.model, seeds.torch_seed("model_init"), "cpu", v) for v in "CD")
        res = train(d, tr, va, settings, seeds, log=lambda s: None)
        finals.append((copy.deepcopy(d.state_dict()), res))
    for k in finals[0][0]:
        assert torch.equal(finals[0][0][k], finals[1][0][k])
    res = finals[0][1]
    assert min(h["val_bce"] for h in res.history[1:]) < res.initial_val_bce


def test_diagnostic_callback_cannot_influence_training(cfg, world, trace, observed):
    tr, va, _ = _tiny_sets(cfg, world, trace, observed)
    settings = TrainSettings(max_epochs=3, batch_graphs=8)

    def run(callback):
        seeds = SeedChain(cfg.seed)
        d = build_trust_head(cfg.model, seeds.torch_seed("model_init"), "cpu", "D")
        res = train(d, tr, va, settings, seeds, epoch_callback=callback, log=lambda s: None)
        return d.state_dict(), res

    def vandal(epoch, model):
        for p in model.parameters():  # tries to wreck the weights it was handed
            p.data.fill_(1e3)
        return "stop now"  # and to steer the loop

    clean, r0 = run(None)
    attacked, r1 = run(vandal)
    for k in clean:
        assert torch.equal(clean[k], attacked[k])
    h0 = [{k: (None if v != v else v) for k, v in h.items()} for h in r0.history]
    h1 = [{k: (None if v != v else v) for k, v in h.items()} for h in r1.history]
    assert h0 == h1


def test_edge_probe_reports_a_change_for_d_and_none_for_c(cfg, world, trace, observed):
    _, _, gs = _tiny_sets(cfg, world, trace, observed)
    c, d = _models(cfg)
    n = world.topology.num_rsus
    probe_d = edge_probe(d, gs.graphs[:10], n)
    probe_c = edge_probe(c, gs.graphs[:10], n)
    assert probe_d["mean_abs_change_same_segment_to_0"] > 0
    assert probe_c["mean_abs_change_same_segment_to_0"] == 0.0


def test_calibration_bins_partition_the_labelled_points():
    rng = np.random.default_rng(0)
    p = rng.random((50, 4))
    y = (rng.random((50, 4)) < p).astype(np.float32)
    valid = rng.random((50, 4)) < 0.8
    rows = calibration_table(p, y, valid, bins=10)
    assert sum(r["n"] for r in rows) == int(valid.sum())
    assert rows[0]["lo"] == 0.0 and rows[-1]["hi"] == 1.0


def test_auc_matches_brute_force_including_ties():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 5, 40).astype(float)
    lab = rng.integers(0, 2, 40)
    pos, neg = s[lab == 1], s[lab == 0]
    expect = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert auc(s, lab) == pytest.approx(expect)
    assert np.isnan(auc(s, np.ones(40, dtype=int)))

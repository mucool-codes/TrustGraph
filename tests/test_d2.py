"""S14 (D68): the contained graph head D2 ("C plus neighbours").

Properties the pre-registration relies on: D2 starts as exactly C's function; only same-segment RSU
neighbours enter (never vehicles, never other-segment RSUs); the neighbour vector is the mean of the encoded
neighbour rows; a node with no such neighbour gets a zero vector and flag 0; the parameter counts are the
hand counts of D68; the neighbour block can be zeroed for the probe.
"""

from __future__ import annotations

import torch

from trustgraph.features import EDGE_COL, EDGE_FEATURE_DIM, NODE_FEATURE_DIM, RSU_BLOCK, RSU_FEATURES
from trustgraph.model import ContainedGraphHead, build_trust_head, num_parameters


def _models(cfg):
    seed = cfg.seeds.torch_seed("model_init")
    return [build_trust_head(cfg.model, seed, "cpu", v) for v in ("C", "D", "D2")]


def _toy():
    """3 RSUs (0,1,2) and 1 vehicle (3). Edges both ways: 0-1 same segment, 0-2 other segment, 0-3 vehicle link."""
    x = torch.rand(4, NODE_FEATURE_DIM)
    x[3, : len(RSU_FEATURES)] = 0.0  # a vehicle row has a zero RSU block
    pairs = [(0, 1, 1.0), (0, 2, 0.0), (0, 3, 0.0)]
    src, dst, ss = [], [], []
    for a, b, s in pairs:
        src += [a, b]
        dst += [b, a]
        ss += [s, s]
    ea = torch.rand(len(src), EDGE_FEATURE_DIM)
    ea[:, EDGE_COL["same_segment"]] = torch.tensor(ss)
    return x, torch.tensor([src, dst]), ea


def test_parameter_counts_are_the_hand_counts(demo_cfg):
    c, d, d2 = _models(demo_cfg)
    assert num_parameters(c) == 224 + 1056 + 33 == 1313
    assert num_parameters(d2) == 1313 + 224 + 33 == 1570
    assert num_parameters(d) > num_parameters(c)


def test_d2_starts_as_exactly_c(cfg):
    c, _, d2 = _models(cfg)
    x, ei, ea = _toy()
    assert torch.equal(c(x, ei, ea), d2(x, ei, ea))
    assert torch.equal(d2.fc1.weight, c.fc1.weight) and torch.equal(d2.head.weight, c.head.weight)
    assert float(d2.head_nbr.weight.abs().sum()) == 0.0


def test_neighbour_block_is_the_mean_of_encoded_same_segment_rsu_neighbours(cfg):
    _, _, d2 = _models(cfg)
    x, ei, ea = _toy()
    blk = d2.neighbour_block(x, ei, ea)
    enc = torch.relu(d2.nbr_enc(x[:, RSU_BLOCK]))
    # node 0: one same-segment neighbour (RSU 1); the other-segment RSU 2 and vehicle 3 are excluded
    assert torch.allclose(blk[0, :-1], enc[1]) and blk[0, -1] == 1.0
    assert torch.allclose(blk[1, :-1], enc[0]) and blk[1, -1] == 1.0
    # nodes with no same-segment neighbour: zero vector, flag 0 (RSU 2, the vehicle)
    for n in (2, 3):
        assert float(blk[n].abs().sum()) == 0.0


def test_mean_over_two_neighbours_and_vehicle_rows_never_contribute(cfg):
    _, _, d2 = _models(cfg)
    x = torch.rand(4, NODE_FEATURE_DIM)
    # RSUs 1 and 2 are both same-segment neighbours of RSU 0; vehicle 3 carries a huge value that must not leak
    src = torch.tensor([1, 2, 3, 0, 0, 0])
    dst = torch.tensor([0, 0, 0, 1, 2, 3])
    ea = torch.zeros(6, EDGE_FEATURE_DIM)
    ea[[0, 1, 3, 4], EDGE_COL["same_segment"]] = 1.0
    x[3, : len(RSU_FEATURES)] = 0.0
    x[3, len(RSU_FEATURES):] = 1e3
    enc = torch.relu(d2.nbr_enc(x[:, RSU_BLOCK]))
    blk = d2.neighbour_block(x, torch.stack([src, dst]), ea)
    assert torch.allclose(blk[0, :-1], (enc[1] + enc[2]) / 2)


def test_output_depends_on_neighbours_only_once_head_nbr_is_nonzero(cfg):
    _, _, d2 = _models(cfg)
    x, ei, ea = _toy()
    x2 = x.clone()
    x2[1, RSU_FEATURES.index("success_ewma")] += 0.5
    assert torch.equal(d2(x, ei, ea)[0], d2(x2, ei, ea)[0])  # zero block: neighbour invisible
    with torch.no_grad():
        d2.head_nbr.weight.copy_(torch.randn_like(d2.head_nbr.weight))
    assert not torch.equal(d2(x, ei, ea)[0], d2(x2, ei, ea)[0])
    assert torch.equal(d2(x, ei, ea, drop_neighbours=True), d2.head(torch.relu(d2.fc2(torch.relu(d2.fc1(x[:, RSU_BLOCK]))))).squeeze(-1).sigmoid())


def test_neighbour_gradient_reaches_the_encoder_only_after_head_nbr_moves(cfg):
    _, _, d2 = _models(cfg)
    x, ei, ea = _toy()
    d2(x, ei, ea).sum().backward()
    assert d2.head_nbr.weight.grad is not None and float(d2.head_nbr.weight.grad.abs().sum()) > 0.0
    assert float(d2.nbr_enc.weight.grad.abs().sum()) == 0.0  # zero head weights pass no gradient yet


def test_no_same_segment_edge_touches_a_vehicle_row_in_a_real_graph(cfg, world, trace, observed):
    from test_trust_models import _graph

    data = _graph(cfg, world, trace, observed)
    n = world.topology.num_rsus
    ss = data.edge_attr[:, EDGE_COL["same_segment"]] > 0.5
    assert bool(ss.any())
    assert bool((data.edge_index[:, ss] < n).all())

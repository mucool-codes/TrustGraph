"""The trust heads: Variant D (message passing) and Variant C (node-local).

L1: the model's ONLY job is producing a scalar trust score per fog node. There is no
learned selector here and there must never be one - selection is the analytic rule in
`selection.py`.

Both variants share one interface, `forward(x, edge_index, edge_attr) -> trust`, and
one structure: two hidden layers of width `hidden_dim`, ReLU, then a linear head and a
sigmoid. The ONLY difference is what the two hidden layers read (DECISIONS.md D44):

  * D (`TrustHead`)          two GINEConv layers over the whole graph, consuming all
                             four edge features including `same_segment` (L6);
  * C (`NodeLocalTrustHead`) two Linear layers over a node's own RSU feature row.

Trained self-supervised on observed deadline outcomes (L3) by `train.py`. Training mode
is the caller's to set: `build_trust_head` returns a model in train mode, and inference
code calls `.eval()` (S0 hard-coded eval here; that assumption is removed).
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv

from .features import EDGE_COL, EDGE_FEATURE_DIM, NODE_FEATURE_DIM, RSU_BLOCK, RSU_FEATURES

VARIANTS = ("C", "D", "D2")


class TrustHead(nn.Module):
    """Variant D: two-layer GINEConv encoder with a scalar sigmoid head.

    Two layers means each RSU's trust is informed by nodes up to 2 hops away - far
    enough to pick up a neighbour on the same backhaul segment, which is the signal L5
    exists to create. `GINEConv` adds a learned projection of each edge's features to
    the neighbour's message, so `same_segment` reaches the aggregation; SAGEConv could
    not do this (D44).

    Each layer's `nn` is a single Linear, so a layer is "aggregate, Linear, ReLU" -
    the same depth per layer as Variant C's "Linear, ReLU".
    """

    def __init__(
        self,
        in_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = 32,
        num_layers: int = 2,
        edge_dim: int = EDGE_FEATURE_DIM,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError(
                "the trust head is fixed at 2 message-passing layers (L1); changing "
                "depth is a modelling decision that belongs in DECISIONS.md first"
            )
        self.conv1 = GINEConv(nn.Linear(in_dim, hidden_dim), edge_dim=edge_dim)
        self.conv2 = GINEConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=edge_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def logits(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = torch.relu(self.conv1(x, edge_index, edge_attr))
        h = torch.relu(self.conv2(h, edge_index, edge_attr))
        return self.head(h).squeeze(-1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Return (num_nodes,) trust scores in [0, 1]."""
        return torch.sigmoid(self.logits(x, edge_index, edge_attr))


class NodeLocalTrustHead(nn.Module):
    """Variant C: an MLP over a single RSU's own feature row.

    It reads exactly the RSU block of `x` (`features.RSU_FEATURES`) and nothing else: no
    message passing, no other node's data, no vehicle block, no position, no edge
    feature. `edge_index` and `edge_attr` are accepted so C and D are interchangeable
    at every call site, and are never read. Rows are independent, so a vehicle row
    produces a meaningless score that callers ignore, exactly as for D.

    Same width, depth, activation and head as D (D44); only the message passing is
    absent.
    """

    def __init__(self, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("the node-local head matches D's depth of 2 (D44)")
        self.fc1 = nn.Linear(len(RSU_FEATURES), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def logits(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = torch.relu(self.fc1(x[:, RSU_BLOCK]))
        h = torch.relu(self.fc2(h))
        return self.head(h).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.sigmoid(self.logits(x, edge_index, edge_attr))


class ContainedGraphHead(nn.Module):
    """Variant D2 ("C plus neighbours", DECISIONS.md D68): C's own branch plus a contained neighbour branch.

    Own branch: identical to `NodeLocalTrustHead` (two Linear layers over the node's own RSU row).
    Neighbour branch: the mean, over the node's RSU neighbours joined by an edge with `same_segment` = 1,
    of a 1-layer encoder (Linear + ReLU) of those neighbours' RSU feature rows; zero when there is no such
    neighbour, with a 0/1 "has neighbours" flag appended. Vehicle nodes and other-segment RSUs never enter.
    Head: logit = head_own(own) + head_nbr(neighbour block), i.e. one linear head over the concatenation,
    with the neighbour weights initialised to ZERO, so D2 starts as exactly C's function.

    The own branch is built first, so under the same torch seed its initial weights equal C's.
    """

    def __init__(self, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("D2 matches C's depth of 2 (D68)")
        self.fc1 = nn.Linear(len(RSU_FEATURES), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.nbr_enc = nn.Linear(len(RSU_FEATURES), hidden_dim)
        self.head_nbr = nn.Linear(hidden_dim + 1, 1, bias=False)  # encoded neighbours + the has-neighbours flag
        nn.init.zeros_(self.head_nbr.weight)

    def neighbour_block(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """(num_nodes, hidden + 1): mean encoded same-segment neighbour vector and the has-neighbours flag."""
        n = x.shape[0]
        sel = edge_attr[:, EDGE_COL["same_segment"]] > 0.5
        src, dst = edge_index[0][sel], edge_index[1][sel]
        enc = torch.relu(self.nbr_enc(x[:, RSU_BLOCK]))
        total = torch.zeros(n, enc.shape[1], dtype=enc.dtype).index_add(0, dst, enc[src])
        count = torch.zeros(n, dtype=enc.dtype).index_add(0, dst, torch.ones(src.shape[0], dtype=enc.dtype))
        mean = total / count.clamp(min=1.0).unsqueeze(-1)
        return torch.cat([mean, (count > 0).to(enc.dtype).unsqueeze(-1)], dim=-1)

    def logits(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, drop_neighbours: bool = False
    ) -> torch.Tensor:
        h = torch.relu(self.fc1(x[:, RSU_BLOCK]))
        h = torch.relu(self.fc2(h))
        out = self.head(h).squeeze(-1)
        if drop_neighbours:  # probe only (D68 section 2): the neighbour block set to zero
            return out
        return out + self.head_nbr(self.neighbour_block(x, edge_index, edge_attr)).squeeze(-1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, drop_neighbours: bool = False
    ) -> torch.Tensor:
        return torch.sigmoid(self.logits(x, edge_index, edge_attr, drop_neighbours))


def build_trust_head(
    cfg_model: dict, torch_seed: int, device: str, variant: str = "D"
) -> nn.Module:
    """Construct a trust head with deterministic initialisation.

    The seed comes from the run's seed chain (DECISIONS.md D17), so weights are a
    function of the master seed and nothing else. The model is returned in train mode;
    inference callers set `.eval()` themselves.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    torch.manual_seed(torch_seed)
    hidden_dim = int(cfg_model["hidden_dim"])
    num_layers = int(cfg_model["num_layers"])
    if variant == "D":
        model: nn.Module = TrustHead(
            in_dim=NODE_FEATURE_DIM, hidden_dim=hidden_dim, num_layers=num_layers
        )
    elif variant == "D2":
        model = ContainedGraphHead(hidden_dim=hidden_dim, num_layers=num_layers)
    else:
        model = NodeLocalTrustHead(hidden_dim=hidden_dim, num_layers=num_layers)
    return model.to(device)


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

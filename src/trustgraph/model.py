"""The trust head.

L1: the GNN's ONLY job is producing a scalar trust score per fog node. There is no
learned selector here and there must never be one - selection is the analytic rule in
`selection.py`.

S0 runs this UNTRAINED, forward pass only. Training is self-supervised on observed
deadline outcomes (L3) and arrives in a later session.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from .features import NODE_FEATURE_DIM


class TrustHead(nn.Module):
    """Two-layer GraphSAGE encoder with a scalar sigmoid head.

    Two layers means each RSU's trust is informed by nodes up to 2 hops away - far
    enough to pick up a neighbour on the same backhaul segment, which is the signal
    L5 exists to create.

    Output is `trust_v` in [0, 1], one scalar per node. Callers take the RSU rows.
    """

    def __init__(
        self,
        in_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = 32,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError(
                "S0 fixes the trust head at 2 GraphSAGE layers; changing depth is a "
                "modelling decision that belongs in DECISIONS.md first"
            )
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return (num_nodes,) trust scores in [0, 1]."""
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        return torch.sigmoid(self.head(h)).squeeze(-1)


def build_trust_head(cfg_model: dict, torch_seed: int, device: str) -> TrustHead:
    """Construct the trust head with deterministic initialisation.

    The seed comes from the run's seed chain (DECISIONS.md D17), so weights are a
    function of the master seed and nothing else.
    """
    torch.manual_seed(torch_seed)
    model = TrustHead(
        in_dim=NODE_FEATURE_DIM,
        hidden_dim=int(cfg_model["hidden_dim"]),
        num_layers=int(cfg_model["num_layers"]),
    )
    model.to(device)
    model.eval()  # S0: untrained, forward pass only
    return model

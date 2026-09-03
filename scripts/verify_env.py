"""Environment verification (S0 task 7).

Confirms the pinned stack actually works: imports torch_geometric, reports CUDA
availability, and runs one GraphSAGE forward pass on the GPU (falling back to CPU and
saying so if CUDA is unavailable).

Run:  python scripts/verify_env.py
Its output is transcribed into FINDINGS.md as the environment entry.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    print("=" * 70)
    print("TrustGraph environment verification")
    print("=" * 70)
    print(f"python            : {platform.python_version()} ({platform.machine()})")
    print(f"platform          : {platform.platform()}")

    import numpy
    import torch
    import torch_geometric
    import yaml

    print(f"numpy             : {numpy.__version__}")
    print(f"pyyaml            : {yaml.__version__}")
    print(f"torch             : {torch.__version__}")
    print(f"torch CUDA build  : {torch.version.cuda}")
    print(f"torch_geometric   : {torch_geometric.__version__}")

    cuda_ok = torch.cuda.is_available()
    print(f"cuda available    : {cuda_ok}")
    if cuda_ok:
        print(f"gpu               : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"compute capability: {cap[0]}.{cap[1]}")

    device = "cuda" if cuda_ok else "cpu"
    print(f"forward-pass device: {device}")

    # One real GraphSAGE forward pass through the project's own trust head, so this
    # verifies the model as built rather than a stand-in.
    from trustgraph.features import NODE_FEATURE_DIM
    from trustgraph.model import TrustHead

    torch.manual_seed(0)
    model = TrustHead(in_dim=NODE_FEATURE_DIM, hidden_dim=32, num_layers=2).to(device)
    x = torch.randn(12, NODE_FEATURE_DIM, device=device)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        trust = model(x, edge_index)

    assert trust.shape == (12,), f"unexpected trust shape {tuple(trust.shape)}"
    assert torch.all(trust >= 0) and torch.all(trust <= 1), "trust outside [0, 1]"
    print(f"GraphSAGE forward : OK - shape {tuple(trust.shape)}, "
          f"range [{trust.min():.4f}, {trust.max():.4f}]")

    print("=" * 70)
    if not cuda_ok:
        print("NOTE: CUDA unavailable - ran on CPU. At the L9 scale (<=30 RSUs) the")
        print("      GPU is a convenience, not a requirement.")
        print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

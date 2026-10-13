"""Self-supervised training of the trust heads (L3).

The objective is binary cross-entropy of `trust_v` against "did node v meet the deadline
of the next task dispatched to it" (`dataset.next_task_labels`). That is the whole
training signal.

THE SEAL (L4, D35). This module never imports the sealed ground truth and has no
parameter that could carry it. Model selection and early stopping read held-out
next-task BCE only. The per-epoch diagnostic enters through `epoch_callback`, which

  * receives a DEEP COPY of the model, so it cannot change a weight,
  * has its return value discarded, so it cannot steer the loop,
  * is called after the epoch's selection decision has been made.

The callback lives in evaluation code that is allowed to hold both the model and the
ground truth (`diagnostic.py`, wired in by `scripts/s3_train.py`).

C and D go through this one function with one `TrainSettings`; nothing here branches
on the variant (D44, S3 Task 7).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data

from .config import SeedChain
from .dataset import GraphSet
from .features import EDGE_COL


@dataclass(frozen=True)
class TrainSettings:
    """Identical for every variant. Chosen a priori, not tuned (FINDINGS.md F13)."""

    lr: float = 1e-3
    batch_graphs: int = 32
    max_epochs: int = 60
    patience: int = 8
    min_delta: float = 1e-4
    weight_decay: float = 0.0


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    initial_val_bce: float = float("nan")
    best_epoch: int = 0
    stopped_epoch: int = 0
    converged_by: str = ""  # "patience" or "max_epochs"
    best_state: dict | None = None


def _batches(graphs: list[Data], size: int, order: np.ndarray | None = None):
    idx = np.arange(len(graphs)) if order is None else order
    for i in range(0, len(idx), size):
        yield Batch.from_data_list([graphs[j] for j in idx[i : i + size]])


@torch.no_grad()
def mean_bce(model: nn.Module, graphs: list[Data], batch_graphs: int = 64) -> float:
    """Mean next-task BCE over every labelled RSU-step in `graphs`."""
    model.eval()
    total, count = 0.0, 0
    for batch in _batches(graphs, batch_graphs):
        logits = model.logits(batch.x, batch.edge_index, batch.edge_attr)
        m = batch.mask
        total += float(F.binary_cross_entropy_with_logits(logits[m], batch.y[m], reduction="sum"))
        count += int(m.sum())
    return total / max(count, 1)


@torch.no_grad()
def predict_trust(model: nn.Module, graphset: GraphSet) -> np.ndarray:
    """(len(graphset), num_rsus) trust for each cached graph's RSU rows."""
    model.eval()
    out = []
    for batch in _batches(graphset.graphs, 64):
        trust = model(batch.x, batch.edge_index, batch.edge_attr)
        # Each graph contributes num_nodes rows; its RSUs come first.
        ptr = batch.ptr.tolist()
        for a in ptr[:-1]:
            out.append(trust[a : a + graphset.num_rsus].numpy())
    return np.stack(out)


def train(
    model: nn.Module,
    train_graphs: list[Data],
    val_graphs: list[Data],
    settings: TrainSettings,
    seeds: SeedChain,
    epoch_callback: Callable[[int, nn.Module], object] | None = None,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Train `model` in place; leave it holding the best-validation-BCE weights."""
    rng = seeds.generator("train_shuffle")
    opt = torch.optim.Adam(
        model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay
    )
    result = TrainResult()
    result.initial_val_bce = mean_bce(model, val_graphs)
    best = result.initial_val_bce
    # Epoch 0 is the untrained model, so a failure to learn is visible as a flat curve.
    result.best_state = copy.deepcopy(model.state_dict())
    result.history.append(
        {"epoch": 0, "train_bce": float("nan"), "val_bce": result.initial_val_bce}
    )
    if epoch_callback is not None:
        epoch_callback(0, copy.deepcopy(model))
    since_best = 0

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        order = rng.permutation(len(train_graphs))
        total, count = 0.0, 0
        for batch in _batches(train_graphs, settings.batch_graphs, order):
            m = batch.mask
            logits = model.logits(batch.x, batch.edge_index, batch.edge_attr)
            loss = F.binary_cross_entropy_with_logits(logits[m], batch.y[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * int(m.sum())
            count += int(m.sum())
        train_bce = total / max(count, 1)
        val_bce = mean_bce(model, val_graphs)
        result.history.append({"epoch": epoch, "train_bce": train_bce, "val_bce": val_bce})

        # Selection and early stopping: held-out next-task BCE, nothing else (Task 7).
        improved = val_bce < best - settings.min_delta
        if val_bce < best:
            best = val_bce
            result.best_epoch = epoch
            result.best_state = copy.deepcopy(model.state_dict())
        since_best = 0 if improved else since_best + 1
        stop = since_best >= settings.patience
        log(f"  epoch {epoch:3d}  train_bce {train_bce:.5f}  val_bce {val_bce:.5f}")

        if epoch_callback is not None:  # diagnostic only; copy in, result discarded
            epoch_callback(epoch, copy.deepcopy(model))

        result.stopped_epoch = epoch
        if stop:
            result.converged_by = "patience"
            break
    else:
        result.converged_by = "max_epochs"

    model.load_state_dict(result.best_state)
    model.eval()
    return result


# --------------------------------------------------------------------------- reports


def calibration_table(
    trust: np.ndarray, y: np.ndarray, valid: np.ndarray, bins: int = 10
) -> list[dict]:
    """Binned predicted trust vs observed next-task success, equal-width bins on [0, 1].

    `trust`, `y`, `valid` are aligned (steps, rsus) arrays.
    """
    p, o = trust[valid], y[valid]
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = which == b
        n = int(sel.sum())
        rows.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": n,
                "mean_pred": float(p[sel].mean()) if n else float("nan"),
                "observed": float(o[sel].mean()) if n else float("nan"),
            }
        )
    return rows


@torch.no_grad()
def edge_probe(model: nn.Module, graphs: list[Data], num_rsus: int) -> dict:
    """D44 fallback probe: does `same_segment` change the output?

    Sets `same_segment` to 0 on every edge, then to 1 on every RSU-RSU edge, and reports
    the mean absolute change in RSU trust. Meaningful only for an edge-aware model.
    """
    model.eval()
    col = EDGE_COL["same_segment"]
    to_zero, to_one = [], []
    for batch in _batches(graphs, 64):
        base = model(batch.x, batch.edge_index, batch.edge_attr)
        # RSU rows of every graph in the batch, labelled or not.
        rsu_rows = torch.zeros(batch.num_nodes, dtype=torch.bool)
        for a in batch.ptr.tolist()[:-1]:
            rsu_rows[a : a + num_rsus] = True

        zero = batch.edge_attr.clone()
        zero[:, col] = 0.0
        one = batch.edge_attr.clone()
        rr = rsu_rows[batch.edge_index[0]] & rsu_rows[batch.edge_index[1]]
        one[rr, col] = 1.0
        t0 = model(batch.x, batch.edge_index, zero)
        t1 = model(batch.x, batch.edge_index, one)
        to_zero.append((t0 - base).abs()[rsu_rows])
        to_one.append((t1 - base).abs()[rsu_rows])
    return {
        "mean_abs_change_same_segment_to_0": float(torch.cat(to_zero).mean()),
        "mean_abs_change_rsu_rsu_same_segment_to_1": float(torch.cat(to_one).mean()),
    }


# ---------------------------------------------------------------------- checkpoints


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    variant: str,
    seed: int,
    settings: TrainSettings,
    result: TrainResult,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant,
            "seed": int(seed),
            "state_dict": model.state_dict(),
            "settings": asdict(settings),
            "history": result.history,
            "best_epoch": result.best_epoch,
            "initial_val_bce": result.initial_val_bce,
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, model: nn.Module) -> dict:
    """Load weights into `model` and freeze it (L12: one model per seed, then frozen)."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return blob

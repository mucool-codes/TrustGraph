"""The sealed-label diagnostic (L4, PROJECT_SPEC.md 4.4). EVALUATION CODE.

This is the one place on the model side of the project that is allowed to hold both a
trained model's trust scores and the sealed ground truth. It is allowlisted in
`tests/test_sealing.py`; `train.py`, `dataset.py` and `model.py` are not, and cannot
import it. Its output is printed; nothing returned from here ever reaches a loss, an
early-stopping rule, or model selection - `train.train` discards the callback's return
value and hands it a copy of the model.

The metric is D43's: per RSU, mean trust over a step window (primary [450, 650)), score
= 1 - trust, label = degraded (1) vs reliable (0); compromised and cold-start RSUs are
excluded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .dataset import GraphSet
from .sealed.ground_truth import COMPROMISED, DEGRADED, RELIABLE, load_ground_truth, sealed_path_for
from .train import predict_trust

PRIMARY_WINDOW = (450, 650)
PLATEAU_WINDOW = (1100, 1200)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney, ties count half). `nan` if a class is empty."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(allv.size)
    i = 0
    while i < allv.size:  # average ranks over ties
        j = i
        while j + 1 < allv.size and allv[j + 1] == allv[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_of = np.empty(allv.size)
    rank_of[order] = ranks
    r_pos = rank_of[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def degraded_vs_reliable_auc(
    trust: np.ndarray, steps: np.ndarray, observed_path: str | Path, window: tuple[int, int]
) -> float:
    """AUC of (1 - window-mean trust) for degraded vs reliable RSUs of one scenario."""
    gt = load_ground_truth(sealed_path_for(observed_path))
    sel = (steps >= window[0]) & (steps < window[1])
    if not sel.any():
        raise ValueError(f"no cached steps inside window {window}")
    mean_trust = trust[sel].mean(axis=0)
    keep = (gt.behavior_class != COMPROMISED) & ~gt.cold_start
    labels = (gt.behavior_class == DEGRADED).astype(int)
    return auc(1.0 - mean_trust[keep], labels[keep])


def make_epoch_diagnostic(
    graphset: GraphSet,
    observed_path: str | Path,
    window: tuple[int, int] = PRIMARY_WINDOW,
    label: str = "",
    log: Callable[[str], None] = print,
) -> Callable[[int, object], None]:
    """An `epoch_callback` that PRINTS the diagnostic AUC and returns nothing."""

    def callback(epoch: int, model) -> None:
        trust = predict_trust(model, graphset)
        value = degraded_vs_reliable_auc(trust, graphset.steps, observed_path, window)
        log(f"           diagnostic AUC (sealed, {label}, steps {window[0]}-{window[1]}): {value:.4f}")

    return callback

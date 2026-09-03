"""The analytic selection rule.

L1:  score(v) = alpha*trust_v - beta*latency_v - gamma*load_v,  argmax over in-range
candidates, with hand-tuned alpha/beta/gamma. Selection is NOT learned.

L2:  Baseline A (no trust) is this same rule with alpha = 0.

Because the score is linear in three terms, each term's contribution to the margin
between the winner and the runner-up is an exact identity, not an approximation. That
is what makes Layer 1 of the explanation exact by construction (L11) - the fields are
carried on `Decision` here so a later session can render them without recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Decision:
    """One offloading decision, with the terms that produced it."""

    timestep: int
    vehicle_id: int
    candidates: tuple[int, ...]
    chosen_rsu: int
    runner_up: int | None
    score: float
    runner_up_score: float | None
    trust: float
    latency: float
    load: float
    # Signed contributions of each term to the chosen node's score. They sum to
    # `score` exactly.
    trust_term: float
    latency_term: float
    load_term: float


def score_candidates(
    trust: np.ndarray,
    latency: np.ndarray,
    load: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """score(v) = alpha*trust_v - beta*latency_v - gamma*load_v, elementwise."""
    return alpha * trust - beta * latency - gamma * load


def select(
    timestep: int,
    vehicle_id: int,
    candidates: np.ndarray,
    trust: np.ndarray,
    latency: np.ndarray,
    load: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> Decision | None:
    """Pick the argmax candidate for one vehicle.

    `candidates` holds global RSU indices; `trust`, `latency` and `load` are aligned
    with it. Returns None when the vehicle has no RSU in range - a real condition
    under mobility, not an error.

    Ties break to the lowest RSU index, so the rule is deterministic (Standing Rule 7)
    rather than dependent on argmax implementation details.
    """
    if candidates.size == 0:
        return None

    scores = score_candidates(trust, latency, load, alpha, beta, gamma)

    # Lexicographic sort on (-score, rsu_index) gives a deterministic tie-break.
    order = np.lexsort((candidates, -scores))
    best = int(order[0])
    second = int(order[1]) if candidates.size > 1 else None

    return Decision(
        timestep=timestep,
        vehicle_id=int(vehicle_id),
        candidates=tuple(int(c) for c in candidates),
        chosen_rsu=int(candidates[best]),
        runner_up=int(candidates[second]) if second is not None else None,
        score=float(scores[best]),
        runner_up_score=float(scores[second]) if second is not None else None,
        trust=float(trust[best]),
        latency=float(latency[best]),
        load=float(load[best]),
        trust_term=float(alpha * trust[best]),
        latency_term=float(-beta * latency[best]),
        load_term=float(-gamma * load[best]),
    )

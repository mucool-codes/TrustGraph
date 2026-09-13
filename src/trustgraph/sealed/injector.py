"""The degradation injector: decides, before the first timestep, what goes wrong where.

Three independent mechanisms, each on its own RNG stream so that turning one on does
not reshuffle the others (DECISIONS.md D17):

  degradation   `fraction` of RSUs degrade, gradually, from a shared onset. Which ones
                is controlled by `rho` - see `select_degraded` and DECISIONS.md D29.
  collusion     groups of RSUs that advertise a load below their true one (L8). They
                execute honestly; the lie is only in the advertisement.
  cold start    RSUs that are absent until `join_step` and join with no history.

SEALED (L4): this module lives in `trustgraph.sealed` and produces the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .ground_truth import COMPROMISED, DEGRADED, RELIABLE, GroundTruth


@dataclass(frozen=True)
class DegradationConfig:
    fraction: float = 0.0
    rho: float = 0.0
    onset_step: int = 0
    ramp_steps: int = 1
    severity: float = 0.6
    drop_prob: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("degradation.fraction must lie in [0, 1]")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("degradation.rho must lie in [0, 1]")
        if self.onset_step < 0:
            raise ValueError("degradation.onset_step must be >= 0")
        if self.ramp_steps < 1:
            raise ValueError("degradation.ramp_steps must be >= 1")
        if not 0.0 <= self.severity < 1.0:
            raise ValueError("degradation.severity must lie in [0, 1)")
        if not 0.0 <= self.drop_prob <= 1.0:
            raise ValueError("degradation.drop_prob must lie in [0, 1]")


@dataclass(frozen=True)
class CollusionConfig:
    num_groups: int = 0
    group_size: int = 0
    under_report: float = 0.5

    def __post_init__(self) -> None:
        if self.num_groups < 0 or self.group_size < 0:
            raise ValueError("collusion.num_groups and group_size must be >= 0")
        if not 0.0 < self.under_report < 1.0:
            raise ValueError("collusion.under_report must lie in (0, 1)")


@dataclass(frozen=True)
class ColdStartConfig:
    num_nodes: int = 0
    join_step: int = 0

    def __post_init__(self) -> None:
        if self.num_nodes < 0:
            raise ValueError("cold_start.num_nodes must be >= 0")
        if self.num_nodes > 0 and self.join_step < 1:
            raise ValueError("cold_start.join_step must be >= 1 when nodes cold-start")


_INT_FIELDS = {"onset_step", "ramp_steps", "num_groups", "group_size", "num_nodes", "join_step"}


def _build(cls, raw: dict | None, section: str):
    if not raw:
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown {section} config keys: {sorted(unknown)}")
    return cls(
        **{k: (int(v) if k in _INT_FIELDS else float(v)) for k, v in raw.items()}
    )


def degraded_count(num_rsus: int, fraction: float) -> int:
    """K = round(fraction * num_rsus), with halves rounded up.

    Python's round() is banker's rounding and would send 2.5 to 2 - an asymmetry
    nobody would think to look for in a sweep table.
    """
    return int(np.floor(fraction * num_rsus + 0.5))


def select_degraded(
    segment_id: np.ndarray, k: int, rho: float, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Choose `k` degraded RSUs with segment correlation `rho`. DECISIONS.md D29.

    Sequential. The first pick is uniform over RSUs. For each later pick, with
    probability `rho` it is *correlated*: uniform over the healthy members of segments
    that already hold a degraded RSU - or, if those segments are exhausted, uniform
    over the healthy RSUs of untouched segments, which opens a new one. Otherwise it is
    *independent*: uniform over all healthy RSUs.

      rho = 0  - uniform sampling without replacement: degraded nodes chosen
                 independently at random.
      rho = 1  - one segment fills completely before the next is opened, so the
                 degraded set is whole backhaul segments plus at most one partial one.

    Every random number is drawn up front in fixed-size blocks of `num_rsus`, and
    pick `i` reads only element `i`. Two consequences worth having: the draws never
    depend on `rho` or `k`, so for one seed the degraded set changes with `rho` only
    through the rule, not through a reshuffled stream; and the set for a smaller `k`
    is a prefix of the set for a larger one, so a fraction sweep at a fixed seed
    degrades nested sets rather than unrelated ones.

    Returns the mask and the number of correlated picks actually taken.
    """
    n = int(segment_id.shape[0])
    if not 0 <= k <= n:
        raise ValueError(f"cannot degrade {k} of {n} RSUs")
    u_branch = rng.random(n)
    u_pick = rng.random(n)

    degraded = np.zeros(n, dtype=bool)
    correlated = 0
    for i in range(k):
        healthy = ~degraded
        if i > 0 and u_branch[i] < rho:
            touched = np.isin(segment_id, segment_id[degraded])
            candidates = np.flatnonzero(healthy & touched)
            if candidates.size == 0:
                candidates = np.flatnonzero(healthy & ~touched)
            correlated += 1
        else:
            candidates = np.flatnonzero(healthy)
        pick = candidates[min(int(u_pick[i] * candidates.size), candidates.size - 1)]
        degraded[pick] = True
    return degraded, correlated


def inject(
    segment_id: np.ndarray,
    cfg_degradation: dict | None,
    cfg_collusion: dict | None,
    cfg_cold_start: dict | None,
    rngs: dict[str, np.random.Generator],
) -> GroundTruth:
    """Build the injection plan. `rngs` needs 'degradation', 'collusion', 'cold_start'."""
    deg = _build(DegradationConfig, cfg_degradation, "degradation")
    col = _build(CollusionConfig, cfg_collusion, "collusion")
    cold = _build(ColdStartConfig, cfg_cold_start, "cold_start")

    segment_id = np.asarray(segment_id, dtype=np.int64)
    n = int(segment_id.shape[0])

    k = degraded_count(n, deg.fraction)
    degraded, correlated = select_degraded(segment_id, k, deg.rho, rngs["degradation"])

    # Colluders are drawn from the RSUs left healthy, in a fixed permutation order, so
    # a node is never both degraded and colluding and the two failure modes stay
    # separately attributable.
    order = rngs["collusion"].permutation(n)
    eligible = [int(i) for i in order if not degraded[i]]
    wanted = col.num_groups * col.group_size
    if wanted > len(eligible):
        raise ValueError(
            f"collusion needs {wanted} RSUs but only {len(eligible)} are not degraded"
        )
    collusion_group = np.full(n, -1, dtype=np.int64)
    for g in range(col.num_groups):
        members = eligible[g * col.group_size : (g + 1) * col.group_size]
        collusion_group[members] = g

    if cold.num_nodes > n:
        raise ValueError(f"cold_start.num_nodes={cold.num_nodes} exceeds {n} RSUs")
    join_step = np.zeros(n, dtype=np.int64)
    joiners = rngs["cold_start"].permutation(n)[: cold.num_nodes]
    join_step[joiners] = cold.join_step

    behavior_class = np.full(n, RELIABLE, dtype=np.int64)
    behavior_class[degraded] = DEGRADED
    behavior_class[collusion_group >= 0] = COMPROMISED

    return GroundTruth(
        behavior_class=behavior_class,
        backhaul_segment_id=segment_id.copy(),
        rho=deg.rho,
        degraded_fraction=deg.fraction,
        onset_step=deg.onset_step,
        ramp_steps=deg.ramp_steps,
        severity=deg.severity,
        drop_prob=deg.drop_prob,
        num_correlated_picks=correlated,
        collusion_group=collusion_group,
        under_report=col.under_report,
        join_step=join_step,
    )

"""Advertised-vs-observed tracking: where `success_ewma` and `latency_dev` come from (L8).

After every task completes or times out, the vehicle compares what it observed with
what the node's advertisement promised. The *discrepancy* feeds the two behavioural
features; neither the advertised value nor the raw outcome is ever a feature by itself.

  success_ewma  EWMA over tasks of "the promise was kept". A promise is broken when
                the advertisement said the task would finish inside its deadline and
                it did not. A task the node never promised to finish in time - an
                honest, busy node whose advertised completion already exceeded the
                deadline - cannot break a promise, however late it is. That is the
                distinction D8 exists for: busy-and-honest is the selection rule's
                problem (gamma), lying is trust's.

  latency_dev   EWMA over tasks of how much slower than advertised the task was, as a
                fraction of the advertised time, clipped at `latency_dev_cap` and
                rescaled to [0, 1]. Faster than advertised counts as zero: beating a
                promise is not misbehaviour. A timed-out task's true completion time
                is unknown to the vehicle, so the deadline stands in for it - a lower
                bound, which under-states the deviation rather than inventing one.

This module sees only what a vehicle sees. It never imports the sealed ground truth,
and nothing in its interface could carry a behaviour class (L4).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

# No observation yet means no evidence of misbehaviour: every promise kept, no
# deviation. The same neutral values the S1 placeholders held (DECISIONS.md D24),
# now the starting point of a real estimate rather than a constant.
NEUTRAL_SUCCESS_EWMA = 1.0
NEUTRAL_LATENCY_DEV = 0.0

# Still placeholders after S2. No SCMS revocation is modelled (CRL propagation takes
# hours to days, far beyond a 20-minute scenario, so a compromised node's certificate
# would stay valid for the whole run anyway), and no restart or session-drop process
# exists to observe. See FINDINGS.md F8.
PLACEHOLDER_CERT_VALID = 1.0
PLACEHOLDER_UPTIME_STABILITY = 1.0


@dataclass(frozen=True)
class TrackingConfig:
    ewma_weight: float = 0.1
    latency_dev_cap: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.ewma_weight <= 1.0:
            raise ValueError("tracking.ewma_weight must lie in (0, 1]")
        if self.latency_dev_cap <= 0:
            raise ValueError("tracking.latency_dev_cap must be positive")


def build_tracking_config(cfg_tracking: dict | None) -> TrackingConfig:
    if not cfg_tracking:
        return TrackingConfig()
    known = {f.name for f in fields(TrackingConfig)}
    unknown = set(cfg_tracking) - known
    if unknown:
        raise ValueError(f"unknown tracking config keys: {sorted(unknown)}")
    return TrackingConfig(**{k: float(v) for k, v in cfg_tracking.items()})


def promise_kept(advertised_s: float, deadline_s: float, met_deadline: bool) -> bool:
    """False only if the node promised to meet the deadline and did not."""
    promised = advertised_s <= deadline_s
    return not (promised and not met_deadline)


def relative_latency_deviation(advertised_s: float, observed_s: float) -> float:
    """(observed - advertised) / advertised, floored at zero."""
    return max(observed_s - advertised_s, 0.0) / max(advertised_s, 1e-9)


class DiscrepancyTracker:
    """Per-RSU behavioural features, updated one observed outcome at a time."""

    def __init__(self, num_rsus: int, config: TrackingConfig) -> None:
        self.config = config
        self.success_ewma = np.full(num_rsus, NEUTRAL_SUCCESS_EWMA, dtype=np.float64)
        self.latency_dev = np.full(num_rsus, NEUTRAL_LATENCY_DEV, dtype=np.float64)
        # Diagnostic only - how much evidence each estimate rests on. Not a feature:
        # PROJECT_SPEC.md 5.1 fixes the feature set, and adding one is a decision.
        self.num_observations = np.zeros(num_rsus, dtype=np.int64)

    def observe(
        self,
        rsu: int,
        advertised_s: float,
        deadline_s: float,
        observed_s: float,
        met_deadline: bool,
    ) -> None:
        """Fold one outcome into the node's features.

        `observed_s` is the completion time the vehicle saw, which for a timeout is
        the deadline itself (see the module docstring).
        """
        w = self.config.ewma_weight
        kept = 1.0 if promise_kept(advertised_s, deadline_s, met_deadline) else 0.0
        cap = self.config.latency_dev_cap
        dev = min(relative_latency_deviation(advertised_s, observed_s), cap) / cap

        self.success_ewma[rsu] = (1.0 - w) * self.success_ewma[rsu] + w * kept
        self.latency_dev[rsu] = (1.0 - w) * self.latency_dev[rsu] + w * dev
        self.num_observations[rsu] += 1

    def reset(self, rsu: int) -> None:
        """Forget a node's history - a cold-start node joins with none."""
        self.success_ewma[rsu] = NEUTRAL_SUCCESS_EWMA
        self.latency_dev[rsu] = NEUTRAL_LATENCY_DEV
        self.num_observations[rsu] = 0

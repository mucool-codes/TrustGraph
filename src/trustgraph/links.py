"""The radio and backhaul link model.

Two quantities, both functions of distance, both named in PROJECT_SPEC.md 5.3:
`signal_strength` from a log-distance path-loss model, and `link_latency` from the
retransmission cost of a weak link. Vehicle-RSU (access) and RSU-RSU (backhaul) links
get separate latency bands because they are different media - one is a 5.9 GHz radio
hop, the other a fixed coordination link.

Kept in its own module because it is the piece most likely to be challenged as
unrealistic and therefore the piece most likely to be swapped. Nothing else in the
codebase encodes an assumption about radio propagation.

Note on L8: these are the *true* physical quantities. The advertised-vs-observed
discrepancy that actually drives trust is a separate thing, introduced with the task
and degradation model in S2. Nothing here lies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinkModel:
    """Distance-to-(signal, latency) for both link types.

    Defaults describe a 5.9 GHz DSRC/C-V2X roadside link: 23 dBm transmit, a 47.9 dB
    reference loss at one metre, and a path-loss exponent of 2.7 for an urban street
    canyon with partial line of sight.
    """

    tx_power_dbm: float = 23.0
    reference_loss_db: float = 47.9
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.7
    rssi_max_dbm: float = -40.0
    rssi_min_dbm: float = -95.0
    access_base_ms: float = 4.0
    access_span_ms: float = 16.0
    backhaul_base_ms: float = 2.0
    backhaul_span_ms: float = 4.0
    latency_norm_ms: float = 25.0

    def rssi_dbm(self, distance_m: np.ndarray) -> np.ndarray:
        """Received power under a log-distance path-loss model."""
        d = np.maximum(np.asarray(distance_m, dtype=np.float64), self.reference_distance_m)
        loss = self.reference_loss_db + 10.0 * self.path_loss_exponent * np.log10(
            d / self.reference_distance_m
        )
        return self.tx_power_dbm - loss

    def signal_strength(self, distance_m: np.ndarray) -> np.ndarray:
        """RSSI rescaled to [0, 1] between the sensitivity floor and a near-field cap.

        The model wants a bounded feature, and dBm is unbounded below. `rssi_min_dbm`
        is the receiver sensitivity - below it there is no link at all - so 0 means
        "at the edge of usability" rather than an arbitrary floor.
        """
        rssi = self.rssi_dbm(distance_m)
        span = self.rssi_max_dbm - self.rssi_min_dbm
        return np.clip((rssi - self.rssi_min_dbm) / span, 0.0, 1.0)

    def latency_ms(self, distance_m: np.ndarray, is_backhaul: bool) -> np.ndarray:
        """One-way latency in milliseconds.

        Latency rises as signal falls because a weak link costs retransmissions, and
        it rises faster than linearly for the same reason - each retry also waits out
        a longer contention window. The quadratic is a stand-in for that, not a
        derivation from one.
        """
        base = self.backhaul_base_ms if is_backhaul else self.access_base_ms
        span = self.backhaul_span_ms if is_backhaul else self.access_span_ms
        deficit = 1.0 - self.signal_strength(distance_m)
        return base + span * deficit**2

    def normalised_latency(
        self, distance_m: np.ndarray, is_backhaul: bool
    ) -> np.ndarray:
        """Latency scaled into [0, 1] for use as a model feature."""
        return np.clip(
            self.latency_ms(distance_m, is_backhaul) / self.latency_norm_ms, 0.0, 1.0
        )


def build_link_model(cfg_link: dict | None) -> LinkModel:
    """Construct the link model from the config, falling back to the defaults."""
    if not cfg_link:
        return LinkModel()
    known = {f for f in LinkModel.__dataclass_fields__}
    unknown = set(cfg_link) - known
    if unknown:
        raise ValueError(f"unknown link config keys: {sorted(unknown)}")
    return LinkModel(**{k: float(v) for k, v in cfg_link.items()})

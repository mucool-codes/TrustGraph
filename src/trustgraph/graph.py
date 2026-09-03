"""Dynamic graph construction.

One PyG `Data` object per timestep of a mobility trace. Single homogeneous edge type
covering both vehicle-RSU and RSU-RSU links, carrying a `same_segment` boolean edge
feature - no HeteroData (L6). Node features are a single matrix with a zero-padded RSU
block and vehicle block (DECISIONS.md D16).

What is real as of S1 and what is not
-------------------------------------
Real, derived from the trace and the geometry:
  `load`, `queue_depth`  - coverage demand at each RSU
  `speed`, `dwell_estimate` - from the vehicle's recorded motion
  `link_latency`, `signal_strength` - from the link model (`links.py`)
  `link_age`     - tracked across timesteps as links form and break
  `same_segment` - from `backhaul_segment_id` (L5/L6)

Placeholder, pending later sessions:
  `success_ewma`, `latency_dev`, `uptime_stability` - the behavioural features. They
    are constants here, not random values, so a result that depends on them is
    obviously degenerate rather than plausibly noisy. Real values need observed task
    outcomes and the advertised-vs-observed discrepancy of L8, which arrive with the
    task and degradation model.
  `cert_valid` - 1.0 everywhere. No revocation or compromise is modelled yet.
  `task_demand` - a fixed per-vehicle draw. Needs the task model.

Because `link_age` is a function of history rather than of the current timestep,
snapshots are produced by a stateful `SnapshotBuilder` walked forward in time, not by
a free function over an arbitrary timestep.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from .features import (
    EDGE_COL,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    RSU_COL,
    VEHICLE_FEATURES,
)
from .links import LinkModel
from .topology import Topology, pairwise_distances
from .trace import Trace

# Neutral values for the behavioural features until S2/S3 make them real. Under L8
# these are discrepancy quantities, so "no evidence of misbehaviour" is the identity:
# every task succeeded, observed latency matched what was advertised, no restarts.
PLACEHOLDER_SUCCESS_EWMA = 1.0
PLACEHOLDER_LATENCY_DEV = 0.0
PLACEHOLDER_UPTIME_STABILITY = 1.0

# No SCMS revocation is modelled yet, so every certificate is valid. Variant B reads
# this column as its entire trust term, and a constant is the honest value for a
# session with no compromised nodes in it.
PLACEHOLDER_CERT_VALID = 1.0


def dwell_estimate_s(
    position: np.ndarray,
    velocity: np.ndarray,
    centre: np.ndarray,
    radius_m: float,
) -> float:
    """Seconds until the vehicle leaves a coverage circle at its current velocity.

    A straight-line extrapolation: the vehicle is assumed to hold its heading until it
    crosses the circle. That is deliberately what a real `dwell_estimate` is - an
    estimate the vehicle itself could compute from its own speed and heading, which
    turns out wrong whenever it turns. It is a feature, not ground truth.

    Returns `inf` for a stationary vehicle, and 0.0 if it is already outside.
    """
    speed_sq = float(velocity @ velocity)
    offset = position - centre
    outside = float(offset @ offset) - radius_m**2
    if outside >= 0.0:
        return 0.0
    if speed_sq <= 1e-12:
        return float("inf")
    b = float(offset @ velocity)
    # outside < 0 guarantees a real positive root, so no discriminant guard is needed.
    return (-b + np.sqrt(b * b - speed_sq * outside)) / speed_sq


class SnapshotBuilder:
    """Turns a mobility trace into a sequence of PyG graphs.

    Holds the only state graph construction needs: when each currently-present link
    first appeared, which is what `link_age` measures. Timesteps must therefore be
    visited in order - `build(t)` accepts `t = 0` or one past the previous call, and
    `reset()` rewinds. `snapshots()` is the normal way in.
    """

    def __init__(
        self,
        topology: Topology,
        trace: Trace,
        link_model: LinkModel,
        cfg_graph: dict,
        rng: np.random.Generator,
    ) -> None:
        self.topology = topology
        self.trace = trace
        self.link_model = link_model

        self.rsu_capacity = float(cfg_graph.get("rsu_capacity_vehicles", 12.0))
        self.link_age_norm_s = float(cfg_graph.get("link_age_norm_s", 60.0))
        self.dwell_norm_s = float(cfg_graph.get("dwell_norm_s", 60.0))
        self.speed_norm_mps = float(cfg_graph.get("speed_norm_mps", 22.0))

        # Static per-vehicle placeholder, drawn once so vehicles are not identical.
        self._task_demand = rng.uniform(
            0.2, 1.0, size=trace.num_vehicles
        ).astype(np.float32)

        self._rsu_rsu_distance = np.linalg.norm(
            topology.positions[topology.rsu_edges[:, 0]]
            - topology.positions[topology.rsu_edges[:, 1]],
            axis=1,
        )
        self._same_segment = (
            topology.backhaul_segment_id[topology.rsu_edges[:, 0]]
            == topology.backhaul_segment_id[topology.rsu_edges[:, 1]]
        ).astype(np.float32)

        self.reset()

    # ------------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Forget link history and rewind to before timestep 0."""
        self._veh_first_seen = np.full(
            (self.trace.num_vehicles, self.topology.num_rsus), -1, dtype=np.int64
        )
        self._last_t: int | None = None

    # ------------------------------------------------------------------- internals

    def _rsu_features(self, in_range_count: np.ndarray) -> np.ndarray:
        """The RSU block of `x`, one row per RSU.

        `load` and `queue_depth` are both driven by coverage demand but are not the
        same quantity: load saturates at capacity, queue_depth is only the excess
        beyond it. A fully loaded RSU with nothing queued and one with a backlog are
        distinguishable, which is what the pair is for.
        """
        num_rsus = self.topology.num_rsus
        block = np.zeros((num_rsus, len(RSU_COL)), dtype=np.float32)
        demand = in_range_count / max(self.rsu_capacity, 1e-9)
        block[:, RSU_COL["load"]] = np.clip(demand, 0.0, 1.0)
        block[:, RSU_COL["queue_depth"]] = np.clip(demand - 1.0, 0.0, 1.0)
        block[:, RSU_COL["cert_valid"]] = PLACEHOLDER_CERT_VALID
        block[:, RSU_COL["success_ewma"]] = PLACEHOLDER_SUCCESS_EWMA
        block[:, RSU_COL["latency_dev"]] = PLACEHOLDER_LATENCY_DEV
        block[:, RSU_COL["uptime_stability"]] = PLACEHOLDER_UPTIME_STABILITY
        return block

    def _vehicle_features(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        serving: np.ndarray,
    ) -> np.ndarray:
        """The vehicle block of `x`, one row per vehicle."""
        num_vehicles = positions.shape[0]
        block = np.zeros((num_vehicles, len(VEHICLE_FEATURES)), dtype=np.float32)
        col = {name: i for i, name in enumerate(VEHICLE_FEATURES)}

        block[:, col["task_demand"]] = self._task_demand
        speed = np.linalg.norm(velocities, axis=1)
        block[:, col["speed"]] = np.clip(speed / self.speed_norm_mps, 0.0, 1.0)

        dwell = np.zeros(num_vehicles, dtype=np.float64)
        for v in np.flatnonzero(serving >= 0):
            dwell[v] = dwell_estimate_s(
                positions[v],
                velocities[v],
                self.topology.positions[serving[v]],
                self.topology.coverage_radius_m,
            )
        block[:, col["dwell_estimate"]] = np.clip(
            dwell / self.dwell_norm_s, 0.0, 1.0
        )
        return block

    # ---------------------------------------------------------------------- public

    def build(self, t: int) -> Data:
        """Build the graph for timestep `t` of the trace.

        Node ordering is RSUs first (indices `0 .. num_rsus-1`), then vehicles. This
        is relied on throughout: `data.is_rsu` marks the split and the selection rule
        indexes RSUs by their global node index.

        Edges are stored in both directions, so the single `edge_index` is effectively
        undirected and `SAGEConv` aggregates over true neighbourhoods.
        """
        t = int(t)
        expected = 0 if self._last_t is None else self._last_t + 1
        if t != expected:
            raise ValueError(
                f"SnapshotBuilder needs timesteps in order: expected t={expected}, "
                f"got t={t}. link_age is a function of history; call reset() to "
                "rewind."
            )
        if not 0 <= t < self.trace.num_steps:
            raise IndexError(f"timestep {t} outside trace of {self.trace.num_steps}")

        topo = self.topology
        num_rsus = topo.num_rsus
        positions = self.trace.positions[t]
        velocities = self.trace.velocities[t]
        num_vehicles = positions.shape[0]
        num_nodes = num_rsus + num_vehicles
        dt = self.trace.dt_s

        # --- coverage geometry ---------------------------------------------------
        veh_dist = pairwise_distances(positions, topo.positions)
        covered = veh_dist <= topo.coverage_radius_m

        # Serving RSU: the nearest one in range, -1 when the vehicle has no coverage.
        # Ties break to the lowest index so the choice is deterministic.
        masked = np.where(covered, veh_dist, np.inf)
        nearest = np.argmin(masked, axis=1)
        serving = np.where(covered.any(axis=1), nearest, -1).astype(np.int64)

        # --- link age ------------------------------------------------------------
        # A link that is present and was not present last step starts its clock now;
        # one that has broken forgets its history, so a re-formed link is genuinely
        # new and carries the extra uncertainty PROJECT_SPEC.md 5.3 describes.
        appeared = covered & (self._veh_first_seen < 0)
        self._veh_first_seen[appeared] = t
        self._veh_first_seen[~covered] = -1

        # --- node features -------------------------------------------------------
        x = np.zeros((num_nodes, NODE_FEATURE_DIM), dtype=np.float32)
        x[:num_rsus, : len(RSU_COL)] = self._rsu_features(covered.sum(axis=0))
        x[num_rsus:, len(RSU_COL) :] = self._vehicle_features(
            positions, velocities, serving
        )

        # --- edges ---------------------------------------------------------------
        # RSU <-> RSU (L6: required; without them segment evidence cannot propagate).
        rr_src = topo.rsu_edges[:, 0]
        rr_dst = topo.rsu_edges[:, 1]
        rr_dist = self._rsu_rsu_distance
        rr_age_s = np.full(rr_dist.shape, t * dt)  # static links, up since t=0
        rr_same = self._same_segment
        rr_backhaul = np.ones(rr_dist.shape, dtype=bool)

        # vehicle <-> RSU coverage links.
        vi, ri = np.nonzero(covered)
        vr_src = num_rsus + vi
        vr_dst = ri
        vr_dist = veh_dist[vi, ri]
        vr_age_s = (t - self._veh_first_seen[vi, ri]) * dt
        # A vehicle sits on no backhaul segment, so a vehicle-RSU link is never
        # same-segment.
        vr_same = np.zeros(vr_dist.shape, dtype=np.float32)
        vr_backhaul = np.zeros(vr_dist.shape, dtype=bool)

        src = np.concatenate([rr_src, vr_src])
        dst = np.concatenate([rr_dst, vr_dst])
        dist = np.concatenate([rr_dist, vr_dist])
        age_s = np.concatenate([rr_age_s, vr_age_s])
        same = np.concatenate([rr_same, vr_same])
        backhaul = np.concatenate([rr_backhaul, vr_backhaul])

        num_undirected = src.shape[0]
        attr = np.zeros((num_undirected, EDGE_FEATURE_DIM), dtype=np.float32)
        if num_undirected:
            latency = np.empty(num_undirected, dtype=np.float64)
            latency[backhaul] = self.link_model.normalised_latency(
                dist[backhaul], is_backhaul=True
            )
            latency[~backhaul] = self.link_model.normalised_latency(
                dist[~backhaul], is_backhaul=False
            )
            attr[:, EDGE_COL["link_latency"]] = latency
            attr[:, EDGE_COL["signal_strength"]] = self.link_model.signal_strength(
                dist
            )
            attr[:, EDGE_COL["link_age"]] = np.clip(
                age_s / self.link_age_norm_s, 0.0, 1.0
            )
            attr[:, EDGE_COL["same_segment"]] = same

        edge_index = torch.tensor(
            np.stack(
                [
                    np.concatenate([src, dst]),
                    np.concatenate([dst, src]),
                ]
            ),
            dtype=torch.long,
        )
        edge_attr = torch.from_numpy(np.concatenate([attr, attr], axis=0))

        is_rsu = torch.zeros(num_nodes, dtype=torch.bool)
        is_rsu[:num_rsus] = True

        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
        )
        data.is_rsu = is_rsu
        data.num_rsus = num_rsus
        data.timestep = t
        data.backhaul_segment_id = torch.from_numpy(topo.backhaul_segment_id)
        data.serving_rsu = torch.from_numpy(serving)
        # Geometry and latency are carried alongside the graph so the selection rule
        # reads a vehicle's candidates without recomputing any of it.
        data.vehicle_rsu_distance = torch.from_numpy(veh_dist.astype(np.float32))
        data.vehicle_rsu_latency_ms = torch.from_numpy(
            self.link_model.latency_ms(veh_dist, is_backhaul=False).astype(np.float32)
        )

        self._last_t = t
        return data

    def snapshots(self, num_steps: int | None = None):
        """Yield graphs for `t = 0 ...`, rewinding first.

        The normal way to consume a trace: `for data in builder.snapshots(): ...`
        """
        limit = self.trace.num_steps if num_steps is None else int(num_steps)
        limit = min(limit, self.trace.num_steps)
        self.reset()
        for t in range(limit):
            yield self.build(t)

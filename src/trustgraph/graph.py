"""Dynamic graph construction.

One PyG `Data` object per timestep of a mobility trace. Single homogeneous edge type
covering both vehicle-RSU and RSU-RSU links, carrying a `same_segment` boolean edge
feature - no HeteroData (L6). Node features are a single matrix with a zero-padded RSU
block and vehicle block (DECISIONS.md D16).

Where each feature comes from as of S2
--------------------------------------
From the observable scenario (`observed.py`), produced by the simulator:
  `load`, `queue_depth` - what each RSU *advertises* (L8). Equal to the true coverage
    demand for an honest or degraded node, below it for a colluder. The true values
    are sealed ground truth and never reach this module.
  `success_ewma`, `latency_dev` - advertised-vs-observed discrepancy from the tracker
    (`tracking.py`), updated only by outcomes already observed at that step.
  `cert_valid`, `uptime_stability` - still the constants named in `tracking.py`; no
    revocation or restart process exists yet (FINDINGS.md F8).
  `task_demand` - the offloading vehicle's task size, 0 for a vehicle with no task.
  which RSUs are present - a cold-start RSU has no edges until it joins.

From the trace and the geometry:
  `speed`, `dwell_estimate` - from the vehicle's recorded motion
  `link_latency`, `signal_strength` - from the link model (`links.py`)
  `link_age`     - tracked across timesteps as links form and break
  `same_segment` - from `backhaul_segment_id` (L5/L6)

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
    RSU_FEATURES,
    VEHICLE_FEATURES,
)
from .links import LinkModel
from .topology import Topology, pairwise_distances
from .trace import Trace


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
    """Turns a mobility trace plus an observable scenario into a sequence of PyG graphs.

    Holds the only state graph construction needs: when each currently-present link
    first appeared, which is what `link_age` measures. Timesteps must therefore be
    visited in order - `build(t)` accepts `t = 0` or one past the previous call, and
    `reset()` rewinds. `snapshots()` is the normal way in.

    Args:
        rsu_features: (num_steps, num_rsus, len(RSU_FEATURES)) the RSU block per step.
        rsu_active: (num_steps, num_rsus) which RSUs are present.
        vehicle_task_demand: (num_steps, num_vehicles) in [0, 1].
    """

    def __init__(
        self,
        topology: Topology,
        trace: Trace,
        link_model: LinkModel,
        cfg_graph: dict,
        rsu_features: np.ndarray,
        rsu_active: np.ndarray,
        vehicle_task_demand: np.ndarray,
    ) -> None:
        self.topology = topology
        self.trace = trace
        self.link_model = link_model

        expected = (trace.num_steps, topology.num_rsus, len(RSU_FEATURES))
        if rsu_features.shape != expected:
            raise ValueError(f"rsu_features has shape {rsu_features.shape}, want {expected}")
        if rsu_active.shape != expected[:2]:
            raise ValueError(f"rsu_active has shape {rsu_active.shape}, want {expected[:2]}")
        if vehicle_task_demand.shape != (trace.num_steps, trace.num_vehicles):
            raise ValueError(
                f"vehicle_task_demand has shape {vehicle_task_demand.shape}, want "
                f"{(trace.num_steps, trace.num_vehicles)}"
            )
        self.rsu_features = rsu_features
        self.rsu_active = rsu_active.astype(bool)
        self.vehicle_task_demand = vehicle_task_demand

        self.link_age_norm_s = float(cfg_graph.get("link_age_norm_s", 60.0))
        self.dwell_norm_s = float(cfg_graph.get("dwell_norm_s", 60.0))
        self.speed_norm_mps = float(cfg_graph.get("speed_norm_mps", 22.0))

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
        # When each RSU first became present. RSU-RSU links are static between present
        # RSUs, so a link's age runs from the later of its two endpoints' arrivals -
        # which is t=0 for every link unless a cold-start RSU is involved.
        self._rsu_first_active = np.full(self.topology.num_rsus, -1, dtype=np.int64)
        self._last_t: int | None = None

    # ------------------------------------------------------------------- internals

    def _vehicle_features(
        self,
        t: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        serving: np.ndarray,
    ) -> np.ndarray:
        """The vehicle block of `x`, one row per vehicle."""
        num_vehicles = positions.shape[0]
        block = np.zeros((num_vehicles, len(VEHICLE_FEATURES)), dtype=np.float32)
        col = {name: i for i, name in enumerate(VEHICLE_FEATURES)}

        block[:, col["task_demand"]] = self.vehicle_task_demand[t]
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
        indexes RSUs by their global node index. An RSU that has not joined yet keeps
        its row (so indices never shift) but has no edges and is marked inactive in
        `data.rsu_active`.

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
        active = self.rsu_active[t]

        # --- coverage geometry ---------------------------------------------------
        veh_dist = pairwise_distances(positions, topo.positions)
        covered = (veh_dist <= topo.coverage_radius_m) & active[None, :]

        # Serving RSU: the nearest present one in range, -1 when the vehicle has no
        # coverage. Ties break to the lowest index so the choice is deterministic.
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
        self._rsu_first_active[active & (self._rsu_first_active < 0)] = t

        # --- node features -------------------------------------------------------
        x = np.zeros((num_nodes, NODE_FEATURE_DIM), dtype=np.float32)
        x[:num_rsus, : len(RSU_FEATURES)] = self.rsu_features[t]
        x[num_rsus:, len(RSU_FEATURES) :] = self._vehicle_features(
            t, positions, velocities, serving
        )

        # --- edges ---------------------------------------------------------------
        # RSU <-> RSU (L6: required; without them segment evidence cannot propagate),
        # between present RSUs only.
        rr_keep = active[topo.rsu_edges[:, 0]] & active[topo.rsu_edges[:, 1]]
        rr_src = topo.rsu_edges[rr_keep, 0]
        rr_dst = topo.rsu_edges[rr_keep, 1]
        rr_dist = self._rsu_rsu_distance[rr_keep]
        rr_since = np.maximum(
            self._rsu_first_active[rr_src], self._rsu_first_active[rr_dst]
        )
        rr_age_s = (t - rr_since) * dt
        rr_same = self._same_segment[rr_keep]
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
        data.rsu_active = torch.from_numpy(active.copy())
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

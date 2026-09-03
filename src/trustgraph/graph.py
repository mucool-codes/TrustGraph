"""Dynamic graph construction.

One PyG `Data` object per timestep. Single homogeneous edge type covering both
vehicle-RSU and RSU-RSU links, carrying a `same_segment` boolean edge feature - no
HeteroData (L6). Node features are a single matrix with a zero-padded RSU block and
vehicle block (DECISIONS.md D16).

S0 fills the feature values with seeded random numbers. The names, order, and shapes
are the real ones from PROJECT_SPEC.md section 5; only the values are fake.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from .features import (
    EDGE_COL,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    RSU_BLOCK,
    RSU_FEATURES,
    VEHICLE_BLOCK,
    VEHICLE_FEATURES,
)
from .topology import Topology


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) euclidean distances."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def _fake_node_features(
    topology: Topology, num_vehicles: int, rng: np.random.Generator
) -> np.ndarray:
    """Random values in the correct columns (S0 placeholder).

    Real features arrive in a later session. `cert_valid` is drawn as a binary rather
    than a uniform so the column already has the right domain - Variant B reads it
    directly as its entire trust term.
    """
    num_rsus = topology.num_rsus
    x = np.zeros((num_rsus + num_vehicles, NODE_FEATURE_DIM), dtype=np.float32)

    rsu_block = rng.uniform(0.0, 1.0, size=(num_rsus, len(RSU_FEATURES)))
    rsu_block[:, RSU_FEATURES.index("cert_valid")] = rng.integers(
        0, 2, size=num_rsus
    ).astype(np.float64)
    x[:num_rsus, RSU_BLOCK] = rsu_block

    x[num_rsus:, VEHICLE_BLOCK] = rng.uniform(
        0.0, 1.0, size=(num_vehicles, len(VEHICLE_FEATURES))
    )
    return x


def build_snapshot(
    topology: Topology,
    vehicle_positions: np.ndarray,
    rng: np.random.Generator,
) -> Data:
    """Build the graph for one timestep.

    Node ordering is RSUs first (indices 0..num_rsus-1), then vehicles. This is relied
    on throughout: `data.is_rsu` marks the split and the selection rule indexes RSUs
    by their global node index.

    Edges are stored in both directions, so the single `edge_index` is effectively
    undirected and `SAGEConv` aggregates over true neighbourhoods.
    """
    num_rsus = topology.num_rsus
    num_vehicles = int(vehicle_positions.shape[0])
    num_nodes = num_rsus + num_vehicles

    x = _fake_node_features(topology, num_vehicles, rng)

    src: list[int] = []
    dst: list[int] = []
    dists: list[float] = []
    same_segment: list[float] = []

    # --- RSU <-> RSU links (L6: required, this is what lets segment evidence move) ---
    rsu_dist = _pairwise_distances(topology.positions, topology.positions)
    for i in range(num_rsus):
        for j in range(i + 1, num_rsus):
            if rsu_dist[i, j] <= topology.rsu_link_radius_m:
                shared = float(
                    topology.backhaul_segment_id[i] == topology.backhaul_segment_id[j]
                )
                for a, b in ((i, j), (j, i)):
                    src.append(a)
                    dst.append(b)
                    dists.append(float(rsu_dist[i, j]))
                    same_segment.append(shared)

    # --- vehicle <-> RSU coverage links ---
    veh_dist = _pairwise_distances(vehicle_positions, topology.positions)
    for v in range(num_vehicles):
        v_idx = num_rsus + v
        for r in range(num_rsus):
            if veh_dist[v, r] <= topology.coverage_radius_m:
                for a, b in ((v_idx, r), (r, v_idx)):
                    src.append(a)
                    dst.append(b)
                    dists.append(float(veh_dist[v, r]))
                    # A vehicle is not on any backhaul segment, so a vehicle-RSU link
                    # is never same-segment.
                    same_segment.append(0.0)

    num_edges = len(src)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = np.zeros((num_edges, EDGE_FEATURE_DIM), dtype=np.float32)
    if num_edges:
        distance = np.asarray(dists, dtype=np.float32)
        # Fake but monotone in distance, so the placeholder values are at least not
        # nonsense: latency rises with distance, signal strength falls.
        span = max(topology.coverage_radius_m, topology.rsu_link_radius_m)
        edge_attr[:, EDGE_COL["link_latency"]] = distance / span
        edge_attr[:, EDGE_COL["signal_strength"]] = 1.0 - np.clip(
            distance / span, 0.0, 1.0
        )
        edge_attr[:, EDGE_COL["link_age"]] = rng.uniform(0.0, 1.0, size=num_edges)
        edge_attr[:, EDGE_COL["same_segment"]] = np.asarray(
            same_segment, dtype=np.float32
        )

    is_rsu = torch.zeros(num_nodes, dtype=torch.bool)
    is_rsu[:num_rsus] = True

    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_attr=torch.from_numpy(edge_attr),
        num_nodes=num_nodes,
    )
    data.is_rsu = is_rsu
    data.num_rsus = num_rsus
    data.backhaul_segment_id = torch.from_numpy(topology.backhaul_segment_id)
    # Distances are kept alongside the graph so the selection rule can read a
    # vehicle's latency to each candidate without recomputing geometry.
    data.vehicle_rsu_distance = torch.from_numpy(veh_dist.astype(np.float32))
    return data

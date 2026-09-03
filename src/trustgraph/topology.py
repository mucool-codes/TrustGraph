"""RSU topology: placement and backhaul segments.

L5: each RSU is assigned a `backhaul_segment_id` at topology build time. Degradation
(later sessions) is applied to a SEGMENT, not to statistically correlated independent
nodes. S0 builds the segments but injects no degradation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Topology:
    """Fixed RSU layout for one run.

    Attributes:
        positions: (num_rsus, 2) float array of x/y positions in metres.
        backhaul_segment_id: (num_rsus,) int array. RSUs sharing an id share a
            backhaul link - the structural handle correlated failure uses (L5).
        coverage_radius_m: a vehicle links to an RSU within this distance.
        rsu_link_radius_m: two RSUs are linked within this distance (L6).
    """

    positions: np.ndarray
    backhaul_segment_id: np.ndarray
    coverage_radius_m: float
    rsu_link_radius_m: float

    @property
    def num_rsus(self) -> int:
        return int(self.positions.shape[0])


def build_topology(cfg_topology: dict, rng: np.random.Generator) -> Topology:
    """Place RSUs on a jittered lattice and assign backhaul segments.

    RSUs are spread over the region on an approximately square lattice with a small
    random offset, so the layout is neither perfectly regular nor degenerate. Segments
    are assigned in contiguous spatial runs (by x-then-y order) rather than at random,
    because a real backhaul link serves a geographic stretch of road - and a segment
    scattered uniformly across the region would give message passing nothing local to
    propagate along.
    """
    num_rsus = int(cfg_topology["num_rsus"])
    num_segments = int(cfg_topology["num_backhaul_segments"])
    extent_m = float(cfg_topology["region_extent_m"])
    jitter_m = float(cfg_topology.get("placement_jitter_m", 0.0))

    side = int(np.ceil(np.sqrt(num_rsus)))
    spacing = extent_m / max(side, 1)
    lattice = np.array(
        [[(i % side) * spacing, (i // side) * spacing] for i in range(num_rsus)],
        dtype=np.float64,
    )
    lattice += spacing / 2.0
    if jitter_m > 0:
        lattice += rng.uniform(-jitter_m, jitter_m, size=lattice.shape)
    positions = np.clip(lattice, 0.0, extent_m)

    # Contiguous spatial runs: sort by (x, y), then cut into near-equal blocks.
    order = np.lexsort((positions[:, 1], positions[:, 0]))
    segment_of_rank = np.array_split(np.arange(num_rsus), num_segments)
    segment_id = np.zeros(num_rsus, dtype=np.int64)
    for seg, ranks in enumerate(segment_of_rank):
        segment_id[order[ranks]] = seg

    return Topology(
        positions=positions,
        backhaul_segment_id=segment_id,
        coverage_radius_m=float(cfg_topology["coverage_radius_m"]),
        rsu_link_radius_m=float(cfg_topology["rsu_link_radius_m"]),
    )

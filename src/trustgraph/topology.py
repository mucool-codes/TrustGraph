"""RSU topology: placement on the road layout, backhaul segments, RSU-RSU edges.

Three things are fixed here, once per run, before any vehicle moves:

  * where the RSUs are (on the road network from `roads.py`),
  * which `backhaul_segment_id` each one carries (L5 - degradation in a later session
    is applied to a SEGMENT, never to statistically correlated independent nodes),
  * which RSU pairs are close enough to coordinate (L6 - RSU-RSU edges are required;
    without them segment-level evidence has no path to propagate along).

Segment assignment is spatially coherent by construction: a real backhaul link, power
feed, or software rollout serves a geographic stretch of road. That coherence is what
makes the structure learnable - `same_segment` on the edge tells message passing where
to look, and a segment scattered uniformly over the region would give it nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .roads import RoadNetwork


@dataclass(frozen=True)
class Topology:
    """Fixed RSU layout for one run.

    Attributes:
        positions: (num_rsus, 2) float array of x/y positions in metres.
        backhaul_segment_id: (num_rsus,) int array. RSUs sharing an id share a
            backhaul link - the structural handle correlated failure uses (L5).
        coverage_radius_m: a vehicle links to an RSU within this distance.
        rsu_link_radius_m: two RSUs are linked within this distance (L6).
        rsu_edges: (num_pairs, 2) int array of undirected RSU-RSU pairs (i < j),
            precomputed because the RSU layout never changes within a run.
        road: the layout the RSUs were placed on, kept so mobility and the plots read
            the same object.
    """

    positions: np.ndarray
    backhaul_segment_id: np.ndarray
    coverage_radius_m: float
    rsu_link_radius_m: float
    rsu_edges: np.ndarray
    road: RoadNetwork

    @property
    def num_rsus(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_segments(self) -> int:
        return int(self.backhaul_segment_id.max()) + 1 if self.num_rsus else 0

    def segment_sizes(self) -> np.ndarray:
        """(num_segments,) count of RSUs per segment."""
        return np.bincount(self.backhaul_segment_id, minlength=self.num_segments)


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) euclidean distances."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def _candidate_sites(road: RoadNetwork) -> np.ndarray:
    """Where an RSU may stand: every intersection and every segment midpoint.

    Mast sites in a real deployment are on the roadway, not at arbitrary coordinates,
    and intersections dominate because that is where power and duct already are.
    Midpoints are included so long blocks are not left uncovered.
    """
    return np.vstack([road.intersections, road.segment_midpoints()])


def _road_sample_points(road: RoadNetwork, spacing_m: float) -> np.ndarray:
    """Points along every road, roughly `spacing_m` apart.

    What coverage is measured against. The quantity an operator cares about is the
    fraction of *road* inside some RSU's radius, not the fraction of the region - the
    region is mostly buildings, and no vehicle is ever in them.
    """
    chunks: list[np.ndarray] = []
    for a, b in road.segments():
        pa, pb = road.intersections[a], road.intersections[b]
        steps = max(int(np.ceil(np.linalg.norm(pb - pa) / max(spacing_m, 1e-6))), 1)
        fractions = np.linspace(0.0, 1.0, steps + 1)[:-1, None]
        chunks.append(pa + fractions * (pb - pa))
    chunks.append(road.intersections)
    return np.vstack(chunks)


def _greedy_coverage_sites(
    sites: np.ndarray,
    samples: np.ndarray,
    radius_m: float,
    k: int,
    redundancy_decay: float,
) -> np.ndarray:
    """Choose `k` sites, greedily, to cover the most road.

    Each site's gain is the total weight of the road points it reaches, where a point
    already reached by `c` chosen sites is worth `redundancy_decay ** c`. So the first
    RSUs go where nothing is covered, and once the road is covered the remaining ones
    add a second layer over it rather than crowding one spot.

    That second layer is the point. The selection rule of L1 is an argmax over
    in-range candidates, so a vehicle that can see exactly one RSU produces a decision
    with nothing to decide - and an evaluation full of them measures coverage, not
    trust. Maximising bare coverage alone would spread the RSUs to minimise overlap
    and produce exactly that.

    Deterministic: `argmax` breaks ties to the lowest index, and no RNG is consulted.
    """
    if k > sites.shape[0]:
        raise ValueError(
            f"cannot place {k} RSUs on {sites.shape[0]} candidate sites; "
            "increase road.blocks_x / road.blocks_y or lower topology.num_rsus"
        )
    within = pairwise_distances(sites, samples) <= radius_m
    hits = np.zeros(samples.shape[0], dtype=np.int64)
    picked: list[int] = []

    for _ in range(k):
        gain = within @ (redundancy_decay**hits)
        gain[picked] = -np.inf
        best = int(np.argmax(gain))
        picked.append(best)
        hits[within[best]] += 1

    return np.array(picked, dtype=np.int64)


def _farthest_point_sample(points: np.ndarray, k: int, first: int) -> np.ndarray:
    """Greedy max-min subset of `points`, starting from index `first`.

    Used to seed the segment clustering below, where spreading the initial centroids
    as far apart as possible is exactly what is wanted.
    """
    if k > points.shape[0]:
        raise ValueError(
            f"cannot place {k} RSUs on {points.shape[0]} candidate sites; "
            "increase road.blocks_x / road.blocks_y or lower topology.num_rsus"
        )
    picked = [int(first)]
    best = np.linalg.norm(points - points[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(best))
        picked.append(nxt)
        best = np.minimum(best, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(picked, dtype=np.int64)


def _spatial_segments(
    positions: np.ndarray,
    num_segments: int,
    swap_prob: float,
    min_segment_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign a `backhaul_segment_id` per RSU, spatially coherent.

    Lloyd's algorithm over RSU positions with a farthest-point initialisation, so the
    segments come out as contiguous geographic regions of comparable size. Then, with
    probability `swap_prob`, an RSU is reassigned to a different segment.

    The swap noise is deliberate. With perfectly contiguous segments `same_segment` is
    a function of position alone, and a model could recover it from geometry without
    ever using the edge feature - which would make an H2 result about that feature
    unfalsifiable. A minority of off-region RSUs (a spur off a neighbouring backhaul
    link, a site re-homed during a build-out) keeps the segment id genuinely extra
    information. A swap is skipped when it would shrink its source segment below
    `min_segment_size`, so every segment keeps several RSUs as L5 requires.
    """
    num_rsus = positions.shape[0]
    if num_segments > num_rsus:
        raise ValueError(
            f"num_backhaul_segments={num_segments} exceeds num_rsus={num_rsus}"
        )

    seeds = _farthest_point_sample(positions, num_segments, first=0)
    centroids = positions[seeds].copy()
    labels = np.full(num_rsus, -1, dtype=np.int64)
    for _ in range(25):
        new_labels = np.argmin(pairwise_distances(positions, centroids), axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for s in range(num_segments):
            members = positions[labels == s]
            if members.size:
                centroids[s] = members.mean(axis=0)

    if swap_prob > 0.0 and num_segments > 1:
        draws = rng.random(num_rsus)
        alternatives = rng.integers(0, num_segments - 1, size=num_rsus)
        sizes = np.bincount(labels, minlength=num_segments)
        for i in np.flatnonzero(draws < swap_prob):
            source = int(labels[i])
            if sizes[source] <= min_segment_size:
                continue
            # Map into the segments other than the current one.
            target = int(alternatives[i])
            if target >= source:
                target += 1
            sizes[source] -= 1
            sizes[target] += 1
            labels[i] = target

    return labels.astype(np.int64)


def build_topology(
    cfg_topology: dict, road: RoadNetwork, rng: np.random.Generator
) -> Topology:
    """Place RSUs on the road layout and assign backhaul segments.

    `topology.positions` in the config, when present, is taken verbatim (a fixed
    deployment can be pinned exactly); otherwise `num_rsus` sites are chosen from the
    road network's intersections and segment midpoints.
    """
    num_segments = int(cfg_topology["num_backhaul_segments"])
    jitter_m = float(cfg_topology.get("placement_jitter_m", 0.0))

    explicit = cfg_topology.get("positions")
    if explicit:
        positions = np.asarray(explicit, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("topology.positions must be a list of [x, y] pairs")
        if positions.shape[0] != int(cfg_topology["num_rsus"]):
            raise ValueError("topology.positions length must equal topology.num_rsus")
    else:
        sites = _candidate_sites(road)
        samples = _road_sample_points(
            road, float(cfg_topology.get("site_sample_spacing_m", 25.0))
        )
        chosen = _greedy_coverage_sites(
            sites,
            samples,
            radius_m=float(cfg_topology["coverage_radius_m"]),
            k=int(cfg_topology["num_rsus"]),
            redundancy_decay=float(cfg_topology.get("redundancy_decay", 0.5)),
        )
        positions = sites[chosen].copy()
        if jitter_m > 0:
            # A mast stands beside the carriageway, not on the centreline. The jitter
            # is small relative to the coverage radius, so it perturbs the geometry
            # without changing which vehicles a site can see.
            positions += rng.uniform(-jitter_m, jitter_m, size=positions.shape)
        positions = np.clip(positions, 0.0, road.extent_m)

    segment_id = _spatial_segments(
        positions,
        num_segments=num_segments,
        swap_prob=float(cfg_topology.get("segment_swap_prob", 0.0)),
        min_segment_size=int(cfg_topology.get("min_segment_size", 2)),
        rng=rng,
    )

    rsu_link_radius_m = float(cfg_topology["rsu_link_radius_m"])
    dist = pairwise_distances(positions, positions)
    i, j = np.triu_indices(positions.shape[0], k=1)
    close = dist[i, j] <= rsu_link_radius_m
    rsu_edges = np.stack([i[close], j[close]], axis=1).astype(np.int64)

    return Topology(
        positions=positions,
        backhaul_segment_id=segment_id,
        coverage_radius_m=float(cfg_topology["coverage_radius_m"]),
        rsu_link_radius_m=rsu_link_radius_m,
        rsu_edges=rsu_edges,
        road=road,
    )

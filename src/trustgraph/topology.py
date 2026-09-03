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
        num_swapped_rsus: how many RSUs are homed off the segment pure geometry would
            have given them. Carried on the topology because it is the one property of
            the segment assignment that cannot be seen in the topology figure - that
            plot colours by assigned segment, so a swapped RSU looks like any other
            (DECISIONS.md D22).
    """

    positions: np.ndarray
    backhaul_segment_id: np.ndarray
    coverage_radius_m: float
    rsu_link_radius_m: float
    rsu_edges: np.ndarray
    road: RoadNetwork
    num_swapped_rsus: int = 0

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


def _adjacency(num_rsus: int, rsu_edges: np.ndarray) -> list[set[int]]:
    """RSU-RSU neighbour sets, indexed by RSU id."""
    adjacency: list[set[int]] = [set() for _ in range(num_rsus)]
    for a, b in rsu_edges:
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    return adjacency


def _segment_stays_connected(
    labels: np.ndarray,
    adjacency: list[set[int]],
    segment: int,
    without: int,
) -> bool:
    """Would `segment` still be one component over same_segment edges without `without`?"""
    members = {
        i for i in np.flatnonzero(labels == segment).tolist() if i != without
    }
    if len(members) <= 1:
        return True
    seen: set[int] = set()
    stack = [min(members)]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend((adjacency[node] & members) - seen)
    return seen == members


def _admissible_swap(
    i: int,
    labels: np.ndarray,
    sizes: np.ndarray,
    adjacency: list[set[int]],
    centroids: np.ndarray,
    position: np.ndarray,
    min_segment_size: int,
) -> int | None:
    """Which segment RSU `i` may be re-homed to, or None.

    Two constraints, both of which exist to keep every segment a single component over
    same_segment edges - the condition that lets segment-level evidence propagate one
    hop rather than having to detour through a cross-segment neighbour:

      * `i` must already have an RSU-RSU edge to a member of the target segment. A
        site cannot be homed to a backhaul link it has no physical path to, and
        without this the swapped RSU lands in its new segment as an isolated node.
      * removing `i` must not split its source segment.

    Among the targets that qualify, the nearest centroid wins, so the result stays as
    close to the geometric assignment as the constraints allow.
    """
    source = int(labels[i])
    if sizes[source] <= min_segment_size:
        return None

    reachable = {int(labels[n]) for n in adjacency[i]} - {source}
    if not reachable:
        return None
    if not _segment_stays_connected(labels, adjacency, source, without=i):
        return None

    candidates = sorted(reachable)
    distances = [float(np.linalg.norm(centroids[s] - position)) for s in candidates]
    return candidates[int(np.argmin(distances))]


def _spatial_segments(
    positions: np.ndarray,
    num_segments: int,
    swap_prob: float,
    min_segment_size: int,
    min_swap_fraction: float,
    adjacency: list[set[int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Assign a `backhaul_segment_id` per RSU, spatially coherent.

    Returns the labels and the number of RSUs homed off their geographic segment.

    Lloyd clustering over RSU positions with a farthest-point initialisation, so the
    segments come out as contiguous geographic regions of comparable size. Then two
    passes reassign a minority of RSUs to a different segment: a probabilistic one at
    rate `swap_prob`, and a deterministic floor that tops the count up to
    `min_swap_fraction` of the RSUs if the first pass fell short.

    Why any swaps at all: without them a segment is exactly a region of the map, and
    the paper's claim under correlated failure reduces to "things that are near each
    other fail together" - which needs no message passing to exploit and no segment id
    to express. Off-region RSUs are physically ordinary (a spur off a neighbouring
    backhaul link, a site re-homed during a build-out) and make segment membership
    something other than a restatement of position, which is what lets a GNN gain be
    attributed to shared *infrastructure* rather than to proximity.

    Why a floor rather than a rate: at `swap_prob = 0.1` over 20 RSUs the realised
    count is a binomial draw, and it comes out zero often enough to matter - seed 4
    produced no swaps at all, which would have made the property silently false for
    one seed of an evaluation that averages over seeds (FINDINGS.md F6). The floor
    makes it a guarantee. Forced swaps take the RSUs nearest their segment boundary
    first, so the assignment stays as close to the geometric one as the floor allows.

    A swap of either kind is skipped when it would shrink its source segment below
    `min_segment_size`, so every segment keeps several RSUs as L5 requires. That means
    the floor is best-effort on layouts too small to satisfy it; the achieved count is
    returned rather than asserted here, and `Topology.num_swapped_rsus` carries it.
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

    geometric = labels.copy()
    sizes = np.bincount(labels, minlength=num_segments)

    def apply(i: int) -> bool:
        target = _admissible_swap(
            i, labels, sizes, adjacency, centroids, positions[i], min_segment_size
        )
        if target is None:
            return False
        sizes[int(labels[i])] -= 1
        sizes[target] += 1
        labels[i] = target
        return True

    if num_segments > 1 and swap_prob > 0.0:
        for i in np.flatnonzero(rng.random(num_rsus) < swap_prob):
            apply(int(i))

    swapped = labels != geometric
    floor = int(np.ceil(min_swap_fraction * num_rsus))

    if num_segments > 1 and swapped.sum() < floor:
        # Distance from each RSU to its own segment centroid and to the nearest other
        # one. The difference is how far inside its region the RSU sits, so the
        # smallest values are the sites on a boundary - the ones a real operator would
        # plausibly have homed either way.
        distance = pairwise_distances(positions, centroids)
        rows = np.arange(num_rsus)
        own = distance[rows, geometric]
        across = distance.copy()
        across[rows, geometric] = np.inf
        margin = across.min(axis=1) - own

        for i in np.argsort(margin, kind="stable"):
            if swapped.sum() >= floor:
                break
            if swapped[i]:
                continue
            if apply(int(i)):
                swapped[i] = True

    return labels.astype(np.int64), int(swapped.sum())


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

    # Coordination edges are computed before segments are assigned, because segment
    # assignment needs them: an RSU may only be re-homed to a segment it actually has
    # a link to (see `_admissible_swap`).
    rsu_link_radius_m = float(cfg_topology["rsu_link_radius_m"])
    dist = pairwise_distances(positions, positions)
    i, j = np.triu_indices(positions.shape[0], k=1)
    close = dist[i, j] <= rsu_link_radius_m
    rsu_edges = np.stack([i[close], j[close]], axis=1).astype(np.int64)

    segment_id, num_swapped = _spatial_segments(
        positions,
        num_segments=num_segments,
        swap_prob=float(cfg_topology.get("segment_swap_prob", 0.0)),
        min_segment_size=int(cfg_topology.get("min_segment_size", 2)),
        min_swap_fraction=float(cfg_topology.get("min_swap_fraction", 0.15)),
        adjacency=_adjacency(positions.shape[0], rsu_edges),
        rng=rng,
    )

    return Topology(
        positions=positions,
        backhaul_segment_id=segment_id,
        coverage_radius_m=float(cfg_topology["coverage_radius_m"]),
        rsu_link_radius_m=rsu_link_radius_m,
        rsu_edges=rsu_edges,
        road=road,
        num_swapped_rsus=num_swapped,
    )

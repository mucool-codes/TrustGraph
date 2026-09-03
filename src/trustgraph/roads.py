"""The road layout.

A Manhattan-style grid of arterials over a square region: `blocks_x + 1` vertical
roads crossed by `blocks_y + 1` horizontal roads. Intersections are the waypoints of
the mobility model (`mobility.py`) and the candidate sites for RSU placement
(`topology.py`), so both read the same object and cannot drift apart.

Deliberately not a real map. L7 puts synthetic mobility on the critical path and SUMO
off it; a grid is the standard synthetic road model in the VANET literature and is
enough to produce the thing S1 actually needs - vehicles that follow roads, enter and
leave coverage zones, and hand off between RSUs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoadNetwork:
    """A grid of roads over `[0, extent_m] x [0, extent_m]`.

    Attributes:
        intersections: (num_intersections, 2) float array of x/y in metres, indexed by
            intersection id. Row-major: id = iy * (blocks_x + 1) + ix.
        neighbours: for each intersection id, the tuple of adjacent intersection ids
            (the road segments leaving it). 2 at corners, 3 on an edge, 4 inside.
        blocks_x, blocks_y: number of blocks along each axis.
        extent_m: side length of the square region in metres.
    """

    intersections: np.ndarray
    neighbours: tuple[tuple[int, ...], ...]
    blocks_x: int
    blocks_y: int
    extent_m: float

    @property
    def num_intersections(self) -> int:
        return int(self.intersections.shape[0])

    @property
    def spacing_x_m(self) -> float:
        return self.extent_m / self.blocks_x

    @property
    def spacing_y_m(self) -> float:
        return self.extent_m / self.blocks_y

    def grid_coords(self, node: int) -> tuple[int, int]:
        """(ix, iy) lattice coordinates of an intersection id."""
        cols = self.blocks_x + 1
        return int(node % cols), int(node // cols)

    def segments(self) -> list[tuple[int, int]]:
        """Every road segment once, as (lower id, higher id) pairs."""
        return [
            (a, b)
            for a, nbrs in enumerate(self.neighbours)
            for b in nbrs
            if a < b
        ]

    def segment_midpoints(self) -> np.ndarray:
        """(num_segments, 2) midpoint of every road segment."""
        seg = self.segments()
        if not seg:
            return np.zeros((0, 2), dtype=np.float64)
        a = self.intersections[[s[0] for s in seg]]
        b = self.intersections[[s[1] for s in seg]]
        return 0.5 * (a + b)


def build_road_network(cfg_road: dict) -> RoadNetwork:
    """Build the grid described by the config. No randomness - layout is fixed."""
    blocks_x = int(cfg_road["blocks_x"])
    blocks_y = int(cfg_road["blocks_y"])
    extent_m = float(cfg_road["extent_m"])

    if blocks_x < 1 or blocks_y < 1:
        raise ValueError("road.blocks_x and road.blocks_y must both be >= 1")
    if extent_m <= 0:
        raise ValueError("road.extent_m must be positive")

    cols, rows = blocks_x + 1, blocks_y + 1
    xs = np.linspace(0.0, extent_m, cols)
    ys = np.linspace(0.0, extent_m, rows)
    points = np.array(
        [[xs[ix], ys[iy]] for iy in range(rows) for ix in range(cols)],
        dtype=np.float64,
    )

    neighbours: list[tuple[int, ...]] = []
    for iy in range(rows):
        for ix in range(cols):
            adj: list[int] = []
            if ix > 0:
                adj.append(iy * cols + ix - 1)
            if ix < cols - 1:
                adj.append(iy * cols + ix + 1)
            if iy > 0:
                adj.append((iy - 1) * cols + ix)
            if iy < rows - 1:
                adj.append((iy + 1) * cols + ix)
            neighbours.append(tuple(adj))

    return RoadNetwork(
        intersections=points,
        neighbours=tuple(neighbours),
        blocks_x=blocks_x,
        blocks_y=blocks_y,
        extent_m=extent_m,
    )

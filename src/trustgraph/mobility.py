"""Mobility sources.

L7: mobility sits behind a `MobilitySource` interface. The synthetic implementation is
the default and is on the critical path; SUMO is an optional later swap and is NOT on
the critical path. Nothing here imports or depends on SUMO.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MobilitySource(ABC):
    """Supplies vehicle positions over time.

    A source yields an (num_vehicles, 2) float array of x/y positions in metres for
    each timestep. Implementations must be fully determined by the generator handed to
    them at construction, so a run reproduces from (config, seed) alone.
    """

    @property
    @abstractmethod
    def num_vehicles(self) -> int: ...

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Return positions at t=0 and rewind any internal state."""

    @abstractmethod
    def step(self) -> np.ndarray:
        """Advance one timestep and return the new positions."""


class SyntheticGridMobility(MobilitySource):
    """Vehicles random-walking on a square grid.

    Deliberately crude - this is the walking skeleton's fake mobility (S0). Each
    vehicle sits on grid intersections and, at each step, moves one cell in one of the
    four cardinal directions, or holds. Movement is reflected at the boundary so the
    population stays inside the region.
    """

    _MOVES = np.array(
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64
    )

    def __init__(
        self,
        num_vehicles: int,
        grid_cells: int,
        cell_size_m: float,
        rng: np.random.Generator,
    ) -> None:
        self._num_vehicles = int(num_vehicles)
        self._grid_cells = int(grid_cells)
        self._cell_size_m = float(cell_size_m)
        self._rng = rng
        self._cells = self._initial_cells()

    @property
    def num_vehicles(self) -> int:
        return self._num_vehicles

    def _initial_cells(self) -> np.ndarray:
        return self._rng.integers(
            0, self._grid_cells, size=(self._num_vehicles, 2), dtype=np.int64
        )

    def _to_metres(self) -> np.ndarray:
        return self._cells.astype(np.float64) * self._cell_size_m

    def reset(self) -> np.ndarray:
        self._cells = self._initial_cells()
        return self._to_metres()

    def step(self) -> np.ndarray:
        choice = self._rng.integers(0, len(self._MOVES), size=self._num_vehicles)
        self._cells = self._cells + self._MOVES[choice]
        # Reflect at the boundary rather than clipping, so vehicles do not pile up
        # along the edges of the region.
        upper = self._grid_cells - 1
        self._cells = np.abs(self._cells)
        over = self._cells > upper
        self._cells[over] = 2 * upper - self._cells[over]
        return self._to_metres()


def build_mobility(cfg_mobility: dict, rng: np.random.Generator) -> MobilitySource:
    """Construct the mobility source named in the config.

    Only `synthetic_grid` exists in S0. SUMO is not installed and not integrated (L7).
    """
    kind = cfg_mobility.get("source", "synthetic_grid")
    if kind == "synthetic_grid":
        return SyntheticGridMobility(
            num_vehicles=cfg_mobility["num_vehicles"],
            grid_cells=cfg_mobility["grid_cells"],
            cell_size_m=cfg_mobility["cell_size_m"],
            rng=rng,
        )
    raise ValueError(
        f"unknown mobility source {kind!r} (S0 provides only 'synthetic_grid'; "
        "SUMO is off the critical path per L7)"
    )

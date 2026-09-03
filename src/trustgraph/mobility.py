"""Mobility sources.

L7: mobility sits behind a `MobilitySource` interface. The synthetic implementation is
the default and is on the critical path; SUMO is an optional later swap and is NOT on
the critical path. Nothing here imports or depends on SUMO.

A source's whole contract is to produce a `Trace` (`trace.py`) for a fixed horizon.
It is never stepped by the pipeline: the trace is written to disk once, and every
consumer reads it back. That keeps a single realisation of vehicle motion shared by
graph construction, statistics, plots, and later training and evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .roads import RoadNetwork
from .trace import Trace


class MobilitySource(ABC):
    """Produces a mobility trace over a road layout.

    Implementations must be fully determined by the road network and the generator
    handed to them at construction, so a trace reproduces from (config, seed) alone
    (Standing Rule 7).
    """

    @property
    @abstractmethod
    def num_vehicles(self) -> int: ...

    @property
    @abstractmethod
    def dt_s(self) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str:
        """The `mobility.source` value that selects this implementation."""

    @abstractmethod
    def generate(self, num_steps: int, seed: int) -> Trace:
        """Simulate `num_steps` timesteps and return the trace."""


class SyntheticRoadMobility(MobilitySource):
    """Vehicles driving the road grid on waypoint trajectories.

    Each vehicle occupies a road segment and drives toward the intersection at its
    far end. On arrival it may pause (a signalised junction), then picks the next
    segment - continuing straight by preference, turning otherwise, and reversing only
    at a dead end. The sequence of intersections a vehicle visits is its waypoint
    trajectory; between waypoints it moves in a straight line at its own speed.

    Speeds are per-vehicle draws from a truncated normal with small per-step jitter,
    so the population has a spread of speeds rather than one shared value. That spread
    is what makes dwell time in a coverage zone a distribution instead of a constant,
    and dwell time is what determines how often handoff actually happens.
    """

    def __init__(
        self,
        road: RoadNetwork,
        cfg_mobility: dict,
        rng: np.random.Generator,
    ) -> None:
        self._road = road
        self._rng = rng
        self._num_vehicles = int(cfg_mobility["num_vehicles"])
        self._dt_s = float(cfg_mobility["dt_s"])

        self._speed_mean = float(cfg_mobility["speed_mean_mps"])
        self._speed_std = float(cfg_mobility["speed_std_mps"])
        self._speed_min = float(cfg_mobility["speed_min_mps"])
        self._speed_max = float(cfg_mobility["speed_max_mps"])
        self._speed_jitter = float(cfg_mobility.get("speed_jitter_mps", 0.0))

        self._stop_prob = float(cfg_mobility.get("intersection_stop_prob", 0.0))
        self._stop_mean_s = float(cfg_mobility.get("intersection_stop_mean_s", 0.0))
        self._straight_pref = float(cfg_mobility.get("straight_preference", 1.0))

        if self._dt_s <= 0:
            raise ValueError("mobility.dt_s must be positive")
        if not 0.0 < self._speed_min <= self._speed_max:
            raise ValueError("require 0 < speed_min_mps <= speed_max_mps")

    @property
    def num_vehicles(self) -> int:
        return self._num_vehicles

    @property
    def dt_s(self) -> float:
        return self._dt_s

    @property
    def name(self) -> str:
        return "synthetic_road"

    # ------------------------------------------------------------------ internals

    def _draw_speeds(self) -> np.ndarray:
        """Per-vehicle desired speed, truncated to the configured band."""
        raw = self._rng.normal(
            self._speed_mean, self._speed_std, size=self._num_vehicles
        )
        return np.clip(raw, self._speed_min, self._speed_max)

    def _spawn(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Place vehicles part-way along randomly chosen road segments.

        Spawning mid-segment rather than at intersections avoids an artificial
        synchronised first handoff, where every vehicle would reach its next junction
        at nearly the same moment.
        """
        segments = self._road.segments()
        pick = self._rng.integers(0, len(segments), size=self._num_vehicles)
        flip = self._rng.random(self._num_vehicles) < 0.5

        from_node = np.empty(self._num_vehicles, dtype=np.int64)
        to_node = np.empty(self._num_vehicles, dtype=np.int64)
        for v in range(self._num_vehicles):
            a, b = segments[int(pick[v])]
            from_node[v], to_node[v] = (b, a) if flip[v] else (a, b)

        progress = self._rng.random(self._num_vehicles)  # fraction along the segment
        return from_node, to_node, progress

    def _next_node(self, came_from: int, at: int) -> int:
        """Pick the next waypoint, preferring to carry straight on."""
        options = [n for n in self._road.neighbours[at] if n != came_from]
        if not options:
            return came_from  # dead end: turn around

        ax, ay = self._road.grid_coords(came_from)
        bx, by = self._road.grid_coords(at)
        heading = (bx - ax, by - ay)

        weights = np.empty(len(options), dtype=np.float64)
        for k, n in enumerate(options):
            cx, cy = self._road.grid_coords(n)
            straight = (cx - bx, cy - by) == heading
            weights[k] = self._straight_pref if straight else 1.0
        weights /= weights.sum()
        return int(options[int(self._rng.choice(len(options), p=weights))])

    # -------------------------------------------------------------------- public

    def generate(self, num_steps: int, seed: int) -> Trace:
        num_steps = int(num_steps)
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        nodes = self._road.intersections
        from_node, to_node, progress = self._spawn()
        desired = self._draw_speeds()
        pause_left = np.zeros(self._num_vehicles, dtype=np.int64)

        seg_len = np.linalg.norm(nodes[to_node] - nodes[from_node], axis=1)
        travelled = progress * seg_len

        def current_positions() -> np.ndarray:
            frac = (travelled / np.maximum(seg_len, 1e-9))[:, None]
            return nodes[from_node] + frac * (nodes[to_node] - nodes[from_node])

        positions = np.empty((num_steps, self._num_vehicles, 2), dtype=np.float64)
        velocities = np.zeros((num_steps, self._num_vehicles, 2), dtype=np.float64)
        positions[0] = current_positions()

        for t in range(1, num_steps):
            previous = positions[t - 1]

            jitter = (
                self._rng.normal(0.0, self._speed_jitter, size=self._num_vehicles)
                if self._speed_jitter > 0
                else np.zeros(self._num_vehicles)
            )
            speed = np.clip(desired + jitter, self._speed_min, self._speed_max)
            distance = speed * self._dt_s

            # Vehicles held at a junction do not advance this step.
            held = pause_left > 0
            pause_left[held] -= 1
            distance[held] = 0.0

            for v in np.flatnonzero(~held):
                remaining = float(distance[v])
                # A step can cross more than one junction only if a block is shorter
                # than one step of travel; the loop handles it either way.
                while remaining > 0.0:
                    to_end = seg_len[v] - travelled[v]
                    if remaining < to_end:
                        travelled[v] += remaining
                        break
                    remaining -= to_end
                    arrived = int(to_node[v])
                    nxt = self._next_node(int(from_node[v]), arrived)
                    from_node[v] = arrived
                    to_node[v] = nxt
                    seg_len[v] = float(
                        np.linalg.norm(nodes[nxt] - nodes[arrived])
                    )
                    travelled[v] = 0.0
                    if self._stop_prob > 0 and self._rng.random() < self._stop_prob:
                        hold_s = self._rng.exponential(self._stop_mean_s)
                        pause_left[v] = max(1, int(round(hold_s / self._dt_s)))
                        break

            positions[t] = current_positions()
            # Velocity is the realised displacement, so it is exactly consistent with
            # the recorded positions - including through a turn and across a pause.
            velocities[t] = (positions[t] - previous) / self._dt_s

        # t=0 has no preceding step; use the first realised velocity so speed is not
        # spuriously zero for every vehicle at the start of the trace.
        if num_steps > 1:
            velocities[0] = velocities[1]

        return Trace(
            positions=positions,
            velocities=velocities,
            dt_s=self._dt_s,
            seed=int(seed),
            source=self.name,
        )


def build_mobility(
    cfg_mobility: dict, road: RoadNetwork, rng: np.random.Generator
) -> MobilitySource:
    """Construct the mobility source named in the config.

    Only `synthetic_road` exists. SUMO is not installed and not integrated (L7); when
    it is, it becomes another `MobilitySource` returning a `Trace` and nothing
    downstream changes.
    """
    kind = cfg_mobility.get("source", "synthetic_road")
    if kind == "synthetic_road":
        return SyntheticRoadMobility(road=road, cfg_mobility=cfg_mobility, rng=rng)
    raise ValueError(
        f"unknown mobility source {kind!r} (only 'synthetic_road' exists; "
        "SUMO is off the critical path per L7)"
    )

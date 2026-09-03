"""Mobility traces on disk.

A trace is the complete record of where every vehicle was at every timestep, written
once by `scripts/generate_trace.py` and read back by everything downstream. Nothing in
the pipeline advances a mobility model inside its own loop.

The separation is not bookkeeping. It means the graph sequence, the statistics, the
plots, and (later) training and evaluation all see the *same* vehicle motion for a
given seed, rather than each re-deriving it and diverging the moment one of them takes
an extra RNG draw. It is also what makes L7 cheap: a SUMO source would write a trace in
this format and every consumer would be unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRACE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Trace:
    """Vehicle motion over a fixed horizon.

    Attributes:
        positions: (num_steps, num_vehicles, 2) x/y in metres.
        velocities: (num_steps, num_vehicles, 2) x/y in metres per second. Kept
            rather than scalar speed because `dwell_estimate` needs the direction of
            travel, not just its magnitude.
        dt_s: seconds per timestep.
        seed: the master seed the trace was generated from.
        source: the `mobility.source` that produced it.
    """

    positions: np.ndarray
    velocities: np.ndarray
    dt_s: float
    seed: int
    source: str

    @property
    def num_steps(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_vehicles(self) -> int:
        return int(self.positions.shape[1])

    @property
    def duration_s(self) -> float:
        return self.num_steps * self.dt_s

    @property
    def speeds(self) -> np.ndarray:
        """(num_steps, num_vehicles) speed magnitude in m/s."""
        return np.linalg.norm(self.velocities, axis=-1)

    def save(self, path: str | Path) -> Path:
        """Write the trace to a compressed .npz and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            format_version=np.int64(TRACE_FORMAT_VERSION),
            positions=self.positions.astype(np.float64),
            velocities=self.velocities.astype(np.float64),
            dt_s=np.float64(self.dt_s),
            seed=np.int64(self.seed),
            source=np.str_(self.source),
        )
        return path


def load_trace(path: str | Path) -> Trace:
    """Read a trace written by `Trace.save`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no mobility trace at {path}. Traces are generated once and read from "
            "disk, never regenerated in a loop - run:\n"
            "    python scripts/generate_trace.py --config <your config>"
        )
    with np.load(path, allow_pickle=False) as blob:
        version = int(blob["format_version"])
        if version != TRACE_FORMAT_VERSION:
            raise ValueError(
                f"trace {path} is format version {version}, this build reads "
                f"version {TRACE_FORMAT_VERSION}; regenerate it"
            )
        return Trace(
            positions=blob["positions"],
            velocities=blob["velocities"],
            dt_s=float(blob["dt_s"]),
            seed=int(blob["seed"]),
            source=str(blob["source"]),
        )


def default_trace_path(config_path: str | Path, seed: int) -> Path:
    """Where a trace for (config, seed) lives by default.

    Named after both, because the trace is a pure function of them and two seeds must
    never silently share one file. `traces/` is gitignored - a trace is a generated
    artefact and is reproduced from the config, not committed.
    """
    stem = Path(config_path).stem
    return Path("traces") / f"{stem}-seed{int(seed)}.npz"

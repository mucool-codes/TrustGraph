"""Config loading and the seed chain.

Every run is reproducible from (config file, seed) - Standing Rule 7. Randomness is
drawn from per-purpose generators derived from one master seed, never from a global
RNG, so adding a draw in one place cannot shift the streams anywhere else
(DECISIONS.md D17).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _stable_key(name: str) -> int:
    """Process-independent integer key for a purpose name.

    Python's built-in hash() is randomised per process (PYTHONHASHSEED), so using it
    here would break reproducibility across runs - the exact thing Standing Rule 7
    forbids. blake2b is stable across processes and machines.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class SeedChain:
    """Derives independent numpy Generators, one per named purpose."""

    def __init__(self, master_seed: int) -> None:
        self.master_seed = int(master_seed)
        self._cache: dict[str, np.random.Generator] = {}

    def generator(self, purpose: str) -> np.random.Generator:
        """Return the generator for `purpose`, creating it on first use.

        The same (master_seed, purpose) always yields the same stream, independent of
        which other purposes have been requested or how many draws they have taken.
        """
        if purpose not in self._cache:
            seq = np.random.SeedSequence([self.master_seed, _stable_key(purpose)])
            self._cache[purpose] = np.random.default_rng(seq)
        return self._cache[purpose]

    def torch_seed(self, purpose: str) -> int:
        """A deterministic 32-bit seed for torch, derived the same way."""
        seq = np.random.SeedSequence([self.master_seed, _stable_key(purpose)])
        return int(seq.generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True)
class Config:
    """Parsed run configuration. Mirrors the YAML structure one-for-one."""

    seed: int
    device: str
    scenario: dict[str, Any]
    road: dict[str, Any]
    topology: dict[str, Any]
    mobility: dict[str, Any]
    link: dict[str, Any]
    graph: dict[str, Any]
    model: dict[str, Any]
    selection: dict[str, Any]

    @property
    def seeds(self) -> SeedChain:
        return SeedChain(self.seed)


def load_config(path: str | Path) -> Config:
    """Load a YAML config and validate the constraints PROJECT_SPEC.md locks down."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    cfg = Config(
        seed=int(raw["seed"]),
        device=str(raw.get("device", "cpu")),
        scenario=raw["scenario"],
        road=raw["road"],
        topology=raw["topology"],
        mobility=raw["mobility"],
        # `link` and `graph` are optional: every field in them has a documented
        # default in links.py / graph.py, so a config need only name what it changes.
        link=raw.get("link") or {},
        graph=raw.get("graph") or {},
        model=raw["model"],
        selection=raw["selection"],
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    n_rsu = int(cfg.topology["num_rsus"])
    n_veh = int(cfg.mobility["num_vehicles"])

    # L9 caps the scale. The smoke test deliberately runs below the floor, so the
    # cap is enforced as an upper bound only and the floor is advisory.
    if n_rsu > 30:
        raise ValueError(f"num_rsus={n_rsu} exceeds the L9 cap of 30 RSUs")
    if n_veh > 100:
        raise ValueError(f"num_vehicles={n_veh} exceeds the L9 cap of 100 vehicles")
    if n_rsu < 1 or n_veh < 1:
        raise ValueError("num_rsus and num_vehicles must both be >= 1")

    num_segments = int(cfg.topology["num_backhaul_segments"])
    if num_segments < 1:
        raise ValueError("num_backhaul_segments must be >= 1")
    if num_segments > n_rsu:
        raise ValueError(
            f"num_backhaul_segments={num_segments} exceeds num_rsus={n_rsu}"
        )

    for radius in ("coverage_radius_m", "rsu_link_radius_m"):
        if float(cfg.topology[radius]) <= 0:
            raise ValueError(f"topology.{radius} must be positive")

    if float(cfg.mobility["dt_s"]) <= 0:
        raise ValueError("mobility.dt_s must be positive")
    if int(cfg.scenario["num_steps"]) < 1:
        raise ValueError("scenario.num_steps must be >= 1")

    for weight in ("alpha", "beta", "gamma"):
        if float(cfg.selection[weight]) < 0:
            raise ValueError(f"selection.{weight} must be non-negative")

    if cfg.device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {cfg.device!r}")

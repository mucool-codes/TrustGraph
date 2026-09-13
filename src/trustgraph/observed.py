"""The observable scenario: everything a deployed system could actually see.

This is what training reads (L3, L4). It holds, per timestep, the RSU feature block in
canonical order - advertised load and queue depth, the behavioural features from the
advertised-vs-observed tracker, `cert_valid`, `uptime_stability` - plus which RSUs are
present and which vehicles offloaded, and the task stream with each task's advertised
and observed completion time and whether it met its deadline.

It holds no behaviour class, no true load, no collusion membership, no degradation
state, and no true completion time for a timed-out task. Those live in the sealed file
written beside it (`trustgraph.sealed.ground_truth`), which this module does not import.
`load_observed` refuses a file containing any key it does not expect, so a sealed field
added to the wrong file fails on load rather than riding along.

The graph sequence is not stored: it is rebuilt from this file, the mobility trace, and
the static world, by `graph.SnapshotBuilder`. One source of truth per quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .features import RSU_FEATURES

OBSERVED_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TaskStream:
    """Every task generated in the scenario, in generation order (step, then vehicle).

    Attributes:
        step, vehicle: when and by whom the task was generated.
        task_type: index into `tasks.TASK_TYPES`.
        cycles, deadline_s: the task itself.
        rsu: the RSU it was dispatched to, or -1 if the vehicle had no active RSU in
            range (unserved - excluded from success rates).
        num_candidates: active RSUs in range when it was dispatched.
        advertised_s: completion time the chosen node's advertisement promised.
        observed_s: completion time the vehicle observed; the deadline for a timeout.
        met_deadline: the L3 self-supervised target.
        observed_at_s: absolute time the outcome became known to the vehicle - the
            earliest time it may influence any feature.
    """

    step: np.ndarray
    vehicle: np.ndarray
    task_type: np.ndarray
    cycles: np.ndarray
    deadline_s: np.ndarray
    rsu: np.ndarray
    num_candidates: np.ndarray
    advertised_s: np.ndarray
    observed_s: np.ndarray
    met_deadline: np.ndarray
    observed_at_s: np.ndarray

    def __len__(self) -> int:
        return int(self.step.shape[0])

    @property
    def dispatched(self) -> np.ndarray:
        return self.rsu >= 0

    def success_rate(self, mask: np.ndarray | None = None) -> float:
        """Deadline success over dispatched tasks, optionally restricted by `mask`."""
        use = self.dispatched if mask is None else (self.dispatched & mask)
        n = int(use.sum())
        return float(self.met_deadline[use].mean()) if n else float("nan")


TASK_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(TaskStream))


@dataclass(frozen=True)
class ObservedScenario:
    """The observable half of a generated scenario.

    Attributes:
        rsu_features: (num_steps, num_rsus, len(RSU_FEATURES)) float32, columns in
            `features.RSU_FEATURES` order. Row `t` is the state *before* any task of
            step `t` is dispatched, and contains only outcomes observed at or before
            time `t * dt`.
        rsu_active: (num_steps, num_rsus) bool. A cold-start RSU is inactive until it
            joins; its presence is observable, its behaviour is not.
        vehicle_task_demand: (num_steps, num_vehicles) float32, the offloading
            vehicle's task cycles over the largest task type, 0 if no task that step.
        tasks: the task stream.
        seed: master seed of the run.
        dispatch_policy: the rule that chose each task's RSU (DECISIONS.md D33).
    """

    rsu_features: np.ndarray
    rsu_active: np.ndarray
    vehicle_task_demand: np.ndarray
    tasks: TaskStream
    seed: int
    dispatch_policy: str

    @property
    def num_steps(self) -> int:
        return int(self.rsu_features.shape[0])

    @property
    def num_rsus(self) -> int:
        return int(self.rsu_features.shape[1])

    def feature(self, name: str) -> np.ndarray:
        """(num_steps, num_rsus) one named RSU feature over time."""
        return self.rsu_features[:, :, RSU_FEATURES.index(name)]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observed_format_version=np.int64(OBSERVED_FORMAT_VERSION),
            rsu_features=self.rsu_features,
            rsu_active=self.rsu_active,
            vehicle_task_demand=self.vehicle_task_demand,
            seed=np.int64(self.seed),
            dispatch_policy=np.str_(self.dispatch_policy),
            **{f"task_{name}": getattr(self.tasks, name) for name in TASK_FIELDS},
        )
        return path


_EXPECTED_KEYS = frozenset(
    {
        "observed_format_version",
        "rsu_features",
        "rsu_active",
        "vehicle_task_demand",
        "seed",
        "dispatch_policy",
    }
    | {f"task_{name}" for name in TASK_FIELDS}
)


def load_observed(path: str | Path) -> ObservedScenario:
    """Read an observable scenario, refusing anything that is not purely observable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no scenario at {path}. Scenarios are generated once and read from disk "
            "- run:\n    python scripts/generate_scenario.py --config <your config>"
        )
    with np.load(path, allow_pickle=False) as blob:
        unexpected = set(blob.files) - _EXPECTED_KEYS
        if unexpected:
            raise ValueError(
                f"{path} contains non-observable keys {sorted(unexpected)}; an "
                "observable scenario file must never carry ground truth (L4)"
            )
        version = int(blob["observed_format_version"])
        if version != OBSERVED_FORMAT_VERSION:
            raise ValueError(
                f"scenario {path} is format version {version}, this build reads "
                f"version {OBSERVED_FORMAT_VERSION}; regenerate it"
            )
        tasks = TaskStream(**{name: blob[f"task_{name}"] for name in TASK_FIELDS})
        return ObservedScenario(
            rsu_features=blob["rsu_features"],
            rsu_active=blob["rsu_active"],
            vehicle_task_demand=blob["vehicle_task_demand"],
            tasks=tasks,
            seed=int(blob["seed"]),
            dispatch_policy=str(blob["dispatch_policy"]),
        )


def scenario_stem(config_path: str | Path, seed: int, rho: float, fraction: float) -> str:
    """File stem for a scenario: config, seed, and the two swept parameters (L12)."""
    return (
        f"{Path(config_path).stem}-seed{int(seed)}"
        f"-rho{rho:.2f}-frac{fraction:.2f}"
    )


def default_observed_path(
    config_path: str | Path, seed: int, rho: float, fraction: float
) -> Path:
    """Where the observable scenario lives. `scenarios/` is gitignored."""
    return Path("scenarios") / f"{scenario_stem(config_path, seed, rho, fraction)}.observed.npz"

"""The sealed ground-truth object (L4).

`GroundTruth` holds two kinds of thing:

  * the injection plan, fixed before the first timestep: behaviour class per RSU, the
    degradation onset and ramp, collusion membership, cold-start join times, and the
    realised structure of the degraded set;
  * the realised hidden state, filled in by the simulator: true load per RSU per step,
    and each task's true completion time and whether it was silently dropped.

It is written to its own file, separate from the observable scenario, so that the
thing a training loop reads from disk physically does not contain it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

SEALED_FORMAT_VERSION = 1

BEHAVIOR_CLASSES: tuple[str, ...] = ("reliable", "degraded", "compromised")
RELIABLE, DEGRADED, COMPROMISED = 0, 1, 2


@dataclass(frozen=True)
class GroundTruth:
    """What the scenario really is. SEALED - evaluation only.

    Attributes:
        behavior_class: (num_rsus,) int, index into `BEHAVIOR_CLASSES`. Static for the
            run: a degraded node is class `degraded` before its onset too, and
            `degradation_level` says how far the fault has progressed at a given step.
        backhaul_segment_id: (num_rsus,) copy of the topology's assignment, so the
            ground truth can be read without rebuilding the world.
        rho, degraded_fraction: the injector's parameters.
        onset_step, ramp_steps: degradation begins at `onset_step` and reaches full
            severity `ramp_steps` later, linearly. Shared by every degraded node.
        severity: fraction of compute capacity lost at full degradation.
        drop_prob: probability of a silent task drop at full degradation.
        num_correlated_picks: how many of the degraded picks took the segment branch
            of the rho interpolation (DECISIONS.md D29). The realised, not nominal,
            correlation.
        collusion_group: (num_rsus,) int, the group id or -1.
        under_report: colluders advertise `(1 - under_report) * true load`.
        join_step: (num_rsus,) int, the step a node becomes active; 0 unless cold-start.
        true_load, true_queue_depth: (num_steps, num_rsus) realised hidden state.
        task_true_completion_s: (num_tasks,) true completion time, `inf` if dropped.
        task_dropped: (num_tasks,) bool.
    """

    behavior_class: np.ndarray
    backhaul_segment_id: np.ndarray
    rho: float
    degraded_fraction: float
    onset_step: int
    ramp_steps: int
    severity: float
    drop_prob: float
    num_correlated_picks: int
    collusion_group: np.ndarray
    under_report: float
    join_step: np.ndarray
    true_load: np.ndarray | None = None
    true_queue_depth: np.ndarray | None = None
    task_true_completion_s: np.ndarray | None = None
    task_dropped: np.ndarray | None = None

    # ------------------------------------------------------------------ membership

    @property
    def num_rsus(self) -> int:
        return int(self.behavior_class.shape[0])

    @property
    def degraded(self) -> np.ndarray:
        return self.behavior_class == DEGRADED

    @property
    def colluding(self) -> np.ndarray:
        return self.collusion_group >= 0

    @property
    def cold_start(self) -> np.ndarray:
        return self.join_step > 0

    def segment_degraded_count(self) -> np.ndarray:
        """(num_segments,) degraded RSUs per backhaul segment."""
        num_segments = int(self.backhaul_segment_id.max()) + 1
        return np.bincount(
            self.backhaul_segment_id[self.degraded], minlength=num_segments
        )

    def degraded_pair_concentration(self) -> float:
        """Fraction of degraded-node pairs that share a backhaul segment.

        The realised correlation of the degraded set. `nan` with fewer than two
        degraded nodes, where the notion is undefined - which is itself a finding
        about the low end of the fraction sweep (FINDINGS.md F10).
        """
        seg = self.backhaul_segment_id[self.degraded]
        k = seg.size
        if k < 2:
            return float("nan")
        counts = np.bincount(seg)
        same = float((counts * (counts - 1) / 2).sum())
        return same / (k * (k - 1) / 2)

    # ------------------------------------------------------------ time-varying state

    def degradation_level(self, step: int) -> np.ndarray:
        """(num_rsus,) fault progress in [0, 1]: zero before onset, linear over the ramp.

        Gradual by construction. With `ramp_steps = R`, the level one step after
        onset is `1/R`, not 1 - the property the H2 test depends on, since a fault
        that lands all at once leaves nothing for neighbourhood evidence to anticipate.
        """
        progress = np.clip((int(step) - self.onset_step) / self.ramp_steps, 0.0, 1.0)
        return np.where(self.degraded, progress, 0.0)

    def capacity_factor(self, step: int) -> np.ndarray:
        return 1.0 - self.severity * self.degradation_level(step)

    def drop_probability(self, step: int) -> np.ndarray:
        return self.drop_prob * self.degradation_level(step)

    def active(self, step: int) -> np.ndarray:
        return self.join_step <= int(step)

    # ------------------------------------------------------------------ persistence

    def with_realised(
        self,
        true_load: np.ndarray,
        true_queue_depth: np.ndarray,
        task_true_completion_s: np.ndarray,
        task_dropped: np.ndarray,
    ) -> "GroundTruth":
        return replace(
            self,
            true_load=true_load,
            true_queue_depth=true_queue_depth,
            task_true_completion_s=task_true_completion_s,
            task_dropped=task_dropped,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.true_load is None:
            raise ValueError("refusing to save a GroundTruth with no realised state")
        np.savez_compressed(
            path,
            sealed_format_version=np.int64(SEALED_FORMAT_VERSION),
            behavior_class=self.behavior_class,
            backhaul_segment_id=self.backhaul_segment_id,
            rho=np.float64(self.rho),
            degraded_fraction=np.float64(self.degraded_fraction),
            onset_step=np.int64(self.onset_step),
            ramp_steps=np.int64(self.ramp_steps),
            severity=np.float64(self.severity),
            drop_prob=np.float64(self.drop_prob),
            num_correlated_picks=np.int64(self.num_correlated_picks),
            collusion_group=self.collusion_group,
            under_report=np.float64(self.under_report),
            join_step=self.join_step,
            true_load=self.true_load,
            true_queue_depth=self.true_queue_depth,
            task_true_completion_s=self.task_true_completion_s,
            task_dropped=self.task_dropped,
        )
        return path


def load_ground_truth(path: str | Path) -> GroundTruth:
    path = Path(path)
    with np.load(path, allow_pickle=False) as blob:
        version = int(blob["sealed_format_version"])
        if version != SEALED_FORMAT_VERSION:
            raise ValueError(
                f"sealed file {path} is version {version}, expected "
                f"{SEALED_FORMAT_VERSION}; regenerate it"
            )
        return GroundTruth(
            behavior_class=blob["behavior_class"],
            backhaul_segment_id=blob["backhaul_segment_id"],
            rho=float(blob["rho"]),
            degraded_fraction=float(blob["degraded_fraction"]),
            onset_step=int(blob["onset_step"]),
            ramp_steps=int(blob["ramp_steps"]),
            severity=float(blob["severity"]),
            drop_prob=float(blob["drop_prob"]),
            num_correlated_picks=int(blob["num_correlated_picks"]),
            collusion_group=blob["collusion_group"],
            under_report=float(blob["under_report"]),
            join_step=blob["join_step"],
            true_load=blob["true_load"],
            true_queue_depth=blob["true_queue_depth"],
            task_true_completion_s=blob["task_true_completion_s"],
            task_dropped=blob["task_dropped"],
        )


def sealed_path_for(observed_path: str | Path) -> Path:
    """The sealed file that sits beside an observable scenario file."""
    observed_path = Path(observed_path)
    stem = observed_path.name.removesuffix(".observed.npz")
    return observed_path.with_name(f"{stem}.SEALED.npz")

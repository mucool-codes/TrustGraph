"""The scenario generator: (config, seed, trace) -> observable scenario + sealed ground truth.

The research instrument. It walks the mobility trace one timestep at a time and, at
each step:

  1. works out coverage among the RSUs that are present (cold-start nodes are absent
     until they join);
  2. computes each RSU's true load from coverage demand, and what it *advertises* -
     the truth for an honest or degraded node, an under-report for a colluder (L8);
  3. folds into the tracker every task outcome that became observable by now, and
     records the RSU feature block the graph for this step will carry;
  4. draws the step's task arrivals and dispatches each one;
  5. executes each task on its node's true state and schedules its outcome to be
     observed at `dispatch time + observed completion time`.

Step 3 precedes step 4 on purpose: a feature at step `t` can only depend on outcomes
already observed, never on a task dispatched at `t` or later.

WORLD SIDE. This module necessarily imports the sealed ground truth - it is the thing
that decides what goes wrong - and is therefore on the allowlist in
`tests/test_sealing.py`. Nothing on the training path may import it; training reads the
observable file this writes.

Dispatch (DECISIONS.md D33, and the tension flagged in FINDINGS.md F9): tasks go to
the RSU chosen by the L1 rule with `alpha = 0` - Baseline A, trust-agnostic - using
advertised load and measured link latency. This is the only policy S2 implements.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .config import Config
from .execution import build_execution_model, coverage_load, round_trip_comm_s
from .features import RSU_COL, RSU_FEATURES
from .observed import ObservedScenario, TaskStream
from .scenario import World
from .sealed.ground_truth import GroundTruth
from .sealed.injector import INJECTION_PURPOSES, inject, injection_stream_name
from .selection import select
from .tasks import build_task_model
from .topology import pairwise_distances
from .trace import Trace
from .tracking import (
    PLACEHOLDER_CERT_VALID,
    PLACEHOLDER_UPTIME_STABILITY,
    DiscrepancyTracker,
    build_tracking_config,
)

DISPATCH_POLICIES: tuple[str, ...] = ("baseline_a",)


@dataclass(frozen=True)
class Scenario:
    observed: ObservedScenario
    ground_truth: GroundTruth


def check_trace_matches(cfg: Config, trace: Trace) -> None:
    """Refuse a trace that was generated for a different config or seed.

    `generate_trace.py` never overwrites without `--force`, so a trace left over from
    an older config would otherwise be picked up silently - and the S2 horizon differs
    from S1's.
    """
    problems = []
    if trace.num_steps != int(cfg.scenario["num_steps"]):
        problems.append(f"num_steps {trace.num_steps} != {cfg.scenario['num_steps']}")
    if trace.num_vehicles != int(cfg.mobility["num_vehicles"]):
        problems.append(
            f"num_vehicles {trace.num_vehicles} != {cfg.mobility['num_vehicles']}"
        )
    if not np.isclose(trace.dt_s, float(cfg.mobility["dt_s"])):
        problems.append(f"dt_s {trace.dt_s} != {cfg.mobility['dt_s']}")
    if trace.seed != cfg.seed:
        problems.append(f"seed {trace.seed} != {cfg.seed}")
    if problems:
        raise ValueError(
            "trace does not match the config (" + "; ".join(problems) + "). "
            "Regenerate it: python scripts/generate_trace.py --config <config> --force"
        )


def generate_scenario(cfg: Config, world: World, trace: Trace) -> Scenario:
    """Simulate the task stream and outcomes over `trace`."""
    check_trace_matches(cfg, trace)

    policy = str((cfg.dispatch or {}).get("policy", "baseline_a"))
    if policy not in DISPATCH_POLICIES:
        raise ValueError(
            f"unknown dispatch.policy {policy!r}; S2 implements only {DISPATCH_POLICIES}"
        )

    topo = world.topology
    link = world.link_model
    task_model = build_task_model(cfg.tasks)
    exec_model = build_execution_model(cfg.execution)
    tracker = DiscrepancyTracker(topo.num_rsus, build_tracking_config(cfg.tracking))

    seeds = cfg.seeds
    plan = inject(
        topo.backhaul_segment_id,
        cfg.degradation,
        cfg.collusion,
        cfg.cold_start,
        rngs={
            purpose: seeds.generator(injection_stream_name(purpose, cfg.injection_draw))
            for purpose in INJECTION_PURPOSES
        },
    )
    task_rng = seeds.generator("tasks")
    exec_rng = seeds.generator("execution")

    beta = float(cfg.selection["beta"])
    gamma = float(cfg.selection["gamma"])

    num_steps, num_vehicles, num_rsus = trace.num_steps, trace.num_vehicles, topo.num_rsus
    dt = trace.dt_s
    colluding = plan.colluding

    rsu_features = np.zeros((num_steps, num_rsus, len(RSU_FEATURES)), dtype=np.float32)
    rsu_active = np.zeros((num_steps, num_rsus), dtype=bool)
    task_demand = np.zeros((num_steps, num_vehicles), dtype=np.float32)
    true_load_series = np.zeros((num_steps, num_rsus), dtype=np.float32)
    true_queue_series = np.zeros((num_steps, num_rsus), dtype=np.float32)

    records: dict[str, list] = {name: [] for name in TaskStream.__dataclass_fields__}
    true_completion: list[float] = []
    dropped_flags: list[bool] = []
    pending: list[tuple[float, int]] = []  # (observed_at_s, task index)

    for t in range(num_steps):
        now_s = t * dt
        active = plan.active(t)
        for r in np.flatnonzero(plan.join_step == t):
            if t > 0:
                tracker.reset(int(r))  # joins with no history

        # --- 1. coverage over present RSUs -----------------------------------------
        veh_dist = pairwise_distances(trace.positions[t], topo.positions)
        covered = (veh_dist <= topo.coverage_radius_m) & active[None, :]

        # --- 2. true and advertised load -------------------------------------------
        load, queue = coverage_load(covered.sum(axis=0), exec_model.rsu_capacity_vehicles)
        adv_load = np.where(colluding, (1.0 - plan.under_report) * load, load)
        adv_queue = np.where(colluding, (1.0 - plan.under_report) * queue, queue)
        true_load_series[t] = load
        true_queue_series[t] = queue

        # --- 3. outcomes observed so far -> features for this step ------------------
        while pending and pending[0][0] <= now_s:
            _, idx = heapq.heappop(pending)
            tracker.observe(
                rsu=records["rsu"][idx],
                advertised_s=records["advertised_s"][idx],
                deadline_s=records["deadline_s"][idx],
                observed_s=records["observed_s"][idx],
                met_deadline=records["met_deadline"][idx],
            )

        block = rsu_features[t]
        block[:, RSU_COL["load"]] = np.where(active, adv_load, 0.0)
        block[:, RSU_COL["queue_depth"]] = np.where(active, adv_queue, 0.0)
        block[:, RSU_COL["cert_valid"]] = PLACEHOLDER_CERT_VALID
        block[:, RSU_COL["success_ewma"]] = tracker.success_ewma
        block[:, RSU_COL["latency_dev"]] = tracker.latency_dev
        block[:, RSU_COL["uptime_stability"]] = PLACEHOLDER_UPTIME_STABILITY
        rsu_active[t] = active

        # --- 4. arrivals and dispatch -----------------------------------------------
        # Fixed-size draws every step, whatever happens: the task stream and the
        # execution noise are then identical across runs that differ only in which
        # nodes misbehave, which makes rho and fraction comparisons paired.
        draws = task_model.draw_step(num_vehicles, dt, task_rng)
        noise_z = exec_rng.standard_normal(num_vehicles)
        drop_u = exec_rng.random(num_vehicles)

        latency_ms = link.latency_ms(veh_dist, is_backhaul=False)
        latency_norm = np.clip(latency_ms / link.latency_norm_ms, 0.0, 1.0)
        capacity_factor = plan.capacity_factor(t)
        drop_prob = plan.drop_probability(t)

        for v in np.flatnonzero(draws.offloads):
            type_idx = int(draws.type_idx[v])
            cycles = float(task_model.cycles[type_idx])
            deadline = float(task_model.deadline_s(type_idx, draws.slack[v]))
            task_demand[t, v] = cycles / task_model.max_cycles
            candidates = np.flatnonzero(covered[v])

            records["step"].append(t)
            records["vehicle"].append(int(v))
            records["task_type"].append(type_idx)
            records["cycles"].append(cycles)
            records["deadline_s"].append(deadline)
            records["num_candidates"].append(int(candidates.size))

            if candidates.size == 0:
                for name, value in (
                    ("rsu", -1),
                    ("advertised_s", np.nan),
                    ("observed_s", np.nan),
                    ("met_deadline", False),
                    ("observed_at_s", np.nan),
                ):
                    records[name].append(value)
                true_completion.append(np.nan)
                dropped_flags.append(False)
                continue

            decision = select(
                timestep=t,
                vehicle_id=int(v),
                candidates=candidates,
                trust=np.zeros(candidates.size),
                latency=latency_norm[v, candidates],
                load=adv_load[candidates],
                alpha=0.0,
                beta=beta,
                gamma=gamma,
            )
            r = decision.chosen_rsu
            comm_s = float(round_trip_comm_s(latency_ms[v, r]))

            # --- 5. execute on the true state ---------------------------------------
            advertised = float(
                exec_model.advertised_completion_s(cycles, comm_s, adv_load[r])
            )
            completion, dropped = exec_model.true_completion_s(
                cycles,
                comm_s,
                load[r],
                capacity_factor[r],
                drop_prob[r],
                noise_z[v],
                drop_u[v],
            )
            completion = float(completion)
            met = bool(completion <= deadline)
            observed = completion if met else deadline

            idx = len(records["rsu"])
            records["rsu"].append(int(r))
            records["advertised_s"].append(advertised)
            records["observed_s"].append(observed)
            records["met_deadline"].append(met)
            records["observed_at_s"].append(now_s + observed)
            true_completion.append(completion)
            dropped_flags.append(bool(dropped))
            heapq.heappush(pending, (now_s + observed, idx))

    dtypes = {
        "step": np.int64,
        "vehicle": np.int64,
        "task_type": np.int64,
        "cycles": np.float64,
        "deadline_s": np.float64,
        "rsu": np.int64,
        "num_candidates": np.int64,
        "advertised_s": np.float64,
        "observed_s": np.float64,
        "met_deadline": bool,
        "observed_at_s": np.float64,
    }
    tasks = TaskStream(
        **{name: np.asarray(records[name], dtype=dtypes[name]) for name in dtypes}
    )

    observed = ObservedScenario(
        rsu_features=rsu_features,
        rsu_active=rsu_active,
        vehicle_task_demand=task_demand,
        tasks=tasks,
        seed=cfg.seed,
        dispatch_policy=policy,
    )
    ground_truth = plan.with_realised(
        true_load=true_load_series,
        true_queue_depth=true_queue_series,
        task_true_completion_s=np.asarray(true_completion, dtype=np.float64),
        task_dropped=np.asarray(dropped_flags, dtype=bool),
    )
    return Scenario(observed=observed, ground_truth=ground_truth)

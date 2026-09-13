"""First calibration measurement - run BEFORE any tuning, recorded verbatim in FINDINGS."""

import sys
import time

import numpy as np

sys.path.insert(0, "src")
from trustgraph.config import load_config, with_overrides
from trustgraph.scenario import build_world
from trustgraph.sealed.ground_truth import DEGRADED
from trustgraph.simulator import generate_scenario
from trustgraph.tasks import TASK_TYPES
from trustgraph.trace import default_trace_path, load_trace

config = sys.argv[1] if len(sys.argv) > 1 else "configs/demo.yaml"
base = load_config(config)
trace = load_trace(default_trace_path(config, base.seed))
world = build_world(base)

for label, frac, rho in (("no degradation", 0.0, 0.0), ("20% rho=0", 0.2, 0.0), ("20% rho=1", 0.2, 1.0)):
    cfg = with_overrides(base, fraction=frac, rho=rho)
    t0 = time.perf_counter()
    s = generate_scenario(cfg, world, trace)
    elapsed = time.perf_counter() - t0
    tasks, gt = s.observed.tasks, s.ground_truth
    onset, ramp = gt.onset_step, gt.ramp_steps
    pre = tasks.step < onset
    post = tasks.step >= onset + ramp
    on_degraded = np.zeros(len(tasks), dtype=bool)
    d = tasks.dispatched
    on_degraded[d] = gt.behavior_class[tasks.rsu[d]] == DEGRADED
    print(f"== {label}  ({elapsed:.1f}s)")
    print(f"   tasks {len(tasks)}  dispatched {int(d.sum())}  unserved {int((~d).sum())}")
    print(f"   success overall        {tasks.success_rate():.4f}")
    for i, name in enumerate(TASK_TYPES):
        print(f"   success {name:<6}         {tasks.success_rate(tasks.task_type == i):.4f}  "
              f"(n={int((d & (tasks.task_type == i)).sum())}, mean deadline "
              f"{tasks.deadline_s[tasks.task_type == i].mean():.3f}s)")
    print(f"   success pre-onset      {tasks.success_rate(pre):.4f}")
    print(f"   success post-ramp      {tasks.success_rate(post):.4f}")
    if frac > 0:
        print(f"   post-ramp on degraded  {tasks.success_rate(post & on_degraded):.4f}  "
              f"(n={int((post & on_degraded).sum())})")
        print(f"   post-ramp on healthy   {tasks.success_rate(post & ~on_degraded & d):.4f}")
        print(f"   degraded per segment   {gt.segment_degraded_count().tolist()}  "
              f"correlated picks {gt.num_correlated_picks}")
    per_rsu = np.bincount(tasks.rsu[d], minlength=world.topology.num_rsus)
    print(f"   tasks per RSU          min {per_rsu.min()} median {int(np.median(per_rsu))} max {per_rsu.max()}")
    print(f"   true load mean         {gt.true_load.mean():.3f}")
    ewma = s.observed.feature("success_ewma")
    print(f"   success_ewma end       healthy {ewma[-1][~gt.degraded].mean():.3f}  "
          f"degraded {ewma[-1][gt.degraded].mean() if gt.degraded.any() else float('nan'):.3f}")

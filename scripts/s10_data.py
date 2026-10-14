"""S10 data stage: the two figure inputs that need closed-loop runs (Figure 2 timeline, Figure 6 worked example).

Called by `make_figures.py` (cached under results/s10/, gitignored; `make_figures.py --recompute` rebuilds).

  timeline.npz   Variant A and Variant D, closed loop, no retuning (D52 weights):
                   * the single illustrative scenario of F22/F25 (seed 100, rho 1.0, fraction 0.20, injection draw 7): per-step
                     trust of every RSU under D, the degraded mask, and per-step dispatch counts to / not to the degraded RSUs;
                   * the evaluation aggregate at the shared point (seeds 1-5 x draws 100-149, the S9 grid): the same series per run.
  example.json   the first qualifying decision of D54 section 9 (the S8 merged example), as structured fields plus the rendered text.

EVALUATION-SIDE: reads the sealed class of each RSU to colour the series. Allowlisted in tests/test_sealing.py. Nothing here is imported
by a model or by the explanation code.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DEMO = dict(seed=100, rho=1.0, fraction=0.20, draw=7)
SWEEP_CONFIG = "configs/sweep.yaml"
OUT = REPO_ROOT / "results" / "s10"


def _series(variant, seed, rho, fraction, draw, sw):
    """One closed-loop run -> per-step arrays (needs the sealed degraded mask and the trust history)."""
    import s9_sweep as S
    from s6_run import load_weights
    from trustgraph.config import with_overrides
    from trustgraph.offload import build_offload_settings
    from trustgraph.simulator import run_closed_loop

    st = S._seed_state(seed)
    cfg = with_overrides(st["base"], seed=seed, rho=rho, fraction=fraction, draw=draw)
    settings = build_offload_settings(
        cfg, variant=variant, weights=load_weights(REPO_ROOT / sw["weights_file"], variant), record_trust=True
    )
    model = st["models"]["D"] if variant == "D" else None
    sc = run_closed_loop(cfg, st["world"], st["trace"], settings, model=model)
    gt, tasks = sc.ground_truth, sc.observed.tasks
    n = sc.observed.num_steps
    sent = tasks.dispatched
    to_deg = np.zeros(len(tasks), bool)
    to_deg[sent] = gt.degraded[tasks.rsu[sent]]
    return {
        "to_deg": np.bincount(tasks.step[to_deg], minlength=n),
        "sent": np.bincount(tasks.step[sent], minlength=n),
        "trust": sc.extras["trust_history"],          # (steps, rsus)
        "degraded": gt.degraded, "segment": gt.backhaul_segment_id,
        "onset": int(gt.onset_step), "ramp": int(gt.ramp_steps),
    }


def _work(task):
    variant, seed, draw = task
    r = _series(variant, seed, 1.0, 0.20, draw, _STATE["sw"])
    out = {"to_deg": r["to_deg"], "sent": r["sent"]}
    if variant == "D":
        t, d = r["trust"], r["degraded"]
        out["trust_deg"] = t[:, d].mean(axis=1)
        out["trust_ok"] = t[:, ~d].mean(axis=1)
    return task, out


def _init(sw):
    import s9_sweep as S

    S._init(sw)
    _STATE["sw"] = sw


_STATE: dict = {}


def build_timeline(workers: int = 12) -> Path:
    import s9_sweep as S

    sw = S.load_sweep(SWEEP_CONFIG)
    S._init(sw)
    _STATE["sw"] = sw
    demo = {}
    for v in ("A", "D"):
        r = _series(v, DEMO["seed"], DEMO["rho"], DEMO["fraction"], DEMO["draw"], sw)
        demo[v] = r
    tasks = [(v, s, d) for v in ("A", "D") for s in sw["seeds"] for d in sw["draws"]]
    t0, res = time.time(), {}
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(sw,)) as pool:
        for i, (task, out) in enumerate(pool.imap_unordered(_work, tasks, chunksize=2), 1):
            res[task] = out
            if i % 100 == 0 or i == len(tasks):
                print(f"  [timeline] {i}/{len(tasks)} runs  {time.time() - t0:.0f}s", flush=True)
    seeds, draws = sw["seeds"], sw["draws"]

    def stack(v, key):
        return np.array([[res[(v, s, d)][key] for d in draws] for s in seeds])  # (seeds, draws, steps)

    arrays = {
        "demo_degraded": demo["D"]["degraded"], "demo_segment": demo["D"]["segment"],
        "demo_trust_D": demo["D"]["trust"],
        "demo_to_deg_A": demo["A"]["to_deg"], "demo_sent_A": demo["A"]["sent"],
        "demo_to_deg_D": demo["D"]["to_deg"], "demo_sent_D": demo["D"]["sent"],
        "onset": demo["D"]["onset"], "ramp": demo["D"]["ramp"],
        "agg_seeds": np.array(seeds), "agg_draws": np.array(draws),
        "agg_to_deg_A": stack("A", "to_deg"), "agg_sent_A": stack("A", "sent"),
        "agg_to_deg_D": stack("D", "to_deg"), "agg_sent_D": stack("D", "sent"),
        "agg_trust_deg_D": stack("D", "trust_deg"), "agg_trust_ok_D": stack("D", "trust_ok"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "timeline.npz"
    np.savez_compressed(path, **arrays)
    return path


def build_example() -> Path:
    """The S8 merged example, as data (same selection rule as scripts/s8_example.py, D54 section 9)."""
    import s8_common as C
    from trustgraph.explain import explain_decision
    from trustgraph.explain_layer2 import explain_node, render_merged
    from trustgraph.sealed.ground_truth import DEGRADED

    cfg, world, trace, scenario, model = C.run_variant("D", DEMO["seed"], draw=DEMO["draw"])
    n = world.topology.num_rsus
    gt = scenario.ground_truth
    deg, seg = gt.behavior_class == DEGRADED, gt.backhaul_segment_id
    qualifying = []
    for rec in scenario.extras["decision_log"]:
        if rec["step"] < 700 or rec["selected_rsu"] is None:
            continue
        if sum(c["dwell_eligible"] for c in rec["candidates"]) < 2:
            continue
        e = explain_decision(rec)
        r = e.runner_up.loser
        if deg[r] and int((deg & (seg == seg[r])).sum()) >= 2 and e.runner_up.decisive == "trust":
            qualifying.append((rec, e))
    rec, e = qualifying[0]
    graph = C.step_graphs(cfg, world, trace, scenario.observed, [rec["step"]])[rec["step"]]
    layer2 = {v: explain_node(model, graph, v, n, 0, rec["decision_id"]) for v in (e.winner, e.runner_up.loser)}
    data = {
        "decision_id": rec["decision_id"], "step": rec["step"], "vehicle": rec["vehicle"], "variant": rec["variant"],
        "n_qualifying": len(qualifying), "weights": rec["weights"], "selected": rec["selected_rsu"],
        "runner_up": rec["runner_up_rsu"], "text": render_merged(e, layer2),
        "degraded_rsus": [int(i) for i in np.flatnonzero(deg)],
        "candidates": [
            {k: c[k] for k in ("rsu", "trust", "trust_term", "latency_term", "load_term", "score", "dwell_eligible")}
            for c in rec["candidates"]
        ],
        "outcome": rec["outcome"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "example.json"
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build_timeline()); print(build_example())

"""Build demo/data.js for the TrustGraph demo front end (demo/index.html).

    python demo/export_demo_data.py            (demo/Rebuild_Data.bat calls this; the result is committed, D90)

READ-ONLY with respect to the project: it reads saved outputs and writes ONE file, demo/data.js.
It does not need PyTorch, does not train or run any model, and never edits code, configs or results.

What it reads (all already on disk after S6/S10):
    configs/demo.yaml, configs/weights*.yaml, configs/gate.yaml   world layout, weights, tau
    traces/demo-seed100.npz                                      vehicle motion of the demo scenario
    scenarios/s6/demo-seed100-rho1.00-frac0.20-draw7-cl-{A,C,D}.*  the closed-loop demo runs + decision logs
    results/s10/timeline.npz                                     Variant D trust for every RSU, 5-seed aggregates
    results/s10/tables/*.csv|*.md                                multi-seed tables
    results/s8/example.txt                                       the Layer 2 merged explanation

The sealed ground truth is read here only to colour the "reveal sealed truth" view and to compute the
demo summary, exactly as scripts/s6_demo.py does (evaluation side). Nothing is written back.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

DEMO_DIR = Path(__file__).resolve().parent
REPO = DEMO_DIR.parent
sys.path.insert(0, str(REPO / "src"))

# numpy-only project modules (no torch import anywhere on this path)
from trustgraph.config import load_config, with_overrides  # noqa: E402
from trustgraph.explain import explain_decision, render_text  # noqa: E402
from trustgraph.features import RSU_FEATURES  # noqa: E402
from trustgraph.roads import build_road_network  # noqa: E402
from trustgraph.topology import build_topology  # noqa: E402

CONFIG = "configs/demo.yaml"
SEED, RHO, FRACTION, DRAW = 100, 1.0, 0.20, 7
STEM = "demo-seed100-rho1.00-frac0.20-draw7-cl-{}"
REPLAY_VARIANTS = ("A", "C", "D")
WINDOW = 25
TASK_NAMES = ("light", "medium", "heavy")


def r(x, nd=3):
    return None if x is None else float(round(float(x), nd))


def arr(a, nd=3):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def read_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def why_line(record: dict) -> str:
    """The exact Layer 1 'Why:' sentence from trustgraph.explain (empty if one candidate)."""
    text = render_text(explain_decision(record))
    for line in text.split("\n"):
        if line.startswith("Why:"):
            return line[5:]
    return ""


def exact_trust(cfg, variant: str, logged: np.ndarray):
    """Optional: re-run the frozen trust head on every step's graph of the closed-loop run.

    Uses the same calls as scripts/s8_common.py (build_snapshot_builder + the S3 checkpoint). Needs PyTorch,
    so it only runs inside the project's .venv. The result is checked against every trust value the engine
    logged; on any mismatch (or any error) it returns None and the caller falls back to the logged values.
    """
    try:
        import torch  # noqa: F401

        from trustgraph.model import build_trust_head
        from trustgraph.observed import load_observed
        from trustgraph.scenario import build_snapshot_builder, build_world
        from trustgraph.trace import load_trace
        from trustgraph.train import load_checkpoint

        ckpt = REPO / "checkpoints" / f"s3-seed{SEED}-{variant}.pt"
        if not ckpt.exists():
            return None
        model = build_trust_head(cfg.model, 0, cfg.device, variant)
        load_checkpoint(ckpt, model)
        model.eval()
        world = build_world(cfg)
        trace = load_trace(REPO / "traces" / f"demo-seed{SEED}.npz")
        observed = load_observed(REPO / "scenarios" / "s6" / (STEM.format(variant) + ".observed.npz"))
        builder = build_snapshot_builder(cfg, world, trace, observed)
        n = world.topology.num_rsus
        out = np.zeros((trace.num_steps, n))
        with torch.no_grad():
            for t in range(trace.num_steps):
                data = builder.build(t)
                out[t] = model(data.x, data.edge_index, data.edge_attr)[:n].numpy()
        seen = ~np.isnan(logged)
        err = float(np.abs(out[seen] - logged[seen]).max()) if seen.any() else 0.0
        if err > 1e-3:
            print(f"    exact trust for {variant} disagrees with the log (max {err:.4f}); using logged values")
            return None
        print(f"    exact trust for {variant}: re-ran the frozen head, matches the log (max diff {err:.1e})")
        return out
    except Exception as exc:  # no torch, missing file, API drift: fall back, never fail the demo
        print(f"    exact trust for {variant} not available ({type(exc).__name__}); using logged values")
        return None


def windowed(values, steps, num_steps):
    edges = np.arange(0, num_steps + 1, WINDOW)
    out = []
    for i in range(len(edges) - 1):
        sel = (steps >= edges[i]) & (steps < edges[i + 1])
        out.append(r(values[sel].mean()) if sel.any() else None)
    return out


# data.js is opened from file:// (fetch is blocked there), so it cannot be compressed. Instead the decision log is
# stored losslessly in a compact form and expanded by a few lines at the end of data.js, before index.html runs:
# every "why" sentence becomes (template id; its numbers), and each decision becomes a positional array.
_NUM = re.compile(r"\d+(?:\.\d+)?")

DECODER = """
(function (D) {
  const T = D.why_templates, TN = D.task_names;
  const why = c => { if (!c) return ""; const i = c.indexOf(";"), nums = c.slice(i + 1).split(",");
    let k = 0; return T[+c.slice(0, i)].replace(/#/g, () => nums[k++]); };
  for (const v in D.runs) {
    const run = D.runs[v];
    run.decisions = run.decisions.map(a => ({ id: a[0], t: a[1], veh: a[2], type: TN[a[3]], dl: a[4], sel: a[5], ru: a[6],
      met: a[7] === 1, obs: a[8], adv: a[9], c: a[10], why: why(a[11]) }));
  }
  delete D.why_templates; delete D.task_names;
})(window.DEMO_DATA);
"""


def pack_decisions(runs: dict) -> tuple[dict, list[str]]:
    """Return a copy of runs with decisions packed, and the shared list of 'why' templates."""
    templates: list[str] = []
    index: dict[str, int] = {}
    packed = {}
    for v, run in runs.items():
        rows = []
        for d in run["decisions"]:
            code = ""
            if d["why"]:
                assert "#" not in d["why"]
                tpl = _NUM.sub("#", d["why"])
                if tpl not in index:
                    index[tpl] = len(templates)
                    templates.append(tpl)
                code = f"{index[tpl]};" + ",".join(_NUM.findall(d["why"]))
            rows.append([d["id"], d["t"], d["veh"], TASK_NAMES.index(d["type"]), d["dl"], d["sel"], d["ru"],
                         1 if d["met"] else 0, d["obs"], d["adv"], d["c"], code])
        packed[v] = {**run, "decisions": rows}
    return packed, templates


def main() -> int:
    print("TrustGraph demo: building demo/data.js from saved outputs (read-only) ...")
    cfg = with_overrides(load_config(REPO / CONFIG), seed=SEED, rho=RHO, fraction=FRACTION, draw=DRAW)
    road = build_road_network(cfg.road)
    topo = build_topology(cfg.topology, road, cfg.seeds.generator("topology"))
    n = topo.num_rsus
    segs = topo.backhaul_segment_id

    trace = np.load(REPO / "traces" / f"demo-seed{SEED}.npz")
    pos = trace["positions"]  # (T, V, 2)
    T, V = pos.shape[:2]

    s6 = REPO / "scenarios" / "s6"
    sealed = np.load(s6 / (STEM.format("A") + ".SEALED.npz"))
    from trustgraph.sealed.ground_truth import DEGRADED  # noqa: E402  (evaluation side, as in s6_demo.py)

    degraded = (sealed["behavior_class"] == DEGRADED)
    onset, ramp = int(sealed["onset_step"]), int(sealed["ramp_steps"])
    assert (sealed["backhaul_segment_id"] == segs).all(), "topology does not match the scenario"

    timeline_path = REPO / "results" / "s10" / "timeline.npz"
    timeline = np.load(timeline_path) if timeline_path.exists() else None

    runs = {}
    for v in REPLAY_VARIANTS:
        print(f"  variant {v}: reading decision log ...")
        obs = np.load(s6 / (STEM.format(v) + ".observed.npz"))
        feats = obs["rsu_features"]  # (T, n, 6)
        log_path = s6 / (STEM.format(v) + ".decisions.jsonl")
        trust_seen = np.full((T, n), np.nan)
        decisions = []
        weights = None
        with log_path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                weights = weights or rec["weights"]
                t = rec["step"]
                cands = []
                for c in rec["candidates"]:
                    if v != "A":
                        trust_seen[t, c["rsu"]] = c["trust"]
                    cands.append([c["rsu"], r(c["trust"], 4), r(c["latency_norm"], 4), r(c["load"], 3),
                                  r(c["latency_ms"], 1), 1 if c["dwell_eligible"] else 0])
                out = rec.get("outcome") or {}
                decisions.append({
                    "id": rec["decision_id"], "t": t, "veh": rec["vehicle"],
                    "type": TASK_NAMES[rec["task_type"]], "dl": r(rec["deadline_s"], 3),
                    "sel": rec["selected_rsu"], "ru": rec.get("runner_up_rsu"),
                    "met": bool(out.get("met_deadline", False)),
                    "obs": r(out.get("observed_s", 0), 3), "adv": r(out.get("advertised_s", 0), 3),
                    "c": cands, "why": why_line(rec) if rec["selected_rsu"] is not None else "",
                })
        # trust for every RSU at every step: D from the saved S10 timeline when present,
        # otherwise the value the engine logged for that RSU, carried forward between decisions
        if v == "A":
            trust = np.zeros((T, n))
            trust_note = "Variant A does not use trust (alpha = 0)."
        elif v == "D" and timeline is not None and "demo_trust_D" in timeline.files:
            trust = timeline["demo_trust_D"]
            trust_note = "Exact per-step trust of every RSU (results/s10/timeline.npz)."
        else:
            trust = exact_trust(cfg, v, trust_seen) if v in ("C", "D") else None
            if trust is not None:
                trust_note = "Exact per-step trust of every RSU (frozen head re-run on the logged graphs)."
            else:
                trust = trust_seen.copy()
                last = np.ones(n)
                for t in range(T):
                    m = np.isnan(trust[t])
                    trust[t, m] = last[m]
                    last = trust[t]
                trust_note = ("Trust as logged at each decision for that RSU, held between decisions "
                              "(approximate between decisions; PyTorch not available to re-run the head).")

        sel = np.array([d["sel"] if d["sel"] is not None else -1 for d in decisions])
        steps = np.array([d["t"] for d in decisions])
        met = np.array([d["met"] for d in decisions], dtype=float)
        disp = sel >= 0
        to_deg = np.zeros(len(sel))
        to_deg[disp] = degraded[sel[disp]]
        post = steps >= onset

        phases = []
        for lo, hi, name in ((0, onset, "pre-onset"), (onset, onset + ramp, "ramp"), (onset + ramp, T, "plateau")):
            m = disp & (steps >= lo) & (steps < hi)
            phases.append({"name": name, "lo": lo, "hi": hi,
                           "share": r(to_deg[m].mean()), "success": r(met[(steps >= lo) & (steps < hi)].mean())})

        runs[v] = {
            "weights": weights,
            "trust": arr(trust, 3),
            "trust_note": trust_note,
            "feat": {  # the per-step behavioural evidence each RSU showed (observable)
                name: arr(feats[:, :, RSU_FEATURES.index(name)], 3)
                for name in ("load", "success_ewma", "latency_dev", "queue_depth")
            },
            "decisions": decisions,
            "summary": {
                "tasks": int(len(decisions)),
                "success_all": r(met.mean(), 4),
                "success_post": r(met[post].mean(), 4),
                "post_tasks": int((post & disp).sum()),
                "post_to_deg": int((post & disp & (to_deg > 0)).sum()),
                "post_share_deg": r(to_deg[post & disp].mean(), 4),
            },
            "phases": phases,
            "win_share_deg": windowed(to_deg[disp], steps[disp], T),
            "win_success": windowed(met[disp], steps[disp], T),
        }

    # 5-seed aggregates of the S10 timeline (evaluation seeds 1-5, draws 100-149)
    agg = None
    if timeline is not None and "agg_trust_deg_D" in timeline.files:
        def share(to_deg, sent):
            s = sent.sum(axis=(0, 1)).astype(float)
            return np.where(s > 0, to_deg.sum(axis=(0, 1)) / np.maximum(s, 1), np.nan)

        def win(x):
            x = np.asarray(x)
            return [r(np.nanmean(x[i:i + WINDOW])) for i in range(0, T, WINDOW)]

        agg = {
            "seeds": timeline["agg_seeds"].tolist(), "draws": [int(timeline["agg_draws"].min()), int(timeline["agg_draws"].max())],
            "trust_deg_D": win(np.nanmean(timeline["agg_trust_deg_D"], axis=(0, 1))),
            "trust_ok_D": win(np.nanmean(timeline["agg_trust_ok_D"], axis=(0, 1))),
            "share_A": win(share(timeline["agg_to_deg_A"], timeline["agg_sent_A"])),
            "share_D": win(share(timeline["agg_to_deg_D"], timeline["agg_sent_D"])),
        }

    curated = []
    picks = [  # same selection rules as scripts/s7_examples.py: first plateau decision of each kind
        ("D", "Runner-up lost on trust", lambda e: e.runner_up.kind == "decisive" and e.runner_up.decisive == "trust"),
        ("D", "Runner-up lost on latency although the winner had lower trust",
         lambda e: e.runner_up.kind == "decisive" and e.runner_up.decisive == "latency" and "trust" in e.runner_up.outweighed),
        ("C", "Winner chosen despite lower trust; load decisive",
         lambda e: e.runner_up.kind == "decisive" and e.runner_up.decisive == "load" and "trust" in e.runner_up.outweighed),
        ("A", "Variant A: trust not used", lambda e: e.trust_mode == "unused"),
    ]
    for v, title, pred in picks:
        with (s6 / (STEM.format(v) + ".decisions.jsonl")).open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["step"] < 700 or rec["selected_rsu"] is None:
                    continue
                e = explain_decision(rec)
                if e.runner_up is not None and pred(e):
                    curated.append({"variant": v, "title": title, "id": rec["decision_id"], "text": render_text(e)})
                    break

    tables_dir = REPO / "results" / "s10" / "tables"
    weights_all = load_yaml(REPO / "configs" / "weights.yaml").get("weights", {})
    wh = load_yaml(REPO / "configs" / "weights_h.yaml").get("weights", {})
    wd2 = load_yaml(REPO / "configs" / "weights_d2.yaml").get("weights", {})
    gate = load_yaml(REPO / "configs" / "gate.yaml")

    data = {
        "generated_from": {"config": CONFIG, "seed": SEED, "rho": RHO, "fraction": FRACTION, "draw": DRAW},
        "world": {
            "extent": float(cfg.road["extent_m"]),
            "intersections": arr(road.intersections, 1),
            "roads": sorted({tuple(sorted((i, j))) for i, nb in enumerate(road.neighbours) for j in nb}),
            "rsu_pos": arr(topo.positions, 1),
            "rsu_seg": segs.tolist(),
            "rsu_edges": topo.rsu_edges.tolist(),
            "coverage_m": float(topo.coverage_radius_m),
            "rsu_link_m": float(topo.rsu_link_radius_m),
            "num_segments": int(topo.num_segments),
        },
        "steps": int(T), "num_vehicles": int(V),
        "veh_pos": np.round(pos).astype(int).reshape(T, -1).tolist(),
        "sealed": {"degraded": np.flatnonzero(degraded).tolist(), "onset": onset, "ramp": ramp,
                   "severity": r(sealed["severity"]), "drop_prob": r(sealed["drop_prob"])},
        "runs": runs,
        "agg": agg,
        "curated": curated,
        "layer2_example": read_text(REPO / "results" / "s8" / "example.txt"),
        "weights": {"file": weights_all, "H": wh.get("H"), "D2": (wd2.get("shared") or {}).get("D2"),
                    "gate_tau": gate.get("tau")},
        "tables": {
            "shared_point": read_csv(tables_dir / "ablation_shared_point.csv"),
            "all_points": read_csv(tables_dir / "ablation_all_points.csv"),
            "graph_tests": read_csv(tables_dir / "paper_graph_tests.csv"),
            "d2_dispatch": read_csv(tables_dir / "paper_d2_dispatch.csv"),
            "latency": read_csv(tables_dir / "paper_latency.csv"),
        },
    }

    out = DEMO_DIR / "data.js"
    data["runs"], data["why_templates"] = pack_decisions(runs)
    data["task_names"] = list(TASK_NAMES)
    out.write_text("window.DEMO_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n" + DECODER,
                   encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    for v in REPLAY_VARIANTS:
        print(f"  {v}: {runs[v]['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

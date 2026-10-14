"""S6 exit condition: the demo plot.

    python scripts/s6_demo.py --config configs/demo.yaml --weights-file configs/weights.yaml

One scenario (tuning seed 100, rho 1.0, fraction 0.20, injection draw 7 - chosen before running, not one of
the tuning draws 0-2), closed loop, run with Variant A, C and D on identical traffic. A backhaul segment
degrades from step 400 (ramp to 700). The figure shows

  1. mean trust of the degraded RSUs and of the healthy RSUs under C and D (the models' own scores);
  2. the share of dispatched tasks that go to the degraded RSUs, per 25-step window, A vs C vs D;
  3. deadline success per 25-step window, A vs C vs D.

Prints the numbers behind it. This is a demonstration on ONE scenario, not an evaluation: no
evaluation seed is used and no sweep is run (S9). EVALUATION-SIDE; allowlisted in tests/test_sealing.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from s6_run import load_weights, summarise  # noqa: E402
from trustgraph.config import load_config, with_overrides  # noqa: E402
from trustgraph.offload import build_offload_settings  # noqa: E402
from trustgraph.scenario import build_world  # noqa: E402
from trustgraph.sealed.ground_truth import DEGRADED  # noqa: E402
from trustgraph.simulator import run_closed_loop  # noqa: E402
from trustgraph.trace import default_trace_path, load_trace  # noqa: E402

SEED, RHO, FRACTION, DRAW = 100, 1.0, 0.20, 7
WINDOW = 25
COLORS = {"A": "#6b7280", "C": "#2a6fb0", "D": "#c2571a"}
LABEL = {"A": "A (no trust)", "C": "C (node-local MLP)", "D": "D (GNN)"}


def windowed(values: np.ndarray, steps: np.ndarray, num_steps: int, fn) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(0, num_steps + 1, WINDOW)
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        sel = (steps >= edges[i]) & (steps < edges[i + 1])
        if sel.any():
            out[i] = fn(values[sel])
    return edges[:-1] + WINDOW / 2, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights-file", required=True)
    ap.add_argument("--out", default="figures/s6_demo.png")
    args = ap.parse_args()

    cfg = with_overrides(load_config(args.config), seed=SEED, rho=RHO, fraction=FRACTION, draw=DRAW)
    world = build_world(cfg)
    trace = load_trace(default_trace_path(args.config, SEED))
    lines: list[str] = []

    def p(s=""):
        print(s)
        lines.append(s)

    runs = {}
    for variant in ("A", "C", "D"):
        weights = load_weights(args.weights_file, variant)
        settings = build_offload_settings(cfg, variant=variant, weights=weights, record_trust=variant != "A")
        runs[variant] = run_closed_loop(cfg, world, trace, settings)
        s = summarise(runs[variant])
        p(f"{variant}: weights (alpha, beta, gamma) = ({settings.effective_alpha:g}, {settings.beta:g}, {settings.gamma:g})")
        p(f"   tasks {s['tasks']}  success all {s['success_all']:.4f}  success post-onset {s['success_post_onset']:.4f}")
        p(f"   post-onset tasks {s['post_onset_tasks']}  to degraded RSUs {s['post_onset_to_degraded']} "
          f"({s['post_onset_share_to_degraded']:.4f})")
        p(f"   engine counters: {runs[variant].extras['counters']}")

    gt = runs["A"].ground_truth
    deg = np.flatnonzero(gt.behavior_class == DEGRADED)
    healthy = np.flatnonzero(gt.behavior_class != DEGRADED)
    p(f"degraded RSUs {deg.tolist()} on segments {gt.backhaul_segment_id[deg].tolist()} "
      f"(onset {gt.onset_step}, ramp {gt.ramp_steps}); realised pair concentration "
      f"{gt.degraded_pair_concentration():.3f}")

    # post-onset share to degraded, in the three phases, per variant
    for lo, hi, name in ((0, gt.onset_step, "pre-onset"), (gt.onset_step, gt.onset_step + gt.ramp_steps, "ramp"),
                         (gt.onset_step + gt.ramp_steps, trace.num_steps, "plateau")):
        row = []
        for v in ("A", "C", "D"):
            t = runs[v].observed.tasks
            sel = t.dispatched & (t.step >= lo) & (t.step < hi)
            share = np.isin(t.rsu[sel], deg).mean() if sel.any() else float("nan")
            succ = t.success_rate((t.step >= lo) & (t.step < hi))
            row.append(f"{v}: share {share:.3f} success {succ:.3f}")
        p(f"  {name:9s} [{lo},{hi}): " + " | ".join(row))
    for v in ("C", "D"):
        th = runs[v].extras["trust_history"]
        p(f"{v} mean trust of degraded RSUs: step 399 {th[399, deg].mean():.3f}, 550 {th[550, deg].mean():.3f}, "
          f"700 {th[700, deg].mean():.3f}, 1199 {th[1199, deg].mean():.3f};  healthy RSUs: "
          f"399 {th[399, healthy].mean():.3f}, 1199 {th[1199, healthy].mean():.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED = "#1f2328", "#6b7280"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "font.size": 10})
    fig, ax = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)
    steps = np.arange(trace.num_steps)
    for v in ("C", "D"):
        th = runs[v].extras["trust_history"]
        for r in deg:
            ax[0].plot(steps, th[:, r], color=COLORS[v], lw=0.6, alpha=0.35)
        ax[0].plot(steps, th[:, deg].mean(axis=1), color=COLORS[v], lw=2.2, label=f"{LABEL[v]}: degraded RSUs (mean)")
        ax[0].plot(steps, th[:, healthy].mean(axis=1), color=COLORS[v], lw=1.4, ls="--", label=f"{LABEL[v]}: healthy RSUs (mean)")
    ax[0].set_ylabel("trust score"); ax[0].set_ylim(0, 1.02)
    ax[0].set_title(f"Backhaul segment degrades mid-run (seed {SEED}, rho {RHO:g}, {FRACTION:.0%} degraded, closed loop)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")

    for v in ("A", "C", "D"):
        t = runs[v].observed.tasks
        to_deg = np.zeros(len(t))
        to_deg[t.dispatched] = np.isin(t.rsu[t.dispatched], deg)
        x, y = windowed(to_deg[t.dispatched], t.step[t.dispatched], trace.num_steps, np.mean)
        ax[1].plot(x, y, color=COLORS[v], lw=2, label=LABEL[v])
    ax[1].set_ylabel(f"share of tasks sent to degraded RSUs\n(per {WINDOW} steps)"); ax[1].legend(frameon=False, fontsize=9)

    for v in ("A", "C", "D"):
        t = runs[v].observed.tasks
        x, y = windowed(t.met_deadline[t.dispatched].astype(float), t.step[t.dispatched], trace.num_steps, np.mean)
        ax[2].plot(x, y, color=COLORS[v], lw=2, label=LABEL[v])
    ax[2].set_ylabel(f"deadline success\n(per {WINDOW} steps)"); ax[2].set_xlabel("step (1 step = 1 s)")
    for a in ax:
        a.axvspan(gt.onset_step, gt.onset_step + gt.ramp_steps, color="#f3e3d6", alpha=0.6, lw=0)
        a.axvline(gt.onset_step, color=MUTED, lw=1, ls=":")
    ax[0].text(gt.onset_step + 5, 0.04, "degradation ramps in", fontsize=8, color=MUTED)
    fig.tight_layout()
    out = REPO_ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    (REPO_ROOT / "results" / "s6").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results" / "s6" / "demo.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

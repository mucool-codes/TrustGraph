"""S10: every figure and table of the paper, regenerated from one command.

    python scripts/make_figures.py                 # figures/s10/*.svg|pdf  and  results/s10/tables/*.md|csv|tex
    python scripts/make_figures.py --recompute     # also rebuild the cached closed-loop inputs (Figures 2 and 6)

Inputs (all produced by earlier sessions; nothing is typed in by hand):
  results/s9/   results_table.csv, paired_comparisons.csv, summary-verdict.json, overhead.json   (S9, F26/F27)
  results/s4/   summary-5seeds.json                                                               (S4, F16/F17)
  results/s10/  timeline.npz, example.json   closed-loop inputs, built by scripts/s10_data.py on first use and cached
Every number on a figure or in a table is read from those files. Colours, fonts and sizes are defined once below.
This script never opens the sealed ground truth (s10_data.py, allowlisted, does).

Figures: 1 architecture, 2 trust timeline with dispatch shifting away (D vs A, closed loop), 3 success vs degraded fraction,
4 ablation A-E (C vs D highlighted), 5 S4 gap vs correlation, 6 a worked explanation as an operator sees it.
Tables: ablation (mean +/- sd), per-decision latency by component, related work (rows left for S11).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

S9 = REPO_ROOT / "results" / "s9"
S4 = REPO_ROOT / "results" / "s4"
S10 = REPO_ROOT / "results" / "s10"
FIG = REPO_ROOT / "figures" / "s10"
TAB = S10 / "tables"

# ------------------------------------------------------------------ one style for everything
INK, MUTED, FAINT = "#1f2328", "#5b6470", "#d5d9de"
COL = {"A": "#6b7280", "B": "#9ca3af", "B_oracle": "#2f855a", "C": "#2a6fb0", "D": "#c2571a", "E": "#c2571a"}
NAME = {"A": "A  no trust", "B": "B  certificate", "B_oracle": "B-oracle  (upper bound)", "C": "C  node-local MLP",
        "D": "D  GNN", "E": "E  GNN + explanation"}
ONSET, RAMP_END, WIN = 400, 700, 25
SHARED = (1.0, 0.20)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "lines.linewidth": 2.4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "path", "svg.hashsalt": "trustgraph-s10", "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.svg", metadata={"Date": None, "Creator": "make_figures.py"}, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": "make_figures.py"},
                bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=110, bbox_inches="tight")  # preview only
    plt.close(fig)
    print(f"  wrote figures/s10/{name}.svg .pdf (.png preview)")


# ------------------------------------------------------------------ loading
def load_table() -> dict:
    """{(rho, fraction): {variant: {metric: (mean, sd)}}} from the S9 results table."""
    p = S9 / "results_table.csv"
    if not p.exists():
        raise SystemExit(f"{p} missing: run `python scripts/s9_sweep.py all --sweep configs/sweep.yaml` first (S9)")
    out: dict = {}
    for r in csv.DictReader(p.open(encoding="utf-8")):
        k = (round(float(r["rho"]), 2), round(float(r["fraction"]), 2))
        out.setdefault(k, {}).setdefault(r["variant"], {})[r["metric"]] = (float(r["mean_over_seeds"]), float(r["sd_over_seeds"]))
    return out


def load_paired() -> dict:
    out = {}
    for r in csv.DictReader((S9 / "paired_comparisons.csv").open(encoding="utf-8")):
        out[(round(float(r["rho"]), 2), round(float(r["fraction"]), 2), r["x"], r["ref"])] = {k: float(v) for k, v in r.items() if k.startswith("h")}
    return out


def ensure_inputs(recompute: bool) -> None:
    need = [S10 / "timeline.npz", S10 / "example.json"]
    if recompute or not all(p.exists() for p in need):
        import s10_data

        print("building the closed-loop inputs (a few minutes) ...", flush=True)
        s10_data.build_timeline()
        s10_data.build_example()


def spans(ax, y0=0.0, y1=1.0, label=True):
    ax.axvspan(ONSET, RAMP_END, color="#f3d9c4", alpha=0.55, lw=0, zorder=0)
    ax.axvline(ONSET, color=MUTED, lw=1.2, ls="--", zorder=1)
    if label:
        ax.text(ONSET - 8, 0.97, "onset", transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=11, color=MUTED)
        ax.text(RAMP_END + 8, 0.97, "ramp ends", transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=11, color=MUTED)
        ax.axvline(RAMP_END, color=MUTED, lw=1.0, ls=":", zorder=1)


# ------------------------------------------------------------------ Figure 1: architecture
def box(ax, x, y, w, h, title, lines, face="#ffffff", edge=INK, dashed=False, tcolor=INK, size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=face, ec=edge,
                                lw=2.0, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="top", fontsize=13.5, fontweight="bold", color=tcolor)
    ax.text(x + w / 2, y + h - 0.9, "\n".join(lines), ha="center", va="top", fontsize=size, linespacing=1.5)


def arrow(ax, p, q, text=None, rad=0.0, color=INK, ls="-", tpos=0.5, toff=(0, 0.18), ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, lw=2.2, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        x = p[0] + (q[0] - p[0]) * tpos + toff[0]
        y = p[1] + (q[1] - p[1]) * tpos + toff[1]
        ax.text(x, y, text, ha=ha, va="bottom", fontsize=11, color=color)


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(16, 7.4))
    ax.set_xlim(0, 16.6); ax.set_ylim(-0.7, 7.6); ax.axis("off")
    w, h, y = 3.4, 3.2, 3.6
    xs = [0.2, 4.4, 8.6, 12.8]
    box(ax, xs[0], y, w, h, "1  Dynamic graph", ["G_t rebuilt every timestep", "nodes: RSUs and vehicles", "edges: vehicle-RSU coverage", "and RSU-RSU coordination",
                                                  "edge flag same_segment", "(shared backhaul segment)"], face="#eef2f6")
    box(ax, xs[1], y, w, h, "2  Trust GNN", ["2-layer GINEConv: message", "passing over G_t", "output: one scalar trust_v", "in [0,1] per RSU",
                                              "self-supervised target:", "meets its next deadline?"], face="#fbeee4", edge=COL["D"])
    box(ax, xs[2], y, w, h, "3  Offloading rule", ["fixed, not learned (L1)", "score = a*trust - b*latency", "- g*load, argmax over in-range,", "dwell-feasible RSUs",
                                                   "trust is a weighted term,", "not a gate;  a = 0 is Variant A"], face="#eef2f6")
    box(ax, xs[3], y, w, h, "4  Task and outcome", ["task runs at the chosen RSU", "vehicle compares observed with", "advertised load and latency;", "the discrepancy updates",
                                                     "success_ewma, latency_dev", "of that RSU"], face="#eef2f6")
    for i in range(3):
        arrow(ax, (xs[i] + w + 0.04, y + h / 2), (xs[i + 1] - 0.04, y + h / 2))
    ax.add_patch(FancyArrowPatch((xs[3] + w / 2, y + h + 0.04), (xs[0] + w / 2, y + h + 0.04), arrowstyle="-|>", mutation_scale=22,
                                 lw=2.0, color=MUTED, connectionstyle="arc3,rad=0.16"))
    ax.text(8.1, 7.3, "observed outcomes become the next timestep's node features", ha="center", fontsize=11.5, color=MUTED)
    box(ax, 8.6, 0.0, 7.6, 2.5, "5  Explanation (read-only)",
        ["Layer 1: exact decomposition of the linear score;", "contrastive: why the runner-up lost",
         "Layer 2: GNNExplainer on the trust head only: why", "trust_v has its value (asynchronous, ~300 ms)"],
        face="#fbeee4", edge=COL["D"], size=10.5)
    arrow(ax, (xs[2] + w / 2 + 0.8, y - 0.04), (xs[2] + w / 2 + 0.8, 2.56), "decision log", toff=(0.12, -0.15), ha="left")
    arrow(ax, (xs[1] + w / 2, y - 0.04), (9.6, 2.56))
    ax.text(7.0, 3.1, "trust head", ha="center", fontsize=11, color=INK)
    box(ax, 0.2, 0.0, 7.6, 2.5, "Sealed ground truth (evaluation only)",
        ["injected behaviour class and segment faults", "used only to score trust and dispatch", "never a feature, a label, or importable", "by training code (L4); no arrow enters the pipeline"],
        face="#f5f5f5", edge=MUTED, dashed=True, tcolor=MUTED, size=10.5)
    ax.text(8.1, -0.55, "Ablation: A no trust  |  B certificate only  |  C node-local MLP trust  |  D GNN trust  |  E = D + explanation.   Same rule in all five (L1, L2).",
            ha="center", fontsize=11.5)
    save(fig, "fig1_architecture")


# ------------------------------------------------------------------ Figure 2: timeline
def smooth(x, k=15):
    """Centred moving average over k steps (edges padded), for display only."""
    x = np.asarray(x, float)
    pad = np.pad(x, [(k // 2, k // 2)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    c = np.cumsum(np.insert(pad, 0, 0, axis=0), axis=0)
    return (c[k:] - c[:-k]) / k


def win_share(to_deg, sent, nsteps):
    """Per-window share of dispatches that went to degraded RSUs; arrays (..., steps) -> (..., windows)."""
    k = nsteps // WIN
    td = to_deg[..., : k * WIN].reshape(*to_deg.shape[:-1], k, WIN).sum(-1)
    sn = sent[..., : k * WIN].reshape(*sent.shape[:-1], k, WIN).sum(-1)
    return np.where(sn > 0, td / np.maximum(sn, 1), np.nan)


def fig_timeline() -> None:
    z = np.load(S10 / "timeline.npz")
    nsteps = z["demo_trust_D"].shape[0]
    centers = np.arange(WIN // 2, nsteps - WIN + 1 + WIN // 2, WIN)[: nsteps // WIN]
    fig, ax = plt.subplots(2, 2, figsize=(15, 8.6), sharex=True, gridspec_kw={"hspace": 0.1, "wspace": 0.13})

    # (a) one scenario
    t = z["demo_trust_D"]; deg = z["demo_degraded"]
    a = ax[0, 0]
    spans(a, label=False)
    lo, hi = np.percentile(t[:, ~deg], [10, 90], axis=1)
    a.fill_between(np.arange(nsteps), smooth(lo), smooth(hi), color=COL["A"], alpha=0.22, lw=0, label="healthy RSUs, 10-90th percentile")
    a.plot(smooth(np.median(t[:, ~deg], axis=1)), color=COL["A"], lw=2.4, label="healthy RSUs, median")
    for j, v in enumerate(np.flatnonzero(deg)):
        a.plot(smooth(t[:, v]), color=COL["D"], lw=1.7, alpha=0.9, label="degraded RSUs, one line each" if j == 0 else None)
    a.set_ylabel("trust from the GNN (Variant D)\n15 s moving average"); a.set_ylim(0, 1.04)
    a.set_title("(a) One scenario (seed 100, draw 7)")
    a.legend(loc="lower left", frameon=False, fontsize=11)
    b = ax[1, 0]
    spans(b, label=True)
    sa = win_share(z["demo_to_deg_A"], z["demo_sent_A"], nsteps); sd = win_share(z["demo_to_deg_D"], z["demo_sent_D"], nsteps)
    b.step(centers, sa, where="mid", color=COL["A"], label=NAME["A"]); b.step(centers, sd, where="mid", color=COL["D"], label=NAME["D"])
    b.set_ylabel("share of tasks sent to degraded RSUs\n(25 s windows; label = mean after the ramp)"); b.set_xlabel("time (s)"); b.set_ylim(0, 0.56)
    plateau = centers >= RAMP_END
    for s_, c_, nm in ((sa, COL["A"], "A"), (sd, COL["D"], "D")):
        b.text(1212, np.nanmean(s_[plateau]), f"{nm}  {np.nanmean(s_[plateau]):.2f}", color=c_, ha="left", va="center",
               fontsize=12, fontweight="bold", clip_on=False)
    b.legend(loc="upper right", frameon=False, fontsize=11)

    # (b) evaluation aggregate: 5 seeds x 50 draws
    def band(arr):  # (seeds, draws, steps|windows) -> mean over draws per seed, then mean/sd over seeds
        per_seed = np.nanmean(arr, axis=1)
        return np.nanmean(per_seed, axis=0), np.nanstd(per_seed, axis=0, ddof=1)

    a = ax[0, 1]
    spans(a, label=False)
    for key, col, lab in (("agg_trust_ok_D", COL["A"], "healthy RSUs"), ("agg_trust_deg_D", COL["D"], "degraded RSUs")):
        m, s = band(z[key]); x = np.arange(m.size)
        a.fill_between(x, m - s, m + s, color=col, alpha=0.2, lw=0); a.plot(x, m, color=col, label=lab)
    a.set_ylim(0, 1.04); a.legend(loc="lower left", frameon=False, fontsize=11)
    a.set_title("(b) 5 seeds x 50 draws, mean +/- sd over seeds"); a.tick_params(labelleft=False)
    b = ax[1, 1]
    spans(b, label=True)
    for v in ("A", "D"):
        sh = win_share(z[f"agg_to_deg_{v}"], z[f"agg_sent_{v}"], nsteps)
        m, s = band(sh)
        b.fill_between(centers, m - s, m + s, color=COL[v], alpha=0.2, lw=0, step="mid"); b.step(centers, m, where="mid", color=COL[v], label=NAME[v])
        b.text(1212, np.nanmean(m[plateau]), f"{v}  {np.nanmean(m[plateau]):.2f}", color=COL[v], ha="left", va="center",
               fontsize=12, fontweight="bold", clip_on=False)
    b.set_xlabel("time (s)"); b.set_ylim(0, 0.56); b.tick_params(labelleft=False)
    fig.suptitle("Trust falls on the degrading segment, and dispatch moves away from it (closed loop, D vs A)", x=0.125, ha="left",
                 fontsize=16, fontweight="bold", y=0.96)
    save(fig, "fig2_trust_timeline")


# ------------------------------------------------------------------ Figure 3: success vs fraction
def fig_success_vs_fraction(T, P) -> None:
    fr = [0.05, 0.10, 0.20, 0.30]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"wspace": 0.22})
    a = ax[0]
    order = [("A", "A = B  (identical decisions)", "o"), ("B_oracle", NAME["B_oracle"], "s"), ("C", NAME["C"], "^"), ("D", "D = E  (same dispatch)", "D")]
    off = {"A": -0.9, "B_oracle": -0.3, "C": 0.3, "D": 0.9}
    for v, lab, mk in order:
        m = [T[(1.0, f)][v]["success_post_onset"][0] for f in fr]; s = [T[(1.0, f)][v]["success_post_onset"][1] for f in fr]
        a.errorbar(np.array(fr) * 100 + off[v], m, yerr=s, color=COL[v], marker=mk, ms=8, capsize=4, lw=2.2,
                   ls="--" if v == "B_oracle" else "-", label=lab)
    a.set_xticks([5, 10, 20, 30]); a.set_xlabel("degraded fraction of RSUs (%)  at rho = 1.0"); a.set_ylabel("task success rate after onset")
    a.set_title("(a) Success, mean +/- sd over 5 seeds"); a.legend(frameon=False, fontsize=11, loc="lower left")
    a.text(30.5, 0.9755, "at 5% only one RSU is degraded:\nno correlated failure", fontsize=10.5, color=MUTED, ha="right", va="top")
    b = ax[1]
    for v, off_ in (("C", -0.4), ("D", 0.4)):
        m = np.array([P[(1.0, f, v, "A")]["h3_mean"] for f in fr]); lo = np.array([P[(1.0, f, v, "A")]["h3_lo"] for f in fr])
        hi = np.array([P[(1.0, f, v, "A")]["h3_hi"] for f in fr])
        b.errorbar(np.array(fr) * 100 + off_, m, yerr=[m - lo, hi - m], color=COL[v], marker="o", ms=8, capsize=4, label=f"{v} minus A")
    b.axhline(0, color=MUTED, lw=1.2); b.axhline(0.01, color=MUTED, lw=1.2, ls=":")
    b.text(30.8, 0.0115, "pre-registered floor 0.01", ha="right", fontsize=10.5, color=MUTED)
    b.set_xticks([5, 10, 20, 30]); b.set_xlabel("degraded fraction of RSUs (%)  at rho = 1.0"); b.set_ylabel("paired gain in success rate")
    b.set_title("(b) Gain over A, 95% interval over seeds"); b.legend(frameon=False, loc="upper left")
    save(fig, "fig3_success_vs_fraction")


# ------------------------------------------------------------------ Figure 4: ablation
def fig_ablation(T, P) -> None:
    S = T[SHARED]
    variants = ["A", "B", "C", "D", "E"]
    src = {"E": "D"}
    panels = [("success_post_onset", "task success rate after onset\n(axis starts at 0.80)", (0.80, 0.96), "{:.3f}", "h3", "{:+.3f}"),
              ("n_av", "tasks sent to degraded RSUs,\nonset to avoidance", (0, 760), "{:.0f}", "h1", "{:+.3f}"),
              ("n_win", "tasks sent to degraded RSUs,\nfixed window [400, 1200)", (0, 1050), "{:.0f}", None, None)]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.0), gridspec_kw={"wspace": 0.42})
    for a, (m, ylab, ylim, f, pk, pf) in zip(ax, panels):
        for i, v in enumerate(variants):
            mean, sd = S[src.get(v, v)][m]
            a.bar(i, mean, 0.7, color=COL[v], hatch="//" if v == "E" else None, edgecolor="white" if v == "E" else COL[v], lw=0,
                  alpha=0.45 if v == "B" else 1.0, zorder=2)
            a.errorbar(i, mean, yerr=sd, color=INK, capsize=5, lw=1.8, zorder=3)
            a.text(i, mean + sd + (ylim[1] - ylim[0]) * 0.015, f.format(mean), ha="center", fontsize=11)
        om = S["B_oracle"][m][0]
        a.axhline(om, color=COL["B_oracle"], ls="--", lw=2.0, zorder=1)
        a.text(4.5, om, "  oracle", color=COL["B_oracle"], ha="left", va="center", fontsize=11, clip_on=False)
        a.set_xticks(range(5)); a.set_xticklabels(variants); a.set_ylim(*ylim); a.set_xlim(-0.6, 4.6); a.set_ylabel(ylab)
        a.spines["bottom"].set_visible(True)
        a.add_patch(plt.Rectangle((1.55, ylim[0]), 1.9, ylim[1] - ylim[0], fc="#fff4c2", ec="none", alpha=0.75, zorder=0))
        a.text(2.5, 1.02, "C vs D: the structural test", transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=11.5, fontweight="bold")
        if pk:
            d = P[(1.0, 0.2, "D", "C")]
            a.set_xlabel(f"D minus C: {pf.format(d[pk + '_mean'])}  [{pf.format(d[pk + '_lo'])}, {pf.format(d[pk + '_hi'])}]", fontsize=11.5)
        else:
            a.set_xlabel("(rule-free count, a guard on the middle panel)", fontsize=11, color=MUTED)
    fig.suptitle("Ablation at the shared operating point (rho 1.0, 20% degraded). Bars: mean over 5 seeds, whiskers: sd.\n"
                 "B equals A and E equals D (identical decisions, asserted). Dashed green: B-oracle, an upper bound.",
                 x=0.125, ha="left", fontsize=13.5, fontweight="bold", y=1.04)
    save(fig, "fig4_ablation")


# ------------------------------------------------------------------ Figure 5: S4 gap vs correlation
def fig_s4_gap() -> None:
    S = json.loads((S4 / "summary-5seeds.json").read_text(encoding="utf-8"))
    rhos = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True, gridspec_kw={"wspace": 0.08})
    a = ax[0]
    d = [S["sweep"][f"{r:.2f}"]["delta"] for r in rhos]
    for j in range(5):
        a.plot(rhos, [S["sweep"][f"{r:.2f}"]["per_seed_delta"][j] for r in rhos], color=FAINT, lw=1.4, marker=".", ms=6,
               label="single seed" if j == 0 else None, zorder=1)
    a.errorbar(rhos, [x[0] for x in d], yerr=[[x[0] - x[2] for x in d], [x[3] - x[0] for x in d]], color=COL["D"], marker="o", ms=8,
               capsize=5, label="mean, 95% interval over seeds", zorder=3)
    a.axhline(0, color=MUTED, lw=1.2); a.axhline(0.02, color=COL["C"], lw=1.6, ls="--")
    a.text(0.02, 0.0225, "pre-registered floor: +0.02 would mean D beats C", color=COL["C"], fontsize=11, va="bottom")
    a.set_xlabel("nominal failure correlation rho"); a.set_ylabel("AUC of D minus AUC of C   (window 450-650)")
    a.set_title("(a) Gap vs nominal rho"); a.legend(frameon=False, loc="lower left", fontsize=11)
    b = ax[1]
    cs, ms, lo, hi, ns = [], [], [], [], []
    for k, v in S["by_c"].items():
        if v["n_seeds"] >= 2:
            cs.append(float(k)); ms.append(v["ci"][0]); lo.append(v["ci"][2]); hi.append(v["ci"][3]); ns.append(v["n_draws"])
    cs, ms, lo, hi = map(np.array, (cs, ms, lo, hi))
    b.errorbar(cs, ms, yerr=[ms - lo, hi - ms], color=COL["D"], marker="o", ms=8, capsize=5)
    for c_, n_ in zip(cs, ns):
        b.text(c_, 0.02, f"n = {n_}", transform=b.get_xaxis_transform(), ha="center", fontsize=10.5, color=MUTED)
    b.axhline(0, color=MUTED, lw=1.2); b.axhline(0.02, color=COL["C"], lw=1.6, ls="--")
    b.set_xlim(-0.1, 1.1); b.set_xlabel("realised pair concentration c  (all draws pooled)"); b.set_title("(b) Gap vs realised correlation")
    fig.suptitle("The graph model does not beat the node-local one, and the gap does not widen with correlation\n"
                 "S4 diagnostic AUC, pre-registered rule: NEGATIVE.  n = number of draws pooled in each bin.",
                 x=0.125, ha="left", fontsize=14, fontweight="bold", y=1.04)
    save(fig, "fig5_s4_gap_vs_correlation")


# ------------------------------------------------------------------ Figure 6: worked explanation
def fig_explanation() -> None:
    d = json.loads((S10 / "example.json").read_text(encoding="utf-8"))
    paras = [p for p in d["text"].split("\n\n") if p.strip()]
    head = next(p for p in paras if p.startswith("Decision "))
    other = next(p for p in paras if p.startswith("Other candidates"))
    ctx = next(p for p in paras if p.startswith("Logged behaviour record"))
    l2 = next(p for p in paras if "Primary factors" in p)
    l2_main = l2.split("\n")[0]
    l2_note = [ln for ln in l2.split("\n")[1:] if ln.startswith("(Layer 2")]
    h_lines = head.split("\n")  # title / who won / why

    fig = plt.figure(figsize=(15, 10.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.82], wspace=0.04, left=0.02, right=0.98, top=0.905, bottom=0.03)
    tx = fig.add_subplot(gs[0]); tx.axis("off"); tx.set_xlim(0, 1); tx.set_ylim(-0.25, 1)
    fig.suptitle("What an operator sees for one real decision (Variant E: D's decision plus the explanation)", x=0.02, ha="left", fontsize=16, fontweight="bold", y=0.99)

    def block(y, title, body, width, color=INK, size=12.2, edge=FAINT, extra=0.0):
        wrapped = textwrap.fill(body, width=width, replace_whitespace=False)
        n = wrapped.count("\n") + 1
        h = n * size * 1.45 / 72 / 7.52 + 0.06 + extra
        tx.add_patch(FancyBboxPatch((0.0, y - h), 1.0, h, boxstyle="round,pad=0.005,rounding_size=0.012", fc="#fafbfc", ec=edge, lw=1.6))
        tx.text(0.02, y - 0.016, title, fontsize=11, fontweight="bold", color=MUTED, va="top")
        tx.text(0.02, y - 0.056, wrapped, fontsize=size, va="top", color=color, linespacing=1.45)
        return y - h - 0.018

    y = 1.0
    y = block(y, "DECISION", h_lines[0] + "\n" + h_lines[1], 74)
    y = block(y, "LAYER 1  -  exact: why the runner-up lost", h_lines[2].replace("Why: ", ""), 74)
    y = block(y, "OTHER CANDIDATES", other.replace("Other candidates:\n", "").strip(), 74, size=11.2)
    y = block(y, "LAYER 2  -  GNNExplainer on the trust head (asynchronous)", l2_main, 74, size=11.2)
    note = (l2_note[0] if l2_note else "") + "\n" + ctx.split("\nOutcome")[0].replace("Logged behaviour record (context, not an attribution): ", "Logged record: ")
    y = block(y, "HOW TO READ IT", note.strip(), 84, color=MUTED, size=10.2, extra=0.03)

    bx = fig.add_subplot(gs[1])
    cands = sorted([c for c in d["candidates"] if c["dwell_eligible"]], key=lambda c: -c["score"])
    names = [f"RSU-{c['rsu']}" + ("  (selected)" if c["rsu"] == d["selected"] else "") + ("  (runner-up)" if c["rsu"] == d["runner_up"] else "") for c in cands]
    ypos = np.arange(len(cands))[::-1]
    tcol, lcol, gcol = "#2a6fb0", "#b7791f", "#8b5cf6"
    for yi, c in zip(ypos, cands):
        left_pos = 0.0; left_neg = 0.0
        pos = c["trust_term"]
        bx.barh(yi, pos, 0.5, left=0, color=tcol, label="alpha * trust" if yi == ypos[0] else None)
        bx.barh(yi, c["latency_term"], 0.5, left=0, color=lcol, label="- beta * latency" if yi == ypos[0] else None)
        bx.barh(yi, c["load_term"], 0.5, left=c["latency_term"], color=gcol, label="- gamma * load" if yi == ypos[0] else None)
        bx.plot([c["score"]], [yi], marker="D", color=INK, ms=9, zorder=4, label="score" if yi == ypos[0] else None)
        bx.text(c["score"] + 0.05, yi + 0.34, f"{c['score']:+.2f}", fontsize=11.5, fontweight="bold")
    bx.set_yticks(ypos); bx.set_yticklabels(names); bx.axvline(0, color=MUTED, lw=1.2)
    bx.set_xlabel("contribution to the selection score")
    w = d["weights"]
    bx.set_title(f"score = {w['alpha']:g} x trust - {w['beta']:g} x latency - {w['gamma']:g} x load", fontsize=13, pad=14)
    bx.legend(frameon=False, loc="lower right", fontsize=11)
    save(fig, "fig6_explanation")


# ------------------------------------------------------------------ tables
def write_table(name: str, header: list[str], rows: list[list[str]], caption: str, note: str = "") -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    with (TAB / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    if note:
        md += ["", note]
    (TAB / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    esc = lambda s: s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("±", r"$\pm$").replace("✓", r"\checkmark").replace("✗", "--")  # noqa: E731
    tex = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\begin{{tabular}}{{l{'r' * (len(header) - 1)}}}", r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    if note:
        tex += [rf"\par\smallskip\footnotesize {esc(note)}"]
    tex += [r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote results/s10/tables/{name}.md .csv .tex")


def pm(ms, nd=3):
    return f"{ms[0]:.{nd}f} ± {ms[1]:.{nd}f}"


def tables(T, P) -> None:
    S = T[SHARED]
    rows = []
    for v, src in (("A", "A"), ("B", "B"), ("B_oracle", "B_oracle"), ("C", "C"), ("D", "D"), ("E", "D")):
        s = S[src]
        rows.append([{"B_oracle": "B-oracle", "E": "E (= D)"}.get(v, v), pm(s["success_post_onset"]), pm(s["success_all"]),
                     pm(s["n_av"], 0), pm(s["n_win"], 0), pm(s["never_avoided"], 2),
                     "never" if s["detected_fraction"][0] == 0 else pm(s["latency"], 0), pm(s["detected_fraction"], 2)])
    write_table("ablation_shared_point",
                ["Variant", "Success after onset", "Success, all steps", "To degraded, onset to avoidance", "To degraded, [400,1200)",
                 "Runs never avoiding", "Detection latency (steps)", "Degraded RSUs detected"], rows,
                "Ablation at the shared operating point (rho 1.0, 20% degraded): mean ± sd across 5 seeds (50 draws each)",
                "B decides identically to A in this scenario and E identically to D (both asserted); B-oracle reads the sealed fault state and is an upper bound, "
                "not a method. Detection latency is censored at 399 steps; A and B cannot detect by construction.")
    pts = [(1.0, 0.05), (1.0, 0.10), (1.0, 0.20), (1.0, 0.30), (0.0, 0.20), (0.25, 0.20), (0.5, 0.20), (0.75, 0.20)]
    rows = []
    for rho, fr in pts:
        for v in ("A", "B_oracle", "C", "D"):
            s = T[(rho, fr)][v]
            rows.append([f"{rho:g}", f"{fr:.2f}", {"B_oracle": "B-oracle"}.get(v, v), pm(s["success_post_onset"]), pm(s["n_av"], 0), pm(s["n_win"], 0)])
    write_table("ablation_all_points", ["rho", "fraction", "Variant", "Success after onset", "To degraded, onset to avoidance", "To degraded, [400,1200)"],
                rows, "All sweep points: mean ± sd across 5 seeds", "Rho sweep at fraction 0.20, fraction sweep at rho 1.0; the (1.0, 0.20) rows are shared. B = A and E = D throughout.")

    verdict = json.loads((S9 / "summary-verdict.json").read_text(encoding="utf-8"))["verdicts"]
    rows = []
    for x, ref in (("C", "A"), ("D", "A"), ("C", "B_oracle"), ("D", "B_oracle"), ("D", "C")):
        e = verdict["1.00/0.20"][f"{x}_vs_{ref}"]
        rows.append([f"{x} vs {ref}".replace("B_oracle", "B-oracle"), f"{e['H3']['mean']:+.4f} [{e['H3']['lo']:+.4f}, {e['H3']['hi']:+.4f}]", e["H3"]["label"],
                     f"{e['H1']['mean']:+.3f} [{e['H1']['lo']:+.3f}, {e['H1']['hi']:+.3f}]", e["H1"]["label"]])
    write_table("paired_effects_shared_point", ["Comparison", "H3 success difference [95% CI]", "H3 label", "H1 reduction [95% CI]", "H1 label"], rows,
                "Paired effects at the shared point (per draw, seed means, Student-t over 5 seeds); labels from the rule fixed in D56/D57",
                "H1 reduction = (n_ref - n_X) / n_A. D vs C is the closed-loop secondary check of H2 and never gates.")

    o = json.loads((S9 / "overhead.json").read_text(encoding="utf-8"))
    rows = []
    comp = [("trust_inference", "Trust inference (one forward pass over the step graph)"), ("selection", "Selection (analytic rule)"),
            ("layer1", "Layer 1 explanation (exact decomposition + text)"), ("layer2_per_decision", "Layer 2 explanation (GNNExplainer, winner and runner-up)")]
    for key, r in o["timing"].items():
        th = key.replace("threads", "")
        for c, lab in comp:
            x = r[c]
            rows.append([lab, f"{th}", f"{x['median']:.3f}", f"{x['mean']:.3f}", f"{x['p95']:.3f}"])
    write_table("latency_breakdown", ["Component", "Torch threads", "Median (ms)", "Mean (ms)", "p95 (ms)"], rows,
                "Per-decision time by component (200 decisions: 5 seeds x 40, shared point)",
                f"Layer 2 against the stated {o['budget_ms']:g} ms assumption: asynchronous only. {o['hardware'].get('cpu', '')}, CPU, one decision at a time.")

    rel = [("1", "GNN-Mamba Spatiotemporal Trust Evaluation", "✓", "✓", "✗", "✗"), ("2", "Multi-Hop Trust via GNN-Aided Agentic AI", "✓", "✓", "✗", "✗"),
           ("3", "Federated GNN Multi-Agent RL for AoI in VEC", "✗", "✓", "✓", "✗"), ("4", "Joint Allocation Under Asymmetric Information", "~ (economic)", "✗", "✓", "✗"),
           ("5", "DRL for Delay-Optimized VFC Offloading", "✗", "✗", "✓", "✗"), ("6", "Explainability-as-a-Service for Edge AI", "✗", "✗", "✗", "✓"),
           ("7", "XRL for RAN Slicing (6G ORAN)", "✗", "✗", "✗", "✓"), ("8", "Self-Explaining RL for Mobile Resource Allocation", "✗", "✗", "✗", "✓"),
           ("9", "XAI for RL-based Networking", "✗", "✗", "✗", "✓"), ("10", "Explainable Fraud Detection (GNNExplainer + Shapley)", "✗", "✓", "✗", "✓"),
           ("this", "This work", "✓", "✓ (built; no measured gain over node-local, F17/F27)", "✓", "✓")]
    write_table("related_work", ["Ref", "Work", "Learned trust", "Structure-aware", "Vehicular offloading", "Explainable"], [list(r) for r in rel],
                "Prior art, mapped (rows from TECHNICAL_AND_NOVELTY.md section 6; S11 updates them)",
                "TO BE UPDATED IN S11. The structure-aware mark for this work records that message passing is implemented, not that it helped: H2 was tested and not supported (S4).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute", action="store_true", help="rebuild the cached closed-loop inputs of Figures 2 and 6")
    a = ap.parse_args()
    ensure_inputs(a.recompute)
    T, P = load_table(), load_paired()
    print("figures:")
    fig_architecture(); fig_timeline(); fig_success_vs_fraction(T, P); fig_ablation(T, P); fig_s4_gap(); fig_explanation()
    print("tables:")
    tables(T, P)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

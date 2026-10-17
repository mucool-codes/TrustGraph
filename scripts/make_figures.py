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
COL = {"A": "#6b7280", "B": "#9ca3af", "B_oracle": "#2f855a", "C": "#2a6fb0", "D": "#c2571a", "E": "#c2571a", "Gate_C": "#7c5cbf", "H": "#a67c00"}
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


def load_gate() -> dict:
    """Gate-C (S9b, D59) at the four fraction-sweep points, from results/s9b/raw (read through s9b_gate's json loader; no sealed access).

    Returns {"T": {(rho, frac): {metric: (mean, sd)}}, "paired": {(rho, frac): Effect of C minus Gate-C success},
    "vs_a": {(rho, frac): Effect of Gate-C minus A success}}.
    """
    if not (REPO_ROOT / "results" / "s9b" / "raw").exists():
        raise SystemExit("results/s9b/raw missing: run `python scripts/s9b_gate.py run` first (S9b)")
    import s9b_gate as G
    from trustgraph import sweep_metrics as M

    g = G.load_gate()
    T, paired, vs_a = {}, {}, {}
    for pt, d in G.load_arrays(g).items():
        a = d["a"]
        sm = lambda x: (float(np.mean(M.seed_means(x))), float(np.std(M.seed_means(x), ddof=1)))  # noqa: E731
        key = (round(pt[0], 2), round(pt[1], 2))
        T[key] = {"success_post_onset": sm(a["g_succ"]), "success_all": sm(a["g_all"]), "n_av": sm(a["g_nav"]), "n_win": sm(a["g_nwin"]),
                  "never_avoided": sm(a["g_never"]), "fallback_all": sm(a["fb_all"]), "fallback_post": sm(a["fb_post"])}
        paired[key] = M.h3_effect(a["c_succ"], a["g_succ"])
        vs_a[key] = M.h3_effect(a["g_succ"], a["a_succ"])
    return {"T": T, "paired": paired, "vs_a": vs_a, "tau": g["tau"]}


def load_h() -> dict:
    """Heuristic baseline H (S13, D65) at the four fraction-sweep points, from results/s13/h/raw (no sealed access).

    Returns {"T": {(rho, frac): {metric: (mean, sd)}}, "paired": {(rho, frac): (mean, lo, hi, D65 label) of C minus H success},
    "vs_a": {(rho, frac): Effect of H minus A success}, "weights": (alpha, beta, gamma)}.
    """
    if not (REPO_ROOT / "results" / "s13" / "h" / "raw").exists():
        raise SystemExit("results/s13/h/raw missing: run `python scripts/s13_h.py run` first (S13)")
    import s13_h as Hm
    from trustgraph import sweep_metrics as M

    g = Hm.load_h()
    T, paired, vs_a = {}, {}, {}
    for pt, d in Hm.load_arrays(g).items():
        a = d["a"]
        sm = lambda x: (float(np.mean(M.seed_means(x))), float(np.std(M.seed_means(x), ddof=1)))  # noqa: E731
        key = (round(pt[0], 2), round(pt[1], 2))
        T[key] = {"success_post_onset": sm(a["h_succ"]), "success_all": sm(a["h_sall"]), "n_av": sm(a["h_nav"]), "n_win": sm(a["h_nwin"]),
                  "never_avoided": sm(a["h_never"]), "latency": sm(a["h_lat"]), "detected_fraction": sm(a["h_det"])}
        m, _, lo, hi = Hm.diff(a["c_succ"], a["h_succ"])
        paired[key] = (m, lo, hi, Hm.label_d65(m, lo, hi))
        vs_a[key] = M.h3_effect(a["h_succ"], a["a_succ"])
    w = Hm.yaml.safe_load(Hm.weights_path(g).read_text())["weights"]["H"]
    return {"T": T, "paired": paired, "vs_a": vs_a, "weights": (w["alpha"], w["beta"], w["gamma"])}


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
    box(ax, xs[2], y, w, h, "3  Offloading rule", ["fixed, not learned", "score = a*trust - b*latency", "- g*load, argmax over in-range,", "dwell-feasible RSUs",
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
        ["injected behaviour class and segment faults", "used only to score trust and dispatch", "never a feature, a label, or importable", "by training code; no arrow enters the pipeline"],
        face="#f5f5f5", edge=MUTED, dashed=True, tcolor=MUTED, size=10.5)
    ax.text(8.1, -0.55, "Variants: A no trust | B certificate only | B-oracle upper bound | C node-local MLP | D GNN | E = D + explanation | Gate-C, M extra baselines.  Same rule in all.",
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
def fig_success_vs_fraction(T, P, GT, HH) -> None:
    fr = [0.05, 0.10, 0.20, 0.30]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"wspace": 0.22})
    a = ax[0]
    order = [("A", "A = B  (identical decisions)", "o"), ("B_oracle", NAME["B_oracle"], "s"), ("C", NAME["C"], "^"),
             ("Gate_C", "Gate-C  (extra baseline)", "v"), ("H", "M  monitor: own success_ewma (extra baseline)", "P"),
             ("D", "D = E  (same dispatch)", "D")]
    off = {"A": -1.5, "B_oracle": -0.9, "C": -0.3, "Gate_C": 0.3, "H": 0.9, "D": 1.5}
    for v, lab, mk in order:
        src_ = {f: (GT["T"][(1.0, f)] if v == "Gate_C" else HH["T"][(1.0, f)] if v == "H" else T[(1.0, f)][v]) for f in fr}
        m = [src_[f]["success_post_onset"][0] for f in fr]; s = [src_[f]["success_post_onset"][1] for f in fr]
        a.errorbar(np.array(fr) * 100 + off[v], m, yerr=s, color=COL[v], marker=mk, ms=8, capsize=4, lw=2.2,
                   ls="--" if v == "B_oracle" else "-", label=lab)
    a.set_xticks([5, 10, 20, 30]); a.set_xlabel("degraded fraction of RSUs (%)  at rho = 1.0"); a.set_ylabel("task success rate after onset")
    a.set_title("(a) Success, mean +/- sd over 5 seeds"); a.legend(frameon=False, fontsize=11, loc="lower left")
    a.text(30.5, 0.9755, "at 5% only one RSU is degraded:\nno correlated failure", fontsize=10.5, color=MUTED, ha="right", va="top")
    b = ax[1]
    for v, off_ in (("C", -0.9), ("Gate_C", -0.3), ("H", 0.3), ("D", 0.9)):
        if v in ("Gate_C", "H"):
            e = [(GT if v == "Gate_C" else HH)["vs_a"][(1.0, f)] for f in fr]
            m = np.array([x.mean for x in e]); lo = np.array([x.lo for x in e]); hi = np.array([x.hi for x in e])
        else:
            m = np.array([P[(1.0, f, v, "A")]["h3_mean"] for f in fr]); lo = np.array([P[(1.0, f, v, "A")]["h3_lo"] for f in fr])
            hi = np.array([P[(1.0, f, v, "A")]["h3_hi"] for f in fr])
        b.errorbar(np.array(fr) * 100 + off_, m, yerr=[m - lo, hi - m], color=COL[v], marker={"Gate_C": "v", "H": "P"}.get(v, "o"), ms=8, capsize=4,
                   label=("Gate-C minus A  (extra baseline)" if v == "Gate_C" else "M minus A  (extra baseline)" if v == "H" else f"{v} minus A"))
    b.axhline(0, color=MUTED, lw=1.2); b.axhline(0.01, color=MUTED, lw=1.2, ls=":")
    b.text(30.8, 0.0115, "pre-registered floor 0.01", ha="right", fontsize=10.5, color=MUTED)
    b.set_xticks([5, 10, 20, 30]); b.set_xlabel("degraded fraction of RSUs (%)  at rho = 1.0"); b.set_ylabel("paired gain in success rate")
    b.set_title("(b) Gain over A, 95% interval over seeds"); b.legend(frameon=False, loc="upper left")
    save(fig, "fig3_success_vs_fraction")


# ------------------------------------------------------------------ Figure 4: ablation
def fig_ablation(T, P, GT, HH) -> None:
    S = T[SHARED]
    variants = ["A", "B", "C", "D", "E", "Gate_C", "H"]
    src = {"E": "D"}
    label = {"Gate_C": "Gate-C", "H": "M"}
    panels = [("success_post_onset", "task success rate after onset\n(axis starts at 0.80)", (0.80, 0.96), "{:.3f}", "h3", "{:+.3f}"),
              ("n_av", "tasks sent to degraded RSUs,\nonset to avoidance", (0, 760), "{:.0f}", "h1", "{:+.3f}"),
              ("n_win", "tasks sent to degraded RSUs,\nfixed window [400, 1200)", (0, 1050), "{:.0f}", None, None)]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.0), gridspec_kw={"wspace": 0.42})
    for a, (m, ylab, ylim, f, pk, pf) in zip(ax, panels):
        for i, v in enumerate(variants):
            mean, sd = (GT["T"][SHARED] if v == "Gate_C" else HH["T"][SHARED] if v == "H" else S[src.get(v, v)])[m]
            a.bar(i, mean, 0.7, color=COL[v], hatch="//" if v == "E" else None, edgecolor="white" if v == "E" else COL[v], lw=0,
                  alpha=0.45 if v == "B" else 1.0, zorder=2)
            a.errorbar(i, mean, yerr=sd, color=INK, capsize=5, lw=1.8, zorder=3)
            a.text(i, mean + sd + (ylim[1] - ylim[0]) * 0.015, f.format(mean), ha="center", fontsize=11)
        om = S["B_oracle"][m][0]
        a.axhline(om, color=COL["B_oracle"], ls="--", lw=2.0, zorder=1)
        a.text(6.5, om, "  oracle", color=COL["B_oracle"], ha="left", va="center", fontsize=11, clip_on=False)
        a.set_xticks(range(len(variants))); a.set_xticklabels([label.get(v, v) for v in variants]); a.set_ylim(*ylim); a.set_xlim(-0.6, 6.6); a.set_ylabel(ylab)
        a.spines["bottom"].set_visible(True)
        a.add_patch(plt.Rectangle((1.55, ylim[0]), 1.9, ylim[1] - ylim[0], fc="#fff4c2", ec="none", alpha=0.75, zorder=0))
        a.text(2.5, 1.02, "C vs D: the structural test", transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=11.5, fontweight="bold")
        if pk:
            d = P[(1.0, 0.2, "D", "C")]
            lab = f"D minus C: {pf.format(d[pk + '_mean'])}  [{pf.format(d[pk + '_lo'])}, {pf.format(d[pk + '_hi'])}]"
            if pk == "h3":
                e = GT["paired"][SHARED]
                lab += f"\nC minus Gate-C: {e.mean:+.4f}  [{e.lo:+.4f}, {e.hi:+.4f}]"
                hm, hlo, hhi, _ = HH["paired"][SHARED]
                lab += f"\nC minus M: {hm:+.4f}  [{hlo:+.4f}, {hhi:+.4f}]"
            a.set_xlabel(lab, fontsize=11.5)
        else:
            a.set_xlabel("(rule-free count, a guard on the middle panel)", fontsize=11, color=MUTED)
    fig.suptitle("Ablation at the shared operating point (rho 1.0, 20% degraded). Bars: mean over 5 seeds, whiskers: sd.\n"
                 "B equals A and E equals D (identical decisions, asserted). Dashed green: B-oracle, an upper bound.\n"
                 f"Gate-C: an extra baseline (frozen C head, candidates with trust below {GT['tau']:g} removed first), run at the fraction-sweep points only.\n"
                 "M: an extra baseline, trust = the RSU's own success_ewma (no learning, same rule, no gate), run at the fraction-sweep points only.",
                 x=0.125, ha="left", fontsize=13.5, fontweight="bold", y=1.12)
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
    fig.suptitle("The graph model is worse than the node-local one, and the gap does not close with correlation\n"
                 "Shared-onset diagnostic AUC, pre-registered rule: NEGATIVE.  n = number of draws pooled in each bin.",
                 x=0.125, ha="left", fontsize=14, fontweight="bold", y=1.04)
    save(fig, "fig5_s4_gap_vs_correlation")


# ------------------------------------------------------------------ Figure 7: S18 D-minus-C gap vs spreading-fault lag (D81)
S18 = REPO_ROOT / "results" / "s18"


def fig_s18_gap() -> None:
    P = json.loads((S18 / "summary-open-primary.json").read_text(encoding="utf-8"))
    Q = json.loads((S18 / "summary-open-secondary.json").read_text(encoding="utf-8"))
    conds, Ls = ["L0", "L20", "L40", "L80"], [0, 20, 40, 80]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.22})
    for a, key, ylab, ttl in ((ax[0], "lat", "D minus C detection latency (steps)", "(a) Detection latency"),
                              (ax[1], "fa", "D minus C healthy false-alarm rate", "(b) False alarms")):
        for S, dx, fill, lab in ((P, -2.0, True, "frozen heads (primary; decides Sa, Sc)"), (Q, 2.0, False, "retrained heads (secondary; descriptive)")):
            m = np.array([S[c][key][0] for c in conds]); lo = np.array([S[c][key][2] for c in conds]); hi = np.array([S[c][key][3] for c in conds])
            a.errorbar(np.array(Ls) + dx, m, yerr=[m - lo, hi - m], color=COL["D"], marker="o", ms=8, capsize=5, ls="-" if fill else ":",
                       mfc=COL["D"] if fill else "white", label=lab)
        a.axhline(0, color=MUTED, lw=1.2)
        if key == "lat":
            a.axhline(-20, color=COL["C"], lw=1.6, ls="--"); a.text(1, -17, "WIN floor: D earlier by 20 steps", color=COL["C"], fontsize=11, va="bottom")
        else:
            a.axhline(0.05, color=COL["C"], lw=1.6, ls="--"); a.text(1, 0.06, "tolerance +0.05", color=COL["C"], fontsize=11, va="bottom")
        a.set_xticks(Ls); a.set_xlabel("lag L per hop (steps); L = 0 is the control"); a.set_ylabel(ylab); a.set_title(ttl)
    ax[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=11)
    fig.suptitle("The graph head detects later than the node-local head at every lag,\n"
                 "including the L = 0 control. Spreading fault, draws 400-449, 5 seeds,\n"
                 "95% intervals; shared-rule latency (not a time-to-alarm), false alarms\n"
                 "beside it. Sa NOT SUPPORTED.",
                 x=0.125, ha="left", fontsize=14, fontweight="bold", y=1.14)
    save(fig, "fig7_s18_gap_vs_lag")


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
def write_table(name: str, header: list[str], rows: list[list[str]], caption: str, note: str = "", wide: bool = False) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    plain = [[c.replace("\n", " ") for c in r] for r in rows]  # a newline inside a cell is a two-line cell in the .tex only
    with (TAB / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(plain)
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in plain]
    if note:
        md += ["", note]
    (TAB / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    esc = lambda s: s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("±", r"$\pm$").replace("✓", r"\checkmark").replace("✗", "--")  # noqa: E731
    paper = name.startswith("paper_")  # S12 paper tables: labelled, and shrunk to the column / page width only if wider than it
    env = "table*" if wide else "table"  # wide = two-column paper layout (S12); the default layout is unchanged
    tex = [rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{esc(caption)}}}"]
    tex += [rf"\label{{tab:{name}}}"] if paper else []
    fit = r"\textwidth" if wide else r"\columnwidth"
    tex += [r"\resizebox{\ifdim\width>" + fit + " " + fit + r"\else\width\fi}{!}{%"] if paper else []
    tex += [rf"\begin{{tabular}}{{l{'r' * (len(header) - 1)}}}", r"\toprule", " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    cell = lambda c: (r"\shortstack[r]{" + r" \\[0pt] ".join(esc(x) for x in c.split("\n")) + "}") if "\n" in c else esc(c)  # noqa: E731
    tex += [" & ".join(cell(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    tex += ["}"] if paper else []
    if note:
        tex += [rf"\par\smallskip\footnotesize {esc(note)}"]
    tex += [rf"\end{{{env}}}"]
    (TAB / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote results/s10/tables/{name}.md .csv .tex")


def pm(ms, nd=3):
    return f"{ms[0]:.{nd}f} ± {ms[1]:.{nd}f}"


def tables(T, P, GT, HH) -> None:
    S = T[SHARED]
    rows = []
    for v, src in (("A", "A"), ("B", "B"), ("B_oracle", "B_oracle"), ("C", "C"), ("D", "D"), ("E", "D")):
        s = S[src]
        rows.append([{"B_oracle": "B-oracle", "E": "E (= D)"}.get(v, v), pm(s["success_post_onset"]), pm(s["success_all"]),
                     pm(s["n_av"], 0), pm(s["n_win"], 0), pm(s["never_avoided"], 2),
                     "never" if s["detected_fraction"][0] == 0 else pm(s["latency"], 0), pm(s["detected_fraction"], 2)])
    g = GT["T"][SHARED]
    rows.append(["Gate-C (extra baseline)", pm(g["success_post_onset"]), pm(g["success_all"]), pm(g["n_av"], 0), pm(g["n_win"], 0),
                 pm(g["never_avoided"], 2), "not reported (D59)", "not reported (D59)"])
    h = HH["T"][SHARED]
    rows.append(["H (heuristic baseline)", pm(h["success_post_onset"]), pm(h["success_all"]), pm(h["n_av"], 0), pm(h["n_win"], 0),
                 pm(h["never_avoided"], 2), pm(h["latency"], 0), pm(h["detected_fraction"], 2)])
    write_table("ablation_shared_point",
                ["Variant", "Success after onset", "Success, all steps", "To degraded, onset to avoidance", "To degraded, [400,1200)",
                 "Runs never avoiding", "Detection latency (steps)", "Degraded RSUs detected"], rows,
                "Ablation at the shared operating point (rho 1.0, 20% degraded): mean ± sd across 5 seeds (50 draws each)",
                "B decides identically to A in this scenario and E identically to D (both asserted); B-oracle reads the sealed fault state and is an upper bound, "
                "not a method. Detection latency is censored at 399 steps; A and B cannot detect by construction. Gate-C is an extra baseline: the frozen C head with candidates "
                f"below trust {GT['tau']:g} removed before the same rule (D59); it was run at the four fraction-sweep points only and its detection figures are not reported. "
                "H is an extra baseline (S13, D65): trust is the RSU's own success_ewma, no learning, the same rule with no gate, weights tuned on the tuning seed; run at the four fraction-sweep points only.")
    pts = [(1.0, 0.05), (1.0, 0.10), (1.0, 0.20), (1.0, 0.30), (0.0, 0.20), (0.25, 0.20), (0.5, 0.20), (0.75, 0.20)]
    rows = []
    for rho, fr in pts:
        for v in ("A", "B_oracle", "C", "D"):
            s = T[(rho, fr)][v]
            rows.append([f"{rho:g}", f"{fr:.2f}", {"B_oracle": "B-oracle"}.get(v, v), pm(s["success_post_onset"]), pm(s["n_av"], 0), pm(s["n_win"], 0)])
        if (round(rho, 2), round(fr, 2)) in GT["T"]:
            gg = GT["T"][(round(rho, 2), round(fr, 2))]
            rows.append([f"{rho:g}", f"{fr:.2f}", "Gate-C (extra baseline)", pm(gg["success_post_onset"]), pm(gg["n_av"], 0), pm(gg["n_win"], 0)])
        if (round(rho, 2), round(fr, 2)) in HH["T"]:
            hh = HH["T"][(round(rho, 2), round(fr, 2))]
            rows.append([f"{rho:g}", f"{fr:.2f}", "H (heuristic baseline)", pm(hh["success_post_onset"]), pm(hh["n_av"], 0), pm(hh["n_win"], 0)])
    write_table("ablation_all_points", ["rho", "fraction", "Variant", "Success after onset", "To degraded, onset to avoidance", "To degraded, [400,1200)"],
                rows, "All sweep points: mean ± sd across 5 seeds", "Rho sweep at fraction 0.20, fraction sweep at rho 1.0; the (1.0, 0.20) rows are shared. B = A and E = D throughout. Gate-C and H were run at the four fraction-sweep points only.")

    verdict = json.loads((S9 / "summary-verdict.json").read_text(encoding="utf-8"))["verdicts"]
    rows = []
    for x, ref in (("C", "A"), ("D", "A"), ("C", "B_oracle"), ("D", "B_oracle"), ("D", "C")):
        e = verdict["1.00/0.20"][f"{x}_vs_{ref}"]
        rows.append([f"{x} vs {ref}".replace("B_oracle", "B-oracle"), f"{e['H3']['mean']:+.4f} [{e['H3']['lo']:+.4f}, {e['H3']['hi']:+.4f}]", e["H3"]["label"],
                     f"{e['H1']['mean']:+.3f} [{e['H1']['lo']:+.3f}, {e['H1']['hi']:+.3f}]", e["H1"]["label"]])
    e = GT["paired"][SHARED]
    rows.append(["C vs Gate-C (extra baseline)", f"{e.mean:+.4f} [{e.lo:+.4f}, {e.hi:+.4f}]", e.label, "not gating (see F29)", "-"])
    hm, hlo, hhi, hlab = HH["paired"][SHARED]
    rows.append(["C vs H (heuristic baseline)", f"{hm:+.4f} [{hlo:+.4f}, {hhi:+.4f}]", hlab, "not gating (see F33)", "-"])
    write_table("paired_effects_shared_point", ["Comparison", "H3 success difference [95% CI]", "H3 label", "H1 reduction [95% CI]", "H1 label"], rows,
                "Paired effects at the shared point (per draw, seed means, Student-t over 5 seeds); labels from the rule fixed in D56/D57",
                "H1 reduction = (n_ref - n_X) / n_A. D vs C is the closed-loop secondary check of H2 and never gates. C vs Gate-C is the D59 comparison (WIN would mean the weighted rule beats the gate; "
                "floor 0.01); its label is the shared-point label, the D59 verdict uses all four fraction points (NOT SEPARATED, F30). "
                "C vs H is the D65 comparison (WIN would mean the learned head beats the heuristic; floor 0.01, NULL = interval inside +-0.025); its label is the shared-point label, the D65 verdict uses all four fraction points (NOT SEPARATED, F34).")

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
           ("11", "SGA-SAC: Bayesian Stackelberg + graph-attention DRL, explainable VEC offloading (Cheng et al., Inf. Fusion 2026)", "✗", "✓ (graph attention)", "✓", "✓ (Grad-CAM)"),
           ("12", "Trust-Based Malicious RSU Detection in Edge-Enabled VANETs (Siddiqua and Jahan, arXiv 2208.05680)", "✗", "~ (RSU-RSU interaction)", "✗", "✗"),
           ("13", "G-TRAC: Trust-Aware Routing for Distributed GenAI Inference at the Edge (Nguyen and Elmroth, DCOSS-IoT 2026)", "✗", "~ (path search over peers)", "✗", "✗"),
           ("this", "This work", "✓", "✓ (built; no measured gain over node-local, F17/F27)", "✓", "✓")]
    write_table("related_work", ["Ref", "Work", "Learned trust", "Structure-aware", "Vehicular offloading", "Explainable"], [list(r) for r in rel],
                "Prior art, mapped (rows 1-10 from TECHNICAL_AND_NOVELTY.md section 6; rows 11-13 added in S11a, see RELATED_WORK.md)",
                "Rows 11-13 are filled from the abstracts only (verified 2026-10-15): a cross means the abstract does not describe the property, not that the paper lacks it; ~ means partly or in a different sense. "
                "The structure-aware mark for this work records that message passing is implemented, not that it helped: H2 was tested and not supported (S4).")


# ------------------------------------------------------------------ S12: tables of the paper (tables only; no figure is redrawn)
import re  # noqa: E402

S13 = REPO_ROOT / "results" / "s13"
S14 = REPO_ROOT / "results" / "s14"
S15 = REPO_ROOT / "results" / "s15"
S16 = REPO_ROOT / "results" / "s16"
# CLAIMS.md (D78): the caveat carried by every detection latency the paper reports
LATENCY_CAVEAT_SHORT = "Latency: the shared-rule caveat of Sec. VI applies."
LATENCY_CAVEAT = ("Latency is compared under one shared rule, not at a detection operating point; the rule's false-alarm rate was 0.32–0.44 of healthy RSUs "
                  "where measured (S16) and was not measured elsewhere.")
_NUM = r"([+-]?\d+\.\d+)"
_CI = rf"{_NUM} \(sd [\d.]+\) 95% \[{_NUM}, {_NUM}\]"


def _ci(m_, lo, hi) -> str:
    return f"{m_} [{lo}, {hi}]"


def _read_report(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"{path} missing: it is written by the S13-S16 report commands (scripts/s13_stag.py, s14_d2.py, s15_horizon.py, s16_confirm.py)")
    return path.read_text(encoding="utf-8").splitlines()


def _latency_blocks(path: Path, who: str) -> dict:
    """{rho: {"C": str, who: str, "diff": "m [lo, hi]", "label": str, "auc": "m [lo, hi] or None"}} from a stored staggered-onset report.

    Strings are copied verbatim from the report, so the table cannot differ from the report in rounding.
    """
    out, rho = {}, None
    for ln in _read_report(path):
        h = re.match(r"rho (\d)\s+\((CONTROL|TEST)\)", ln)
        if h:
            rho = h.group(1)
            out[rho] = {"auc": None}
            continue
        if rho is None:
            continue
        m_ = re.search(rf"detection latency .*: C ([\d.]+) \+/- [\d.]+\s+{who} ([\d.]+) \+/- [\d.]+", ln)
        if m_:
            out[rho]["C"], out[rho][who] = m_.group(1), m_.group(2)
        m_ = re.search(rf"PRIMARY\s+{who} - C latency: {_CI}\s+(\w+)", ln)
        if m_:
            out[rho]["diff"], out[rho]["label"] = _ci(*m_.group(1, 2, 3)), m_.group(4)
        m_ = re.search(rf"\(secondary\) window AUC: .*{who} - C {_NUM} \(sd [\d.]+\) 95% \[{_NUM}, {_NUM}\]", ln)
        if m_:
            out[rho]["auc"] = _ci(*m_.group(1, 2, 3))
    return out


def _shared_auc(path: Path) -> dict:
    """{rho: ("m [lo, hi]", label)} for S14 secondary (a), D2 minus C diagnostic AUC on the shared regime."""
    out, rho, on = {}, None, False
    for ln in _read_report(path):
        on = on or "SECONDARY (a)" in ln
        h = re.match(r"rho (\d)\s", ln)
        if on and h:
            rho = h.group(1)
        m_ = re.search(rf"D2 - C: {_CI}\s+(\w+)", ln)
        if on and rho is not None and m_:
            out[rho] = (_ci(*m_.group(1, 2, 3)), m_.group(4))
    return out


def _s14_closed(path: Path) -> dict:
    """{regime: {"diff": "m [lo, hi]", "label": str, "A"/"C"/"D2": (n_av, n_win, never)}} from the S14 closed-loop verdict report."""
    out, reg = {}, None
    for ln in _read_report(path):
        h = re.match(r"--- regime (\w+):", ln)
        if h:
            reg = h.group(1)
            out[reg] = {}
            continue
        if reg is None:
            continue
        m_ = re.search(rf"^\s+D2 - C: {_CI}\s+(\w+)", ln)
        if m_:
            out[reg]["diff"], out[reg]["label"] = _ci(*m_.group(1, 2, 3)), m_.group(4)
        m_ = re.match(r"\s+(A|C|D2)\s+n_av\s+([\d.]+) \+/-\s+[\d.]+\s+n_win\s+([\d.]+) \+/-\s+[\d.]+\s+never-avoided ([\d.]+)", ln)
        if m_:
            out[reg][m_.group(1)] = m_.group(2, 3, 4)
    return out


def _s13_closed(path: Path) -> dict:
    """{rho: "m [lo, hi]"} of D minus C closed-loop post-onset success under staggered onset (descriptive, no label)."""
    out, rho = {}, None
    for ln in _read_report(path):
        h = re.match(r"rho (\d)\s+attempted", ln)
        if h:
            rho = h.group(1)
        m_ = re.search(rf"^\s+D - C: {_CI}", ln)
        if m_ and rho is not None:
            out[rho] = _ci(*m_.group(1, 2, 3))
    return out


def _labelled(path: Path, pattern: str) -> dict:
    """{rho: ("m [lo, hi]", label)} for one labelled line per rho block of a stored S15/S16 eval report (the label may be several words)."""
    out, rho = {}, None
    for ln in _read_report(path):
        h = re.match(r"rho (\d)\s+\((CONTROL|TEST)\)", ln)
        if h:
            rho = h.group(1)
        m_ = re.search(rf"{pattern}: {_CI}\s+([A-Z][A-Z ]*[A-Z])\s*$", ln)
        if m_ and rho is not None:
            out[rho] = (_ci(*m_.group(1, 2, 3)), m_.group(4))
    return out


def _s16_overall(path: Path) -> dict:
    """{rho: overall D76 label} from the verdict block of the stored S16 eval report."""
    out = {}
    for ln in _read_report(path):
        m_ = re.match(r"\s+rho (\d) \((?:test|control)\): .* -> ([A-Z ]+?)\s*$", ln)
        if m_:
            out[m_.group(1)] = m_.group(2)
    return out


def _closed_diff(path: Path, who: str) -> str:
    """"m [lo, hi]" of one paired closed-loop success difference (descriptive) from a stored S15/S16 closed-loop report."""
    for ln in _read_report(path):
        m_ = re.search(rf"^\s+{re.escape(who)}: {_CI}", ln)
        if m_:
            return _ci(*m_.group(1, 2, 3))
    raise SystemExit(f"{who} not found in {path}")


def paper_tables(T, GT, HH) -> None:
    S = T[SHARED]
    # (1) ablation at the shared point, H1 triplet always together (R2)
    rows = []
    for v, src in (("A", "A"), ("B", "B"), ("B_oracle", "B_oracle"), ("C", "C"), ("D", "D"), ("E", "D")):
        s = S[src]
        rows.append([{"B_oracle": "B-oracle (upper bound)", "E": "E (= D)", "B": "B (= A here)"}.get(v, v), pm(s["success_post_onset"]), pm(s["n_av"], 1),
                     pm(s["n_win"], 1), pm(s["never_avoided"], 2), "never" if s["detected_fraction"][0] == 0 else pm(s["latency"], 0),
                     pm(s["detected_fraction"], 2)])
    g = GT["T"][SHARED]
    rows.append(["Gate-C (extra baseline)", pm(g["success_post_onset"]), pm(g["n_av"], 1), pm(g["n_win"], 1), pm(g["never_avoided"], 2), "not reported", "not reported"])
    h = HH["T"][SHARED]
    rows.append(["M (extra baseline)", pm(h["success_post_onset"]), pm(h["n_av"], 1), pm(h["n_win"], 1), pm(h["never_avoided"], 2), pm(h["latency"], 0),
                 pm(h["detected_fraction"], 2)])
    write_table("paper_ablation_shared_point",
                ["Variant", "Success after onset", "n_av", "n_win", "Never avoided", "Detection latency (steps)", "Degraded RSUs detected"], rows,
                "Closed-loop dispatch at the shared point (rho 1.0, 20% degraded), in this simulation: mean ± sd across 5 seeds",
                "n_av: dispatches to degraded RSUs from onset to the run's own avoidance step; n_win: the same over the fixed window [400, 1200); never avoided: fraction of runs "
                "that never reach the avoidance rule (the three are always read together). Detection latency is descriptive and censored at 399 steps; A and B cannot detect by construction. " + LATENCY_CAVEAT_SHORT,
                wide=True)

    # (2) effects against the references at the four fraction-sweep points
    verdict = json.loads((S9 / "summary-verdict.json").read_text(encoding="utf-8"))["verdicts"]
    fr = (0.05, 0.10, 0.20, 0.30)
    eff = lambda e: f"{e['mean']:+.4f}\n[{e['lo']:+.4f}, {e['hi']:+.4f}] {e['label']}"  # noqa: E731
    rows = [[f"{x} minus {ref}".replace("B_oracle", "B-oracle")] + [eff(verdict[f"1.00/{f:.2f}"][f"{x}_vs_{ref}"]["H3"]) for f in fr]
            for x, ref in (("C", "A"), ("D", "A"), ("C", "B_oracle"), ("D", "B_oracle"), ("D", "C"))]
    rows.append(["C minus Gate-C (extra baseline)"] + [f"{e.mean:+.4f}\n[{e.lo:+.4f}, {e.hi:+.4f}] {e.label}" for e in (GT["paired"][(1.0, round(f, 2))] for f in fr)])
    rows.append(["C minus M (extra baseline)"] + [f"{m_:+.4f}\n[{lo:+.4f}, {hi:+.4f}] {lab}" for m_, lo, hi, lab in (HH["paired"][(1.0, round(f, 2))] for f in fr)])
    rows.append(["M minus A (descriptive)"] + [f"{e.mean:+.4f}\n[{e.lo:+.4f}, {e.hi:+.4f}]" for e in (HH["vs_a"][(1.0, round(f, 2))] for f in fr)])
    write_table("paper_fraction_effects", ["Paired difference in post-onset success"] + [f"{int(f * 100)}% degraded" for f in fr], rows,
                "Paired differences in post-onset deadline success at rho 1.0, in this simulation: mean [95% t-interval over 5 seeds] and the pre-registered label",
                "Labels from the pre-registered rules (Table III); floor 0.01. D minus C is the closed-loop secondary check of H2 and is never gating. "
                "M minus A is descriptive (no pre-registered label).", wide=True)

    # (2b) the rho sweep at 20% degraded: success only (Gate-C and H were not run at these points, D59 and D65)
    rows = []
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        s_ = T[(round(rho, 2), 0.20)]
        rows.append([f"{rho:g}"] + [pm(s_[v]["success_post_onset"]) for v in ("A", "B_oracle", "C", "D")])
    write_table("paper_rho_sweep", ["rho", "A", "B-oracle (upper bound)", "C", "D"], rows,
                "Post-onset deadline success along the correlation sweep (20% degraded), in this simulation: mean ± sd across 5 seeds",
                "Gate-C and M were run at the four fraction-sweep points only and are not part of this sweep. B = A and E = D.")

    # (3) every graph-versus-node-local test, in the order it was run
    S4j = json.loads((S4 / "summary-5seeds.json").read_text(encoding="utf-8"))["sweep"]
    d43 = lambda r: f"{S4j[r]['delta'][0]:+.4f} [{S4j[r]['delta'][2]:+.4f}, {S4j[r]['delta'][3]:+.4f}]"  # noqa: E731
    sh = verdict["1.00/0.20"]["D_vs_C"]["H3"]
    sh0 = verdict["0.00/0.20"]["D_vs_C"]["H3"]
    L13 = _latency_blocks(S13 / "stag" / "report-eval-verdict.txt", "D")
    L14 = _latency_blocks(S14 / "report-eval-verdict-5seeds.txt", "D2")
    A14 = _shared_auc(S14 / "report-eval-verdict-5seeds.txt")
    C14 = _s14_closed(S14 / "report-closed-verdict.txt")
    C13 = _s13_closed(S13 / "stag" / "report-closed.txt")
    L15 = _labelled(S15 / "report-eval-verdict-5seeds.txt", r"PRIMARY\s+D2-h - C-h latency")
    C15 = _closed_diff(S15 / "report-closed-raw.txt", "D2_h - C_h")
    rows = [
        ["Shared onset", "shared", "D minus C", "diagnostic AUC [450, 650)", "1.0", d43("1.00"), "NEGATIVE (primary)"],
        ["Shared onset", "shared", "D minus C", "diagnostic AUC [450, 650)", "0", d43("0.00"), "no separate label (judged at rho 1.0)"],
        ["Closed-loop sweep", "shared", "D minus C", "closed-loop post-onset success", "1.0", f"{sh['mean']:+.4f} [{sh['lo']:+.4f}, {sh['hi']:+.4f}]", f"{sh['label']} (secondary)"],
        ["Closed-loop sweep", "shared", "D minus C", "closed-loop post-onset success", "0", f"{sh0['mean']:+.4f} [{sh0['lo']:+.4f}, {sh0['hi']:+.4f}]", f"{sh0['label']} (secondary)"],
        ["Staggered onset", "staggered", "D minus C", "latency, late-onset RSUs (steps)", "1.0", L13["1"]["diff"], f"{L13['1']['label']} (primary)"],
        ["Staggered onset", "staggered", "D minus C", "latency, late-onset RSUs (steps)", "0", L13["0"]["diff"], f"{L13['0']['label']} (control)"],
        ["Staggered onset", "staggered", "D minus C", "closed-loop post-onset success", "1.0", C13["1"], "descriptive"],
        ["Staggered onset", "staggered", "D minus C", "closed-loop post-onset success", "0", C13["0"], "descriptive"],
        ["Contained head", "staggered", "D2 minus C", "latency, late-onset RSUs (steps)", "1.0", L14["1"]["diff"], f"{L14['1']['label']} (primary)"],
        ["Contained head", "staggered", "D2 minus C", "latency, late-onset RSUs (steps)", "0", L14["0"]["diff"], f"{L14['0']['label']} (control)"],
        ["Contained head", "shared", "D2 minus C", "diagnostic AUC [450, 650)", "1.0", A14["1"][0], f"{A14['1'][1]} (secondary)"],
        ["Contained head", "shared", "D2 minus C", "diagnostic AUC [450, 650)", "0", A14["0"][0], f"{A14['0'][1]} (secondary)"],
        ["Contained head", "shared", "D2 minus C", "closed-loop post-onset success", "1.0", C14["shared"]["diff"], f"{C14['shared']['label']} (secondary)"],
        ["Contained head", "staggered", "D2 minus C", "closed-loop post-onset success", "1.0", C14["stag"]["diff"], f"{C14['stag']['label']} (secondary)"],
        ["Horizon objective", "staggered", "D2-h minus C-h", "latency, late-onset RSUs (steps)", "1.0", L15["1"][0], f"{L15['1'][1]} (primary)"],
        ["Horizon objective", "staggered", "D2-h minus C-h", "latency, late-onset RSUs (steps)", "0", L15["0"][0], f"{L15['0'][1]} (control)"],
        ["Horizon objective", "staggered", "D2-h minus C-h", "closed-loop post-onset success", "1.0", C15, "descriptive"],
    ]
    write_table("paper_graph_tests", ["Experiment", "Onset", "Comparison", "Quantity", "rho", "Difference [95% CI]", "Pre-registered label"], rows,
                "Every test of a graph head against the node-local head C, in this simulation (5 seeds; 95% t-interval over seeds)",
                "Negative latency means the first-named model detects earlier. Every rule was fixed before its numbers (Table III). "
                "Heads in the staggered-onset rows were trained on the staggered regime, the others on the shared-onset regime. C-h and D2-h were trained on the "
                "forward-looking (horizon) objective, every other head on the next-task objective. " + LATENCY_CAVEAT_SHORT, wide=True)

    # (3b) the training objective (A29, A30): the S15 primary and the S16 confirmatory test, side by side
    E16 = S16 / "report-eval-verdict-5seeds.txt"
    L16 = _labelled(E16, r"CO-PRIMARY 1\s+C-h - C latency")
    F16 = _labelled(E16, r"CO-PRIMARY 2\s+C-h - C false-alarm rate")
    O16 = _s16_overall(E16)
    C16 = _closed_diff(S16 / "report-closed-raw.txt", "C_h - C")
    rows = [
        ["Horizon objective", "D2-h minus C-h", "latency, late-onset RSUs (steps)", "1.0", L15["1"][0], f"{L15['1'][1]} (primary)"],
        ["Horizon objective", "D2-h minus C-h", "latency, late-onset RSUs (steps)", "0", L15["0"][0], f"{L15['0'][1]} (control)"],
        ["Horizon objective", "D2-h minus C-h", "closed-loop post-onset success", "1.0", C15, "descriptive"],
        ["Confirmatory test", "C-h minus C", "latency, all degraded RSUs (steps)", "1.0", L16["1"][0], f"{L16['1'][1]} (co-primary 1)"],
        ["Confirmatory test", "C-h minus C", "false-alarm rate, healthy RSUs", "1.0", F16["1"][0], f"{F16['1'][1]} (co-primary 2)"],
        ["Confirmatory test", "C-h minus C", "overall verdict", "1.0", "", f"{O16['1']}"],
        ["Confirmatory test", "C-h minus C", "latency, all degraded RSUs (steps)", "0", L16["0"][0], f"{L16['0'][1]} (control)"],
        ["Confirmatory test", "C-h minus C", "false-alarm rate, healthy RSUs", "0", F16["0"][0], f"{F16['0'][1]} (control)"],
        ["Confirmatory test", "C-h minus C", "overall verdict", "0", "", f"{O16['0']} (control)"],
        ["Confirmatory test", "C-h minus C", "closed-loop post-onset success", "1.0", C16, "descriptive"],
    ]
    write_table("paper_objective", ["Experiment", "Comparison", "Quantity", "rho", "Difference [95% CI]", "Pre-registered label"], rows,
                "The training objective, in this simulation: staggered onset (S = 300), 20% degraded, 5 seeds, 95% t-interval over seeds",
                "C-h and D2-h are C's and D2's architectures trained on the forward-looking objective (label 1 if fewer than 0.75 of an RSU's tasks in the next 150 steps meet "
                "their deadline; windows with fewer than 5 tasks excluded); C is the staggered-regime head trained on the next-task objective. Horizon objective: draws 300-349, "
                "late-onset RSUs. Confirmatory test: fresh draws 350-399, all degraded RSUs; its verdict is OBJECTIVE HELPS only if the latency label is EARLIER and the "
                "false-alarm label ACCEPTABLE (upper end of the interval at most 0.05). A negative difference means the first-named model detects earlier or raises fewer false alarms. "
                + LATENCY_CAVEAT_SHORT, wide=True)

    # (4) H1 triplet for D2 in both regimes (A28)
    rows = []
    for reg, name in (("shared", "shared onset"), ("stag", "staggered onset")):
        for v in ("A", "C", "D2"):
            n_av, n_win, nv = C14[reg][v]
            rows.append([name, v, n_av, n_win, nv])
    write_table("paper_d2_dispatch", ["Regime", "Variant", "n_av", "n_win", "Never avoided"], rows,
                "Dispatches to degraded RSUs at rho 1.0, 20% degraded, draws 250-299, in this simulation (mean over 5 seeds of per-seed means; n_av, n_win and the never-avoided rate together)",
                "D2 is the contained graph head; D was not run on draws 250-299. A is identical in both regimes.")

    # (5) per-decision time by component (A14), medians only
    o = json.loads((S9 / "overhead.json").read_text(encoding="utf-8"))["timing"]
    comp = [("trust_inference", "Trust inference"), ("selection", "Selection (rule)"), ("layer1", "Layer 1"), ("layer2_per_decision", "Layer 2")]
    rows = []
    for c, lab in comp:
        f = (lambda v: f"{v:.0f}") if c == "layer2_per_decision" else (lambda v: f"{v:.2f}")
        rows.append([lab, f(o["threads1"][c]["median"]), f(o["threads10"][c]["median"])])
    write_table("paper_latency", ["Component", "Median, 1 thread (ms)", "Median, 10 threads (ms)"], rows,
                "Per-decision time by component, median over 200 decisions at the shared point, in this simulation",
                "13th Gen Intel i5-13450HX, CPU only, one decision at a time. Layer 2 is judged against a stated 100 ms assumption: asynchronous only.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recompute", action="store_true", help="rebuild the cached closed-loop inputs of Figures 2 and 6")
    a = ap.parse_args()
    ensure_inputs(a.recompute)
    T, P, GT, HH = load_table(), load_paired(), load_gate(), load_h()
    print("figures:")
    fig_architecture(); fig_timeline(); fig_success_vs_fraction(T, P, GT, HH); fig_ablation(T, P, GT, HH); fig_s4_gap(); fig_explanation()
    if (S18 / "summary-open-primary.json").exists() and (S18 / "summary-open-secondary.json").exists():
        fig_s18_gap()
    print("tables:")
    tables(T, P, GT, HH)
    paper_tables(T, GT, HH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

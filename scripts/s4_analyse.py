"""S4: aggregate the raw per-scenario files into the D43 tables, plots, and verdict.

    python scripts/s4_analyse.py --seeds 1 2 3 4 5            # RAW numbers + plots only
    python scripts/s4_analyse.py --seeds 1 2 3 4 5 --verdict  # also apply D43 section 4

The raw section never prints a verdict, so FINDINGS.md can be written before any
interpretation. Everything is defined by DECISIONS.md D43 (and D42 for cold start).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from trustgraph.diagnostic import auc  # noqa: E402

import s4_eval as ev  # noqa: E402

RAW = REPO_ROOT / "results" / "s4" / "raw"
T0 = ev.TRUST_STEPS[0]  # trust array row i = step T0 + i
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          8: 2.306, 9: 2.262}
RELIABLE, DEGRADED, COMPROMISED = 0, 1, 2
ONSET, SEARCH_END, RUN = 400, 799, 10


def w(lo, hi):
    return slice(lo - T0, hi - T0)


def tci(values):
    """(mean, std, lo, hi): Student-t 95% interval over per-seed values."""
    v = np.asarray(values, float)
    n = v.size
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    h = T_CRIT[n - 1] * sd / np.sqrt(n) if n > 1 else float("nan")
    return m, sd, m - h, m + h


def fmt(ci, d=4):
    m, sd, lo, hi = ci
    return f"{m:+.{d}f} sd {sd:.{d}f} CI [{lo:+.{d}f}, {hi:+.{d}f}]"


def fmt_u(ci, d=4):
    m, sd, lo, hi = ci
    return f"{m:.{d}f} sd {sd:.{d}f} CI [{lo:.{d}f}, {hi:.{d}f}]"


def load(seeds):
    out = {}
    infeasible = {}
    for task in ev.all_tasks(seeds):
        key = ev.task_key(*task)
        if (RAW / f"{key}.infeasible").exists():
            infeasible[task] = True
            continue
        p = RAW / f"{key}.npz"
        if not p.exists():
            raise SystemExit(f"missing raw file {p.name}; run s4_eval.py first")
        z = np.load(p)
        out[task] = {k: z[k] for k in z.files}
    return out, infeasible


def class_auc(trust, rec, window, positive):
    """AUC of 1 - window-mean trust; `positive` vs RELIABLE; the other fault class and
    cold-start RSUs are excluded (D43 section 3)."""
    cls = rec["behavior_class"]
    other = COMPROMISED if positive == DEGRADED else DEGRADED
    keep = (cls != other) & (rec["join_step"] == 0)
    score = 1.0 - trust[window].mean(axis=0)
    return auc(score[keep], (cls[keep] == positive).astype(int))


def detection(trust, cls, tau):
    """Mean latency (steps) over the draw's degraded RSUs, and the detected fraction."""
    lat, det = [], []
    for v in np.flatnonzero(cls == DEGRADED):
        below = (trust[:, v] < tau).astype(int)
        cs = np.concatenate([[0], np.cumsum(below)])
        found = None
        for t in range(ONSET, SEARCH_END + 1):
            i = t - T0
            if cs[i + RUN] - cs[i] == RUN:
                found = t
                break
        det.append(found is not None)
        lat.append((found if found is not None else SEARCH_END) - ONSET)
    return float(np.mean(lat)), float(np.mean(det))


def per_seed(values_by_seed):
    return [float(np.mean(v)) for v in values_by_seed]


def run(args):
    seeds = args.seeds
    recs, infeasible = load(seeds)
    S = {}  # summary for json
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    # ------------------------------------------------------------- correlation sweep
    p(f"seeds {seeds}   draws {ev.DRAWS[0]}..{ev.DRAWS[-1]} per seed per cell   fraction {ev.FRACTION}")
    p("")
    p("=== A. CORRELATION SWEEP (fraction 0.20, K=4): diagnostic AUC of 1-trust vs sealed class")
    p("    primary window [450,650); per-seed value = mean over 50 draws; CI = Student-t over seeds")
    sweep = {}  # rho -> seed -> arrays
    for rho in ev.RHOS_SWEEP:
        sweep[rho] = {}
        for s in seeds:
            rs = [recs[(s, "sweep", rho, "none", d)] for d in ev.DRAWS]
            sweep[rho][s] = {
                "aucC": np.array([class_auc(r["trust_C"], r, w(450, 650), DEGRADED) for r in rs]),
                "aucD": np.array([class_auc(r["trust_D"], r, w(450, 650), DEGRADED) for r in rs]),
                "pC": np.array([_plateau_auc(r, "C") for r in rs]),
                "pD": np.array([_plateau_auc(r, "D") for r in rs]),
                "c": np.array([float(r["pair_concentration"]) for r in rs]),
                "recs": rs,
            }
    S["sweep"] = {}
    delta_ci = {}
    for key, tag in (("aucC", "AUC_C"), ("aucD", "AUC_D")):
        pass
    for rho in ev.RHOS_SWEEP:
        d = sweep[rho]
        aC = [d[s]["aucC"].mean() for s in seeds]
        aD = [d[s]["aucD"].mean() for s in seeds]
        dl = [(d[s]["aucD"] - d[s]["aucC"]).mean() for s in seeds]
        pC = [d[s]["pC"].mean() for s in seeds]
        pD = [d[s]["pD"].mean() for s in seeds]
        pdl = [(d[s]["pD"] - d[s]["pC"]).mean() for s in seeds]
        cmean = [d[s]["c"].mean() for s in seeds]
        ciC, ciD, ciL = tci(aC), tci(aD), tci(dl)
        overlap = not (ciC[3] < ciD[2] or ciD[3] < ciC[2])
        delta_ci[rho] = ciL
        p(f"  rho {rho:.2f}  realised c (mean over seeds) {np.mean(cmean):.3f}")
        p(f"     AUC_C  {fmt_u(ciC)}")
        p(f"     AUC_D  {fmt_u(ciD)}")
        p(f"     D-C    {fmt(ciL)}   separate 95% CIs overlap: {overlap}")
        p(f"     per-seed AUC_C {np.round(aC, 4).tolist()}")
        p(f"     per-seed AUC_D {np.round(aD, 4).tolist()}")
        p(f"     per-seed D-C   {np.round(dl, 4).tolist()}")
        p(f"     plateau [1100,1200) (secondary): C {np.mean(pC):.4f}  D {np.mean(pD):.4f}  D-C {fmt(tci(pdl))}")
        S["sweep"][f"{rho:.2f}"] = {"aucC": ciC, "aucD": ciD, "delta": ciL, "per_seed_delta": dl,
                                    "plateau_delta": tci(pdl), "mean_c": float(np.mean(cmean))}

    # W3: per-seed slope of per-draw Delta on realised c, all five rho pooled
    slopes = []
    for s in seeds:
        c = np.concatenate([sweep[r][s]["c"] for r in ev.RHOS_SWEEP])
        dd = np.concatenate([sweep[r][s]["aucD"] - sweep[r][s]["aucC"] for r in ev.RHOS_SWEEP])
        slopes.append(float(np.polyfit(c, dd, 1)[0]))
    slope_ci = tci(slopes)
    p("")
    p(f"  W3 slope of per-draw (AUC_D - AUC_C) on realised c, 250 draws/seed: per-seed {np.round(slopes, 4).tolist()}")
    p(f"     mean slope {fmt(slope_ci)}")
    S["slope"] = {"per_seed": slopes, "ci": slope_ci}

    # by realised c
    p("")
    p("  D-C by realised pair concentration c (all rho pooled; per-seed means, then across seeds)")
    p("     c      draws(total)  seeds   mean D-C     CI(t over seeds)")
    by_c = {}
    for cval in (0.0, 1 / 6, 1 / 3, 1 / 2, 1.0):
        per = []
        n_draw = 0
        for s in seeds:
            c = np.concatenate([sweep[r][s]["c"] for r in ev.RHOS_SWEEP])
            dd = np.concatenate([sweep[r][s]["aucD"] - sweep[r][s]["aucC"] for r in ev.RHOS_SWEEP])
            sel = np.isclose(c, cval)
            n_draw += int(sel.sum())
            if sel.any():
                per.append(float(dd[sel].mean()))
        if len(per) >= 2:
            ci = tci(per)
            p(f"     {cval:.3f}  {n_draw:6d}        {len(per)}      {ci[0]:+.4f}     [{ci[2]:+.4f}, {ci[3]:+.4f}]")
        else:
            ci = (per[0] if per else float("nan"),) + (float("nan"),) * 3
            p(f"     {cval:.3f}  {n_draw:6d}        {len(per)}      {ci[0]:+.4f}     (fewer than 2 seeds)")
        by_c[cval] = {"n_draws": n_draw, "n_seeds": len(per), "ci": ci}
    S["by_c"] = {f"{k:.4f}": v for k, v in by_c.items()}

    # ------------------------------------------------------------- detection latency
    p("")
    p("=== B. DETECTION LATENCY (secondary; sweep scenarios). tau = 1st percentile of the model's trust over")
    p("    all RSU-steps in [100,400) pooled over the seed's 250 sweep draws; detected = first t in [400,799]")
    p("    from which trust < tau for 10 consecutive steps; undetected censored at 799 (latency 399).")
    taus = {}
    for s in seeds:
        for v in ("C", "D"):
            pool = np.concatenate([r[f"trust_{v}"][w(100, 400)].ravel()
                                   for rho in ev.RHOS_SWEEP for r in sweep[rho][s]["recs"]])
            taus[(s, v)] = float(np.percentile(pool, 1))
    p("  tau per seed (C): " + str([round(taus[(s, 'C')], 4) for s in seeds]))
    p("  tau per seed (D): " + str([round(taus[(s, 'D')], 4) for s in seeds]))
    S["latency"] = {}
    for rho in ev.RHOS_SWEEP:
        res = {}
        for v in ("C", "D"):
            lat_s, det_s = [], []
            for s in seeds:
                ld = [detection(r[f"trust_{v}"], r["behavior_class"], taus[(s, v)]) for r in sweep[rho][s]["recs"]]
                lat_s.append(np.mean([x[0] for x in ld]))
                det_s.append(np.mean([x[1] for x in ld]))
            res[v] = (lat_s, det_s)
        dlat = [res["D"][0][i] - res["C"][0][i] for i in range(len(seeds))]
        ciC, ciD, ciL = tci(res["C"][0]), tci(res["D"][0]), tci(dlat)
        p(f"  rho {rho:.2f}  latency(steps) C {fmt_u(ciC, 1)} | detected {np.mean(res['C'][1]):.3f}")
        p(f"            latency(steps) D {fmt_u(ciD, 1)} | detected {np.mean(res['D'][1]):.3f}")
        p(f"            D-C latency {fmt(ciL, 1)}   separate CIs overlap: {not (ciC[3] < ciD[2] or ciD[3] < ciC[2])}")
        S["latency"][f"{rho:.2f}"] = {"C": ciC, "D": ciD, "delta": ciL,
                                      "detC": float(np.mean(res['C'][1])), "detD": float(np.mean(res['D'][1]))}

    # ------------------------------------------------------------------- collusion
    p("")
    p("=== C. COLLUSION (1 group of 3, under_report 0.5; fraction 0.20): AUC of 1-trust, compromised vs reliable,")
    p("    degraded RSUs excluded, window [200,400)")
    S["collusion"] = {}
    coll_ci = {}
    for rho in ev.RHOS_BINARY:
        aC, aD, dl = [], [], []
        for s in seeds:
            rs = [recs[(s, "collusion", rho, "none", d)] for d in ev.DRAWS]
            ac = np.array([class_auc(r["trust_C"], r, w(200, 400), COMPROMISED) for r in rs])
            ad = np.array([class_auc(r["trust_D"], r, w(200, 400), COMPROMISED) for r in rs])
            aC.append(ac.mean()); aD.append(ad.mean()); dl.append((ad - ac).mean())
        ciC, ciD, ciL = tci(aC), tci(aD), tci(dl)
        coll_ci[rho] = ciL
        p(f"  rho {rho:.2f}  AUC_C {fmt_u(ciC)}")
        p(f"            AUC_D {fmt_u(ciD)}")
        p(f"            D-C   {fmt(ciL)}   separate CIs overlap: {not (ciC[3] < ciD[2] or ciD[3] < ciC[2])}")
        p(f"            per-seed AUC_C {np.round(aC, 4).tolist()}  AUC_D {np.round(aD, 4).tolist()}  D-C {np.round(dl, 4).tolist()}")
        S["collusion"][f"{rho:.2f}"] = {"aucC": ciC, "aucD": ciD, "delta": ciL, "per_seed_delta": dl}

    # ------------------------------------------------------------------ cold start
    p("")
    p("=== D. COLD START (1 RSU joins at step 600; fraction 0.20). D42 accounting beside every number.")
    p("    AUC of 1 - mean trust over [600,630) of the joining RSU; label test=1 / control=0; per seed, pooled")
    p("    over draws feasible in BOTH conditions. Secondary window [600,660).")
    S["cold"] = {}
    cold_ci = {}
    for rho in ev.RHOS_BINARY:
        att = len(ev.DRAWS) * len(seeds)
        inf = {pl: sum(1 for s in seeds for d in ev.DRAWS if (s, "cold", rho, pl, d) in infeasible)
               for pl in ev.PLACEMENTS}
        paired = {s: [d for d in ev.DRAWS
                      if all((s, "cold", rho, pl, d) not in infeasible for pl in ev.PLACEMENTS)]
                  for s in seeds}
        npair = sum(len(v) for v in paired.values())
        p(f"  rho {rho:.2f}: attempted {att} per condition | control infeasible {inf['healthy_segment']} "
          f"aggregated {att - inf['healthy_segment']} | test infeasible {inf['degraded_segment']} "
          f"aggregated {att - inf['degraded_segment']} | paired (both feasible) aggregated {npair}")
        p(f"     paired draws per seed: {[len(paired[s]) for s in seeds]}")
        row = {}
        for win, tag in ((w(600, 630), "30 steps"), (w(600, 660), "60 steps")):
            aC, aD, dl = [], [], []
            mt = {"C": {"control": [], "test": []}, "D": {"control": [], "test": []}}
            for s in seeds:
                sc = {"C": [], "D": []}
                lab = []
                for d in paired[s]:
                    for pl, y in (("healthy_segment", 0), ("degraded_segment", 1)):
                        r = recs[(s, "cold", rho, pl, d)]
                        cold = np.flatnonzero(r["join_step"] > 0)
                        assert cold.size == 1
                        for v in ("C", "D"):
                            m = float(r[f"trust_{v}"][win][:, cold[0]].mean())
                            sc[v].append(1.0 - m)
                            mt[v]["test" if y else "control"].append(m)
                        lab.append(y)
                lab = np.array(lab)
                ac = auc(np.array(sc["C"]), lab); ad = auc(np.array(sc["D"]), lab)
                aC.append(ac); aD.append(ad); dl.append(ad - ac)
            ciC, ciD, ciL = tci(aC), tci(aD), tci(dl)
            p(f"     window {tag}: AUC_C {fmt_u(ciC)}")
            p(f"                  AUC_D {fmt_u(ciD)}")
            p(f"                  D-C   {fmt(ciL)}   separate CIs overlap: {not (ciC[3] < ciD[2] or ciD[3] < ciC[2])}")
            p(f"                  per-seed AUC_C {np.round(aC, 4).tolist()} AUC_D {np.round(aD, 4).tolist()} D-C {np.round(dl, 4).tolist()}")
            p("                  mean trust of the joiner: " + "  ".join(
                f"{v} control {np.mean(mt[v]['control']):.4f} test {np.mean(mt[v]['test']):.4f}" for v in ("C", "D")))
            row[tag] = {"aucC": ciC, "aucD": ciD, "delta": ciL, "per_seed_delta": dl,
                        "mean_trust": {v: {k: float(np.mean(x)) for k, x in mt[v].items()} for v in mt}}
        cold_ci[rho] = row["30 steps"]["delta"]
        row.update({"attempted": att, "infeasible": inf, "paired": npair})
        S["cold"][f"{rho:.2f}"] = row

    # -------------------------------------------------------------------- verdicts
    S["verdict"] = None
    if args.verdict:
        p("")
        p("=== VERDICT against D43 section 4")
        V = {}
        m1, _, lo1, hi1 = delta_ci[1.0]
        m0, _, lo0, hi0 = delta_ci[0.0]
        W1 = lo1 > 0 and m1 >= 0.02
        W2 = (lo0 <= 0 <= hi0) or abs(m0) < 0.02
        W3 = slope_ci[2] > 0
        if hi1 < 0 and m1 <= -0.02:
            core = "NEGATIVE"
        elif W1 and W2 and W3:
            core = "WIN"
        elif W1 and not W2:
            core = "WIN-NONSTRUCTURAL"
        elif hi1 < 0.05:
            core = "NULL"
        else:
            core = "NOISY"
        p(f"  H2 core (rho=1.0): Delta {m1:+.4f} CI [{lo1:+.4f}, {hi1:+.4f}]  W1 {W1} | rho=0 Delta {m0:+.4f} "
          f"CI [{lo0:+.4f}, {hi0:+.4f}]  W2 {W2} | W3 mean slope {slope_ci[0]:+.4f} CI lower {slope_ci[2]:+.4f}  W3 {W3}")
        p(f"  -> {core}")
        V["core"] = core

        def w1_only(ci):
            m, _, lo, hi = ci
            if hi < 0 and m <= -0.02:
                return "NEGATIVE"
            if lo > 0 and m >= 0.02:
                return "WIN"
            if hi < 0.05:
                return "NULL"
            return "NOISY"

        for rho in ev.RHOS_BINARY:
            V[f"collusion_rho{rho:g}"] = w1_only(coll_ci[rho])
            p(f"  collusion rho={rho:g}: Delta {coll_ci[rho][0]:+.4f} CI [{coll_ci[rho][2]:+.4f}, {coll_ci[rho][3]:+.4f}] -> {V[f'collusion_rho{rho:g}']}")
        V["cold_rho1"] = w1_only(cold_ci[1.0])
        p(f"  cold start rho=1.0: Delta {cold_ci[1.0][0]:+.4f} CI [{cold_ci[1.0][2]:+.4f}, {cold_ci[1.0][3]:+.4f}] -> {V['cold_rho1']}")
        p(f"  (informational, not a verdict) cold start rho=0: Delta {cold_ci[0.0][0]:+.4f} CI [{cold_ci[0.0][2]:+.4f}, {cold_ci[0.0][3]:+.4f}]")
        full = core == "WIN" and V["collusion_rho1"] == "WIN" and V["collusion_rho0"] == "WIN" and V["cold_rho1"] == "WIN"
        p(f"  H2 holds in full (all three WIN): {full}")
        V["h2_full"] = full
        S["verdict"] = V

    out = REPO_ROOT / "results" / "s4"
    (out / f"summary-{len(seeds)}seeds.json").write_text(json.dumps(S, indent=1, default=float), encoding="utf-8")
    (out / f"report-{len(seeds)}seeds{'-verdict' if args.verdict else ''}.txt").write_text("\n".join(lines), encoding="utf-8")
    plots(S, sweep, seeds, recs, infeasible)


def _plateau_auc(r, v):
    cls = r["behavior_class"]
    keep = (cls != COMPROMISED) & (r["join_step"] == 0)
    return auc(1.0 - r[f"plateau_{v}"][keep], (cls[keep] == DEGRADED).astype(int))


def plots(S, sweep, seeds, recs, infeasible):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = REPO_ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    INK, MUTED, BLUE, ORANGE = "#1f2328", "#6b7280", "#2a6fb0", "#c2571a"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "font.size": 10})
    n = len(seeds)

    # 1. gap vs realised correlation (+ nominal rho)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    rhos = list(ev.RHOS_SWEEP)
    m = [S["sweep"][f"{r:.2f}"]["delta"] for r in rhos]
    ax[0].errorbar(rhos, [x[0] for x in m], yerr=[[x[0] - x[2] for x in m], [x[3] - x[0] for x in m]],
                   color=BLUE, marker="o", capsize=3, lw=2, label="mean over seeds, 95% t-CI")
    for s in seeds:
        ax[0].plot(rhos, [S["sweep"][f"{r:.2f}"]["per_seed_delta"][seeds.index(s)] for r in rhos],
                   color=MUTED, lw=0.8, alpha=0.6, marker=".", label="single seed" if s == seeds[0] else None)
    ax[0].axhline(0, color=MUTED, lw=1); ax[0].axhline(0.02, color=ORANGE, lw=1, ls="--", label="0.02 effect floor")
    ax[0].set_xlabel("nominal rho"); ax[0].set_ylabel("AUC_D - AUC_C  [450,650)"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Gap vs nominal rho")
    cs, ms, errs = [], [], []
    for k, v in S["by_c"].items():
        if v["n_seeds"] >= 2:
            cs.append(float(k)); ms.append(v["ci"][0]); errs.append([v["ci"][0] - v["ci"][2], v["ci"][3] - v["ci"][0]])
    errs = np.array(errs).T
    ax[1].errorbar(cs, ms, yerr=errs, color=BLUE, marker="o", capsize=3, lw=2)
    for c_, mm in zip(cs, ms):
        ax[1].annotate(f"n_seeds={S['by_c'][f'{c_:.4f}']['n_seeds']}", (c_, mm), textcoords="offset points", xytext=(4, 6), fontsize=7, color=MUTED)
    ax[1].axhline(0, color=MUTED, lw=1); ax[1].axhline(0.02, color=ORANGE, lw=1, ls="--")
    ax[1].set_xlabel("realised pair concentration c"); ax[1].set_title("Gap vs realised correlation (all draws pooled)")
    fig.tight_layout(); fig.savefig(fig_dir / "s4_gap_vs_correlation.png", dpi=150); plt.close(fig)

    # 2. cold start control vs test
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, rho in enumerate(ev.RHOS_BINARY):
        row = S["cold"][f"{rho:.2f}"]["30 steps"]
        xs = np.arange(2)
        for j, (v, col) in enumerate((("C", BLUE), ("D", ORANGE))):
            ci = row[f"auc{v}"]
            ax[i].errorbar(xs[j], ci[0], yerr=[[ci[0] - ci[2]], [ci[3] - ci[0]]], color=col, marker="o", capsize=4, lw=2)
        ax[i].set_xticks(xs); ax[i].set_xticklabels(["C (node-local)", "D (GNN)"])
        ax[i].axhline(0.5, color=MUTED, lw=1, ls=":")
        ax[i].set_ylabel("AUC test vs control (joiner, [600,630))")
        ax[i].set_title(f"Cold start, rho={rho:g}  (paired draws {S['cold'][f'{rho:.2f}']['paired']}, "
                        f"infeasible ctrl {S['cold'][f'{rho:.2f}']['infeasible']['healthy_segment']})", fontsize=9)
        ax[i].set_xlim(-0.5, 1.5)
    fig.tight_layout(); fig.savefig(fig_dir / "s4_cold_start.png", dpi=150); plt.close(fig)

    # 3. collusion
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, rho in enumerate(ev.RHOS_BINARY):
        row = S["collusion"][f"{rho:.2f}"]
        for j, (v, col) in enumerate((("C", BLUE), ("D", ORANGE))):
            ci = row[f"auc{v}"]
            ax.errorbar(i + (j - 0.5) * 0.15, ci[0], yerr=[[ci[0] - ci[2]], [ci[3] - ci[0]]], color=col,
                        marker="o", capsize=4, lw=2, label=("C (node-local)" if v == "C" else "D (GNN)") if i == 0 else None)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["rho=0", "rho=1"]); ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_ylabel("AUC compromised vs reliable, [200,400)"); ax.set_title("Collusion"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / "s4_collusion.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--verdict", action="store_true")
    run(ap.parse_args())

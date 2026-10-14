"""S9: aggregate the sweep's result files into the results table, the paired comparisons and the verdicts.

    python scripts/s9_analyse.py --sweep configs/sweep.yaml                 # RAW tables, no labels
    python scripts/s9_analyse.py --sweep configs/sweep.yaml --verdict       # also the D56 labels and verdicts

Reads only results/s9/raw/*.json (+ checks.json, overhead.json); it never opens the sealed ground truth.
Writes results/s9/results_table.csv (one row per sweep point x variant x metric, mean and sd across the 5
seeds, with attempted / failed / aggregated draws), results/s9/summary.json and the report text.
Seed means are the unit (D43/D56); mean +/- sd is across seeds; intervals are Student-t, df 4.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from trustgraph import sweep_metrics as M  # noqa: E402
from s9_sweep import load_sweep, out_dir, points, shared_point, task_path  # noqa: E402

TABLE_METRICS = [
    ("success_post_onset", "H3 success post-onset"),
    ("success_all", "success (all steps)"),
    ("n_av", "H1 dispatches to degraded, onset->avoidance"),
    ("n_win", "dispatches to degraded, [400,1200)"),
    ("never_avoided", "fraction of runs never avoiding"),
    ("latency", "H1 detection latency (steps, censored 399)"),
    ("detected_fraction", "fraction of degraded RSUs detected"),
]
PRIMARY = [("C", "A"), ("D", "A"), ("C", "B_oracle"), ("D", "B_oracle")]
SECONDARY = [("D", "C")]
EXTRA = [("B", "A")]


def load_cells(sw: dict) -> dict:
    """{(rho, fraction): dict of arrays} with shape (seeds, draws); NaN where a task has no result file."""
    seeds, draws, variants = sw["seeds"], sw["draws"], sw["variants"]
    cells = {}
    for pt in points(sw):
        shape = (len(seeds), len(draws))
        arr = {v: {m: np.full(shape, np.nan) for m, _ in TABLE_METRICS} for v in variants}
        meta = {"k": np.full(shape, np.nan), "conc": np.full(shape, np.nan), "ok": np.zeros(shape, bool),
                "failed": 0, "b_eq_a_bad": 0, "post_unequal": 0}
        for i, s in enumerate(seeds):
            for j, d in enumerate(draws):
                p = task_path(sw, pt[0], pt[1], s, d)
                if not p.exists():
                    if task_path(sw, pt[0], pt[1], s, d, failed=True).exists():
                        meta["failed"] += 1
                    continue
                r = json.loads(p.read_text(encoding="utf-8"))
                meta["ok"][i, j] = True
                meta["k"][i, j] = r["k_degraded"]
                meta["conc"][i, j] = r["pair_concentration"]
                meta["b_eq_a_bad"] += int(r["b_equals_a"] is False)
                meta["post_unequal"] += int(not r["post_dispatched_equal"])
                for v in variants:
                    for m, _ in TABLE_METRICS:
                        x = r["variants"][v][m]
                        arr[v][m][i, j] = float(x)
        cells[pt] = {"arr": arr, "meta": meta}
    return cells


def agg(a: np.ndarray) -> tuple[float, float]:
    sm = M.seed_means(a)
    return float(np.mean(sm)), float(np.std(sm, ddof=1))


def fmt(mean, sd, nd=4):
    return f"{mean:.{nd}f} +/- {sd:.{nd}f}"


def pname(sw, pt):
    tags = "+".join(points(sw)[pt])
    return f"rho={pt[0]:.2f} frac={pt[1]:.2f} [{tags}]"


def comparisons(arr, x, ref):
    """Effects of X relative to ref on one point: (H3 effect, H1 primary, H1 guard, latency diff tuple)."""
    h3 = M.h3_effect(arr[x]["success_post_onset"], arr[ref]["success_post_onset"])
    h1, guard = M.h1_effect(arr[x]["n_av"], arr[ref]["n_av"], arr[x]["n_win"], arr[ref]["n_win"])
    lat = M.t_interval(M.seed_means(arr[x]["latency"] - arr[ref]["latency"]))
    return h3, h1, guard, lat


def report(sw: dict, cells: dict, verdict: bool) -> tuple[str, dict]:
    L: list[str] = []
    out: dict = {"points": {}, "verdicts": {}}
    L.append("=" * 118)
    L.append("S9 RESULTS TABLE" + (" WITH D56 LABELS AND VERDICTS" if verdict else " (RAW: no labels, no verdicts)"))
    L.append(f"seeds {sw['seeds']}, draws {sw['draws'][0]}-{sw['draws'][-1]} ({len(sw['draws'])} per seed per point), "
             f"variants {sw['variants']}; mean +/- sd across the 5 seed means; closed loop, D52 weights, dwell filter + handoff on")
    L.append("=" * 118)

    # ---- D42 block
    L.append("\nD42 accounting: draws attempted / failed / aggregated per point (every variant shares the same paired draws)")
    L.append("  The sweep injects no cold-start RSU (cold_start.num_nodes = 0), so no cold-start aggregate exists; the counts below are "
             "the whole-run accounting.")
    L.append(f"  {'point':44s} {'attempted':>9s} {'failed':>7s} {'aggregated':>10s}   K degraded (mean)  realised pair concentration (mean over draws with K>=2)")
    for pt, c in cells.items():
        att = len(sw["seeds"]) * len(sw["draws"])
        agg_n = int(c["meta"]["ok"].sum())
        conc = c["meta"]["conc"]
        cm = float(np.nanmean(conc)) if np.isfinite(conc).any() else float("nan")
        L.append(f"  {pname(sw, pt):44s} {att:9d} {att - agg_n:7d} {agg_n:10d}   {np.nanmean(c['meta']['k']):5.2f}              {cm:.3f}")
        out["points"][f"{pt[0]:.2f}/{pt[1]:.2f}"] = {"attempted": att, "failed": att - agg_n, "aggregated": agg_n}
    bad_b = sum(c["meta"]["b_eq_a_bad"] for c in cells.values())
    bad_p = sum(c["meta"]["post_unequal"] for c in cells.values())
    L.append(f"  consistency: runs where B's task stream differs from A's: {bad_b}; runs where the post-onset dispatched count "
             f"differs across variants: {bad_p}")
    out["b_not_equal_a"], out["post_dispatched_unequal"] = bad_b, bad_p

    # ---- the table
    for sweep_name, key in (("rho sweep (fraction fixed)", "rho"), ("fraction sweep (rho fixed)", "fraction")):
        L.append("\n" + "-" * 118)
        L.append(f"{sweep_name}")
        for pt in [p for p, tags in points(sw).items() if key in tags]:
            c = cells[pt]
            L.append(f"\n  {pname(sw, pt)}" + ("   [K = 1: no correlated failure possible, F10]" if abs(pt[1] - 0.05) < 1e-9 else ""))
            hdr = f"    {'metric':46s}" + "".join(f"{v:>20s}" for v in sw["variants"])
            L.append(hdr)
            for m, desc in TABLE_METRICS:
                nd = 1 if m in ("n_av", "n_win", "latency") else 4
                cellsx = []
                for v in sw["variants"]:
                    mean, sd = agg(c["arr"][v][m])
                    cellsx.append(f"{fmt(mean, sd, nd):>20s}")
                L.append(f"    {desc:46s}" + "".join(cellsx))
                for v in sw["variants"]:
                    mean, sd = agg(c["arr"][v][m])
                    out["points"][f"{pt[0]:.2f}/{pt[1]:.2f}"].setdefault(v, {})[m] = {"mean": mean, "sd": sd}

    # ---- paired comparisons
    L.append("\n" + "=" * 118)
    L.append("PAIRED COMPARISONS (X - ref per draw, seed means, Student-t 95% over 5 seeds). H3 in absolute success rate;"
             " H1 as relative reduction in dispatches to degraded RSUs (1 - X/ref, positive = X sends fewer); latency diff in steps.")
    pairs = PRIMARY + SECONDARY + EXTRA
    csv_rows = []
    for pt, c in cells.items():
        L.append(f"\n  {pname(sw, pt)}")
        for x, ref in pairs:
            tag = "primary" if (x, ref) in PRIMARY else ("secondary (H2 check)" if (x, ref) in SECONDARY else "sanity")
            try:
                h3, h1, g, lat = comparisons(c["arr"], x, ref)
            except ValueError as e:
                L.append(f"    {x:>2s} vs {ref:8s} [{tag}] undefined: {e}")
                continue
            lab3 = f"  {h3.label}" if verdict else ""
            lab1 = f"  {h1.label}" if verdict else ""
            L.append(f"    {x:>2s} vs {ref:8s} [{tag}]")
            L.append(f"        H3  success diff   {h3.mean:+.4f} (sd {h3.sd:.4f}) 95% [{h3.lo:+.4f}, {h3.hi:+.4f}]{lab3}")
            L.append(f"        H1  n_av reduction {h1.mean:+.3f} (sd {h1.sd:.3f}) 95% [{h1.lo:+.3f}, {h1.hi:+.3f}]{lab1}")
            L.append(f"            n_win guard    {g.mean:+.3f} (sd {g.sd:.3f}) 95% [{g.lo:+.3f}, {g.hi:+.3f}]")
            L.append(f"        latency diff       {lat[0]:+.1f} (sd {lat[1]:.1f}) 95% [{lat[2]:+.1f}, {lat[3]:+.1f}]  (descriptive; never gates)")
            out["verdicts"].setdefault(f"{pt[0]:.2f}/{pt[1]:.2f}", {})[f"{x}_vs_{ref}"] = {
                "H3": {"mean": h3.mean, "sd": h3.sd, "lo": h3.lo, "hi": h3.hi, "label": h3.label},
                "H1": {"mean": h1.mean, "sd": h1.sd, "lo": h1.lo, "hi": h1.hi, "label": h1.label},
                "H1_guard": {"mean": g.mean, "lo": g.lo, "hi": g.hi},
                "latency_diff": {"mean": lat[0], "sd": lat[1], "lo": lat[2], "hi": lat[3]},
            }
            csv_rows.append((pt[0], pt[1], x, ref, tag, h3.mean, h3.sd, h3.lo, h3.hi, h1.mean, h1.sd, h1.lo, h1.hi))

    # ---- non-overlap (spec's literal H3 criterion), printed
    L.append("\nH3 'non-overlapping error bars' (PROJECT_SPEC 4.1), fraction sweep: separate 95% t intervals of the variants' seed means")
    for pt in [p for p, tags in points(sw).items() if "fraction" in tags]:
        iv = {}
        for v in ("A", "B_oracle", "C", "D"):
            sm = M.seed_means(cells[pt]["arr"][v]["success_post_onset"])
            m_, sd_, lo_, hi_ = M.t_interval(sm)
            iv[v] = (lo_, hi_)
        row = []
        for x in ("C", "D"):
            for ref in ("A", "B_oracle"):
                nov = iv[x][0] > iv[ref][1] or iv[x][1] < iv[ref][0]
                row.append(f"{x} vs {ref}: {'non-overlap' if nov else 'overlap'}")
        L.append(f"  {pname(sw, pt):44s} " + "; ".join(row))

    # ---- verdicts
    if verdict:
        L.append("\n" + "=" * 118)
        L.append("D56 VERDICTS")
        sp = shared_point(sw)
        fr_pts = [p for p, tags in points(sw).items() if "fraction" in tags]
        for x, ref in PRIMARY:
            wins = []
            for pt in fr_pts:
                h3 = comparisons(cells[pt]["arr"], x, ref)[0]
                wins.append((pt[1], h3.label))
            n_win = sum(1 for _, l in wins if l == "WIN")
            holds = "HOLDS" if n_win >= 2 else "DOES NOT HOLD"
            L.append(f"  H3 {x} vs {ref}: labels by fraction {wins} -> WIN at {n_win} of 4 points -> {holds}")
            out["verdicts"][f"H3_{x}_vs_{ref}"] = {"labels": wins, "wins": n_win, "holds": n_win >= 2}
        for x, ref in PRIMARY:
            h1 = comparisons(cells[sp]["arr"], x, ref)[1]
            L.append(f"  H1 {x} vs {ref} at the shared point (rho {sp[0]}, frac {sp[1]}): {h1.label}  "
                     f"(reduction {h1.mean:+.3f}, 95% [{h1.lo:+.3f}, {h1.hi:+.3f}])")
            out["verdicts"][f"H1_{x}_vs_{ref}"] = h1.label
        for x, ref in SECONDARY:
            labs = [(p[1], comparisons(cells[p]["arr"], x, ref)[0].label) for p in fr_pts]
            L.append(f"  secondary, closed-loop H2 check {x} vs {ref} (H3 metric) by fraction: {labs}  [never gating; F17 unchanged]")
        L.append("  PRE-REGISTERED LABEL KEY: NEGATIVE / WIN / NULL / NOISY exactly as D56 section 4; vs B_oracle, NEGATIVE reads 'the oracle "
                 "is at least the floor better' (headroom) and NULL reads 'within the bound of the oracle'.")
    # ---- csv
    d = out_dir(sw)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "results_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rho", "fraction", "variant", "metric", "mean_over_seeds", "sd_over_seeds", "n_seeds", "attempted", "failed", "aggregated"])
        for pt, cc in cells.items():
            att = len(sw["seeds"]) * len(sw["draws"])
            ok = int(cc["meta"]["ok"].sum())
            for v in sw["variants"]:
                for m, _ in TABLE_METRICS:
                    mean, sd = agg(cc["arr"][v][m])
                    w.writerow([pt[0], pt[1], v, m, f"{mean:.6f}", f"{sd:.6f}", len(sw["seeds"]), att, att - ok, ok])
    with (d / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rho", "fraction", "x", "ref", "kind", "h3_mean", "h3_sd", "h3_lo", "h3_hi", "h1_mean", "h1_sd", "h1_lo", "h1_hi"])
        w.writerows(csv_rows)
    return "\n".join(L), out


def overhead_report(sw: dict, verdict: bool) -> str:
    p = out_dir(sw) / "overhead.json"
    if not p.exists():
        return "\n(overhead stage not run: no overhead.json)"
    d = json.loads(p.read_text(encoding="utf-8"))
    L = ["\n" + "=" * 118, f"H4 FIDELITY REPLICATION and H5 TIMING (D56 section 7): Variant {d['variant']} (head D), shared point, draw {d['draw']}"]
    L.append(f"  hardware: {d['hardware']}")
    L.append(f"  decisions per seed {d['samples']}; graph rebuild max |trust diff| {d['checks']['max_trust_diff']:.2e}; "
             f"decisions whose re-run differs from the logged selection: {d['checks']['decision_mismatch']} of {d['checks']['decisions']}")
    for kind, floor in (("trust", 0.20), ("decision", 0.05)):
        h = d["h4"][kind]
        L.append(f"  H4 {kind}-change: per-seed (top, low, gap):")
        for s, (t, lo, g) in zip(sw["seeds"], h["per_seed"]):
            L.append(f"      seed {s}: top {t:.3f}  low {lo:.3f}  gap {g:+.3f}")
        tm, ts, _, _ = h["top"]; lm, ls, _, _ = h["low"]; gm, gs, glo, ghi = h["gap"]
        line = f"    mean top {tm:.3f} (sd {ts:.3f})  low {lm:.3f} (sd {ls:.3f})  gap {gm:+.3f} (sd {gs:.3f}) 95% [{glo:+.3f}, {ghi:+.3f}]  floor {floor}"
        if verdict:
            v = "FAITHFUL" if (glo > 0 and gm >= floor) else ("WEAK" if glo > 0 else "NOT SUPPORTED")
            line += f"  -> {v}"
        L.append(line)
    for key, r in d["timing"].items():
        L.append(f"  H5 per-decision times in ms, {key} ({r['trust_inference']['n']} decisions); median / mean / p95:")
        for comp in ("trust_inference", "selection", "layer1", "layer2_per_decision", "layer2_per_explanation"):
            x = r[comp]
            L.append(f"      {comp:24s} {x['median']:9.3f} / {x['mean']:9.3f} / {x['p95']:9.3f}")
        if verdict:
            med = r["layer2_per_decision"]["median"]
            L.append(f"      Layer 2 median {med:.0f} ms vs the 100 ms assumption -> {'synchronous-practical' if med <= d['budget_ms'] else 'asynchronous only'}")
    return "\n".join(L)


def checks_report(sw: dict) -> str:
    p = out_dir(sw) / "checks.json"
    if not p.exists():
        return "\n(checks not run: no checks.json)"
    d = json.loads(p.read_text(encoding="utf-8"))
    ok_cells = all(c["identical"] for c in d["cells"])
    ok_e = all(c["identical"] for c in d["e_equals_d"])
    return (f"\nshared-point agreement (D56 1b): config-level assert passed at load; shared cell (rho {d['shared_point'][0]}, frac "
            f"{d['shared_point'][1]}, draw {d['draw']}) recomputed bit-identical for all 5 seeds: {ok_cells}; "
            f"E's task stream equal to D's for all 5 seeds: {ok_e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args(argv)
    sw = load_sweep(a.sweep)
    cells = load_cells(sw)
    text, summary = report(sw, cells, a.verdict)
    text += checks_report(sw) + overhead_report(sw, a.verdict)
    d = out_dir(sw)
    name = "report-verdict.txt" if a.verdict else "report-raw.txt"
    (d / name).write_text(text + "\n", encoding="utf-8")
    (d / ("summary-verdict.json" if a.verdict else "summary-raw.json")).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

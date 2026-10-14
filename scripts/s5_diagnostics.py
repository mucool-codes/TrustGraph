"""S5 gate-response diagnostics (DECISIONS.md D48). Diagnostics only: nothing here changes a
model, a scenario, a window or a threshold of D43, and nothing re-opens F16/F17.

    python scripts/s5_diagnostics.py check1      # does D use same_segment? is it set correctly?
    python scripts/s5_diagnostics.py check2      # does neighbour evidence lead own-node evidence?
    python scripts/s5_diagnostics.py check3      # seed 1 at 3 torch threads vs F16 (after s4_eval --threads 3)

EVALUATION-SIDE (reads sealed classes next to model outputs); allowlisted in tests/test_sealing.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import s4_eval as ev  # noqa: E402
from s4_analyse import T_CRIT, class_auc, tci, w  # noqa: E402

CONFIG = "configs/demo.yaml"
DEGRADED = 1


def fmt(ci, d=3):
    m, sd, lo, hi = ci
    return f"{m:+.{d}f} sd {sd:.{d}f} CI [{lo:+.{d}f}, {hi:+.{d}f}]"


# ------------------------------------------------------------------------------ check 1
def check1(seeds):
    import torch
    from torch_geometric.data import Batch, Data

    from trustgraph.config import load_config, with_overrides
    from trustgraph.diagnostic import auc
    from trustgraph.features import EDGE_COL
    from trustgraph.model import build_trust_head
    from trustgraph.observed import load_observed, scenario_stem
    from trustgraph.scenario import build_snapshot_builder, build_world
    from trustgraph.sealed.ground_truth import load_ground_truth, sealed_path_for
    from trustgraph.train import load_checkpoint
    from trustgraph.trace import default_trace_path, load_trace

    torch.set_num_threads(2)
    base = load_config(CONFIG)
    col = EDGE_COL["same_segment"]
    print("CHECK 1  same_segment -> D")
    print("  structural: same_segment == (segment_i == segment_j) on RSU-RSU edges, 0 on vehicle-RSU edges")
    rows = []
    for seed in seeds:
        cfg = with_overrides(base, seed=seed, rho=1.0, fraction=ev.FRACTION, draw=ev.DRAWS[0])
        world = build_world(cfg)
        trace = load_trace(default_trace_path(CONFIG, seed))
        n = world.topology.num_rsus
        seg = world.topology.backhaul_segment_id
        d_model = build_trust_head(base.model, 0, base.device, "D")
        load_checkpoint(REPO_ROOT / "checkpoints" / f"s3-seed{seed}-D.pt", d_model)

        bad_rr = bad_vr = n_rr = n_vr = n_rr_same = 0
        d_zero, d_one, d_perm = [], [], []
        auc_base, auc_zero, auc_perm = [], [], []
        rng = np.random.default_rng(20261014 + seed)
        for draw in ev.DRAWS[:5]:
            cfg = with_overrides(base, seed=seed, rho=1.0, fraction=ev.FRACTION, draw=draw)
            stem = scenario_stem(CONFIG, cfg)
            obs_path = Path("scenarios") / "s4" / f"{stem}.observed.npz"
            observed = load_observed(obs_path)
            gt = load_ground_truth(sealed_path_for(obs_path))
            builder = build_snapshot_builder(cfg, world, trace, observed)
            graphs, steps = [], []
            for t in range(650):
                data = builder.build(t)
                if t >= 450 and t % 5 == 0:
                    graphs.append(data)
                    steps.append(t)
            for data in graphs:
                ei, ea = data.edge_index, data.edge_attr
                rr = (ei[0] < n) & (ei[1] < n)
                want = torch.from_numpy((seg[ei[0][rr].numpy()] == seg[ei[1][rr].numpy()]).astype(np.float32))
                bad_rr += int((ea[rr, col] != want).sum())
                bad_vr += int((ea[~rr, col] != 0).sum())
                n_rr += int(rr.sum()); n_vr += int((~rr).sum()); n_rr_same += int(want.sum())

            def run(mod):
                out = []
                for data in graphs:
                    ea = data.edge_attr.clone()
                    mod(data, ea)
                    with torch.no_grad():
                        tr = d_model(data.x, data.edge_index, ea)
                    out.append(tr[:n].numpy())
                return np.stack(out)

            def zero(data, ea):
                ea[:, col] = 0.0

            def one(data, ea):
                rr = (data.edge_index[0] < n) & (data.edge_index[1] < n)
                ea[rr, col] = 1.0

            def perm(data, ea):
                ei = data.edge_index.numpy()
                rr = np.flatnonzero((ei[0] < n) & (ei[1] < n))
                key = {}
                for e in rr:
                    key.setdefault((min(ei[0][e], ei[1][e]), max(ei[0][e], ei[1][e])), []).append(e)
                pairs = list(key.values())
                vals = np.array([ea[p[0], col].item() for p in pairs])
                vals = rng.permutation(vals)
                for p, v in zip(pairs, vals):
                    ea[p, col] = float(v)

            base_t = run(lambda d, e: None)
            z_t, o_t, p_t = run(zero), run(one), run(perm)
            d_zero.append(np.abs(z_t - base_t).mean()); d_one.append(np.abs(o_t - base_t).mean())
            d_perm.append(np.abs(p_t - base_t).mean())
            keep = (gt.behavior_class != 2) & (gt.join_step == 0)
            lab = (gt.behavior_class == DEGRADED).astype(int)
            for store, tr in ((auc_base, base_t), (auc_zero, z_t), (auc_perm, p_t)):
                store.append(auc(1.0 - tr.mean(axis=0)[keep], lab[keep]))
        rows.append((seed, n_rr, n_rr_same, bad_rr, n_vr, bad_vr, np.mean(d_zero), np.mean(d_one),
                     np.mean(d_perm), np.mean(auc_base), np.mean(auc_zero), np.mean(auc_perm)))
        print(f"  seed {seed}: RSU-RSU edge-instances {n_rr} (same-segment {n_rr_same}), mismatches {bad_rr}; "
              f"vehicle-RSU {n_vr}, nonzero {bad_vr}")
    print("")
    print("  behavioural (rho=1.0, draws 100-104, steps [450,650) stride 5): mean |d RSU trust| and window AUC")
    print("  seed  zero-all  ->1(RSU-RSU)  permuted   AUC base  AUC zeroed  AUC permuted")
    defect = False
    for r in rows:
        print(f"  {r[0]}     {r[6]:.4f}    {r[7]:.4f}        {r[8]:.4f}     {r[9]:.4f}    {r[10]:.4f}      {r[11]:.4f}")
        defect |= r[3] > 0 or r[5] > 0 or r[6] < 0.01 or r[8] < 0.01
    print(f"  D44 probe on validation scenarios (F15), restated: seed1 0.0571/0.0405 seed2 0.1330/0.0353 "
          f"seed3 0.0632/0.0358 seed4 0.1172/0.0474 seed5 0.0677/0.0573")
    print(f"  DEFECT (any mismatch, or mean |dtrust| < 0.01 for zero/permute on any seed): {defect}")


# ------------------------------------------------------------------------------ check 2
def moves_pctile(series, base_slice):
    """First t in [400,799] with series below its own pre-onset 1st percentile for 10 steps."""
    theta = np.percentile(series[base_slice], 1)
    return _first_run(series < theta)


def moves_drop(series, base_slice):
    return _first_run(series < series[base_slice].mean() - 0.10)


def _first_run(below, onset=400, end=799, run=10):
    cs = np.concatenate([[0], np.cumsum(below.astype(int))])
    for t in range(onset, end + 1):
        if cs[t + run] - cs[t] == run:
            return t
    return None


def check2(seeds):
    from trustgraph.config import load_config, with_overrides
    from trustgraph.observed import load_observed, scenario_stem
    from trustgraph.scenario import build_world
    from trustgraph.sealed.ground_truth import load_ground_truth, sealed_path_for

    base = load_config(CONFIG)
    print("CHECK 2  own-node vs same-segment-neighbour first movement of success_ewma (seconds; dt = 1 s)")
    out = {}
    for variant, mover in (("primary: own 1st-percentile, 10 consecutive", moves_pctile),
                           ("secondary: drop >= 0.10 below own pre-onset mean, 10 consecutive", moves_drop)):
        print(f"\n--- {variant}")
        print("  rho   n(v)   own t after onset  ramp level   own-lead(mean nbr)    own-lead(min nbr)     "
              "censored own  have deg nbr / only healthy nbr / none  healthy nbr moves")
        for rho in ev.RHOS_SWEEP:
            per_seed = {k: [] for k in ("own", "lead_mean", "lead_min", "cens", "hasdeg", "onlyh", "none", "hmove")}
            nv = 0
            for seed in seeds:
                cfg0 = with_overrides(base, seed=seed)
                topo = build_world(cfg0).topology
                seg = topo.backhaul_segment_id
                nbrs = [[] for _ in range(topo.num_rsus)]
                for i, j in topo.rsu_edges:
                    if seg[i] == seg[j]:
                        nbrs[int(i)].append(int(j)); nbrs[int(j)].append(int(i))
                own, lm, ln, cens, hd, oh, no, hm = [], [], [], [], 0, 0, 0, []
                for draw in ev.DRAWS:
                    cfg = with_overrides(base, seed=seed, rho=rho, fraction=ev.FRACTION, draw=draw)
                    p = Path("scenarios") / "s4" / f"{scenario_stem(CONFIG, cfg)}.observed.npz"
                    ewma = load_observed(p).feature("success_ewma")
                    gt = load_ground_truth(sealed_path_for(p))
                    deg = np.flatnonzero(gt.behavior_class == DEGRADED)
                    t_of = {}
                    for v in range(topo.num_rsus):
                        t_of[v] = mover(ewma[:, v], slice(100, 400))
                    for v in deg:
                        tv = t_of[v]
                        own.append((tv if tv is not None else 799) - 400)
                        cens.append(tv is None)
                        dn = [u for u in nbrs[v] if gt.behavior_class[u] == DEGRADED]
                        hn = [u for u in nbrs[v] if gt.behavior_class[u] != DEGRADED]
                        if dn:
                            hd += 1
                            tu = [(t_of[u] if t_of[u] is not None else 799) for u in dn]
                            tv_ = tv if tv is not None else 799
                            lm.append(np.mean(tu) - tv_); ln.append(np.min(tu) - tv_)
                        elif hn:
                            oh += 1
                        else:
                            no += 1
                        hm += [t_of[u] is not None for u in hn]
                nvd = len(own)
                per_seed["own"].append(np.mean(own)); per_seed["cens"].append(np.mean(cens))
                per_seed["lead_mean"].append(np.mean(lm) if lm else np.nan)
                per_seed["lead_min"].append(np.mean(ln) if ln else np.nan)
                per_seed["hasdeg"].append(hd / nvd); per_seed["onlyh"].append(oh / nvd); per_seed["none"].append(no / nvd)
                per_seed["hmove"].append(np.mean(hm) if hm else np.nan)
                nv += nvd
            def ci(k):
                v = np.array(per_seed[k], float)
                v = v[~np.isnan(v)]
                return tci(v) if v.size >= 2 else (float("nan"),) * 4
            o = ci("own")
            print(f"  {rho:.2f}  {nv:5d}  {o[0]:6.1f} (sd {o[1]:.1f})   {o[0] / 300:5.2f}      "
                  f"{fmt(ci('lead_mean'), 1)}   {fmt(ci('lead_min'), 1)}   {np.mean(per_seed['cens']):.3f}   "
                  f"{np.mean(per_seed['hasdeg']):.2f}/{np.mean(per_seed['onlyh']):.2f}/{np.mean(per_seed['none']):.2f}"
                  f"        {np.nanmean(per_seed['hmove']):.3f}")
            out[(variant[:7], rho)] = per_seed
    return out


# ------------------------------------------------------------------------------ check 3
def check3():
    base_dir = ev.OUT_DIR
    alt_dir = REPO_ROOT / "results" / "s5" / "raw_t3"
    keys = [ev.task_key(*t) for t in ev.all_tasks([1])]
    max_d = {"C": 0.0, "D": 0.0}
    n_cmp = n_bad = 0
    auc_diff = []
    lat_note = 0
    for key in keys:
        a, b = base_dir / f"{key}.npz", alt_dir / f"{key}.npz"
        if (base_dir / f"{key}.infeasible").exists():
            assert (alt_dir / f"{key}.infeasible").exists(), key
            continue
        za, zb = np.load(a), np.load(b)
        n_cmp += 1
        for v in ("C", "D"):
            d = float(np.abs(za[f"trust_{v}"] - zb[f"trust_{v}"]).max())
            max_d[v] = max(max_d[v], d)
        ra = {k: za[k] for k in za.files}; rb = {k: zb[k] for k in zb.files}
        if ra["kind"] == "sweep":
            for v in ("C", "D"):
                auc_diff.append(abs(class_auc(ra[f"trust_{v}"], ra, w(450, 650), DEGRADED)
                                    - class_auc(rb[f"trust_{v}"], rb, w(450, 650), DEGRADED)))
    print(f"CHECK 3  seed 1, torch threads 1 (F16) vs 3: {n_cmp} scenarios compared, "
          f"infeasible sets identical")
    print(f"  max |trust difference|: C {max_d['C']:.3e}  D {max_d['D']:.3e}")
    print(f"  max |AUC difference| over the 250 sweep draws, C and D: {max(auc_diff):.3e}")
    beyond = max(max_d.values()) > 1e-5 or max(auc_diff) > 5e-5
    print(f"  beyond rounding (max |dtrust| > 1e-5 or |dAUC| >= 5e-5): {beyond}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("check1", "check2", "check3"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    {"check1": lambda: check1(a.seeds), "check2": lambda: check2(a.seeds), "check3": check3}[a.which]()

"""S10: `scripts/make_figures.py` regenerates every figure and table, deterministically, from the S9/S4 result files."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NEEDED = [REPO / "results" / "s9" / "results_table.csv", REPO / "results" / "s9b" / "tuning.json", REPO / "results" / "s4" / "summary-5seeds.json",
          REPO / "results" / "s10" / "timeline.npz", REPO / "results" / "s10" / "example.json"]
FIGURES = ["fig1_architecture", "fig2_trust_timeline", "fig3_success_vs_fraction", "fig4_ablation",
           "fig5_s4_gap_vs_correlation", "fig6_explanation"]
TABLES = ["ablation_shared_point", "ablation_all_points", "paired_effects_shared_point", "latency_breakdown", "related_work",
          "paper_ablation_shared_point", "paper_fraction_effects", "paper_rho_sweep", "paper_graph_tests", "paper_d2_dispatch", "paper_latency",
          "paper_objective"]


def _hashes():
    files = [REPO / "figures" / "s10" / f"{n}.{e}" for n in FIGURES for e in ("svg", "pdf")]
    files += [REPO / "results" / "s10" / "tables" / f"{n}.{e}" for n in TABLES for e in ("md", "csv", "tex")]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="needs the S9 sweep results and cached S10 inputs (generated artefacts)")
def test_make_figures_writes_everything_and_is_byte_identical_on_rerun():
    run = lambda: subprocess.run([sys.executable, str(REPO / "scripts" / "make_figures.py")], cwd=REPO, capture_output=True, text=True)  # noqa: E731
    r1 = run()
    assert r1.returncode == 0, r1.stderr[-2000:]
    h1 = _hashes()
    assert len(h1) == len(FIGURES) * 2 + len(TABLES) * 3
    r2 = run()
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert _hashes() == h1

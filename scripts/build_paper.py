"""Build the paper PDF:  python scripts/build_paper.py

Fetches the pinned, checksum-verified Tectonic into tools/ if missing (scripts/get_tectonic.py), regenerates figures and tables
if they are missing (scripts/make_figures.py), compiles paper/main.tex, and prints the page count.
Everything stays inside the repository (tools/ is gitignored); nothing is installed system-wide.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"
TECTONIC = REPO / "tools" / ("tectonic.exe" if sys.platform == "win32" else "tectonic")
NEED = [REPO / "figures" / "s10" / f"{n}.pdf" for n in ("fig1_architecture", "fig2_trust_timeline", "fig3_success_vs_fraction",
                                                         "fig4_ablation", "fig5_s4_gap_vs_correlation", "fig6_explanation")]
NEED += [REPO / "results" / "s10" / "tables" / f"{n}.tex" for n in ("paper_ablation_shared_point", "paper_fraction_effects", "paper_rho_sweep",
                                                                    "paper_graph_tests", "paper_d2_dispatch", "paper_latency",
                                                                    "paper_objective")]


def main() -> int:
    if not TECTONIC.exists():
        if subprocess.run([sys.executable, str(REPO / "scripts" / "get_tectonic.py")]).returncode:
            return 1
    if not all(p.exists() for p in NEED):
        if subprocess.run([sys.executable, str(REPO / "scripts" / "make_figures.py")], cwd=REPO).returncode:
            return 1
    r = subprocess.run([str(TECTONIC), "--keep-logs", "main.tex"], cwd=PAPER)
    if r.returncode:
        return r.returncode
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Output written on main\.\w+ \((\d+) pages?", log)
    print(f"paper/main.pdf built; pages: {m.group(1) if m else 'unknown'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

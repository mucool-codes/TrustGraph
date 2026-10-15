# paper/

LaTeX source of the S12 draft (IEEEtran, two-column conference class; venue undecided).

**Build (one command):**

    python scripts/build_paper.py        # use the project venv: .venv/Scripts/python.exe on Windows

It fetches the pinned, SHA-256-verified Tectonic into `tools/` if missing (`scripts/get_tectonic.py`; project-local,
gitignored, nothing on PATH or in the registry; delete `tools/` to remove it), regenerates figures and tables with
`scripts/make_figures.py` if they are missing, compiles `paper/main.tex`, and prints the page count. Output: `paper/main.pdf`
(generated, not committed). The first build downloads TeX packages into Tectonic's cache and needs network access.

Layout: `main.tex`, `sections/*.tex` (one per section), `sections/refs.tex` (verified references only).
Figures come from `figures/s10/` and tables from `results/s10/tables/paper_*.tex`, both written by `scripts/make_figures.py`;
nothing is redrawn or retyped. Every number in the text has its source in a `% src:` comment beside it.
Claims are gated by `CLAIMS.md` (not edited by this session): every sentence matches an ALLOWED entry, no FORBIDDEN entry appears.
`\TODO{...}` marks something the repository cannot supply (author block, authors of references 1-10, three missing citations).

"""SEALED ground truth (L4). Evaluation and scenario generation only.

Everything in this package describes what the scenario *really* is: which RSUs are
degraded and when, which segments carry the degradation, who colludes, every node's
true load, and every task's true completion time. None of it may reach a feature
vector, a loss, early stopping, or model selection.

The seal is structural, not a convention. `tests/test_sealing.py` walks the import
graph of the whole `trustgraph` package and fails if any module other than the ones
explicitly allowed to hold ground truth can reach this package, directly or
transitively - so a training module that imports it fails a test, rather than quietly
inflating a result. Adding a module to that allowlist is a deliberate act that belongs
in DECISIONS.md.
"""

"""L4: the sealed ground truth is structurally unreachable from the training path.

Two independent checks, because each can miss what the other catches:

  * static: parse every module in `trustgraph` (and every script) with `ast`, build the
    import graph - relative imports, `from x import submodule`, imports inside
    functions, and the implicit import of every parent package - and assert no module
    outside an explicit allowlist can reach `trustgraph.sealed`, directly or
    transitively;
  * runtime: import every non-allowlisted module in a fresh interpreter and assert that
    nothing under `trustgraph.sealed` ended up in `sys.modules`.

The check is default-deny. S3 will add training modules and they are covered without
anyone remembering to list them; a module that genuinely needs ground truth (evaluation
code computing the L4 diagnostic) has to be added to the allowlist below, which is a
visible, reviewable change that belongs in DECISIONS.md.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from conftest import REPO_ROOT

PACKAGE_DIR = REPO_ROOT / "src" / "trustgraph"
SCRIPTS_DIR = REPO_ROOT / "scripts"
SEALED = "trustgraph.sealed"

# Package modules allowed to hold ground truth. The simulator decides what goes wrong,
# so it must. Nothing else in the package may.
GROUND_TRUTH_MODULES = frozenset({"trustgraph.simulator"})

# Scripts allowed to hold ground truth: they generate scenarios or evaluate against
# the sealed state (PROJECT_SPEC.md 4.4 - evaluation code may hold both).
GROUND_TRUTH_SCRIPTS = frozenset(
    {"generate_scenario.py", "s2_calibration.py", "s2_report.py", "s2_rho_quantization.py"}
)


def _module_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in PACKAGE_DIR.rglob("*.py"):
        parts = list(path.relative_to(PACKAGE_DIR.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        index[".".join(parts)] = path
    return index


INDEX = _module_index()


def _with_parents(name: str) -> set[str]:
    """Importing a.b.c executes a and a.b first."""
    parts = name.split(".")
    return {".".join(parts[:i]) for i in range(1, len(parts) + 1)}


def _direct_imports(source: str, package: str) -> set[str]:
    """Every `trustgraph` module a source file imports, resolved to indexed names."""
    targets: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            targets.append(module)
            # `from pkg import name` may import a submodule called `name`.
            targets.extend(f"{module}.{alias.name}" for alias in node.names)

    resolved: set[str] = set()
    for target in targets:
        for name in _with_parents(target):
            if name in INDEX:
                resolved.add(name)
    return resolved


def _module_imports(name: str) -> set[str]:
    path = INDEX[name]
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    return _direct_imports(path.read_text(encoding="utf-8"), package) - {name}


GRAPH = {name: _module_imports(name) for name in INDEX}


def _reachable(start: set[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(start)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(GRAPH.get(node, set()) - seen)
    return seen


def _reaches_seal(modules: set[str]) -> set[str]:
    return {m for m in _reachable(modules) if m == SEALED or m.startswith(SEALED + ".")}


def _is_sealed(name: str) -> bool:
    return name == SEALED or name.startswith(SEALED + ".")


# ---------------------------------------------------------------- walker sanity


def test_walker_resolves_relative_and_package_imports():
    """If the walker cannot see ordinary edges, a clean result below means nothing."""
    assert "trustgraph.features" in GRAPH["trustgraph.graph"]
    assert "trustgraph.sealed.ground_truth" in GRAPH["trustgraph.sealed.injector"]
    assert {"trustgraph.sealed", "trustgraph.sealed.injector"} <= GRAPH["trustgraph.simulator"]


def test_walker_detects_the_simulator_reaching_the_seal():
    assert _reaches_seal({"trustgraph.simulator"}), "walker is vacuous"


def test_walker_catches_a_transitive_leak():
    """A synthetic module importing a harmless module that imports the simulator."""
    GRAPH["trustgraph._probe_mid"] = {"trustgraph.simulator"}
    GRAPH["trustgraph._probe_top"] = {"trustgraph._probe_mid"}
    try:
        assert _reaches_seal({"trustgraph._probe_top"})
    finally:
        del GRAPH["trustgraph._probe_mid"], GRAPH["trustgraph._probe_top"]


# ------------------------------------------------------------------- the seal


def test_no_package_module_outside_the_allowlist_can_reach_ground_truth():
    leaks = {}
    for name in INDEX:
        if _is_sealed(name) or name in GROUND_TRUTH_MODULES:
            continue
        reached = _reaches_seal({name})
        if reached:
            leaks[name] = sorted(reached)
    assert not leaks, f"modules reaching sealed ground truth (L4): {leaks}"


def test_allowlist_is_no_wider_than_it_needs_to_be():
    for name in GROUND_TRUTH_MODULES:
        assert name in INDEX, f"allowlisted module {name} does not exist"
        assert _reaches_seal({name}), f"{name} is allowlisted but never touches the seal"


def test_training_path_modules_are_all_covered():
    """The modules S3's trainer will import are clean today - named explicitly so the
    failure message says which part of the training path broke."""
    training_path = {
        "trustgraph.config",
        "trustgraph.features",
        "trustgraph.graph",
        "trustgraph.model",
        "trustgraph.observed",
        "trustgraph.scenario",
        "trustgraph.selection",
        "trustgraph.tasks",
        "trustgraph.tracking",
        "trustgraph.trace",
    }
    assert training_path <= set(INDEX)
    assert not _reaches_seal(training_path)


def test_no_script_outside_the_allowlist_can_reach_ground_truth():
    leaks = {}
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        if path.name in GROUND_TRUTH_SCRIPTS:
            continue
        direct = _direct_imports(path.read_text(encoding="utf-8"), package="")
        reached = _reaches_seal(direct)
        if reached:
            leaks[path.name] = sorted(reached)
    assert not leaks, f"scripts reaching sealed ground truth (L4): {leaks}"


def test_importing_every_training_safe_module_loads_no_ground_truth():
    """Runtime check in a clean interpreter - catches anything the AST walk cannot see."""
    safe = sorted(
        n for n in INDEX if not _is_sealed(n) and n not in GROUND_TRUTH_MODULES
    )
    program = "\n".join(
        [
            "import importlib, sys",
            f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r})",
            f"for name in {safe!r}:",
            "    importlib.import_module(name)",
            f"leaked = sorted(m for m in sys.modules if m == {SEALED!r} or m.startswith({SEALED + '.'!r}))",
            "print('LEAKED', leaked)",
            "importlib.import_module('trustgraph.simulator')",
            f"print('CONTROL', any(m.startswith({SEALED!r}) for m in sys.modules))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    assert "LEAKED []" in result.stdout, result.stdout
    assert "CONTROL True" in result.stdout, "control import did not load the seal"

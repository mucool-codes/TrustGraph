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
GROUND_TRUTH_MODULES = frozenset({"trustgraph.simulator", "trustgraph.diagnostic"})

# Scripts allowed to hold ground truth: they generate scenarios or evaluate against
# the sealed state (PROJECT_SPEC.md 4.4 - evaluation code may hold both).
GROUND_TRUTH_SCRIPTS = frozenset(
    {
        "generate_scenario.py",
        "s2_calibration.py",
        "s2_report.py",
        "s2_rho_quantization.py",
        "s2b_cold_start_conditions.py",
        "s3_train.py",  # wires the sealed-label diagnostic into the trainer
        "s4_eval.py",  # evaluates frozen C/D against sealed classes (D43)
        "s4_analyse.py",  # aggregates the above; verdict against D43
        "s5_diagnostics.py",  # S5 gate-response diagnostics (D48)
        "s6_run.py",  # closed-loop run + sealed side files (D51)
        "s6_tune.py",  # weight tuning on the tuning seed (D50)
        "s6_demo.py",  # the S6 demo plot
        "s8_common.py",  # S8 shared sample and closed-loop helpers
        "s8_fidelity.py",  # H4 fidelity, stability, neighbour-clause rate (D54)
        "s8_timing.py",  # H5 timing (D54)
        "s8_example.py",  # the merged example (D54)
        "s9_sweep.py",  # the sweep runner scores dispatches against the true degraded set (D56)
        "s9b_gate.py",  # Gate-C tuning and evaluation (D59)
        "s13_h.py",  # heuristic-baseline tuning and evaluation (D65)
        "s13_stag.py",  # staggered-onset mechanism check, evaluation and closed loop (D65)
        "s14_d2.py",  # D2 evaluation: open-loop latency and AUC, closed loop (D68)
        "s15_horizon.py",  # horizon-objective evaluation: open-loop latency, closed loop (D72)
        "s16_confirm.py",  # S16 confirmatory C-h vs C: latency, false alarms, closed loop (D76)
        "s17_smoke.py",  # S17 smoke run: sealed fault timeline of the new scenario modes (D81)
        "s18_run.py",  # S18 spreading-fault and cold-start run: scenarios, open-loop and closed-loop evaluation, secondary arm (D81, D83)
        "s10_data.py",  # figure inputs that need closed-loop runs (timeline, worked example)
        "s9_overhead.py",  # H4/H5 serial stage (D56 section 7)
    }
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
        "trustgraph.dataset",
        "trustgraph.features",
        "trustgraph.graph",
        "trustgraph.model",
        "trustgraph.observed",
        "trustgraph.scenario",
        "trustgraph.selection",
        "trustgraph.tasks",
        "trustgraph.tracking",
        "trustgraph.train",
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


# ------------------------------------------------- S17 (D81 section 3): the new generator code stays sealed
#
# The spreading-fault and restart generators decide which RSU is the seed, which hop each degraded RSU is
# from it, which RSUs restart and which of those fail. All of that is ground truth. Four checks: the names
# that carry it appear only in the sealed package and the world-side allowlists; the helpers that compute it
# are defined only under `trustgraph.sealed`; the one config section a non-sealed module may touch is the
# plain `restart` dict in config.py (a scenario knob like `degradation`, never a per-RSU fact); and the
# observable surface (scenario fields, feature columns, file keys) is the same with the modes on and off.

# Names that only ground-truth code may use (code identifiers, attribute names, keyword names, and string
# keys; comments and docstrings are not names). `restart_plan`, `hop_distances` and `spread_onsets` are the
# helpers; the rest are the sealed per-RSU fields and the config key that switches the spread on.
S17_SEALED_NAMES = frozenset(
    {"spread_lag_steps", "spread_hop", "restart_steps", "restart_plan", "hop_distances", "spread_onsets", "RestartConfig"}
)
S17_HELPERS = frozenset({"restart_plan", "hop_distances", "spread_onsets"})
# `restart` (the config section / Config field) may appear here and in the world-side modules, nowhere else.
S17_RESTART_ALLOWED = frozenset({"trustgraph.config"})


def _names_in(source: str) -> set[str]:
    """Every identifier, attribute, argument, keyword, definition, import alias and string constant in `source`."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.update({node.name.split(".")[-1], node.asname or ""})
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def test_s17_name_scanner_sees_names_strings_and_keywords():
    seen = _names_in("def f(spread_hop):\n    cfg.get('restart_steps'); g(spread_onsets=1); h = x.hop_distances\n")
    assert {"spread_hop", "restart_steps", "spread_onsets", "hop_distances"} <= seen


def test_s17_generator_names_appear_only_in_sealed_and_world_side_modules():
    leaks = {}
    for name, path in INDEX.items():
        if _is_sealed(name) or name in GROUND_TRUTH_MODULES:
            continue
        found = _names_in(path.read_text(encoding="utf-8")) & S17_SEALED_NAMES
        if "restart" in _names_in(path.read_text(encoding="utf-8")) and name not in S17_RESTART_ALLOWED:
            found = found | {"restart"}
        if found:
            leaks[name] = sorted(found)
    assert not leaks, f"modules outside the seal naming the S17 generator state (L4): {leaks}"


def test_s17_generator_names_appear_in_no_script_outside_the_allowlist():
    leaks = {}
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        if path.name in GROUND_TRUTH_SCRIPTS:
            continue
        found = _names_in(path.read_text(encoding="utf-8")) & (S17_SEALED_NAMES | {"restart"})
        if found:
            leaks[path.name] = sorted(found)
    assert not leaks, f"scripts outside the allowlist naming the S17 generator state (L4): {leaks}"


def test_s17_helpers_are_defined_only_under_the_seal():
    defined_in: dict[str, set[str]] = {h: set() for h in S17_HELPERS}
    for name, path in INDEX.items():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name in S17_HELPERS:
                defined_in[node.name].add(name)
    assert all(mods for mods in defined_in.values()), f"helpers not found: {defined_in}"
    assert all(all(_is_sealed(m) for m in mods) for mods in defined_in.values()), defined_in


def test_s17_the_training_path_still_cannot_reach_the_generator():
    """The new modes added no import edge from the training path to the seal (the default-deny test above covers
    every module; this names the S17 additions so the failure message says what broke)."""
    assert "trustgraph.sealed.injector" not in _reachable({"trustgraph.config", "trustgraph.dataset", "trustgraph.train", "trustgraph.model"})
    assert "trustgraph.sealed.injector" in GRAPH["trustgraph.simulator"]


def test_s17_observable_surface_is_unchanged_by_the_new_modes(demo_bundle, tmp_path):
    import dataclasses

    import numpy as np

    from trustgraph.config import load_config
    from trustgraph.features import RSU_FEATURES
    from trustgraph.observed import ObservedScenario
    from trustgraph.simulator import generate_scenario

    assert tuple(f.name for f in dataclasses.fields(ObservedScenario)) == (
        "rsu_features", "rsu_active", "vehicle_task_demand", "tasks", "seed", "dispatch_policy",
    )
    assert RSU_FEATURES == ("load", "queue_depth", "cert_valid", "success_ewma", "latency_dev", "uptime_stability")
    _, world, trace = demo_bundle
    keys = {}
    for name in ("demo", "spread_L80", "restart_f20"):
        sc = generate_scenario(load_config(REPO_ROOT / "configs" / f"{name}.yaml"), world, trace)
        sc.observed.save(tmp_path / f"{name}.observed.npz")
        keys[name] = set(np.load(tmp_path / f"{name}.observed.npz").files)
    assert keys["spread_L80"] == keys["demo"] == keys["restart_f20"]
    # the sealed-only state is never in the observable file
    assert not any("hop" in k or "restart" in k or "onset" in k for k in keys["demo"] | keys["restart_f20"])


def test_s17_sealed_file_keys_with_both_modes_off_are_the_pre_s17_keys(tmp_path):
    import numpy as np

    from trustgraph.sealed.injector import INJECTION_PURPOSES, inject

    rngs = {p: np.random.default_rng([1, k, 0]) for k, p in enumerate(INJECTION_PURPOSES)}
    gt = inject(np.repeat(np.arange(4), 5), {"fraction": 0.2, "rho": 1.0, "onset_step": 400, "ramp_steps": 300}, None, None, rngs)
    z = np.zeros((2, 20), dtype=np.float32)
    gt.with_realised(z, z, np.zeros(1), np.zeros(1, dtype=bool)).save(tmp_path / "a.SEALED.npz")
    assert set(np.load(tmp_path / "a.SEALED.npz").files) == {
        "sealed_format_version", "behavior_class", "backhaul_segment_id", "rho", "degraded_fraction", "onset_step",
        "ramp_steps", "severity", "drop_prob", "num_correlated_picks", "collusion_group", "under_report", "join_step",
        "true_load", "true_queue_depth", "task_true_completion_s", "task_dropped",
    }

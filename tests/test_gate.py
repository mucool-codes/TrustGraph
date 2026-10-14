"""S9b: the Gate-C baseline (DECISIONS.md D59): candidates below tau are removed, the L1 rule runs on the survivors,
an empty survivor set falls back to the un-gated set, and no existing variant changes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trustgraph.offload import OffloadSettings
from trustgraph.simulator import run_closed_loop


class StubHead(torch.nn.Module):
    """Trust = 0.9 for even node indices, 0.1 for odd ones (RSUs are the first rows of the node matrix)."""

    def forward(self, x, edge_index, edge_attr):
        idx = torch.arange(x.shape[0])
        return torch.where(idx % 2 == 0, torch.tensor(0.9), torch.tensor(0.1))


def _settings(variant, tau=None, **kw):
    base = dict(variant=variant, alpha=1.0, beta=0.25, gamma=0.6, log_decisions=True, trust_gate_tau=tau)
    base.update(kw)
    return OffloadSettings(**base)


def test_gate_needs_a_tau_and_only_gate_c_may_have_one():
    with pytest.raises(ValueError):
        OffloadSettings(variant="Gate_C")
    with pytest.raises(ValueError):
        OffloadSettings(variant="C", trust_gate_tau=0.5)
    with pytest.raises(ValueError):
        OffloadSettings(variant="Gate_C", trust_gate_tau=0.0)
    assert OffloadSettings(variant="Gate_C", trust_gate_tau=0.5).effective_alpha == 1.0  # trust stays a weighted term too


def test_candidates_below_tau_are_removed_and_a_dispatch_never_goes_below_tau_unless_nobody_passes(cfg, world, trace):
    sc = run_closed_loop(cfg, world, trace, _settings("Gate_C", 0.5), model=StubHead())
    log, c = sc.extras["decision_log"], sc.extras["counters"]
    falls = excluded = 0
    for rec in log:
        if rec["selected_rsu"] is None:
            continue
        pool = [x for x in rec["candidates"] if x["dwell_eligible"]]
        passing = [x["rsu"] for x in pool if x["trust"] >= 0.5]
        if passing:
            assert rec["selected_rsu"] in passing and rec["gate_fallback"] is False
            excluded += int(len(passing) < len(pool))
        else:
            assert rec["gate_fallback"] is True and rec["selected_rsu"] in [x["rsu"] for x in pool]
            falls += 1
            excluded += 1  # the gate did remove candidates (all of them) before the fallback restored the set
    assert c.gate_fallback == falls and c.gate_excluded == excluded
    assert len(sc.extras["gate_fallback_steps"]) == falls
    assert falls > 0 and excluded > 0, "the stub should exercise both the gate and the fallback"


def test_a_gate_nobody_can_pass_is_variant_C(cfg, world, trace):
    c = run_closed_loop(cfg, world, trace, _settings("C"), model=StubHead())
    g = run_closed_loop(cfg, world, trace, _settings("Gate_C", 1.0), model=StubHead())
    for name in c.observed.tasks.__dataclass_fields__:
        assert np.array_equal(getattr(c.observed.tasks, name), getattr(g.observed.tasks, name), equal_nan=True), name
    n_dispatch = sum(1 for r in g.extras["decision_log"] if r["selected_rsu"] is not None)
    assert g.extras["counters"].gate_fallback == n_dispatch  # every dispatch fell back


def test_existing_variants_have_no_gate_state(cfg, world, trace):
    sc = run_closed_loop(cfg, world, trace, _settings("C"), model=StubHead())
    assert sc.extras["counters"].gate_fallback == sc.extras["counters"].gate_excluded == 0
    assert len(sc.extras["gate_fallback_steps"]) == 0
    assert all("gate_tau" not in r and all("gate_pass" not in x for x in r["candidates"]) for r in sc.extras["decision_log"])

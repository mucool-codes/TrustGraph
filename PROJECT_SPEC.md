# PROJECT_SPEC.md

**Trust-Aware, Explainable Task Offloading for Vehicular Fog Networks using Graph Neural Networks**

Status: authoritative. Created S0, 2026-09-03.

> **Precedence.** This document supersedes `DEVELOPMENT_GUIDE.md` and
> `TECHNICAL_AND_NOVELTY.md`. Where any of them conflict, **PROJECT_SPEC.md wins.**
> The two design docs remain the reference for motivation, prior-art positioning,
> and the parts of the design this spec does not contradict.

---

## 1. System summary

A vehicle in a connected transport network continuously generates compute tasks it
cannot process onboard (sensor fusion, hazard classification, cooperative perception,
path re-planning). It offloads them to a roadside fog node (RSU) one radio hop away,
because a cloud round trip (~50-100 ms) exceeds the latency budget for anything
safety-adjacent.

Today the choice of RSU is computed from proximity, load, and estimated latency.
Trust is not part of that decision — it was settled once, at certificate presentation.
Certificate-based trust (SCMS / IEEE 1609.2.1, ETSI TS 102 941) answers *"is this node
who it claims to be?"*, not *"is this node currently working correctly?"* A node can
hold a valid certificate while overloaded, faulty, under attack, or compromised, and
SCMS CRL revocation propagates in hours to days. The system is therefore well-defended
against impostors and undefended against degradation.

This project adds a **behavioral trust layer** to the offloading decision. It has three
parts, executed each timestep:

1. **Dynamic graph construction.** The network is rebuilt as a graph `G_t = (V_t, E_t)`
   at every timestep. Nodes are RSUs and vehicles; edges are vehicle-RSU coverage links
   and RSU-RSU coordination links. Topology churn from mobility is the normal operating
   condition, not a perturbation.
2. **Trust scoring via message passing.** A GNN propagates behavioral evidence across
   `G_t`. After `k` rounds, each RSU's embedding encodes its own behavioral history and
   that of nodes within `k` hops. A scalar head maps that embedding to `trust_v`.
   This is the **only** thing the GNN produces.
3. **Trust-integrated selection and explanation.** A fixed analytic rule combines trust
   with latency and load to pick a target; a two-layer explanation says why that target
   won and why the runner-up lost.

**The novelty claim** is the conjunction of four properties: trust that is (a) learned
from behavior rather than asserted by certificate, (b) propagated across network
structure via message passing, (c) integrated as a *weighted term* in the offloading
objective rather than as an admission gate, and (d) attributable per decision. Each has
prior art individually; the conjunction, in the vehicular offloading setting, does not.

**The load-bearing hypothesis is H2** — that message passing beats a node-local model.
It is decided almost entirely by the degradation injection model, which is why
correlated failure is specified structurally (L5) rather than statistically.

### 1.1 What the system does not do

- It does not replace SCMS. `cert_valid` remains an input feature; this is a
  complementary behavioral layer, not a substitute for cryptographic identity.
- It does not diagnose the *cause* of degradation (hardware fault vs. active attack).
- It does not assess drivers or humans. Trust here is infrastructure node reliability.
- It does not handle multi-hop offloading chains.
- It does not learn node selection (L1), and it does not defend against an adaptive
  adversary (L10).

---

## 2. Locked decisions

These are locked. They are not to be silently overridden. If one appears wrong, raise a
DECISION REQUEST rather than changing it. Rationale and alternatives for each are in
`DECISIONS.md`.

**L1 — The GNN's only job is producing a scalar trust score per fog node.**
Node selection is NOT learned. Selection is the fixed analytic rule
`score(v) = alpha*trust_v - beta*latency_v - gamma*load_v`, argmax over in-range
candidates, with hand-tuned `alpha`, `beta`, `gamma`.

**L2 — Baseline A (no trust) is that same analytic rule with `alpha = 0`.**
There is no learned baseline selector anywhere in this project.

**L3 — Trust is trained SELF-SUPERVISED**: predict whether node `v` completes the next
task within its deadline, learned from observed task outcomes. The injected behavior
class is NEVER a training label.

**L4 — The injected behavior class is sealed for evaluation only.** It may be used as a
printed DIAGNOSTIC (AUC of `trust_v` against true class, logged each epoch) but must
never enter a loss function or feature vector. Enforce this structurally: keep ground
truth in a separate object that the training path cannot import.

**L5 — Correlated failure is implemented via explicit backhaul segments.** Each RSU gets
a `backhaul_segment_id` at topology build time. Degradation is applied to a SEGMENT, not
to statistically correlated independent nodes.

**L6 — RSU-RSU edges are required.** Single homogeneous edge type, with a `same_segment`
boolean edge feature. NO `HeteroData`, no heterogeneous edge types. This is a deliberate
scope cut.

**L7 — Mobility sits behind a `MobilitySource` interface.** The synthetic implementation
is the default and is on the critical path. SUMO is an optional later swap and is NOT on
the critical path. Do not install or integrate SUMO in any session unless explicitly
told to.

**L8 — Advertised-vs-observed discrepancy is the trust signal.** Nodes advertise
load/latency; after a task completes or times out, observed outcome is compared to what
was advertised, and the DISCREPANCY (not the advertised value) feeds `success_ewma` and
`latency_dev`.

**L9 — Scale: 15-30 RSUs, 50-100 vehicles.** Do not scale beyond this. Scalability is
explicitly out of scope.

**L10 — No adaptive adversary.** Degraded nodes do not strategically game the trust
score. Stated as a limitation.

**L11 — Explanation is two layers.** Layer 1 is an exact analytic decomposition of the
linear selection score (contrastive: why the runner-up lost). Layer 2 is GNNExplainer
applied ONLY to the trust head. Fidelity testing (H4) applies to Layer 2 only; Layer 1
is exact by construction.

**L12 — Evaluation uses two 1-D sweeps, not a cross product.** Sweep `rho` (failure
correlation) at fixed 20% degraded fraction. Sweep degraded fraction at fixed high
`rho`. Models are trained ONCE per seed on mixed conditions, then frozen and evaluated
across all conditions.

### 2.1 Consequences worth stating explicitly

- **L1 + L2 mean the ablation isolates the trust signal, not the selector.** Every
  variant A-E uses the identical selection rule. The only thing that varies is where
  `trust_v` comes from (or whether it is used at all). This is what makes the ablation
  interpretable.
- **L3 + L4 mean the trust model never sees the answer.** The self-supervised target
  (did this node meet its deadline?) is an *observable*; the behavior class is not.
  A model that scores well on the sealed diagnostic has genuinely inferred behavior
  from outcomes.
- **L5 + L6 mean the H2 test has a real structural signal to find.** Backhaul segments
  give the GNN something propagable that a node-local model provably cannot see, and
  `same_segment` on the edge tells it where to look — without heterogeneous edge types.
- **L1's fixed weights make Layer 1 of the explanation exact.** Because the selection
  score is linear in three terms, the contribution of each term to the margin between
  winner and runner-up is an identity, not an approximation. That is why H4 fidelity
  testing applies only to Layer 2.

---

## 3. Ablation variants

Five variants (from `DEVELOPMENT_GUIDE.md` section 5.4, re-expressed under L1/L2). All
five use the **same** analytic selection rule from L1. They differ only in the trust
term.

| Variant | Trust source | Message passing | Explainability | `alpha` | Purpose |
|---|---|:---:|:---:|:---:|---|
| **A** | none | — | no | `0` | Baseline — load/latency only |
| **B** | `cert_valid` only (binary, no behavioral component) | no | no | > 0 | Mimics deployed SCMS trust |
| **C** | behavioral, node-local (MLP over the node's own features) | no | no | > 0 | **Isolates whether graph structure actually matters** |
| **D** | behavioral + GNN (message passing) | yes | no | > 0 | Full trust model, no explanation |
| **E** | behavioral + GNN + explanation (L11) | yes | yes | > 0 | Full proposed system |

Notes:

- **Variant A** is `alpha = 0` in the identical rule (L2), not a separate model and not
  a learned selector.
- **Variant B** sets `trust_v = cert_valid` (1.0 or 0.0). It cannot react to behavior by
  construction — that is the point, and it is why H1 against B is necessary but not
  persuasive on its own.
- **Variant C is the most important and the easiest to skip.** C and D are trained on
  the *identical* self-supervised objective (L3) over the *identical* features; the only
  difference is whether the trust model aggregates over neighbors. If C matches D, the
  graph contributes nothing and the novelty claim collapses. The C-vs-D probe runs early
  (~day 15), before explainability, so a null result leaves time to respond.
- **E vs. D** differ only in that E emits explanations. E's *decisions* are identical to
  D's — explanation is read-only and does not alter selection. Any measured difference
  between D and E decisions is a bug, and the evaluation should assert this.

---

## 4. Evaluation plan

### 4.1 Hypotheses

- **H1 — Detection latency.** Behavioral trust routes away from a degrading node faster
  than certificate-based trust. *Criterion:* reduction in tasks dispatched to a degraded
  node between degradation onset and effective avoidance, vs. Baseline B. *Caveat:* B
  cannot react by construction; a reviewer will discount this as a strawman. Necessary,
  not sufficient.
- **H2 — Structural advantage (critical).** Under correlated failure, cold-start, and
  collusion, GNN trust (D) outperforms node-local trust (C). *Criterion:* a
  statistically meaningful gap across >= 5 seeds, widening as `rho` strengthens.
  *If it fails:* report it. The fallback contribution is the integrated + explainable
  formulation plus an honest negative result on structural propagation.
- **H3 — Task success under degradation.** End-to-end completion rate is higher for the
  proposed system across degraded fractions 5-30%. *Criterion:* consistent improvement
  with non-overlapping error bars at >= 2 operating points.
- **H4 — Explanation fidelity.** *Applies to Layer 2 only* (L11). Removing the
  top-attributed feature flips the decision often; removing a random low-attribution
  feature flips it rarely. The gap between those two rates is the result — report both.
  Layer 1 is exact by construction and is not fidelity-tested.
- **H5 — Decision overhead.** Trust scoring and explanation generation stay inside a
  plausible fog-layer budget. *Criterion:* per-decision inference time reported
  explicitly, explanation generation measured separately. It is expected to be the
  expensive component and may be practical only asynchronously — an acceptable finding.

### 4.2 Sweep design (L12)

Two 1-D sweeps, **not** a cross product:

1. **Correlation sweep.** Vary `rho` (failure correlation) with degraded fraction fixed
   at **20%**.
2. **Fraction sweep.** Vary degraded fraction (5%, 10%, 20%, 30%) with `rho` fixed at a
   **high** value.

**Train-once protocol.** For each seed, each learned variant (C, D, E) is trained
**once** on mixed conditions, then **frozen** and evaluated across every point in both
sweeps. Models are never retrained per sweep point. This keeps the sweeps a test of
generalization rather than of per-condition fitting, and it keeps the cost linear.

### 4.3 Statistical protocol

- Every configuration runs across **>= 5 random seeds** (different vehicle spawns,
  different degraded-segment selections).
- Report **mean +/- standard deviation**, never single-run numbers.
- The seed controls: mobility, topology (including `backhaul_segment_id` assignment),
  degraded-segment selection, task arrivals, and model init. One seed reproduces a run
  exactly (Standing Rule 7).

### 4.4 Sealed-label diagnostic (L4)

Each epoch, log AUC of `trust_v` against the true injected behavior class. This is a
**printed diagnostic only**. Structural enforcement: ground truth lives in a separate
object that the training path cannot import. The diagnostic is computed outside the
training path, by evaluation code that is allowed to hold both.

### 4.5 Metrics

- Task success / failure rate under degradation.
- Tasks dispatched to degraded nodes before effective avoidance (detection latency).
- Trust-violation detection rate against sealed ground truth.
- Per-decision latency, broken down by component (trust inference vs. explanation).
- Explanation fidelity rates (top-attributed vs. random-low, Layer 2 only).

### 4.6 Limitations to state in the paper

Simulation-only, no physical RSU testbed. Synthetic behavior injection — no public
dataset pairs vehicular mobility with fog-node trust ground truth, so results are
conditional on the realism of the injection model, which is the most attackable
assumption in the work. Single-hop offloading only. No adaptive adversary (L10). Fixed
objective weights `alpha, beta, gamma`, hand-tuned. Scale capped at 15-30 RSUs / 50-100
vehicles (L9); scalability is out of scope.

---

## 5. Glossary — node and edge features

Feature names from `TECHNICAL_AND_NOVELTY.md` section 3.1, plus fields added by the
locked decisions. **These names are canonical.** Code uses them verbatim.

### 5.1 Fog node (RSU) features

| Feature | Type | Description |
|---|---|---|
| `load` | float | Current compute utilization. **Advertised** by the node (L8) — may be false. |
| `queue_depth` | float | Pending tasks awaiting execution. Advertised (L8). |
| `cert_valid` | binary | Certificate validity from SCMS. The whole of Variant B's trust. |
| `success_ewma` | float | EWMA of recent task completions. Fed by advertised-vs-observed **discrepancy** (L8), not by raw outcomes. |
| `latency_dev` | float | Deviation of observed latency from advertised/expected (L8). |
| `uptime_stability` | float | Frequency of observed restarts or dropped sessions. |

`success_ewma` and `latency_dev` are the two behavioral features and the core of the
trust signal. Under L8 both are *discrepancy* quantities: what the node promised minus
what was observed after the task completed or timed out. A node that advertises low load
and then misses deadlines is penalized precisely because the two disagree.

### 5.2 Vehicle features

| Feature | Type | Description |
|---|---|---|
| `task_demand` | float | Size / compute requirement of the task to offload. |
| `speed` | float | Current speed. |
| `dwell_estimate` | float | Estimated remaining time in the current coverage zone. |

### 5.3 Edge features

Single homogeneous edge type (L6) — one `edge_attr` matrix covering both vehicle-RSU and
RSU-RSU edges.

| Feature | Type | Description |
|---|---|---|
| `link_latency` | float | Link latency estimate. |
| `signal_strength` | float | Signal strength. |
| `link_age` | float | How long the link has existed. Newly formed links are less characterized and carry more uncertainty. |
| `same_segment` | binary | Both endpoints on the same `backhaul_segment_id` (L6). The structural handle for correlated failure. |

### 5.4 Topology / scenario fields (not model inputs)

| Field | Type | Description |
|---|---|---|
| `backhaul_segment_id` | int | Assigned per RSU at topology build time (L5). Degradation is applied per segment. Used to derive `same_segment`; not itself a node feature. |
| `behavior_class` | enum | **SEALED (L4).** `reliable` / `degraded` / `compromised`. Evaluation only. Never a feature, never a label, never importable from the training path. |
| `rho` | float | Failure correlation parameter. Swept in the correlation sweep (L12). |
| `degraded_fraction` | float | Fraction of nodes degraded. Swept in the fraction sweep (L12). |

### 5.5 Selection-rule symbols

| Symbol | Description |
|---|---|
| `trust_v` | Scalar trust score for node `v`, in [0, 1]. The GNN's only output (L1). |
| `latency_v` | Estimated latency to node `v` for the deciding vehicle. |
| `load_v` | Advertised load at node `v`. |
| `alpha` | Weight on trust. `0` for Variant A (L2). |
| `beta` | Weight on latency (penalty). |
| `gamma` | Weight on load (penalty). |

**Trust enters the selection function, not a pre-filter.** A threshold that excludes
nodes below a cutoff is a gate — architecturally the same as certificate checking with a
different signal. Trust as a weighted term means a moderately-trusted-but-much-closer
node can still win, which is correct under a latency budget. This is the difference
between "we added a better filter" and "we changed the decision function", and it is
worth stating explicitly in the paper.

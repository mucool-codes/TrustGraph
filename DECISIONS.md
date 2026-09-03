# DECISIONS.md

**What was CHOSEN and WHY.** Every non-trivial design choice: the choice, the
alternatives considered, the rationale, the date. This is the review-defense artifact —
when a reviewer asks "why did you do it this way", the answer is here.

Companion document: `FINDINGS.md` records what was *measured*. Keep them distinct — a
decision is a choice, a finding is a number.

**Amendment rule.** If a later session reverses an earlier decision, add a **new entry**
that supersedes the old one. Never edit or delete the original. Mark the old entry with
a `SUPERSEDED BY` line and leave its text intact.

**Entry format.**

```
### D<n> — <title>
Date: YYYY-MM-DD | Session: S<n> | Status: active | superseded by D<m>
Decision: <what was chosen>
Alternatives: <what else was considered>
Rationale: <why this one>
```

---

## Locked decisions (L1-L12)

These twelve were fixed during design review, before S0, and supersede the corresponding
choices in `DEVELOPMENT_GUIDE.md` and `TECHNICAL_AND_NOVELTY.md`. They are recorded here
with the reasoning behind each. They are **locked**: a future session that believes one
is wrong raises a DECISION REQUEST rather than changing it.

### D1 (L1) — The GNN produces only a scalar trust score; selection is not learned
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** The GNN's only output is a scalar `trust_v` per fog node. Node selection is
the fixed analytic rule `score(v) = alpha*trust_v - beta*latency_v - gamma*load_v`,
argmax over in-range candidates, with hand-tuned `alpha, beta, gamma`.

**Alternatives:** (a) a learned selector — GNN scores candidates directly and the argmax
is over learned scores, as in `DEVELOPMENT_GUIDE.md` Phase 2; (b) a learned combination
layer over `(trust, latency, load)`.

**Rationale:** A fixed weighting is interpretable, ablatable, and defensible. It makes
the ablation isolate the *trust signal* rather than the selector — every variant A-E uses
the identical rule, so any measured difference is attributable to trust and nothing else.
It also makes Layer 1 of the explanation (D11) an exact identity rather than an
approximation. A learned selector would confound the trust claim with selector capacity
and would make the explainability story materially harder.

### D2 (L2) — Baseline A is the same analytic rule with alpha = 0
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** Baseline A (no trust) is the D1 rule with `alpha = 0`. There is no learned
baseline selector anywhere in this project.

**Alternatives:** Train a GNN on a proxy objective (lowest-load node was the right
choice) as the no-trust baseline, per `DEVELOPMENT_GUIDE.md` Step 2.3.

**Rationale:** A learned baseline introduces a second confound — the baseline could lose
because its *selector* is undertrained, not because it lacks trust. Setting `alpha = 0`
makes A and D differ in exactly one term, which is what a clean ablation requires. It
also removes an entire training pipeline from the critical path.

### D3 (L3) — Trust is trained self-supervised on observed task outcomes
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** The trust head is trained to predict whether node `v` completes its next
task within its deadline, learned from observed outcomes. The injected behavior class is
never a training label.

**Alternatives:** Supervised training against the injected `behavior_class`
(reliable/degraded/compromised).

**Rationale:** Supervising on the injected class trains the model to invert the injection
model, not to infer reliability from behavior — a reviewer would correctly read the
result as circular. Deadline completion is an *observable* a real deployment would have.
Self-supervision also means the approach transfers to real traces where no behavior
labels exist, which is the honest framing of the contribution.

### D4 (L4) — The behavior class is sealed and structurally isolated
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** `behavior_class` is evaluation-only. It may be printed as a diagnostic
(AUC of `trust_v` against true class, logged each epoch) but never enters a loss function
or a feature vector. Enforced structurally: ground truth lives in a separate object that
the training path cannot import.

**Alternatives:** Convention and code review only ("just don't use it"); or a runtime
assertion at the loss boundary.

**Rationale:** Label leakage in a self-supervised setup is easy to introduce accidentally
and nearly invisible once present — a stray join on node id is enough. A structural
barrier (the training path physically cannot import the object) fails loudly at import
time instead of silently inflating results. Since the entire novelty claim rests on trust
being *inferred*, this is worth enforcing at the architecture level rather than trusting
discipline.

### D5 (L5) — Correlated failure via explicit backhaul segments
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** Each RSU is assigned a `backhaul_segment_id` at topology build time.
Degradation is applied to a SEGMENT, not to statistically correlated independent nodes.

**Alternatives:** Sample node failures from a correlated distribution (e.g. a Gaussian
copula with correlation `rho`) without any explicit shared-infrastructure object.

**Rationale:** H2 — does message passing beat a node-local model — is decided almost
entirely by the degradation injection model. Statistical correlation gives the GNN a
signal that is real but has no structural handle: nothing in the graph tells it *which*
nodes are coupled. An explicit segment is both more physically faithful (RSUs really do
share backhaul links, power feeds, and software builds) and gives the graph an actual
edge to propagate along. If the GNN cannot win with this, it will not win at all, and
that is a cleaner negative result than one confounded by an unlearnable coupling.

### D6 (L6) — RSU-RSU edges required; single homogeneous edge type
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** RSU-RSU edges are required. One homogeneous edge type covering both
vehicle-RSU and RSU-RSU links, with a `same_segment` boolean edge feature. No
`HeteroData`, no heterogeneous edge types.

**Alternatives:** `HeteroData` with distinct `(vehicle, covers, rsu)` and
`(rsu, backhaul, rsu)` relations and separate convolution weights per relation.

**Rationale:** RSU-RSU edges are non-negotiable — without them there is no path for
segment-level evidence to propagate and D5's whole point is lost. But heterogeneous edge
types roughly double the model surface, the config surface, and the debugging cost, for a
benefit that is unmeasured. `same_segment` as an edge feature gives the model the same
information in a form a homogeneous `SAGEConv` can use. This is a deliberate scope cut,
recorded as such; it can be revisited if the homogeneous version is demonstrably the
bottleneck (Standing Rule 3 — confirm the simple version is failing first).

### D7 (L7) — Mobility behind a MobilitySource interface; synthetic is the default
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** Mobility sits behind a `MobilitySource` interface. The synthetic
implementation is the default and is on the critical path. SUMO is an optional later
swap and is NOT on the critical path. No session installs or integrates SUMO unless
explicitly instructed.

**Alternatives:** SUMO-first, per `DEVELOPMENT_GUIDE.md` Phase 0 and Phase 1.

**Rationale:** SUMO is a heavy dependency (installer, TraCI, network authoring in
netedit) that gates *nothing* the core hypotheses need. H2 depends on the degradation
model, not on mobility realism; H1, H3, H4, H5 likewise. Putting SUMO on the critical
path risks spending days of a 40-day budget on tooling before the load-bearing question
has been asked. The interface keeps the swap cheap if realism is later challenged in
review, and "mobility is synthetic" is already in the limitations section either way.

### D8 (L8) — Advertised-vs-observed discrepancy is the trust signal
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** Nodes advertise load and latency. After a task completes or times out, the
observed outcome is compared to what was advertised, and the DISCREPANCY — not the
advertised value — feeds `success_ewma` and `latency_dev`.

**Alternatives:** Feed raw observed outcomes directly, ignoring what was advertised.

**Rationale:** Raw outcomes conflate "this node is busy and honest about it" with "this
node is lying". The first is fine — the selection rule already penalizes load via
`gamma`. The second is the actual trust failure, and it is precisely a *mismatch* between
claim and behavior. This also makes the compromised/colluding class meaningful: a node
that falsely advertises low load is individually plausible on its advertised features and
only detectable through discrepancy.

### D9 (L9) — Scale capped at 15-30 RSUs, 50-100 vehicles
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** 15-30 RSUs, 50-100 vehicles. Do not scale beyond this. Scalability is
explicitly out of scope.

**Alternatives:** Include a scaling study (100+ RSUs) to pre-empt a "does this scale"
reviewer question.

**Rationale:** None of H1-H5 is a scaling claim, and the project makes no scaling claim,
so a scaling study would answer a question nobody asked at the cost of days. Small graphs
also debug vastly faster, and 5+ seeds across two sweeps is already the dominant compute
cost. The cap is stated in limitations rather than defended with data.

### D10 (L10) — No adaptive adversary
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** Degraded nodes do not strategically manipulate their behavior to game the
trust score. Stated as a limitation.

**Alternatives:** Model an adversary that observes its own trust score and modulates
behavior to stay above the selection threshold.

**Rationale:** An adaptive adversary is a research contribution in its own right and
would need its own threat model, its own evaluation, and its own baselines. Bolting a
weak version onto this project would produce an untrustworthy robustness claim, which is
worse than no claim. Naming the gap explicitly is the stronger move in review — the
limitation is already written into `PROJECT_SPEC.md` section 4.6.

### D11 (L11) — Two-layer explanation; fidelity testing applies to Layer 2 only
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** Layer 1 is an exact analytic decomposition of the linear selection score,
contrastive — why the runner-up lost. Layer 2 is GNNExplainer applied ONLY to the trust
head. H4 fidelity testing applies to Layer 2; Layer 1 is exact by construction.

**Alternatives:** A single GNNExplainer pass over the whole decision pipeline.

**Rationale:** Under D1 the selection score is linear in three terms, so each term's
contribution to the winner-runner-up margin is an *identity*. Running an approximate
attribution method over an exactly decomposable function would be strictly worse and
would invite the reviewer question "why approximate something you can compute". Splitting
the layers also sharpens H4: the only thing being fidelity-tested is the part that is
genuinely opaque, the trust head. Contrastive explanation — why the alternative lost — is
also the specific differentiator against prior-art ref [10], which explains a
classification rather than an allocation.

### D12 (L12) — Two 1-D sweeps; train once per seed, then freeze
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** Sweep `rho` at fixed 20% degraded fraction; sweep degraded fraction at
fixed high `rho`. Not a cross product. Models are trained ONCE per seed on mixed
conditions, then frozen and evaluated across all conditions.

**Alternatives:** Full cross product of `rho` x degraded fraction; and/or retrain the
model at each sweep point.

**Rationale:** The cross product multiplies runs by the product of both axes for
information that two 1-D slices already carry — the axes are not expected to interact in
a way the slices would miss, and 5+ seeds per point makes the difference large. Training
once per seed and freezing is the more important half: retraining per sweep point would
measure how well each condition can be *fitted*, when the claim is that structural trust
*generalizes* across conditions. Freezing also matches deployment, where a model does not
get retrained when the failure correlation changes.

---

## Session decisions

### D13 — Two design docs recovered from Downloads, not re-authored
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** `DEVELOPMENT_GUIDE.md` and `TECHNICAL_AND_NOVELTY.md` were located in
`D:\Downloads (dont delete)\` and copied into the repo root unchanged, then committed.

**Alternatives:** Reconstruct the missing content (ablation table, feature glossary) from
the locked decisions alone.

**Rationale:** The ablation variants and the canonical feature names exist only in those
documents. Inventing them would have fabricated exactly the content `PROJECT_SPEC.md` is
meant to supersede, and every later session would have inherited the invention. Stopping
to locate the real files cost one round trip and removed that risk. `CLAUDE_CODE_SESSIONS.md`
was not found on disk; it is context-only for S0 and non-blocking, so S0 proceeded without
it.

### D14 — PROJECT_SPEC.md is the single authoritative document
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** `PROJECT_SPEC.md` supersedes both design docs wherever they conflict. The
design docs are kept in the repo, unedited, as the reference for motivation, prior-art
positioning, and everything the spec does not contradict. A superseded-notice banner
already exists at the top of `DEVELOPMENT_GUIDE.md`.

**Alternatives:** Edit the design docs in place to remove the superseded parts.

**Rationale:** The design docs carry the prior-art analysis, the CVE motivation, and the
novelty framing — content the spec deliberately does not duplicate. Editing them in place
would destroy the record of what changed during design review, which is itself
review-defense material. One authoritative document plus unedited history is clearer than
two partially-correct documents.

### D15 — Feature vectors are ordered, named tuples fixed at S0
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** Node and edge feature *names and order* are fixed in code at S0 as explicit
ordered lists (`RSU_FEATURES`, `VEHICLE_FEATURES`, `EDGE_FEATURES`), matching
`PROJECT_SPEC.md` section 5 verbatim. The walking skeleton generates fake values but the
correct names, order, and shapes.

**Alternatives:** Leave feature layout implicit until real features exist in a later
session.

**Rationale:** Column order is the kind of thing that silently diverges between the graph
builder, the model, and the explainer, and the bug surfaces as a quietly wrong
attribution rather than a crash. Fixing the order once, in one module both sides import,
makes Layer 2 attributions nameable for free and makes a mismatch a shape error instead
of a wrong answer.

### D16 — Vehicles and RSUs share one node feature matrix via zero-padded blocks
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** A single homogeneous node feature matrix `x` of width
`len(RSU_FEATURES) + len(VEHICLE_FEATURES)`. RSU rows fill the RSU block and zero the
vehicle block; vehicle rows do the reverse. A separate `is_rsu` mask marks node kind.

**Alternatives:** (a) `HeteroData` with separate node types; (b) project each node type
through its own encoder into a shared dimension before message passing.

**Rationale:** D6 already rules out `HeteroData` for edges, and splitting node types
while keeping edges homogeneous would reintroduce the same complexity through a different
door. Zero-padded blocks keep one `x`, one `edge_index`, one `SAGEConv` stack, and keep
the feature-name mapping in D15 a straight index into `x`'s columns — which Layer 2
attribution needs. Per-type encoders are the obvious upgrade if the padding proves to be
the bottleneck; Standing Rule 3 says confirm that first.

### D17 — Determinism via one explicit seed chain, no global RNG state
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** One master seed from the config derives a set of independent
`numpy.random.Generator` streams by purpose (topology, mobility, features, model init).
No module reads a global RNG. `torch.manual_seed` is set once from the same master seed
for model init only.

**Alternatives:** A single global `np.random.seed(...)` plus `torch.manual_seed(...)` at
process start.

**Rationale:** Global seeding makes reproducibility depend on *call order* — adding one
extra draw anywhere silently shifts every downstream stream, so a run stops reproducing
for reasons unrelated to the change. Per-purpose generators mean the mobility stream is
unaffected by adding a feature draw, which matters over 12 sessions of edits. It also
makes the L12 protocol expressible: the same seed must reproduce topology and degraded
segments across sweep points that differ in other respects.

### D18 — Torch pinned to CUDA 12.4 build; PyG core only, no compiled extensions
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** `torch==2.6.0+cu124` from the PyTorch cu124 index, plus `torch_geometric`
from PyPI. No `torch-scatter` / `torch-sparse` / `pyg-lib`.

**Alternatives:** (a) CPU-only torch; (b) full PyG stack including the compiled
extension wheels.

**Rationale:** cu124 is a well-tested build for Ada-generation cards and the installed
driver (581.86) is far newer than its floor. Since PyG 2.3 the compiled extensions are
optional — `SAGEConv` and the rest of the core run on pure-PyTorch fallbacks — and those
wheels are the single most common source of version-matrix breakage on Windows. Omitting
them removes that failure mode at no functional cost at D9's scale. Measured environment
state is recorded in `FINDINGS.md` F1.

### D19 — The skeleton runs on CPU by default; GPU is verified separately
Date: 2026-09-03 | Session: S0 | Status: active

**Decision:** `configs/demo.yaml` sets `device: cpu`. GPU availability is verified
independently by `scripts/verify_env.py`, which runs a real GraphSAGE forward pass
through the project's own `TrustHead` on CUDA.

**Alternatives:** (a) `device: auto`, preferring CUDA when present; (b) CUDA by default
with `torch.use_deterministic_algorithms(True)`.

**Rationale:** GraphSAGE aggregation uses scatter reductions, and their GPU
implementations are not bit-reproducible across runs — floating-point addition is not
associative and the reduction order varies. That would break the S0 exit condition
(byte-identical output across two runs) for a reason that has nothing to do with the
project. Option (b) would restore determinism but at a real speed cost and with ops that
raise rather than fall back. At the L9 scale — at most 30 RSUs and 100 vehicles, so a
graph in the low hundreds of nodes — the GPU is a convenience, not a requirement, and
`device` stays a config field so training sessions can opt in where exact
reproducibility of a forward pass matters less than throughput.

### D20 — Road layout is a synthetic Manhattan grid, not a real map
Date: 2026-09-03 | Session: S1 | Status: active

**Decision:** The road layout is a grid of arterials over a square region:
`blocks_x + 1` vertical roads crossing `blocks_y + 1` horizontal roads, all parameters
from the config. RSU candidate sites and vehicle waypoints both read this one object.

**Alternatives:** (a) an OpenStreetMap extract of a real city district; (b) keeping S0's
bare lattice with no notion of roads at all.

**Rationale:** D7 already put SUMO and map realism off the critical path, and importing
a real map would drag in the same dependency question through a different door — a
parser, a projection, and a set of degenerate geometries to handle — for realism that
none of H1-H5 depends on. A grid is the standard synthetic road model in the VANET
literature and is enough to produce what S1 actually needs: vehicles that follow roads,
enter and leave coverage zones, and hand off. Option (b) is what S0 had and is what this
session was asked to replace: without roads there is no reason for RSUs to be anywhere
in particular, and "RSU placement" has no meaning. The grid is named in the limitations
alongside synthetic mobility.

### D21 — RSUs are sited by greedy coverage maximisation with a redundancy discount
Date: 2026-09-03 | Session: S1 | Status: active

**Decision:** RSUs are placed on road intersections and segment midpoints, chosen
greedily to maximise covered road, where a road point already reached by `c` chosen
RSUs contributes `redundancy_decay ** c` (default 0.5) rather than nothing.

**Alternatives:** (a) farthest-point sampling over candidate sites, which was
implemented first; (b) uniform random sampling of sites; (c) pure coverage maximisation
with no redundancy discount.

**Rationale:** (b) clusters, leaving bare patches whose size is an artefact of the seed.
(a) was tried and measured (FINDINGS.md F3): maximising the minimum pairwise distance
drives sites onto the region boundary, which left the interior thin, the RSU-RSU graph
fragmented at 14 edges over 20 nodes, and 22% of vehicle-timesteps with no RSU in range
at all. (c) fixes coverage but then spreads the remaining RSUs to *minimise* overlap,
which is the opposite of what is wanted: under L1 selection is an argmax over in-range
candidates, so a vehicle that can see exactly one RSU produces a decision with nothing
to decide, and an evaluation full of those measures coverage rather than trust. The
discount makes the RSUs left over once the road is covered build a second layer instead.
Measured effect (F4): 100% of vehicle-timesteps covered, 2.44 RSUs in range on average,
and a single connected RSU-RSU component with mean degree 4.6.

### D22 — Backhaul segments by spatial clustering, plus a minority of off-region swaps
Date: 2026-09-03 | Session: S1 | Status: active

**Decision:** `backhaul_segment_id` is assigned by Lloyd clustering over RSU positions
with a farthest-point initialisation, then each RSU is reassigned to a different segment
with probability `segment_swap_prob` (0.10), skipping any swap that would take a segment
below `min_segment_size`.

**Alternatives:** (a) S0's contiguous runs in sorted-x order; (b) pure spatial clustering
with no swap noise; (c) uniformly random assignment.

**Rationale:** (c) is what D5 already rejects — a segment scattered across the region
gives message passing nothing local to propagate along. (a) is coherent only along one
axis and produces stripes rather than regions. (b) is the obvious choice and is most of
what is implemented, but it makes `same_segment` a deterministic function of position:
a model could then recover the segment structure from geometry without ever using the
edge feature, and an H2 result about that feature would be unfalsifiable. The swap noise
corresponds to something real — a spur off a neighbouring backhaul link, a site re-homed
during a build-out — and measurably breaks the equivalence: a distance-threshold
predictor recovers `same_segment` with only 63.0% accuracy (F4). The minimum-size guard
keeps L5's "several RSUs each" true, and every segment stays internally connected through
same_segment edges, which is the precondition for propagating segment-level evidence.

**Note added 2026-09-03 (S1, post-merge verification).** Two corrections to the
rationale above; the decision itself stands.

1. The 63.0% figure does not support the claim it is attached to. The same predictor
   scores 60.9% against a pure-geometry assignment of the same seed, so it measures the
   weakness of the probe rather than the effect of the swap — see the dated note on F4.
   The swap does make `same_segment` non-geometric, but by a margin this metric cannot
   resolve.
2. `figures/s1_topology.png` does not visually demonstrate the swap and must not be read
   as evidence either way. The plot colours each RSU by its *assigned* segment, so a
   swapped RSU appears in its new segment's colour; on seed 20260903 the single swapped
   node (RSU 19) also sits on the boundary between the two clusters rather than inside
   the wrong one, leaving nothing visible to spot. A reader looking at that image alone
   will conclude the segments are purely geometric. Verify with the assignment diff, not
   the figure.

### D23 — Mobility sources emit a trace; the pipeline never steps a mobility model
Date: 2026-09-03 | Session: S1 | Status: active
Supersedes the `reset()` / `step()` interface introduced in S0.

**Decision:** `MobilitySource` exposes `generate(num_steps, seed) -> Trace` instead of
`reset()` / `step()`. `scripts/generate_trace.py` writes the trace to a compressed .npz;
the pipeline, the statistics, and the plots all read it back. `run.py` fails with an
actionable error rather than generating a trace itself.

**Alternatives:** (a) keep the stepped interface and have each consumer drive its own
copy of the mobility model; (b) generate the trace in-process on first use and cache it.

**Rationale:** Under (a) every consumer re-derives the motion, and they diverge the
moment one of them takes an extra RNG draw — the exact failure mode D17 exists to
prevent, but at the level of whole components rather than single streams. Since the
evaluation plan (L12) freezes a model and replays it across many conditions, a single
on-disk realisation per (config, seed) is also what makes those runs comparable at all.
(b) keeps the convenience but loses the artefact: nothing on disk to point at when a
result needs reproducing months later. It also makes the L7 swap cheaper, not more
expensive — a SUMO source becomes another implementation returning a `Trace`, and
nothing downstream changes.

### D24 — Placeholder features are named constants, never random values
Date: 2026-09-03 | Session: S1 | Status: active

**Decision:** The behavioural features (`success_ewma`, `latency_dev`,
`uptime_stability`) and `cert_valid` are module-level constants in `graph.py` with the
neutral value for each — no discrepancy, no revocation — not random draws. S0 filled
them with seeded uniforms.

**Alternatives:** keep S0's random placeholder values so the columns have variance.

**Rationale:** Random placeholders have exactly the property that makes them dangerous:
a model trained on them produces a plausible-looking result, and nothing distinguishes
"learned from the real signal" from "fitted noise in a column that means nothing yet".
Constants make any dependence on them degenerate and obvious — a trust score that varies
cannot be varying because of `success_ewma`. Under L8 these are discrepancy quantities,
so the neutral value is also the honest one: as of S1 no task has been observed, so
there is no evidence of misbehaviour anywhere. A test asserts the constants, so the
session that makes them real has to delete that test deliberately.

### D25 — link_age is tracked across timesteps by a stateful snapshot builder
Date: 2026-09-03 | Session: S1 | Status: active

**Decision:** Snapshots come from a `SnapshotBuilder` that remembers when each present
link first appeared. `build(t)` accepts only `t = 0` or one past the previous call and
raises otherwise; `reset()` rewinds. A link that breaks forgets its history, so a
re-formed link starts again at age zero.

**Alternatives:** (a) a free function over an arbitrary timestep, with `link_age`
dropped or faked; (b) precomputing the whole age tensor for the trace up front.

**Rationale:** `link_age` is in the glossary (PROJECT_SPEC.md 5.3) precisely because a
newly formed link is less characterised and carries more uncertainty — that is a
statement about history, and (a) cannot express it. Making the ordering requirement an
explicit error rather than a silent wrong answer matters because the failure would
otherwise be a quietly too-old age on a link, which nothing would catch. (b) is a valid
optimisation but at 300 steps x 20 RSUs x 60 vehicles the state is a single (V, R)
integer array updated in place, so there is nothing to optimise yet (Standing Rule 3).

### D26 — Radio and backhaul link model isolated in one module
Date: 2026-09-03 | Session: S1 | Status: active

**Decision:** `links.py` holds the only assumption in the codebase about radio
propagation: a log-distance path-loss model for `signal_strength` and a
retransmission-cost model for `link_latency`, with separate latency bands for access
(vehicle-RSU) and backhaul (RSU-RSU) links. matplotlib is added as a dependency for the
two S1 figures and is imported nowhere on the model or evaluation path.

**Alternatives:** compute both quantities inline in the graph constructor, as S0 did
with a normalised-distance stand-in.

**Rationale:** The propagation model is the piece of S1 most likely to be challenged in
review as unrealistic, and therefore the piece most likely to need swapping. Keeping it
in one dataclass means a reviewer's objection is answered by changing one file, and it
made a real defect findable: the receiver sensitivity floor and the coverage radius are
not independent, and at the initial values the floor landed at exactly the coverage
radius, saturating every cell-edge link to `signal_strength = 0` (FINDINGS.md F5). That
was caught by a monotonicity test over the model in isolation, which would have been
awkward to write against geometry embedded in the graph builder.

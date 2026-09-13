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
   the figure — `scripts/s1_report.py` now prints the count.

3. **The stated purpose above is wrong and is corrected here.** The rationale claims the
   swap protects an H2 result from being "unfalsifiable" because a model could recover
   `same_segment` from geometry. That cannot happen in the ablation as specified:
   Variant C is an MLP over a node's *own* feature vector, which contains no position
   and no neighbour information, so it could not infer segment membership from geometry
   at any swap rate — and Variant D is handed `same_segment` on the edge regardless. The
   swap changes nothing about C-vs-D fairness.

   Its real value is narrative, and it is worth keeping for that alone. Without it a
   backhaul segment is exactly a region of the map, and a reviewer can fairly read any
   GNN gain under correlated failure as "things that are near each other fail together"
   — a spatial-smoothing result that needs no notion of shared infrastructure. Because
   segment membership is *not* purely spatial, the gain can be attributed to the
   backhaul structure the model was given rather than to proximity it could have
   inferred anyway. That is a claim about what the result means, not about whether the
   experiment is sound, and D27 makes it hold on every seed rather than on average.

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
SUPERSEDED BY D37 for `success_ewma`, `latency_dev`, `load`, `queue_depth` and
`task_demand`, which are real as of S2. Still active for `cert_valid` and
`uptime_stability` (FINDINGS.md F8).

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

### D27 — Swap count is a deterministic floor, and a swap must be physically admissible
Date: 2026-09-03 | Session: S1b | Status: active
Amends the mechanism of D22; the decision to have off-geometry segments stands.

**Decision:** Two changes to how `backhaul_segment_id` is perturbed.

1. **Floor, not rate.** After the probabilistic pass at `segment_swap_prob`, if fewer
   than `min_swap_fraction` of the RSUs (0.15, so 3 of 20) ended up off their geometric
   segment, further RSUs are moved until the floor is met, taking those nearest a
   segment boundary first. No randomness is involved in the top-up.
2. **Admissibility.** A swap of either kind is applied only if the RSU already has an
   RSU-RSU coordination edge to a member of the target segment, and only if removing it
   leaves its source segment connected over `same_segment` edges. Among qualifying
   targets the nearest centroid wins. `Topology.num_swapped_rsus` reports the achieved
   count, and `scripts/s1_report.py` prints it.

**Alternatives:** (a) leave the swap purely probabilistic and accept the variance;
(b) raise `segment_swap_prob` until zero-swap seeds become unlikely; (c) drop the
off-geometry property and rest H2 entirely on segment-level degradation, which does not
depend on it.

**Rationale:** (a) is what D22 did, and it made the property seed-dependent: at
`swap_prob = 0.1` over 20 RSUs the realised count is a binomial draw that came out zero
on seed 4 (FINDINGS.md F6). Under L12 every configuration is averaged over >= 5 seeds,
so a property that holds on four seeds and silently fails on the fifth is worse than one
that fails everywhere — the aggregate still looks fine and nothing in the output says
which seed was different. (b) shifts the distribution without removing the tail and
costs geometric coherence to do it. (c) is defensible and remains the fallback, but the
narrative in D22 note 3 is cheap to keep once the mechanism is deterministic.

The admissibility constraint fixes a separate defect the floor exposed rather than
caused (F6): a swapped RSU was joining its new segment with no `same_segment` edge into
it, leaving that segment split into two components. Every segment being a single
component over `same_segment` edges is the condition that lets segment-level evidence
propagate in one hop instead of detouring through a cross-segment neighbour, and D22 and
F4 both name it as a precondition for H2. It was true on seed 20260903 by luck and false
on four of six seeds tested. The constraint also has the better physical reading: a site
cannot be re-homed onto a backhaul link it has no path to.

The regression test for internal connectivity now runs across six seeds. Checking a
single default seed is what let both defects hide.

### D28 — Scenario horizon extended from 300 to 1200 steps
Date: 2026-09-13 | Session: S2 | Status: active

**Decision:** `scenario.num_steps` is 1200 (20 minutes at dt = 1 s). Degradation onset is
at step 400 and the ramp lasts 300 steps, leaving a 400-step pre-onset baseline and a
500-step post-ramp plateau. `simulator.check_trace_matches` refuses a trace whose steps,
vehicles, dt or seed differ from the config.

**Alternatives:** (a) keep S1's 300 steps and compress onset and ramp into them; (b) a
longer horizon still (e.g. 3600 steps).

**Rationale:** Each RSU sees roughly one task every three seconds (F7: median 366 tasks
over 1200 s), and the EWMA weight of 0.1 means a feature integrates about ten outcomes.
In 300 steps a ramp long enough to be gradual would leave no stable baseline before it
and no plateau after it, and neither detection latency nor the no-collapse property could
be measured. (b) buys nothing S2's exit condition or S3's training needs, and triples
generation time for every sweep cell. S1's F4 statistics were measured at 300 steps and
remain valid as a statement about that horizon; the stale-trace check exists because
`generate_trace.py` never overwrites without `--force`, so a 300-step trace left on disk
would otherwise have been read silently.

### D29 — rho interpolates by sequential picks with a per-pick segment branch
Date: 2026-09-13 | Session: S2 | Status: active

**Decision:** `K = floor(fraction * num_rsus + 0.5)` RSUs degrade. They are picked one at
a time. The first pick is uniform over RSUs. Each later pick, with probability `rho`,
is *correlated* — uniform over the healthy members of segments that already contain a
degraded RSU, or, if none remain, uniform over healthy RSUs in untouched segments (which
opens a new segment) — and otherwise *independent*, uniform over all healthy RSUs. All
random numbers are drawn up front as fixed blocks of `num_rsus` uniforms; pick `i` reads
element `i` only.

So `rho = 0` is uniform sampling without replacement, and `rho = 1` fills one whole
segment before opening the next, giving whole segments plus at most one partial segment.
In between, the expected fraction of degraded pairs sharing a segment rises monotonically
with rho (F10, and a test). The realised number of correlated picks and the pair
concentration are recorded in the sealed ground truth, because the realised correlation
is what an evaluation should be plotted against.

**Alternatives:** (a) mixture at the set level: with probability rho degrade whole
segments, otherwise independent nodes — binary per seed, so intermediate rho would only
change how often each extreme occurs; (b) a latent-Gaussian copula over nodes with
within-segment correlation rho — continuous, but it is exactly the statistical
correlation L5 rejects, and it cannot hold K fixed; (c) degrade whole segments at every
rho and scale severity by rho — confounds correlation with severity; (d) round K up to
whole segments at high rho — lets the realised fraction vary per seed with segment sizes,
confounding the fraction sweep.

**Rationale:** The sequential rule keeps K exact at every rho, so the correlation sweep
changes *which* nodes degrade and nothing else, and it reaches L5's "degradation applied
to a segment" exactly at rho = 1 using the segments S1 built, including the off-geometry
swaps of D27. Fixed-size draws give two properties worth having for free: at one seed,
changing rho alters the set only through the rule, never through a reshuffled stream
(D17), and the degraded set for a smaller fraction is a subset of the set for a larger
one, so the fraction sweep degrades nested sets rather than unrelated ones. Its cost is
quantisation at small K, measured in F10 and raised as a decision request rather than
worked around. The halves-up rounding avoids Python's banker's rounding sending 2.5 to 2.

### D30 — Task model: three types, deadline = allowance + slack x idle compute time, capped
Date: 2026-09-13 | Session: S2 | Status: active

**Decision:** Light / medium / heavy tasks of 0.2 / 1.0 / 3.0 GHz-seconds of compute
(20 / 100 / 300 ms on an idle 10 GHz reference node), shares 0.50 / 0.35 / 0.15, arriving
as a per-vehicle Poisson process at 0.1 Hz thinned to one per step. Deadline =
`min(1.0 s, 0.05 s + slack * idle_time)` with slack uniform in [2, 4] per task. All
parameters were set a priori and not tuned (F7).

**Alternatives:** (a) deadline proportional to compute with no cap; (b) a fixed deadline
per type; (c) deadlines tied to the vehicle's dwell time in coverage.

**Rationale:** A proportional deadline makes a heavy task exactly as easy to meet as a
light one, so degradation would fail every type at the same rate and task type would
carry no information. The cap is the physical statement that a late perception result is
worthless however expensive it was; it makes heavy tasks the tightest in relative terms,
which is why their baseline success (91.8%) sits below light (99.2%). Per-task slack
rather than (b) gives the deadline a distribution, so "met the deadline" is not a
deterministic function of node state and type — without it, the L3 target would be
learnable from load alone. (c) ties outcomes to handoff mid-task, which is S6's scope; S2
does not model a vehicle leaving coverage before its result returns, and says so.

### D31 — Degradation onset is shared by all degraded RSUs and ramps linearly
Date: 2026-09-13 | Session: S2 | Status: active

**Decision:** Every degraded RSU begins degrading at `onset_step` and reaches full
severity `ramp_steps` later, linearly: capacity factor `1 - severity * level` and silent
drop probability `drop_prob * level`, with `level` rising from 0 to 1. Severity 0.6 and
drop probability 0.15 at full degradation. The onset is the same at every rho.

**Alternatives:** (a) instantaneous onset; (b) per-node random onsets; (c) onset shared
within a segment but staggered across segments, or staggered within a segment as a fault
spreads.

**Rationale:** (a) is ruled out by the S2 brief: if a node's own evidence collapses the
instant it degrades there is nothing for neighbourhood evidence to anticipate, and D
cannot beat C by construction. A regression test asserts the degraded mean
`success_ewma` has not fallen 0.1 below its pre-onset value 10 s after onset and is still
above its pre/post midpoint a quarter of the way into the ramp. (b) and (c) would make
rho change *when* nodes fail as well as *which* nodes fail — temporal correlation would
move with spatial correlation, and a gap-vs-rho curve could not say which of the two the
GNN exploited. A shared onset keeps rho a purely structural knob. It also means any
D-over-C advantage at high rho comes from pooling weak simultaneous evidence across
segment neighbours rather than from neighbours failing first — the weaker and more
defensible version of the claim. Staggering (c) is the obvious next lever if pooling
alone proves insufficient, and would be a decision of its own.

### D32 — Execution model is closed-form; true load is coverage demand and ignores dispatch
Date: 2026-09-13 | Session: S2 | Status: active

**Decision:** Completion time = round-trip access latency + `cycles / (C * capacity_factor
* (1 - 0.6 * load))` times a mean-one lognormal (sigma 0.25). True load is S1's coverage
demand (`in-range vehicles / 10`, clipped), unchanged in meaning, moved from `graph.py`
into `execution.py`. No discrete-event queue; dispatched tasks do not add to load.

**Alternatives:** (a) a discrete-event FCFS queue per RSU in which offloaded tasks consume
capacity; (b) an M/M/1 sojourn time with its pole at load 1; (c) true load as coverage
demand plus in-flight offloaded work.

**Rationale:** At about 0.3 tasks per second per RSU and 20-300 ms of compute each,
offloaded work is a few percent of a 10 GHz node's capacity, so (a) and (c) change load
by a small amount while making the world's behaviour a function of the dispatch policy
under test — every variant would then face a different world, confounding the ablation.
(b)'s pole makes every saturated node infinitely slow, and since coverage demand
routinely saturates it would turn load into a binary. The load sensitivity of 0.6 keeps a
fully loaded node at 40% speed. The noise is mean-one so an honest node's advertised
completion time is an unbiased estimate of its true one, and the latency deviation of an
honest node is pure scatter (F7: 0.037 at the end of the run). This is revisited only if
the simple version is shown to fail (Standing Rule 3).

### D33 — S2 dispatches trust-agnostically with the Baseline A rule (provisional)
Date: 2026-09-13 | Session: S2 | Status: active, pending decision request
Status resolved by D41 (Q2): confirmed for S3 training data; closed loop for H1/H3.

**Decision:** Every task in an S2 scenario is dispatched by the L1 rule with alpha = 0 over
the vehicle's in-range present RSUs, using advertised load and normalised access latency
with the configured beta and gamma. The policy name is recorded in the observable file.
`baseline_a` is the only policy implemented; any other value is refused.

**Alternatives:** (a) nearest RSU; (b) uniform random over candidates; (c) closed-loop
trust-aware dispatch with a model in the loop.

**Rationale:** Something has to choose an RSU for an outcome to exist. Of the
trust-agnostic options, Baseline A is already a locked part of the project (L2), so it
introduces no new policy, and because it reads advertised load it is the policy a
colluder's lie actually works against. (a) ignores load and would let colluders go
unexploited. (b) gives the most even coverage of node behaviour and is a real candidate
for training data. (c) needs a trained model, which does not exist until S3, and whether
to use it is exactly the choice the S2 brief reserves: outcome statistics, feature
trajectories and L3 labels all depend on the dispatcher (F9). This entry records what S2
did, not a settled answer.

### D34 — success_ewma counts broken promises; latency_dev is relative excess over the promise
Date: 2026-09-13 | Session: S2 | Status: active

**Decision:** A node advertises its load. Its promised completion time for a task is the
nominal model at that load plus the vehicle's measured round-trip latency. After the
outcome, `success_ewma` is updated with 0 if the promise was inside the deadline and the
task missed it, and 1 otherwise. `latency_dev` is updated with `max(observed - promised, 0)
/ promised`, capped at 2 and rescaled to [0, 1]; a timeout uses the deadline as its
observed time. Both are EWMAs with weight 0.1, neutral at 1 and 0 before any observation,
and a feature row at step t contains only outcomes observed by time t * dt.

**Alternatives:** (a) `success_ewma` as the EWMA of raw deadline success; (b) signed
latency deviation; (c) a timeout counted at the maximum deviation; (d) a tolerance band
around the promise before it counts as broken.

**Rationale:** (a) is the reading of the glossary's "EWMA of recent task completions"
that L8 explicitly forbids — it penalises an honest node that says it is busy, and it
would duplicate the L3 target as an input. Counting only promises made and broken is the
direct statement of D8. A degraded node advertises nominal capacity because it does not
know it is degraded, and a colluder advertises a low load deliberately; both break promises,
an honest busy node does not. (b) would let an honest node's fast tasks cancel a slow
node's lateness in an average of scatter. (c) invents information: the vehicle knows only
that the task took at least until the deadline, so the lower bound is the honest value.
(d) adds a parameter the discrete promise/deadline comparison does not need; runtime
scatter already makes an honest node's `success_ewma` sit near 0.97 rather than 1, which
the model has to learn around either way. The causal ordering is verified by a test that
replays the task stream through a fresh tracker and reproduces every feature row exactly.

### D35 — Ground truth is sealed by a separate package, a separate file, and a default-deny import test
Date: 2026-09-13 | Session: S2 | Status: active

**Decision:** Behaviour class, degradation onset and level, per-segment degradation,
collusion membership, cold-start plan, true load, and true task completion times live in
`trustgraph.sealed` (`GroundTruth`, and the injector that creates it). The simulator
writes two files per scenario: `<stem>.observed.npz`, whose loader rejects any key it
does not expect, and `<stem>.SEALED.npz`. `tests/test_sealing.py` builds the package's
import graph with `ast` — relative imports, imports inside functions, submodules imported
by `from x import y`, and implicit parent packages — and asserts that no package module
outside the allowlist `{trustgraph.simulator}`, and no script outside
`{generate_scenario, s2_calibration, s2_report, s2_rho_quantization}`, can reach
`trustgraph.sealed` directly or transitively. A subprocess then imports every
non-allowlisted module and asserts nothing under `trustgraph.sealed` was loaded, with a
control import proving the check can fail. `config.py` deliberately does not validate the
degradation sections, because importing the injector there would put the seal on the
training path.

**Alternatives:** (a) a single scenario object with a "do not use" field; (b) a runtime
assertion at the loss boundary; (c) an allowlist of forbidden importers (e.g. modules
named `train*`) rather than an allowlist of permitted ones.

**Rationale:** D4 requires the barrier to be structural. (a) is one attribute access from
a leak. (b) catches a leak into the loss but not into a feature. (c) protects only modules
someone remembered to name; default-deny covers S3's trainer the moment it is created, and
the only way to give a module ground truth is a visible edit to the allowlist. True load
and true completion times are sealed along with the class because they leak it: a
colluder is exactly a node whose true load differs from its advertised load, and a
dropped task is exactly one with infinite true completion time. `true_load` is not a
behaviour label, but storing it observably would let a stray join recover collusion
membership.

### D36 — Colluders are drawn from healthy RSUs anywhere; cold-start RSUs are uniform
Date: 2026-09-13 | Session: S2 | Status: active
SUPERSEDED BY D38 for cold-start placement in S4's conditions (uniform remains the
generator default). The collusion half of this entry stands.

**Decision:** Collusion: `num_groups` groups of `group_size` RSUs, taken in a fixed random
permutation order from the RSUs not already degraded, each advertising `(1 - under_report)`
times its true load and queue depth for the whole run, and executing honestly. Class
`compromised`. A node is never both degraded and colluding. Cold start: `num_nodes` RSUs
chosen uniformly (independent of class), absent — no edges, no tasks — until `join_step`,
then present with neutral features. Both mechanisms are off in the demo config; each has
its own RNG stream.

**Alternatives:** (a) colluding groups aligned with a backhaul segment (a compromised
software build); (b) colluders that also execute badly; (c) cold-start nodes placed
preferentially on degraded segments; (d) keeping absent RSUs out of the graph entirely.

**Rationale:** TECHNICAL_AND_NOVELTY.md 3.2 describes collusion as "individually
plausible but structurally anomalous relative to their neighbourhood", which requires
colluders to sit among honest neighbours whose advertisements they undercut; (a) would
surround each colluder with others telling the same lie. Keeping execution honest (not
(b)) makes the advertisement the only thing wrong, so collusion is detectable only
through L8's discrepancy, and it stays separable from degradation in the evaluation.
Uniform cold-start placement (not (c)) is the neutral default for the generator; whether
S4's cold-start condition should place joiners on degraded segments — the case where a
propagated prior can help — is a question for the decision request. (d) would shift every
RSU index when a node joins, breaking the RSUs-first layout D16 and the selection path
rely on; an absent RSU keeps its row and simply has no edges. A cold-start RSU joins with
the same neutral features as a node with a perfect record, because the locked feature set
has no observation count; the decision request raises this too.

### D37 — The graph reads the observable scenario; graph.py no longer derives node state
Date: 2026-09-13 | Session: S2 | Status: active
Supersedes D24 for the features listed there.

**Decision:** `SnapshotBuilder` takes the observable scenario's per-step RSU feature block,
RSU presence mask, and per-vehicle task demand, and writes them into `x` verbatim. It no
longer computes `load` or `queue_depth` from coverage and no longer holds placeholder
constants; the two remaining placeholders (`cert_valid`, `uptime_stability`) are defined
in `tracking.py` and written by the simulator. RSU-RSU edges exist only between present
RSUs, and their `link_age` counts from the later endpoint's arrival. `run.py` reads the
scenario from disk and fails with an actionable error if it is missing, as it does for
the trace (D23).

**Alternatives:** (a) keep computing true load in `graph.py` and add advertised values as
extra columns; (b) have the graph builder call the simulator.

**Rationale:** Under L8 the `load` feature *is* the advertised value (PROJECT_SPEC.md 5.1),
so there is no column for true load to occupy, and adding one would put collusion
membership in the feature matrix (D35). (b) would make the graph builder — which training
imports — import the sealed ground truth. With the builder reading only the observable
file, the graph a model sees is by construction something a deployment could have
produced, and a test asserts each graph's RSU block equals the stored row exactly.

### D38 — Cold start is evaluated as two conditions: control on healthy segments, test on degraded ones
Date: 2026-09-13 | Session: S2b | Status: active
Supersedes the cold-start half of D36 for S4. Adjusts the lean of S2 decision request Q5.

**Decision:** `cold_start.placement` selects the pool cold-start RSUs are drawn from, in a
fixed permutation order per draw:
`healthy_segment` (CONTROL) — RSUs on segments containing no degraded RSU, excluding
colluders, so the node is reliable and its segment neighbourhood is healthy;
`degraded_segment` (TEST) — the degraded RSUs themselves, so the node is degraded and
shares its segment with the degradation.
`uniform` stays the default and reproduces S2 bit-for-bit. A draw where the pool is too
small is refused with an "infeasible" error rather than falling back to another pool.
S4 runs control and test as separate conditions under the frozen model, each over many
injection draws (D39). No observation-count feature is added.

**Alternatives:** (a) place cold-start nodes only on degraded segments (the S2 lean);
(b) `degraded_segment` drawing from *every* member of a degraded segment, including its
healthy members; (c) fall back to a different pool when the requested one is empty;
(d) add an observation-count feature so a new node is distinguishable from a perfect one.

**Rationale:** (a) was rejected in review: with every cold-start node on a degraded
segment, "message passing gives a new node a useful prior" is confounded with "this
segment happened to be under test", and a D-over-C gap could not say which it measured.
A control where the propagated prior should say "healthy" and a test where it should say
"degrading" make the two separable — the effect is the difference between them. (b) was
considered and not chosen: under exact-count degradation (D29, Q3) a degraded segment
often still has healthy members, most of them at low rho (F11: 11 healthy RSUs share a
segment with degradation at rho = 0), so (b) would make the test condition mostly a
*false*-prior condition — a real question, but a different one from the one S4 asks.
Under L5 a node on a degrading backhaul shares its fate, which is what drawing from the
degraded RSUs encodes; a healthy-member "misleading prior" condition can be added later
as a third pool if wanted. (c) would silently mix conditions, the confound this decision
exists to remove. (d) stays deferred as instructed: revisit only if C and D fail to
separate on these conditions once measured. F12 records how often each condition is
feasible and how much segment evidence the test node actually has.

### D39 — Injection draws: many degraded/cold-start/collusion realisations per seed, traffic fixed
Date: 2026-09-13 | Session: S2b | Status: active
Implements S2 decision request Q1 (b).

**Decision:** `scenario.injection_draw = k` seeds the three injection streams
(degradation, collusion, cold start) from `"<purpose>#draw<k>"` instead of `"<purpose>"`.
Draw 0 uses the bare names, so every existing scenario reproduces unchanged (verified:
the default demo graph-sequence hash is still `f3f247ac...`). Mobility, task arrivals and
execution noise keep their seed-level streams. Scenario file stems gain `-draw<k>` and
`-cold<n>-<placement>` only when those differ from the default.

**Alternatives:** (a) more seeds instead of draws; (b) redraw traffic along with the
injection; (c) a draw index folded into the master seed.

**Rationale:** F10 showed a seed samples rho's structure very coarsely. Under train-once
(L12) a frozen model can be evaluated on many draws of one seed for the cost of
generation alone, whereas (a) multiplies training too. Keeping traffic fixed across draws
(not (b)) makes draws paired — two draws differ only in which RSUs misbehave — which is
the same property D29's fixed-size draws give across rho. (c) would change mobility and
topology with the draw, turning a draw into a new seed and discarding exactly that
pairing.

### D40 — cert_valid and uptime_stability stay placeholders until S6; S4 is unaffected
Date: 2026-09-13 | Session: S2b | Status: active

**Decision:** The two placeholders of F8 are deliberately deferred, not forgotten. No
revocation/compromise model and no restart process are built before S4. The question is
reopened as the first task of S6 (the session that makes Variants A-E selectable), which
must decide before any A-E comparison is run whether Variant B keeps constant
certificates or gains a revocation model.

**Alternatives:** (a) build a revocation model in S2b or S3, before S4; (b) defer
indefinitely and report B identical to A.

**Rationale:** S4 compares C and D only (CLAUDE_CODE_SESSIONS.md S4); Variant B first
matters for H1 and H3, which are measured in closed loop in S6 and S9. Both C and D
receive the same constant `cert_valid` and `uptime_stability` columns, so the placeholders
cannot favour either side of the H2 comparison — a zero-variance input is ignored equally
by both. (a) would add a mechanism before the simple version is shown to be insufficient
(Standing Rule 3) and before S6 has settled dispatch in closed loop (D41 Q2), which a
revocation model's effect depends on. (b) is not acceptable silently: with every
certificate valid, B's trust term is the same constant for every candidate and B's
decisions are identical to Baseline A's in every scenario (F8). That may even be the
faithful model — CRL revocation takes hours to days, longer than the 20-minute horizon —
but it makes "behavioural trust beats certificate trust" a comparison against A under
another name, and that has to be a stated choice, which is why S6 must make it
explicitly rather than inherit it.

### D41 — Resolution of the S2 decision request
Date: 2026-09-13 | Session: S2b | Status: active

**Decision:** As answered in design review:
- **Q1 (rho quantisation, F10):** adopt (b) + (a). S4 evaluates the frozen models over
  many injection draws per seed (D39), and plots against realised segment concentration
  as well as nominal rho. The 5% fraction point is reported as containing no correlated
  failure.
- **Q2 (dispatch, F9):** adopt (a). S3 trains on open-loop Baseline A data as generated;
  H1 and H3 are measured closed-loop when dispatch reads trust (S6/S9). D33 confirmed.
- **Q3 (whole segments vs exact count):** exact K with a possible partial segment, as in
  D29.
- **Q4 (onset):** shared onset for every degraded RSU at every rho, as in D31.
- **Q5 (cold start):** adjusted, not approved as proposed — two conditions, control and
  test (D38); no observation-count feature unless C and D fail to separate on them.

**Alternatives:** the options listed in the decision request itself.

**Rationale:** Recorded so later sessions inherit the answers from this file rather than
from the conversation that produced them; the reasoning for each lives in D29, D31, D33,
D38 and D39 and in the review answer quoted in D38.

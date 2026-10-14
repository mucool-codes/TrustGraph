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

### D42 — S4 must report infeasible cold-start draws explicitly, never drop them silently
Date: 2026-09-13 | Session: S2b (post-merge review) | Status: active
Confirms D38 as implemented (test nodes are degraded RSUs specifically; no third
"healthy member of a degraded segment" placement unless a later session needs to
separate the misleading-prior question).

**Decision:** Wherever S4 (or any later session) aggregates results for the cold-start
conditions, it reports, per condition and per rho, the number of draws attempted, the
number infeasible, and the number aggregated — alongside the metric, not in a footnote.
Infeasible draws are never silently excluded. At minimum S4 also reports the control-vs-
test comparison restricted to draws where *both* conditions were feasible, so the two
samples are drawn from the same set of scenarios.

**Alternatives:** (a) drop infeasible draws and aggregate the rest; (b) substitute a
different pool when the requested one is empty.

**Rationale:** F12 measured the control condition as infeasible on 11.6% of draws at
rho = 0 (29 of 250) and on none at rho = 1. The infeasible draws are not random: they are
the ones where degradation touched every segment. Dropping them (a) skews the control
sample toward scenarios where degradation happened to be concentrated, which is exactly
the property the rho axis is meant to vary, and would make the control condition's
rho = 0 result quietly less "rho = 0" than the test condition's. (b) is the silent mixing
of conditions D38 already rejects.

### D43 — S4 pre-registration: conditions, metrics, and the H2 verdict rule, fixed before any C-vs-D number exists
Date: 2026-10-13 | Session: S3 (Task 0) | Status: active — **FROZEN. Do not revise after any
C-vs-D measurement.** A change requires a new entry that supersedes this one, and the S4
result already obtained stays reported alongside it.

State of knowledge at writing: no trust model (C or D) has been trained, and no C-vs-D
quantity of any kind has been computed. Everything below is derived from the spec, from
F10-F12 (properties of the *scenarios*, not of any model), and from D29/D31/D38/D39/D42.
This entry does not depend on the Variant D architecture; "D" means whichever Variant D
the S3 architecture decision fixes, which is itself fixed before training.

**1. Conditions that constitute the structural test (nothing else counts toward H2)**

- **Correlation sweep.** rho in {0, 0.25, 0.5, 0.75, 1.0}, degraded fraction 0.20 (K = 4),
  no collusion, no cold start.
- **Collusion.** 1 group of 3 colluders, under_report 0.5 (the config values), at fraction
  0.20, at rho = 0 and rho = 1.0.
- **Cold start.** 1 cold-start RSU joining at step 600, as the control/test pair of D38
  (`healthy_segment` / `degraded_segment`), at fraction 0.20, at rho = 0 and rho = 1.0.
  D42 applies: attempted / infeasible / aggregated draws are reported beside every cold-start
  number, and the control-vs-test comparison uses only draws feasible in *both*.
- **The 5% degraded point is excluded from the structural claim.** At K = 1, rho has no
  effect and there is no correlated failure (F10). It belongs to the fraction sweep (S9)
  and is reported there as containing no correlated failure. The 10% point (K = 2) is
  likewise not in the structural test (every draw is binary, F10).

**2. Seeds and draws**

- Seeds 1, 2, 3, 4, 5 (the seeds of F10/F12). One model per seed per variant, trained on
  mixed conditions and frozen (L12), selected by held-out next-task BCE only.
- Evaluation draws are injection draws 100-149 (50 per seed per cell, D39). Training uses
  draw indices < 100 only; no evaluation draw is ever seen in training. Traffic and
  execution noise are fixed per seed (D39), so training and evaluation share a seed's
  traffic and differ in which RSUs misbehave. This is a stated limitation of the design,
  not something to be fixed after the fact.
- **The unit of analysis is the seed.** Draws within a seed share a model and traffic, so
  they are not independent samples. Each draw yields a paired (D, C) value; draws are
  averaged within a seed; the 95% intervals are Student-t over the 5 per-seed values
  (df = 4, t = 2.776) of the *paired* difference D - C.

**3. Metrics**

- **Primary: diagnostic AUC of trust vs the sealed class.** Per draw: each active RSU's
  trust is its mean over steps **[450, 650)** (the contested part of the ramp: one sixth
  to five sixths of the way through, D31); score = 1 - trust; label = `degraded` (1) vs
  `reliable` (0); compromised and cold-start RSUs excluded from this AUC. The window is
  chosen from the design (D31: the advantage claimed is pooling weak simultaneous
  evidence), not from any model output, because at the plateau both models can be expected
  to be at ceiling (degraded `success_ewma` ~0.3 vs healthy ~0.97, F11) and a ceiling
  comparison can only return a null. **Secondary, reported never gating:** the same AUC
  over the plateau [1100, 1200).
- **Collusion AUC:** trust over **[200, 400)** (before any degradation can act), label
  `compromised` (1) vs `reliable` (0), degraded RSUs excluded.
- **Cold-start AUC:** per seed, over the pooled control and test draws feasible in both,
  score = 1 - (cold-start RSU's mean trust over the 30 steps after it joins, [600, 630)),
  label = test (1) / control (0). Secondary window: 60 steps.
- **Secondary: detection latency.** Per model and seed, tau = the 1st percentile of that
  model's trust over all active RSU-steps in [100, 400) (pre-onset, so tau depends on no
  degradation). A degraded RSU is detected at the first step t >= onset (400) from which
  its trust stays below tau for 10 consecutive steps. Latency = t - 400, in steps; a
  never-detected RSU is censored at 799 and the detected fraction is reported beside the
  mean. Latency never enters the H2 verdict.
- **Realised correlation.** Every draw records its realised pair concentration c (the
  five possible shapes at K = 4 are c = 0, 1/6, 1/3, 1/2, 1; D29/F10). All results are
  reported against realised c as well as nominal rho (D41 Q1).

**4. The H2 verdict rule** (Delta = per-seed mean AUC_D - AUC_C, interval = 95% t over seeds)

Let [lo, hi] be the interval for Delta at **rho = 1.0** in the correlation sweep.

- **WIN (structural)** requires all three:
  - **W1** lo > 0 **and** mean Delta >= 0.02 at rho = 1.0.
  - **W2** at rho = 0 the interval for Delta contains 0, **or** |mean Delta| < 0.02. (If
    D also beats C when failures are independent, the advantage is not structural.)
  - **W3** per seed, regress the per-draw Delta on realised c over all 250 draws of that
    seed (all five nominal rho pooled); the mean of the 5 slopes is > 0 with its t-interval
    excluding 0.
- **WIN-NONSTRUCTURAL:** W1 holds and W2 fails. D is better, but not for the reason H2
  claims; reported as exactly that, not as support for H2.
- **NEGATIVE:** hi < 0 and mean Delta <= -0.02 at rho = 1.0 (C beats D).
- **NULL:** not WIN/NEGATIVE, and hi < 0.05 (an advantage of 0.05 or more is ruled out).
- **NOISY:** anything else — the interval at rho = 1.0 is too wide to either confirm 0.02
  or rule out 0.05. Check order: NEGATIVE, WIN / WIN-NONSTRUCTURAL, NULL, NOISY.
- **Response to NOISY (decided now):** one extension only, to seeds 6-10, same draws, same
  code, same hyperparameters; the verdict is re-evaluated on all 10 seeds (t = 2.262,
  df = 9). Still NOISY at 10 seeds is reported as inconclusive. Draws are not increased in
  place of seeds: they do not reduce seed-level variance.
- **Cold start and collusion are judged separately**, each by W1 alone on its own AUC (at
  rho = 1.0 for cold start; at rho = 1.0 and rho = 0 for collusion, reported separately),
  using the same WIN / NULL / NOISY thresholds. No pooling across conditions and no
  multiplicity correction: the three conditions are named in H2 and are reported as three
  verdicts. "H2 holds in full" means all three are WIN; any other combination is stated
  as the combination it is.

**5. What may not happen after the result**

- No change to the windows, thresholds, seeds, draws, or this rule. No model or
  hyperparameter change between seeds or after seeing S4 output.
- If S4 is NULL or NOISY and a response is wanted (staggered onset D31(c), an
  observation-count feature D38(d), a different architecture), it is a new DECISIONS
  entry and its run is labelled post-hoc; it never replaces the S4 result.

**Alternatives:** (a) non-overlap of the two models' separate 95% intervals (the
illustrative criterion in the S3 brief) — rejected as the primary: C and D share seeds,
traffic and draws, so the paired difference is the correct statistic, and non-overlap of
unpaired intervals at n = 5 mostly measures between-seed spread rather than the gap; the
non-overlap check is still printed. (b) plateau-window AUC as primary — rejected as a
ceiling comparison (above). (c) a threshold-free "gap widens" criterion only — rejected: a
significance-only rule would count a gap of 0.003.

**Rationale:** L12 and the H2 text commit to a gap that is "statistically meaningful
across >= 5 seeds, widening as rho strengthens", which cannot be judged unless "meaningful"
and "widening" are numbers. Fixing them before any model exists is what makes a positive
S4 result credible and a negative one publishable (PROJECT_SPEC.md 4.1: the fallback is an
honest negative result). The 0.02 effect floor stops a statistically detectable but
practically empty gap counting as a win; the 0.05 ruling-out bound is what lets a NULL be
stated as "an advantage of 0.05 is excluded" rather than "no significant difference".

### D44 — Variant D is a 2-layer GINEConv over all four edge features; C matches its depth and width; a C-independent fallback trigger is fixed before training
Date: 2026-10-13 | Session: S3 | Status: active
Resolves the S3 decision request. Amends the layer type of the S0 TrustHead (D1/L1 fix the
role and depth, not the conv type). D43 is untouched.

**Decision:**
1. **Q1 — edge-aware layer.** Variant D uses two `GINEConv` layers with `edge_dim = 4`
   (link_latency, signal_strength, link_age, same_segment), followed by the same scalar
   sigmoid head. Depth stays 2 (L1); the edge type stays single and homogeneous (L6). A
   LayerNorm on the layer output is allowed if training is unstable, applied identically
   in any later variant of D, and recorded if used.
2. **Q2 — capacity matching.** Variant C is an MLP over the RSU feature block only
   (`RSU_FEATURES`, no vehicle block, no position, no neighbour data) with 2 hidden layers
   of width 32 and the same head. Only message passing differs. Both parameter counts and
   D's count as a multiple of C's are reported in FINDINGS.md.
3. **Fallback trigger, fixed before training and independent of C.** After D is trained,
   on the *validation* scenarios only, two checks:
   - **Edge probe:** set `same_segment` to 0 on every edge (and, separately, to 1 on every
     RSU-RSU edge) at inference; the mean absolute change in RSU trust must be >= 0.01.
   - **Loss check:** held-out next-task BCE after training is below the untrained model's
     held-out BCE.
   If D fails either check it is replaced **once** by `GATv2Conv` with `edge_dim = 4`,
   the reason is recorded, and the choice is then frozen. No switch of architecture is
   made on the basis of any C-vs-D number (that is the choice-after-outcome D43 exists to
   prevent), and (c) "same_segment-gated SAGE" is not a further fallback.

**Why SAGEConv was ruled out:** it cannot see `same_segment` (and `forward` never received
`edge_attr`), so D could not distinguish a same-segment from a cross-segment neighbour and
H2 would be untestable rather than false.

**Alternatives:** (b) GATv2Conv with edge_dim=4 as the first choice - more parts, kept as
the single pre-committed fallback; (c) SAGE with aggregation split by `same_segment` -
custom code, no evidence it is needed (Standing Rule 3); matching C by parameter count
instead of depth and width.

**Rationale:** GINEConv is the simplest off-the-shelf layer that exposes every edge
feature. Its sum aggregation scales with degree, but the graph is capped at 15-30 RSUs
(L9) so degree is bounded. Matching depth and width keeps message passing the only
difference between C and D; D has more parameters (edge encoders), which is reported
rather than hidden.

**Limitation queued for the paper (S11):** training and evaluation share a seed's traffic
and differ only in injection draws (D39, D43 section 2).

### D45 — Epoch cap raised from 60 to 200 for both C and D, because the cap was binding (F13)
Date: 2026-10-13 | Session: S3 | Status: active

**Decision:** `TrainSettings.max_epochs` goes from 60 to 200 for BOTH variants. Every other
setting (lr 1e-3, Adam, batch 32 graphs, patience 8, min_delta 1e-4, selection on held-out
next-task BCE) is unchanged. C and D are re-run from scratch under the new budget and F13 stays
as recorded.

**Alternatives:** (a) accept D at epoch 60 - rejected, the S3 exit condition is convergence and
D's best epoch was its last; (b) raise D's learning rate or change its optimiser - rejected, it
would give C and D different settings (S3 Task 7); (c) a larger cap only for D - rejected for the
same reason.

**Rationale:** A cap that binds for one variant and not the other is a truncation of the
slower-learning model, not a convergence criterion. The change uses only the training/validation
BCE curves (F13), not any AUC or C-vs-D comparison, and the patience rule that actually decides
when each model stops is untouched - C stopped at epoch 24 by patience and will do so again, so
the larger cap only releases D. The F13 numbers are not edited.

### D46 — S4 implementation choices D43 left open, and training of the missing seeds
Date: 2026-10-14 | Session: S4 | Status: active
D43 is untouched; these fill gaps in it and were fixed before any C-vs-D number was computed.

**Decision:** (1) Seeds 2-5 had no S3 checkpoints, so each was trained once with `s3_train.py` and
unchanged D45 settings (F15); seed 1's frozen S3 checkpoints are used as they are. (2) The
detection threshold tau is the 1st percentile of a model's trust over [100,400) pooled over the
seed's 250 sweep-condition draws (no collusion, no cold start). (3) A never-detected RSU has latency
399; a draw's latency is the mean over its degraded RSUs. (4) Collusion AUC uses compromised vs
reliable with degraded RSUs excluded; cold-start AUC pools control and test draws feasible in both,
per seed. (5) Evaluation scenarios live under `scenarios/s4/` (collusion files carry `-coll1`).

**Alternatives:** (1) halt and ask before training; (2) tau per scenario or per cell; (3) censor
at the end of the horizon (1199).

**Rationale:** (1) D43 section 2 requires one model per seed and training is deterministic from
(config, seed); the cost of asking was a stalled session, and the choice is reported rather than
hidden. (2) per-scenario tau would let a degraded RSU's own history set its threshold.
(3) matches D43's "censored at 799".

### D47 — Resolution of the S4 decision request: the NEGATIVE stands; no rescue runs
Date: 2026-10-14 | Session: S5 | Status: active
Records the design-review answers to the S4 decision request. F17 is not edited.

**Decision:**
- **Q1:** (a) accept the NEGATIVE (F17) as the pre-registered H2 result. No staggered onset
  (D31(c)), no observation-count feature (D38(d)) and no new D variant (GATv2, LayerNorm): each is a
  scenario-generator or model change chosen after the result exists, i.e. a rescue attempt (D43 section 5).
  S5 runs the two diagnostics of its brief and nothing else. A later study of regimes where a GNN should
  help is allowed only if labelled exploratory and pre-registered before it runs.
- **Q2:** the two collusion verdicts (NOISY at rho 0 and 1) are reported as inconclusive; seeds 6-10 are
  not run for them. The core verdict has a CI excluding 0, so the NOISY extension is not triggered
  for it. The models never saw colluders in training, so collusion is a stress test, not H2 evidence. If a
  later session trains seeds 6-10, collusion is evaluated on them then, and that is stated in advance.
- **Q3:** D46 accepted. Training seeds 2-5 is recorded as a **deviation from the "no retraining" wording
  of the S4 brief** (read as covering the existing seed-1 checkpoints); it was necessary to execute D43's
  5-seed protocol. Seed 1 was trained at the default torch thread count and seeds 2-5 at 3 threads; the
  inference-side effect is measured in S5 (D48 check 3), without retraining.

**Alternatives:** rescue runs (b)/(c) of the S4 request; extending collusion to seeds 6-10.

**Rationale:** D43 fixed the response to each outcome in advance; changing the generator or model after
seeing the number would make the S4 result unfalsifiable. H1 and H3 compare trust-aware dispatch with
certificate-only and no-trust baselines, do not depend on D beating C, and are tested in the closed loop
(S6, S9).

### D48 — S5 protocol: what each diagnostic measures and what counts as a defect, fixed before any S5 number
Date: 2026-10-14 | Session: S5 | Status: active — frozen once check 2 has been computed
No S5 quantity has been computed when this entry is written. Diagnostics only: no change to the
generator, models, seeds, draws, windows or thresholds of D43, and none to F16/F17.

**Check 1 (are RSU-RSU edges carrying segment membership into message passing?)**
- Structural: on built graphs of every seed, `same_segment` must equal (segment_i == segment_j) on every
  RSU-RSU edge and be 0 on every vehicle-RSU edge.
- Behavioural, per seed's own D checkpoint, on rho=1.0 sweep draws 100-104, steps [450,650) at stride 5:
  mean |change in RSU trust| when `same_segment` is (a) zeroed on all edges, (b) set to 1 on all RSU-RSU
  edges, (c) permuted among RSU-RSU undirected pairs (fixed seed). Also the change in the D43 window AUC
  for (a) and (c). The D44 probe magnitudes (validation, F15) are restated per seed.
- **Defect** = any structural mismatch, or mean |dtrust| < 0.01 for (a) or (c) on any seed (D44's
  threshold). A defect stops the session with a decision request.

**Check 2 (does neighbour evidence lead own-node evidence?)** on the observable `success_ewma`
(the L8 discrepancy feature) of every sweep scenario (5 seeds x 5 rho x 50 draws), dt = 1 s so steps = s.
- RSU v "moves" at t_v = first step t in [400,799] with `success_ewma_v` below theta_v for 10
  consecutive steps, theta_v = 1st percentile of v's own `success_ewma` over [100,400). Never moved:
  censored at 799 and counted.
- Neighbours of v = RSU-RSU neighbours (topology edges) on the same backhaul segment. For a degraded v with
  >=1 degraded same-segment neighbour u: **primary** own-lead = mean_u(t_u) - t_v, in seconds (positive =
  own evidence first); **secondary** = min_u(t_u) - t_v (biased toward neighbour-first because a minimum over
  noisy times is early; reported, not primary). Per-seed mean over all such (draw, v), 95% t-interval over 5
  seeds, per rho.
- Healthy same-segment neighbours: fraction that "move" in [400,799] (the false-alarm baseline of the same
  rule); share of degraded RSUs having a degraded / only a healthy / no same-segment neighbour. Also the
  ramp level (t_v - 400)/300 at which own evidence appears.
- Secondary operationalisation, reported alongside: "moves" = `success_ewma_v` at least 0.10 below its own
  [100,400) mean for 10 consecutive steps.
- **Reading, fixed now:** own-first if the own-lead interval lies above 0; neighbour-first if below 0;
  otherwise simultaneous. Own-first or simultaneous means neighbour evidence does not precede own-node
  evidence in this scenario, so a structural advantage cannot come from lead time (consistent with D31's
  shared onset, which makes degraded neighbours move together by construction).

**Check 3 (thread sensitivity of inference):** re-evaluate the seed-1 checkpoints on all 550 seed-1
evaluation scenarios with 3 torch threads instead of 1, writing to a separate directory, and compare to F16.
"Beyond rounding" = max |trust difference| > 1e-5, or any F16 metric changing at the 4th decimal. No
retraining; the training-thread difference itself cannot be tested without retraining and is stated as such.

**Alternatives:** choosing thresholds after looking at the series; a single threshold-free lag measure
(cross-correlation) - rejected because it would not say when evidence *first* appears.

**Rationale:** the S4 negative has two candidate causes that are checkable without touching the result: a
defect in how the graph carries segment membership (check 1) and a scenario in which neighbour evidence
cannot arrive before own-node evidence (check 2). Fixing the operationalisation first prevents choosing
the threshold that produces the story.

### D49 — H2 fallback framing adopted: a pre-registered negative on graph-based trust, with a stated boundary condition
Date: 2026-10-14 | Session: S5 | Status: active
Applies the S5 brief's fallback and the H2 fallback clause of PROJECT_SPEC.md 4.1. D43 and F17 are untouched.

**Decision:** The paper reports H2 as tested and not supported: a 2-layer GINEConv trust head did not beat a
capacity-matched node-local MLP on the pre-registered diagnostic AUC (F17, NEGATIVE), cold start NULL at rho 1,
collusion inconclusive (D47). The stated boundary condition is F18/F19: in this scenario own-node evidence
appears at the same time as neighbouring degraded RSUs' (shared onset, D31), the segment feature is correctly
supplied and used by D, and C already sits near ceiling (0.98 AUC). The contribution is carried by H1 and H3
(trust-aware vs certificate-only and no-trust dispatch, tested in the closed loop in S6/S9, which do not
depend on D beating C) and by the integrated, explainable formulation (L11). No model, generator, window or
threshold is changed in response to S4 or S5. Any later study of regimes where a GNN should help (e.g. staggered
onset) is exploratory, pre-registered before it runs, and never replaces the S4 result.

**Open for S6 (not decided here):** which trust head feeds Variant E and the closed-loop comparison. Variants C and
D both exist and are frozen per seed (L1/L12); S6 must raise this as a decision request rather than inherit a
default. D40's question (Variant B's certificate model) also remains S6's first task.

**Alternatives:** (a) fix D and re-test; (b) change the scenario until the gap appears; (c) drop H2 from the paper.

**Rationale:** (a)/(b) are post-result changes of the thing the pre-registration fixed. (c) hides a result that
the spec says must be reported. A measured, pre-registered negative with an identified boundary condition is a
defensible finding; the same numbers after a rescue would not be.

### D50 — S6 pre-answered decisions: Variant B, B-oracle, which trust heads, tuning fairness
Date: 2026-10-14 | Session: S6 | Status: active
Resolves D40 and the open item of D49. Written before any S6 code and before any B or closed-loop number.

**Decision:**
1. **Variant B** stays certificate-only: `trust_v = cert_valid`, which is the constant 1.0 for every node in this
   scenario (D40, F8). **B therefore equals A in this scenario, and this is stated as a fact about the scenario,
   not a result:** a backhaul degradation is a performance fault, not a certificate event, and no CRL
   revocation can occur inside a 20-minute run. B is run anyway and the report asserts its decisions are identical to A's.
2. **B-oracle (a labelled sensitivity, not a realistic baseline).** `trust_v = cert_valid_oracle(v, t)`, with
   `cert_valid_oracle = 0` for every faulty RSU from the step its fault begins, with no delay, and 1 otherwise.
   "Faulty" = behaviour class degraded or compromised. A degraded RSU's fault begins at `onset_step` (400; its
   degradation level is positive from step 401); a compromised (colluding) RSU's fault is present from step 0.
   Cold-start RSUs are faulty only if they are also degraded. It reads sealed ground truth by construction, so it
   is world-side (the simulator hands the array to the engine; the engine never imports `sealed`). It is an
   **upper bound** for any detector, never a competing method, and is reported beside B, never in place of it.
   The `cert_valid` input feature seen by C and D is NOT changed by B-oracle (they are trained on constant 1).
3. **Trust heads.** C and D are both reported throughout H1/H3. D is not presented as the headline (S4: C >= D, F17).
   The closed-loop comparison covers A, B, B-oracle, C, D. **E stays D-based** (Layer 2 needs the graph); Layer 1
   works for any head. In S6, E selects exactly as D (explanation lands in S7/S8) and is an alias in config.
4. **Tuning fairness.** alpha/beta/gamma are tuned on ONE tuning seed, **seed 100**, by the same procedure for
   every variant, A included (beta and gamma only). Seed 100 is excluded from all later evaluation; seeds 1-5
   (the S4 seeds) are the evaluation seeds, and seeds 6-10 stay unreserved. C and D for seed 100 are trained once
   with the unchanged D45 settings (same as F15). Procedure, fixed now:
   - scenario: seed 100, rho 1.0, fraction 0.20, injection draws {0, 1, 2}; closed loop, dwell filter and
     handoff on (D51); objective = mean over the three draws of the deadline-success rate over steps [400, 1200)
     (onset to end) of dispatched tasks;
   - grid (full factorial): alpha in {0.25, 0.5, 1, 2, 4}, beta in {0.25, 0.5, 1.0}, gamma in {0.15, 0.3, 0.6}
     for C, D, B-oracle; beta x gamma only (alpha = 0) for A; B is not tuned separately because its trust term is a
     constant (it takes A's weights, and any alpha gives the same decisions);
   - selection: highest objective; exact ties go to the first grid point in (alpha, beta, gamma) ascending order;
   - the whole grid table is recorded raw in FINDINGS.md before the weights are used anywhere; the chosen weights
     are committed in `configs/weights.yaml`. No trust threshold, cutoff or pre-filter is ever added (L1).
5. **Limitation to log (F-entry).** The trust models were trained on Baseline-A (open-loop) dispatch; the closed loop
   changes who receives tasks, hence which RSUs' outcomes and `success_ewma` update. This is a distribution shift
   between training and evaluation. (Correction to the S6 brief's wording: true load in this world is coverage
   demand and does not respond to dispatch, D32, so the shift is in the observation/feedback distribution, not in
   load.)

**Alternatives:** (a) B with a revocation model - not chosen, a backhaul fault is not a certificate event;
(b) presenting D as the headline - contradicted by F17; (c) tuning per evaluation seed - leaks the evaluation.

**Rationale:** B = A is the honest outcome of the scenario and B-oracle brackets how much any certificate-style
signal could help if it were perfect, so a reviewer cannot read B as a strawman or B-oracle as a method.

### D51 — S6 offloading engine: candidate set, handoff rule, closed-loop hook, decision log
Date: 2026-10-14 | Session: S6 | Status: active
Design choices the S6 brief delegates (Tasks 1, 4, 5, 6). None changes the injector, task model, onset ramp or topology.

**Decision:**
1. **Closed-loop hook, not a new generator.** `simulator.generate_scenario` takes an optional `engine`
   (`offload.OffloadEngine`). With `engine=None` the code path is the existing open-loop Baseline A and its output
   is byte-identical (regression: same graph-sequence hash as F11/F12; tested). With an engine, only the dispatch
   decision and the post-execution handoff rule differ; arrivals, execution noise and injection draws are the same
   fixed-size streams, so closed-loop runs of different variants on one (seed, draw) are paired.
2. **Order inside a step.** Task arrivals are drawn first (unchanged); `task_demand[t]` for all offloaders is
   written; then the step's graph is built from the live arrays and the frozen trust head is run once for all RSUs;
   then each offloader dispatches. The graph at step t therefore carries exactly what the training graphs carried.
3. **Candidate set (Task 1).** RSUs that are active and in range, and whose straight-line dwell estimate
   (`graph.dwell_estimate_s`, the vehicle's own speed and heading, observable) is at least that RSU's advertised
   completion time for the task. If no in-range RSU satisfies this, all in-range RSUs are candidates (the vehicle
   still has to try) and the decision is flagged `dwell_fallback`. No candidate in range = unserved, as before.
4. **Handoff (Task 5), relay rule.** The result returns at dispatch time + completion time. Using the trace
   velocity, the vehicle's position at that moment is extrapolated. (a) still inside the serving RSU's coverage:
   no change; (b) outside it but inside an active RSU r' that covers it: the result is relayed r -> r' over the
   backhaul, adding the one-way backhaul latency of `links.py` for the r-r' distance; (c) outside every RSU's
   coverage: the result is lost and the vehicle sees a timeout (observed time = deadline, deadline missed). The
   vehicle cannot tell a lost result from a silent drop, so a (c) outcome is fed to the tracker like any miss.
   `handoff: off` disables this rule and `dwell_filter: off` disables item 3; with both off, closed-loop variant A
   reproduces the open-loop scenario exactly (tested).
5. **Variant trust sources.** A: alpha = 0 (trust unread). B: the `cert_valid` feature column. B_oracle: D50.
   C and D: the frozen per-seed checkpoint's output on the step's graph. E: as D.
6. **Decision log (Task 6).** One JSON line per task that had at least one in-range RSU, schema version 1,
   observable fields only: ids, time, vehicle, vehicle kinematics, task type/cycles/deadline, the weights, every
   in-range candidate with its full RSU feature row, link latency/distance, dwell estimate, trust, trust source,
   the three signed score terms and the total, dwell eligibility; the selected node and runner-up; `dwell_fallback`;
   the outcome (advertised, observed, met, relayed). Sealed quantities (true completion, silent drop, lost-by-
   handoff, behaviour class) go in the run's separate sealed file, joined on `decision_id` (= task index), so
   the log can feed explanation and training code without importing ground truth (L4). Format is jsonl: no
   pyarrow in the pinned environment, and nested candidate lists are natural in it.
7. **Selection rule.** `selection.select` is used unchanged: argmax of `alpha*trust - beta*latency - gamma*load`,
   no threshold on trust anywhere.

**Alternatives:** a separate closed-loop simulator (duplicates 150 lines and risks drift from the open-loop
world); parquet (not installed); dropping tasks whose result is lost (hides a real cost); candidate set = all
in-range (ignores dwell as the brief asks).

**Rationale:** one simulator with one hook keeps the world identical across modes, makes the open-loop/closed-loop
equivalence a testable property, and keeps ground truth out of the observable log.

### D52 — S6 weights adopted: `configs/weights.yaml`, applied unchanged on every evaluation seed
Date: 2026-10-14 | Session: S6 | Status: active
Records the outcome of the D50 procedure (raw grid: F20).

**Decision:** The weights selected on the tuning seed are used as they are, per variant, for every later
evaluation: A (alpha 0, beta 0.25, gamma 0.6), B = A's, B-oracle (0.5, 0.25, 0.6), C (1, 0.25, 0.6), D (2, 0.25, 0.6),
E = D's. They are never re-tuned on an evaluation seed or after seeing an evaluation number; a change needs a new
entry. Tuning seed 100 and the demo draw (seed 100, draw 7) are excluded from the evaluation (the S4 seeds 1-5 remain the evaluation seeds).

**Alternatives:** one shared weight triple for all variants (hides that a variant may need a different alpha);
re-tuning per evaluation seed (leaks the evaluation).

**Rationale:** the surface is flat for the trust-bearing variants (F20: the spread between the best and the
config-default point is 0.002-0.005, inside the draw-to-draw spread), so the exact weights should not be read as
carrying a finding; what matters is that every variant, A included, was tuned by the identical procedure, so no
variant has a handicap the others do not.

### D53 — S7 Layer 1: exact decomposition from the log, the dominance rule, and a log-schema amendment
Date: 2026-10-14 | Session: S7 | Status: active
Amends D51 item 6 (log schema) and fixes Layer 1's wording rules before any explanation is written.

**Decision:**
1. **Source of truth is the decision log.** An explanation reads the weights and every candidate's inputs from the
   logged record, never from the config or `weights.yaml`, so a later weight change cannot alter an old
   explanation. Each candidate's terms are recomputed from the logged inputs as `alpha*trust`, `-beta*latency_norm`,
   `-gamma*load`, in that summation order, and must equal the logged terms and the logged score **exactly**
   (`==`, not approximately); a mismatch, or a selected node that is not the rule's argmax over the dwell-eligible
   candidates (ties to the lowest RSU index), raises `ExplanationError`. This is what "exact by construction" means.
2. **Schema 2.** The S6 log stored the advertised load only as the float32-rounded feature, while the score used
   the float64 value, so the load term could not be reproduced bit-for-bit. Candidates now carry `load` (the value
   the score used) and the schema number is 2. Layer 1 accepts schema 2 only; S6's schema-1 files are gitignored
   artefacts and are regenerated. No other S6 behaviour changes (open-loop equivalence is unaffected).
3. **Every variant.** The same code explains A, B, B-oracle, C, D and E. Trust enters only as `alpha*trust`:
   - alpha = 0 (Variant A): "trust did not influence this decision"; no trust story is told;
   - alpha > 0 but every contender has the same trust term (Variant B, constant certificate): "trust was the
     same for every candidate, so it did not discriminate";
   - otherwise trust is a term like the other two.
4. **Contrastive margin** is winner minus top runner-up (the best-scoring other dwell-eligible candidate). It
   splits exactly into three signed differences `d_trust + d_latency + d_load`. A term is **decisive** if its
   difference is positive and at least **0.6 of the margin** (fixed here, before any explanation was produced).
   Otherwise the decision is **mixed**: the two largest positive differences are named together. Terms that
   favoured the runner-up (negative difference) are always named as outweighed, which is how a winner with
   *lower trust* than the runner-up is explained. A margin of 0 is reported as a tie broken to the lower RSU index;
   a single eligible candidate has no runner-up and is reported as such. RSUs in range but excluded by the dwell
   filter are listed with that reason.
5. **What Layer 1 does not say.** It explains the *score*, not why trust has its value: trust is an input here.
   The logged own-node behaviour record (promise-keeping EWMA, latency deviation) is shown as context, without a
   causal claim. The neighbourhood clause of TECHNICAL_AND_NOVELTY.md 3.4 is Layer 2's (S8) and is not produced.
6. Interface: `trustgraph.explain.explain_decision(record) -> Explanation` (data) and `render_text(explanation)`;
   CLI `python explain.py --decision <id> [--log <file>]`, plus `--survey` to list example decision ids by category.

**Alternatives:** approximate tolerance instead of exact equality (hides schema drift); recomputing weights from
config (a later retune would silently change old explanations); a dominance threshold chosen after looking at
examples; a single sentence with no contrast (the point of L11 is the contrast).

**Rationale:** the selection score is linear, so the winner-runner-up margin is an identity in three terms; testing it
bit-for-bit makes any divergence between what the system did and what it says it did a failing test.

### D54 — S8 pre-registration: Layer 2 attribution, the H4 fidelity test, stability, the neighbour clause, H5 timing, and the merged example
Date: 2026-10-14 | Session: S8 | Status: active — **FROZEN. Do not revise after any fidelity, stability or timing number exists.**
Written before any S8 code ran on the sample; no attribution, fidelity, stability or timing figure exists when this is committed
(only a one-explanation API smoke test on the seed-100 demo scenario, whose numbers were discarded).

**1. What is explained, and how.** The trust head only, never the selection rule (L11). For a logged decision, Layer 2 explains
the trust of the **winner** and of the **top runner-up** at the decision's step: why that RSU's trust is what it is. Variant E is
D-based; C is explained too for the fidelity comparison (it can only attribute to its own features). GNNExplainer from
`torch_geometric.explain` with PyG defaults (100 epochs, lr 0.01), `explanation_type="model"`, `mode="regression"`,
`return_type="raw"` on the head's sigmoid output, `node_mask_type="attributes"`, `edge_mask_type="object"` for D and none for C.
The graph of the decision's step is rebuilt from the closed-loop observable scenario and trace; the rebuilt head's trust must
reproduce the logged trust of every candidate (max difference reported). The mask-initialisation seed is
`run_seed*1_000_003 + decision_id*101 + rsu`, with primary run_seed 0.

**2. Units and attribution.** A *unit* is one `(node, feature)` pair in the receptive field of the explained RSU v - nodes within
2 hops of v (breadth-first on the step's edges, independent of any mask) - using only valid feature slots (an RSU row's 6 RSU
features, a vehicle row's 3 vehicle features; zero-padded slots are not units). A unit's importance is its GNNExplainer node-mask
value. **Top** = the single highest-importance unit (ties to the lowest (node, feature)); **top-3** is used for stability.
**Low pool** = the units in the bottom half by importance within that explanation (inside the receptive field, so units that are
unreachable by construction are not counted as "low"); one unit is drawn uniformly from the pool with `default_rng(20261015+seed)`
in explanation order. A mask cannot see an input that is exactly zero (zero gradient); that blind spot is part of what is tested.

**3. Perturbation and what counts as a change.** One scalar is moved by delta = 0.3 toward the other end of its [0, 1] range
(x -> x + 0.3 if x < 0.5, else x - 0.3), identically for the top and the low unit. *Trust change*: the explained RSU's trust moves
by at least 0.05 (absolute). *Decision change*: after perturbing, trust is recomputed for every RSU on the whole graph and the
rule is re-run on the logged latency and load with the logged weights; the decision changes if the selected RSU differs.

**4. Sample.** Evaluation seeds 1-5, rho 1.0, fraction 0.20, injection draw 100, closed loop with the D52 weights; one run each of
Variant D (D head) and Variant C (C head). Decisions with step >= 400 (degradation present) and at least two dwell-eligible
candidates; 40 per seed drawn without replacement from the sorted decision ids with `default_rng(20261014+seed)`; each explains its
winner and its runner-up: 80 explanations per seed, 400 per head. Both perturbations are applied to the same explanation (paired).

**5. The fidelity criterion** (per head; unit of analysis = seed, as in D43). Per seed: top-perturbation change rate, random-low
change rate, and gap = top - low, over that seed's 80 explanations. Mean gap over the 5 seeds with a 95% Student-t interval
(df 4, t = 2.776). **Primary metric = trust change**: FAITHFUL if the interval's lower bound is > 0 and the mean gap >= 0.20;
WEAK if lower bound > 0 but mean gap < 0.20; NOT SUPPORTED otherwise. **Secondary = decision change** (the H4 wording): same rule
with floor 0.05, reported alongside, never instead. The top rate, the low rate and the gap are always reported together.
The floors are stated assumptions, not derived quantities.

**6. Stability.** GNNExplainer is stochastic, so the primary explanations are repeated with run_seed 0, 1, 2 on all 400
explanations per head. Reported: the fraction of explanations whose top unit is identical in all three runs; mean pairwise
top-unit agreement; mean pairwise Jaccard of the top-3 units; the same for the top *node* (own vs which neighbour). The masks are
called **stable** if the all-three top-unit agreement is >= 0.80, otherwise **unstable**, and the report says which.

**7. The neighbour clause is conditional.** A same-segment neighbour is an RSU joined to v by an RSU-RSU edge with
`same_segment = 1`. Node mass = the sum of importance over a node's units; share = node mass / total mass in the receptive field.
The clause is **included only if** the same-segment RSU neighbours' summed share is >= 0.30; it then names up to two such
neighbours with share >= 0.05, by mass. Direction is measured, not inferred: the named neighbour's RSU feature row is replaced by the
median RSU row at that step and the change in v's trust is reported ("lowers"/"raises" if |change| >= 0.01, else "no measurable
effect"). Own-evidence clause: the highest-importance own feature plus a second if its share of v's own mass is >= 0.25, with
the same occlusion for direction (median RSU value at that step). The rate at which the clause is included is reported plainly,
overall and split by whether the explained RSU is degraded (sealed class, used only by the evaluation, never by the explanation).
If neighbours rarely drive the score, no neighbour story is templated.

**8. H5 timing**, on the same sample (decisions, same steps), per decision, four components separately: (a) trust inference - one
forward pass of the head over the step's graph; (b) selection - `select` over the decision's eligible candidates; (c) Layer 1 -
`explain_decision` + `render_text`; (d) Layer 2 - GNNExplainer for the winner and the runner-up plus clause construction and
occlusions (also per single explanation). Median, mean and 95th percentile in ms, `time.perf_counter`, after a warm-up, one decision
at a time, no other job running; hardware and torch thread count named in FINDINGS. Practicality is judged against a stated
assumption of 100 ms per decision (the order of a vehicular safety-message latency budget): Layer 2 is **synchronous-practical** if
its median per decision is <= 100 ms, otherwise **asynchronous only**.

**9. The merged example** is the first decision in the seed-100 / injection-draw-7 Variant D log with step >= 700 whose runner-up
is a degraded RSU sharing its backhaul segment with at least one other degraded RSU, with at least two eligible candidates, and
with trust the decisive term of Layer 1 (D53). The sealed class is used only to choose the decision. It is shown whether or not the
neighbour clause is supported.

**Alternatives:** thresholds chosen after seeing masks; random units from the whole subgraph as the low pool (unreachable units
make the gap trivially large); one perturbation size per unit tuned to flip the decision; a single GNNExplainer run (hides mask
noise); templating a neighbour clause unconditionally (the rejected "neighbour story the attribution does not support").

**Rationale:** H4 needs a rule that fixes what "top" and "random-low" mean and what gap counts before any rate exists; the
low pool is restricted to the receptive field and the units are perturbed identically so that the gap measures the attribution, not
the perturbation or the geometry. Seed-level intervals match D43; the stability check and the conditional clause keep the explanation
from claiming more than a stochastic mask supports.

### D55 — What Layer 2 may claim, given the H4, stability, clause-rate and timing results
Date: 2026-10-14 | Session: S8 | Status: active
Records the consequences of F24/F25. D54 and the S8 numbers are unchanged.

**Decision:**
1. **H4 is reported as measured:** Layer 2 on the GNN head is *weakly* faithful (trust-change gap +0.16, decision-change gap +0.03; both
   verdicts WEAK against D54's floors), and the C-head result is reported with its structural caveat (the random-low pool is unreadable by C),
   so it is not offered as evidence that C's explanations are better. Top, random-low and gap are always quoted together.
2. **The neighbour clause** is stated as conditional and rare (10% of D explanations; 16% for degraded RSUs), never as the general
   behaviour of the system. The paper may say a graph model *can* name neighbouring evidence and show a real instance; it may not say
   the system routinely does, and it must not use the clause as evidence for the graph claim beyond what these rates support. This is
   consistent with the S4 result and D49's fallback framing.
3. **GNNExplainer's masks are unstable** (top-unit agreement 0.22 for D); every Layer 2 explanation is reported with its run seed
   (seed 0 in this project) and described as one draw. No fix is made in this session: averaging masks over runs, more epochs or a
   different explainer would each be a new, labelled-post-hoc entry evaluated against D54's criterion, never a replacement for it.
4. **Layer 2 is asynchronous only** (307-598 ms per decision on the stated laptop against a stated 100 ms assumption): it is generated
   after the decision, off the dispatch path; Layer 1 (0.09 ms) is cheap enough to run inline.
5. Variant E is D with Layer 1 + Layer 2; E's decisions are D's by construction, so no E-versus-D decision difference exists to measure.

**Alternatives:** tuning epochs/lr or the thresholds until the gap clears 0.20 (choice after outcome); hiding the weak clause rate;
presenting C's FAITHFUL verdict as better explanations.

**Rationale:** H4 asks for rates and a gap; D54 fixed what counts. The honest summary is a measured, modest fidelity with an unstable
explainer and a rarely supported neighbour clause, which is what a reviewer will check first.

### D56 — S9 pre-registration: the sweep grid, the H1/H3 metrics, the comparisons, and the WIN / NULL / NOISY rule, fixed before any sweep number exists
Date: 2026-10-14 | Session: S9 | Status: active — **FROZEN. Do not revise after the first sweep result exists.**
State of knowledge at writing: no S9 harness code exists and no sweep quantity has been computed. Known beforehand and disclosed: F20/F22
(seed 100, the tuning seed, excluded from evaluation) and the cost of ONE closed-loop run (seed 1, draw 149, wall time only, outcome not
read): 1.0 s (A), 1.5 s (D), 1.2 s (C), single thread. A harness bug found before the first evaluation run may be fixed; once any
evaluation draw (seed 1-5, draw 100-149) has produced a result, a change needs a new entry and the earlier result stays reported.
Harness debugging uses draws 900+ only, which are not evaluation draws.

**1. Grid (L12, D12).** Seeds 1-5, each with its frozen S3 checkpoints (C, D) trained once and used unchanged at every point. Injection
draws 100-149 (50 per seed per point; training used draws < 100). Excluded: tuning seed 100, the demo draw (seed 100, draw 7), draws 0-2.
Variants A, B, B_oracle, C, D with the D52 weights from `configs/weights.yaml`, applied unchanged; dwell filter and handoff on (D51);
closed loop. **No retuning of any kind.** E has D's dispatch (D50, D55 item 5) and is not a performance cell: one assertion run per seed
at the shared point checks E's task stream equals D's; E is otherwise measured only for H4/H5 overhead (section 7).
- rho sweep: rho in {0, 0.25, 0.5, 0.75, 1.0} at degraded fraction 0.20. Fraction sweep: fraction in {0.05, 0.10, 0.20, 0.30} at rho 1.0
  ("high" = 1.0, as in D43). The shared operating point is (rho 1.0, fraction 0.20): it is run once, used by both sweeps, and asserted
  (a) at config level, both sweep definitions resolve to the same (rho, fraction) key, and (b) at data level, the shared cell recomputed
  independently (seed 1-5, draw 100, all variants) equals the stored cell bit-for-bit. 8 distinct points x 5 seeds x 50 draws x 5 variants
  = 10,000 closed-loop runs, estimated 3-4 h serial, under an hour on 12 workers. This is below the 12 h threshold, so **no reduction of
  draws** is made.
- The fraction 0.05 point has K = 1 degraded RSU, so no correlated failure; the 0.10 point has K = 2 (every draw binary, F10). Both are
  reported and labelled as such; they are not part of any structural claim (and none is made here: H2 was decided in S4).
- Unit of analysis = the seed (D43): for each seed, the mean over its 50 draws of the per-draw paired quantity; mean +/- sd across the 5
  seeds is reported everywhere; 95% intervals are Student-t over the 5 seed values (df 4, t = 2.776).

**2. Metrics (per run; the sealed class is read only by the evaluation, never by a model).**
- **H3, task success rate post-onset (primary).** Deadline-success rate of dispatched tasks at steps >= 400 (the F20 objective).
  `success_all` is recorded beside it.
- **H1 primary, dispatches to a degraded node between onset and effective avoidance.** n_av = number of post-onset dispatches whose RSU is
  a `degraded` RSU, counted over steps [400, t_avoid). s0 = the variant's own share of dispatches going to those RSUs over [100, 400).
  r(t) = the same share over the 60 steps [t, t+60) (0 if no dispatch). **t_avoid = the first t in [400, 1110] with r(t') <= 0.75 * s0 for
  every t' in [t, t+30]**; if there is none, t_avoid = 1200 (never avoided: every post-onset dispatch to a degraded RSU counts) and the run is
  flagged. The 0.75 is chosen from F22's phase shares (tuning seed: a degraded share falling from 0.21 to 0.13-0.14) so that a 25% cut
  in the degraded share counts as avoidance, and a bar of 0.5 would have censored the trust variants; it is a stated assumption.
  Reported with it: the fraction of runs never avoiding, and **n_win = post-onset dispatches to degraded RSUs over the fixed window
  [400, 1200)**, which does not depend on the avoidance rule.
- **H1 secondary, detection latency (D43's rule, applied to the closed-loop trust).** Per run, tau = the 1st percentile of that run's trust
  over all active RSU-steps in [100, 400); a degraded RSU is detected at the first t >= 400 from which its trust stays strictly below tau for 10
  consecutive steps; latency = t - 400, censored at 399 if never detected (D43's 799). The run value is the mean over its degraded RSUs,
  and the detected fraction is reported beside it. A has no trust and B's is constant, so neither can detect (stated, not a result).
  **Latency never enters the H1 verdict** (D43 treated latency the same way).
- Also recorded per run: post-onset dispatched count (asserted equal across variants of a draw; a violation is counted and reported, not
  hidden), engine counters, realised pair concentration, wall time.

**3. Comparisons (paired per draw, same seed, same draw, same traffic and execution noise; D51).** Primary: **C vs A, D vs A, C vs B_oracle,
D vs B_oracle.** B equals A in this scenario (D50): asserted as identical task streams on every run; B is reported beside A, never as an
extra win. B_oracle is an upper bound, not a method: against it the labels read as "headroom". **Secondary, closed-loop check of H2:
D vs C**, same metrics and rule, labelled secondary, never gating, and it does not revise F17.
- H3 effect: Delta_H3 = per-seed mean of (success_X - success_ref) over draws, in absolute success rate.
- H1 effect: per-seed relative reduction rho_s = 1 - mean_draws(n_av_X) / mean_draws(n_av_ref); the interval is over the 5 seeds' values.
  (A per-draw ratio is unstable; this one is a ratio of seed means.) The same is computed for n_win as a guard.

**4. The rule (D43 style; check order NEGATIVE, WIN, NULL, NOISY).** Let [lo, hi] be the 95% interval of the effect, floor f and bound b:
- H3: f = 0.01, b = 0.025 (absolute deadline-success rate). H1: f = 0.10, b = 0.25 (relative reduction in dispatches).
- **NEGATIVE:** hi < 0 and mean <= -f (X is worse than the reference, or the oracle is at least f better than X).
- **WIN:** lo > 0 and mean >= f. For H1 the n_win reduction must also have lo > 0, otherwise the label is NOISY (guards against an
  avoidance rule that moves the censoring point rather than the dispatches).
- **NULL:** not WIN or NEGATIVE, and hi < b (an effect of b or more is ruled out).
- **NOISY:** anything else.
The f and b are stated assumptions: f is a third of the A-to-C gap seen on the tuning seed (F20, ~0.03) for H3 and 10% of the count for H1.
They are fixed now, not derived.

**5. Where each hypothesis is judged.** H3 is judged on the fraction sweep (rho 1.0), at each of its 4 points, for each primary
comparison; **H3 holds against a reference only if the label is WIN at >= 2 of the 4 points** (PROJECT_SPEC 4.1). The spec's literal
"non-overlapping error bars" is also printed (separate 95% intervals of the two variants' seed means) but does not decide. H1 is judged at the
shared point (rho 1.0, fraction 0.20), for each primary comparison; every other point is reported with its label, descriptively. With 4
primary comparisons x 2 hypotheses there is no multiplicity correction and each is a separate verdict (as D43).
A NOISY outcome is reported as inconclusive. The seed-6-10 extension of D43 would need new frozen checkpoints and is not run; whether to
extend is the DECISION REQUEST's question, not decided here.

**6. Resumability, determinism.** One result file per (point, seed, draw), written atomically (temp file then rename), holding all five
variants' metrics for that paired draw. A restart skips existing files, and refuses to resume if the stored config hash differs. Every run is
reproducible from (configs/demo.yaml, configs/sweep.yaml, weights.yaml, seed, rho, fraction, draw); workers use one torch thread each.
Infeasible or failed (point, seed, draw) tasks are recorded as such and reported per point (attempted / failed / aggregated, D42), never
dropped. The sweep contains no cold-start RSU (cold_start.num_nodes = 0), so no cold-start aggregate exists; the D42 table is nonetheless
printed beside every cell so a future cold-start addition cannot omit it.

**7. H4 and H5 (serial stage after the sweep, nothing else running).** H4: the D54 procedure unchanged (units, delta 0.3, floors 0.20 / 0.05,
D54 sampler, GNNExplainer run seed 0, no stability repeat) applied to Variant D's head at the shared point on **injection draw 101** (D54
used 100): seeds 1-5, 40 decisions each, winner and runner-up. It is a replication on a fresh draw, reported with D54's verdict labels, and
does not replace F24/F25. H5: per-component timing (trust inference, selection, Layer 1, Layer 2) on the same decisions, at the default torch
thread count and at 1 thread, median / mean / p95; Layer 2 judged against D54's 100 ms assumption. Per-run timing inside the parallel sweep is
not reported (workers contend for the CPU); the wall time per cell is only for the cost estimate.

**Alternatives:** (a) a bar of 0.5 for avoidance (rejected: censors the trust variants on the F22 evidence); (b) the fixed-window count as
the only H1 metric (kept only as a guard: the brief asks for dispatches between onset and avoidance); (c) a cross product (L12 forbids);
(d) fewer draws (not needed: the grid costs less than the threshold); (e) per-draw bootstrap intervals (draws within a seed share a model and
traffic, so they are not independent: D43).

**Rationale:** L12 and PROJECT_SPEC 4.1 give H1 and H3 criteria in words ("reduction", "consistent improvement with non-overlapping error
bars at >= 2 points") that cannot be judged unless the metric, the floor and the reference are numbers; fixing them before the run is what
lets a null be reported as a null. B_oracle brackets how much any certificate-like signal could add, so a reviewer can read A-vs-C/D as a
gain over no trust and B_oracle-vs-C/D as the remaining headroom, and B (equal to A here) is not mistaken for a strawman win.

### D57 — S9: the H1 effect is normalised by Variant A's count, because the oracle's count is zero by construction (amends D56 section 3 only)
Date: 2026-10-14 | Session: S9 | Status: active — **FROZEN with D56. Written before any evaluation draw (seed 1-5, draw 100-149) produced a result.**
Supersedes the H1-effect formula of D56 section 3 ("rho_s = 1 - mean(n_av_X) / mean(n_av_ref)"). Everything else in D56 stands, including the
metrics, f = 0.10 and b = 0.25, the check order and the n_win guard.

**What happened.** The harness smoke test (configs/sweep_smoke.yaml: five seeds, draws 900 and 901 only, which D56 reserves for harness
debugging; nothing from it is a result) crashed the analysis with a zero reference count. Cause: B_oracle's trust is 0 for every faulty RSU from
the step its fault begins (D50), so its dispatch share to degraded RSUs is under the avoidance bar from the first window, t_avoid = 400 and
n_av = 0 in nearly every cell. The ratio X / ref against the oracle is therefore undefined (a zero denominator) or enormous, which is a flaw in D56's
formula, not a property of the variants. The same smoke output was NOT used to change any floor, bound, window, ratio or other rule, and is not
reported as a finding.

**Decision.** The H1 effect is the per-seed reduction **in units of the no-trust baseline's count**:
`rho_s = (mean_draws(n_ref) - mean_draws(n_X)) / mean_draws(n_A)`, with the 95% t interval over the 5 seeds. Positive = X sends fewer tasks to
degraded RSUs than the reference. For ref = A this equals D56's `1 - n_X / n_A` exactly, so the vs-A comparisons are unchanged. Against B_oracle
the effect is non-positive when the oracle is better (here: by that fraction of A's count) and labels as D56 section 4 (NEGATIVE: the oracle is at
least f = 0.10 of A's count better; NULL: within b = 0.25 of A's count of the oracle). The n_win guard uses the same normalisation with n_win.
A seed whose A count is zero is an error (undefined), not dropped.

**Alternatives:** (a) an absolute difference in dispatches with new absolute floors (arbitrary scale, no link to the baseline); (b) dropping the
oracle comparison for H1 (the addendum asks for it); (c) adding a constant to the oracle count (an invented smoothing).

**Rationale:** the oracle comparison answers "how much of the available headroom does the learned trust recover", and the baseline's count is
the natural unit of that headroom; the vs-A rows stay identical to D56 so no reader has to re-derive them.

### D58 — S10 figures: one script, no hand edits, and the presentation choices that could be read as claims
Date: 2026-10-14 | Session: S10 | Status: active
The S9 decision request (Q1-Q4) had not been answered when S10 started. The presentation below follows the stated leans and changes no S9 number, label or rule.

**Decision.**
1. **One command, one script:** `python scripts/make_figures.py` writes figures 1-6 (SVG and PDF, plus a PNG preview) and the tables (md, csv, tex). Every plotted or tabulated number is read from `results/s9/` (results_table.csv, paired_comparisons.csv, summary-verdict.json, overhead.json) or `results/s4/summary-5seeds.json`; the two closed-loop inputs (Figure 2 timeline, Figure 6 worked example) are built by `scripts/s10_data.py` (sealed-aware, allowlisted) and cached under `results/s10/`. Output is byte-identical across runs (fixed SVG hash salt, no dates in metadata). Generated figures and tables are not committed (CLAUDE.md).
2. **C and D shown side by side everywhere (S9 Q1, lean c)**; C is not hidden and D is not presented as the winner. Figure 5 and Figure 4's "C vs D: the structural test" band show the pre-registered negative.
3. **Both H1 counts plotted (S9 Q4, lean c):** the avoidance-bounded n_av and the rule-free fixed-window n_win, side by side, because A's 47% "never avoided" makes n_av alone look larger than the rule-free count supports (F27 item 3).
4. **Figure 2** shows the same story twice: one illustrative scenario (seed 100, draw 7 - the F22/F25 demo scenario, a tuning seed and not an evaluation result, labelled as such) and the evaluation aggregate (seeds 1-5 x draws 100-149 at the shared point, mean +/- sd over seeds). The single-scenario trust lines are drawn as a 15 s moving average for legibility (raw per-step trust is noisy); the aggregate and all dispatch shares are unsmoothed (25 s windows).
5. **Figure 4 truncates the success axis at 0.80** and says so on the axis label; the other two panels start at zero. E is drawn hatched with D's values (identical decisions, asserted in S9), B in a lighter A colour; B-oracle is a dashed reference line, never a bar among the variants.
6. **Figure 6** renders the text the system itself produced (`render_merged`) in blocks, plus the score decomposition from the logged candidate terms; it adds no wording. It is the S8 merged example (D54 section 9 rule, not chosen for its looks), in which the neighbour clause is not produced.
7. **Related-work table:** the ten rows are copied from TECHNICAL_AND_NOVELTY.md section 6 for S11 to update. The "This work" structure-aware cell is annotated "built; no measured gain over node-local (F17/F27)" so the table cannot be read as claiming the graph helped.

**Alternatives:** committing the figures (violates the generated-artifact rule); smoothing the aggregate (hides the per-step spread); one H1 panel only; a y-axis from zero for success (hides a 0.05 difference).

**Rationale:** a figure is a claim. Every choice above is either neutral between C and D or marked on the figure, so a reviewer can reproduce each number from the results table.

### D59 — S9b pre-registration: the Gate-C baseline, its tau grid and tuning, the evaluation, and the "weighted beats gate" criterion, fixed before any Gate-C run
Date: 2026-10-14 | Session: S9b | Status: active — **FROZEN. Do not revise after any Gate-C number exists.**
State of knowledge at writing: no Gate-C code exists and no Gate-C run of any kind has been made. Known and disclosed: F20 (the tuning-seed grids of A, B-oracle, C, D), F26/F27 (S9, including C's success and n_win at every fraction point), D56/D57. Gate-C is **one added variant**; nothing in A, B, B_oracle, C, D, E, the scenario generator, the weights file or the S9 harness's existing behaviour changes, and this is checked (section 6).

**Why it exists.** PROJECT_SPEC section 5.5 states that trust enters the rule "as a weighted term, not a pre-filter". That is a design claim, and a reviewer will ask what the weighted form buys over a gate with the same trust head. Gate-C is that comparison. L1 is not amended by this: the project's system remains the weighted rule; Gate-C is a labelled baseline, in the same sense as B-oracle (a baseline that is not a proposed method).

**1. The variant.** Gate-C uses the frozen seed-specific C head (the same trust values Variant C sees at each step). At each decision the candidate set is the engine's usual one (in range, dwell-feasible, D51 item 3). **Candidates with trust < tau are removed**; the L1 rule `alpha*trust - beta*latency - gamma*load` with **C's D52 weights unchanged (alpha 1, beta 0.25, gamma 0.6)** then selects among the survivors, so trust still appears as a weighted term as well as a gate. If **no candidate survives**, the decision falls back to the **un-gated candidate set** (the engine's candidate set: the dwell-eligible in-range RSUs, or all in-range RSUs where the dwell filter itself fell back; this is the reading of "all in-range candidates" that keeps Gate-C's candidate set identical to every other variant's) and is counted as a gate fallback. Reported for every run: the number of decisions where the gate removed at least one candidate, and the **empty-candidate-set (fallback) rate** = fallbacks / dispatched tasks, over all steps and over steps >= 400. The same tau is used on every seed and every scenario (no per-seed tau).

**2. tau grid and tuning (tuning seed 100 only, as D50/D52).** tau in {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} (8 values; tau = 0 is Variant C itself and is not a gate). Scenario: seed 100, rho 1.0, fraction 0.20, injection draws {0, 1, 2}, closed loop, dwell filter and handoff on; objective = mean over the three draws of the deadline-success rate over steps [400, 1200) (the F20 objective); the weights are not tuned (C's, fixed). Selection: highest objective; exact ties go to the lowest tau. The whole grid, with the fallback rate of each point, is recorded raw in FINDINGS.md before the chosen tau is used anywhere; the chosen tau is written to `configs/gate.yaml` and never re-tuned on an evaluation seed. Seed 100's C checkpoint is the F20 one. The seed-100 demo draw (7) is not used.

**3. Evaluation.** Seeds 1-5 with their frozen C checkpoints; injection draws 100-149 (50 per seed per point); the four fraction-sweep points of L12 at rho 1.0: fractions 0.05, 0.10, **0.20 (the shared operating point)** and 0.30. ("The shared point and the 4 fraction-sweep points" is read as these four distinct points: the shared point is one of the fraction sweep's four. The rho-sweep points are not run for Gate-C.) Same harness, draws, seeds, D52 weights, dwell filter and handoff as S9. Gate-C's runs are paired per draw with the **stored S9 Variant C results**; Variant C is also recomputed in the same task and must equal the stored S9 values bit-for-bit (wall time excepted), which checks that nothing changed in the existing variants. Unit of analysis = the seed (D43/D56): per-seed mean of the per-draw paired quantity, 95% Student-t interval over the 5 seed values (df 4, t = 2.776).

**4. Criterion for "weighted beats gate".** Effect = per-seed mean over draws of (success_C - success_Gate-C), deadline success of dispatched tasks at steps >= 400 (the S9 H3 metric); interval over the 5 seeds. Floor f = 0.01 and bound b = 0.025, as D56. Per fraction point, check order NEGATIVE, WIN, NULL, NOISY, with **WIN = weighted beats gate** (lo > 0 and mean >= 0.01), **NEGATIVE = gate beats weighted** (hi < 0 and mean <= -0.01), **NULL** = neither and hi < 0.025 (a weighted advantage of 0.025 or more is ruled out), NOISY otherwise. **Overall verdict, fixed now:** WEIGHTED BEATS GATE if WIN at >= 2 of the 4 points and NEGATIVE at fewer than 2; GATE BEATS WEIGHTED if NEGATIVE at >= 2 and WIN at fewer than 2; MIXED if both have >= 2; otherwise NOT SEPARATED, and the four labels are quoted. The label at the shared point is also reported. No multiplicity correction.
Also reported, never gating: n_win (dispatches to degraded RSUs over [400, 1200)) and n_av for Gate-C beside C and A, the paired C - Gate-C difference in n_win as a reduction in units of A's count (the D57 normalisation, positive = C sends fewer than Gate-C), the Gate-C-vs-A success difference, the fallback rate, and detection latency is not defined for a variant whose trust is the C head's (identical to C's) and is not reported.

**5. What may not happen after the result.** No change to the grid, tau, f, b, the points, the seeds, the draws or the verdict rule; no per-seed or per-condition tau; no weights tuned for Gate-C. If Gate-C wins or loses, that is reported; any further gate design (a soft gate, a gate with different weights) is a new entry, labelled post-hoc, and never replaces this result.

**6. Implementation constraints (checked by tests and by the C recompute).** `Gate_C` is added to `offload.VARIANTS`; with every existing variant the engine's behaviour, log and counters are unchanged (the new counters default to 0 and the log keys are written only for Gate_C). Gate-C's decision log is not supported by the Layer 1/Layer 2 explanation code (its selected node is not the argmax over the dwell-eligible set, which Layer 1 asserts), and no explanation is claimed for it.

**Alternatives:** (a) a gate with alpha = 0 among survivors (a pure filter): rejected as the baseline because the brief asks for the same L1 rule with C's weights, and a pure filter would conflate "gate" with "no trust weighting"; (b) a gate threshold from a quantile of pre-onset trust (like the D43 tau): rejected, it would depend on the scenario, not on one fixed number; (c) tau tuned per evaluation seed: leaks the evaluation; (d) falling back to the single highest-trust candidate: a different rule, not asked for.

**Rationale:** the comparison isolates how trust enters the decision (a threshold plus the same weighted score, versus the weighted score alone) with the head, the weights, the candidates and the data held fixed. Fixing the grid, the single tuning seed, the effect floor and the verdict rule beforehand makes a gate that wins as reportable as a gate that loses.

### D60 — Resolution of the S9 decision request: how the paper reports the heads, H4, the negative on D, and H1
Date: 2026-10-14 | Session: after S9/S10 (design review) | Status: active
Records the review answers to the S9 decision request. D56-D58 and F26/F27 are not edited; nothing is re-run. The S9b decision request (F28-F30, D59) is a separate request and is still open.

**Decision:**
- **Q1 (headline head): both heads shown equally, C named as the strongest result.** D is the graph variant and the text says plainly that it did not beat the node-local head (S4 NEGATIVE, F17;
  closed loop F27 item 4). Nothing in the title or abstract implies the graph helped. This is what D58 already does in the figures.
- **Q2 (H4 decision-change result): report both passes with their exact numbers and call the decision-change result borderline.** S8 (draw 100): +0.033, WEAK. S9 replication (draw 101): +0.0500,
  exactly on the 0.05 floor, which the literal D54 rule (>=) labels FAITHFUL. The mean of the two is about 0.042. Each pass carries its own literal D54 label; the two are not pooled into a new test
  after the fact, and the mean is quoted only as a description. The trust-change criterion is WEAK on both passes (+0.160 and +0.158). D54 stays frozen.
- **Q3 (why D trails C): accept the negative.** No rescue runs (D47 stands). The F21 dispatch-shift (trust heads trained on Baseline-A dispatch, evaluated in the closed loop) and training on closed-loop
  dispatch are recorded as **untested future work** in the paper's limitations, stated as hypotheses.
- **Q4 (H1 presentation): both n_av and n_win, as D58 does.** Every H1 statement in the paper quotes the fixed-window n_win figure and the censoring (never-avoided) rate beside n_av; n_av is
  never quoted alone. At the shared point: C n_av 153 / n_win 506, D 211 / 610, A 627 / 912 (A never avoids in 47% of runs, C and D in 3%, B-oracle in 1%).

**Alternatives:** presenting D as the headline (contradicted by F17 and F27); pooling the two H4 passes into one interval (a post-hoc test); a rescue study of D (D47); quoting n_av alone (A's censoring makes it look larger than the rule-free count supports).

**Rationale:** each choice reports the measured result at its measured strength. The H4 decision-change gap straddles the floor across two honest passes, so "borderline, with both numbers" is what the data supports; the headline and the H1 rules keep the paper from claiming more than F26/F27 show.

### D61 — Gate-C added to the ablation figure and tables as a labelled extra baseline (amends the variant list of D58 items 2-5 only)
Date: 2026-10-14 | Session: S9b (record completion) | Status: active
Adds the S9b result to the S10 outputs. D58 is not edited and every number of the existing variants is unchanged (the regenerated tables reproduce the previous rows exactly).

**Decision:** Figure 4 and the ablation tables show **Gate-C** (frozen C head, candidates with trust below tau = 0.9 removed first, C's D52 weights, un-gated fallback; D59/F28-F30) as a clearly labelled extra baseline:
its own colour, drawn like the other variants, not highlighted (the highlight stays on C vs D), with the caption stating what it is and that it was run at the four fraction-sweep points only. In the tables it appears in the shared-point ablation,
in the all-points table at the four fraction-sweep points, and in the paired-effects table as "C vs Gate-C" with the shared-point D59 label (the D59 verdict itself uses all four points: NOT SEPARATED, F30). Its detection figures are not shown ("not reported", D59 section 4).
Data are read from `results/s9b/raw` and the stored S9 files; `make_figures.py` stops with a message if they are missing. Figure 3 is not changed.

**Alternatives:** leaving Gate-C out of the figures (the paper would then omit the baseline a reviewer is most likely to ask for); showing it highlighted (would present a null as a finding).

**Rationale:** the comparison is part of the record (F29/F30) and the figures must stay current with it, in the same style as the other variants.

### D62 — Resolution of the S9b decision request: how "weighted, not a gate" is framed, no further gate designs, where Gate-C appears (amends D61's "Figure 3 is not changed")
Date: 2026-10-14 | Session: after S9b (design review) | Status: active
Records the review answers to the S9b decision request. D59, F28-F31 and D61's other content are not edited; nothing is re-run.

**Decision:**
- **Q1: "weighted term, not a gate" is framed as a design argument, not a measured gain.** The paper states what the data showed: on the same trust head, gating performed equivalently on success (C minus Gate-C within +/-0.001 at every fraction, D59 verdict NOT SEPARATED, F30).
  The weighted form's benefits are stated as design properties: an exactly decomposable score (Layer 1) and no empty-candidate-set failure mode (the gate fell back on 4-19% of decisions, F29/F30; the weighted rule has no such case).
- **Q2: accept the null. No further gate designs** (no gate on D's head, no other alpha, no soft gate); D59 section 5 stands.
- **Q3: Gate-C appears in Figures 3 and 4 and in the ablation table, labelled as an extra baseline;** not in the timeline (Figure 2) or the S4/S8 figures (Figures 5, 6), where it was not evaluated. This extends D61, which had left Figure 3 unchanged; Figure 3 now shows Gate-C in both panels
  (success per fraction, and its paired gain over A), in its own colour, not highlighted.
- **Tags:** `s9b-complete` stays on `1c8c67e` (the first S9b merge) and is not moved; `s9b-final` is added on `99065b0` (S9b after the record completion, F31 and D61). The D62 / Figure 3 change comes after `s9b-final`.

**Alternatives:** presenting the weighted form as a measured improvement (contradicted by F30); a further gate study (D59 section 5); Gate-C in every figure (it was not evaluated in the timeline or the S4/S8 scenarios).

**Rationale:** the paper may claim only what F30 supports; the exact decomposition and the absence of a fallback case are properties of the design that do not depend on a success gap.

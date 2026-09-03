# Claude Code Session Pack
## Trust-Aware, Explainable Task Offloading for Vehicular Fog Networks using GNNs

**How to use this file.** Run sessions in order, one at a time. Copy the prompt block for the session into a fresh Claude Code session. When a session emits a `DECISION REQUEST`, stop, paste it into the design-review chat, get the answer, paste the answer back into the same Claude Code session, and continue.

**Session map (40 days, review on Oct 12):**

| Session | Days | Task | Branch | Decision point? |
|---|---|---|---|---|
| S0 | 1–3 | Repo bootstrap + walking skeleton | `session/s0-bootstrap` | No |
| S1 | 4–7 | Topology, backhaul segments, mobility | `session/s1-topology-mobility` | Likely |
| S2 | 8–10 | Task model, degradation injector, outcome tracking | `session/s2-scenario-generator` | Likely |
| S3 | 11–14 | Trust models (MLP + GNN) + training loop | `session/s3-trust-models` | Likely |
| S4 | 15–16 | C-vs-D probe sweep | `session/s4-cd-probe` | **Mandatory gate** |
| S5 | 17–18 | Gate response (contents depend on S4 outcome) | `session/s5-gate-response` | Conditional |
| S6 | 19–24 | Offloading engine + decision log | `session/s6-offloading-engine` | Likely |
| S7 | 25–27 | Explanation Layer 1 (analytic, contrastive) | `session/s7-explain-layer1` | No |
| S8 | 28–30 | Explanation Layer 2 (GNNExplainer) + templating | `session/s8-explain-layer2` | Likely |
| S9 | 31–34 | Evaluation harness + sweep runner | `session/s9-eval-harness` | No |
| S10 | 35–36 | Figures and tables | `session/s10-figures` | No |
| S11 | 37–40 | Demo script, slides, DECISIONS.md cleanup | `session/s11-demo-slides` | No |

Each session branches from `main`, commits incrementally, then merges back with
`--no-ff` and tags `s<n>-complete` at its exit condition. You never run a git
command — the sessions handle it. The tags give you a clean rollback point per
session, which matters most at the S4 gate.

Reference verification (the 10 papers + CVE-2026-37555) is a manual task for Day 17 or 18 — not a Claude Code session.

---

## Standing rules block

**S0 is the only session where you paste this block manually.** S0 writes it
into `CLAUDE.md`, which Claude Code auto-loads at the start of every session
thereafter — so from S1 onward you paste only the session's task prompt.

```
STANDING RULES

1. Read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md before doing anything
   else. They contain locked decisions and everything measured so far. Do not
   silently override a locked decision. If one appears wrong, raise a
   DECISION REQUEST rather than changing it.

2. Scope discipline. Do only this session's task. Do not start the next
   session's work even if it seems trivial. If you finish early, improve
   tests and documentation for work already done.

3. Simplicity rule. Before adding complexity (a fancier layer, an extra
   feature, an abstraction), first confirm the simple version is actually
   failing. Do not add complexity pre-emptively.

4. Stop at the exit condition. Show me a runnable state. Do not continue
   past it.

5. You handle ALL git operations yourself — see GIT WORKFLOW below. I make
   no commits manually and will not run any git commands.

6. Maintain two living documents, and keep them distinct:

   DECISIONS.md — what was CHOSEN and WHY. Every non-trivial design choice,
   one entry each: the choice, the alternatives considered, the rationale,
   and the date. This is the review-defense artifact. If a later session
   reverses an earlier decision, add a new entry that supersedes the old one
   — never edit or delete the original. The reversal and its reason are
   themselves evidence of judgment.

   FINDINGS.md — what was MEASURED and what it means. Every empirical
   result: calibration numbers, training diagnostics, sweep outcomes,
   timings, failures. Each entry records the date, the commit hash, the
   config/seed used, the raw numbers, and a one-line interpretation.

   CRITICAL: write the FINDINGS.md entry BEFORE analysing, tuning, or
   attempting to improve a result. A number recorded after three rounds of
   tuning is not the same number. This matters most at S4 — if the C-vs-D
   gap is absent, that must be written down as absent before anyone tries
   to make it appear. Negative results are recorded with the same weight as
   positive ones and are never overwritten.

7. Determinism. Every run must be reproducible from (config file, seed).
   No unseeded randomness anywhere.

8. When you hit a design question you cannot resolve from PROJECT_SPEC.md,
   emit a DECISION REQUEST in the exact format below and STOP. Do not
   guess and proceed. Do not pick "the reasonable default" silently.

GIT WORKFLOW

Repo: https://github.com/mucool-codes/TrustGraph — already cloned locally.
The GitHub CLI (gh) is installed and authenticated. You manage branching,
committing, merging, tagging, and pushing entirely yourself.

At session start:
  git checkout main && git pull origin main
  git checkout -b session/s<n>-<short-name>

During the session:
  Commit in small, logical increments with clear messages — several per
  session, not one at the end. Never commit generated artifacts: mobility
  traces, model checkpoints, results tables, figures, __pycache__, venv.
  Add them to .gitignore. Code, configs, tests, and docs are committed.

At the exit condition, ONLY once the work actually runs and tests pass:
  git push -u origin session/s<n>-<short-name>
  git checkout main
  git merge --no-ff session/s<n>-<short-name> -m "S<n>: <one-line summary>"
  git tag s<n>-complete
  git push origin main --tags
  git branch -d session/s<n>-<short-name>
  git push origin --delete session/s<n>-<short-name>

Rules:
  - Merge only after the exit condition is met. A failing session stays on
    its branch, unmerged, and you tell me why.
  - If a merge conflicts, STOP and raise a DECISION REQUEST. Do not resolve
    conflicts by discarding work, and do not rebase main.
  - Never force-push. Never rewrite published history.
  - The s<n>-complete tags are rollback points. Do not delete or move them.
  - Report the merge commit hash and tag at the end of the session.

DECISION REQUEST FORMAT

===== DECISION REQUEST: S<n> =====
CONTEXT: <3-6 sentences. What you built, where you are, what forced this
         question. Enough that someone who has not seen the code can follow.>
QUESTION 1: <the question>
  OPTIONS: <the options you see, with the trade-off of each in one line>
  YOUR LEAN: <what you would pick and why, in one line>
QUESTION 2: ...
BLOCKING: <yes/no — can you continue on other work while waiting?>
===== END =====

Emit this as plain text I can copy. Keep the whole block under 400 words.
Batch questions — do not emit one request per question if several can wait.
```

---

## S0 — Repo bootstrap and walking skeleton (Days 1–3)

**Goal:** kill integration risk. End-to-end pipeline with fake everything, running from one command.

**Before starting:** put `DEVELOPMENT_GUIDE.md`, `TECHNICAL_AND_NOVELTY.md`,
and this file in the repo root, then start Claude Code from inside the clone.
Nothing else is needed — S0 sets up git config, `.gitignore`, and the branch
workflow itself.

```
You are starting a 40-day solo research project. I will run you in a series of
sessions, one task each. This is session S0.

REPO: you are inside a clone of https://github.com/mucool-codes/TrustGraph.
gh is installed and authenticated. Confirm this before anything else with
`git remote -v` and `git status`. If the remote is missing or wrong, STOP and
tell me — do not run `git init` or add a remote yourself.

You own all git operations for the whole project. I will never run a git
command or make a commit. Follow the GIT WORKFLOW in the standing rules
exactly, starting with this session: branch `session/s0-bootstrap`, incremental
commits, then merge to main and tag `s0-complete` at the exit condition.
Create a .gitignore as one of your first commits, covering Python artifacts,
virtualenvs, and the generated-data directories this project will produce
(traces, checkpoints, results, figures).

PROJECT: Trust-Aware, Explainable Task Offloading for Vehicular Fog Networks
using Graph Neural Networks. Two design documents are in the repo root:
DEVELOPMENT_GUIDE.md and TECHNICAL_AND_NOVELTY.md. Read both fully before
starting.

Those documents describe the intended system, but several of their decisions
have been superseded during design review. Your first task is to write
PROJECT_SPEC.md capturing the LOCKED DECISIONS below verbatim in meaning.
Where PROJECT_SPEC.md conflicts with the two design docs, PROJECT_SPEC.md wins.

LOCKED DECISIONS

L1. The GNN's only job is producing a scalar trust score per fog node.
    Node selection is NOT learned. Selection is the fixed analytic rule
    score(v) = alpha*trust_v - beta*latency_v - gamma*load_v, argmax over
    in-range candidates, with hand-tuned alpha/beta/gamma.

L2. Baseline A (no trust) is that same analytic rule with alpha = 0.
    There is no learned baseline selector anywhere in this project.

L3. Trust is trained SELF-SUPERVISED: predict whether node v completes the
    next task within its deadline, learned from observed task outcomes.
    The injected behavior class is NEVER a training label.

L4. The injected behavior class is sealed for evaluation only. It may be used
    as a printed DIAGNOSTIC (AUC of trust_v against true class, logged each
    epoch) but must never enter a loss function or feature vector. Enforce
    this structurally: keep ground truth in a separate object that the
    training path cannot import.

L5. Correlated failure is implemented via explicit backhaul segments. Each
    RSU gets a backhaul_segment_id at topology build time. Degradation is
    applied to a SEGMENT, not to statistically correlated independent nodes.

L6. RSU-RSU edges are required. Single homogeneous edge type, with a
    same_segment boolean edge feature. NO HeteroData, no heterogeneous edge
    types. This is a deliberate scope cut.

L7. Mobility sits behind a MobilitySource interface. The synthetic
    implementation is the default and is on the critical path. SUMO is an
    optional later swap and is NOT on the critical path. Do not install or
    integrate SUMO in any session unless explicitly told to.

L8. Advertised-vs-observed discrepancy is the trust signal. Nodes advertise
    load/latency; after a task completes or times out, observed outcome is
    compared to what was advertised, and the DISCREPANCY (not the advertised
    value) feeds success_ewma and latency_dev.

L9. Scale: 15-30 RSUs, 50-100 vehicles. Do not scale beyond this. Scalability
    is explicitly out of scope.

L10. No adaptive adversary. Degraded nodes do not strategically game the
     trust score. Stated as a limitation.

L11. Explanation is two layers. Layer 1 is an exact analytic decomposition of
     the linear selection score (contrastive: why the runner-up lost). Layer 2
     is GNNExplainer applied ONLY to the trust head. Fidelity testing (H4)
     applies to Layer 2 only; Layer 1 is exact by construction.

L12. Evaluation uses two 1-D sweeps, not a cross product. Sweep rho (failure
     correlation) at fixed 20% degraded fraction. Sweep degraded fraction at
     fixed high rho. Models are trained ONCE per seed on mixed conditions,
     then frozen and evaluated across all conditions.

YOUR TASK FOR S0

1. Read the two design docs. Write PROJECT_SPEC.md: system summary, the five
   ablation variants (A-E), the locked decisions above, the evaluation plan,
   and a glossary of the feature names used in TECHNICAL_AND_NOVELTY.md
   section 3.1.

1b. Write CLAUDE.md in the repo root. This is auto-loaded into every future
   session, so it must be SHORT — target under 150 lines. It contains:
   - a two-sentence project description
   - the STANDING RULES block from this prompt, verbatim
   - the GIT WORKFLOW block, verbatim
   - the DECISION REQUEST FORMAT, verbatim
   - an instruction to read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md
     at the start of every session
   - a one-line pointer to CLAUDE_CODE_SESSIONS.md for the session plan
   Do NOT duplicate the locked decisions, the feature glossary, or any
   findings into CLAUDE.md — those live in their own files and would bloat
   every session's context. CLAUDE.md holds behaviour and process only.
2. Create DECISIONS.md with an initial entry per locked decision (each with
   choice, alternatives, rationale, date). Create FINDINGS.md with a header
   explaining its purpose and format, and a first entry recording the
   environment: pinned library versions, GPU availability, and whether the
   CUDA path worked. Both files are read at the start of every subsequent
   session.
3. Scaffold ONLY what this session needs. Do not create empty folders for
   future phases.
4. Build a walking skeleton that runs end-to-end with fake components:
   - fake mobility: vehicles random-walking on a grid
   - fake node features: random values, correct shapes and names
   - a 2-layer GraphSAGE trust head, UNTRAINED, forward pass only
   - the analytic selection rule from L1
   - a printed decision sequence
5. Config-driven via a YAML file. Seeded and reproducible.
6. Add a smoke test that runs the whole pipeline on a tiny graph and asserts
   the output shape and determinism across two runs with the same seed.

EXIT CONDITION
`python run.py --config configs/demo.yaml` prints a sequence of offloading
decisions, and running it twice with the same seed produces identical output.
Show me the output. Then stop.

STANDING RULES
<paste the standing rules block here>
```

---

## S1 — Topology, backhaul segments, mobility (Days 4–7)

```
This is session S1. Read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md first.

S0 built a walking skeleton with fake mobility and fake features. This session
replaces the fake topology and mobility with the real thing. It does NOT touch
degradation, tasks, or trust — those are S2 and S3.

YOUR TASK

1. Topology builder:
   - 15-30 RSUs placed on a road layout, positions from config.
   - Each RSU assigned a backhaul_segment_id. A handful of segments, several
     RSUs each. Segment assignment must be spatially coherent (co-located RSUs
     tend to share a segment) — this is what makes the structure learnable.
   - RSU-RSU edges from physical adjacency within a coordination radius, with
     same_segment as a boolean edge feature (L6).
   - Coverage radius per RSU from config.

2. MobilitySource interface (L7) with a synthetic implementation:
   - 50-100 vehicles on waypoint trajectories over the road layout.
   - Realistic speed distribution and resulting dwell time in coverage zones.
   - Traces written to disk. The rest of the pipeline reads traces from disk,
     never generates them live in a loop.

3. Graph constructor: given a timestep of a trace, emit a PyTorch Geometric
   Data object with the node and edge features named in PROJECT_SPEC.md's
   glossary. Behavioral features (success_ewma, latency_dev, uptime_stability)
   exist with correct shape but are placeholder values this session.

4. Two sanity visualisations saved to disk: a graph snapshot with RSUs
   colored by backhaul segment, and a plot of vehicle-RSU edge count over
   time confirming handoff is actually happening.

EXIT CONDITION
Given a config and seed, you can produce a reproducible sequence of graph
snapshots, and the two visualisations look correct. Show me the visualisations
and the summary statistics (mean vehicles per RSU, mean dwell time, handoffs
per vehicle per minute). Then stop.

STANDING RULES
Already loaded from CLAUDE.md. Confirm you have read them, plus
PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md, before starting.
```

---

## S2 — Task model, degradation injector, outcome tracking (Days 8–10)

**This is the session that determines whether H2 is answerable. Watch it closely.**

```
This is session S2. Read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md first.

This session builds the research instrument: the thing that generates the
conditions under which the graph either helps or does not. Its design decides
whether H2 is answerable. Take it seriously and raise DECISION REQUESTs
liberally.

YOUR TASK

1. Task model:
   - Task types light/medium/heavy, each with a compute-cycle requirement.
   - Deadline drawn from a distribution correlated with task type — heavier
     tasks get proportionally longer but not unlimited deadlines.
   - Calibrate so that baseline success rate under no degradation is high but
     not 100% (target roughly 90-97%), and drops meaningfully under
     degradation. Report the calibration you land on.

2. Node execution model: given a task assigned to a node with a given true
   load and true behavior class, determine completion time and success.

3. Degradation injector, parameterised by rho:
   - rho = 0: degraded nodes chosen independently at random.
   - rho high: degradation applied to whole backhaul segments (L5).
   - rho in between: interpolate, in a way you document precisely in
     DECISIONS.md.
   - Degraded-node fraction is a separate config parameter (5/10/20/30%).
   - CRITICAL: degradation onset must be GRADUAL, not instantaneous. If a
     node's own success_ewma collapses the moment it degrades, there is
     nothing left for neighborhood evidence to contribute and the GNN cannot
     beat the MLP by construction. Make onset ramp over a configurable window.
   - Also implement: cold-start nodes (join mid-simulation with no history)
     and colluding groups (advertise lower load than true).

4. Advertised-vs-observed tracking (L8):
   - Nodes advertise load and expected latency.
   - After each task completes or times out, compare observed to advertised.
   - The discrepancy feeds success_ewma and latency_dev. The advertised value
     alone must not be the feature.

5. Ground truth sealing (L4): behavior class, segment degradation state, and
   collusion membership live in a separate object with no import path from
   the training code. Add a test that asserts this.

EXIT CONDITION
From a config and seed you produce: graph sequence + task stream + sealed
ground truth. Show me a plot of mean success_ewma over time for degraded vs
healthy nodes at rho=0 and at high rho, so I can see the onset ramp and the
segment structure. Then stop.

STANDING RULES
Already loaded from CLAUDE.md. Confirm you have read them, plus
PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md, before starting.
```

---

## S3 — Trust models and training loop (Days 11–14)

```
This is session S3. Read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md first.

YOUR TASK

1. Variant C trust head: an MLP over a single node's own features only. No
   message passing. Same input feature set as D, same output, same training
   objective. It must be a fair comparison — if C is handicapped in any way
   other than the absence of message passing, the ablation is worthless.

2. Variant D trust head: 2-3 layer GraphSAGE (or GCN) over the full graph,
   same input features, same output.

3. Self-supervised training objective (L3): predict whether node v completes
   the next task within deadline, from observed outcomes. Binary
   cross-entropy. The injected class is never a label.

4. Diagnostic (L4): each epoch, print AUC of predicted trust against the
   sealed true behavior class. This is monitoring only and must not
   influence training, early stopping, or model selection.

5. Training harness: train on mixed conditions (a spread of rho and degraded
   fraction), one model per seed, frozen afterward (L12). Checkpointing.

6. Report for both C and D on a held-out scenario: loss curve, diagnostic
   AUC, and calibration of predicted trust.

EXIT CONDITION
Both C and D train to convergence on the same data with the same objective,
and you can report their diagnostic AUC side by side on one held-out
scenario. Do NOT run the rho sweep — that is S4. Show me the numbers,
then stop.

STANDING RULES
Already loaded from CLAUDE.md. Confirm you have read them, plus
PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md, before starting.
```

---

## S4 — The C-vs-D probe (Days 15–16) — MANDATORY GATE

```
This is session S4. Read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md first.

This session answers the project's critical hypothesis H2: does message
passing beat a node-local model, and does the gap widen as failure
correlation strengthens? Everything downstream depends on the answer.

YOUR TASK

1. Sweep rho across at least 4 values from 0 to high, at fixed 20% degraded
   fraction, across 5 seeds. Train once per seed per condition set per L12.

2. Report for C and D at each rho:
   - diagnostic AUC of trust vs sealed true class, mean +- std across seeds
   - detection latency: time from degradation onset to trust dropping below
     threshold, mean +- std
   - the C-vs-D gap and whether error bars overlap

3. Separately report the cold-start and collusion conditions, which are the
   other two situations where the graph is claimed to help.

4. Produce the gap-vs-rho plot with error bars.

5. Do NOT tune anything to make the gap appear. Write the FINDINGS.md entry
   with the raw numbers FIRST, before any analysis or interpretation, and
   commit it. Only then write your interpretation. If the gap is absent,
   report it as absent. A clean negative result is a usable finding; a
   tuned-until-positive result is not, and the FINDINGS.md commit history is
   what makes that distinction verifiable later.

EXIT CONDITION
Emit a DECISION REQUEST containing the full numbers and the plot description,
regardless of outcome. This gate is mandatory even if results look good — I
need to see them before you proceed. Then stop and wait.

STANDING RULES
Already loaded from CLAUDE.md. Confirm you have read them, plus
PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md, before starting.
```

---

## S5 — Gate response (Days 17–18) — CONDITIONAL

Do not write this prompt in advance. Its content depends on the S4 outcome. Bring the S4 decision request to the design chat; the reply will specify what S5 does. The three anticipated shapes:

- **Gap widens with rho, ~zero at rho=0** → S5 is a short buffer session (tests, cleanup, docs) and you proceed to S6 early.
- **Gap flat and near zero** → S5 attempts exactly two fixes: verify RSU-RSU edges carry segment membership into message passing, and verify degradation onset is gradual enough that neighborhood evidence leads own-node evidence. If neither works by end of Day 18, the paper pivots to the H2 fallback framing.
- **Gap present but noisy** → S5 raises seed count to 8-10 and re-runs.

Reference verification (10 papers + CVE) happens manually on one of these two days.

---

## S6 — Offloading engine and decision log (Days 19–24)

```
This is session S6. Read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md first.

YOUR TASK

1. Candidate set construction: for a vehicle needing to offload, the in-range
   fog nodes, accounting for coverage and remaining dwell time.

2. The analytic selection rule (L1), with alpha/beta/gamma from config.
   Trust is a WEIGHTED TERM, not a pre-filter or threshold gate. This
   distinction is central to the novelty claim — do not add a trust cutoff.

3. All five variants selectable by config:
   A = alpha 0 (load/latency only)
   B = certificate-only binary trust
   C = MLP trust
   D = GNN trust
   E = D plus explanation (explanation itself comes in S7/S8)

4. Handoff handling: what happens when a vehicle leaves coverage mid-task.

5. Decision log schema — design this carefully, it cannot be reconstructed
   later. Per decision, record: timestamp, vehicle, task type and deadline,
   EVERY candidate with its full feature vector and its individual score
   terms, the selected node, and the eventual outcome. Write to a structured
   format (parquet or jsonl).

6. Rough alpha/beta/gamma tuning on one scenario. Document the procedure.

EXIT CONDITION
The Phase 3 checkpoint from DEVELOPMENT_GUIDE.md: degrade a backhaul segment
partway through a run, and show me trust for those nodes visibly dropping and
dispatch shifting away from them, as a plot. This plot is your review demo.
Then stop.

STANDING RULES
Already loaded from CLAUDE.md. Confirm you have read them, plus
PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md, before starting.
```

---

## S7 — Explanation Layer 1, analytic and contrastive (Days 25–27)

```
This is session S7. Read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md first.

Layer 1 is an EXACT decomposition, not an approximation. The selection score
is linear, so the contribution of each term to each candidate's score is
computable exactly. This layer is cheap, exact, and produces the contrastive
clause ("why the runner-up lost").

YOUR TASK

1. For any logged decision, decompose the score of the winner and of each
   losing candidate into its alpha*trust, -beta*latency, -gamma*load terms.

2. Compute the contrastive margin: for the top runner-up, which term accounts
   for the gap, and by how much. Handle the case where no single term
   dominates.

3. Templating into operator-readable text. Target the structure of the
   example in TECHNICAL_AND_NOVELTY.md section 3.4, minus the neighborhood
   clause (that needs Layer 2, coming in S8).

4. A CLI: given a decision id from the log, print the explanation.

EXIT CONDITION
For any decision in a logged run, `python explain.py --decision <id>` prints
a correct contrastive explanation. Show me three examples, including one
where the runner-up lost on latency rather than trust. Then stop.

STANDING RULES
Already loaded from CLAUDE.md. Confirm you have read them, plus
PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md, before starting.
```

---

## S8 — Explanation Layer 2 and fidelity test (Days 28–30)

```
This is session S8. Read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md first.

Layer 2 explains the trust score itself: why is this node's trust 0.42, in
terms of its own features AND its neighbors. This is what produces the
neighborhood-evidence clause, which a node-local model cannot generate — so
this layer is also evidence for the H2 structural claim.

YOUR TASK

1. Apply GNNExplainer (torch_geometric.explain) to the trust head only, not
   to the selection rule. Get per-feature and per-neighbor importance.

2. Merge Layer 1 and Layer 2 into a single explanation matching the full
   structure of the section 3.4 example, including the clause about adjacent
   nodes on the same backhaul segment.

3. H4 fidelity test:
   - remove/perturb the TOP-attributed feature, measure how often the trust
     score and resulting decision change
   - remove/perturb a RANDOM LOW-attribution feature, same measurement
   - the GAP between those two rates is the result. Report both rates, never
     just the first.

4. Measure and report timing separately for: trust inference, selection,
   Layer 1, Layer 2. Layer 2 is expected to be the expensive one, and finding
   that it is only practical asynchronously is an acceptable, honest result
   (H5).

EXIT CONDITION
A full merged explanation for a real decision involving segment-correlated
degradation, plus the H4 fidelity numbers and the H5 timing breakdown.
Then stop.

STANDING RULES
Already loaded from CLAUDE.md. Confirm you have read them, plus
PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md, before starting.
```

---

## S9 — Evaluation harness and sweep runner (Days 31–34)

```
This is session S9. Read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md first.

YOUR TASK

1. A single sweep runner driven by one config, runnable unattended overnight,
   writing all results to one structured results table. No notebooks, no
   manual steps, no babysitting.

2. Per L12, two 1-D sweeps, NOT a cross product:
   - rho sweep at fixed 20% degraded fraction
   - degraded-fraction sweep {5,10,20,30}% at fixed high rho
   Both across 5 variants and >=5 seeds. They share one operating point —
   assert that the shared point agrees across both sweeps as a consistency
   check.

3. Metrics per run: task success rate, detection latency, tasks dispatched to
   a degraded node between onset and avoidance, per-decision timing by
   component, and the H4 fidelity gap on a sample of decisions.

4. Resumability: if the sweep dies at hour 3, it resumes without redoing
   completed cells.

5. Results table with mean +- std across seeds. Never report single-run
   numbers.

EXIT CONDITION
The full sweep completes and produces the results table. Show me the table.
Then stop.

STANDING RULES
Already loaded from CLAUDE.md. Confirm you have read them, plus
PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md, before starting.
```

---

## S10 — Figures and tables (Days 35–36)

```
This is session S10. Read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md first.

All figures are generated by script from the S9 results table. Nothing is
hand-made or hand-edited. Consistent font sizes, consistent color scheme
across all figures, vector output (SVG/PDF), legible when projected.

YOUR TASK — the five figures from TECHNICAL_AND_NOVELTY.md section 10:

1. System architecture: graph construction -> trust GNN -> offloading ->
   explanation.
2. Trust-score timeline for a degrading segment, overlaid with dispatch
   decisions shifting away. THIS IS THE STRONGEST SINGLE FIGURE — spend
   disproportionate effort here.
3. Task success rate vs degraded-node fraction, all variants, error bars.
4. Ablation bar chart A-E, with C vs D visually highlighted as the
   structural test.
5. A worked explanation example rendered as it would appear to an operator.

Plus the gap-vs-rho plot from S4, cleaned up to the same standard.

TABLES: related-work comparison (from TECHNICAL_AND_NOVELTY.md section 6),
ablation results with mean +- std, per-decision latency breakdown.

EXIT CONDITION
All figures and tables regenerate from one command. Show me each one.
Then stop.

STANDING RULES
Already loaded from CLAUDE.md. Confirm you have read them, plus
PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md, before starting.
```

---

## S11 — Demo, slides, DECISIONS cleanup (Days 37–40)

```
This is session S11. Read PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md first.

This is a college review, not a paper submission. The deliverables are a live
demo, slides, and a defensible account of design choices.

YOUR TASK

1. A demo script that runs one scenario end-to-end in a few minutes and
   shows, in order: normal operation, a backhaul segment degrading, trust
   dropping for those nodes, dispatch shifting away, and a generated
   explanation naming the neighborhood evidence. Also record it as a fallback
   in case the live run misbehaves during the review.

2. Rewrite DECISIONS.md from a running log into a clean summary of key design
   choices and trade-offs, organised by theme rather than chronologically.
   Include the cuts (no HeteroData, no SUMO, no adaptive adversary) with
   their justifications — a defended scope cut reads as judgment, an
   unexplained absence reads as an oversight. Preserve any superseded
   entries in an appendix rather than deleting them.

   Leave FINDINGS.md as the chronological record it is — do NOT tidy it into
   a narrative. Its value at review is that it shows what was measured, when,
   and in what order, including anything that did not work. Add only a short
   index at the top pointing to the entries that back each headline result.

3. Slide skeleton: problem, the certificate-vs-behavior gap, system
   architecture, the four-property novelty claim, ablation results with C-vs-D
   highlighted, a worked explanation, limitations, future work.

4. A limitations slide stating plainly: simulation-only, synthetic behavior
   injection, single-hop offloading, no adaptive adversary, fixed objective
   weights. Reviewers calibrate their trust in everything else from this
   slide.

5. Anticipate the reviewer question "is the GNN necessary, or would a
   per-node score suffice?" and prepare the answer from your actual C-vs-D
   numbers.

EXIT CONDITION
Demo runs reliably, slides drafted, DECISIONS.md clean. Then stop.

STANDING RULES
Already loaded from CLAUDE.md. Confirm you have read them, plus
PROJECT_SPEC.md, DECISIONS.md, and FINDINGS.md, before starting.
```

---

## Notes on running this

- **Paste the standing rules block into every session.** It is the mechanism that stops a session from silently making a decision that should come to design review.
- **S4 is the gate.** If it slips past Day 18, cut S8's Layer 2 rather than compressing S9. A weaker explainability story survives review; an evaluation without error bars does not.
- **If a session emits a DECISION REQUEST that looks like it is asking permission for something trivial**, answer it yourself and move on — the batching rule should mostly prevent this, but early sessions may over-ask while calibrating.
- **Keep DECISIONS.md in every session's context.** It is both the review artifact and the mechanism that stops Day 30 from contradicting Day 8.
- **A failed session should not merge.** If a session cannot meet its exit condition, its branch stays unmerged and you decide what happens next. An unmerged branch is a visible, recoverable state; a half-working merge to `main` is not.
- **The `s<n>-complete` tags are your safety net.** If S5 has to undo an S4-driven change, `git checkout s3-complete` gets you a known-good state without archaeology.
- **Keep CLAUDE.md lean.** It loads into every session, so anything added there costs context on all remaining sessions. Behaviour and process belong there; decisions, findings, and specs belong in their own files. If a session proposes appending substantive content to CLAUDE.md, push back.

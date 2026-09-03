# Development Guide
## Trust-Aware, Explainable Task Offloading for Vehicular Fog Networks using GNNs

> **⚠ Partially superseded — read PROJECT_SPEC.md first.**
>
> This guide is sequenced for *learning*: each phase builds understanding on the last. The project is now sequenced for *risk*, because the critical hypothesis (H2 — does message passing beat a node-local model?) is decided almost entirely by the degradation injection model, which this guide treats as a sub-step of Phase 3.
>
> Where this guide and `PROJECT_SPEC.md` conflict, **PROJECT_SPEC.md wins**. The main divergences:
>
> - **The C-vs-D probe moves much earlier**, to roughly day 15, before SUMO and before explainability. Deferring it to Phase 5 risks discovering a null result with no time to respond.
> - **The scenario generator is built before the model**, as a first-class parameterised instrument (backhaul segments, correlation parameter ρ, cold-start arrivals, colluding groups) rather than as scenario fixtures.
> - **Phase 2's learned baseline selector is dropped.** Node selection is never learned. The GNN's only job is producing a trust score; selection is a fixed analytic rule, and Baseline A is that same rule with α = 0.
> - **Trust is trained self-supervised** on observed task outcomes. The injected behaviour class is sealed and used only as an evaluation diagnostic.
> - **SUMO is off the critical path**, behind a `MobilitySource` interface with a synthetic default.
> - **Evaluation uses two 1-D sweeps, not a cross product** (ρ at fixed 20% degraded fraction; degraded fraction at fixed high ρ).
>
> Everything else below — the phase checkpoints, the ablation table, the statistical-rigour requirements, the general development practices — still applies and is worth reading in full.

This guide walks through building the project step by step, phase by phase. Each phase produces a working, testable increment — don't move to the next phase until the current one runs end-to-end, even if the results aren't good yet. A working simple version beats a half-built complex one, especially with a review deadline.

---

## Phase 0 — Environment Setup

**Goal:** Get all tools installed and talking to each other before writing any real logic.

1. Install Python 3.10+ and set up a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install core libraries:
   ```bash
   pip install torch torch_geometric networkx numpy pandas matplotlib plotly
   ```
3. Install SUMO (Simulation of Urban Mobility) — follow the OS-specific installer from SUMO's official site, then verify:
   ```bash
   sumo --version
   ```
4. Set up a project folder structure:
   ```
   project/
     data/           # generated mobility traces, synthetic behavior logs
     graph/          # graph construction code
     models/         # GNN model definitions
     trust/          # trust scoring module
     offload/        # offloading decision engine
     explain/        # explainability module
     eval/           # evaluation scripts and metrics
     notebooks/      # exploration and debugging
     DECISIONS.md    # log every non-trivial design choice and why
   ```
5. Create a `DECISIONS.md` file now and start logging choices as you make them (e.g., "chose SUMO over iFogSim because X"). This becomes invaluable when your reviewer asks "why did you do it this way."

**Checkpoint:** You can import `torch_geometric` without errors, and `sumo` runs from the command line.

---

## Phase 1 — Mobility Simulation and Graph Schema

**Goal:** Generate a realistic, moving vehicular network and define what a "graph snapshot" of it looks like.

### Step 1.1 — Build a simple SUMO scenario
- Create a small road network (SUMO's `netedit` tool, or use one of SUMO's bundled example networks to start).
- Define a handful of routes and inject a modest number of vehicles (start with ~20–30 vehicles — you can scale up later).
- Run the simulation and export vehicle positions over time (SUMO's TraCI Python API lets you pull live position data step by step).

### Step 1.2 — Place fog nodes (RSUs)
- Decide on fixed RSU positions along your road network (e.g., every few hundred meters, or at intersections).
- Each RSU has a coverage radius — a vehicle is "connected" to an RSU if it's within that radius.

### Step 1.3 — Define your graph schema
Decide the exact features before writing code — this saves a lot of rework later.

- **Node features (per fog node):** current compute load, certificate validity flag, rolling behavioral history score (starts neutral, updated in Phase 3).
- **Node features (per vehicle, optional but recommended):** current task demand, position.
- **Edge features:** distance, signal strength/latency estimate, whether the connection is newly formed or established.

### Step 1.4 — Write the graph constructor
- Using NetworkX (for readability) or directly building PyTorch Geometric `Data` objects, write a function that takes a SUMO simulation snapshot and outputs a graph object for that timestep.
- Test this on a single timestep first, then loop it across the full simulation to confirm the graph updates correctly as vehicles move.

**Checkpoint:** You can run your SUMO scenario and, for any timestep, produce a valid graph object with the right nodes, edges, and features. Visualize a couple of snapshots (NetworkX + Matplotlib) to sanity-check that connectivity looks right.

---

## Phase 2 — Baseline Offloading Model (No Trust, No Explainability Yet)

**Goal:** Get a working GNN that takes a graph and outputs an offloading decision — deliberately simple, so you have an end-to-end pipeline before adding complexity.

### Step 2.1 — Define a simple GNN model
- Start with a basic 2–3 layer GNN using PyTorch Geometric (`GCNConv` or `GraphSAGE` are good starting points — simpler than attention-based GNNs, easier to debug).
- Input: node features (load, distance-derived signal, etc.).
- Output: a score per fog-node-candidate for a given vehicle's task.

### Step 2.2 — Define the offloading rule
- For a given vehicle needing to offload a task, gather the candidate fog nodes within range.
- Run the GNN forward pass, take the highest-scoring candidate as the offloading target.
- This is your baseline: essentially "pick the best node by load/latency, learned rather than hand-coded."

### Step 2.3 — Train on a simple proxy objective
- Since you don't have trust yet, train this baseline to minimize a simple cost (e.g., predicted latency + load penalty) using synthetic labels you generate from your simulation (e.g., "the node that actually had the lowest load at that timestep was the best choice").
- This doesn't need to be sophisticated — its only job is proving the pipeline works.

**Checkpoint:** Given a simulation run, your system outputs a sequence of offloading decisions, and you can print/log them. This is your working skeleton — everything else builds on top of it.

---

## Phase 3 — Trust Integration

**Goal:** Add the core novelty — a learned, behavior-based trust score that feeds into the offloading decision.

### Step 3.1 — Simulate node behavior patterns
- Since no real dataset exists for this, you need to synthetically inject behavior into your simulation:
  - **Reliable nodes:** consistent low latency, no dropped connections.
  - **Degraded nodes:** occasional latency spikes or dropped tasks (simulating overload or minor faults).
  - **Compromised/malicious nodes:** deliberately inconsistent or manipulated behavior (e.g., falsely reporting low load, then failing tasks).
- Assign each fog node a "true" behavior class when you set up the scenario — this becomes your ground truth for evaluation later (Phase 5), even though the model itself won't see this label directly.

### Step 3.2 — Build the trust-scoring module
- Add a rolling behavioral history feature per node (e.g., exponentially weighted moving average of recent task success/failure, latency deviation).
- Extend your graph's node features to include this.
- Retrain (or extend) your GNN so that message passing allows a node's trust score to be influenced by its neighbors too — this is what captures "this node looks fine alone, but its neighborhood has been unreliable."

### Step 3.3 — Feed trust into the offloading decision
- Modify the offloading scoring function from Phase 2 to combine: trust score + load + latency, with weights you can tune.
- Start with simple weighted combination before trying anything fancier (e.g., a learned combination layer) — get the simple version working and defensible first.

**Checkpoint:** Run a scenario where you manually degrade or "compromise" a specific node partway through the simulation, and confirm the system's trust score for that node visibly drops and offloading shifts away from it. This is your core proof-of-concept demo.

---

## Phase 4 — Explainability Layer

**Goal:** Make each offloading decision explainable, not just accurate.

### Step 4.1 — Integrate GNNExplainer
- PyTorch Geometric's `torch_geometric.explain` module provides `GNNExplainer` and `PGExplainer` out of the box — start with `GNNExplainer` since it's simpler to set up.
- For a given offloading decision, run the explainer on your trained GNN to get an importance score for each input feature and each contributing neighbor node.

### Step 4.2 — Turn raw attribution into a readable explanation
- Write a small templating function that converts the raw importance scores into a sentence, e.g.:
  *"Node RSU-7 was selected over RSU-3 primarily due to a higher trust score (driven by consistent recent behavior) and lower current load."*
- This translation layer matters as much as the raw explainability output — a reviewer or end user won't read attribution weights directly.

### Step 4.3 — Log every decision with its explanation
- Store each offloading decision alongside its explanation and the raw feature values that drove it — you'll need this for both your evaluation (Phase 5) and your review presentation.

**Checkpoint:** For any offloading decision in a simulation run, you can retrieve a clear, human-readable explanation, and it should make intuitive sense when you check it against the underlying data manually.

---

## Phase 5 — Evaluation

**Goal:** Prove the system does what it claims, against fair baselines.

### Step 5.1 — Define your baselines
- **Baseline A:** Load/latency-only offloading (no trust) — this is your Phase 2 model.
- **Baseline B:** Certificate-only trust (binary valid/invalid, no behavioral component) — mimics current real-world SCMS-style trust.
- **Your system:** Full trust-aware, explainable offloading.

### Step 5.2 — Run controlled scenarios
- Re-run your degraded/compromised-node scenario (from Phase 3) against all three systems.
- Track: how quickly each system detects and routes around the bad node, and how many tasks get sent to it before being avoided.

### Step 5.3 — Compute your metrics
- **Offloading latency** — time to reach a decision.
- **Trust-violation detection rate** — how often a degraded node is correctly down-weighted, compared to your synthetic ground truth from Phase 3.
- **Task success/failure rate** under node degradation.
- **Explanation fidelity** — pick a handful of decisions, remove the top-attributed feature, and confirm the decision actually changes (this validates your explanations aren't just plausible-sounding noise).

### Step 5.4 — Run the ablation study (required for a conference paper)

A results table alone will not survive review. Reviewers need to see *which component* produces the gain. Run these variants under identical scenarios and seeds:

| Variant | Trust | GNN (neighborhood) | Explainability | Purpose |
|---|---|---|---|---|
| A | ✗ | ✓ | ✗ | Baseline — load/latency only |
| B | Certificate-only | ✗ | ✗ | Mimics deployed SCMS trust |
| C | Behavioral, node-local (MLP, no message passing) | ✗ | ✗ | **Isolates whether the graph structure actually matters** |
| D | Behavioral + GNN | ✓ | ✗ | Full trust model, no explanation |
| E | Behavioral + GNN + Explainability | ✓ | ✓ | Full proposed system |

Variant **C is the most important one** and the easiest to skip. If C performs as well as D, your GNN adds nothing over a simple per-node score, and a reviewer will find that. Run it early — ideally at the end of Phase 3, not at the end of Phase 5 — because if the gap is small you need time to strengthen the neighborhood signal (e.g., correlated/colluding node failures, shared-backhaul degradation) rather than discovering it a week before submission.

### Step 5.5 — Statistical rigor
- Run every configuration across **at least 5 random seeds** (different vehicle spawn patterns, different degraded-node selections).
- Report **mean ± standard deviation**, not single-run numbers. Single-run results get desk-rejected.
- Vary the **fraction of degraded nodes** (e.g., 5%, 10%, 20%, 30%) and plot performance as a curve — this shows where the approach helps most and where it breaks down, which is far more informative than one operating point.

### Step 5.6 — Visualize results
- Trust-score timeline for a degrading node, overlaid with the offloading decisions shifting away from it (your strongest single figure).
- Task success rate vs. fraction of degraded nodes, all variants on one plot with error bars.
- Ablation bar chart (variants A–E).
- One worked explanation example, formatted as it would appear to an operator.

**Checkpoint:** You have multi-seed results with error bars, a completed ablation table including Variant C, and at least one figure that makes the core claim visually obvious.

---

## Phase 6 — Documentation and Presentation

**Goal:** Package everything for review and future publication.

1. Write up your methodology clearly — reuse the structure from your BRD (problem, novelty, architecture, evaluation).
2. Prepare 3–4 key visuals: graph snapshot, trust-score-over-time plot, comparison bar chart, one example explanation output.
3. Update `DECISIONS.md` into a clean summary of key design choices and trade-offs — reviewers respond well to seeing that you understand *why* you made each choice, not just what you built.
4. Prepare a short demo script: run one scenario live (or via a saved recording) showing a node degrading and the system reacting and explaining itself.

---

## Phase 7 — Conference Paper Preparation

**Goal:** Convert the working system and evaluation results into a submittable paper.

1. **Pick a target venue early** (before finishing Phase 5, if possible) — venue page limits and formatting requirements (IEEE conference template vs. ACM template) affect how much detail you can include, and some venues have specific tracks for fog/edge/vehicular networking that are a better fit than a general AI venue.
2. **Draft the paper structure in parallel with Phase 5**, not after:
   - Abstract, Introduction, Related Work (this maps directly to your novelty positioning table), System Model, Methodology (Trust Scoring, Offloading Decision, Explainability), Experimental Setup, Results, Discussion, Limitations, Conclusion.
3. **Reserve space for a limitations section.** Reviewers respond better to authors who state their own boundaries (simulation-only validation, synthetic behavior injection, single-hop offloading) than to authors who let a reviewer discover them.
4. **Keep result figures paper-ready as you generate them in Phase 5** — same font sizes, consistent color scheme, vector formats (SVG/PDF) rather than raster where possible, since conference formatting often penalizes blurry raster plots.
5. **Have your related-work table (from the BRD, Section 5.3) double as your paper's related work section skeleton** — it's already structured as a comparison table, which many venues favor for clearly stating a novelty gap.

---

## General Development Practices Throughout

- **Commit often, in small increments.** Each phase above should be several commits, not one giant one at the end.
- **Keep a simplicity rule:** if you're about to add complexity (a fancier GNN layer, a new feature), first ask whether the simple version is actually failing — don't add complexity pre-emptively.
- **Test on small graphs first.** Debugging a 10-node graph is vastly easier than debugging a 200-node one. Only scale up once the small case works correctly.
- **Re-check Phase 3's checkpoint constantly.** The "does trust actually change behavior visibly" test is your core proof of novelty — if that stops working after a change, treat it as a regression, not a minor issue.

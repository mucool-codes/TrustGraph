# Trust-Aware, Explainable Task Offloading for Vehicular Fog Networks using Graph Neural Networks

### Technical Specification, Novelty Analysis, and Target Results

**Document purpose:** This is the reference document for the system's internal workings, its precise novelty claim, and the results the project aims to produce. It is written with conference submission in mind, so it separates what is *claimed*, what is *assumed*, and what must be *demonstrated*.

---

## Part I — System Workings

### 1. The setting

A vehicle in a connected transport network continuously generates compute tasks it cannot fully process onboard: sensor fusion, hazard classification, cooperative perception, path re-planning. These are offloaded to nearby roadside fog nodes (RSUs) rather than the cloud, because the cloud round trip (~50–100 ms) exceeds the latency budget for anything safety-adjacent, while a fog node one radio hop away is typically in the single-digit-to-low-tens of milliseconds.

The vehicle must therefore answer, repeatedly and quickly: *which fog node should receive this task?*

Today that answer is computed from proximity, current load, and estimated latency. Trust is not part of it. Trust was already settled — once — when the node presented a valid certificate.

### 2. Why that separation is the problem

Certificate-based trust (SCMS under IEEE 1609.2.1, or ETSI TS 102 941) answers *"is this node who it claims to be?"* It does not answer *"is this node currently working correctly?"* Those are different questions, and only the first is currently asked.

A node can hold a perfectly valid certificate and still be:

- **Overloaded** — accepting tasks it cannot complete within their deadline.
- **Faulty** — degraded hardware, intermittent backhaul, silent packet loss.
- **Under attack** — CVE-2026-37554 demonstrates a single malformed packet crashing an ETSI ITS-G5 receiver with no credential compromise required. It is not an isolated case: CVE-2026-43988 (uncaught `std::runtime_error` on malformed ASN.1/OER input) and CVE-2026-44905 (unenforced Psid semantic constraints in the cryptographic verification path) are the same failure mode in the same stack within the same disclosure window. A node holding a valid certificate can be taken offline by any unauthenticated neighbour within radio range.
- **Compromised** — under attacker control while its certificate remains valid, because revocation under the SCMS CRL model propagates in hours to days.

That last point is the structural one. Even when misbehavior *is* eventually detected centrally, the revocation latency means the offloading layer keeps treating a known-bad node as fully trustworthy for an extended window. There is no mechanism for a receiver to locally contain a compromise in real time.

So the system has a blind spot with a specific shape: **it is well-defended against impostors and completely undefended against degradation.**

### 3. The proposed system

Three components, in sequence.

#### 3.1 Dynamic graph construction

The network is represented as a graph `G_t = (V_t, E_t)` rebuilt at each timestep `t`.

- **Nodes `V_t`:** fog nodes (RSUs) and vehicles currently in the region.
- **Edges `E_t`:** vehicle↔RSU links where the vehicle is within coverage; RSU↔RSU links where nodes share backhaul or coordination range.

**Node features (fog node):**

| Feature | Description |
|---|---|
| `load` | Current compute utilization |
| `queue_depth` | Pending tasks awaiting execution |
| `cert_valid` | Binary — certificate validity from SCMS |
| `success_ewma` | Exponentially weighted moving average of recent task completions |
| `latency_dev` | Deviation of observed latency from advertised/expected |
| `uptime_stability` | Frequency of observed restarts or dropped sessions |

**Node features (vehicle):** task demand, current speed, dwell-time estimate in current coverage zone.

**Edge features:** link latency estimate, signal strength, link age (newly formed links are less characterized and should be treated with more uncertainty).

The graph is *dynamic* in a specific, non-cosmetic way: as a vehicle moves at highway speed, its edge set changes every few seconds. Coverage handoff is not an occasional event; it is the normal operating condition. A model trained on a static topology assumption does not transfer.

#### 3.2 Trust scoring via message passing

A GNN (starting with GraphSAGE or GCN layers) propagates information across `G_t`. After `k` rounds of message passing, each fog node's embedding encodes its own behavioral history *and* the behavioral history of nodes within `k` hops.

The trust score is a scalar head on that embedding:

```
h_v^(k) = AGGREGATE(h_v^(k-1), {h_u^(k-1) : u ∈ N(v)})
trust_v  = σ(W_t · h_v^(k))
```

**Why the graph matters here** — this is the load-bearing part of the claim, and it needs to be stated precisely:

A node-local model (an MLP over `success_ewma`, `latency_dev`, etc.) can already detect a node that is *individually* misbehaving. That is not the interesting case. The graph earns its place in three specific situations:

1. **Correlated infrastructure failure.** RSUs sharing a backhaul link, power feed, or software build degrade together. A node whose *neighbors* are failing carries elevated risk before its own metrics move — a node-local model cannot see this, because the signal is not in that node's own features yet.

2. **Cold-start and sparse observation.** A newly deployed or newly encountered node has almost no behavioral history. A node-local model has nothing to score. A GNN can propagate a prior from structurally similar, co-located neighbors.

3. **Collusion and coordinated misreporting.** Nodes that falsely advertise low load will show a behavioral pattern that is individually plausible but structurally anomalous relative to their neighborhood.

**This is the hypothesis that the ablation must test.** If Variant C (node-local MLP trust, no message passing) matches Variant D (GNN trust), the graph contributes nothing and the novelty claim collapses to "we added behavioral trust to offloading," which is a substantially weaker paper. The experimental design must therefore *construct scenarios where these three conditions actually occur* — correlated failures, cold-start nodes, colluding groups. If the evaluation only contains independent random node failures, the GNN will not beat the MLP, and it will not deserve to.

#### 3.3 Trust-integrated offloading decision

For a vehicle `i` with candidate set `C_i` (fog nodes in range), the selection is:

```
score(v) = α · trust_v  −  β · latency_v  −  γ · load_v
target   = argmax_{v ∈ C_i} score(v)
```

Start with fixed, hand-tuned `α, β, γ`. This is deliberate: a fixed weighting is interpretable, ablatable, and defensible. A learned combination layer can be added later as an extension, but it makes the explainability story harder and should not be the first version.

The critical design decision is that **trust enters the selection function, not a pre-filter.** A trust threshold that simply excludes nodes below a cutoff is a gate — that is architecturally the same as certificate checking, just with a different signal. Trust as a *weighted term* means a moderately-trusted-but-much-closer node can still win, which is the correct behavior under a latency budget. This distinction is worth stating explicitly in the paper; it is the difference between "we added a better filter" and "we changed the decision function."

#### 3.4 Decision explanation

After a target is selected, `GNNExplainer` (or `PGExplainer`) identifies which input features and which neighboring nodes most influenced the selection. A templating layer converts the attribution vector into an operator-readable statement:

> *"RSU-7 selected over RSU-3. Primary factors: RSU-7 trust score 0.91 (28 consecutive successful completions); RSU-3 trust score 0.42, down-weighted following latency deviation on RSU-3 and two adjacent nodes on the same backhaul segment over the preceding 40 seconds."*

Note what that sentence contains: not just *this node scored higher*, but *why the alternative was penalized*, including the neighborhood evidence. A node-local model cannot generate the second clause. **The explanation is therefore also evidence for the graph claim** — it surfaces the propagated signal in a form a reviewer can inspect directly.

### 4. What the system does not do

Stating this plainly is a strength in review, not a weakness.

- It does not replace SCMS. Certificate validity remains an input feature. The proposal is a complementary behavioral layer, not a substitute for cryptographic identity.
- It does not detect the *cause* of degradation. It observes that a node is behaving unreliably; it does not distinguish a hardware fault from an active attack.
- It does not perform driver or human behavioral assessment. Trust here refers exclusively to infrastructure node reliability.
- It does not handle multi-hop offloading chains in this version.

---

## Part II — Novelty Analysis

### 5. The precise claim

> **A vehicular fog offloading system in which node trust is (a) learned from behavior rather than asserted by certificate, (b) propagated across network structure via message passing so that neighborhood evidence informs node-level trust, (c) integrated as a weighted term in the offloading objective rather than as an admission gate, and (d) attributable, such that each offloading decision yields an inspectable justification.**

Four properties. The novelty is their conjunction. Each individually has prior art, and the paper must say so.

### 6. Prior art, honestly mapped

| Ref | Work | Learned trust | Structure-aware | Vehicular offloading | Explainable |
|---|---|:---:|:---:|:---:|:---:|
| 1 | GNN-Mamba Spatiotemporal Trust Evaluation | ✓ | ✓ | ✗ | ✗ |
| 2 | Multi-Hop Trust via GNN-Aided Agentic AI | ✓ | ✓ | ✗ | ✗ |
| 3 | Federated GNN Multi-Agent RL for AoI in VEC | ✗ | ✓ | ✓ | ✗ |
| 4 | Joint Allocation Under Asymmetric Information | ~ (economic) | ✗ | ✓ | ✗ |
| 5 | DRL for Delay-Optimized VFC Offloading | ✗ | ✗ | ✓ | ✗ |
| 6 | Explainability-as-a-Service for Edge AI | ✗ | ✗ | ✗ | ✓ |
| 7 | XRL for RAN Slicing (6G ORAN) | ✗ | ✗ | ✗ | ✓ |
| 8 | Self-Explaining RL for Mobile Resource Allocation | ✗ | ✗ | ✗ | ✓ |
| 9 | XAI for RL-based Networking | ✗ | ✗ | ✗ | ✓ |
| 10 | Explainable Fraud Detection (GNNExplainer + Shapley) | ✗ | ✓ | ✗ | ✓ |
| — | **This work** | **✓** | **✓** | **✓** | **✓** |

**The two closest threats to the claim, and the response to each:**

*Refs 1–2* already do learned, structure-aware trust with GNNs. The differentiation is not the trust mechanism — it is that trust there produces a *collaborator selection* in a general distributed setting, decoupled from a resource-allocation objective under a hard latency budget, and with no explanation. Overstating originality against these papers is the fastest way to lose a reviewer. The correct framing is: *"we adopt the learned structural trust formulation of [1,2] and integrate it into the vehicular offloading objective, which those works do not address, and make the resulting decision attributable."*

Note that [1] and [2] share authorship (Zhu and Wang, with Niyato on [2]); they are one continuing research line rather than independent convergent results. This slightly narrows the differentiation surface — there is a single formulation to position against, not two — but it also means a reviewer drawn from that community will know both papers well. The framing must be precise rather than dismissive.

*Ref 10* already applies GNNExplainer to GNN decisions. The differentiation is domain and object: explaining a *classification* (fraudulent / not) versus explaining a *resource allocation* (why this node, over that node, under this constraint). The second requires contrastive explanation — why the alternative lost — which is a different explanation target. Say this explicitly; do not let a reviewer raise it first.

### 7. Novelty categorization

**Primary — Algorithmic.** The conjunction above, specifically the integration of structurally-propagated behavioral trust into the offloading objective as a weighted term.

**Secondary — Design.** The dynamic-graph formulation, where topology churn from vehicle mobility is the normal operating condition rather than a perturbation. Supporting, not standalone.

**Not claimed — Technology.** The project uses existing frameworks (PyTorch Geometric, GNNExplainer, SUMO). No new tooling is contributed. Claiming technological novelty here would be inaccurate and would weaken the credible claims by association.

### 8. The honest weak point

The single most likely reviewer objection: *"Is the GNN necessary, or would a per-node reliability score suffice?"*

This is a fair question and the paper lives or dies on the answer. The response cannot be rhetorical; it must be Variant C in the ablation, run under scenarios containing correlated failures, cold-start nodes, and collusion. If the graph does not help in those conditions, that is a real finding and should be reported as such — a paper that honestly reports "structural propagation helps under correlated failure but not under independent failure" is publishable and useful. A paper that avoids the question is not.

---

## Part III — Target Results

These are the outcomes the project aims to demonstrate. They are stated as hypotheses with success criteria, not as predictions — no numbers here are results, and none should be cited as such until measured.

### 9. Primary hypotheses

**H1 — Detection latency.** The behavioral trust model identifies and routes away from a degrading node substantially faster than certificate-based trust, which cannot react at all until central revocation completes.
*Success criterion:* measurable reduction in tasks dispatched to a degraded node between onset of degradation and effective avoidance, versus Baseline B.
*Note:* this comparison is close to guaranteed to favor the proposed system, because the baseline cannot react by construction. It is necessary to include but it is **not** persuasive on its own — a reviewer will discount it as a strawman. H2 is the real result.

**H2 — Structural advantage (the critical hypothesis).** Under correlated failure, cold-start, and collusion conditions, GNN-propagated trust (Variant D) outperforms node-local behavioral trust (Variant C).
*Success criterion:* a statistically meaningful gap across ≥5 seeds, with the gap widening as the correlation structure of failures strengthens.
*If this fails:* report it. The fallback contribution becomes the integrated + explainable offloading formulation, and the honest negative result on structural propagation. Plan the paper so it survives this outcome.

**H3 — Task success under degradation.** End-to-end task completion rate is higher for the proposed system across degraded-node fractions of 5–30%.
*Success criterion:* consistent improvement with non-overlapping error bars at ≥2 operating points.

**H4 — Explanation fidelity.** Generated explanations reflect the factors that actually drove the decision.
*Success criterion:* removing the top-attributed feature flips the decision in a high proportion of sampled cases; removing a randomly-chosen low-attribution feature flips it rarely. The gap between those two rates is the result — report both.

**H5 — Decision overhead.** Trust scoring and explanation generation do not push decision time outside a plausible fog-layer budget.
*Success criterion:* per-decision inference time reported explicitly, with explanation generation measured separately (it is expected to be the expensive component and may be practical only asynchronously — that is an acceptable and honest finding).

### 10. Deliverables

**Artifacts:**
- Simulation harness (SUMO + custom fog topology layer) producing reproducible dynamic graph sequences.
- Trained trust-scoring GNN and offloading decision engine.
- Explanation generator with the natural-language templating layer.
- Evaluation suite covering all five ablation variants across ≥5 seeds.
- Decision log with per-decision feature values, attributions, and generated explanations.

**Figures for the paper:**
1. System architecture (graph construction → trust GNN → offloading → explanation).
2. Trust-score timeline for a degrading node, overlaid with dispatch decisions shifting away.
3. Task success rate vs. degraded-node fraction, all variants, error bars.
4. Ablation bar chart (A–E), with C vs. D highlighted as the structural test.
5. Worked explanation example as rendered to an operator.

**Tables:**
- Related-work comparison (Section 6).
- Ablation results with mean ± std.
- Per-decision latency breakdown by component.

### 11. Limitations to state in the paper

Write this section early. It is the section reviewers use to calibrate how much to trust everything else.

- **Simulation-only.** No physical RSU testbed; mobility is SUMO-generated.
- **Synthetic behavior labels.** Degradation and compromise patterns are injected, not observed. No public dataset pairs vehicular mobility with fog-node trust ground truth — this is a genuine gap in the field, and it should be named as such rather than glossed. It also means results are conditional on the realism of the injection model, which is the most attackable assumption in the work.
- **Single-hop offloading only.**
- **No adaptive adversary.** Degraded nodes do not strategically manipulate their own behavior to game the trust score. A trust model that is not evaluated against an adaptive attacker has an untested robustness claim.
- **Fixed objective weights.** `α, β, γ` are hand-tuned; no sensitivity analysis unless time permits (and if it does, include it — it is cheap and strengthens the result).

### 12. Publication positioning

Target venues are IEEE conferences in vehicular networking, fog/edge computing, or intelligent transportation systems. Confirm the current year's CFP, scope, and deadlines directly from the venue — do not rely on remembered dates.

Two framings are available and they are not equally strong:

- *"We add trust to offloading"* — incremental, easily positioned against refs 4–5.
- *"Offloading decisions in vehicular fog are made on an incomplete trust model, and we show that structural behavioral evidence both improves them and makes them auditable"* — a gap-driven framing that carries the explainability and the auditability argument, and connects to a real regulatory pressure (post-incident reconstruction requirements for automated driving systems).

The second is the stronger paper. It also sets a higher bar: it requires H2 and H4 to hold, not just H1 and H3.

---

## Appendix — Reference literature

1. Spatiotemporal Trust Evaluation for Collaborator Selection via Customized GNN-Mamba (2026) — https://www.alphaxiv.org/abs/2605.07658
2. Task-Specific Trust Evaluation for Multi-Hop Collaborator Selection via GNN-Aided Distributed Agentic AI (2025) — https://www.alphaxiv.org/abs/2512.05788
3. Optimizing Age of Information in Vehicular Edge Computing with Federated GNN Multi-Agent RL (2024) — https://www.alphaxiv.org/abs/2407.02342
4. Joint Computing Resource Allocation and Task Offloading in Vehicular Fog Computing Under Asymmetric Information (2025) — https://www.alphaxiv.org/abs/2510.26256
5. Deep Reinforcement Learning for Delay-Optimized Task Offloading in Vehicular Fog Computing (2024) — https://www.alphaxiv.org/abs/2410.03472
6. Scalable Explainability-as-a-Service (XaaS) for Edge AI Systems (2026) — https://www.alphaxiv.org/abs/2602.04120
7. Enhancing AI Transparency: XRL-Based Resource Management and RAN Slicing for 6G ORAN (2025) — https://www.alphaxiv.org/abs/2501.10292
8. Self-Explaining Reinforcement Learning for Mobile Network Resource Allocation (2025) — https://www.alphaxiv.org/abs/2509.14925
9. eXplainable AI for RL-based Networking Solutions (2025) — https://www.alphaxiv.org/abs/2509.21649
10. Explainable Fraud Detection with GNNExplainer and Shapley Values (2025) — https://www.alphaxiv.org/abs/2509.12262

**Motivating incidents:**

- **CVE-2026-37554** — Vanetza V2X v26.02, DoS via uncaught OpenSSL ECC point-validation exception escaping `Router::indicate()` to `std::terminate`. Reserved 2026-04-06, published 2026-05-01, MITRE-assigned, CVSS 3.1 base 7.5 (AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H). https://www.cve.org/CVERecord?id=CVE-2026-37554
- **CVE-2026-43988** — Vanetza ≤26.02, uncaught `std::runtime_error` in the ASN.1/OER parsing pipeline (CWE-248). Fixed in commit `62dfe58`.
- **CVE-2026-44905** — Vanetza ≤26.02, DoS in the cryptographic verification pipeline via unenforced Psid semantic constraints during ASN.1 decoding. Fixed in commit `e1a2e27`.

Cite the CVE records rather than vendor write-ups. The original 37554 disclosure exists as a public gist from Innora Security Research if the technical detail is needed, but a security vendor's own gist is a weaker citation than the MITRE record.

*Verified 2026-09-02: all ten references confirmed to exist with matching titles and authors. Ref [6] is peer-reviewed (IEEE SoutheastCon 2026, DOI 10.1109/SoutheastCon63549.2026.11476268); the remaining nine are arXiv preprints and should be described as such. Refs [1] and [2] share authorship. The CVE originally cited as 2026-37555 was a numbering error and has been corrected to 2026-37554.*

# CLAIMS.md

**What this paper may and may not say.** Written in S11a from F16-F31 and D43-D62 (and the locked decisions
L1-L12 of PROJECT_SPEC.md). Status: **DRAFT for design review. S12 must not start until this file is approved.**
Once approved, a change needs a new numbered DECISIONS entry (the next free number) saying what changed and why;
this file is then re-issued, never silently edited.

## How S12 uses this file

1. Every claim in the text must match an **ALLOWED** entry (A-numbers) in wording and in strength. A sentence that says
   more than its entry says is not allowed; soften it or drop it.
2. Nothing in the **FORBIDDEN** list (X-numbers) may appear, in any section, including the abstract, title, captions
   and conclusion.
3. Every number in the text or a table must be copied from the FINDINGS.md entry (or the generated tables in
   `results/s10/tables/`) named in its ALLOWED entry. Put the source entry in a LaTeX comment next to the number.
   A number that is not in the repo is left as a TODO.
4. The **REPORTING CONVENTIONS** (R-numbers) apply to every result sentence.
5. Negative and null results are results: they go in the Results section with the same prominence as positive ones.
6. If a sentence cannot be made to fit, say so in the S12 report rather than bending it.

All results are for the simulated scenario of this repository (synthetic road grid, 20 RSUs, 60 vehicles, injected
degradation, 5 seeds). Every result sentence carries that scope ("in this simulation") at least once per paragraph.

---

## ALLOWED CLAIMS

### Setting and design

**A1. The system.** The paper studies task offloading from vehicles to roadside fog nodes (RSUs) with a behavioural trust
layer: a graph neural network (or a node-local MLP) scores each RSU's trustworthiness from observed task outcomes, and a
**fixed analytic rule** `score = alpha*trust - beta*latency - gamma*load` (argmax over in-range, dwell-feasible RSUs)
selects the target. Selection is not learned (L1). Variant A is the same rule with alpha = 0 (L2).
*Sources:* PROJECT_SPEC 1-3, L1, L2; D1, D2, D51.

**A2. Trust is learned self-supervised from outcomes.** The trust head is trained to predict whether an RSU meets its
next task deadline; the injected behaviour class is never a label or a feature (L3, L4), enforced structurally (the
ground truth is a separate sealed package that the training path cannot import, tested). Trust models are trained once per
seed on mixed conditions and then frozen across both sweeps (L12).
*Sources:* D3, D4, D35, D12.

**A3. The scenario.** Synthetic Manhattan-grid mobility (not SUMO, L7), 20 RSUs, 60 vehicles, 4 backhaul segments, a
task model with three task types, and a closed-form execution model; degradation is injected per **backhaul segment**
(L5) with a shared onset at step 400 and a linear ramp over 300 steps; RSU-RSU edges carry a `same_segment` flag (L6).
Failure correlation rho interpolates how concentrated the degraded RSUs are on one segment (D29), and realised
correlation is reported beside nominal rho. Advertised-versus-observed discrepancy is the trust signal (L8).
*Sources:* configs/demo.yaml; D20-D22, D29-D32, D34, D41; F10.

**A4. The variants.** A: no trust. B: certificate-only trust; in this scenario `cert_valid` is constant, so **B makes
the same decisions as A** (asserted on all 2000 paired draws of S9, F26). B-oracle: trust read from the sealed fault state,
zero delay; an **upper bound, not a method**. C: node-local MLP trust. D: 2-layer GINEConv trust over the graph. E: D plus
explanation (its decisions equal D's by construction, asserted). Gate-C: an **extra baseline**, C's frozen head with candidates
below tau = 0.9 removed first (D59). Weights are the D52 weights, tuned once on seed 100 and applied unchanged.
*Sources:* PROJECT_SPEC 3; D44, D50, D52, D59; F26 (checks), F28.

### Closed-loop dispatch results (H1, H3)

**A5. Task success against no trust (H3).** In the closed loop, C and D raise post-onset deadline success over A at all
four degraded fractions (5%, 10%, 20%, 30%; rho 1.0), with non-overlapping seed error bars, and at every point the paired 95% t-interval
over the 5 seeds lies above 0 with a mean of at least the 0.01 floor (D56 label WIN at 4 of 4, so H3 holds against A).
C: +0.021, +0.038, +0.050, +0.054. D: +0.016, +0.029, +0.034, +0.039 (absolute success rate). At the shared point
(rho 1.0, 20%): A 0.871 +/- 0.003, C 0.921 +/- 0.010, D 0.905 +/- 0.008.
*Sources:* F26 (table and paired effects), F27 (verdicts); D56. Tables: `ablation_shared_point`, `paired_effects_shared_point`.

**A6. Against the oracle.** C is within 0.001-0.003 of B-oracle's success at every fraction (label NULL: the 0.01 floor is not
reached by the oracle's advantage). D is 0.006-0.018 below the oracle and is NEGATIVE against it at 10%, 20% and 30%.
B-oracle is described as an upper bound that reads sealed ground truth.
*Sources:* F27 (items 2 and verdict block), F26.

**A7. Dispatches to degraded RSUs (H1).** Dispatch counts are always stated **together**: n_av (dispatches to degraded RSUs
from onset to the run's own effective-avoidance step), n_win (the fixed window, steps 400-1200, which does not depend on
the avoidance rule) and the never-avoided rate (fraction of runs that never reach the avoidance rule). At the shared point:
A n_av 627.5 +/- 100.4, n_win 912 +/- 21, never-avoided 0.47 +/- 0.12; C 153 +/- 48, 506 +/- 75, 0.03 +/- 0.04;
D 211 +/- 84, 610 +/- 84, 0.03 +/- 0.04; B-oracle 24 +/- 26, 395 +/- 90, 0.01 +/- 0.03. Relative to A (in units of A's count)
the n_av reduction is +0.756 [+0.676, +0.836] for C and +0.664 [+0.523, +0.806] for D (WIN), and the fixed-window reduction
is smaller, +0.446 [+0.351, +0.541] and +0.331 [+0.221, +0.442]. Against B-oracle both are NEGATIVE (the oracle is better by
0.205 and 0.296 of A's count). Detection latency (descriptive, never gating): C 151 +/- 22 steps, D 220 +/- 25 steps after
onset; A and B cannot detect by construction; the oracle detects at once by construction; D leaves 4% of degraded RSUs
undetected inside the 400-step window (detected fraction 0.96 +/- 0.07).
*Sources:* F26, F27 (item 3); D56, D57, D60 (Q4). Table `ablation_shared_point`.

**A8. The shift is partial.** Trust is one weighted term against latency and load, so degraded RSUs still receive a
share of tasks. In the demo scenario (seed 100, draw 7, a tuning seed, labelled as such) the plateau share of dispatches to degraded RSUs
is 0.226 for A, 0.128 for C and 0.142 for D, and the degraded RSUs still receive about 13-14% of plateau tasks under C and D. It may be said that
dispatch moves away from the degrading segment while trust falls (Figure 2, both panels); any figure value read off Figure 2 that is not in
FINDINGS is a TODO, not a quoted number.
*Sources:* F22 (demo), Figure 2 (`fig2_trust_timeline`, from `results/s10/timeline.npz`), D58 item 4.

### H2: the structural test (pre-registered negative and nulls)

**A9. H2 was tested and is not supported.** On the pre-registered diagnostic AUC (D43), the 2-layer GNN head D did **not**
beat the capacity-matched node-local head C: the paired difference D minus C is negative at every rho (-0.0297 [-0.0542, -0.0051]
at rho 1.0; -0.0517 [-0.0902, -0.0131] at rho 0), the slope of the gap on realised correlation is +0.0300 [-0.0017, +0.0617]
(not significant), so the verdict is NEGATIVE (C beats D). Collusion is NOISY at both rho (inconclusive: the models never saw
colluders in training). Cold start is NULL at rho 1.0 (-0.0273 [-0.0935, +0.0389]; an advantage of 0.05 is ruled out), and
at rho 0 D is worse and the 95% interval over seeds excludes 0 (-0.1431 [-0.2226, -0.0635], reported as information). C detects degradation about
77-97 steps earlier than D at every rho. "H2 holds in full" is false.
*Sources:* F16, F17; D43, D47, D49. Figure 5.

**A10. The closed loop agrees.** D is not better than C on any closed-loop point: the paired success difference D minus C is
-0.016 [-0.027, -0.006] at the shared point (NEGATIVE at 20% and 30%, NULL at 5% and 10%), about the same at rho 0 and rho 1.0
(-0.016), and it does not widen with rho. This is the secondary closed-loop check of H2 and does not revise A9.
*Sources:* F26, F27 (item 4); D56 section 3.

**A11. Why, as far as it was measured.** `same_segment` is correct on every sampled RSU-RSU edge (0 mismatches over 92,000 instances,
0 on vehicle-RSU edges) and D's output depends on it (mean |change in trust| 0.024-0.103 when it is zeroed, set to 1 or permuted),
but removing it does not systematically cost detection. A degraded RSU's own evidence moves at the same time as its degraded
same-segment neighbours' (own-lead -0.3 to +0.5 s, all intervals include 0; about 110 s after onset), because onset is shared (D31).
At rho 0, 59% of degraded RSUs have only healthy same-segment neighbours; at rho 1, 95% have degraded neighbours but those move
simultaneously. So in this scenario own-node evidence is never later than neighbour evidence, and a lead-time advantage for message
passing is not available by construction. This is stated as a **boundary condition of this scenario**, not as a general result.
*Sources:* F18, F19; D48, D49.

### Explanation (Layer 1, Layer 2) and overhead (H4, H5)

**A12. Layer 1 is exact.** The decomposition of the winner-versus-runner-up margin into trust, latency and load terms is an identity
for the linear score; it is checked as bit-for-bit equality with the logged terms and score, and the selected node must be the rule's
argmax. It was run on all 6873 decisions of each of A, C and D on the demo scenario (20,619 decisions): 0 failures. It explains the
score, **not** why a node's trust has its value.
*Sources:* F23; D53.

**A13. Layer 2 is weakly faithful, unstable, rarely names neighbours, and asynchronous only.** GNNExplainer applied to the trust
head only (L11). Two passes of the D54 protocol, each with its own literal label (not pooled):
S8 (draw 100): trust-change gap +0.160 [+0.043, +0.277] -> WEAK; decision-change gap +0.033 [+0.0003, +0.065] -> WEAK.
S9 replication (draw 101): trust-change gap +0.158 [+0.031, +0.284] -> WEAK; decision-change gap +0.0500 [+0.017, +0.083] ->
FAITHFUL by the literal >= rule, **exactly at the 0.05 floor, so called borderline** (the mean of the two passes, about 0.042, is
a description only). Top and random-low rates are always quoted with the gap (S8 trust change: top 20.5%, low 4.5%; S9: top 0.192,
low 0.035). The masks are unstable (top-unit agreement 0.22 for D against a 0.80 threshold), so every explanation is one draw with its
run seed. The neighbour clause is conditional and rare (10% of D explanations; 16% when the explained RSU is degraded).
Layer 2 takes 307-598 ms (S8) and 310 ms at 1 thread / 675 ms at 10 threads (S9) per decision against a stated 100 ms
assumption, so it is **asynchronous only**. C's higher fidelity verdict is an artefact of its random-low pool (units C cannot read) and is not
offered as evidence that C's explanations are better.
*Sources:* F24, F25, F27 (item 5), F26 (H4/H5 block); D54, D55, D60 (Q2).

**A14. Overhead.** Per decision, median at 1 thread: trust inference 0.27 ms, selection 0.04 ms, Layer 1 0.09 ms, Layer 2 310 ms
(at 10 threads 0.94, 0.05, 0.11, 675 ms; Layer 2 is faster on one thread for graphs this small). Hardware as named in
the table caption (13th Gen Intel i5-13450HX, CPU only, one decision at a time).
*Sources:* F26 (H5 block), F24; table `latency_breakdown`.

**A22. What decides decisions, in the demo scenario (descriptive only).** In the demo scenario (seed 100, draw 7, a tuning seed, labelled as such;
one run per variant, 6873 decisions each), trust is the decisive term (a term worth at least 0.6 of the winner-versus-runner-up margin, D53) in 15% of
C's decisions and 31% of D's; the winner has lower trust than the runner-up in 12.5% of C's decisions and 18.9% of D's. Load and latency decide
most of the rest. This describes how the fixed rule behaved in this one scenario; it carries no interval, is not a performance result and is not
used to compare C with D. (A22 is numbered out of sequence; it sits here because it belongs with Layer 1.)
*Sources:* F23 (table of decisive terms; 12.5% and 18.9% are 857 and 1296 of 6873, which F23's prose rounds to 13% and 19%); D53.

### Weighted rule versus gate

**A15. Gate-C: no difference on success.** On the same trust head (C), removing candidates below tau = 0.9 before the same rule
performed equivalently on task success: the paired difference C minus Gate-C is +0.0003 [-0.0003, +0.0009] (5%), +0.0001
[-0.0004, +0.0005] (10%), -0.0002 [-0.0006, +0.0003] (20%), -0.0003 [-0.0006, +0.0001] (30%); NULL at all four points; the pre-registered
verdict is NOT SEPARATED. Gate-C sends slightly fewer tasks to degraded RSUs than C (n_win 493.5 vs 506.3 at the shared point) without a
success difference (descriptive, no pre-registered test; D59 judged success only). The gate acted often (removed a candidate on 31-44% of decisions) and its empty-candidate-set fallback rate at
steps >= 400 was 4.8%, 6.2%, 12.1%, 19.1% (5% to 30% degraded). tau was chosen on one seed from a flat grid (all values within 0.0007).
**Framing (D62):** "weighted term, not a gate" is a **design argument**, not a measured gain. The weighted form's benefits are an exactly
decomposable score (Layer 1) and no empty-candidate-set failure mode.
*Sources:* F28-F31; D59, D61, D62.

### Protocol

**A16. Pre-registration and statistics.** The H2 rule (D43), the explanation protocol (D54), the sweep, metrics and verdict rule
(D56, with the single amendment D57 made before any evaluation draw ran) and the Gate-C protocol (D59) were each committed before the
numbers they judge (verified from commit order for D59: D59 = 3710795 at 19:26, first Gate-C code = 86b3a20, first Gate-C number (F28) = eead7f1 at 19:29, evaluation results at 19:33).
Seeds 1-5 with frozen checkpoints; evaluation draws 100-149; tuning seed 100 and the demo draw excluded from every evaluation. The unit
of analysis is the seed: per-seed means of per-draw paired differences, 95% Student-t intervals over 5 seeds (df 4). Mean +/- sd is across
seeds. No multiplicity correction. D42: the sweeps contain no cold-start RSU; every sweep point reports 250 draws attempted, 0 failed, 250 aggregated
(S4's cold-start conditions report attempted / infeasible / aggregated per D42).
*Sources:* D43, D54, D56, D57, D59, D42; F16, F26, F29; commit log.

**A17. Reproducibility.** Every run is reproducible from (config file, seed); figures and tables regenerate from one command and are
byte-identical on re-run.
*Sources:* CLAUDE.md rule 7; D17; D58; `tests/test_figures.py`.

**A18. The exact tuned weights are not a finding.** The weights were tuned by one procedure on one seed for every variant, A included; the
surface is flat for the trust-bearing variants (best vs default point within 0.002-0.005, F20), so the specific weights carry no claim.
*Sources:* F20; D50, D52.

### Limitations (may be stated, and must be stated)

**A19.** Simulation only, no testbed. The injection model is synthetic and is the most attackable assumption; no public dataset pairs vehicular
mobility with fog-node trust ground truth. Single-hop offloading only. No adaptive adversary (L10): degraded nodes do not game the trust score.
Hand-tuned weights, tuned on one seed. Scale capped at 15-30 RSUs and 50-100 vehicles (L9). One topology per seed, 5 seeds. Onset is shared by all
degraded RSUs (D31). The trust heads were trained on Baseline-A (open-loop) dispatch and are evaluated in the closed loop, a distribution
shift (D50 item 5, F21 item 3). Collusion models never saw colluders in training. B-oracle reads sealed truth. The avoidance bar of 0.75 is a stated
assumption. Seed 1 was trained at the default thread count and seeds 2-5 at 3 threads (D47; inference is not thread-sensitive, F18 check 3).
Layer 2 uses one explainer run per explanation and a 200-decision sample. Training and evaluation share a seed's traffic and differ only in injection draws (D44).
*Sources:* PROJECT_SPEC 4.6; L9, L10; D31, D47, D50, D56; F21, F18.

**A20. Untested future work (stated as such, as hypotheses).** Staggered onset (D31(c)); an observation-count feature (D38(d)); other architectures;
training the trust heads on closed-loop dispatch and the F21 dispatch-shift as a possible cause of D trailing C; D's aggregation over many healthy
neighbours diluting its own-node signal (F19); cold-start and collusion conditions where a graph model might help; gating a different head.
None of these was run and none may be reported as a finding.
*Sources:* D47, D49, D60 (Q3), F19, F21.

### Contribution statement

**A21. The contribution, as it may be stated.** In this simulation, the paper contributes: (i) a behavioural trust layer that drives a fixed weighted
selection rule, validated in the closed loop against a tuned no-trust baseline (A) and a perfect-knowledge oracle upper bound (B-oracle), in which the
node-local head (C) is the strongest; (ii) an exactly decomposable explanation of each selection (Layer 1); (iii) pre-registered negative and null
results on graph structure, with the measured boundary condition of this scenario (F18). It must not be worded so as to imply that the graph helped (X1, X2).
*Sources:* A5, A6 (i); A12 (ii); A9, A10, A11 (iii); F18, F26, F27; D56, D60.

---

## FORBIDDEN CLAIMS

**X1.** That graph structure, message passing, or the GNN **improves** trust, detection, dispatch or success over the node-local model; "outperforms C",
"structural advantage", "propagation helps", "neighbourhood evidence improves trust". (A9, A10: negative.)

**X2.** Any title, abstract, contribution list or conclusion wording that implies the graph helped, or that the novelty rests on a demonstrated graph benefit.
The conjunction of properties may be described as the design; the graph's benefit may only be described as tested and not supported. (D60 Q1.)

**X3.** That the weighted rule beats a gate, or that "weighted, not a gate" is a measured advantage. (A15, D62.)

**X4.** That the system improves on **deployed** certificate trust (SCMS) in general. B equals A in this scenario because `cert_valid` is constant; the comparison is against
no behavioural signal. B must not be presented as a strawman that was beaten, and B-oracle must not be presented as a competing method.

**X5.** That Layer 2 explanations are "faithful", "reliable", "stable", "trustworthy" or "real-time"; that the neighbour clause is the system's general
behaviour; that the H4 decision-change result is FAITHFUL without the S8 pass beside it; any pooling of the two H4 passes into a new test. (A13, D55, D60.)

**X6.** Any claim about real deployments, a testbed, safety, certification, or SUMO (mobility is synthetic, L7); any claim that results hold outside the injection model,
other topologies, other scales, other degradation shapes, or adaptive adversaries.

**X7.** Robustness to collusion (inconclusive), a benefit of the graph for cold start (NULL at rho 1, worse at rho 0), or any claim about adaptive adversaries (L10).

**X8.** Any comparison of performance with other published systems (SGA-SAC, G-TRAC, Siddiqua & Jahan, refs 1-10): none was run. Related work may be
characterised only as RELATED_WORK.md allows, and never as "outperforming", "state of the art", "first", or "novel in all of" without that support.

**X9.** Quoting n_av **alone**, or any H1 sentence without the fixed-window n_win and the never-avoided rate beside it. Quoting the "76% fewer" figure without
the smaller fixed-window figure. (D60 Q4.)

**X10.** Stating mechanisms for D trailing C (dispatch shift, dilution, training distribution, seed fit) as findings; they are hypotheses (A20).

**X11.** "Statistically significant" or p-values: none was computed. Allowed: "the 95% t-interval over 5 seeds excludes 0 / lies above the floor".
Single-run numbers; draw-level standard errors (draws within a seed share a model and traffic and are not independent samples).

**X12.** That Variant E is a separate performance result (its decisions are D's by construction), or that explanation changes any decision.

**X13.** Background statements that are not in the repo and not cited from a verified reference, for example the time SCMS CRL revocation takes to propagate,
regulatory pressure for post-incident reconstruction, or CVE details. They may be used only with a verified citation (otherwise leave a TODO). The CVE
records listed in TECHNICAL_AND_NOVELTY.md were not re-verified in S11a and may not be cited until they are.

**X14.** Re-describing any pre-registered result with a different rule after the fact (changed floor, window or label); describing D57 as anything other than a
pre-evaluation correction; describing the tuning-seed grids (F20, F28) as findings about the weights or tau.

**X15.** That Layer 1 explains why a node's trust is high or low (it explains the score only), or that the explanation is causal.

**X16.** Any performance, fidelity or "result" claim based on seed 100 or draw 7 (the tuning seed and the demo draw). Allowed only: the A12 exactness
identity check, A8 as a labelled illustration, and A22 as descriptive.

---

## REPORTING CONVENTIONS

**R1. Both heads, equally.** C and D appear together in every table and figure; C is named the strongest result; the text says plainly that D did not beat the node-local
head. (D60 Q1, D58.)
**R2. H1 triplet.** n_av with n_win and the never-avoided rate, always. (D60 Q4.)
**R3. H4 two passes.** Both passes with exact numbers and their own literal labels; "borderline" for the decision-change result. (D60 Q2.)
**R4. Effect and interval.** Every difference carries its 95% interval over seeds and, where a pre-registered rule applies, its D43/D54/D56/D59 label.
**R5. Scope.** "In this simulation" at least once per results paragraph; limitations in their own section but results never moved into it.
**R6. Dated protocol.** The pre-registrations are described with their entries (D43, D54, D56 + D57, D59) and the fact that the rule preceded the numbers.
**R7. Gate-C and B-oracle labelled.** Gate-C as an extra baseline, B-oracle as an upper bound, B as equal to A in this scenario, E as D plus explanation.
**R8. Figures from the script.** Figures come from `figures/s10/` via `scripts/make_figures.py`; none is redrawn. Every number in the text matches a source entry cited in a LaTeX comment.

---

## ITEMS THE PAPER DRAFT MAY NOT FILL FROM MEMORY (S12 TODO list seeds)

- The motivating-incident evidence (CVEs) and any statement about SCMS revocation delay or regulation: needs verified citations (X13).
- Venue, page limit, and class: undecided.
- Any numbers for the rho-sweep H1/H3 points not in `results/s10/tables/ablation_all_points` or F26.
- Gate-C at rho-sweep points: not run (D59: fraction-sweep points only); the paper must say so.

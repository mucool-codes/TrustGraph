# RELATED_WORK.md

Written in S11a (2026-10-15). The related-work set for the paper, with what was verified and what may be said about each work.
Companion to CLAIMS.md (X8: no performance comparison with any of these works was run).

## Verification record

Rule: cite only what was confirmed to exist. "Verified" below means the title, authors and venue or identifier were read from the
publisher or arXiv page on the date shown. **Depth is metadata and abstract only (and, for SGA-SAC, the abstract, highlights and introduction
shown on the publisher page; the full text is paywalled).** Any statement about a work's content is therefore limited to what its abstract
says, and is phrased "per its abstract" or "to our reading of the abstract".

| # | Work | Verified | Source read |
|---|---|---|---|
| 1-10 | The ten works of TECHNICAL_AND_NOVELTY.md section 6 and its appendix (refs 1-10) | 2026-09-02 (S0 record in that file: all ten confirmed with matching titles and authors; ref 6 peer-reviewed, the other nine arXiv preprints; refs 1 and 2 share authorship) | not re-checked in S11a |
| 11 | **SGA-SAC**: X. Cheng, B. Xu, H. Yin, S. Shen, Z. Liu, T. Wang, R. Li. "SGA-SAC: Fusing Bayesian Stackelberg games to DRL with graph attention for explainable task offloading in vehicular edge computing." *Information Fusion* 136, article 104506, 2026 (online 2026-06-01; issue date December 2026). DOI 10.1016/j.inffus.2026.104506 | 2026-10-15 | ScienceDirect article page (abstract, highlights, introduction, reference list); DOI and volume from its metadata |
| 12 | **Siddiqua & Jahan**: F. Siddiqua, M. Jahan. "A Trust-Based Malicious RSU Detection Mechanism in Edge-Enabled Vehicular Ad Hoc Networks." arXiv:2208.05680 (cs.CR), submitted 2022-08-11; arXiv lists the journal reference *Journal of the Asiatic Society of Bangladesh, Science*, 2025 (DOI 10.3329/jasbs.v51i1.82796) | 2026-10-15 | arXiv abstract page (title, authors, abstract, journal reference) |
| 13 | **G-TRAC**: C. Nguyen, E. Elmroth. "Trust-Aware Routing for Distributed Generative AI Inference at the Edge." arXiv:2603.28622 (cs.DC), submitted 2026-03-30; accepted at the 22nd Int. Conf. on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT 2026), DOI 10.1109/DCOSS-IoT69657.2026.00036 | 2026-10-15 | arXiv abstract page (title, authors, abstract, comments, journal reference); also found as the top result of a search for "G-TRAC trust-floor pruning" |

The identifiers supplied for review (SGA-SAC in Information Fusion 136:104506, 2026; Siddiqua & Jahan arXiv 2208.05680) matched exactly.
For G-TRAC the paper and its authors were found by search and confirmed on its arXiv page, so it may be cited.
Not verified and not to be cited until it is: the CVE records of TECHNICAL_AND_NOVELTY.md (CLAIMS.md X13).

## What may be said about each new work (abstract level)

**11. SGA-SAC.** Per its abstract: a task-offloading and resource-allocation strategy for vehicular edge computing that models vehicles and
RSUs as a hierarchical Bayesian Stackelberg game, uses a deep-reinforcement-learning algorithm (soft actor-critic) with a **graph attention
mechanism in the policy network**, and uses **Grad-CAM** to visualise the multi-agent decision; evaluated on latency and energy consumption.
It may be described as an explainable, graph-based vehicular offloading method. Its abstract does not describe a node-trust model.
It may **not** be said to be non-explainable, to lack trust anywhere in the full text, or to be outperformed. Differentiation that may be
stated: its explanation is a gradient visualisation of a learned policy, while ours is an exact decomposition of a fixed linear score plus a
separate attribution on the trust head; its objective is latency and energy, ours is deadline success under injected node degradation.

**12. Siddiqua & Jahan.** Per its abstract: a trust-based scheme to detect malicious RSUs in edge-enabled VANETs that uses stable, frequent
RSU-to-RSU interaction to analyse an RSU's behaviour and a check for altered sensor-detected data in beacons; it reports detecting about 92%
of malicious RSUs (its own figure) and that integrating the scheme with VANET routing improves routing efficiency. It may be described as
behavioural trust for RSUs (not certificate-only) using RSU-RSU interaction, and as prior art for trust scoring of RSUs. The abstract does not
describe offloading, a learned model, or an explanation; do not state that it has none of these beyond "the abstract does not describe".

**13. G-TRAC.** Per its abstract: a trust-aware coordination framework for distributed generative-AI inference across heterogeneous edge
devices, formulating routing as a Risk-Bounded Shortest Path problem solved by **trust-floor pruning combined with Dijkstra's search**, with a
Hybrid Trust Architecture (global reputation state at stable anchors, lightweight updates to peers); evaluated on a testbed of commodity devices
for inference completion rate. It is not vehicular and is about routing of inference, not task offloading from vehicles. It may be cited as prior art
for a **trust floor that removes candidates below a threshold before selection**, which is the same structural idea as our Gate-C baseline;
we did not implement G-TRAC, and Gate-C is our own construction on our trust head. No performance comparison with it may be stated.

## The related-work table (13 rows + this work)

Rows 1-10 are copied from TECHNICAL_AND_NOVELTY.md section 6. Rows 11-13 were filled from the abstracts only: a check mark means the abstract
describes the property; "~" means partly or in a different sense (stated in the cell); a cross means the abstract does not describe it, which is
a statement about the abstract, not about the paper. "This work" records that message passing is implemented, not that it helped (S4: no
measured gain over node-local). The table is generated into `results/s10/tables/related_work.{md,csv,tex}` by `scripts/make_figures.py`.

| Ref | Work | Learned trust | Structure-aware | Vehicular offloading | Explainable |
|---|---|:---:|:---:|:---:|:---:|
| 11 | SGA-SAC (Cheng et al., Information Fusion 2026) | ✗ | ✓ (graph attention among agents) | ✓ | ✓ (Grad-CAM) |
| 12 | Siddiqua & Jahan (arXiv 2208.05680) | ✗ (trust scores, not described as learned) | ~ (RSU-RSU interaction) | ✗ (RSU detection, routing) | ✗ |
| 13 | G-TRAC (Nguyen & Elmroth, DCOSS-IoT 2026) | ✗ (reputation state, not described as learned) | ~ (path search over peers) | ✗ (edge inference routing) | ✗ |

## Positioning sentences that are allowed

- Refs 1-2 already do learned, structure-aware trust with GNNs; we adopt the formulation and integrate it into the offloading objective (TECHNICAL_AND_NOVELTY section 6), and our own result is that the graph did not help in our scenario (A9).
- Ref 11 is the closest explainable vehicular-offloading work found; the contribution difference is the exactly decomposable score and a trust head under injected degradation, not a performance claim.
- Ref 13's trust floor is the nearest analogue of the gate we tested; in our scenario the gate and the weighted rule performed equivalently (A15).
- Ref 12 is prior art for behavioural trust of RSUs.

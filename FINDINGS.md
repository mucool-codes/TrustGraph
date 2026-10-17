# FINDINGS.md

**What was MEASURED and what it means.** Every empirical result: calibration numbers,
training diagnostics, sweep outcomes, timings, failures.

Companion document: `DECISIONS.md` records what was *chosen*. Keep them distinct — a
decision is a choice, a finding is a number. A finding never argues for a design; if a
finding motivates a change, the change gets its own `DECISIONS.md` entry that cites the
finding.

## Rules for this file

- **Record the number BEFORE analysing, tuning, or trying to improve it.** A number
  written down after three rounds of tuning is not the same number. The first honest
  measurement is the one that has evidential value; everything after it is conditioned
  on what was already seen.
- **Negative results are recorded with the same weight as positive ones, and are never
  overwritten.** H2 failing is a publishable finding (`PROJECT_SPEC.md` section 4.1) —
  but only if it was written down when it happened.
- **Never edit a past entry's numbers.** If a result is superseded, add a new entry that
  references the old one. Corrections to *interpretation* are appended to the entry as a
  dated note; the raw numbers stay untouched.
- Every entry records the commit hash it was measured at. A finding without a commit is
  not reproducible and is therefore not a finding.

## Entry format

```
### F<n> — <title>
Date: YYYY-MM-DD | Session: S<n> | Commit: <short hash>
Config: <config file> | Seed(s): <seeds>
Command: <the exact command run>

Numbers:
  <raw measurements, verbatim - no rounding beyond what was printed>

Interpretation: <one line - what this means, not what to do about it>
```

---

### F1 — Environment: pinned stack, CUDA path works on the RTX 4050
Date: 2026-09-03 | Session: S0 | Commit: caf77de
Config: n/a | Seed(s): n/a
Command: `python scripts/verify_env.py`

Numbers:
```
python            : 3.11.9 (AMD64)
platform          : Windows-10-10.0.26200-SP0
numpy             : 2.4.6
pyyaml            : 6.0.3
torch             : 2.6.0+cu124
torch CUDA build  : 12.4
torch_geometric   : 2.8.0.post1
cuda available    : True
gpu               : NVIDIA GeForce RTX 4050 Laptop GPU
compute capability: 8.9
forward-pass device: cuda
GraphSAGE forward : OK - shape (12,), range [0.4056, 0.5423]
```
Driver 581.86. `torch-scatter` / `torch-sparse` / `pyg-lib` are NOT installed
(DECISIONS.md D18); `SAGEConv` runs on the pure-PyTorch fallback.

Install note: the cu124 Windows wheel is 2,532,350,702 bytes and the link dropped
twice mid-download (`NameResolutionError` on `download-r2.pytorch.org`, once at
506 MB). `pip install` has no usable resume across those failures. What worked was
fetching the wheel with a resumable downloader (`curl -C - --retry`) and then
`pip install`ing the local file. Measured link throughput ~1.8 MB/s.

Interpretation: the CUDA path is fully working, so no CPU fallback was needed - but
note that the S0 pipeline still runs on CPU by choice, for determinism (D19), not
because CUDA failed.

### F2 — Walking skeleton runs end-to-end and is bit-reproducible
Date: 2026-09-03 | Session: S0 | Commit: caf77de
Config: `configs/demo.yaml` | Seed(s): 20260903
Command: `python run.py --config configs/demo.yaml` (run twice, output compared)

Numbers:
```
scenario         : 20 RSUs, 4 backhaul segments, 60 vehicles, 8 steps
weights          : alpha=1.0 beta=0.5 gamma=0.3
total decisions  : 23
distinct RSUs    : 14
mean score       : 0.026274
trust range      : [0.3850, 0.4265]   (UNTRAINED model)
run 1 sha256     : d04344427fc4471526033b85b06d5d3dbf4da08cecf1995debe7bf2072b298f0
run 2 sha256     : d04344427fc4471526033b85b06d5d3dbf4da08cecf1995debe7bf2072b298f0
pytest           : 20 passed in 10.03s
```

Interpretation: the pipeline is reproducible from (config, seed) as Standing Rule 7
requires; the narrow trust range (0.385-0.427, spread 0.042) is the expected signature
of an untrained sigmoid head on random features and is a baseline to compare against
once training exists - not a result about trust.

### F3 — First S1 scenario: coverage too sparse, vehicles usually have no choice
Date: 2026-09-03 | Session: S1 | Commit: 4d09f25
Config: `configs/demo.yaml` (2400 m region, 5x5 blocks, 20 RSUs, coverage 350 m,
60 vehicles, 300 steps at dt=1 s) | Seed(s): 20260903
Command: `python scripts/generate_trace.py --config configs/demo.yaml`
         `python scripts/s1_report.py --config configs/demo.yaml`

Numbers:
```
backhaul segment sizes     : [5, 5, 4, 6]
RSU-RSU edges (undirected) : 14
mean speed                 : 12.44 m/s (sd 4.29) = 44.8 km/h
mean vehicles per RSU      : 2.595 (sd 1.492)
mean vehicle-RSU edges     : 51.90 per step
vehicle-timesteps covered  : 78.2%
mean RSUs in range         : 0.865
mean dwell time            : 45.76 s (median 44.00, sd 18.45)
completed dwell episodes   : 250
handoffs per veh per min   : 0.403
total handoffs             : 121
total coverage gaps        : 352
directed edges per snapshot: 131.8 (min 110, max 152)
same_segment edges         : 18.0 per snapshot
sequence sha256            : 38dc8a778104676c3dca3af00b383a8ae2f3de3b1772a9d0d9256865aa4ae8dc
```

Interpretation: the scenario is not usable as configured — with 0.865 RSUs in range
on average and 22% of vehicle-timesteps uncovered, the typical offloading decision
has one candidate or none, so the selection rule of L1 has nothing to choose between
and there are more coverage gaps (352) than handoffs (121). The cause is geometric,
not a bug: 28.8 km of road against 20 RSUs covering ~700 m of road each cannot
produce redundant coverage. Farthest-point RSU placement also drives sites onto the
region boundary, leaving the interior thin and the RSU-RSU graph fragmented at 14
edges over 20 nodes.

### F4 — Calibrated S1 scenario: full coverage, real choice, realistic handoff
Date: 2026-09-03 | Session: S1 | Commit: 4d09f25 (code) + calibration in this session
Config: `configs/demo.yaml` (1600 m region, 3x3 blocks, 20 RSUs, coverage 400 m,
coordination 650 m, 60 vehicles, 300 steps at dt=1 s) | Seed(s): 20260903, and 1/2/3
Command: `python scripts/generate_trace.py --config configs/demo.yaml --force`
         `python scripts/s1_report.py --config configs/demo.yaml`

Numbers (seed 20260903):
```
backhaul segment sizes     : [5, 5, 4, 6]
RSU-RSU edges (undirected) : 46      (degree min 3 / mean 4.6 / max 6, one component)
same_segment RSU-RSU edges : 32/46 = 70%
mean speed                 : 12.59 m/s (sd 4.14) = 45.3 km/h
mean vehicles per RSU      : 7.307 (sd 2.952)
mean vehicle-RSU edges     : 146.14 per step
vehicle-timesteps covered  : 100.0%
mean RSUs in range         : 2.436   (>=2 candidates on 96.4% of vehicle-timesteps)
mean dwell time            : 29.55 s (median 32.00, sd 17.82)
completed dwell episodes   : 528
handoffs per veh per min   : 1.960
total handoffs             : 588
total coverage gaps        : 0
directed edges per snapshot: 384.3 (min 362, max 410)
sequence sha256            : fb35eab950af5915f15646ded0711c12d28e0b9eb1fe8bf176cd5b9f2b5edce8
   (identical on a second invocation - the graph sequence is reproducible)
pytest                     : 84 passed in 16.02s
```

Cross-seed stability (topology and mobility both re-drawn):
```
seed  segment sizes   RSUs in range  dwell (s)  handoffs/veh/min  covered
   1  [6, 4, 5, 5]           2.347      31.16              1.837    100.0%
   2  [5, 3, 6, 6]           2.419      30.16              1.900    100.0%
   3  [7, 5, 4, 4]           2.409      28.38              2.000    100.0%
```

Structural checks: the RSU-RSU graph is a single connected component, and each of the
four backhaul segments is *internally* connected through same_segment edges alone.
A distance-threshold predictor recovers `same_segment` from geometry with only 63.0%
accuracy (seed 20260903 has 1 RSU homed off its geographic segment; across seeds 1-5
the count is 2, 3, 5, 0, 3).

Interpretation: the scenario is now usable — vehicles are always covered, have 2.4
candidate RSUs on average so the selection rule of L1 has a genuine choice, and hand
off about twice a minute with a ~30 s dwell, which is what a 45 km/h urban arterial
with 400 m cells should produce. The structural preconditions for H2 hold: segment
members can reach each other over same_segment edges, and `same_segment` is not
recoverable from position alone. These are scenario properties, not results about
trust — the behavioural features are still constants (S2/S3).

**Note added 2026-09-03 (S1, post-merge verification; raw numbers above unchanged):**
the 63.0% figure is real but does not support the conclusion drawn from it, and the
last sentence of the interpretation above is withdrawn. Rebuilding seed 20260903 with
`segment_swap_prob: 0.0` — which isolates the swap exactly, since the Lloyd step
consumes no randomness — gives:

```
                    median-split   best-threshold
actual (swap 0.10)         63.0%            78.3%
pure geometry (swap 0)     60.9%            80.4%
```

The predictor scores about the same either way, so 63.0% measures the weakness of the
probe — edge *length* cannot express which 2D region an edge lies in — not the effect
of the swap. What the swap actually did on this seed: exactly one RSU (19, at
(793, 0), geometric segment 0, assigned segment 3) differs from the pure-geometry
assignment, flipping 5 of 46 RSU-RSU edges. So `same_segment` is *not* purely
geometric, but by a small and, on this metric, unquantified margin. A probe that
actually separates the two conditions is still owed; until then no claim about the
recoverability of `same_segment` from position should rest on this number.

### F5 — Link model saturated at the cell edge under the -95 dBm sensitivity floor
Date: 2026-09-03 | Session: S1 | Commit: 4d09f25
Config: `configs/demo.yaml` | Seed(s): n/a (a property of the model, not a run)
Command: `pytest tests/test_graph.py::test_signal_strength_falls_with_distance`

Numbers, with the original `rssi_min_dbm: -95.0`:
```
distance   RSSI (dBm)   signal_strength
   300 m       -91.78            0.0585
   400 m       -95.16            0.0000   <- floor reached exactly at coverage radius
   500 m       -97.77            0.0000
```
After moving the floor to `rssi_min_dbm: -101.0` (a typical 10 MHz C-V2X sensitivity):
```
     1 m       -24.90            1.0000   (clipped at the near-field cap)
   100 m       -78.90            0.3623
   300 m       -91.78            0.1511
   400 m       -95.16            0.0958
   650 m      -100.85            0.0025
access latency spans 4.00 ms (1 m) to 17.08 ms (400 m); backhaul 2.00 to 5.98 ms.
```

Interpretation: the receiver sensitivity floor and the coverage radius are not
independent knobs — with the floor at -95 dBm it landed at exactly 400 m, so every
link near the cell edge pinned to `signal_strength = 0` and maximum latency, and the
two edge features stopped distinguishing a boundary link from one well outside. Found
by a test asserting monotonicity, not by inspection. A regression test now asserts the
edge-of-cell signal lies in (0.02, 0.5) for the operating config.

### F6 — Segment swaps were seed-dependent, and split segments on four of six seeds
Date: 2026-09-03 | Session: S1b | Commit: 15da028 (state measured), fixed in this session
Config: `configs/demo.yaml` | Seed(s): 1, 2, 3, 4, 5, 20260903
Command: rebuild the topology per seed with `segment_swap_prob` and `min_swap_fraction`
varied; diff against the pure-geometry assignment (both disabled), and test each
segment for connectivity over `same_segment` edges only.

Numbers, **before** the fix (probabilistic swaps only, as merged at 85b54f5):
```
seed        swapped RSUs   per-segment components over same_segment edges
       1               1   [[6], [4], [4, 1], [5]]        SPLIT
       2               2   [[5], [3], [6], [5, 1]]        SPLIT
       3               3   [[6, 1], [5], [3, 1], [3, 1]]  SPLIT
       4               0   [[6], [4], [6], [4]]           ok (no swaps at all)
       5               3   [[6], [4, 1, 1], [5], [3]]     SPLIT
20260903               1   [[5], [5], [4], [6]]           ok
pure geometry (all seeds): every segment a single component
```

**After** the fix (deterministic floor of 3, plus the admissibility constraint of D27):
```
seed   prob pass   final   clears floor   swapped RSU ids   segment sizes   connected
   1           1       3            yes   [0, 1, 15]        [5, 3, 5, 7]          yes
   2           1       3            yes   [1, 11, 18]       [6, 5, 5, 4]          yes
   3           3       3            yes   [10, 11, 15]      [7, 7, 3, 3]          yes
   4           0       3            yes   [1, 18, 19]       [5, 5, 4, 6]          yes
   5           2       3            yes   [4, 18, 19]       [6, 4, 5, 5]          yes
20260903       1       3            yes   [1, 18, 19]       [5, 5, 5, 5]          yes

same_segment RSU-RSU edges: 29-32 of 46 across the six seeds
scenario statistics unchanged (only segment labels moved, not positions or mobility):
  coverage 100.0%, 2.436 RSUs in range, 29.55 s mean dwell, 1.960 handoffs/veh/min
pytest: 88 passed
```

Interpretation: two separate seed-luck defects, neither of which the S1 exit condition
would have caught. First, the swap count is a binomial draw and came out zero on seed 4,
so the off-geometry property that D22 argues for was simply absent for one seed of a
protocol that averages over five (L12). Second, and worse, a swapped RSU could join a
segment it had no coordination edge into, splitting that segment into two components
over `same_segment` edges — true on four of six seeds, and false on seed 20260903, which
is the only seed S1 checked. The S1 test asserting internal connectivity therefore
passed while the property was broken for most seeds. Both are now structural rather than
probabilistic (D27), and the connectivity test runs across six seeds. The lesson worth
carrying: a structural precondition asserted on the default seed alone is not asserted.

### F7 — First S2 calibration: baseline success 96.6%, inside the target untuned
Date: 2026-09-13 | Session: S2 | Commit: 17ab365
Config: `configs/demo.yaml` as committed at 17ab365 (1200 steps; tasks, execution,
tracking and degradation blocks at their committed values) | Seed(s): 20260903
Command: `python scripts/s2_calibration.py --config configs/demo.yaml`
(run first from a byte-identical copy in the session scratchpad, before the script was
committed; the trace was regenerated at 1200 steps with `generate_trace.py --force`)

This is the first measurement of the task and execution model. Every parameter was set
a priori from the physical reasoning in `tasks.py` / `execution.py` before anything was
run; nothing has been tuned against these numbers.

Numbers:
```
== no degradation  (0.3s)
   tasks 6957  dispatched 6957  unserved 0
   success overall        0.9659
   success light          0.9920  (n=3490, mean deadline 0.110s)
   success medium         0.9487  (n=2454, mean deadline 0.351s)
   success heavy          0.9181  (n=1013, mean deadline 0.896s)
   success pre-onset      0.9696
   success post-ramp      0.9659
   tasks per RSU          min 185 median 366 max 495
   true load mean         0.700
   success_ewma end       healthy 0.968  degraded nan
== 20% rho=0  (0.3s)
   tasks 6957  dispatched 6957  unserved 0
   success overall        0.9211
   success light          0.9521  (n=3490, mean deadline 0.110s)
   success medium         0.8969  (n=2454, mean deadline 0.351s)
   success heavy          0.8727  (n=1013, mean deadline 0.896s)
   success pre-onset      0.9696
   success post-ramp      0.8774
   post-ramp on degraded  0.2405  (n=370)
   post-ramp on healthy   0.9672
   degraded per segment   [1, 0, 1, 2]  correlated picks 0
   tasks per RSU          min 185 median 366 max 495
   true load mean         0.700
   success_ewma end       healthy 0.974  degraded 0.279
== 20% rho=1  (0.3s)
   tasks 6957  dispatched 6957  unserved 0
   success overall        0.8886
   success light          0.9295  (n=3490, mean deadline 0.110s)
   success medium         0.8549  (n=2454, mean deadline 0.351s)
   success heavy          0.8292  (n=1013, mean deadline 0.896s)
   success pre-onset      0.9696
   success post-ramp      0.8096
   post-ramp on degraded  0.2632  (n=665)
   post-ramp on healthy   0.9657
   degraded per segment   [4, 0, 0, 0]  correlated picks 3
   tasks per RSU          min 185 median 366 max 495
   true load mean         0.700
   success_ewma end       healthy 0.963  degraded 0.299
```
("pre-onset" / "post-ramp" in the no-degradation block use the default config's onset
step 400 and ramp 300 as time windows only; nothing degrades in that run.)

Interpretation: the first measurement already meets the S2 target — 96.6% success with no
degradation (inside 90-97%, failures concentrated in heavy tasks on loaded nodes), falling
to 24-26% on degraded nodes once the ramp completes and to 81-88% network-wide — so the
calibration is recorded as landed without a tuning round.

Two things in these numbers are worth carrying forward, neither acted on here. First,
the rho=1 degraded set received 665 post-ramp tasks against 370 at rho=0: the degraded
segment happens to be a lightly loaded region that Baseline A dispatch favours, so the
network-wide success drop at a sweep point depends on *where* degradation lands relative
to demand, not only on how much of it there is. Second, post-ramp success on healthy
nodes (96.6-96.7%) is unchanged from baseline, confirming degradation does not leak into
healthy nodes' outcomes — including the one healthy RSU sharing the degraded segment at
rho=1 (4 of that segment's 5 RSUs are degraded, see F10).

### F8 — cert_valid and uptime_stability remain documented placeholders after S2
Date: 2026-09-13 | Session: S2 | Commit: 17ab365
Config: n/a | Seed(s): n/a (a property of the code, not a run)
Command: `grep -n PLACEHOLDER src/trustgraph/tracking.py src/trustgraph/simulator.py`

Numbers:
```
cert_valid        = PLACEHOLDER_CERT_VALID       = 1.0 for every RSU at every step
uptime_stability  = PLACEHOLDER_UPTIME_STABILITY = 1.0 for every RSU at every step
success_ewma, latency_dev: REAL as of 17ab365 (tracker, L8) - no longer constants
load, queue_depth: REAL advertised values as of 17ab365
task_demand: REAL (offloading vehicle's task size) as of 17ab365
```

Interpretation: no revocation/compromise model is in S2's scope, so `cert_valid` is still
the constant 1.0 — and would be even with one, since SCMS CRL propagation (hours to days)
exceeds the 20-minute horizon and a compromised node's certificate stays valid throughout.
The consequence for later sessions is exact, not approximate: Variant B's trust term is
the same constant for every candidate, so B's decisions are identical to Baseline A's in
every scenario this generator produces. `uptime_stability` is constant because no restart
or dropped-session process exists to observe.

### F9 — Observed outcomes depend on the dispatch policy that generated them
Date: 2026-09-13 | Session: S2 | Commit: 17ab365
Config: `configs/demo.yaml` | Seed(s): 20260903
Command: `python scripts/s2_calibration.py configs/demo.yaml` (numbers below are
extracted from the F7 output, not a separate run)

Numbers:
```
dispatch policy for every S2 scenario : baseline_a  (L1 rule, alpha = 0, advertised load)
post-ramp tasks landing on degraded RSUs, 20% degraded:
  rho = 0   n = 370   (degraded set spread over segments [1, 0, 1, 2])
  rho = 1   n = 665   (degraded set = 4 of segment 0's 5 RSUs)
tasks per RSU over the run (both rho): min 185, median 366, max 495
```

Interpretation: every observable the trust model will learn from — each RSU's outcome
count, its `success_ewma` trajectory, and the L3 "did the next task meet its deadline"
labels — is a function of which RSUs the dispatcher chose, so S2's trust-agnostic
dispatch is an experimental choice, not a neutral default.

This is the tension S2 was asked to flag rather than resolve (DECISIONS.md D33). Under
Baseline A a degraded node keeps receiving tasks after onset, so its features keep
updating and its failures stay visible. Under trust-aware dispatch (Variants C-E) the
same node is avoided as its trust falls, stops producing outcomes, and its
`success_ewma` freezes at whatever value it had when traffic stopped: the training
distribution generated here would differ from the distribution a deployed trust model
creates for itself. H1 ("tasks dispatched to a degraded node before effective
avoidance") and H3 are only meaningful in closed loop, where dispatch reads trust. The
simulator is structured so that a second policy is one more entry in
`DISPATCH_POLICIES`, but whether S3 trains on open-loop Baseline A data, and whether
S4/S5 re-run the generator closed-loop, is left to a decision request.

### F10 — rho is coarsely quantised at the L12 operating points
Date: 2026-09-13 | Session: S2 | Commit: 6295c04
Config: `configs/demo.yaml` (20 RSUs, 4 segments, min_segment_size 3) | Seed(s): 1..200
(each seed re-draws topology and degraded set), evaluation seeds 1..5 shown separately
Command: `python scripts/s2_rho_quantization.py --config configs/demo.yaml`

Numbers:
```
config configs/demo.yaml   RSUs 20   seeds 1..200
concentration of two independently chosen RSUs (segment geometry only): 0.221

=== fraction 0.05  ->  K = 1 degraded RSUs
  rho   E[conc]  distinct           [1]
  0.00      nan         1         1.000
  0.25      nan         1         1.000
  0.50      nan         1         1.000
  0.75      nan         1         1.000
  1.00      nan         1         1.000
  evaluation seeds 1..5: realised concentration per rho
    rho 0.00:   nan   nan   nan   nan   nan   distinct 0
    rho 0.25:   nan   nan   nan   nan   nan   distinct 0
    rho 0.50:   nan   nan   nan   nan   nan   distinct 0
    rho 0.75:   nan   nan   nan   nan   nan   distinct 0
    rho 1.00:   nan   nan   nan   nan   nan   distinct 0

=== fraction 0.10  ->  K = 2 degraded RSUs
  rho   E[conc]  distinct           [2]        [1, 1]
  0.00    0.220         2         0.220         0.780
  0.25    0.375         2         0.375         0.625
  0.50    0.600         2         0.600         0.400
  0.75    0.800         2         0.800         0.200
  1.00    1.000         1         1.000         0.000
  evaluation seeds 1..5: realised concentration per rho
    rho 0.00:  0.00  0.00  0.00  0.00  0.00   distinct 1
    rho 0.25:  0.00  0.00  0.00  0.00  0.00   distinct 1
    rho 0.50:  0.00  1.00  0.00  0.00  0.00   distinct 2
    rho 0.75:  0.00  1.00  0.00  1.00  1.00   distinct 2
    rho 1.00:  1.00  1.00  1.00  1.00  1.00   distinct 1

=== fraction 0.20  ->  K = 4 degraded RSUs
  rho   E[conc]  distinct           [4]        [3, 1]        [2, 2]     [2, 1, 1]  [1, 1, 1, 1]
  0.00    0.217         4         0.000         0.155         0.115         0.610         0.120
  0.25    0.310         5         0.045         0.235         0.240         0.405         0.075
  0.50    0.483         5         0.235         0.260         0.240         0.230         0.035
  0.75    0.735         5         0.565         0.220         0.155         0.050         0.010
  1.00    0.985         2         0.970         0.030         0.000         0.000         0.000
  evaluation seeds 1..5: realised concentration per rho
    rho 0.00:  0.00  0.17  0.00  0.17  0.17   distinct 2
    rho 0.25:  0.00  0.33  0.33  0.17  0.17   distinct 3
    rho 0.50:  0.17  1.00  0.33  0.17  0.17   distinct 3
    rho 0.75:  0.17  1.00  0.33  0.50  1.00   distinct 4
    rho 1.00:  1.00  1.00  0.50  1.00  1.00   distinct 2

=== fraction 0.30  ->  K = 6 degraded RSUs
  rho   E[conc]  distinct           [6]        [5, 1]        [4, 2]     [4, 1, 1]        [3, 3]     [3, 2, 1]  [3, 1, 1, 1]     [2, 2, 2]  [2, 2, 1, 1]
  0.00    0.217         7         0.000         0.000         0.010         0.070         0.015         0.325         0.105         0.105         0.370
  0.25    0.279         9         0.005         0.005         0.100         0.080         0.100         0.350         0.075         0.080         0.205
  0.50    0.372         9         0.020         0.105         0.165         0.120         0.130         0.290         0.020         0.060         0.090
  0.75    0.551         9         0.175         0.205         0.260         0.070         0.130         0.110         0.005         0.030         0.015
  1.00    0.761         4         0.420         0.360         0.190         0.000         0.030         0.000         0.000         0.000         0.000
  evaluation seeds 1..5: realised concentration per rho
    rho 0.00:  0.13  0.13  0.13  0.13  0.27   distinct 2
    rho 0.25:  0.13  0.27  0.40  0.13  0.27   distinct 3
    rho 0.50:  0.27  0.47  0.40  0.27  0.27   distinct 3
    rho 0.75:  0.27  0.47  0.40  0.40  0.67   distinct 4
    rho 1.00:  0.67  0.47  0.40  0.67  0.67   distinct 3
```

Interpretation: at the L12 correlation-sweep operating point (20%, K = 4) the degraded set
can take only five segment shapes, and five evaluation seeds realise just 2-4 distinct
correlation values per rho — the expected concentration rises smoothly with rho
(0.217 -> 0.310 -> 0.483 -> 0.735 -> 0.985) but a single seed samples it very coarsely,
so a gap-vs-rho curve over 5 seeds would be dominated by which shape each seed drew.

Three further facts in the table, stated without remedy (raised as a decision request):
at 5% (K = 1) rho has no effect whatsoever, so the fraction sweep's lowest point contains
no correlated failure even at "high" rho; at 10% (K = 2) every seed is binary, either
both degraded RSUs share a segment or they do not; and at 30% (K = 6) rho = 1 reaches
only 0.761 concentration, because once one segment is full the remainder spills into a
second — whole-segment degradation cannot hold K fixed and be fully concentrated at the
same time. Seed 3's smallest segment has 3 RSUs, which is why it realises [3, 1] (0.50)
at 20% and rho = 1.

### F11 — S2 exit: gradual onset in success_ewma, degradation confined to one segment at rho = 1
Date: 2026-09-13 | Session: S2 | Commit: 559259f
Config: `configs/demo.yaml` (fraction 0.20, onset 400, ramp 300) | Seed(s): 20260903
Command: `python scripts/s2_report.py --config configs/demo.yaml`
         `python scripts/generate_scenario.py --config configs/demo.yaml [--rho 0] --force`
Figure: `figures/s2_success_ewma.png` (generated, not committed)

Numbers:
```
config configs/demo.yaml   seed 20260903   fraction 0.2   segment sizes [5, 5, 5, 5]
no-degradation deadline success: 0.9659

--- rho = 0
  degraded RSUs          : [2, 4, 10, 12]  (segments [3, 2, 0, 3])
  degraded per segment   : [1, 0, 1, 2] of [5, 5, 5, 5]
  correlated picks       : 0   pair concentration 0.167
  success overall        : 0.9211   post-ramp 0.8774   post-ramp on degraded 0.2405
  mean success_ewma        step   degraded   healthy-same-seg   healthy-other
    pre-onset               399      0.947              0.968           0.999
    onset + 10              410      0.947              0.970           1.000
    ramp 1/4                475      0.930              0.980           1.000
    ramp 1/2                550      0.865              0.981           0.960
    ramp end                700      0.562              0.972           0.968
    end                    1199      0.279              0.971           0.980
  latency_dev at end     : degraded 0.331   healthy 0.037

--- rho = 1
  degraded RSUs          : [0, 5, 7, 10]  (segments [0, 0, 0, 0])
  degraded per segment   : [4, 0, 0, 0] of [5, 5, 5, 5]
  correlated picks       : 3   pair concentration 1.000
  success overall        : 0.8886   post-ramp 0.8096   post-ramp on degraded 0.2632
  mean success_ewma        step   degraded   healthy-same-seg   healthy-other
    pre-onset               399      0.975              0.909           0.975
    onset + 10              410      0.981              0.918           0.974
    ramp 1/4                475      0.918              0.998           0.990
    ramp 1/2                550      0.783              0.957           0.977
    ramp end                700      0.586              0.995           0.962
    end                    1199      0.299              0.875           0.969
  latency_dev at end     : degraded 0.327   healthy 0.037

graph sequence sha256 (on-disk scenario, rebuilt from the observable file):
  rho 1.00: f3f247ac6c9becb1d2ddf859bf0b635a3c0d4778076ff028e46cfc0c97130de7  (identical on two runs)
  rho 0.00: a3af5951e1b12d00bc67d0953625a851aa78ec327a9339589cf6daf35fe27100
run.py on the stored rho=1 scenario: 6957 decisions, 20 distinct RSUs, 2.4091 mean candidates
pytest: 137 passed
```

Interpretation: onset is gradual in the observable feature at both rho — the degraded
mean is unchanged 10 s after onset and still 0.93/0.92 a quarter into the ramp, then lags
the true fault by roughly 100-150 s — and at rho = 1 the four degraded RSUs sit in one
segment while the rest of the network stays near 0.97, so the correlated structure H2
needs is present in the data.

Noted, not acted on: the "healthy, same segment" group at rho = 1 is a single RSU (16),
and its line swings between 0.75 and 1.0 *before* onset as well as after, so it is EWMA
scatter on one node rather than leakage; at a feature weight of 0.1 a single RSU's
`success_ewma` is noisy enough that per-node pre-onset dips of 0.2 occur with no
degradation at all. That noise floor is what any detection-latency threshold in S4 has
to sit above.

### F12 — Cold-start control and test conditions: both place cleanly; control is not always feasible
Date: 2026-09-13 | Session: S2b | Commit: e0d1a15
Config: `configs/demo.yaml`, fraction 0.20, one cold-start RSU joining at step 600 |
Seed(s): 1..5, injection draws 0..49 per seed (250 draws per cell)
Command: `python scripts/s2b_cold_start_conditions.py --config configs/demo.yaml`

Numbers:
```
config configs/demo.yaml   seeds 1..5   draws 0..49   fraction 0.2   cold-start nodes 1

=== rho 0
  placement          feasible   classes                         degraded nbrs (mean)   any same-seg   distinct/seed
  healthy_segment       0.884   {'reliable': 221}                              0.000          1.000   [15, 18, 14, 17, 17]
  degraded_segment      1.000   {'degraded': 250}                              0.500          1.000   [18, 18, 17, 19, 19]

=== rho 1
  placement          feasible   classes                         degraded nbrs (mean)   any same-seg   distinct/seed
  healthy_segment       1.000   {'reliable': 250}                              0.000          1.000   [20, 19, 19, 19, 18]
  degraded_segment      1.000   {'degraded': 250}                              2.052          1.000   [17, 19, 19, 18, 18]

regression: default demo scenario (draw 0, uniform) graph sequence sha256 unchanged,
  f3f247ac6c9becb1d2ddf859bf0b635a3c0d4778076ff028e46cfc0c97130de7 (as in F11)
pytest: 147 passed
```

Interpretation: both conditions are pure by construction — every control node is
reliable on a degradation-free segment, and every test node is degraded — and draws place
the new node on 14-20 distinct RSUs per seed, so cold start does not share F10's
quantisation problem.

Two facts S4 will have to handle, recorded without remedy. First, the control condition
is infeasible on 11.6% of draws at rho = 0 (29 of 250), because four independently chosen
degraded RSUs sometimes touch all four segments; the feasible draws are therefore not a
random sample of all draws at rho = 0, while at rho = 1 every draw is feasible. Second, at
rho = 0 the test node has on average only 0.5 degraded same-segment coordination
neighbours, against 2.05 at rho = 1: at low rho the "test" condition mostly offers the
GNN no degraded neighbour to propagate from, so the cold-start test condition only probes
the claim it is meant to probe at high rho.

### F13 — S3 first training run (seed 1): C converges by patience; D hits the 60-epoch cap still improving
Date: 2026-10-13 | Session: S3 | Commit: cc1dfdb (code identical to bce1464 apart from the dataset refactor and tests)
Config: `configs/demo.yaml`, seed 1; `TrainSettings` lr 1e-3, Adam, batch 32 graphs, max 60 epochs,
patience 8, min_delta 1e-4 - all set a priori, shared by C and D, not tuned |
Data: Baseline A dispatch; train rho{0,.25,.5,.75,1} x frac{.1,.2,.3} x draws{0,1} = 30 scenarios
(9000 graphs at stride 4); validation rho{0,.5,1} x frac{.1,.2,.3} draw 40 = 9 scenarios;
held-out rho 1.0, frac 0.2, draw 90 (stride 1)
Command: `python scripts/s3_train.py --config configs/demo.yaml --seed 1` (253 s)
(A 2-epoch plumbing run preceded this; its numbers were not recorded and not used.)

Numbers:
```
parameters           C 1313   D 1614   D/C 1.23x
                          C                     D
untrained val BCE         0.68385               0.93336
best val BCE              0.16614 (epoch 16)    0.16226 (epoch 60)
stopped                   epoch 24, patience    epoch 60, MAX_EPOCHS (best epoch = last)
D val BCE, epochs 50..60 : 0.16352 0.16608 0.16509 0.16424 0.16309 0.16306 0.16614 0.16252 0.16711 0.16596 0.16226
held-out (rho 1.0, frac 0.2, draw 90):
  next-task BCE                 0.16804               0.16845
  diagnostic AUC [450,650)      1.0000                0.9844
  diagnostic AUC [1100,1200)    1.0000                1.0000
validation-scenario diagnostic AUC [450,650) (rho 1.0, frac 0.2, draw 40), per epoch, final:
  C 0.7812 (rose 0.52 -> 0.78 over training)   D 0.5625 (range 0.42-0.72, noisy, no trend)
D edge probe (validation): mean |dtrust| same_segment->0 on all edges 0.0423; ->1 on RSU-RSU edges 0.0199
Calibration, held-out (predicted bin: n, mean_pred, observed):
 C: 0.0-0.1 49 .083 .000 | .1-.2 371 .156 .140 | .2-.3 592 .249 .297 | .3-.4 679 .345 .317 | .4-.5 437 .438 .474
    .5-.6 237 .532 .616 | .6-.7 86 .626 .616 | .7-.8 138 .764 .797 | .8-.9 1391 .867 .874 | .9-1.0 19913 .976 .976
 D: 0.0-0.1 46 .075 .022 | .1-.2 238 .156 .092 | .2-.3 317 .251 .211 | .3-.4 373 .352 .332 | .4-.5 493 .450 .323
    .5-.6 422 .547 .429 | .6-.7 373 .651 .552 | .7-.8 494 .752 .692 | .8-.9 1308 .861 .839 | .9-1.0 19829 .982 .979
```
D44 fallback checks for D: edge probe >= 0.01 in both directions - PASS; held-out BCE below the
untrained model's - PASS. The architecture stays GINEConv.

Interpretation: C converged; D did NOT converge under the a-priori budget - its validation BCE
was still falling at the epoch cap (0.1623 vs C's 0.1661), so "C and D converge" is unmet and
the comparison above is between a converged C and a truncated D. D over-predicts trust at
mid-range (bins .4-.8 observed below predicted), C is closer to calibrated there. One held-out
scenario (4 degraded RSUs) says nothing about H2; the per-epoch validation-scenario AUC (0.78
vs 0.56) and the held-out AUC (1.00 vs 0.98) disagree by a wide margin for both models,
showing the single-scenario AUC is dominated by which RSUs happened to degrade.

### F14 — S3 exit run (seed 1, cap 200): C and D both converge by patience on identical data
Date: 2026-10-13 | Session: S3 | Commit: 808f29d
Config: `configs/demo.yaml`, seed 1; settings as F13 except `max_epochs` 200 (D45) | Data: as F13
Command: `python scripts/s3_train.py --config configs/demo.yaml --seed 1`
Reproducibility: D's val BCE at epoch 60 is 0.1623, equal to F13's 0.16226; C's curve is identical to F13.

Numbers:
```
                      C                          D
parameters            1313                       1614   (D/C 1.23x)
untrained val BCE     0.6838                     0.9334
stopped               epoch 24 (patience)        epoch 143 (patience)
best epoch            16                         135
best val BCE          0.16614                    0.13845
loss curve (epoch: train/val)
  C   1: .2839/.1756   2: .1713/.1709   5: .1659/.1689   10: .1641/.1667   16: .1638/.1661   24: .1634/.1663
  D   1: .2668/.2188   5: .2044/.1986   10: .1857/.1891   24: .1705/.1773   60: .1539/.1623   100: .1416/.1527
      120: .1334/.1437   135: .1291/.1385   143: .1290/.1392
held-out (rho 1.0, frac 0.2, draw 90, 4 degraded RSUs of 20):
  next-task BCE                  0.16804                    0.15568
  diagnostic AUC [450,650)       1.0000                     1.0000
  diagnostic AUC [1100,1200)     1.0000                     1.0000
validation-scenario diagnostic AUC [450,650) (draw 40), per epoch: C 0.52 -> 0.78 (final 0.7812);
  D wanders 0.42-0.88 with no clear trend, final 0.7812 (epoch 143: 0.78)
D edge probe (validation): |dtrust| same_segment->0: 0.0571; RSU-RSU same_segment->1: 0.0405  (threshold 0.01: PASS)
Calibration, held-out (bin: n, mean_pred, observed):
 C: 0.0-0.1 49 .083 .000 | .1-.2 371 .156 .140 | .2-.3 592 .249 .297 | .3-.4 679 .345 .317 | .4-.5 437 .438 .474
    .5-.6 237 .532 .616 | .6-.7 86 .626 .616 | .7-.8 138 .764 .797 | .8-.9 1391 .867 .874 | .9-1.0 19913 .976 .976
 D: 0.0-0.1 327 .055 .113 | .1-.2 352 .150 .239 | .2-.3 438 .253 .265 | .3-.4 489 .351 .385 | .4-.5 404 .448 .473
    .5-.6 389 .545 .506 | .6-.7 344 .652 .677 | .7-.8 448 .754 .734 | .8-.9 1128 .858 .861 | .9-1.0 19574 .986 .984
pytest: 159 passed
```
Interpretation: both models converge and both calibrate well in the dominant 0.9-1.0 bin; D reaches a lower next-task BCE (0.138 vs 0.166 validation, 0.156 vs 0.168 held-out) and
is better calibrated at the low-trust end than C was at the same stage, which is a statement about
predicting the next outcome, not about H2. The held-out diagnostic AUC is 1.0 for both on a single
scenario with 4 degraded RSUs - a ceiling that cannot separate C from D - while the same metric
on the draw-40 validation scenario sits at ~0.78 for both, so a one-scenario AUC mostly reflects
which RSUs degraded; the D43 multi-draw protocol in S4 is what measures the comparison.

### F15 — S4 preparation: per-seed C and D for seeds 2-5 trained with the unchanged S3 protocol
Date: 2026-10-14 | Session: S4 | Commit: 548c6d6
Config: `configs/demo.yaml`, seeds 2, 3, 4, 5; `TrainSettings` exactly as D45 (lr 1e-3, Adam, batch 32,
max 200, patience 8, min_delta 1e-4), no setting touched | Data: as F13/F14 per seed
Command: `python scripts/s3_train.py --config configs/demo.yaml --seed <n>` (4 runs in parallel, OMP/MKL
threads 3 each; seed 1 was trained earlier at default threads - thread count not equalised)

Why this run exists: S3 trained and checkpointed seed 1 only (`checkpoints/` held s3-seed1-{C,D}.pt),
but D43 section 2 requires one frozen model per seed per variant for seeds 1-5. The seed-1 checkpoints
are used as they are; seeds 2-5 were trained once each, no re-runs, no selection among runs. The
traces for seeds 2 and 3 on disk were stale 300-step S1 traces and were regenerated (gitignored,
reproducible from config + seed); seeds 4 and 5 traces were generated new.

Numbers (stopped = epoch, all by patience; BCE = validation next-task BCE at the best epoch):
```
seed  variant  best_ep stopped  untrained_val  best_val   held-out BCE  held-out AUC[450,650)
1     C           16     24       0.6838        0.16614    0.16804       1.0000
1     D          135    143       0.9334        0.13845    0.15568       1.0000
2     C           47     48       0.7527        0.14829    0.16773       1.0000
2     D           44     52       3.8947        0.20394    0.20472       0.9062
3     C           24     24       0.7988        0.16789    0.18315       1.0000
3     D           78     86       2.5810        0.17090    0.21488       0.9375
4     C           16     21       0.6671        0.18668    0.17118       1.0000
4     D          133    141       0.3448        0.16280    0.15460       1.0000
5     C           21     22       0.6422        0.18586    0.17221       1.0000
5     D          118    126       0.5050        0.16622    0.15965       1.0000
parameters C 1313, D 1614 (1.23x) for every seed
D44 fallback checks (validation): edge probe |dtrust| same_segment->0 / RSU-RSU ->1:
  seed 1 0.0571/0.0405  seed 2 0.1330/0.0353  seed 3 0.0632/0.0358  seed 4 0.1172/0.0474  seed 5 0.0677/0.0573
  all >= 0.01: PASS.  best val BCE below untrained val BCE: PASS on all five seeds.
```
Interpretation: all ten models converged by patience and D passes both pre-committed checks on every
seed, so the architecture stays GINEConv (D44). D on seeds 2 and 3 stopped early (epochs 52, 86) at a
worse validation BCE than C (0.204 vs 0.148; 0.171 vs 0.168) - a property of those D runs recorded
here, not corrected (D43 section 5: no model or hyperparameter change between seeds).

### F16 — S4 raw results: frozen C and D, D43 conditions, seeds 1-5 (RAW - no verdict recorded here)
Date: 2026-10-14 | Session: S4 | Commit: 548c6d6
Config: `configs/demo.yaml`, seeds 1-5, injection draws 100-149 (50 per seed per cell), fraction 0.20 (K=4) |
Checkpoints: `checkpoints/s3-seed{1..5}-{C,D}.pt` (F14, F15), frozen, no retraining
Commands: `python scripts/s4_eval.py --config configs/demo.yaml --seeds 1..5` ;
          `python scripts/s4_analyse.py --seeds 1 2 3 4 5` (full text: results/s4/report-5seeds.txt, not committed)
Figures (not committed): figures/s4_gap_vs_correlation.png, s4_cold_start.png, s4_collusion.png
Conventions as D43 section 3. Intervals: Student-t over the 5 per-seed values (t=2.776). Two
implementation choices D43 left open, made before any C-vs-D number: (i) detection threshold tau is the
1st percentile of a model's trust over [100,400) for all RSUs of that seed's 250 sweep-condition draws
(no collusion/cold start, so it depends on no injected fault); (ii) a never-detected RSU has latency 399
(censored at step 799), per-draw latency is the mean over that draw's 4 degraded RSUs.
D = AUC_D - AUC_C throughout (paired per draw, averaged within seed).

A. Correlation sweep, diagnostic AUC [450,650), degraded vs reliable:
```
rho   c(real.)  AUC_C mean sd         AUC_D mean sd         D-C mean   sd      95% CI              sep. CIs overlap
0.00  0.216     0.9803 0.0338         0.9286 0.0145         -0.0517    0.0310  [-0.0902, -0.0131]  yes
0.25  0.333     0.9801 0.0345         0.9320 0.0129         -0.0481    0.0273  [-0.0821, -0.0142]  yes
0.50  0.500     0.9806 0.0306         0.9366 0.0156         -0.0440    0.0289  [-0.0798, -0.0082]  yes
0.75  0.717     0.9836 0.0228         0.9427 0.0116         -0.0409    0.0246  [-0.0714, -0.0104]  yes
1.00  0.956     0.9852 0.0195         0.9555 0.0122         -0.0297    0.0198  [-0.0542, -0.0051]  yes
per-seed D-C (seeds 1..5):
0.00 [+0.0016 -0.0625 -0.0578 -0.0597 -0.0800]   0.25 [-0.0003 -0.0591 -0.0512 -0.0631 -0.0669]
0.50 [+0.0034 -0.0494 -0.0456 -0.0750 -0.0534]   0.75 [-0.0016 -0.0522 -0.0441 -0.0678 -0.0391]
1.00 [-0.0025 -0.0394 -0.0338 -0.0541 -0.0188]
per-seed AUC_C at rho 1.0 [0.9528 1.0 1.0 0.9906 0.9825]; AUC_D [0.9503 0.9606 0.9662 0.9366 0.9638]
plateau [1100,1200) (secondary) D-C: rho0 -0.0129 CI[-0.0297,+0.0038]; .25 -0.0123 [-0.0283,+0.0037];
  .5 -0.0111 [-0.0242,+0.0021]; .75 -0.0136 [-0.0277,+0.0004]; 1.0 -0.0123 [-0.0257,+0.0010]
  (AUC_C plateau 1.0000 in every cell; AUC_D plateau 0.9864-0.9889)
W3 per-seed slope of per-draw D-C on realised c (250 draws/seed): [-0.0020 +0.0191 +0.0488 +0.0219 +0.0622]
  mean +0.0300 sd 0.0255, 95% CI [-0.0017, +0.0617]
D-C by realised c (all rho pooled; draws total; mean of per-seed means; CI over 5 seeds):
  c=0     32 draws  -0.0586 [-0.1237,+0.0064]    c=1/6  331  -0.0438 [-0.0783,-0.0093]
  c=1/3  193        -0.0655 [-0.1181,-0.0130]    c=1/2  266  -0.0470 [-0.0843,-0.0098]
  c=1    428        -0.0271 [-0.0523,-0.0020]
```
B. Detection latency, steps after onset (400), mean over seeds; detected fraction in brackets:
```
rho   C latency sd   (detected)    D latency sd    (detected)    D-C latency  95% CI            sep. CIs overlap
0.00  137.7 10.2 (1.000)           234.1 22.2 (0.974)           +96.5   [+64.0,+129.0]    no
0.25  137.2  9.5 (1.000)           230.2 23.4 (0.973)           +93.0   [+61.9,+124.2]    no
0.50  137.8  9.3 (1.000)           224.9 23.2 (0.976)           +87.1   [+55.8,+118.4]    no
0.75  136.5  5.7 (1.000)           220.9 23.3 (0.976)           +84.4   [+55.7,+113.0]    no
1.00  135.3  5.5 (1.000)           212.0 24.4 (0.982)           +76.7   [+48.2,+105.2]    no
tau (1st pct of pre-onset trust) per seed: C [0.8393 0.8396 0.8396 0.8580 0.8577]  D [0.5808 0.6952 0.5774 0.5292 0.6128]
```
C. Collusion (1 group of 3, under_report 0.5), AUC compromised vs reliable, [200,400):
```
rho   AUC_C mean sd        AUC_D mean sd        D-C mean   sd      95% CI              per-seed D-C
0.00  0.7344 0.1304        0.7913 0.1614        +0.0569    0.2091  [-0.2027, +0.3166]  [+0.3282 -0.2523 +0.0272 +0.0549 +0.1267]
1.00  0.7332 0.1308        0.7958 0.1654        +0.0626    0.2325  [-0.2261, +0.3513]  [+0.3538 -0.2877 +0.0287 +0.0682 +0.1497]
per-seed AUC_C (rho0) [0.5882 0.7954 0.7928 0.8877 0.6077]  AUC_D [0.9164 0.5431 0.8200 0.9426 0.7344]
```
D. Cold start (1 joiner at step 600), AUC test vs control of 1 - joiner mean trust. D42 accounting:
```
rho 0.00: attempted 250/condition; control infeasible 25 (aggregated 225); test infeasible 0 (aggregated 250);
          paired (both feasible) aggregated 225; per seed [46 45 49 43 42]
rho 1.00: attempted 250/condition; control infeasible 0; test infeasible 0; paired aggregated 250 (50 per seed)
                          AUC_C mean sd       AUC_D mean sd       D-C mean   sd      95% CI              per-seed D-C
rho 0  [600,630)           0.7951 0.0889       0.6520 0.1420       -0.1431    0.0641  [-0.2226, -0.0635]  [-0.1843 -0.1037 -0.0820 -0.2347 -0.1105]
rho 0  [600,660) secondary  0.9348 0.0742       0.8232 0.0812       -0.1116    0.1098  [-0.2479, +0.0248]  [-0.0198 -0.0795 -0.0296 -0.2883 -0.1406]
rho 1  [600,630)           0.8193 0.1020       0.7920 0.0957       -0.0273    0.0533  [-0.0935, +0.0389]  [-0.1156 +0.0272 -0.0280 -0.0104 -0.0096]
rho 1  [600,660) secondary  0.9419 0.0210       0.8976 0.0501       -0.0443    0.0482  [-0.1041, +0.0155]  [-0.0272 -0.0016 -0.0256 -0.1268 -0.0404]
mean joiner trust [600,630): rho0  C control 0.9737 test 0.8853 | D control 0.9127 test 0.8706
                             rho1  C control 0.9757 test 0.8888 | D control 0.9245 test 0.7936
```

### F17 — S4 verdict against D43 (applied to F16 after F16 was committed in 09cc126)
Date: 2026-10-14 | Session: S4 | Commit: 09cc126 (numbers) | Config/seeds: as F16
Command: `python scripts/s4_analyse.py --seeds 1 2 3 4 5 --verdict`

```
H2 core, correlation sweep:   rho=1.0  Delta -0.0297  CI [-0.0542, -0.0051]   W1 False (lo > 0 fails)
                              rho=0.0  Delta -0.0517  CI [-0.0902, -0.0131]   W2 False
                              W3 mean slope +0.0300  CI [-0.0017, +0.0617]    W3 False (CI includes 0)
  NEGATIVE check: hi < 0 and mean <= -0.02 at rho=1.0 -> TRUE.   VERDICT: NEGATIVE (C beats D)
Collusion rho=0:  Delta +0.0569 CI [-0.2027, +0.3166]  -> NOISY
Collusion rho=1:  Delta +0.0626 CI [-0.2261, +0.3513]  -> NOISY
Cold start rho=1: Delta -0.0273 CI [-0.0935, +0.0389]  -> NULL (hi < 0.05)
(info) cold start rho=0: Delta -0.1431 CI [-0.2226, -0.0635]
H2 holds in full (three WIN): False.  Combination: core NEGATIVE, collusion NOISY x2, cold start NULL.
```
Interpretation: on the pre-registered metric, the graph model D is *worse* than the node-local MLP C
at every rho in the sweep (by 0.03-0.05 AUC), D's deficit is smallest where failures are most
correlated (-0.052 at rho 0 to -0.030 at rho 1; per-seed slopes mostly positive, mean +0.030 but its
interval includes 0), and seed 1 is the only seed where D is level with C. The gap is therefore not
structural in the sense H2 claims: D does not beat C at any correlation, and the narrowing with
correlation is not statistically established. Cold start is a NULL at rho=1 (an advantage of 0.05 is
ruled out) and D is significantly worse at rho=0. Collusion cannot separate the models: per-seed gaps
run from -0.29 to +0.35 and the seed-level sd of the gap is 0.21-0.23, so it is NOISY. Secondary
metric, latency: C detects degradation ~77-97 steps earlier than D at every rho (CIs do not overlap)
with all degraded RSUs detected, again favouring C. Caveats that bear on the reading, not
re-analysed: (i) C reaches AUC 0.98 so there is little headroom, and the gap could partly reflect
D's weaker fit on seeds 2-3 (F15) rather than a property of message passing; (ii) the F14 held-out
AUC ceiling of 1.0 for both was a one-scenario artefact, as predicted; (iii) seeds 2-5 were trained
for S4 (F15), under identical settings. D43 section 5 applies: no window, threshold, seed, draw, model
or hyperparameter change; any response (e.g. staggered onset D31(c), observation-count feature D38(d),
other architecture) is a new DECISIONS entry labelled post-hoc and never replaces this result.
The NOISY-response extension to seeds 6-10 is not triggered by the core (NEGATIVE) and was not run for
collusion (see the decision request).

### F18 — S5 raw diagnostics: same_segment use, lead-lag of own vs neighbour evidence, thread sensitivity (RAW - no interpretation here)
Date: 2026-10-14 | Session: S5 | Commit: a746c02 (script scripts/s5_diagnostics.py; protocol D48 committed 568e536 before any number)
Config: `configs/demo.yaml`, seeds 1-5, S4 evaluation scenarios (draws 100-149) | Frozen checkpoints of F15/F14
Commands: `python scripts/s5_diagnostics.py check1|check2|check3`;
          `python scripts/s4_eval.py --config configs/demo.yaml --seeds 1 --workers 4 --threads 3 --out-dir results/s5/raw_t3`

Check 1 - is `same_segment` correct, and does D's output depend on it?
```
structural (steps 450..645 stride 5, rho 1.0, draws 100-104, per seed): RSU-RSU edge-instances 18400 per seed,
  same-segment instances 12000 / 12000 / 12000 / 12800 / 11600 (seeds 1-5); mismatches vs segment equality: 0 on all
  seeds; vehicle-RSU edge-instances 56630-58170 per seed, nonzero same_segment: 0.
behavioural (mean |d RSU trust|; window AUC [450,650) degraded vs reliable, mean of the 5 draws):
 seed  zero-all  ->1 (RSU-RSU)  permuted   AUC base   AUC zeroed   AUC permuted
 1     0.0613    0.0418         0.0328     0.9000     0.6312       0.9406
 2     0.1025    0.0259         0.0238     0.9469     0.9719       0.9719
 3     0.0574    0.0451         0.0443     0.9656     0.9469       0.9531
 4     0.1030    0.0654         0.0490     0.8969     0.8688       0.8688
 5     0.0669    0.0685         0.0457     0.9625     0.9719       0.9938
D44 probe (validation scenarios, F15): zero-all / ->1: seed1 0.0571/0.0405  seed2 0.1330/0.0353  seed3 0.0632/0.0358
  seed4 0.1172/0.0474  seed5 0.0677/0.0573.   D48 defect criterion (any mismatch, or mean |dtrust| < 0.01): not met.
```
Check 2 - first movement of `success_ewma` (seconds after onset at 400; dt = 1 s), sweep scenarios, 1000 degraded
RSU-instances per rho (5 seeds x 50 draws x 4), 95% t-intervals over 5 seeds. "own-lead" = mean (or min) over the RSU's
degraded same-segment neighbours of their first-move time, minus the RSU's own first-move time (positive = own first).
```
primary rule (own-RSU 1st percentile of [100,400), 10 consecutive steps):
 rho   own first move (s)   ramp level  own-lead, mean over nbrs          own-lead, min over nbrs (biased early)  own censored
 0.00  111.8 sd 9.2         0.37        +0.2 sd 1.8 CI [-2.0, +2.5]       -3.6 CI [-4.4, -2.8]                     0.000
 0.25  109.5 sd 7.1         0.37        +0.5 sd 2.6 CI [-2.7, +3.6]       -9.4 CI [-14.6, -4.3]                    0.000
 0.50  110.9 sd 6.3         0.37        -0.2 sd 4.3 CI [-5.6, +5.2]       -18.5 CI [-25.7, -11.3]                  0.000
 0.75  110.5 sd 7.1         0.37        +0.5 sd 5.3 CI [-6.0, +7.0]       -24.1 CI [-32.8, -15.5]                  0.000
 1.00  110.1 sd 7.1         0.37        -0.3 sd 6.0 CI [-7.8, +7.2]       -33.0 CI [-44.8, -21.2]                  0.000
 share of degraded RSUs with: >=1 degraded same-segment neighbour / only healthy ones / none:
   rho0 0.41/0.59/0.00   rho.25 0.53/0.47/0.00   rho.5 0.66/0.34/0.00   rho.75 0.81/0.19/0.00   rho1 0.95/0.05/0.00
 fraction of healthy same-segment neighbours that also "move" in [400,799] under the same rule: 0.542 0.546 0.542 0.530 0.512
secondary rule (drop >= 0.10 below own pre-onset mean, 10 consecutive):
 rho   own first move (s)   ramp level  own-lead, mean over nbrs          own-lead, min over nbrs                 healthy nbrs "moving"
 0.00  122.8 sd 7.3         0.41        -0.3 sd 1.9 CI [-2.7, +2.1]       -4.1 CI [-6.1, -2.0]                     0.429
 0.25  121.3 sd 5.4         0.40        +0.3 sd 2.0 CI [-2.2, +2.7]       -8.7 CI [-14.9, -2.4]                    0.437
 0.50  122.1 sd 6.6         0.41        -0.1 sd 3.9 CI [-4.9, +4.8]       -17.8 CI [-27.8, -7.8]                   0.434
 0.75  121.7 sd 6.1         0.41        +1.1 sd 5.1 CI [-5.2, +7.4]       -22.0 CI [-33.5, -10.6]                  0.431
 1.00  120.0 sd 7.2         0.40        +0.9 sd 5.5 CI [-6.0, +7.8]       -30.1 CI [-46.5, -13.7]                  0.447
```
Check 3 - seed-1 checkpoints evaluated with 3 torch threads vs the 1-thread F16 run:
```
546 feasible scenarios compared (4 infeasible cold-start draws identical in both runs)
max |trust difference|: C 0.0   D 6.855e-07;   max |AUC difference| over 250 sweep draws, C and D: 0.0
beyond rounding (D48: max |dtrust| > 1e-5 or |dAUC| >= 5e-5): False
```
Not measured: the effect of the training-time thread count (seed 1 at default threads, seeds 2-5 at 3), which
cannot be tested without retraining.

### F19 — S5 interpretation of F18, and the gap-narrowing observation on F16/F17 (written after F18 was committed in 1bcedac)
Date: 2026-10-14 | Session: S5 | Commit: 1bcedac (numbers)

Observation on F16/F17, added as asked: the D-C gap narrows from -0.052 at rho 0 (realised c 0.22) to -0.030 at
rho 1 (c 0.96), but the per-seed slope on realised c is +0.030 with 95% CI [-0.002, +0.062], so it is not
significant. It is a direction consistent with H2 and does not support it; D is below C at every rho.

Check 1 - no defect. `same_segment` equals segment equality on every sampled RSU-RSU edge (0 mismatches over
92,000 instances) and is 0 on every vehicle-RSU edge. D's output moves by 0.024-0.103 mean |trust| when it is
zeroed, set to 1 or permuted, on every seed (threshold 0.01), so D reads the feature. But removing it does not
systematically cost detection: window AUC after zeroing is higher on seeds 2 and 5 and lower on 1, 3, 4; after
permuting it is higher on seeds 1, 2, 5 and lower on 3, 4 (only 5 draws per seed). D uses the segment signal; there is no evidence it uses it to
separate degraded from reliable RSUs better. The architecture and graph are not the cause of the S4 negative.

Check 2 - neighbour evidence does not lead own-node evidence. A degraded RSU's own `success_ewma` first moves
about 110 s after onset (ramp level 0.37) at every rho; the mean first-move time of its degraded same-segment
neighbours differs from its own by -0.3 to +0.5 s (CIs all include 0, widest [-7.8, +7.2] at rho 1; secondary rule
the same within +-1.1 s). Under D48's pre-declared reading this is "simultaneous": degraded RSUs share their
onset (D31) and carry the same fault, so neighbouring degraded RSUs show it at the same moment as the RSU itself.
The "min over neighbours" lead is negative and grows with rho (-3.6 to -33 s), as it must for an order
statistic over more neighbours (more degraded neighbours at higher rho); it is not a lead. Two structural facts
limit what message passing could add here: at rho 0, 59% of degraded RSUs have only healthy same-segment
neighbours (no degraded one to learn from), and at rho 1, 95% have degraded neighbours but those move
simultaneously. In this scenario own-node evidence is therefore never later than neighbour evidence, so a
lead-time advantage for D is not available by construction. Caveats: the pre-declared "moves" rule is loose -
43-55% of healthy neighbours also trip it in [400,799] - so the absolute times are first-excursion times, not
clean detections; the own-vs-neighbour comparison applies the same rule to both sides. The lead-lag result
explains why D has no advantage; it does not by itself explain why D detects 77-97 s later than C (F16B). That
is unmeasured; a candidate (D's aggregation over many healthy neighbours diluting its own-node signal) is a
hypothesis, not a finding.

Check 3 - inference is not thread-sensitive: the seed-1 checkpoints give identical C and D outputs to within
6.9e-7 at 3 threads versus 1, with no AUC change. The training-time thread difference (seed 1 default, seeds 2-5
three threads) is not tested and stays a stated limitation of F15.

Conclusion: neither check finds a defect, so the S4 NEGATIVE (F17) stands unmodified. Under the S5 brief's second
branch the paper moves to the H2 fallback framing (D49).

### F20 — S6 tuning grid on the tuning seed (RAW, recorded before the weights are used anywhere)
Date: 2026-10-14 | Session: S6 | Commit: see `git log` for the commit that adds this entry
Config: `configs/demo.yaml`, tuning seed 100 (excluded from all later evaluation), rho 1.0, fraction 0.20,
injection draws 0, 1, 2, closed loop with the dwell filter and handoff on (D51) | Protocol: D50 item 4, unchanged
Models: seed-100 C and D trained once with the D45 settings (`python scripts/s3_train.py --config configs/demo.yaml --seed 100`):
  C stopped epoch 48 (best 46, val BCE 0.1762), D stopped epoch 105 (best 97, val BCE 0.1607), both by patience;
  D edge probe 0.0853 / 0.0561 (>= 0.01, D44 PASS); held-out AUC [450,650) 1.0000 for both (one scenario).
Command: `python scripts/s6_tune.py --config configs/demo.yaml --workers 12 --write-weights` (results/s6/tuning-grid.json)
Objective: deadline success of dispatched tasks over steps [400,1200), mean over the three draws. 432 runs.

Numbers (all grid points):
```
A  (rows: beta, gamma; columns: alpha; value: mean post-onset deadline success over draws 0,1,2)
   beta gamma | a=0    
   0.25 0.15  | 0.8559 
   0.25 0.3   | 0.8633 
   0.25 0.6   | 0.8670 
   0.5  0.15  | 0.8511 
   0.5  0.3   | 0.8559 
   0.5  0.6   | 0.8633 
   1    0.15  | 0.8462 
   1    0.3   | 0.8511 
   1    0.6   | 0.8559 
   chosen (alpha,beta,gamma) = (0, 0.25, 0.6), objective 0.8670, per-draw [0.8834, 0.8416, 0.876]

B_oracle  (rows: beta, gamma; columns: alpha; value: mean post-onset deadline success over draws 0,1,2)
   beta gamma | a=0.25   a=0.5    a=1      a=2      a=4    
   0.25 0.15  | 0.8957   0.8957   0.8957   0.8957   0.8957 
   0.25 0.3   | 0.8995   0.8995   0.8995   0.8995   0.8995 
   0.25 0.6   | 0.8996   0.9009   0.9009   0.9009   0.9009 
   0.5  0.15  | 0.8920   0.8925   0.8925   0.8925   0.8925 
   0.5  0.3   | 0.8943   0.8957   0.8957   0.8957   0.8957 
   0.5  0.6   | 0.8962   0.8995   0.8995   0.8995   0.8995 
   1    0.15  | 0.8776   0.8892   0.8895   0.8895   0.8895 
   1    0.3   | 0.8812   0.8920   0.8925   0.8925   0.8925 
   1    0.6   | 0.8854   0.8943   0.8957   0.8957   0.8957 
   chosen (alpha,beta,gamma) = (0.5, 0.25, 0.6), objective 0.9009, per-draw [0.9432, 0.8516, 0.9078]

C  (rows: beta, gamma; columns: alpha; value: mean post-onset deadline success over draws 0,1,2)
   beta gamma | a=0.25   a=0.5    a=1      a=2      a=4    
   0.25 0.15  | 0.8949   0.8973   0.8986   0.8992   0.8987 
   0.25 0.3   | 0.8951   0.8982   0.8992   0.8993   0.8991 
   0.25 0.6   | 0.8923   0.8985   0.8999   0.8999   0.8994 
   0.5  0.15  | 0.8872   0.8945   0.8968   0.8980   0.8988 
   0.5  0.3   | 0.8889   0.8949   0.8973   0.8986   0.8992 
   0.5  0.6   | 0.8878   0.8951   0.8982   0.8992   0.8993 
   1    0.15  | 0.8733   0.8851   0.8933   0.8963   0.8972 
   1    0.3   | 0.8758   0.8872   0.8945   0.8968   0.8980 
   1    0.6   | 0.8780   0.8889   0.8949   0.8973   0.8986 
   chosen (alpha,beta,gamma) = (1, 0.25, 0.6), objective 0.8999, per-draw [0.9406, 0.8516, 0.9073]

D  (rows: beta, gamma; columns: alpha; value: mean post-onset deadline success over draws 0,1,2)
   beta gamma | a=0.25   a=0.5    a=1      a=2      a=4    
   0.25 0.15  | 0.8906   0.8922   0.8920   0.8915   0.8912 
   0.25 0.3   | 0.8910   0.8930   0.8938   0.8926   0.8919 
   0.25 0.6   | 0.8878   0.8919   0.8937   0.8938   0.8926 
   0.5  0.15  | 0.8814   0.8891   0.8912   0.8912   0.8909 
   0.5  0.3   | 0.8838   0.8906   0.8922   0.8920   0.8915 
   0.5  0.6   | 0.8838   0.8910   0.8930   0.8938   0.8926 
   1    0.15  | 0.8685   0.8788   0.8867   0.8891   0.8896 
   1    0.3   | 0.8722   0.8814   0.8891   0.8912   0.8912 
   1    0.6   | 0.8746   0.8838   0.8906   0.8922   0.8920 
   chosen (alpha,beta,gamma) = (2, 0.25, 0.6), objective 0.8938, per-draw [0.9347, 0.8471, 0.8997]

```
B is not tuned separately (D50): its trust term is the constant cert_valid = 1, so it takes A's weights.
E takes D's weights. Chosen weights are in `configs/weights.yaml`.
Interpretation: the weights matter much less than the choice of trust source - the best A point (0.8670) is
0.03 below the weakest-tuned trust variant, while within C, D and B-oracle the surface is flat (best vs
config default (1, 0.5, 0.3): C 0.8999 vs 0.8973, D 0.8938 vs 0.8922, B-oracle 0.9009 vs 0.8957), and several points tie
to four decimals (ties go to the first grid point, D50).
All three trust-bearing variants sit near 0.89-0.90 and B-oracle is only about 0.001-0.007 above C and D on the
tuning seed, so on this seed a learned trust head recovers nearly all of what perfect fault knowledge could.
One seed, three draws; per-draw values differ by up to 0.09, so differences of ~0.005 between variants here are
inside the draw-to-draw spread and say nothing about evaluation seeds.

### F21 — S6 engine checks and limitations
Date: 2026-10-14 | Session: S6 | Commit: see `git log`
1. Equivalence (tests/test_offload.py, and a full-size check on seed 1, rho 1.0, fraction 0.2, draw 0, 1200 steps,
   6957 tasks): closed-loop A and B with the dwell filter and handoff off reproduce the open-loop task stream, RSU
   features and task demand exactly; the open-loop generator still gives graph-sequence hash f3f247ac...30de7
   (F11) after the simulator change. B's decisions equal A's, as D50 states.
2. The dwell filter and the handoff rule are inert on that scenario: over 6880 decisions, dwell exclusions 0,
   fallbacks 0, relayed 0, lost 0 (same deadline success, 0.9036, as the open loop). With a 400 m radius, speeds
   <= 22 m/s and completions <= 1 s, a task seldom outlives its vehicle's dwell. Both rules are implemented and
   unit-tested on hand-made geometry (tests/test_offload.py); their counters are reported with every closed-loop
   run (`EngineCounters`), so any later claim that they matter must show the counts.
3. Limitation (D50 item 5): the trust heads were trained on Baseline-A open-loop dispatch; the closed loop changes
   which RSUs receive tasks and so which RSUs' `success_ewma` and `latency_dev` keep being updated (an RSU that
   stops receiving tasks stops producing evidence, so its features freeze). That is a distribution shift between
   training and evaluation. True load does not respond to dispatch in this world (D32), so the shift is in the
   observation and feedback distribution, not in load.

### F22 — S6 demo scenario: closed-loop A vs C vs D on one degrading backhaul segment
Date: 2026-10-14 | Session: S6 | Commit: see `git log`
Config: `configs/demo.yaml`, seed 100 (tuning seed, not an evaluation seed), rho 1.0, fraction 0.20, injection draw 7
(fixed before running; the tuning draws were 0-2), closed loop, dwell filter and handoff on, weights from
`configs/weights.yaml` (F20) | One scenario, one run per variant - a demonstration, not an evaluation.
Command: `python scripts/s6_demo.py --config configs/demo.yaml --weights-file configs/weights.yaml`
Figure: `figures/s6_demo.png` (generated, not committed); numbers: results/s6/demo.txt

Numbers:
```
degraded RSUs [0, 5, 7, 16], all on backhaul segment 3 (onset 400, ramp 300); realised pair concentration 1.000
variant  weights (a,b,g)   tasks  success all  success post-onset  post-onset tasks to degraded RSUs (share)
A        0,    0.25, 0.6   6873   0.8980      0.8593              1094 of 4597 (0.2380)
C        1,    0.25, 0.6   6873   0.9259      0.9012               715 of 4597 (0.1555)
D        2,    0.25, 0.6   6873   0.9185      0.8908               767 of 4597 (0.1668)
phase               share of dispatched tasks to degraded RSUs / deadline success
pre-onset [0,400)   A 0.216 / 0.976   C 0.208 / 0.976   D 0.222 / 0.975
ramp [400,700)      A 0.257 / 0.927   C 0.201 / 0.942   D 0.209 / 0.934
plateau [700,1200)  A 0.226 / 0.819   C 0.128 / 0.877   D 0.142 / 0.865
mean trust of the degraded RSUs (step 399 / 550 / 700 / 1199):  C 0.909 / 0.879 / 0.469 / 0.306   D 0.938 / 0.840 / 0.434 / 0.283
mean trust of the healthy RSUs (step 399 / 1199):               C 0.981 / 0.958                  D 0.978 / 0.896
engine counters (identical across the three runs): decisions 6873, dwell exclusions 52, dwell fallbacks 0, relayed 2, lost to handoff 0
```
Interpretation: in this one scenario trust for the degraded segment falls from ~0.9 to ~0.3 under both C and D
while healthy RSUs stay at 0.9-0.98, dispatch shifts away from the degraded RSUs (plateau share 0.23 under A vs 0.13
under C and 0.14 under D) and plateau deadline success is 0.82 under A vs 0.88 (C) and 0.87 (D). The shift is partial,
not total: the degraded RSUs still receive about 13-14% of plateau tasks, because trust is one weighted term against
latency and load and many vehicles have a degraded RSU as their only or best in-range candidate (mean 2.4 candidates,
F11). C is not behind D here (consistent with F17). The dwell filter removed an RSU on 52 decisions and the relay rule
fired twice, so both are live but rare on this seed. D's mean healthy-RSU trust drifts down to 0.90 late in the run
(C: 0.96), a possible sign of the training/evaluation dispatch shift (F21 item 3); it is not investigated here.
One scenario says nothing about H1/H3 magnitudes; those are S9's.

### F23 — S7 Layer 1: exactness on every logged decision, what decides decisions, and four real explanations
Date: 2026-10-14 | Session: S7 | Commit: see `git log`
Config: `configs/demo.yaml`, seed 100, rho 1.0, fraction 0.20, injection draw 7, closed loop with the D52 weights; one run
each of Variant A, C and D (decision logs, schema 2 per D53) | Commands: `python scripts/s6_run.py ... --variant <A|C|D> --weights-file
configs/weights.yaml --log`; `python explain.py --survey --log <file>`; `python scripts/s7_examples.py`

Exactness: `explain_decision` recomputes every candidate's `alpha*trust`, `-beta*latency`, `-gamma*load` from the logged inputs and
requires bit-for-bit equality with the logged terms and score, and the selected node to be the rule's argmax. The survey ran it
on all 6873 decisions of each of A, C and D (20,619 decisions): 0 failures. The unit tests also run it on every decision of
smoke-scenario runs of A, B, B-oracle, C, D and E.

What decides the winner-versus-runner-up margin (D53 rule: a term is decisive if it is >= 0.6 of the margin), seed 100 draw 7:
```
variant   decisions  single eligible RSU  decisive: latency  load   trust   mixed   winner has LOWER trust than runner-up
A         6873       248                  1995               4276   -       354     - (trust unused, alpha = 0)
C         6873       248                  1196               3329   1027    1073    857  (12.5%)
D         6873       248                   759               2765   2150     951    1296 (18.9%)
```
Source of the S7 examples (selection rule fixed before reading them: the first decision at step >= 700 in each category).
```
##### Example: D - runner-up lost on trust  (decision 3993, Variant D)

Decision 3993: step 700 (t = 700 s), vehicle 31, medium task, deadline 0.37 s - Variant D
RSU-19 was selected over RSU-7 (score 1.387 vs 0.833, margin 0.554).
Why: RSU-7 lost on trust: RSU-19's trust was 0.93 against RSU-7's 0.66, worth +0.532 of the 0.554 score margin (96%). The rest of the margin came from load (+0.060). RSU-7 had lower link latency (12.7 ms against 16.5 ms for RSU-19), worth 0.038 to RSU-7, but that was outweighed.

Score = 2 * trust - 0.25 * latency - 0.6 * load   (weights as logged)
   RSU      trust  alpha*trust  latency  -beta*lat   load  -gamma*load     score
  *  19   0.926     +1.8526    0.662    -0.1654  0.500     -0.3000   +1.3873
      7   0.660     +1.3208    0.509    -0.1273  0.600     -0.3600   +0.8335
     10   0.325     +0.6507    0.608    -0.1521  0.700     -0.4200   +0.0786
  (* selected, x excluded by the dwell filter)

Other candidates:
  RSU-10 lost on trust: RSU-19's trust was 0.93 against RSU-10's 0.33, worth +1.202 of the 1.309 score margin (92%). The rest of the margin came from load (+0.120). RSU-10 had lower link latency (15.2 ms against 16.5 ms for RSU-19), worth 0.013 to RSU-10, but that was outweighed.

Logged behaviour record (context, not an attribution): RSU-19: promises kept 1.00, latency deviation 0.02; RSU-7: promises kept 0.63, latency deviation 0.32; RSU-10: promises kept 0.95, latency deviation 0.03.
Why a node's trust has its value is not explained here (Layer 2).
Outcome: deadline met (observed 0.161 s, advertised 0.176 s).

##### Example: D - runner-up lost on latency; the winner had LOWER trust  (decision 4002, Variant D)

Decision 4002: step 702 (t = 702 s), vehicle 26, light task, deadline 0.10 s - Variant D
RSU-2 was selected over RSU-14 (score 1.492 vs 1.392, margin 0.100).
Why: RSU-14 lost on latency: RSU-2's link latency was 5.6 ms against RSU-14's 16.5 ms, worth +0.109, more than the whole 0.100 score margin (the terms below offset part of it). RSU-14 had higher trust (0.99 against 0.98 for RSU-2), worth 0.009 to RSU-14, but that was outweighed.

Score = 2 * trust - 0.25 * latency - 0.6 * load   (weights as logged)
   RSU      trust  alpha*trust  latency  -beta*lat   load  -gamma*load     score
  *   2   0.984     +1.9677    0.223    -0.0558  0.700     -0.4200   +1.4920
     14   0.988     +1.9767    0.658    -0.1645  0.700     -0.4200   +1.3922
     19   0.927     +1.8541    0.666    -0.1666  0.500     -0.3000   +1.3875
     12   0.935     +1.8710    0.676    -0.1690  0.900     -0.5400   +1.1620
      0   0.537     +1.0750    0.669    -0.1673  0.600     -0.3600   +0.5477
  (* selected, x excluded by the dwell filter)

Other candidates:
  RSU-19 lost on trust: RSU-2's trust was 0.98 against RSU-19's 0.93, worth +0.114, more than the whole 0.104 score margin (the terms below offset part of it). The rest of the margin came from latency (+0.111). RSU-19 had lower advertised load (0.50 against 0.70 for RSU-2), worth 0.120 to RSU-19, but that was outweighed.
  No single term decided it: RSU-2 led by 0.330 overall, from load (+0.120, 36%) and latency (+0.113, 34%) (RSU-2's advertised load was 0.70 against RSU-12's 0.90; RSU-2's link latency was 5.6 ms against RSU-12's 16.9 ms).
  RSU-0 lost on trust: RSU-2's trust was 0.98 against RSU-0's 0.54, worth +0.893 of the 0.944 score margin (95%). The rest of the margin came from latency (+0.112). RSU-0 had lower advertised load (0.60 against 0.70 for RSU-2), worth 0.060 to RSU-0, but that was outweighed.

Logged behaviour record (context, not an attribution): RSU-2: promises kept 0.97, latency deviation 0.02; RSU-0: promises kept 0.48, latency deviation 0.21; RSU-12: promises kept 0.91, latency deviation 0.02; RSU-14: promises kept 1.00, latency deviation 0.08; RSU-19: promises kept 1.00, latency deviation 0.02.
Why a node's trust has its value is not explained here (Layer 2).
Outcome: deadline met (observed 0.053 s, advertised 0.046 s).

##### Example: C - winner chosen despite lower trust; load decisive  (decision 4030, Variant C)

Decision 4030: step 707 (t = 707 s), vehicle 20, light task, deadline 0.11 s - Variant C
RSU-0 was selected over RSU-12 (score 0.461 vs 0.253, margin 0.208).
Why: RSU-12 lost on advertised load: RSU-0's advertised load was 0.40 against RSU-12's 0.90, worth +0.300, more than the whole 0.208 score margin (the terms below offset part of it). The rest of the margin came from latency (+0.003). RSU-12 had higher trust (0.96 against 0.86 for RSU-0), worth 0.095 to RSU-12, but that was outweighed.

Score = 1 * trust - 0.25 * latency - 0.6 * load   (weights as logged)
   RSU      trust  alpha*trust  latency  -beta*lat   load  -gamma*load     score
  *   0   0.864     +0.8637    0.652    -0.1630  0.400     -0.2400   +0.4606
     12   0.959     +0.9588    0.665    -0.1662  0.900     -0.5400   +0.2526
      3   0.880     +0.8796    0.674    -0.1684  1.000     -0.6000   +0.1111
  (* selected, x excluded by the dwell filter)

Other candidates:
  RSU-3 lost on advertised load: RSU-0's advertised load was 0.40 against RSU-3's 1.00, worth +0.360, more than the whole 0.349 score margin (the terms below offset part of it). The rest of the margin came from latency (+0.005). RSU-3 had higher trust (0.88 against 0.86 for RSU-0), worth 0.016 to RSU-3, but that was outweighed.

Logged behaviour record (context, not an attribution): RSU-0: promises kept 0.85, latency deviation 0.21; RSU-3: promises kept 0.80, latency deviation 0.05; RSU-12: promises kept 0.98, latency deviation 0.02.
Why a node's trust has its value is not explained here (Layer 2).
Outcome: deadline met (observed 0.101 s, advertised 0.059 s).

##### Example: A - trust unused  (decision 3993, Variant A)

Decision 3993: step 700 (t = 700 s), vehicle 31, medium task, deadline 0.37 s - Variant A
RSU-19 was selected over RSU-7 (score -0.465 vs -0.487, margin 0.022).
Why: RSU-7 lost on advertised load: RSU-19's advertised load was 0.50 against RSU-7's 0.60, worth +0.060, more than the whole 0.022 score margin (the terms below offset part of it). RSU-7 had lower link latency (12.7 ms against 16.5 ms for RSU-19), worth 0.038 to RSU-7, but that was outweighed.
Trust did not influence this decision: Variant A scores on latency and load only (alpha = 0).

Score = - 0.25 * latency - 0.6 * load   (weights as logged; alpha = 0, trust unused)
   RSU      trust  alpha*trust  latency  -beta*lat   load  -gamma*load     score
  *  19     n/a     +0.0000    0.662    -0.1654  0.500     -0.3000   -0.4654
      7     n/a     +0.0000    0.509    -0.1273  0.600     -0.3600   -0.4873
     10     n/a     +0.0000    0.608    -0.1521  0.700     -0.4200   -0.5721
  (* selected, x excluded by the dwell filter)

Other candidates:
  RSU-10 lost on advertised load: RSU-19's advertised load was 0.50 against RSU-10's 0.70, worth +0.120, more than the whole 0.107 score margin (the terms below offset part of it). RSU-10 had lower link latency (15.2 ms against 16.5 ms for RSU-19), worth 0.013 to RSU-10, but that was outweighed.

Logged behaviour record (context, not an attribution): RSU-19: promises kept 1.00, latency deviation 0.01; RSU-7: promises kept 0.73, latency deviation 0.32; RSU-10: promises kept 0.97, latency deviation 0.04.
Why a node's trust has its value is not explained here (Layer 2).
Outcome: deadline met (observed 0.161 s, advertised 0.176 s).
```
Interpretation: Layer 1 reproduces the system's own scores exactly on every logged decision, for every variant, so what the
explanation says is what the rule computed. Trust is a term, not a rule: under D it is the decisive term in 31% of decisions
and under C in 15%, while load decides 40-62% and latency 11-29%, and the winner has lower trust than the runner-up in 13% (C)
and 19% (D) of decisions - an explanation that said "highest trust wins" would be wrong for roughly one decision in five.
Variant A's explanations never mention trust. A share above 100% in an example is a term worth more than the whole margin
because another term offset part of it; the text says so. Layer 1 explains the score, not why a node's trust has its value:
that, and the neighbourhood clause, are Layer 2's (S8).

### F24 — S8 raw results: H4 fidelity, GNNExplainer stability, neighbour-clause rate, H5 timing, merged example (RAW - no verdict recorded here)
Date: 2026-10-14 | Session: S8 | Commit: 47d5622 (protocol D54 committed in aa97280 before any number)
Config: `configs/demo.yaml`; heads C and D; evaluation seeds 1-5, rho 1.0, fraction 0.20, injection draw 100, closed loop with the D52
weights (one run per head per seed); sample = 40 decisions per seed (step >= 400, >= 2 eligible candidates), the winner and the
runner-up explained: 400 explanations per head | GNNExplainer (PyG 2.8.0), 100 epochs, run_seed 0 primary, 1 and 2 for stability
Commands: `python scripts/s8_fidelity.py collect`, `python scripts/s8_fidelity.py report` (raw only), `python scripts/s8_timing.py`,
`python scripts/s8_example.py` (results/s8/*; not committed)
Hardware for H5: Windows 10.0.26300 laptop, 13th Gen Intel Core i5-13450HX (10 cores, 16 logical), 24 GB RAM, CPU only (no GPU used),
torch 2.6.0, default 10 intra-op threads and a 1-thread rerun; nothing else running during the timing.

Rebuild check (both heads): the graph rebuilt from the closed-loop scenario reproduces every logged trust exactly (max difference
0.0) and the rule re-run on it reproduces every logged selection (0 mismatches in 200 decisions per head).

H4 fidelity (perturb one (node, feature) unit by 0.3; per-seed rates over 80 explanations; mean over 5 seeds, 95% t-interval):
```
HEAD D                         top rate         random-low rate   gap (top - low)   gap 95% CI          per-seed gaps
trust change  (>= 0.05)        0.205 (sd 0.095) 0.045 (sd 0.027)  +0.160 (sd 0.094) [+0.0431, +0.2769]  .1375 .100 .100 .1375 .325
decision change                0.065 (sd 0.041) 0.032 (sd 0.019)  +0.0325 (sd 0.026) [+0.0003, +0.0647] .050 .000 .0375 .0125 .0625
HEAD C                         top rate         random-low rate   gap               gap 95% CI          per-seed gaps
trust change  (>= 0.05)        0.580 (sd 0.112) 0.000 (sd 0.000)  +0.580 (sd 0.112) [+0.4405, +0.7195]  .5625 .6625 .4875 .4625 .725
decision change                0.163 (sd 0.029) 0.000 (sd 0.000)  +0.163 (sd 0.029) [+0.1261, +0.1989]  .1625 .1375 .150 .150 .2125
per-seed top / low trust-change rates, D: s1 .150/.013, s2 .138/.037, s3 .150/.050, s4 .225/.087, s5 .362/.037
per-seed top / low decision-change rates, D: s1 .100/.050, s2 .013/.013, s3 .050/.013, s4 .050/.037, s5 .113/.050
pooled over explanations, D: trust-change top 0.205 low 0.045; decision-change top 0.065 low 0.033; mean |trust delta| top 0.043, low 0.009
pooled, C: trust-change top 0.580 low 0.000; decision-change top 0.163 low 0.000; mean |trust delta| top 0.136, low 0.000
where the units are: D top unit is the RSU's own feature in 47.7% of explanations, the random-low unit in 3.0%;
                     C top unit is own in 100%, the random-low unit in 0.0% (the low pool, drawn from the graph's 2-hop
                     receptive field, holds only units on other nodes, which C's architecture cannot read)
```
GNNExplainer stability (run_seed 0, 1, 2 on the same 400 explanations per head):
```
                                                   D       C
top unit identical in all three runs               0.223   0.370
mean pairwise top-unit agreement                   0.303   0.496
top node identical in all three runs               0.297   1.000
mean pairwise Jaccard of the top-3 units           0.253   0.548
```
Neighbour clause (D54 section 7: same-segment RSU neighbours carry >= 30% of the attribution and a neighbour has >= 5%):
```
HEAD D: included in 10.2% of 400 explanations; mean same-segment share 0.189, mean share on the RSU's own units 0.070
        explained RSU degraded (n=80, sealed class used by the evaluation only): clause 16.3%, mean same-segment share 0.214
        explained RSU not degraded (n=320): clause 8.7%, mean same-segment share 0.182
        every explained RSU had at least one same-segment RSU neighbour (100%)
HEAD C: included in 0.0% (same-segment share 0.000, own share 1.000 by construction)
```
H5 timing, ms per decision (median / mean / 95th percentile), 200 decisions per head, Layer 2 = winner + runner-up
(per explanation in the last column):
```
head / torch threads  trust inference          selection              Layer 1                Layer 2 per decision          Layer 2 per explanation
D, 10 threads         0.771 / 0.846 / 1.280    0.041 / 0.044 / 0.055  0.091 / 0.095 / 0.134  598.412 / 603.443 / 666.889   295.134 / 301.718 / 337.988
D, 1 thread           0.286 / 0.326 / 0.620    0.041 / 0.044 / 0.055  0.091 / 0.095 / 0.138  307.423 / 308.836 / 331.501   153.594 / 154.415 / 167.846
C, 10 threads         0.089 / 0.100 / 0.185    0.042 / 0.043 / 0.055  0.089 / 0.096 / 0.141  231.389 / 231.574 / 255.720   115.510 / 115.784 / 131.246
C, 1 thread           0.074 / 0.086 / 0.160    0.038 / 0.041 / 0.051  0.085 / 0.086 / 0.134  141.362 / 143.861 / 159.514    70.695 / 71.928 / 82.870
```
Merged explanation (D54 section 9 rule; first of 457 qualifying decisions): decision 3993, seed 100, draw 7, Variant D. The runner-up RSU-7
is degraded, on backhaul segment 3 with degraded RSUs 0, 5, 7, 16. Layer 2 numbers: RSU-19 trust 0.926, share on its own units 0.063,
same-segment RSU neighbours [2, 14], same-segment share 0.129, clause False; RSU-7 trust 0.660, own share 0.089, same-segment neighbours
[5, 10, 16], same-segment share 0.205, clause False. The full text is in the next entry.

### F25 — S8 verdicts against D54, interpretation, and the merged example (written after F24 was committed in 3487836)
Date: 2026-10-14 | Session: S8 | Commit: 3487836 (numbers)
Command: `python scripts/s8_fidelity.py report --verdict`

Verdicts, as D54 section 5-6 define them:
```
Head D  trust change   gap +0.160, 95% CI [+0.043, +0.277], floor 0.20   -> WEAK       (CI above 0, mean below the floor)
Head D  decision change gap +0.033, 95% CI [+0.0003, +0.065], floor 0.05 -> WEAK       (CI barely above 0, mean below the floor)
Head C  trust change   gap +0.580, 95% CI [+0.441, +0.720], floor 0.20   -> FAITHFUL   (but see the low-pool artefact below)
Head C  decision change gap +0.163, 95% CI [+0.126, +0.199], floor 0.05  -> FAITHFUL   (same caveat)
GNNExplainer masks (all-three top-unit agreement, threshold 0.80): D 0.223, C 0.370 -> UNSTABLE for both heads.
```
Post-hoc, exploratory (not part of D54, computed from the F24 rows, labelled as such): for head D, when the top unit is the RSU's own
feature (191 explanations) perturbing it changes trust in 34.6% of cases (mean |change| 0.070); when the top unit sits on a neighbour (209
explanations) it changes trust in 7.7% (mean |change| 0.018), against 3.9% for the random-low units (388 of 400 are on neighbours; the 12 own
ones change trust in 25%).

Interpretation: (1) The attribution is only weakly faithful for the GNN head. Perturbing D's top-attributed unit changes the trust of the
explained RSU in 20.5% of cases against 4.5% for a random low unit, a gap of 0.16 that is clearly above zero but below the 0.20 floor fixed in
advance; for the decision itself the gap is 0.033 (top 6.5%, low 3.2%) and its interval touches zero, so a decision rarely flips because of any
single input. (2) Most of D's measured fidelity comes from its own features: top-own units move trust in 35% of cases, top-neighbour units in 8%,
barely above the 4% of random neighbour units. The part of the explanation that only a graph model can give - which neighbours drive the score -
is the least supported part. (3) C's gap is large but not comparable: its low pool, taken from the graph's receptive field as D54 specified,
consists entirely of units on other nodes that C cannot read, so its random-low rate is 0.000 by construction; the honest reading is that C's
own-feature attribution is sound and that the C-versus-D comparison on H4 is not informative. D54 is not changed. (4) The masks are unstable:
GNNExplainer's top unit agrees across three seeds for only 22% (D) and 37% (C) of explanations, and the top node for 30% (D), so the same
decision can be explained by a different leading feature on another run; explanations should carry their run seed and be read as one draw.
(5) The neighbour clause is rarely supported: 10.2% of D explanations (16.3% when the explained RSU is degraded, 8.7% otherwise) meet the 30%
same-segment-share rule, although every explained RSU has at least one same-segment neighbour. TECHNICAL_AND_NOVELTY.md 3.4 presents the
neighbour clause as evidence for the graph claim; on these numbers the claim is thinly supported, consistent with S4 (D is not better than C).
(6) H5: Layer 2 costs 307 ms per decision (two GNNExplainer runs, 1 thread, D) or 598 ms at 10 threads, against 0.3-0.8 ms for trust
inference, 0.04 ms for selection and 0.09 ms for Layer 1 - three orders of magnitude more than the decision itself and 3-6 times the stated
100 ms budget; Layer 2 is asynchronous only (explain after the decision, off the dispatch path). It is faster on one thread than on ten for
graphs this small. C's explanation costs 141-231 ms per decision.

Merged explanation, the first qualifying decision under D54 section 9 (not the best-looking one). The neighbour clause is NOT produced because
the attribution does not support it (20% < 30%) - the text says so instead of telling a neighbour story:
```
##### Merged explanation - first qualifying decision (D54 section 9): decision 3993, seed 100, draw 7, Variant D; 457 qualifying decisions at step >= 700
(runner-up RSU-7 is degraded, on backhaul segment 3 with degraded RSUs [0, 5, 7, 16]; sealed class used only to choose the decision)

Decision 3993: step 700 (t = 700 s), vehicle 31, medium task, deadline 0.37 s - Variant D
RSU-19 was selected over RSU-7 (score 1.387 vs 0.833, margin 0.554).
Why: RSU-7 lost on trust: RSU-19's trust was 0.93 against RSU-7's 0.66, worth +0.532 of the 0.554 score margin (96%). The rest of the margin came from load (+0.060). RSU-7 had lower link latency (12.7 ms against 16.5 ms for RSU-19), worth 0.038 to RSU-7, but that was outweighed.

Score = 2 * trust - 0.25 * latency - 0.6 * load   (weights as logged)
   RSU      trust  alpha*trust  latency  -beta*lat   load  -gamma*load     score
  *  19   0.926     +1.8526    0.662    -0.1654  0.500     -0.3000   +1.3873
      7   0.660     +1.3208    0.509    -0.1273  0.600     -0.3600   +0.8335
     10   0.325     +0.6507    0.608    -0.1521  0.700     -0.4200   +0.0786
  (* selected, x excluded by the dwell filter)

Other candidates:
  RSU-10 lost on trust: RSU-19's trust was 0.93 against RSU-10's 0.33, worth +1.202 of the 1.309 score margin (92%). The rest of the margin came from load (+0.120). RSU-10 had lower link latency (15.2 ms against 16.5 ms for RSU-19), worth 0.013 to RSU-10, but that was outweighed.

Logged behaviour record (context, not an attribution): RSU-19: promises kept 1.00, latency deviation 0.02; RSU-7: promises kept 0.63, latency deviation 0.32; RSU-10: promises kept 0.95, latency deviation 0.03.
Outcome: deadline met (observed 0.161 s, advertised 0.176 s).

RSU-19 selected over RSU-7. Primary factors: RSU-19 trust score 0.93, the attribution rests on its own advertised load 0.50 (lowers trust by 0.01 compared with a typical value), neighbouring RSUs did not drive this score (same-segment neighbours carry 13% of the attribution, below the 30% needed to name them); RSU-7 trust score 0.66, the attribution rests on its own latency deviation 0.32 (lowers trust by 0.29 compared with a typical value), neighbouring RSUs did not drive this score (same-segment neighbours carry 20% of the attribution, below the 30% needed to name them).
(Layer 2: GNNExplainer on the trust head. Own-evidence and neighbour effects are measured by replacing the evidence with the step's median RSU value; a neighbour is named only when same-segment neighbours carry at least 30% of the attribution.)
[Layer 2 numbers, RSU-19: trust 0.926, own share 0.063, same-segment RSU neighbours [2, 14], same-segment share 0.129, clause False]
[Layer 2 numbers, RSU-7: trust 0.660, own share 0.089, same-segment RSU neighbours [5, 10, 16], same-segment share 0.205, clause False]
```

### F26 — S9 raw results: the L12 sweeps, closed loop, 5 variants x 5 seeds x 50 draws (RAW - no labels, no verdicts recorded here)
Date: 2026-10-14 | Session: S9 | Commit: 9f2a8f9 (harness; D56 = 26e6e79, D57 = 9f2a8f9; code and rules unchanged since the grid was launched)
Config: `configs/sweep.yaml` over `configs/demo.yaml`, weights `configs/weights.yaml` (D52, unchanged), frozen seed 1-5 checkpoints; draws 100-149
(seed 100 and the demo draw excluded); closed loop, dwell filter and handoff on; 12 workers, one torch thread each.
Command: `python scripts/s9_sweep.py all --sweep configs/sweep.yaml` (log results/s9/sweep.log; tables results/s9/results_table.csv,
paired_comparisons.csv, report-raw.txt - gitignored). Grid: 2000 of 2000 tasks completed, 0 failed, 10,000 closed-loop runs, about 35 min wall
(the 12 h threshold was not approached, so no draws were dropped). Harness history: a smoke test on non-evaluation draws 900-901 exposed a
zero oracle reference in D56's H1 formula and led to D57 before any evaluation draw ran; its numbers are not reported.
Checks: B's task stream equals A's on every one of 2000 paired draws; the post-onset dispatched count is equal across variants on every draw;
the shared cell (rho 1, fraction 0.2, draw 100) recomputed serially is bit-identical for all five seeds; E's task stream equals D's for all five.
Reading key: "dispatches to degraded" n_av is counted from onset to the run's own effective-avoidance step (D56); mean +/- sd is across the five
seed means. Latency is censored at 399 and A, B cannot detect by construction. Realised pair concentration is nominal-rho dependent (0.22 at rho 0
to 0.96 at rho 1; F10 shapes).

Numbers:
```
D42 accounting: draws attempted / failed / aggregated per point (every variant shares the same paired draws)
  The sweep injects no cold-start RSU (cold_start.num_nodes = 0), so no cold-start aggregate exists; the counts below are the whole-run accounting.
  point                                        attempted  failed aggregated   K degraded (mean)  realised pair concentration (mean over draws with K>=2)
  rho=0.00 frac=0.20 [rho]                           250       0        250    4.00              0.216
  rho=0.25 frac=0.20 [rho]                           250       0        250    4.00              0.333
  rho=0.50 frac=0.20 [rho]                           250       0        250    4.00              0.500
  rho=0.75 frac=0.20 [rho]                           250       0        250    4.00              0.717
  rho=1.00 frac=0.20 [rho+fraction]                  250       0        250    4.00              0.956
  rho=1.00 frac=0.05 [fraction]                      250       0        250    1.00              nan
  rho=1.00 frac=0.10 [fraction]                      250       0        250    2.00              1.000
  rho=1.00 frac=0.30 [fraction]                      250       0        250    6.00              0.761
  consistency: runs where B's task stream differs from A's: 0; runs where the post-onset dispatched count differs across variants: 0

----------------------------------------------------------------------------------------------------------------------
rho sweep (fraction fixed)

  rho=0.00 frac=0.20 [rho]
    metric                                                           A                   B            B_oracle                   C                   D
    H3 success post-onset                            0.8709 +/- 0.0035   0.8709 +/- 0.0035   0.9516 +/- 0.0034   0.9484 +/- 0.0041   0.9325 +/- 0.0053
    success (all steps)                              0.9060 +/- 0.0039   0.9060 +/- 0.0039   0.9597 +/- 0.0038   0.9574 +/- 0.0044   0.9459 +/- 0.0032
    H1 dispatches to degraded, onset->avoidance         619.1 +/- 23.4      619.1 +/- 23.4         0.2 +/- 0.6        91.8 +/- 9.1      137.8 +/- 25.6
    dispatches to degraded, [400,1200)                  909.8 +/- 15.6      909.8 +/- 15.6      125.8 +/- 17.3      299.4 +/- 18.7      412.1 +/- 69.2
    fraction of runs never avoiding                  0.4960 +/- 0.0167   0.4960 +/- 0.0167   0.0000 +/- 0.0000   0.0000 +/- 0.0000   0.0000 +/- 0.0000
    H1 detection latency (steps, censored 399)           399.0 +/- 0.0       399.0 +/- 0.0         0.0 +/- 0.0      163.0 +/- 29.3      250.0 +/- 21.8
    fraction of degraded RSUs detected               0.0000 +/- 0.0000   0.0000 +/- 0.0000   1.0000 +/- 0.0000   1.0000 +/- 0.0000   0.9470 +/- 0.0821

  rho=0.25 frac=0.20 [rho]
    metric                                                           A                   B            B_oracle                   C                   D
    H3 success post-onset                            0.8710 +/- 0.0026   0.8710 +/- 0.0026   0.9478 +/- 0.0034   0.9448 +/- 0.0042   0.9289 +/- 0.0060
    success (all steps)                              0.9060 +/- 0.0035   0.9060 +/- 0.0035   0.9572 +/- 0.0042   0.9550 +/- 0.0047   0.9435 +/- 0.0043
    H1 dispatches to degraded, onset->avoidance         616.3 +/- 26.3      616.3 +/- 26.3         2.3 +/- 3.1      100.0 +/- 13.9      145.7 +/- 27.9
    dispatches to degraded, [400,1200)                   909.7 +/- 8.2       909.7 +/- 8.2      161.6 +/- 16.9      327.2 +/- 17.5      438.2 +/- 72.5
    fraction of runs never avoiding                  0.4760 +/- 0.0410   0.4760 +/- 0.0410   0.0000 +/- 0.0000   0.0000 +/- 0.0000   0.0000 +/- 0.0000
    H1 detection latency (steps, censored 399)           399.0 +/- 0.0       399.0 +/- 0.0         0.0 +/- 0.0      161.6 +/- 29.7      244.4 +/- 21.2
    fraction of degraded RSUs detected               0.0000 +/- 0.0000   0.0000 +/- 0.0000   1.0000 +/- 0.0000   1.0000 +/- 0.0000   0.9520 +/- 0.0794

  rho=0.50 frac=0.20 [rho]
    metric                                                           A                   B            B_oracle                   C                   D
    H3 success post-onset                            0.8708 +/- 0.0018   0.8708 +/- 0.0018   0.9398 +/- 0.0046   0.9370 +/- 0.0051   0.9215 +/- 0.0056
    success (all steps)                              0.9059 +/- 0.0029   0.9059 +/- 0.0029   0.9519 +/- 0.0053   0.9498 +/- 0.0056   0.9386 +/- 0.0047
    H1 dispatches to degraded, onset->avoidance         607.0 +/- 32.2      607.0 +/- 32.2         8.8 +/- 8.4      114.4 +/- 20.3      161.9 +/- 33.5
    dispatches to degraded, [400,1200)                   912.0 +/- 9.8       912.0 +/- 9.8      237.6 +/- 31.0      386.1 +/- 31.9      490.7 +/- 68.2
    fraction of runs never avoiding                  0.4520 +/- 0.0756   0.4520 +/- 0.0756   0.0040 +/- 0.0089   0.0040 +/- 0.0089   0.0040 +/- 0.0089
    H1 detection latency (steps, censored 399)           399.0 +/- 0.0       399.0 +/- 0.0         0.0 +/- 0.0      157.8 +/- 26.7      236.6 +/- 21.9
    fraction of degraded RSUs detected               0.0000 +/- 0.0000   0.0000 +/- 0.0000   1.0000 +/- 0.0000   1.0000 +/- 0.0000   0.9520 +/- 0.0726

  rho=0.75 frac=0.20 [rho]
    metric                                                           A                   B            B_oracle                   C                   D
    H3 success post-onset                            0.8718 +/- 0.0021   0.8718 +/- 0.0021   0.9332 +/- 0.0069   0.9306 +/- 0.0073   0.9139 +/- 0.0061
    success (all steps)                              0.9066 +/- 0.0033   0.9066 +/- 0.0033   0.9475 +/- 0.0068   0.9455 +/- 0.0070   0.9335 +/- 0.0056
    H1 dispatches to degraded, onset->avoidance         602.2 +/- 69.1      602.2 +/- 69.1       19.2 +/- 24.5      132.9 +/- 31.9      183.4 +/- 47.0
    dispatches to degraded, [400,1200)                  907.6 +/- 13.5      907.6 +/- 13.5      301.3 +/- 53.0      434.3 +/- 46.2      545.4 +/- 67.0
    fraction of runs never avoiding                  0.4400 +/- 0.0762   0.4400 +/- 0.0762   0.0120 +/- 0.0268   0.0200 +/- 0.0283   0.0200 +/- 0.0283
    H1 detection latency (steps, censored 399)           399.0 +/- 0.0       399.0 +/- 0.0         0.0 +/- 0.0      154.4 +/- 22.2      229.9 +/- 24.7
    fraction of degraded RSUs detected               0.0000 +/- 0.0000   0.0000 +/- 0.0000   1.0000 +/- 0.0000   1.0000 +/- 0.0000   0.9520 +/- 0.0780

  rho=1.00 frac=0.20 [rho+fraction]
    metric                                                           A                   B            B_oracle                   C                   D
    H3 success post-onset                            0.8713 +/- 0.0026   0.8713 +/- 0.0026   0.9234 +/- 0.0099   0.9211 +/- 0.0100   0.9049 +/- 0.0083
    success (all steps)                              0.9062 +/- 0.0036   0.9062 +/- 0.0036   0.9409 +/- 0.0087   0.9392 +/- 0.0088   0.9276 +/- 0.0074
    H1 dispatches to degraded, onset->avoidance        627.5 +/- 100.4     627.5 +/- 100.4       24.4 +/- 25.6      152.7 +/- 47.9      210.6 +/- 84.5
    dispatches to degraded, [400,1200)                  912.4 +/- 20.6      912.4 +/- 20.6      394.7 +/- 89.9      506.3 +/- 74.6      610.1 +/- 84.2
    fraction of runs never avoiding                  0.4680 +/- 0.1188   0.4680 +/- 0.1188   0.0120 +/- 0.0268   0.0280 +/- 0.0390   0.0280 +/- 0.0390
    H1 detection latency (steps, censored 399)           399.0 +/- 0.0       399.0 +/- 0.0         0.0 +/- 0.0      150.7 +/- 21.8      220.2 +/- 25.1
    fraction of degraded RSUs detected               0.0000 +/- 0.0000   0.0000 +/- 0.0000   1.0000 +/- 0.0000   1.0000 +/- 0.0000   0.9590 +/- 0.0701

----------------------------------------------------------------------------------------------------------------------
fraction sweep (rho fixed)

  rho=1.00 frac=0.20 [rho+fraction]
    metric                                                           A                   B            B_oracle                   C                   D
    H3 success post-onset                            0.8713 +/- 0.0026   0.8713 +/- 0.0026   0.9234 +/- 0.0099   0.9211 +/- 0.0100   0.9049 +/- 0.0083
    success (all steps)                              0.9062 +/- 0.0036   0.9062 +/- 0.0036   0.9409 +/- 0.0087   0.9392 +/- 0.0088   0.9276 +/- 0.0074
    H1 dispatches to degraded, onset->avoidance        627.5 +/- 100.4     627.5 +/- 100.4       24.4 +/- 25.6      152.7 +/- 47.9      210.6 +/- 84.5
    dispatches to degraded, [400,1200)                  912.4 +/- 20.6      912.4 +/- 20.6      394.7 +/- 89.9      506.3 +/- 74.6      610.1 +/- 84.2
    fraction of runs never avoiding                  0.4680 +/- 0.1188   0.4680 +/- 0.1188   0.0120 +/- 0.0268   0.0280 +/- 0.0390   0.0280 +/- 0.0390
    H1 detection latency (steps, censored 399)           399.0 +/- 0.0       399.0 +/- 0.0         0.0 +/- 0.0      150.7 +/- 21.8      220.2 +/- 25.1
    fraction of degraded RSUs detected               0.0000 +/- 0.0000   0.0000 +/- 0.0000   1.0000 +/- 0.0000   1.0000 +/- 0.0000   0.9590 +/- 0.0701

  rho=1.00 frac=0.05 [fraction]   [K = 1: no correlated failure possible, F10]
    metric                                                           A                   B            B_oracle                   C                   D
    H3 success post-onset                            0.9474 +/- 0.0028   0.9474 +/- 0.0028   0.9697 +/- 0.0025   0.9685 +/- 0.0029   0.9637 +/- 0.0027
    success (all steps)                              0.9570 +/- 0.0035   0.9570 +/- 0.0035   0.9718 +/- 0.0035   0.9708 +/- 0.0037   0.9667 +/- 0.0040
    H1 dispatches to degraded, onset->avoidance           71.3 +/- 8.3        71.3 +/- 8.3         0.0 +/- 0.0        22.3 +/- 3.0        31.0 +/- 4.1
    dispatches to degraded, [400,1200)                   226.9 +/- 6.2       226.9 +/- 6.2        11.5 +/- 2.9        58.7 +/- 4.0       82.6 +/- 13.8
    fraction of runs never avoiding                  0.0560 +/- 0.0555   0.0560 +/- 0.0555   0.0000 +/- 0.0000   0.0000 +/- 0.0000   0.0000 +/- 0.0000
    H1 detection latency (steps, censored 399)           399.0 +/- 0.0       399.0 +/- 0.0         0.0 +/- 0.0      161.3 +/- 26.6      264.1 +/- 20.8
    fraction of degraded RSUs detected               0.0000 +/- 0.0000   0.0000 +/- 0.0000   1.0000 +/- 0.0000   1.0000 +/- 0.0000   0.9160 +/- 0.1152

  rho=1.00 frac=0.10 [fraction]
    metric                                                           A                   B            B_oracle                   C                   D
    H3 success post-onset                            0.9223 +/- 0.0024   0.9223 +/- 0.0024   0.9619 +/- 0.0023   0.9601 +/- 0.0027   0.9513 +/- 0.0027
    success (all steps)                              0.9402 +/- 0.0033   0.9402 +/- 0.0033   0.9666 +/- 0.0037   0.9652 +/- 0.0040   0.9585 +/- 0.0034
    H1 dispatches to degraded, onset->avoidance         235.9 +/- 33.7      235.9 +/- 33.7         0.8 +/- 0.7        48.6 +/- 8.0       70.9 +/- 12.8
    dispatches to degraded, [400,1200)                   451.7 +/- 9.4       451.7 +/- 9.4       66.1 +/- 16.2      150.7 +/- 17.2      204.4 +/- 37.7
    fraction of runs never avoiding                  0.2640 +/- 0.0590   0.2640 +/- 0.0590   0.0000 +/- 0.0000   0.0000 +/- 0.0000   0.0000 +/- 0.0000
    H1 detection latency (steps, censored 399)           399.0 +/- 0.0       399.0 +/- 0.0         0.0 +/- 0.0      159.2 +/- 26.1      247.9 +/- 26.2
    fraction of degraded RSUs detected               0.0000 +/- 0.0000   0.0000 +/- 0.0000   1.0000 +/- 0.0000   1.0000 +/- 0.0000   0.9280 +/- 0.1008

  rho=1.00 frac=0.30 [fraction]
    metric                                                           A                   B            B_oracle                   C                   D
    H3 success post-onset                            0.8212 +/- 0.0023   0.8212 +/- 0.0023   0.8777 +/- 0.0128   0.8750 +/- 0.0129   0.8604 +/- 0.0085
    success (all steps)                              0.8729 +/- 0.0033   0.8729 +/- 0.0033   0.9105 +/- 0.0107   0.9084 +/- 0.0108   0.8979 +/- 0.0073
    H1 dispatches to degraded, onset->avoidance       1065.0 +/- 195.5    1065.0 +/- 195.5       62.4 +/- 51.1      250.0 +/- 70.6     344.3 +/- 125.1
    dispatches to degraded, [400,1200)                 1358.9 +/- 27.9     1358.9 +/- 27.9     796.7 +/- 134.6     915.0 +/- 107.4     1001.0 +/- 92.8
    fraction of runs never avoiding                  0.6560 +/- 0.1670   0.6560 +/- 0.1670   0.0000 +/- 0.0000   0.0000 +/- 0.0000   0.0080 +/- 0.0179
    H1 detection latency (steps, censored 399)           399.0 +/- 0.0       399.0 +/- 0.0         0.0 +/- 0.0      146.8 +/- 19.3      210.7 +/- 23.4
    fraction of degraded RSUs detected               0.0000 +/- 0.0000   0.0000 +/- 0.0000   1.0000 +/- 0.0000   1.0000 +/- 0.0000   0.9633 +/- 0.0522

Paired effects (X - ref; seed means; 95% t interval over 5 seeds). H3 = absolute success rate. H1 = (n_ref - n_X)/n_A (D57), n_av primary, [n_win guard]. Latency diff in steps.
point(rho/frac)  X vs ref      H3 diff [95% CI]                  H1 n_av reduction [95% CI]       n_win guard   lat diff
1.00/0.05   C vs A        +0.0211 [+0.0203,+0.0219]  +0.681 [+0.591,+0.771]  +0.741 [+0.720,+0.762]  -238
1.00/0.05   D vs A        +0.0162 [+0.0146,+0.0179]  +0.558 [+0.448,+0.669]  +0.635 [+0.551,+0.719]  -135
1.00/0.05   C vs B_oracle -0.0012 [-0.0019,-0.0006]  -0.319 [-0.409,-0.229]  -0.208 [-0.219,-0.198]  +161
1.00/0.05   D vs B_oracle -0.0061 [-0.0073,-0.0048]  -0.442 [-0.552,-0.331]  -0.315 [-0.401,-0.228]  +264
1.00/0.05   D vs C        -0.0049 [-0.0060,-0.0037]  -0.123 [-0.200,-0.045]  -0.106 [-0.188,-0.024]  +103

1.00/0.10   C vs A        +0.0379 [+0.0353,+0.0404]  +0.791 [+0.738,+0.844]  +0.667 [+0.622,+0.711]  -240
1.00/0.10   D vs A        +0.0291 [+0.0240,+0.0341]  +0.697 [+0.636,+0.759]  +0.547 [+0.440,+0.653]  -151
1.00/0.10   C vs B_oracle -0.0018 [-0.0026,-0.0010]  -0.205 [-0.258,-0.153]  -0.187 [-0.197,-0.177]  +159
1.00/0.10   D vs B_oracle -0.0106 [-0.0147,-0.0066]  -0.299 [-0.360,-0.239]  -0.307 [-0.392,-0.222]  +248
1.00/0.10   D vs C        -0.0088 [-0.0131,-0.0045]  -0.094 [-0.120,-0.068]  -0.120 [-0.211,-0.028]  +89

1.00/0.20   C vs A        +0.0498 [+0.0392,+0.0605]  +0.756 [+0.676,+0.836]  +0.446 [+0.351,+0.541]  -248
1.00/0.20   D vs A        +0.0337 [+0.0221,+0.0452]  +0.664 [+0.523,+0.806]  +0.331 [+0.221,+0.442]  -179
1.00/0.20   C vs B_oracle -0.0023 [-0.0028,-0.0017]  -0.205 [-0.287,-0.123]  -0.123 [-0.147,-0.099]  +151
1.00/0.20   D vs B_oracle -0.0184 [-0.0287,-0.0082]  -0.296 [-0.447,-0.146]  -0.237 [-0.344,-0.130]  +220
1.00/0.20   D vs C        -0.0162 [-0.0267,-0.0056]  -0.091 [-0.179,-0.004]  -0.114 [-0.209,-0.019]  +69

1.00/0.30   C vs A        +0.0538 [+0.0382,+0.0694]  +0.758 [+0.663,+0.853]  +0.327 [+0.234,+0.420]  -252
1.00/0.30   D vs A        +0.0392 [+0.0267,+0.0517]  +0.669 [+0.517,+0.820]  +0.263 [+0.181,+0.346]  -188
1.00/0.30   C vs B_oracle -0.0028 [-0.0034,-0.0021]  -0.183 [-0.246,-0.120]  -0.087 [-0.114,-0.061]  +147
1.00/0.30   D vs B_oracle -0.0173 [-0.0302,-0.0045]  -0.273 [-0.391,-0.154]  -0.151 [-0.248,-0.054]  +211
1.00/0.30   D vs C        -0.0146 [-0.0279,-0.0012]  -0.090 [-0.166,-0.013]  -0.064 [-0.141,+0.013]  +64

0.00/0.20   C vs A        +0.0775 [+0.0742,+0.0808]  +0.852 [+0.839,+0.865]  +0.671 [+0.648,+0.694]  -236
0.00/0.20   D vs A        +0.0615 [+0.0508,+0.0723]  +0.777 [+0.723,+0.831]  +0.546 [+0.445,+0.647]  -149
0.00/0.20   C vs B_oracle -0.0032 [-0.0043,-0.0020]  -0.148 [-0.161,-0.135]  -0.191 [-0.199,-0.182]  +163
0.00/0.20   D vs B_oracle -0.0191 [-0.0289,-0.0093]  -0.222 [-0.276,-0.169]  -0.316 [-0.422,-0.210]  +250
0.00/0.20   D vs C        -0.0160 [-0.0266,-0.0054]  -0.075 [-0.122,-0.028]  -0.125 [-0.236,-0.014]  +87

0.25/0.20   C vs A        +0.0738 [+0.0703,+0.0773]  +0.838 [+0.815,+0.861]  +0.640 [+0.617,+0.664]  -237
0.25/0.20   D vs A        +0.0579 [+0.0477,+0.0681]  +0.764 [+0.713,+0.815]  +0.518 [+0.417,+0.619]  -155
0.25/0.20   C vs B_oracle -0.0030 [-0.0040,-0.0020]  -0.158 [-0.180,-0.136]  -0.182 [-0.189,-0.175]  +162
0.25/0.20   D vs B_oracle -0.0189 [-0.0284,-0.0094]  -0.232 [-0.281,-0.183]  -0.304 [-0.403,-0.206]  +244
0.25/0.20   D vs C        -0.0159 [-0.0261,-0.0057]  -0.074 [-0.118,-0.030]  -0.122 [-0.227,-0.018]  +83

0.50/0.20   C vs A        +0.0662 [+0.0607,+0.0717]  +0.811 [+0.767,+0.855]  +0.577 [+0.535,+0.619]  -241
0.50/0.20   D vs A        +0.0507 [+0.0417,+0.0597]  +0.734 [+0.673,+0.795]  +0.462 [+0.369,+0.555]  -162
0.50/0.20   C vs B_oracle -0.0028 [-0.0037,-0.0020]  -0.174 [-0.206,-0.143]  -0.163 [-0.170,-0.156]  +158
0.50/0.20   D vs B_oracle -0.0183 [-0.0277,-0.0090]  -0.252 [-0.308,-0.195]  -0.278 [-0.372,-0.184]  +237
0.50/0.20   D vs C        -0.0155 [-0.0253,-0.0057]  -0.077 [-0.128,-0.027]  -0.115 [-0.211,-0.019]  +79

0.75/0.20   C vs A        +0.0588 [+0.0515,+0.0661]  +0.779 [+0.723,+0.835]  +0.522 [+0.461,+0.582]  -245
0.75/0.20   D vs A        +0.0421 [+0.0329,+0.0512]  +0.696 [+0.616,+0.776]  +0.399 [+0.307,+0.491]  -169
0.75/0.20   C vs B_oracle -0.0026 [-0.0032,-0.0019]  -0.190 [-0.242,-0.138]  -0.147 [-0.158,-0.136]  +154
0.75/0.20   D vs B_oracle -0.0193 [-0.0295,-0.0091]  -0.273 [-0.356,-0.190]  -0.269 [-0.374,-0.165]  +230
0.75/0.20   D vs C        -0.0167 [-0.0273,-0.0062]  -0.083 [-0.131,-0.036]  -0.123 [-0.221,-0.024]  +75

H3 'non-overlapping error bars' (PROJECT_SPEC 4.1), fraction sweep: separate 95% t intervals of the variants' seed means
  rho=1.00 frac=0.20 [rho+fraction]            C vs A: non-overlap; C vs B_oracle: overlap; D vs A: non-overlap; D vs B_oracle: overlap
  rho=1.00 frac=0.05 [fraction]                C vs A: non-overlap; C vs B_oracle: overlap; D vs A: non-overlap; D vs B_oracle: overlap
  rho=1.00 frac=0.10 [fraction]                C vs A: non-overlap; C vs B_oracle: overlap; D vs A: non-overlap; D vs B_oracle: non-overlap
  rho=1.00 frac=0.30 [fraction]                C vs A: non-overlap; C vs B_oracle: overlap; D vs A: non-overlap; D vs B_oracle: overlap
shared-point agreement (D56 1b): config-level assert passed at load; shared cell (rho 1.0, frac 0.2, draw 100) recomputed bit-identical for all 5 seeds: True; E's task stream equal to D's for all 5 seeds: True
======================================================================================================================
H4 FIDELITY REPLICATION and H5 TIMING (D56 section 7): Variant E (head D), shared point, draw 101
  hardware: {'platform': 'Windows-10-10.0.26300-SP0', 'logical_cpus': 16, 'torch': '2.6.0+cu124', 'default_torch_threads': 10, 'device': 'cpu', 'cpu': '13th Gen Intel(R) Core(TM) i5-13450HX | 10 cores', 'ram_gb': '24'}
  decisions per seed {'1': 40, '2': 40, '3': 40, '4': 40, '5': 40}; graph rebuild max |trust diff| 0.00e+00; decisions whose re-run differs from the logged selection: 0 of 200
  H4 trust-change: per-seed (top, low, gap):
      seed 1: top 0.125  low 0.013  gap +0.113
      seed 2: top 0.150  low 0.037  gap +0.112
      seed 3: top 0.113  low 0.050  gap +0.062
      seed 4: top 0.212  low 0.037  gap +0.175
      seed 5: top 0.362  low 0.037  gap +0.325
    mean top 0.192 (sd 0.103)  low 0.035 (sd 0.014)  gap +0.158 (sd 0.102) 95% [+0.031, +0.284]  floor 0.2
  H4 decision-change: per-seed (top, low, gap):
      seed 1: top 0.075  low 0.013  gap +0.062
      seed 2: top 0.025  low 0.000  gap +0.025
      seed 3: top 0.050  low 0.025  gap +0.025
      seed 4: top 0.100  low 0.050  gap +0.050
      seed 5: top 0.100  low 0.013  gap +0.088
    mean top 0.070 (sd 0.033)  low 0.020 (sd 0.019)  gap +0.050 (sd 0.027) 95% [+0.017, +0.083]  floor 0.05
  H5 per-decision times in ms, threads1 (200 decisions); median / mean / p95:
      trust_inference              0.269 /     0.326 /     0.631
      selection                    0.039 /     0.044 /     0.059
      layer1                       0.089 /     0.091 /     0.130
      layer2_per_decision        310.236 /   310.883 /   356.479
      layer2_per_explanation     152.895 /   155.439 /   191.024
  H5 per-decision times in ms, threads10 (200 decisions); median / mean / p95:
      trust_inference              0.939 /     0.958 /     1.317
      selection                    0.049 /     0.051 /     0.061
      layer1                       0.107 /     0.112 /     0.153
      layer2_per_decision        674.923 /   672.157 /   734.458
      layer2_per_explanation     334.565 /   336.075 /   372.970
```

### F27 — S9 verdicts against D56/D57, and interpretation (written after F26 was committed in 0b6585e)
Date: 2026-10-14 | Session: S9 | Commit: see `git log` | No rule, floor, bound, window or number of F26 was changed.
Labels from `python scripts/s9_analyse.py --sweep configs/sweep.yaml --verdict` (results/s9/report-verdict.txt), applied exactly as D56 section 4
and D57:
```
H3 (fraction sweep, rho 1.0; label at fractions 0.05 / 0.10 / 0.20 / 0.30; H3 holds only with WIN at >= 2 of 4)
  C vs A        WIN WIN WIN WIN           -> 4 of 4  -> HOLDS      (non-overlapping error bars at all 4: yes)
  D vs A        WIN WIN WIN WIN           -> 4 of 4  -> HOLDS      (non-overlapping error bars at all 4: yes)
  C vs B_oracle NULL NULL NULL NULL       -> 0 of 4  (within the bound of the oracle at every point; oracle better by 0.001-0.003)
  D vs B_oracle NULL NEGATIVE NEGATIVE NEGATIVE -> 0 of 4  (the oracle is at least the 0.01 floor better at 0.10, 0.20, 0.30)
H1 (n_av, shared point rho 1.0 / fraction 0.20)
  C vs A  WIN  +0.756 [+0.676, +0.836]      D vs A  WIN  +0.664 [+0.523, +0.806]
  C vs B_oracle NEGATIVE -0.205 [-0.287, -0.123]      D vs B_oracle NEGATIVE -0.296 [-0.447, -0.146]
  (the n_win guard is positive for the vs-A rows: C +0.446 [+0.351, +0.541], D +0.331 [+0.221, +0.442])
Secondary, closed-loop H2 check, D vs C (H3 metric), fraction 0.05 / 0.10 / 0.20 / 0.30: NULL NULL NEGATIVE NEGATIVE (never gating; F17 unchanged)
Every other point (rho 0 ... 0.75) has the same labels as the shared point for vs-A (WIN) and vs-oracle (C NULL on H3, D NEGATIVE; both NEGATIVE on H1).
B vs A: identical on all 2000 paired draws (sanity row exactly zero).
H4 replication (draw 101, D54 procedure): trust-change gap +0.158 [+0.031, +0.284], floor 0.20 -> WEAK (F25: +0.160, WEAK).
  decision-change gap +0.0500 [+0.017, +0.083], floor 0.05 -> FAITHFUL by the rule's >=, at the boundary (F25: +0.033, WEAK).
H5: Layer 2 median 310 ms/decision at 1 thread, 675 ms at 10 threads vs the 100 ms assumption -> asynchronous only; trust inference
  0.27 ms (0.94 at 10 threads), selection 0.04 ms, Layer 1 0.09 ms.
```
Interpretation (what the numbers say, in the order of the hypotheses):
1. **H3 holds against no-trust dispatch.** Both learned heads raise post-onset deadline success over A at all four degraded fractions, with
   non-overlapping seed intervals: C by +0.021 (5%), +0.038, +0.050, +0.054 (30%); D by +0.016, +0.029, +0.034, +0.039. The gain is smallest at 5%
   (little to avoid) and largest where more nodes fail, but at rho 1.0, fraction 0.2 it is smaller than at rho 0 (C +0.050 vs +0.078): the more the
   failures share a segment, the fewer healthy alternatives a vehicle has.
2. **B is A here**, as D50 states; the "improvement over certificate trust" is an improvement over no behavioural signal, and the oracle shows what
   a perfect certificate-style signal could do. C recovers all but 0.001-0.003 of the oracle's success at every fraction (NULL: the floor of 0.01 is not
   reached); D falls 0.006-0.018 short of it and is NEGATIVE against it at three fractions.
3. **H1 holds against A with a large effect, and is far from the oracle.** C sends 76% fewer tasks (of A's count) to degraded RSUs than A before its
   own avoidance point, D 66%; the oracle is a further 21% (C) and 30% (D) of A's count better because it avoids at once. Caution: A reaches the
   avoidance rule in only 53% of runs (47% "never avoided", so its n_av is a full-window count), so the n_av reduction partly measures where the rule
   stops counting; the fixed-window n_win reduction, which does not depend on the rule, is smaller (C +0.45, D +0.33 at the shared point) and is the
   more conservative statement. Detection latency (descriptive, not gating): C 151 steps, D 220 steps after onset at the shared point against 0 for
   the oracle and "never" for A and B; D leaves 4% of degraded RSUs undetected inside the 400-step window.
4. **The closed loop agrees with S4 about the graph.** D is not better than C on any point and is worse on H3 at two of four fractions (NEGATIVE) and
   on latency by 64-103 steps; the D-minus-C gap does not shrink or grow with rho (-0.016 at rho 0, -0.016 at rho 1.0). The structural claim H2 gets no
   support from the closed loop either. Possible contributors, none tested: D's weights were tuned by the same procedure as C's (F20 showed D weaker
   on the tuning seed too) and the training/evaluation dispatch shift of F21 item 3 and F22 (D's healthy-RSU trust drifts down). These are hypotheses.
5. **H4 replicates as WEAK on the trust-change criterion**, with essentially the F25 gap (+0.158 vs +0.160). The decision-change criterion lands exactly
   on its 0.05 floor and so reads FAITHFUL by the literal rule, but F25 measured +0.033 (WEAK) on draw 100 with the same procedure; their mean is
   0.042, and a single boundary pass is not a claim. **H5 replicates F24:** Layer 2 is asynchronous only, three orders of
   magnitude above the decision itself.
6. **D42:** no cold-start RSU was injected, so no cold-start aggregate exists; every point has 250 draws attempted, 0 failed, 250 aggregated.
Limits that apply to all of the above: simulation-only, one topology per seed, the injection model is the most attackable assumption (PROJECT_SPEC
4.6), the trust heads were trained on A-dispatch (F21), the avoidance bar 0.75 is a stated assumption (D56), the vs-oracle rows are headroom and not a
competing method, and the H4/H5 stage uses the D54 sample size (200 decisions, 400 explanations), not larger.

### F28 — S9b Gate-C tuning grid on the tuning seed (RAW, recorded before the chosen tau is used anywhere)
Date: 2026-10-14 | Session: S9b | Commit: see `git log` (code in the commit before this entry; D59 = 3710795)
Config: `configs/demo.yaml`, tuning seed 100 (excluded from evaluation), rho 1.0, fraction 0.20, injection draws 0, 1, 2, closed loop, dwell filter and handoff on,
seed-100 C checkpoint (F20), C's D52 weights (alpha 1, beta 0.25, gamma 0.6) | Protocol: D59 section 2, unchanged
Command: `python scripts/s9b_gate.py tune` (results/s9b/tuning.json). 24 runs. Objective: deadline success of dispatched tasks over steps [400, 1200), mean over the 3 draws.
```
tau    objective   per-draw (0, 1, 2)            empty-set (fallback) rate   gate-removed-at-least-one rate
0.2    0.8999   [0.9406, 0.8519, 0.9073]   0.0158   0.0642
0.3    0.8998   [0.9406, 0.8516, 0.9071]   0.0254   0.0873
0.4    0.8998   [0.9406, 0.8514, 0.9073]   0.0405   0.1155
0.5    0.8997   [0.9404, 0.8516, 0.9071]   0.0526   0.1336
0.6    0.9001   [0.9408, 0.8521, 0.9073]   0.0600   0.1432
0.7    0.9000   [0.9410, 0.8514, 0.9075]   0.0626   0.1492
0.8    0.9001   [0.9410, 0.8516, 0.9075]   0.0646   0.1582
0.9    0.9004   [0.9415, 0.8521, 0.9078]   0.1031   0.4025
reference (F20): Variant C, same seed, draws, weights, no gate: 0.8999, per-draw [0.9406, 0.8516, 0.9073]
```
Chosen by the D59 rule (highest objective, ties to the lowest tau): **tau = 0.9**, written to `configs/gate.yaml`.
Interpretation: the whole grid lies within 0.0007 of C's un-gated 0.8999 (draw-to-draw spread on this seed is 0.09), so on the tuning seed a gate on C's own head changes the success rate by less than
anything this seed can resolve; the choice of tau 0.9 rests on a 0.0005 margin and should not be read as a finding about thresholds. The fallback rate grows with tau (1.6% at 0.2 to 10.3% at 0.9)
and at tau 0.9 the gate removes a candidate on 40% of decisions, so the gate is doing work there while the outcome stays flat.

### F29 — S9b raw results: Gate-C (tau 0.9) vs weighted C vs A, the four fraction-sweep points, seeds 1-5 x draws 100-149 (RAW - no labels, no verdict recorded here)
Date: 2026-10-14 | Session: S9b | Commit: see `git log` (D59 = 3710795; tau from F28)
Config: `configs/gate.yaml` over `configs/sweep.yaml` and `configs/demo.yaml`; C's D52 weights; frozen seed 1-5 C checkpoints; closed loop, dwell filter and handoff on; tau 0.9 for every seed and point.
Command: `python scripts/s9b_gate.py run` then `python scripts/s9b_gate.py report` (results/s9b/report-raw.txt). 1000 tasks, 0 failed, about 7.5 min on 12 workers.
Check: Variant C was recomputed in every one of the 1000 tasks and equals the stored S9 values bit-for-bit (wall time excepted): 0 mismatches at every point. Gate-C is paired per draw with the stored S9 C and A.
Reading key: n_win = dispatches to degraded RSUs over [400,1200); the "n_win reduction in units of A" row is (Gate-C minus C) / A, so a negative value means Gate-C sent MORE tasks to degraded RSUs than C.
```
================================================================================================================
S9b GATE-C (RAW: no labels, no verdict)   tau = 0.9   seeds [1, 2, 3, 4, 5]  draws 100-149  weights: C's (D52)
mean +/- sd across the 5 seed means; closed loop; Gate-C paired per draw with the stored S9 Variant C; A from S9
================================================================================================================

D42-style accounting (draws attempted / failed / aggregated) and the C-recompute check
  rho 1.00 frac 0.05: attempted 250  failed 0  aggregated 250   Variant C recompute differs from stored S9 on 0 draws
  rho 1.00 frac 0.10: attempted 250  failed 0  aggregated 250   Variant C recompute differs from stored S9 on 0 draws
  rho 1.00 frac 0.20: attempted 250  failed 0  aggregated 250   Variant C recompute differs from stored S9 on 0 draws
  rho 1.00 frac 0.30: attempted 250  failed 0  aggregated 250   Variant C recompute differs from stored S9 on 0 draws

----------------------------------------------------------------------------------------------------------------
rho 1.00, fraction 0.05   [K = 1: no correlated failure]
  metric                                                           A          C (weighted)                Gate-C
  H3 success post-onset                            0.9474 +/- 0.0028     0.9685 +/- 0.0029     0.9682 +/- 0.0033
  dispatches to degraded, onset->avoidance              71.3 +/- 8.3          22.3 +/- 3.0          20.6 +/- 1.9
  n_win: dispatches to degraded, [400,1200)            226.9 +/- 6.2          58.7 +/- 4.0          54.0 +/- 3.4
  empty-candidate-set (fallback) rate, all steps : 0.04010 +/- 0.01399   steps >= 400 : 0.04774 +/- 0.01654
  decisions where the gate removed >= 1 candidate: 0.3118 +/- 0.0448  (includes the fallback decisions)
  PAIRED  C - Gate-C  success: +0.0003 (sd 0.0005) 95% [-0.0003, +0.0009]
          Gate-C - A  success: +0.0208 (sd 0.0011) 95% [+0.0194, +0.0222]  (descriptive)
          (Gate-C - C) n_win reduction in units of A: -0.021 [-0.031, -0.010]   n_av: -0.026 [-0.058, +0.007]   (positive = C sends fewer; descriptive)

----------------------------------------------------------------------------------------------------------------
rho 1.00, fraction 0.10
  metric                                                           A          C (weighted)                Gate-C
  H3 success post-onset                            0.9223 +/- 0.0024     0.9601 +/- 0.0027     0.9601 +/- 0.0030
  dispatches to degraded, onset->avoidance            235.9 +/- 33.7          48.6 +/- 8.0          46.3 +/- 7.3
  n_win: dispatches to degraded, [400,1200)            451.7 +/- 9.4        150.7 +/- 17.2        141.6 +/- 17.8
  empty-candidate-set (fallback) rate, all steps : 0.04941 +/- 0.01494   steps >= 400 : 0.06172 +/- 0.01802
  decisions where the gate removed >= 1 candidate: 0.3484 +/- 0.0401  (includes the fallback decisions)
  PAIRED  C - Gate-C  success: +0.0001 (sd 0.0004) 95% [-0.0004, +0.0005]
          Gate-C - A  success: +0.0378 (sd 0.0023) 95% [+0.0349, +0.0407]  (descriptive)
          (Gate-C - C) n_win reduction in units of A: -0.020 [-0.026, -0.014]   n_av: -0.009 [-0.022, +0.003]   (positive = C sends fewer; descriptive)

----------------------------------------------------------------------------------------------------------------
rho 1.00, fraction 0.20   [shared operating point]
  metric                                                           A          C (weighted)                Gate-C
  H3 success post-onset                            0.8713 +/- 0.0026     0.9211 +/- 0.0100     0.9213 +/- 0.0102
  dispatches to degraded, onset->avoidance           627.5 +/- 100.4        152.7 +/- 47.9        145.2 +/- 45.7
  n_win: dispatches to degraded, [400,1200)           912.4 +/- 20.6        506.3 +/- 74.6        493.5 +/- 76.8
  empty-candidate-set (fallback) rate, all steps : 0.08900 +/- 0.02182   steps >= 400 : 0.12114 +/- 0.02723
  decisions where the gate removed >= 1 candidate: 0.3964 +/- 0.0321  (includes the fallback decisions)
  PAIRED  C - Gate-C  success: -0.0002 (sd 0.0004) 95% [-0.0006, +0.0003]
          Gate-C - A  success: +0.0500 (sd 0.0088) 95% [+0.0391, +0.0610]  (descriptive)
          (Gate-C - C) n_win reduction in units of A: -0.014 [-0.019, -0.010]   n_av: -0.012 [-0.023, -0.002]   (positive = C sends fewer; descriptive)

----------------------------------------------------------------------------------------------------------------
rho 1.00, fraction 0.30
  metric                                                           A          C (weighted)                Gate-C
  H3 success post-onset                            0.8212 +/- 0.0023     0.8750 +/- 0.0129     0.8752 +/- 0.0131
  dispatches to degraded, onset->avoidance          1065.0 +/- 195.5        250.0 +/- 70.6        231.2 +/- 67.8
  n_win: dispatches to degraded, [400,1200)          1358.9 +/- 27.9       915.0 +/- 107.4       901.5 +/- 111.5
  empty-candidate-set (fallback) rate, all steps : 0.13581 +/- 0.02334   steps >= 400 : 0.19138 +/- 0.02970
  decisions where the gate removed >= 1 candidate: 0.4425 +/- 0.0336  (includes the fallback decisions)
  PAIRED  C - Gate-C  success: -0.0003 (sd 0.0003) 95% [-0.0006, +0.0001]
          Gate-C - A  success: +0.0541 (sd 0.0128) 95% [+0.0382, +0.0699]  (descriptive)
          (Gate-C - C) n_win reduction in units of A: -0.010 [-0.014, -0.006]   n_av: -0.019 [-0.043, +0.004]   (positive = C sends fewer; descriptive)
```

### F30 — S9b verdict against D59 and interpretation (written after F29 was committed)
Date: 2026-10-14 | Session: S9b | Commit: see `git log` | No tau, floor, bound, point, seed, draw or rule of D59 was changed.
Correction to F29's reading key (F29 itself is not edited): the row "(Gate-C - C) n_win reduction in units of A" is (Gate-C count minus C count) / A's count with the sign of
`relative_reduction(C, Gate-C, A)`, so a NEGATIVE value means Gate-C sent FEWER tasks to degraded RSUs than C, not more. F29's printed numbers are unchanged and correct; only its one-line gloss was wrong.
Labels from `python scripts/s9b_gate.py report --verdict` (results/s9b/report-verdict.txt), D59 section 4 (effect = C minus Gate-C success; floor 0.01, bound 0.025):
```
fraction   C - Gate-C success difference [95% interval over seeds]     label
0.05       +0.0003 [-0.0003, +0.0009]                                  NULL
0.10       +0.0001 [-0.0004, +0.0005]                                  NULL
0.20       -0.0002 [-0.0006, +0.0003]    (shared operating point)      NULL
0.30       -0.0003 [-0.0006, +0.0001]                                  NULL
WIN at 0 of 4, NEGATIVE at 0 of 4  ->  D59 verdict: NOT SEPARATED (all four labels NULL)
```
Interpretation:
1. **The weighted rule is not shown to beat the gate on task success, and the gate is not shown to beat the weighted rule.** At every point the paired difference is below 0.001 in absolute value and every interval
   lies inside +/-0.001, an order of magnitude below the 0.01 floor. NULL here rules out a weighted advantage of 0.025 or more (the D59 bound); because the lower ends of the intervals are above -0.0007 it also rules out a
   gate advantage of 0.01 or more. Against no trust, Gate-C gains +0.021 to +0.054 over A, the same as C (+0.021 to +0.054). So the project's design statement that trust enters as a weighted term and not as a gate
   (PROJECT_SPEC 5.5) is **not supported by a success-rate advantage in this scenario**; it stands as a design argument (a moderately trusted, much closer node can still win; an explanation of the score is exact), not as a measured gain.
2. **Gate-C sends slightly fewer tasks to degraded RSUs than C**, consistently: n_win 493.5 vs 506.3 at the shared point, 54.0 vs 58.7, 141.6 vs 150.7, 901.5 vs 915.0 at the other fractions, i.e. 1.0-2.1% of A's count, with seed
   intervals excluding 0 on n_win at all four points (n_av excludes 0 only at the shared point). That reduction does not translate into a success difference above 0.001. Descriptive only (D59: not gating).
3. **The gate is not rare.** With tau 0.9 it removed at least one candidate on 31-44% of decisions and the empty-candidate-set (fallback) rate was 4.0% (5%), 4.9% (10%), 8.9% (20%), 13.6% (30%) over all steps, and
   4.8%, 6.2%, 12.1% and 19.1% over steps >= 400. On those decisions Gate-C is C by construction. So the null is not an artefact of the gate never acting: it acts often and the outcome still matches C.
4. Limits. tau was chosen on one seed from a flat grid (F28: all eight values within 0.0007, chosen on a 0.0005 margin); D59 forbids re-tuning on evaluation seeds, so a better tau, a gate on D's head, or a gate with different weights is not excluded
   and would be a new, labelled post-hoc entry. A hypothesis, untested: C's weighted rule with alpha 1 already steers away from low-trust candidates, so a threshold adds little. The result is for this scenario and injection model only.

### F31 — S9b addendum to F29: the never-avoided rate that F29 omitted (RAW, from the same stored runs; no new run, no new number elsewhere)
Date: 2026-10-14 | Session: S9b (record completion) | Commit: see `git log`
F29 listed success, n_av, n_win and the empty-candidate-set rate but not the fraction of runs that never reached the D56 effective-avoidance rule, which the S9b record is required to carry (D60 Q4 also asks that
censoring be quoted beside n_av). F29 and F30 are not edited. The rows below are computed from the same 1000 stored Gate-C result files and the stored S9 A and C files by `python scripts/s9b_gate.py report`
(the report now prints this row; no other part of the analysis changed, and every F29/F30 number reproduces). Mean +/- sd across the 5 seed means.
```
fraction   fraction of runs never avoiding:    A                C                Gate-C (tau 0.9)
0.05                                           0.056 +/- 0.055  0.000 +/- 0.000  0.000 +/- 0.000
0.10                                           0.264 +/- 0.059  0.000 +/- 0.000  0.000 +/- 0.000
0.20 (shared)                                  0.468 +/- 0.119  0.028 +/- 0.039  0.028 +/- 0.039
0.30                                           0.656 +/- 0.167  0.000 +/- 0.000  0.000 +/- 0.000
```
Consistency: the A and C columns equal F26's (0.056 / 0.264 / 0.468 / 0.656 and 0.000 / 0.000 / 0.028 / 0.000), as they must since they are read from the same S9 files.
Interpretation: Gate-C's censoring is identical to C's at every point, so the n_av comparison between them is not distorted by different censoring; the large censoring is A's (up to 66% at 30% degraded), as F27 item 3 says.

### F32 � S13 Experiment 1: tuning grid of the heuristic baseline H on the tuning seed (RAW, recorded before the chosen weights are used anywhere)
Date: 2026-10-15 | Session: S13 | Commit: 9b5c139 (code); pre-registration D65 = 245705b (2026-10-15 09:16:12 +0530)
Config: `configs/demo.yaml`, `configs/h.yaml`, tuning seed 100 (excluded from all evaluation), rho 1.0, fraction 0.20, injection draws 0, 1, 2, closed loop, dwell filter and handoff on | Protocol: D65 section 1.2 = D52 procedure, unchanged
Variant H: trust_v = the RSU's own `success_ewma` (no model, no gate); weights alpha/beta/gamma tuned over alpha {0.25,0.5,1,2,4} x beta {0.25,0.5,1} x gamma {0.15,0.3,0.6}
Command: `python scripts/s13_h.py tune --write` (results/s13/h/tuning.json; configs/weights_h.yaml). 135 runs, 1m32s on 12 workers. First result of Experiment 1 written 2026-10-15T03:50:31Z (09:20:31 +0530).
Objective: deadline success of dispatched tasks over steps [400, 1200), mean over the 3 draws. Ties go to the first grid point in ascending (alpha, beta, gamma) order.
```
H  (rows: alpha, beta, gamma; value: mean post-onset deadline success over draws 0,1,2; per-draw in brackets)
   alpha beta gamma | objective  per-draw
   0.25  0.25 0.15  | 0.8922    [0.9304, 0.8458, 0.9006]
   0.25  0.25 0.3   | 0.8929    [0.9282, 0.8495, 0.901]
   0.25  0.25 0.6   | 0.8907    [0.9219, 0.8495, 0.9006]
   0.25  0.5  0.15  | 0.8825    [0.9149, 0.8412, 0.8912]
   0.25  0.5  0.3   | 0.8859    [0.9165, 0.846, 0.8954]
   0.25  0.5  0.6   | 0.8862    [0.9145, 0.8479, 0.896]
   0.25  1.0  0.15  | 0.8692    [0.8919, 0.8355, 0.8801]
   0.25  1.0  0.3   | 0.8725    [0.8949, 0.8388, 0.8838]
   0.25  1.0  0.6   | 0.8754    [0.8978, 0.8419, 0.8864]
   0.5   0.25 0.15  | 0.8933    [0.9339, 0.8462, 0.8999]
   0.5   0.25 0.3   | 0.8965    [0.936, 0.849, 0.9043]  <== chosen
   0.5   0.25 0.6   | 0.8957    [0.9321, 0.8499, 0.9052]
   0.5   0.5  0.15  | 0.8886    [0.9271, 0.8421, 0.8967]
   0.5   0.5  0.3   | 0.8922    [0.9304, 0.8458, 0.9006]
   0.5   0.5  0.6   | 0.8929    [0.9282, 0.8495, 0.901]
   0.5   1.0  0.15  | 0.8796    [0.9112, 0.8382, 0.8895]
   0.5   1.0  0.3   | 0.8825    [0.9149, 0.8412, 0.8912]
   0.5   1.0  0.6   | 0.8859    [0.9165, 0.846, 0.8954]
   1.0   0.25 0.15  | 0.8895    [0.9317, 0.8416, 0.8951]
   1.0   0.25 0.3   | 0.8926    [0.9345, 0.8432, 0.9002]
   1.0   0.25 0.6   | 0.8957    [0.936, 0.8469, 0.9041]
   1.0   0.5  0.15  | 0.8891    [0.9304, 0.8412, 0.8956]
   1.0   0.5  0.3   | 0.8933    [0.9339, 0.8462, 0.8999]
   1.0   0.5  0.6   | 0.8965    [0.936, 0.849, 0.9043]
   1.0   1.0  0.15  | 0.8856    [0.9243, 0.8386, 0.8938]
   1.0   1.0  0.3   | 0.8886    [0.9271, 0.8421, 0.8967]
   1.0   1.0  0.6   | 0.8922    [0.9304, 0.8458, 0.9006]
   2.0   0.25 0.15  | 0.8870    [0.9295, 0.8379, 0.8934]
   2.0   0.25 0.3   | 0.8891    [0.9317, 0.8403, 0.8954]
   2.0   0.25 0.6   | 0.8913    [0.9347, 0.8403, 0.8988]
   2.0   0.5  0.15  | 0.8869    [0.9282, 0.8397, 0.8928]
   2.0   0.5  0.3   | 0.8895    [0.9317, 0.8416, 0.8951]
   2.0   0.5  0.6   | 0.8926    [0.9345, 0.8432, 0.9002]
   2.0   1.0  0.15  | 0.8860    [0.928, 0.8375, 0.8925]
   2.0   1.0  0.3   | 0.8891    [0.9304, 0.8412, 0.8956]
   2.0   1.0  0.6   | 0.8933    [0.9339, 0.8462, 0.8999]
   4.0   0.25 0.15  | 0.8834    [0.9284, 0.8316, 0.8901]
   4.0   0.25 0.3   | 0.8848    [0.9278, 0.8342, 0.8923]
   4.0   0.25 0.6   | 0.8864    [0.9289, 0.8375, 0.893]
   4.0   0.5  0.15  | 0.8803    [0.9234, 0.8292, 0.8882]
   4.0   0.5  0.3   | 0.8870    [0.9295, 0.8379, 0.8934]
   4.0   0.5  0.6   | 0.8891    [0.9317, 0.8403, 0.8954]
   4.0   1.0  0.15  | 0.8832    [0.9243, 0.8342, 0.891]
   4.0   1.0  0.3   | 0.8869    [0.9282, 0.8397, 0.8928]
   4.0   1.0  0.6   | 0.8895    [0.9317, 0.8416, 0.8951]
chosen (alpha,beta,gamma) = (0.5, 0.25, 0.3), objective 0.8965, per-draw [0.936, 0.849, 0.9043]
```
Spread over the grid: best 0.8965, worst 0.8692. Exact ties at the best value: (0.5,0.25,0.3), (1.0,0.5,0.6) (first in grid order is taken).
For reference only (F20, same seed, draws, objective): A 0.8670, C 0.8999, D 0.8938, B_oracle 0.9009.
Interpretation: recorded before use; the chosen weights are written to configs/weights_h.yaml and applied unchanged in the evaluation. One seed, three draws, per-draw values differ by up to 0.09, so differences of ~0.005 between grid points are inside the draw-to-draw spread (as F20).

### F33 — S13 Experiment 1 raw results: heuristic H vs learned C, the four fraction-sweep points, seeds 1-5 x draws 100-149 (RAW - no labels, no verdict recorded here)
Date: 2026-10-15 | Session: S13 | Commit: 9b5c139 (code), 776beff (F32, weights_h.yaml); D65 = 245705b
Config: `configs/h.yaml` over `configs/sweep.yaml`, `configs/demo.yaml`; H weights `configs/weights_h.yaml` (F32: 0.5, 0.25, 0.3); C weights D52 (1, 0.25, 0.6); frozen seed 1-5 C checkpoints (not used by H); closed loop, dwell filter and handoff on.
Command: `python scripts/s13_h.py run` then `python scripts/s13_h.py report` (results/s13/h/report-raw.txt). 1000 tasks, 0 failed, 112 min on 12 workers (the machine was shared; the F29 run of the same size took 7.5 min). First evaluation result written 2026-10-15T03:52:23Z (09:22:23 +0530), after F32 (09:21:52 +0530).
Check: Variant C was recomputed in every task and equals the stored S9 values bit-for-bit (wall time excepted): 0 mismatches at every point. H is paired per draw with stored S9 C and A.
Reading key: n_win = dispatches to degraded RSUs over [400,1200); (n_H - n_C)/n_A positive = C sends fewer; latency negative = C detects earlier; mean +/- sd across the 5 seed means; intervals Student-t over 5 seeds.
```
D42-style accounting (draws attempted / failed / aggregated) and the C-recompute check
  rho 1.00 frac 0.05: attempted 250  failed 0  aggregated 250   Variant C recompute differs from stored S9 on 0 draws
  rho 1.00 frac 0.10: attempted 250  failed 0  aggregated 250   Variant C recompute differs from stored S9 on 0 draws
  rho 1.00 frac 0.20: attempted 250  failed 0  aggregated 250   Variant C recompute differs from stored S9 on 0 draws
  rho 1.00 frac 0.30: attempted 250  failed 0  aggregated 250   Variant C recompute differs from stored S9 on 0 draws

----------------------------------------------------------------------------------------------------------------------
rho 1.00, fraction 0.05   [K = 1: no correlated failure]
  metric                                                             A           C (learned)         H (heuristic)
  H3 success post-onset                              0.9474 +/- 0.0028     0.9685 +/- 0.0029     0.9651 +/- 0.0026
  n_av: dispatches to degraded, onset->avoidance          71.3 +/- 8.3          22.3 +/- 3.0          21.0 +/- 1.4
  n_win: dispatches to degraded, [400,1200)              226.9 +/- 6.2          58.7 +/- 4.0          60.3 +/- 2.8
  never-avoided rate (D56 rule)                        0.056 +/- 0.055       0.000 +/- 0.000       0.000 +/- 0.000
  detection latency (steps, censored 399)                          n/a        161.3 +/- 26.6        171.1 +/- 31.8
  detected fraction of degraded RSUs                               n/a       1.000 +/- 0.000       1.000 +/- 0.000
  PRIMARY  C - H  success: +0.0034 (sd 0.0006) 95% [+0.0027, +0.0041]
  (descr.) H - A success: +0.0177 [+0.0169, +0.0185]   C - A success: +0.0211 [+0.0203, +0.0219]
  (descr.) detection latency C - H: -9.8 steps [-34.4, +14.8]   (negative = C detects earlier)
  (descr.) (n_H - n_C)/n_A, positive = C sends fewer: n_win +0.007 [-0.008, +0.022]   n_av -0.021 [-0.065, +0.024]

----------------------------------------------------------------------------------------------------------------------
rho 1.00, fraction 0.10
  metric                                                             A           C (learned)         H (heuristic)
  H3 success post-onset                              0.9223 +/- 0.0024     0.9601 +/- 0.0027     0.9566 +/- 0.0024
  n_av: dispatches to degraded, onset->avoidance        235.9 +/- 33.7          48.6 +/- 8.0          48.2 +/- 8.1
  n_win: dispatches to degraded, [400,1200)              451.7 +/- 9.4        150.7 +/- 17.2        153.4 +/- 17.2
  never-avoided rate (D56 rule)                        0.264 +/- 0.059       0.000 +/- 0.000       0.000 +/- 0.000
  detection latency (steps, censored 399)                          n/a        159.2 +/- 26.1        164.3 +/- 30.5
  detected fraction of degraded RSUs                               n/a       1.000 +/- 0.000       1.000 +/- 0.000
  PRIMARY  C - H  success: +0.0035 (sd 0.0006) 95% [+0.0028, +0.0042]
  (descr.) H - A success: +0.0344 [+0.0319, +0.0368]   C - A success: +0.0379 [+0.0353, +0.0404]
  (descr.) detection latency C - H: -5.1 steps [-24.9, +14.8]   (negative = C detects earlier)
  (descr.) (n_H - n_C)/n_A, positive = C sends fewer: n_win +0.006 [-0.009, +0.021]   n_av -0.002 [-0.018, +0.015]

----------------------------------------------------------------------------------------------------------------------
rho 1.00, fraction 0.20   [shared operating point]
  metric                                                             A           C (learned)         H (heuristic)
  H3 success post-onset                              0.8713 +/- 0.0026     0.9211 +/- 0.0100     0.9173 +/- 0.0098
  n_av: dispatches to degraded, onset->avoidance       627.5 +/- 100.4        152.7 +/- 47.9        155.1 +/- 48.7
  n_win: dispatches to degraded, [400,1200)             912.4 +/- 20.6        506.3 +/- 74.6        510.7 +/- 73.5
  never-avoided rate (D56 rule)                        0.468 +/- 0.119       0.028 +/- 0.039       0.028 +/- 0.039
  detection latency (steps, censored 399)                          n/a        150.7 +/- 21.8        157.1 +/- 28.0
  detected fraction of degraded RSUs                               n/a       1.000 +/- 0.000       1.000 +/- 0.000
  PRIMARY  C - H  success: +0.0038 (sd 0.0005) 95% [+0.0032, +0.0045]
  (descr.) H - A success: +0.0460 [+0.0355, +0.0565]   C - A success: +0.0498 [+0.0392, +0.0605]
  (descr.) detection latency C - H: -6.5 steps [-23.9, +11.0]   (negative = C detects earlier)
  (descr.) (n_H - n_C)/n_A, positive = C sends fewer: n_win +0.005 [-0.002, +0.012]   n_av +0.003 [-0.015, +0.021]

----------------------------------------------------------------------------------------------------------------------
rho 1.00, fraction 0.30
  metric                                                             A           C (learned)         H (heuristic)
  H3 success post-onset                              0.8212 +/- 0.0023     0.8750 +/- 0.0129     0.8705 +/- 0.0130
  n_av: dispatches to degraded, onset->avoidance      1065.0 +/- 195.5        250.0 +/- 70.6        272.7 +/- 87.6
  n_win: dispatches to degraded, [400,1200)            1358.9 +/- 27.9       915.0 +/- 107.4       918.5 +/- 106.6
  never-avoided rate (D56 rule)                        0.656 +/- 0.167       0.000 +/- 0.000       0.008 +/- 0.018
  detection latency (steps, censored 399)                          n/a        146.8 +/- 19.3        152.0 +/- 27.4
  detected fraction of degraded RSUs                               n/a       1.000 +/- 0.000       1.000 +/- 0.000
  PRIMARY  C - H  success: +0.0045 (sd 0.0009) 95% [+0.0033, +0.0057]
  (descr.) H - A success: +0.0493 [+0.0336, +0.0650]   C - A success: +0.0538 [+0.0382, +0.0694]
  (descr.) detection latency C - H: -5.1 steps [-23.9, +13.7]   (negative = C detects earlier)
  (descr.) (n_H - n_C)/n_A, positive = C sends fewer: n_win +0.003 [-0.003, +0.008]   n_av +0.018 [-0.020, +0.056]
```

### F34 — S13 Experiment 1 verdict against D65 and interpretation (written after F33 was committed in 9926c5d)
Date: 2026-10-15 | Session: S13 | Commit: see `git log` | No weight, floor, bound, point, seed, draw or rule of D65 was changed.
Labels from `python scripts/s13_h.py report --verdict` (results/s13/h/report-verdict.txt), D65 section 1.4 (effect = C minus H post-onset deadline success; floor 0.01; NULL = whole interval inside +-0.025):
```
fraction   C - H success [95% interval over seeds]      label
0.05       +0.0034 [+0.0027, +0.0041]                   NULL
0.10       +0.0035 [+0.0028, +0.0042]                   NULL
0.20       +0.0038 [+0.0032, +0.0045]   (shared point)  NULL
0.30       +0.0045 [+0.0033, +0.0057]                   NULL
WIN at 0 of 4, NEGATIVE at 0 of 4  ->  D65 overall verdict: NOT SEPARATED (all four labels NULL)
```
Rule applied: per point, NEGATIVE needs hi < 0 and mean <= -0.01 (no); WIN needs lo > 0 and mean >= 0.01 (lo > 0 holds at all four, but the mean is 0.003-0.005, below the floor); NULL needs lo > -0.025 and hi < 0.025 (yes). Overall: LEARNING HELPS needs WIN at >= 2 (0), HEURISTIC BETTER needs NEGATIVE at >= 2 (0), so NOT SEPARATED.
Interpretation:
1. Under the pre-registered rule learning is not shown to help: the learned head's advantage over the node's own success_ewma is positive and its interval excludes 0 at every point, but it is 0.003-0.005 in absolute success, below the 0.01 floor, and a difference of 0.025 or more is ruled out in either direction. The label is a statement about the floor, not about the sign; the sign is stated here as measured.
2. Descriptively, H recovers most of C's gain over no trust: H - A is +0.0177 / +0.0344 / +0.0460 / +0.0493 against C - A +0.0211 / +0.0379 / +0.0498 / +0.0538 (5% to 30%).
3. n_av, n_win and the never-avoided rate (together, D60 Q4): at the shared point C 152.7 / 506.3 / 0.028, H 155.1 / 510.7 / 0.028, A 627.5 / 912.4 / 0.468; the C-vs-H difference in units of A's count is +0.005 [-0.002, +0.012] (n_win) and +0.003 [-0.015, +0.021] (n_av), intervals including 0 at all four points. At 30% H never avoids in 0.8% of runs against C's 0.0%.
4. Detection latency (descriptive): C 5-10 steps earlier than H at every point, intervals [-34, +15] to [-24, +14] all including 0; all degraded RSUs are detected by both.
5. D42: every point 250 attempted, 0 failed, 250 aggregated; the C recompute matched S9 bit-for-bit on all 1000 tasks.
Limits: H's weights were tuned on one seed like every other variant (F32; the surface is flat, spread within draw-to-draw noise); the result is for this scenario and injection model only; it does not revise F17, F27 or any H2 verdict; no mechanism for why the head adds so little is tested.

### F35 — S13 Experiment 2 mechanism check: does neighbour evidence lead own evidence under staggered onset? (RAW; the D65 2.3 gate result is stated beside it)
Date: 2026-10-15 | Session: S13 | Commit: 84923e3 (code); D65 = 245705b; onset_spread_steps option and F11 regression test in the preceding commit
Config: `configs/staggered.yaml` (demo.yaml + degradation.onset_spread_steps 300), seeds 1-5, fraction 0.20, rho 0 and 1.0, **injection draws 50-79** (30 per seed per rho; < 100, not evaluation draws, not among the S3 draws 0, 1, 40, 90), observable `success_ewma` of Baseline-A scenarios; no model trained or used.
Command: `python scripts/s13_stag.py gen mech` then `python scripts/s13_stag.py mech` (results/s13/stag/mech.json). First number of Experiment 2 written 2026-10-15T06:35:28Z (12:05:28 +0530). The staggered traces equal the demo traces bit-for-bit for seeds 1-5 and 100 (checked).
Definition (D65 2.3): late-onset RSU v = degraded RSU with >= 1 degraded same-segment RSU-RSU-edge neighbour u with onset_u < onset_v. t_w = first t in [onset_w, onset_w+400) with success_ewma_w below its own pre-onset 1st percentile (secondary: 0.10 below its pre-onset mean) for 10 consecutive steps; lead_v = mean over earlier-onset neighbours u of (t_v - t_u), positive = neighbour evidence first. Per-seed mean over all (draw, v); 95% t-interval over the 5 seeds (t = 2.776).
```
rule       rho   lead (steps) [95% interval]      sd     mean onset gap   censored (late / earlier)   late-onset RSUs per seed (seeds 1-5)
primary    0     +103.6 [ +70.9, +136.2]          26.3   103.6            0.000 / 0.000               28 25 34 30 28
primary    1.0   +92.3  [ +79.2, +105.5]          10.6   98.1             0.000 / 0.000               73 74 66 82 84
secondary  0     +103.5 [ +84.5, +122.5]          15.3   103.6            0.000 / 0.000               28 25 34 30 28
secondary  1.0   +89.9  [ +72.6, +107.2]          13.9   98.1             0.000 / 0.000               73 74 66 82 84
```
D65 2.3 gate (primary rule, rho 1.0, interval above 0): lo = +79.2 > 0 -> **PASS**; Experiment 2 proceeds to training.
Interpretation: under staggered onset a late-onset RSU's earlier-onset same-segment neighbours show degraded evidence about 90-100 steps before it does, almost exactly the mean onset gap (98 steps at rho 1.0), so the lead is the injected stagger carried through the same observable; neighbour evidence therefore does precede own evidence in this regime, at both rho. This says nothing yet about whether any model uses it.

### F36 � S13 Experiment 2 training: staggered-regime C and D, seeds 1-5 and tuning seed 100 (RAW diagnostics; no C-vs-D evaluation number exists yet)
Date: 2026-10-15 | Session: S13 | Commit: 8ebd9a7 (scripts; training ran at the preceding s3_train.py --ckpt-dir/--threads commit); D65 = 245705b
Config: `configs/staggered.yaml` (onset_spread_steps 300), `TrainSettings` exactly D45 (lr 1e-3, Adam, batch 32, cap 200, patience 8, min_delta 1e-4), D44 architecture (C MLP 1313 params, D 2-layer GINEConv 1614 = 1.23x), S3 data protocol (30 train scenarios, 9 validation draw 40, held-out draw 90 per seed; all draws < 100), Baseline-A dispatch.
Command: `python scripts/s3_train.py --config configs/staggered.yaml --seed <n> --ckpt-dir checkpoints/s13 --threads 3 --out results/s13` for n in 1 2 3 4 5 100, six processes in parallel, 3 torch threads each (the same for every seed). Checkpoints in checkpoints/s13/ (the S3 checkpoints are untouched). All twelve stopped by patience; each trained once, no re-runs, no selection among runs.
```
seed var best_ep stopped untrained_val best_val  heldout_BCE heldout_AUC[450,650) edge probe (same_segment->0 / RSU-RSU->1)
1    C      43      44        0.6833  0.14863     0.15392      0.8125            
1    D     115     123        0.9394  0.13660     0.15297      0.8125            0.0583 / 0.0434
2    C      38      46        0.7564  0.13990     0.14468      0.7344            
2    D      31      39        3.9948  0.18445     0.17249      0.6094            0.1272 / 0.0247
3    C      41      42        0.8007  0.16176     0.16518      0.2656            
3    D     101     109        2.6054  0.15582     0.17827      0.5156            0.1150 / 0.0328
4    C      16      24        0.6658  0.17145     0.16019      0.8438            
4    D      65      73        0.2977  0.16462     0.15456      0.9375            0.0792 / 0.0351
5    C      17      18        0.6396  0.17479     0.16217      0.6250            
5    D     115     123        0.4111  0.15711     0.15225      0.7812            0.0654 / 0.0578
100  C      40      40        0.7284  0.17125     0.17472      0.9062            
100  D      47      55        0.4784  0.17436     0.18048      0.5469            0.0530 / 0.0217
```
D44 fallback checks (validation): edge probe >= 0.01 PASS on all six seeds; best validation BCE below untrained validation BCE PASS on all twelve models. The one-time GATv2 swap is not applied (D65 2.4).
Note: the per-epoch printed sealed diagnostic uses the fixed shared-onset window [450,650) on one validation scenario (draw 40); under staggered onset it printed values of 0.06-0.11 for both C and D on seed 1 (and the held-out value of that window ranges 0.27-0.94 across seeds). It never enters loss, early stopping or selection (L4); it is a window-mismatch diagnostic of a single scenario, not an evaluation number. The pre-registered evaluation uses per-RSU windows (D65 2.5).
Interpretation: all twelve models converged by patience and pass the pre-committed D44 checks; they are frozen as trained.

### F37 — S13 Experiment 2 raw open-loop results: frozen staggered C and D, draws 200-249, rho 1.0 (test) and rho 0 (control) (RAW - no labels, no verdict recorded here)
Date: 2026-10-15 | Session: S13 | Commit: befb34f (analysis code in s13_stag.py unchanged since its commit); models F36; D65 = 245705b
Config: `configs/staggered.yaml`, frozen `checkpoints/s13/` models, seeds 1-5, fraction 0.20, injection draws 200-249 (50 per seed per rho), Baseline-A open-loop scenarios, trust at stride 1 over steps 100-1199.
Command: `python scripts/s13_stag.py gen eval`, `python scripts/s13_stag.py trust`, `python scripts/s13_stag.py eval` (results/s13/stag/report-eval-raw.txt). First evaluation trust file written 2026-10-15T06:50:12Z (12:20:12 +0530), after F36 (12:19:31 +0530).
Definitions (D65 2.5): latency = steps from the RSU's own onset to the first run of 10 consecutive steps with trust strictly below tau inside [onset, onset+400), censored at 399; tau = 1st percentile of the model's trust over all RSU-steps in [100,400) pooled over the seed's 50 draws at that rho; per draw the mean over its late-onset RSUs (draws with none are not aggregated); negative D - C = D detects earlier. Secondary AUC: 1 - mean trust over [first onset on the RSU's segment, +250) (RSUs on a segment with no degraded RSU use the draw's earliest onset), degraded vs reliable, per draw. Intervals: Student-t over 5 seeds (2.776).
```
rho 0  (CONTROL)
  D42 accounting per seed: draws attempted / aggregated for the latency primary (draws with a late-onset RSU) / late-onset RSUs
    seed 1: attempted 50  failed 0  aggregated 37  late-onset RSUs 43  tau C 0.8461 D 0.7380
    seed 2: attempted 50  failed 0  aggregated 38  late-onset RSUs 46  tau C 0.8602 D 0.7423
    seed 3: attempted 50  failed 0  aggregated 30  late-onset RSUs 36  tau C 0.8137 D 0.5951
    seed 4: attempted 50  failed 0  aggregated 29  late-onset RSUs 33  tau C 0.8598 D 0.6547
    seed 5: attempted 50  failed 0  aggregated 30  late-onset RSUs 36  tau C 0.8682 D 0.6161
  detection latency (steps after own onset, late-onset RSUs, censored 399): C 130.1 +/- 15.3   D 226.4 +/- 34.4   detected fraction C 1.000 D 0.946
  PRIMARY  D - C latency: +96.3 (sd 23.8) 95% [+66.8, +125.9]
  per-seed D - C latency: [65.4, 89.9, 130.4, 105.2, 90.7]
  (secondary) window AUC: C 0.9328 D 0.8057   D - C -0.1271 (sd 0.0623) 95% [-0.2044, -0.0498]

rho 1  (TEST)
  D42 accounting per seed: draws attempted / aggregated for the latency primary (draws with a late-onset RSU) / late-onset RSUs
    seed 1: attempted 50  failed 0  aggregated 50  late-onset RSUs 127  tau C 0.8461 D 0.7380
    seed 2: attempted 50  failed 0  aggregated 50  late-onset RSUs 134  tau C 0.8602 D 0.7423
    seed 3: attempted 50  failed 0  aggregated 50  late-onset RSUs 113  tau C 0.8137 D 0.5951
    seed 4: attempted 50  failed 0  aggregated 50  late-onset RSUs 135  tau C 0.8598 D 0.6547
    seed 5: attempted 50  failed 0  aggregated 50  late-onset RSUs 133  tau C 0.8682 D 0.6161
  detection latency (steps after own onset, late-onset RSUs, censored 399): C 126.5 +/- 8.7   D 197.0 +/- 28.8   detected fraction C 1.000 D 0.968
  PRIMARY  D - C latency: +70.5 (sd 22.3) 95% [+42.8, +98.2]
  per-seed D - C latency: [44.6, 56.9, 97.3, 63.9, 89.8]
  (secondary) window AUC: C 0.7942 D 0.7484   D - C -0.0457 (sd 0.0434) 95% [-0.0996, +0.0081]

(description) (D - C latency at rho 1.0) minus (at rho 0), per-seed paired: -25.8 (sd 15.7) 95% [-45.4, -6.3]
```

### F38 — S13 Experiment 2 verdict against D65 and interpretation (written after F37 was committed in aaba1a3)
Date: 2026-10-15 | Session: S13 | Commit: see `git log` | No S, floor, window, seed, draw or rule of D65 was changed.
Labels from `python scripts/s13_stag.py eval --verdict` (results/s13/stag/report-eval-verdict.txt), D65 section 2.5 (primary: detection latency of late-onset RSUs, D - C, steps; WIN for D = hi < 0 and mean <= -20; NEGATIVE = lo > 0 and mean >= 20; NULL = interval inside +-20; else NOISY):
```
                          D - C latency [95% interval over 5 seeds]    label        C latency   D latency   detected C / D
rho 1.0 (test)            +70.5 [+42.8, +98.2]                         NEGATIVE     126.5       197.0       1.000 / 0.968
rho 0   (control)         +96.3 [+66.8, +125.9]                        NEGATIVE     130.1       226.4       1.000 / 0.946
```
Rule applied: at rho 1.0, WIN needs hi < 0 (hi = +98.2, no); NEGATIVE needs lo > 0 (+42.8 yes) and mean >= 20 (+70.5 yes) -> NEGATIVE. The label is not NOISY, so the D65 seeds-6-10 extension is not triggered and was not run. D65 reading of the control: the label at rho 1.0 is not WIN, so there is no structural reading to make; D is later than C at rho 0 as well, by more (+96.3 vs +70.5).
Secondary, descriptive (no label): window AUC D - C = -0.0457 [-0.0996, +0.0081] at rho 1.0 and -0.1271 [-0.2044, -0.0498] at rho 0; (D - C latency at rho 1.0) minus (at rho 0), per-seed paired: -25.8 [-45.4, -6.3].
Interpretation:
1. Under staggered onset, where F35 shows neighbour evidence leading own evidence by about 90-100 steps, the GNN head D did not detect late-onset RSUs earlier than the node-local head C: D was 43-98 steps later (95% interval) at rho 1.0 and 67-126 later at rho 0, with all five seeds having D later at both rho (per seed +44.6 to +97.3 at rho 1.0). The pre-registered label is NEGATIVE (C beats D) at rho 1.0.
2. D's deficit is smaller at rho 1.0 than at rho 0 (-25.8 [-45.4, -6.3] on the difference): the direction of F19's gap-narrowing with correlation, here with an interval excluding 0. It is a narrowing of a deficit, not an advantage, and was not a pre-registered test.
3. So the F18/F19 boundary condition (shared onset removes the lead-time advantage) is not the whole explanation of why D trails C in this simulation: with a neighbour lead present (F35) and the same training protocol, D still trails C. That is a statement about these two heads trained under D44/D45 in this scenario; mechanisms (dilution by healthy neighbours F19, the D45 training budget, seed fit) are untested hypotheses, not findings. This result does not revise F17 or any H2 verdict; it is a separate regime.
4. D42: rho 1.0: 250 draws attempted, 0 failed, 250 aggregated; rho 0: 250 attempted, 0 failed, 164 aggregated (draws with no late-onset RSU are not aggregated: 37/38/30/29/30 per seed), late-onset RSUs 194 in total at rho 0 against 642 at rho 1.0.
Limits: one injection model, S = 300 only, one scenario per seed (topology and traffic), models trained on Baseline-A open-loop dispatch, latency tau is the 1st percentile of pre-onset trust (so C's and D's thresholds differ: C 0.81-0.87, D 0.60-0.74, F37), detection of a gradual ramp with a 10-step run rule; and secondary AUC windows that do not match the training diagnostic windows.

### F39 — S13 Experiment 2 secondary (b): closed-loop tuning grids on the tuning seed under staggered onset (RAW, recorded before the chosen weights are used anywhere)
Date: 2026-10-15 | Session: S13 | Commit: 1b7bf19; D65 section 2.5 = D52 procedure
Config: `configs/staggered.yaml`, tuning seed 100 (excluded from evaluation), rho 1.0, fraction 0.20, injection draws 0, 1, 2, closed loop, dwell filter and handoff on, seed-100 staggered C and D checkpoints (F36); A over beta x gamma at alpha 0; C and D over alpha {0.25,0.5,1,2,4} x beta {0.25,0.5,1} x gamma {0.15,0.3,0.6}. Objective: deadline success of dispatched tasks over steps [400, 1200), mean over the 3 draws; ties to the first grid point in ascending order.
Command: `python scripts/s13_stag.py tune --write` (results/s13/stag/tuning-grid.json; configs/weights_staggered.yaml). 378 runs. Chosen: A (0, 0.25, 0.6) 0.8860; C (2, 0.25, 0.15) 0.9139; D (2, 0.25, 0.6) 0.9060.
```
=== A: objective = mean post-onset deadline success over draws (0, 1, 2) (staggered onset, seed 100)
  alpha  beta  gamma   objective   per-draw
   0.00  0.25  0.15   0.8762    [0.8875, 0.8562, 0.8849]
   0.00  0.25  0.30   0.8830    [0.8971, 0.8619, 0.8899]
   0.00  0.25  0.60   0.8860    [0.9049, 0.8634, 0.8897]  <== chosen
   0.00  0.50  0.15   0.8712    [0.8834, 0.8516, 0.8786]
   0.00  0.50  0.30   0.8762    [0.8875, 0.8562, 0.8849]
   0.00  0.50  0.60   0.8830    [0.8971, 0.8619, 0.8899]
   0.00  1.00  0.15   0.8663    [0.8764, 0.8462, 0.8762]
   0.00  1.00  0.30   0.8712    [0.8834, 0.8516, 0.8786]
   0.00  1.00  0.60   0.8762    [0.8875, 0.8562, 0.8849]

=== C: objective = mean post-onset deadline success over draws (0, 1, 2) (staggered onset, seed 100)
  alpha  beta  gamma   objective   per-draw
   0.25  0.25  0.15   0.9092    [0.9415, 0.8751, 0.911]
   0.25  0.25  0.30   0.9090    [0.9397, 0.8751, 0.9121]
   0.25  0.25  0.60   0.9062    [0.9347, 0.874, 0.9099]
   0.25  0.50  0.15   0.9014    [0.9304, 0.8695, 0.9043]
   0.25  0.50  0.30   0.9031    [0.9308, 0.8721, 0.9065]
   0.25  0.50  0.60   0.9028    [0.9276, 0.8725, 0.9084]
   0.25  1.00  0.15   0.8894    [0.9106, 0.8614, 0.8962]
   0.25  1.00  0.30   0.8922    [0.9136, 0.8643, 0.8988]
   0.25  1.00  0.60   0.8944    [0.9154, 0.8669, 0.9008]
   0.50  0.25  0.15   0.9115    [0.9441, 0.8767, 0.9139]
   0.50  0.25  0.30   0.9126    [0.945, 0.8775, 0.9152]
   0.50  0.25  0.60   0.9113    [0.943, 0.8764, 0.9145]
   0.50  0.50  0.15   0.9085    [0.9415, 0.874, 0.9099]
   0.50  0.50  0.30   0.9092    [0.9415, 0.8751, 0.911]
   0.50  0.50  0.60   0.9090    [0.9397, 0.8751, 0.9121]
   0.50  1.00  0.15   0.9000    [0.9295, 0.8671, 0.9034]
   0.50  1.00  0.30   0.9014    [0.9304, 0.8695, 0.9043]
   0.50  1.00  0.60   0.9031    [0.9308, 0.8721, 0.9065]
   1.00  0.25  0.15   0.9130    [0.9469, 0.8773, 0.9147]
   1.00  0.25  0.30   0.9135    [0.9474, 0.8769, 0.9162]
   1.00  0.25  0.60   0.9136    [0.9465, 0.8777, 0.9167]
   1.00  0.50  0.15   0.9107    [0.9439, 0.8758, 0.9126]
   1.00  0.50  0.30   0.9115    [0.9441, 0.8767, 0.9139]
   1.00  0.50  0.60   0.9126    [0.945, 0.8775, 0.9152]
   1.00  1.00  0.15   0.9073    [0.9406, 0.8725, 0.9089]
   1.00  1.00  0.30   0.9085    [0.9415, 0.874, 0.9099]
   1.00  1.00  0.60   0.9092    [0.9415, 0.8751, 0.911]
   2.00  0.25  0.15   0.9139    [0.9476, 0.8771, 0.9169]  <== chosen
   2.00  0.25  0.30   0.9136    [0.9471, 0.8773, 0.9162]
   2.00  0.25  0.60   0.9139    [0.9476, 0.8777, 0.9162]
   2.00  0.50  0.15   0.9120    [0.9458, 0.876, 0.9141]
   2.00  0.50  0.30   0.9130    [0.9469, 0.8773, 0.9147]
   2.00  0.50  0.60   0.9135    [0.9474, 0.8769, 0.9162]
   2.00  1.00  0.15   0.9102    [0.9441, 0.8745, 0.9119]
   2.00  1.00  0.30   0.9107    [0.9439, 0.8758, 0.9126]
   2.00  1.00  0.60   0.9115    [0.9441, 0.8767, 0.9139]
   4.00  0.25  0.15   0.9133    [0.9474, 0.8764, 0.9162]
   4.00  0.25  0.30   0.9137    [0.9476, 0.8769, 0.9167]
   4.00  0.25  0.60   0.9137    [0.9471, 0.8777, 0.9162]
   4.00  0.50  0.15   0.9138    [0.948, 0.8769, 0.9165]
   4.00  0.50  0.30   0.9139    [0.9476, 0.8771, 0.9169]
   4.00  0.50  0.60   0.9136    [0.9471, 0.8773, 0.9162]
   4.00  1.00  0.15   0.9113    [0.9454, 0.8749, 0.9136]
   4.00  1.00  0.30   0.9120    [0.9458, 0.876, 0.9141]
   4.00  1.00  0.60   0.9130    [0.9469, 0.8773, 0.9147]

=== D: objective = mean post-onset deadline success over draws (0, 1, 2) (staggered onset, seed 100)
  alpha  beta  gamma   objective   per-draw
   0.25  0.25  0.15   0.9015    [0.9291, 0.8693, 0.906]
   0.25  0.25  0.30   0.9025    [0.9286, 0.871, 0.9078]
   0.25  0.25  0.60   0.8991    [0.925, 0.8701, 0.9023]
   0.25  0.50  0.15   0.8921    [0.9158, 0.8627, 0.8978]
   0.25  0.50  0.30   0.8955    [0.9184, 0.8667, 0.9015]
   0.25  0.50  0.60   0.8963    [0.9191, 0.8684, 0.9015]
   0.25  1.00  0.15   0.8806    [0.8973, 0.8551, 0.8895]
   0.25  1.00  0.30   0.8849    [0.9019, 0.859, 0.8938]
   0.25  1.00  0.60   0.8883    [0.906, 0.8636, 0.8951]
   0.50  0.25  0.15   0.9039    [0.9354, 0.869, 0.9071]
   0.50  0.25  0.30   0.9042    [0.9347, 0.8699, 0.908]
   0.50  0.25  0.60   0.9036    [0.9332, 0.8701, 0.9073]
   0.50  0.50  0.15   0.8998    [0.9295, 0.8662, 0.9036]
   0.50  0.50  0.30   0.9015    [0.9291, 0.8693, 0.906]
   0.50  0.50  0.60   0.9025    [0.9286, 0.871, 0.9078]
   0.50  1.00  0.15   0.8896    [0.9145, 0.8595, 0.8949]
   0.50  1.00  0.30   0.8921    [0.9158, 0.8627, 0.8978]
   0.50  1.00  0.60   0.8955    [0.9184, 0.8667, 0.9015]
   1.00  0.25  0.15   0.9039    [0.9367, 0.8669, 0.908]
   1.00  0.25  0.30   0.9058    [0.9389, 0.8695, 0.9091]
   1.00  0.25  0.60   0.9056    [0.938, 0.8701, 0.9086]
   1.00  0.50  0.15   0.9015    [0.9334, 0.8656, 0.9054]
   1.00  0.50  0.30   0.9039    [0.9354, 0.869, 0.9071]
   1.00  0.50  0.60   0.9042    [0.9347, 0.8699, 0.908]
   1.00  1.00  0.15   0.8971    [0.9265, 0.8645, 0.9004]
   1.00  1.00  0.30   0.8998    [0.9295, 0.8662, 0.9036]
   1.00  1.00  0.60   0.9015    [0.9291, 0.8693, 0.906]
   2.00  0.25  0.15   0.9039    [0.9382, 0.8656, 0.908]
   2.00  0.25  0.30   0.9046    [0.9389, 0.8664, 0.9086]
   2.00  0.25  0.60   0.9060    [0.9395, 0.8684, 0.9102]  <== chosen
   2.00  0.50  0.15   0.9023    [0.935, 0.8651, 0.9069]
   2.00  0.50  0.30   0.9039    [0.9367, 0.8669, 0.908]
   2.00  0.50  0.60   0.9058    [0.9389, 0.8695, 0.9091]
   2.00  1.00  0.15   0.8995    [0.9321, 0.864, 0.9023]
   2.00  1.00  0.30   0.9015    [0.9334, 0.8656, 0.9054]
   2.00  1.00  0.60   0.9039    [0.9354, 0.869, 0.9071]
   4.00  0.25  0.15   0.9022    [0.9363, 0.8638, 0.9065]
   4.00  0.25  0.30   0.9038    [0.938, 0.8653, 0.908]
   4.00  0.25  0.60   0.9046    [0.9384, 0.8667, 0.9089]
   4.00  0.50  0.15   0.9023    [0.9369, 0.864, 0.906]
   4.00  0.50  0.30   0.9039    [0.9382, 0.8656, 0.908]
   4.00  0.50  0.60   0.9046    [0.9389, 0.8664, 0.9086]
   4.00  1.00  0.15   0.9007    [0.9334, 0.8634, 0.9054]
   4.00  1.00  0.30   0.9023    [0.935, 0.8651, 0.9069]
   4.00  1.00  0.60   0.9039    [0.9367, 0.8669, 0.908]

```
For reference only (F20, shared onset): A 0.8670, C 0.8999, D 0.8938. One seed, three draws; per-draw values differ by up to 0.09, so differences of ~0.005 between grid points are inside the draw-to-draw spread (as F20).
Interpretation: recorded before use; the chosen weights are applied unchanged in the evaluation (D65).

### F40 — S13 Experiment 2 secondary (b): closed-loop post-onset success under staggered onset, A / C / D with their own tuned weights (RAW, descriptive: D65 assigns no label to this)
Date: 2026-10-15 | Session: S13 | Commit: see `git log` (weights F39 = 301255e); D65 = 245705b
Config: `configs/staggered.yaml`, frozen staggered C and D (F36), weights `configs/weights_staggered.yaml` (F39), seeds 1-5, rho 1.0 and 0, fraction 0.20, injection draws 200-249, closed loop, dwell filter and handoff on; 1500 runs (3 variants x 5 seeds x 2 rho x 50 draws), 0 failed, 6 min on 12 workers.
Command: `python scripts/s13_stag.py closed` then `python scripts/s13_stag.py report` (results/s13/stag/report-closed.txt). Metric: deadline success of dispatched tasks at steps >= 400; mean +/- sd across the 5 seed means; differences paired per draw, per-seed means, Student-t 95% over 5 seeds.
```
rho 0  attempted 250 failed 0 aggregated 250
  A: 0.8978 +/- 0.0020
  C: 0.9547 +/- 0.0029
  D: 0.9376 +/- 0.0056
  D - C: -0.0171 (sd 0.0075) 95% [-0.0265, -0.0078]   (descriptive, no label)
  C - A: +0.0569 (sd 0.0009) 95% [+0.0558, +0.0581]   (descriptive, no label)
  D - A: +0.0398 (sd 0.0069) 95% [+0.0313, +0.0483]   (descriptive, no label)

rho 1  attempted 250 failed 0 aggregated 250
  A: 0.8958 +/- 0.0027
  C: 0.9336 +/- 0.0066
  D: 0.9178 +/- 0.0033
  D - C: -0.0158 (sd 0.0041) 95% [-0.0210, -0.0107]   (descriptive, no label)
  C - A: +0.0378 (sd 0.0056) 95% [+0.0308, +0.0448]   (descriptive, no label)
  D - A: +0.0219 (sd 0.0027) 95% [+0.0186, +0.0253]   (descriptive, no label)
```
Interpretation: under staggered onset, with each model's own weights, C is above D in post-onset success by 0.016-0.017 at both rho (intervals exclude 0), and both are above no trust (A) by 0.022-0.057. Descriptive only: no pre-registered label applies to this secondary, and it is a closed-loop run of heads trained on open-loop dispatch (F21 item 3).

### F41 — S13b note on F36: the per-epoch sealed-diagnostic window under staggered onset is print-only (dated note; F36's numbers are not edited)
Date: 2026-10-15 | Session: S13b | Commit: see `git log` | Resolves the S13 review's Q4 (D67)
What F36 observed: the per-epoch sealed diagnostic that `s3_train.py` prints (AUC of trust against the true class over the fixed shared-onset window [450, 650), one validation scenario, draw 40, rho 1.0, fraction 0.2) printed 0.06-0.11 for both C and D on seed 1 during training, and the held-out value of the same window ranged 0.27-0.94 across the twelve models. Under staggered onset an RSU's degradation can begin anywhere in [400, 700], so a fixed window of 200 steps from 450 is not matched to when the evidence appears; a value well below 0.5 on one scenario with 4 degraded and 16 reliable RSUs is a window-and-sample artefact of that diagnostic, not an evaluation number.
Status: the diagnostic is print-only by construction (L4): it never enters the loss, early stopping or the held-out-BCE model selection, and the S3 protocol and D45 settings were applied unchanged, so the twelve models are what the protocol produced. No evaluation number uses this window; the pre-registered staggered evaluation uses per-RSU onsets and windows (D65 2.5; F37). No model was re-trained or re-selected. This is recorded as it is.

### F42 — S14 D2 training: parameter counts, convergence, neighbour-block probe (RAW diagnostics; no D2-vs-C evaluation number exists yet)
Date: 2026-10-15 | Commit: see git log (this entry) | Config: configs/demo.yaml (D2-shared), configs/staggered.yaml (D2-stagger); D68 section 2; D45 settings; 3 torch threads; seeds 1-5 and 100.
Parameters (from the code, asserted by tests/test_d2.py for C and D2): C 1313, **D2 1570** (C + 224 neighbour encoder + 33 neighbour head weights), D 1614 (S3 checkpoint). D2 / C = 1.196, D / C = 1.229.
Per seed, validation BCE (selection quantity) and the D68 neighbour-block probe (mean |change in RSU trust| when the neighbour block is zeroed, on the seed's validation graphs; threshold 0.01). C values from the existing S3 / S13 result files, same protocol.

| regime | seed | D2 initial val BCE | D2 best val BCE (epoch / stop) | C best val BCE (epoch) | D2 held-out BCE | C held-out BCE | probe |
|---|---|---|---|---|---|---|---|
| shared | 1 | 0.6838 | 0.1656 (30 / 37) | 0.1661 (16) | 0.1669 | 0.1680 | 0.0475 |
| shared | 2 | 0.7527 | 0.1487 (41 / 49) | 0.1483 (47) | 0.1669 | 0.1677 | 0.0511 |
| shared | 3 | 0.7988 | 0.1668 (34 / 34) | 0.1679 (24) | 0.1813 | 0.1831 | 0.0804 |
| shared | 4 | 0.6671 | 0.1851 (14 / 21) | 0.1867 (16) | 0.1698 | 0.1712 | 0.0625 |
| shared | 5 | 0.6422 | 0.1846 (33 / 35) | 0.1859 (21) | 0.1710 | 0.1722 | 0.0148 |
| shared | 100 | 0.7277 | 0.1773 (22 / 28) | 0.1762 (46) | 0.1800 | 0.1788 | 0.0526 |
| stag | 1 | 0.6833 | 0.1484 (43 / 51) | 0.1486 (43) | 0.1537 | 0.1539 | 0.0738 |
| stag | 2 | 0.7564 | 0.1409 (38 / 46) | 0.1399 (38) | 0.1432 | 0.1447 | 0.0717 |
| stag | 3 | 0.8007 | 0.1608 (34 / 40) | 0.1618 (41) | 0.1646 | 0.1652 | 0.1151 |
| stag | 4 | 0.6658 | 0.1714 (12 / 20) | 0.1715 (16) | 0.1590 | 0.1602 | 0.1067 |
| stag | 5 | 0.6396 | 0.1746 (16 / 18) | 0.1748 (17) | 0.1622 | 0.1622 | 0.0177 |
| stag | 100 | 0.7284 | 0.1700 (37 / 40) | 0.1713 (40) | 0.1742 | 0.1747 | 0.0657 |

All 12 D2 trainings stopped by patience. D44-style checks: best validation BCE is below the untrained model's in 12 of 12; the neighbour probe is >= 0.01 in 12 of 12 (smallest 0.0148), so the output depends on the neighbour block in every model and the "does not depend" report of D68 is not triggered. Interpretation (one line): D2 fits the self-supervised objective about as well as C (validation BCE within 0.002 in either direction) and uses its neighbour block; nothing about detection is implied. Held-out diagnostic AUC (S3 draw 90, printed by the trainer, not a result of this experiment): D2 equals C to 4 decimals in the shared regime (1.0000, all seeds) and is within +0.03 of C in the staggered regime. The per-epoch printed AUC window under staggered onset is print-only (F41). Seed 1 of the shared C was trained at the default thread count (D47); all D2 at 3.

### F43 � S14 D2 closed-loop tuning grids on the tuning seed, one per regime (RAW, recorded before the chosen weights are used anywhere)
Date: 2026-10-15 | Commit: this entry | D68 section 5(b) = D52 procedure: seed 100, rho 1.0, fraction 0.20, draws {0,1,2}, closed loop (dwell filter, handoff on), objective = mean post-onset (steps >= 400) deadline success; ties to the first grid point. Seed-100 D2 checkpoints of each regime. Run by `scripts/s14_d2.py tune --write`; first tuning result logged 2026-10-15T08:51:57Z (14:21:57 +0530). Full grid in results/s14/tuning-grid.json (untracked).

**D2-shared** chosen (alpha, beta, gamma) = (2.0, 0.25, 0.6), objective 0.9002. Compare: C shared weights (1.0, 0.25, 0.6), C staggered (2.0, 0.25, 0.15).

| alpha | beta | gamma | objective | draw 0 | draw 1 | draw 2 |
|---|---|---|---|---|---|---|
| 0.25 | 0.25 | 0.15 | 0.8946 | 0.9337 | 0.8490 | 0.9012 |
| 0.25 | 0.25 | 0.3 | 0.8950 | 0.9315 | 0.8503 | 0.9032 |
| 0.25 | 0.25 | 0.6 | 0.8924 | 0.9265 | 0.8503 | 0.9004 |
| 0.25 | 0.5 | 0.15 | 0.8858 | 0.9202 | 0.8438 | 0.8934 |
| 0.25 | 0.5 | 0.3 | 0.8880 | 0.9213 | 0.8471 | 0.8958 |
| 0.25 | 0.5 | 0.6 | 0.8877 | 0.9182 | 0.8488 | 0.8960 |
| 0.25 | 1.0 | 0.15 | 0.8725 | 0.8978 | 0.8369 | 0.8830 |
| 0.25 | 1.0 | 0.3 | 0.8748 | 0.8988 | 0.8395 | 0.8860 |
| 0.25 | 1.0 | 0.6 | 0.8779 | 0.9028 | 0.8425 | 0.8884 |
| 0.5 | 0.25 | 0.15 | 0.8972 | 0.9374 | 0.8497 | 0.9045 |
| 0.5 | 0.25 | 0.3 | 0.8979 | 0.9374 | 0.8512 | 0.9052 |
| 0.5 | 0.25 | 0.6 | 0.8975 | 0.9352 | 0.8519 | 0.9054 |
| 0.5 | 0.5 | 0.15 | 0.8934 | 0.9339 | 0.8471 | 0.8993 |
| 0.5 | 0.5 | 0.3 | 0.8946 | 0.9337 | 0.8490 | 0.9012 |
| 0.5 | 0.5 | 0.6 | 0.8950 | 0.9315 | 0.8503 | 0.9032 |
| 0.5 | 1.0 | 0.15 | 0.8838 | 0.9204 | 0.8403 | 0.8908 |
| 0.5 | 1.0 | 0.3 | 0.8858 | 0.9202 | 0.8438 | 0.8934 |
| 0.5 | 1.0 | 0.6 | 0.8880 | 0.9213 | 0.8471 | 0.8958 |
| 1.0 | 0.25 | 0.15 | 0.8988 | 0.9404 | 0.8506 | 0.9056 |
| 1.0 | 0.25 | 0.3 | 0.8996 | 0.9406 | 0.8514 | 0.9069 |
| 1.0 | 0.25 | 0.6 | 0.8998 | 0.9397 | 0.8525 | 0.9071 |
| 1.0 | 0.5 | 0.15 | 0.8969 | 0.9378 | 0.8495 | 0.9034 |
| 1.0 | 0.5 | 0.3 | 0.8972 | 0.9374 | 0.8497 | 0.9045 |
| 1.0 | 0.5 | 0.6 | 0.8979 | 0.9374 | 0.8512 | 0.9052 |
| 1.0 | 1.0 | 0.15 | 0.8925 | 0.9339 | 0.8451 | 0.8986 |
| 1.0 | 1.0 | 0.3 | 0.8934 | 0.9339 | 0.8471 | 0.8993 |
| 1.0 | 1.0 | 0.6 | 0.8946 | 0.9337 | 0.8490 | 0.9012 |
| 2.0 | 0.25 | 0.15 | 0.8999 | 0.9417 | 0.8512 | 0.9067 |
| 2.0 | 0.25 | 0.3 | 0.8997 | 0.9415 | 0.8506 | 0.9071 |
| 2.0 | 0.25 | 0.6 | 0.9002 **chosen** | 0.9415 | 0.8519 | 0.9073 |
| 2.0 | 0.5 | 0.15 | 0.8982 | 0.9397 | 0.8499 | 0.9049 |
| 2.0 | 0.5 | 0.3 | 0.8988 | 0.9404 | 0.8506 | 0.9056 |
| 2.0 | 0.5 | 0.6 | 0.8996 | 0.9406 | 0.8514 | 0.9069 |
| 2.0 | 1.0 | 0.15 | 0.8962 | 0.9378 | 0.8482 | 0.9028 |
| 2.0 | 1.0 | 0.3 | 0.8969 | 0.9378 | 0.8495 | 0.9034 |
| 2.0 | 1.0 | 0.6 | 0.8972 | 0.9374 | 0.8497 | 0.9045 |
| 4.0 | 0.25 | 0.15 | 0.8999 | 0.9419 | 0.8506 | 0.9073 |
| 4.0 | 0.25 | 0.3 | 0.8997 | 0.9417 | 0.8503 | 0.9071 |
| 4.0 | 0.25 | 0.6 | 0.8999 | 0.9417 | 0.8510 | 0.9071 |
| 4.0 | 0.5 | 0.15 | 0.8998 | 0.9415 | 0.8512 | 0.9067 |
| 4.0 | 0.5 | 0.3 | 0.8999 | 0.9417 | 0.8512 | 0.9067 |
| 4.0 | 0.5 | 0.6 | 0.8997 | 0.9415 | 0.8506 | 0.9071 |
| 4.0 | 1.0 | 0.15 | 0.8979 | 0.9397 | 0.8495 | 0.9045 |
| 4.0 | 1.0 | 0.3 | 0.8982 | 0.9397 | 0.8499 | 0.9049 |
| 4.0 | 1.0 | 0.6 | 0.8988 | 0.9404 | 0.8506 | 0.9056 |

Spread: best 0.9002, worst 0.8725; within 0.002 of the best: 15 of 45 points.

**D2-stagger** chosen (alpha, beta, gamma) = (1.0, 0.25, 0.3), objective 0.9140. Compare: C shared weights (1.0, 0.25, 0.6), C staggered (2.0, 0.25, 0.15).

| alpha | beta | gamma | objective | draw 0 | draw 1 | draw 2 |
|---|---|---|---|---|---|---|
| 0.25 | 0.25 | 0.15 | 0.9097 | 0.9417 | 0.8756 | 0.9119 |
| 0.25 | 0.25 | 0.3 | 0.9093 | 0.9395 | 0.8754 | 0.9130 |
| 0.25 | 0.25 | 0.6 | 0.9064 | 0.9343 | 0.8740 | 0.9108 |
| 0.25 | 0.5 | 0.15 | 0.9015 | 0.9300 | 0.8693 | 0.9052 |
| 0.25 | 0.5 | 0.3 | 0.9039 | 0.9315 | 0.8723 | 0.9080 |
| 0.25 | 0.5 | 0.6 | 0.9028 | 0.9273 | 0.8727 | 0.9084 |
| 0.25 | 1.0 | 0.15 | 0.8891 | 0.9102 | 0.8610 | 0.8960 |
| 0.25 | 1.0 | 0.3 | 0.8917 | 0.9130 | 0.8636 | 0.8984 |
| 0.25 | 1.0 | 0.6 | 0.8941 | 0.9149 | 0.8664 | 0.9010 |
| 0.5 | 0.25 | 0.15 | 0.9110 | 0.9439 | 0.8758 | 0.9134 |
| 0.5 | 0.25 | 0.3 | 0.9126 | 0.9450 | 0.8773 | 0.9154 |
| 0.5 | 0.25 | 0.6 | 0.9115 | 0.9426 | 0.8767 | 0.9154 |
| 0.5 | 0.5 | 0.15 | 0.9082 | 0.9415 | 0.8730 | 0.9102 |
| 0.5 | 0.5 | 0.3 | 0.9097 | 0.9417 | 0.8756 | 0.9119 |
| 0.5 | 0.5 | 0.6 | 0.9093 | 0.9395 | 0.8754 | 0.9130 |
| 0.5 | 1.0 | 0.15 | 0.8989 | 0.9278 | 0.8664 | 0.9025 |
| 0.5 | 1.0 | 0.3 | 0.9015 | 0.9300 | 0.8693 | 0.9052 |
| 0.5 | 1.0 | 0.6 | 0.9039 | 0.9315 | 0.8723 | 0.9080 |
| 1.0 | 0.25 | 0.15 | 0.9128 | 0.9463 | 0.8773 | 0.9149 |
| 1.0 | 0.25 | 0.3 | 0.9140 **chosen** | 0.9476 | 0.8777 | 0.9167 |
| 1.0 | 0.25 | 0.6 | 0.9133 | 0.9463 | 0.8773 | 0.9165 |
| 1.0 | 0.5 | 0.15 | 0.9107 | 0.9437 | 0.8758 | 0.9126 |
| 1.0 | 0.5 | 0.3 | 0.9110 | 0.9439 | 0.8758 | 0.9134 |
| 1.0 | 0.5 | 0.6 | 0.9126 | 0.9450 | 0.8773 | 0.9154 |
| 1.0 | 1.0 | 0.15 | 0.9067 | 0.9402 | 0.8710 | 0.9089 |
| 1.0 | 1.0 | 0.3 | 0.9082 | 0.9415 | 0.8730 | 0.9102 |
| 1.0 | 1.0 | 0.6 | 0.9097 | 0.9417 | 0.8756 | 0.9119 |
| 2.0 | 0.25 | 0.15 | 0.9139 | 0.9476 | 0.8775 | 0.9167 |
| 2.0 | 0.25 | 0.3 | 0.9135 | 0.9476 | 0.8769 | 0.9160 |
| 2.0 | 0.25 | 0.6 | 0.9140 | 0.9476 | 0.8777 | 0.9167 |
| 2.0 | 0.5 | 0.15 | 0.9122 | 0.9458 | 0.8764 | 0.9143 |
| 2.0 | 0.5 | 0.3 | 0.9128 | 0.9463 | 0.8773 | 0.9149 |
| 2.0 | 0.5 | 0.6 | 0.9140 | 0.9476 | 0.8777 | 0.9167 |
| 2.0 | 1.0 | 0.15 | 0.9098 | 0.9434 | 0.8747 | 0.9112 |
| 2.0 | 1.0 | 0.3 | 0.9107 | 0.9437 | 0.8758 | 0.9126 |
| 2.0 | 1.0 | 0.6 | 0.9110 | 0.9439 | 0.8758 | 0.9134 |
| 4.0 | 0.25 | 0.15 | 0.9136 | 0.9471 | 0.8769 | 0.9167 |
| 4.0 | 0.25 | 0.3 | 0.9136 | 0.9474 | 0.8769 | 0.9167 |
| 4.0 | 0.25 | 0.6 | 0.9136 | 0.9474 | 0.8771 | 0.9165 |
| 4.0 | 0.5 | 0.15 | 0.9136 | 0.9476 | 0.8769 | 0.9162 |
| 4.0 | 0.5 | 0.3 | 0.9139 | 0.9476 | 0.8775 | 0.9167 |
| 4.0 | 0.5 | 0.6 | 0.9135 | 0.9476 | 0.8769 | 0.9160 |
| 4.0 | 1.0 | 0.15 | 0.9121 | 0.9458 | 0.8764 | 0.9141 |
| 4.0 | 1.0 | 0.3 | 0.9122 | 0.9458 | 0.8764 | 0.9143 |
| 4.0 | 1.0 | 0.6 | 0.9128 | 0.9463 | 0.8773 | 0.9149 |

Spread: best 0.9140, worst 0.8891; within 0.002 of the best: 20 of 45 points.

Interpretation (one line): the surface is flat at the top for D2 in both regimes (many points within 0.002), as for C and D (F20); the weights are written to the new configs/weights_d2.yaml and carry no claim (A18).

### F44 — S14 raw results: frozen C and D2, draws 250-299, seeds 1-5 (RAW - no labels, no verdict recorded here)
Date: 2026-10-15 | Commit: this entry | Pre-registration D68 = ac72b82 (2026-10-15 13:34:48 +0530). Scripts: `scripts/s14_d2.py eval`, `report` (code at c6bb684/fd167d5). C = existing frozen checkpoints (shared: checkpoints/s3-seed*-C.pt; staggered: checkpoints/s13/), D2 = F42 checkpoints, weights: C and A from weights.yaml / weights_staggered.yaml, D2 from weights_d2.yaml (F43). First evaluation number on draws 250-299 (open-loop trust): **2026-10-15T08:58:01Z (14:28:01 +0530)**; first closed-loop evaluation result 09:07:54Z; the tuning grid (seed 100, draws 0-2) 08:51:57Z preceded both. Harness check before the runs: C closed loop recomputed on draw 100 (shared, vs S9 stored) and draw 200 (staggered, vs S13 stored), seed 1, equal bit-for-bit (those draws are not results).

Open-loop (PRIMARY and SECONDARY (a)):

```
======================================================================================================================
S14 PRIMARY: STAGGERED ONSET (S = 300), open-loop frozen C and D2-stagger, draws 250-299, fraction 0.20, seeds [1, 2, 3, 4, 5] (RAW: no labels)
======================================================================================================================

rho 0  (CONTROL)
  D42 accounting per seed: draws attempted / failed / aggregated (draws with a late-onset RSU) / late-onset RSUs
    seed 1: attempted 50  failed 0  aggregated 41  late-onset RSUs 54  tau C 0.8461 D2 0.8387
    seed 2: attempted 50  failed 0  aggregated 33  late-onset RSUs 38  tau C 0.8602 D2 0.8791
    seed 3: attempted 50  failed 0  aggregated 40  late-onset RSUs 49  tau C 0.8137 D2 0.8137
    seed 4: attempted 50  failed 0  aggregated 32  late-onset RSUs 35  tau C 0.8598 D2 0.8483
    seed 5: attempted 50  failed 0  aggregated 37  late-onset RSUs 50  tau C 0.8682 D2 0.8651
  detection latency (steps after own onset, late-onset RSUs, censored 399): C 131.0 +/- 15.7   D2 130.2 +/- 13.2   detected fraction C 1.000 D2 1.000
  PRIMARY  D2 - C latency: -0.8 (sd 2.6) 95% [-4.0, +2.5]
  per-seed D2 - C latency: [0.5, -0.7, -5.3, 1.1, 0.6]

rho 1  (TEST)
  D42 accounting per seed: draws attempted / failed / aggregated (draws with a late-onset RSU) / late-onset RSUs
    seed 1: attempted 50  failed 0  aggregated 50  late-onset RSUs 126  tau C 0.8461 D2 0.8387
    seed 2: attempted 50  failed 0  aggregated 50  late-onset RSUs 125  tau C 0.8602 D2 0.8791
    seed 3: attempted 50  failed 0  aggregated 50  late-onset RSUs 110  tau C 0.8137 D2 0.8137
    seed 4: attempted 50  failed 0  aggregated 50  late-onset RSUs 138  tau C 0.8598 D2 0.8483
    seed 5: attempted 50  failed 0  aggregated 50  late-onset RSUs 124  tau C 0.8682 D2 0.8651
  detection latency (steps after own onset, late-onset RSUs, censored 399): C 128.6 +/- 9.5   D2 126.9 +/- 7.8   detected fraction C 1.000 D2 1.000
  PRIMARY  D2 - C latency: -1.7 (sd 4.2) 95% [-7.0, +3.5]
  per-seed D2 - C latency: [2.8, -0.3, -8.6, -0.4, -2.0]

(description) (D2 - C latency at rho 1.0) minus (at rho 0), per-seed paired: -0.9 (sd 2.3) 95% [-3.8, +1.9]

======================================================================================================================
S14 SECONDARY (a): SHARED ONSET, diagnostic AUC of 1 - trust over [450, 650), degraded vs reliable (D43), draws 250-299, fraction 0.20, seeds [1, 2, 3, 4, 5] (RAW: no labels)
======================================================================================================================

rho 0   D42: per seed attempted 50 failed 0 aggregated (draws with both classes present) shown
    seed 1: attempted 50  failed 0  aggregated 50
    seed 2: attempted 50  failed 0  aggregated 50
    seed 3: attempted 50  failed 0  aggregated 50
    seed 4: attempted 50  failed 0  aggregated 50
    seed 5: attempted 50  failed 0  aggregated 50
  AUC  C 0.9838 +/- 0.0276   D2 0.9829 +/- 0.0302
  D2 - C: -0.0008 (sd 0.0028) 95% [-0.0043, +0.0026]
  per-seed D2 - C: [-0.0056, 0.0, 0.0, 0.0, 0.0016]

rho 1   D42: per seed attempted 50 failed 0 aggregated (draws with both classes present) shown
    seed 1: attempted 50  failed 0  aggregated 50
    seed 2: attempted 50  failed 0  aggregated 50
    seed 3: attempted 50  failed 0  aggregated 50
    seed 4: attempted 50  failed 0  aggregated 50
    seed 5: attempted 50  failed 0  aggregated 50
  AUC  C 0.9773 +/- 0.0385   D2 0.9785 +/- 0.0384
  D2 - C: +0.0012 (sd 0.0025) 95% [-0.0018, +0.0043]
  per-seed D2 - C: [0.0006, 0.0, 0.0, 0.0, 0.0056]
```

Closed loop (SECONDARY (b)):

```
======================================================================================================================
S14 SECONDARY (b): closed-loop post-onset success at the shared point (rho 1.0, fraction 0.20), draws 250-299, seeds [1, 2, 3, 4, 5] (RAW: no labels)
======================================================================================================================

--- regime shared: weights (alpha, beta, gamma) A (0.0, 0.25, 0.6)  C (1.0, 0.25, 0.6)  D2 (2.0, 0.25, 0.6)
    D42: attempted 250  failed 0  aggregated 250
  A   success 0.8713 +/- 0.0032
  C   success 0.9216 +/- 0.0097
  D2  success 0.9219 +/- 0.0096
  D2 - C: +0.0003 (sd 0.0004) 95% [-0.0002, +0.0009]
  C - A: +0.0503 (sd 0.0076) 95% [+0.0408, +0.0598]
  D2 - A: +0.0506 (sd 0.0076) 95% [+0.0412, +0.0600]
  n_av / n_win / never-avoided (mean +/- sd over seeds of per-seed means), always together (D60 Q4):
    A   n_av   627.1 +/-  112.0   n_win   907.1 +/-   17.6   never-avoided 0.488 +/- 0.126
    C   n_av   132.9 +/-   24.1   n_win   499.3 +/-   64.5   never-avoided 0.004 +/- 0.009
    D2  n_av   116.9 +/-   18.7   n_win   479.2 +/-   65.6   never-avoided 0.004 +/- 0.009
  D57-normalised (n_C - n_D2)/n_A on n_av: +0.027 [+0.008, +0.045]  (positive = D2 sends fewer to degraded RSUs; descriptive)
  D57-normalised (n_C - n_D2)/n_A on n_win: +0.022 [+0.011, +0.033]  (positive = D2 sends fewer to degraded RSUs; descriptive)
  detection latency (descriptive) C: 149.9 steps, detected fraction 1.000
  detection latency (descriptive) D2: 150.2 steps, detected fraction 1.000

--- regime stag: weights (alpha, beta, gamma) A (0.0, 0.25, 0.6)  C (2.0, 0.25, 0.15)  D2 (1.0, 0.25, 0.3)
    D42: attempted 250  failed 0  aggregated 250
  A   success 0.8953 +/- 0.0036
  C   success 0.9346 +/- 0.0078
  D2  success 0.9349 +/- 0.0077
  D2 - C: +0.0004 (sd 0.0004) 95% [-0.0001, +0.0008]
  C - A: +0.0393 (sd 0.0051) 95% [+0.0330, +0.0456]
  D2 - A: +0.0396 (sd 0.0051) 95% [+0.0334, +0.0459]
  n_av / n_win / never-avoided (mean +/- sd over seeds of per-seed means), always together (D60 Q4):
    A   n_av   627.1 +/-  112.0   n_win   907.1 +/-   17.6   never-avoided 0.488 +/- 0.126
    C   n_av   217.0 +/-   22.2   n_win   561.5 +/-   48.7   never-avoided 0.004 +/- 0.009
    D2  n_av   232.3 +/-   20.4   n_win   577.7 +/-   49.8   never-avoided 0.004 +/- 0.009
  D57-normalised (n_C - n_D2)/n_A on n_av: -0.024 [-0.042, -0.006]  (positive = D2 sends fewer to degraded RSUs; descriptive)
  D57-normalised (n_C - n_D2)/n_A on n_win: -0.018 [-0.022, -0.013]  (positive = D2 sends fewer to degraded RSUs; descriptive)
  detection latency (descriptive) C: 288.8 steps, detected fraction 0.826
  detection latency (descriptive) D2: 278.4 steps, detected fraction 0.850
```

Note: A is identical in the two regimes in n_av / n_win / never-avoided (627.1 / 907.1 / 0.488) because A ignores trust and its routing does not depend on when RSUs degrade; its success differs (0.8713 vs 0.8953).

### F45 — S14 verdicts against D68 and interpretation (written after F44 was committed in e74a1a0)
Date: 2026-10-15 | Rules exactly as in D68; no rule, window, floor or draw changed. Output: `scripts/s14_d2.py eval --verdict`, `report --verdict`.

| item | quantity | mean [95% t, df 4] | D68 label |
|---|---|---|---|
| PRIMARY, staggered, rho 1.0 | D2 - C latency of late-onset RSUs (steps) | -1.7 [-7.0, +3.5] (C 128.6, D2 126.9) | **NULL** (interval inside +-20) |
| control, staggered, rho 0 | D2 - C latency (steps) | -0.8 [-4.0, +2.5] (C 131.0, D2 130.2) | NULL |
| (a) shared AUC [450,650), rho 1.0 | D2 - C | +0.0012 [-0.0018, +0.0043] | NULL (D43 thresholds, W1 only) |
| (a) shared AUC, rho 0 | D2 - C | -0.0008 [-0.0043, +0.0026] | NULL |
| (b) closed loop, shared regime | success D2 - C | +0.0003 [-0.0002, +0.0009] | NULL (D56 H3 thresholds) |
| (b) closed loop, staggered regime | success D2 - C | +0.0004 [-0.0001, +0.0008] | NULL |

D42 counts: primary rho 1.0 250 attempted / 0 failed / 250 aggregated (late-onset RSUs per seed 110-138); rho 0 250 / 0 / 183 aggregated (draws with no late-onset RSU are not aggregated); AUC 250 / 0 / 250 per rho; closed loop 250 / 0 / 250 per regime. The NOISY extension (seeds 6-10) was **not triggered** (no NOISY label); none was run. The neighbour-block probe was >= 0.01 in 12 of 12 models (F42), so the "no dependence" report is not triggered. Rho 0 reading (D68 section 4): rho 1.0 label NULL, so there is no structural reading; the rho 1.0 minus rho 0 difference of D2-C latency is -0.9 [-3.8, +1.9], nothing.
Closed loop, n_av / n_win / never-avoided together (F44): shared regime A 627.1 / 907.1 / 0.488, C 132.9 / 499.3 / 0.004, D2 116.9 / 479.2 / 0.004; staggered regime A 627.1 / 907.1 / 0.488, C 217.0 / 561.5 / 0.004, D2 232.3 / 577.7 / 0.004. Descriptive D57-normalised (n_C - n_D2)/n_A on n_av: +0.027 [+0.008, +0.045] shared, -0.024 [-0.042, -0.006] staggered (opposite signs, intervals exclude 0 but there is no label and no success difference).
Interpretation (one line): with a head that sees only same-segment RSU neighbours (no vehicles, mean not sum) and starts as C, detection latency, shared-regime AUC and closed-loop success are indistinguishable from C (a latency difference of 20 steps either way is ruled out at rho 1.0), so removing vehicle nodes and the all-neighbour sum did not make the graph help in this simulation, although the models use the neighbour block (probe 0.015-0.115). This is a test of one specific dilution route (D2's construction), not of dilution in general; it does not revise F17, F26, F37 or any H2 verdict.

### F46 - S15 training: C-h and D2-h on the horizon objective, parameter counts, convergence, D2-h neighbour-block probe, label counts (RAW diagnostics; no D2-h-vs-C-h evaluation number exists yet)
Date: 2026-10-15 | Commit: see git log (this entry) | Pre-registration D72 = e7c7d54 (2026-10-15 18:51:15 +0530) | Config: configs/staggered.yaml (S = 300), `scripts/s3_train.py --objective horizon`, D45 settings, 3 torch threads, seeds 1-5 and 100; checkpoints/s15/ (untracked). The first training run crashed in the end-of-run summary line (it assumed variants C and D; no S15 result file was written and none was read); the line was guarded (commit "s3_train parameter-ratio line only for the C/D pair") and all twelve trainings were re-run from scratch; the numbers below are from the re-run. Training is deterministic from (config, seed, threads).
Parameters (from the code): C-h 1313, D2-h 1570. Objective: BCE of trust against z = 1 - y, y = 1 if the fraction of the RSU's tasks in (t, t+150] meeting their deadline is < 0.75, windows with < 5 tasks excluded (D72 section 1).
Validation BCE is the selection quantity (horizon objective, so it is not comparable with the next-task BCEs of F42). Probe = D68/D72 neighbour-block probe: mean |change in RSU trust| on the seed's validation graphs when D2-h's neighbour block is zeroed (threshold 0.01).

| seed | model | initial val BCE | best val BCE (epoch / stop) | held-out BCE | probe |
|---|---|---|---|---|---|
| 1 | C-h | 0.6842 | 0.0541 (46 / 47) | 0.0318 | - |
| 1 | D2-h | 0.6842 | 0.0558 (31 / 39) | 0.0357 | **0.0014** |
| 2 | C-h | 0.7537 | 0.0577 (19 / 27) | 0.0423 | - |
| 2 | D2-h | 0.7537 | 0.0592 (12 / 20) | 0.0446 | 0.0272 |
| 3 | C-h | 0.7968 | 0.0524 (30 / 34) | 0.0583 | - |
| 3 | D2-h | 0.7968 | 0.0501 (34 / 35) | 0.0518 | 0.3856 |
| 4 | C-h | 0.6663 | 0.0412 (40 / 43) | 0.0402 | - |
| 4 | D2-h | 0.6663 | 0.0397 (41 / 48) | 0.0443 | 0.0872 |
| 5 | C-h | 0.6403 | 0.0363 (35 / 37) | 0.0501 | - |
| 5 | D2-h | 0.6403 | 0.0321 (51 / 59) | 0.0431 | 0.0241 |
| 100 | C-h | 0.7271 | 0.0316 (32 / 33) | 0.0331 | - |
| 100 | D2-h | 0.7271 | 0.0257 (69 / 74) | 0.0261 | 0.1554 |

All 12 trainings stopped by patience. D44-style checks: best validation BCE below the untrained model's in 12 of 12. **Probe >= 0.01 in 5 of 6 D2-h models; D2-h seed 1 is 0.0014, below the 0.01 threshold, so for that seed the output does not depend on the neighbour block after training (reported as D72 prescribes; the evaluation includes it).** Probe over the five evaluation seeds: 0.0014, 0.0272, 0.3856, 0.0872, 0.0241 (mean 0.1051); seed 100 (tuning only) 0.1554. The probe is larger than for the next-task D2 (F42: 0.015-0.115 in the staggered regime), with one seed far below.
Label counts (RSU-steps over the 30 staggered training scenarios of the seed, 720000 active RSU-steps each; `scripts/s15_horizon.py counts`): next-task labelled share 0.996-0.997, share of labelled points with a failed next task 0.078-0.087; horizon labelled share 0.980-0.984 (the 5-task rule excludes 1.6-2.0% of active RSU-steps), share of labelled points with y = 1 (f < 0.75) 0.088-0.098 (seeds 1-5; seed 100: 0.997 / 0.081 / 0.980 / 0.098).
Held-out diagnostic AUC over the fixed shared-onset window [450, 650) printed by the trainer (print-only, a window-and-sample artefact under staggered onset, F41; not a result of this experiment): C-h 0.73, 0.84, 0.61, 0.80, 0.78; D2-h 0.69, 0.86, 0.69, 0.83, 0.88 (seeds 1-5).
Interpretation (one line): both heads fit the horizon objective (validation BCE far below the untrained 0.64-0.80; the objective is easier in BCE terms because only about 9% of labelled points are y = 1), D2-h uses its neighbour block in 5 of 6 models but not in seed 1; nothing about detection is implied.

### F47 - S15 closed-loop tuning grids for C-h and D2-h on the tuning seed (RAW, recorded before the chosen weights are used anywhere)
Date: 2026-10-15 | Commit: this entry | D72 section 5a = D52 procedure: seed 100, rho 1.0, fraction 0.20, draws {0,1,2}, staggered regime, closed loop (dwell filter, handoff on), objective = mean post-onset (steps >= 400) deadline success; ties to the first grid point in ascending order. Seed-100 C-h and D2-h checkpoints (F46). Run by `scripts/s15_horizon.py tune --write`; first tuning result logged 2026-10-15T13:31:56Z (UTC). Full grid in results/s15/tuning-grid.json (untracked). Harness check before the grid: the original staggered C, run through the S15 closed-loop and open-loop paths on seed 1, draw 200 (an S13 draw, not a result), equals the stored S13 closed-loop result and the stored S13 trust array exactly.

**C-h** chosen (alpha, beta, gamma) = (0.25, 0.25, 0.3), objective 0.9126. For reference: original staggered C (2.0, 0.25, 0.15) and D2 (1.0, 0.25, 0.3) (F43), A (weights_staggered.yaml).

| alpha | beta | gamma | objective | draw 0 | draw 1 | draw 2 |
|---|---|---|---|---|---|---|
| 0.25 | 0.25 | 0.15 | 0.9089 | 0.9439 | 0.8725 | 0.9104 |
| 0.25 | 0.25 | 0.3 | 0.9126 | 0.9469 | 0.8756 | 0.9154 |
| 0.25 | 0.25 | 0.6 | 0.9124 | 0.9467 | 0.8749 | 0.9156 |
| 0.25 | 0.5 | 0.15 | 0.9058 | 0.9417 | 0.8684 | 0.9073 |
| 0.25 | 0.5 | 0.3 | 0.9084 | 0.9437 | 0.8717 | 0.9099 |
| 0.25 | 0.5 | 0.6 | 0.9099 | 0.9424 | 0.8747 | 0.9126 |
| 0.25 | 1.0 | 0.15 | 0.8926 | 0.9210 | 0.8593 | 0.8975 |
| 0.25 | 1.0 | 0.3 | 0.8962 | 0.9241 | 0.8634 | 0.9010 |
| 0.25 | 1.0 | 0.6 | 0.9005 | 0.9291 | 0.8673 | 0.9052 |
| 0.5 | 0.25 | 0.15 | 0.9060 | 0.9410 | 0.8693 | 0.9078 |
| 0.5 | 0.25 | 0.3 | 0.9085 | 0.9419 | 0.8712 | 0.9123 |
| 0.5 | 0.25 | 0.6 | 0.9126 | 0.9461 | 0.8758 | 0.9158 |
| 0.5 | 0.5 | 0.15 | 0.9034 | 0.9382 | 0.8669 | 0.9052 |
| 0.5 | 0.5 | 0.3 | 0.9089 | 0.9439 | 0.8725 | 0.9104 |
| 0.5 | 0.5 | 0.6 | 0.9126 | 0.9469 | 0.8756 | 0.9154 |
| 0.5 | 1.0 | 0.15 | 0.9028 | 0.9389 | 0.8653 | 0.9041 |
| 0.5 | 1.0 | 0.3 | 0.9058 | 0.9417 | 0.8684 | 0.9073 |
| 0.5 | 1.0 | 0.6 | 0.9084 | 0.9437 | 0.8717 | 0.9099 |
| 1.0 | 0.25 | 0.15 | 0.9041 | 0.9397 | 0.8667 | 0.9060 |
| 1.0 | 0.25 | 0.3 | 0.9068 | 0.9417 | 0.8701 | 0.9086 |
| 1.0 | 0.25 | 0.6 | 0.9094 | 0.9424 | 0.8727 | 0.9132 |
| 1.0 | 0.5 | 0.15 | 0.9040 | 0.9395 | 0.8680 | 0.9045 |
| 1.0 | 0.5 | 0.3 | 0.9060 | 0.9410 | 0.8693 | 0.9078 |
| 1.0 | 0.5 | 0.6 | 0.9085 | 0.9419 | 0.8712 | 0.9123 |
| 1.0 | 1.0 | 0.15 | 0.9020 | 0.9397 | 0.8643 | 0.9021 |
| 1.0 | 1.0 | 0.3 | 0.9034 | 0.9382 | 0.8669 | 0.9052 |
| 1.0 | 1.0 | 0.6 | 0.9089 | 0.9439 | 0.8725 | 0.9104 |
| 2.0 | 0.25 | 0.15 | 0.9037 | 0.9395 | 0.8662 | 0.9054 |
| 2.0 | 0.25 | 0.3 | 0.9054 | 0.9382 | 0.8686 | 0.9093 |
| 2.0 | 0.25 | 0.6 | 0.9086 | 0.9439 | 0.8712 | 0.9106 |
| 2.0 | 0.5 | 0.15 | 0.9017 | 0.9354 | 0.8662 | 0.9034 |
| 2.0 | 0.5 | 0.3 | 0.9041 | 0.9397 | 0.8667 | 0.9060 |
| 2.0 | 0.5 | 0.6 | 0.9068 | 0.9417 | 0.8701 | 0.9086 |
| 2.0 | 1.0 | 0.15 | 0.9013 | 0.9367 | 0.8638 | 0.9034 |
| 2.0 | 1.0 | 0.3 | 0.9040 | 0.9395 | 0.8680 | 0.9045 |
| 2.0 | 1.0 | 0.6 | 0.9060 | 0.9410 | 0.8693 | 0.9078 |
| 4.0 | 0.25 | 0.15 | 0.9023 | 0.9356 | 0.8660 | 0.9052 |
| 4.0 | 0.25 | 0.3 | 0.9046 | 0.9389 | 0.8680 | 0.9071 |
| 4.0 | 0.25 | 0.6 | 0.9072 | 0.9406 | 0.8706 | 0.9104 |
| 4.0 | 0.5 | 0.15 | 0.8984 | 0.9332 | 0.8608 | 0.9012 |
| 4.0 | 0.5 | 0.3 | 0.9037 | 0.9395 | 0.8662 | 0.9054 |
| 4.0 | 0.5 | 0.6 | 0.9054 | 0.9382 | 0.8686 | 0.9093 |
| 4.0 | 1.0 | 0.15 | 0.8997 | 0.9354 | 0.8627 | 0.9010 |
| 4.0 | 1.0 | 0.3 | 0.9017 | 0.9354 | 0.8662 | 0.9034 |
| 4.0 | 1.0 | 0.6 | 0.9041 | 0.9397 | 0.8667 | 0.9060 |

Spread of the 45 objectives: min 0.8926, max 0.9126; points within 0.002 of the best: 4.

**D2-h** chosen (alpha, beta, gamma) = (0.5, 0.25, 0.6), objective 0.9132. For reference: original staggered C (2.0, 0.25, 0.15) and D2 (1.0, 0.25, 0.3) (F43), A (weights_staggered.yaml).

| alpha | beta | gamma | objective | draw 0 | draw 1 | draw 2 |
|---|---|---|---|---|---|---|
| 0.25 | 0.25 | 0.15 | 0.9089 | 0.9443 | 0.8708 | 0.9115 |
| 0.25 | 0.25 | 0.3 | 0.9124 | 0.9467 | 0.8751 | 0.9154 |
| 0.25 | 0.25 | 0.6 | 0.9109 | 0.9432 | 0.8743 | 0.9152 |
| 0.25 | 0.5 | 0.15 | 0.9057 | 0.9419 | 0.8680 | 0.9073 |
| 0.25 | 0.5 | 0.3 | 0.9081 | 0.9434 | 0.8708 | 0.9099 |
| 0.25 | 0.5 | 0.6 | 0.9093 | 0.9417 | 0.8743 | 0.9119 |
| 0.25 | 1.0 | 0.15 | 0.8921 | 0.9202 | 0.8588 | 0.8973 |
| 0.25 | 1.0 | 0.3 | 0.8959 | 0.9236 | 0.8630 | 0.9010 |
| 0.25 | 1.0 | 0.6 | 0.9002 | 0.9282 | 0.8677 | 0.9045 |
| 0.5 | 0.25 | 0.15 | 0.9081 | 0.9430 | 0.8701 | 0.9110 |
| 0.5 | 0.25 | 0.3 | 0.9120 | 0.9465 | 0.8743 | 0.9152 |
| 0.5 | 0.25 | 0.6 | 0.9132 | 0.9480 | 0.8749 | 0.9167 |
| 0.5 | 0.5 | 0.15 | 0.9059 | 0.9421 | 0.8682 | 0.9073 |
| 0.5 | 0.5 | 0.3 | 0.9089 | 0.9443 | 0.8708 | 0.9115 |
| 0.5 | 0.5 | 0.6 | 0.9124 | 0.9467 | 0.8751 | 0.9154 |
| 0.5 | 1.0 | 0.15 | 0.9025 | 0.9389 | 0.8643 | 0.9043 |
| 0.5 | 1.0 | 0.3 | 0.9057 | 0.9419 | 0.8680 | 0.9073 |
| 0.5 | 1.0 | 0.6 | 0.9081 | 0.9434 | 0.8708 | 0.9099 |
| 1.0 | 0.25 | 0.15 | 0.9067 | 0.9417 | 0.8688 | 0.9095 |
| 1.0 | 0.25 | 0.3 | 0.9102 | 0.9447 | 0.8732 | 0.9126 |
| 1.0 | 0.25 | 0.6 | 0.9128 | 0.9474 | 0.8751 | 0.9160 |
| 1.0 | 0.5 | 0.15 | 0.9042 | 0.9404 | 0.8658 | 0.9065 |
| 1.0 | 0.5 | 0.3 | 0.9081 | 0.9430 | 0.8701 | 0.9110 |
| 1.0 | 0.5 | 0.6 | 0.9120 | 0.9465 | 0.8743 | 0.9152 |
| 1.0 | 1.0 | 0.15 | 0.9029 | 0.9400 | 0.8643 | 0.9045 |
| 1.0 | 1.0 | 0.3 | 0.9059 | 0.9421 | 0.8682 | 0.9073 |
| 1.0 | 1.0 | 0.6 | 0.9089 | 0.9443 | 0.8708 | 0.9115 |
| 2.0 | 0.25 | 0.15 | 0.9044 | 0.9400 | 0.8660 | 0.9071 |
| 2.0 | 0.25 | 0.3 | 0.9065 | 0.9402 | 0.8684 | 0.9110 |
| 2.0 | 0.25 | 0.6 | 0.9119 | 0.9458 | 0.8738 | 0.9160 |
| 2.0 | 0.5 | 0.15 | 0.9037 | 0.9393 | 0.8658 | 0.9060 |
| 2.0 | 0.5 | 0.3 | 0.9067 | 0.9417 | 0.8688 | 0.9095 |
| 2.0 | 0.5 | 0.6 | 0.9102 | 0.9447 | 0.8732 | 0.9126 |
| 2.0 | 1.0 | 0.15 | 0.9017 | 0.9389 | 0.8623 | 0.9041 |
| 2.0 | 1.0 | 0.3 | 0.9042 | 0.9404 | 0.8658 | 0.9065 |
| 2.0 | 1.0 | 0.6 | 0.9081 | 0.9430 | 0.8701 | 0.9110 |
| 4.0 | 0.25 | 0.15 | 0.9042 | 0.9400 | 0.8669 | 0.9058 |
| 4.0 | 0.25 | 0.3 | 0.9063 | 0.9402 | 0.8690 | 0.9097 |
| 4.0 | 0.25 | 0.6 | 0.9089 | 0.9432 | 0.8714 | 0.9121 |
| 4.0 | 0.5 | 0.15 | 0.9015 | 0.9367 | 0.8647 | 0.9030 |
| 4.0 | 0.5 | 0.3 | 0.9044 | 0.9400 | 0.8660 | 0.9071 |
| 4.0 | 0.5 | 0.6 | 0.9065 | 0.9402 | 0.8684 | 0.9110 |
| 4.0 | 1.0 | 0.15 | 0.8988 | 0.9350 | 0.8603 | 0.9012 |
| 4.0 | 1.0 | 0.3 | 0.9037 | 0.9393 | 0.8658 | 0.9060 |
| 4.0 | 1.0 | 0.6 | 0.9067 | 0.9417 | 0.8688 | 0.9095 |

Spread of the 45 objectives: min 0.8921, max 0.9132; points within 0.002 of the best: 7.

Interpretation (one line): the weights are written to the new configs/weights_horizon.yaml and carry no claim (A18); no other weight file was touched.

### F48 - S15 raw results: frozen C-h and D2-h (horizon objective), staggered onset, draws 300-349, seeds 1-5 (RAW - no labels, no verdict recorded here)
Date: 2026-10-15 | Commit: this entry | Pre-registration D72 = e7c7d54 (2026-10-15 18:51:15 +0530, commit log). Scripts: `scripts/s15_horizon.py trust`, `eval`, `closed`, `report` (code at the S15 commits before this entry). C-h, D2-h = F46 checkpoints; the original staggered C = checkpoints/s13 (S13), re-evaluated on the same draws; weights: A from weights_staggered.yaml, C-h and D2-h from weights_horizon.yaml (F47). Order of the script-logged UTC times (results/s15/first_number.json): tuning grid (seed 100, draws 0-2) first result 2026-10-15T13:31:56Z; **first evaluation number (open-loop trust on draws 300-349) 2026-10-15T13:33:17Z** (= 19:03:17 +0530); first closed-loop evaluation result 2026-10-15T13:34:52Z. The tuning grid therefore preceded both, as D72 section 6 orders it. No weight, window, floor or draw was changed after the first evaluation number. (This run's open-loop and closed-loop numbers were each produced once.)

Open-loop (PRIMARY and SECONDARY (b)); "C(S13)" is the original staggered C trained on the next-task objective:

```
======================================================================================================================
S15 PRIMARY: STAGGERED ONSET (S = 300), open-loop frozen C-h and D2-h (horizon objective), draws 300-349, fraction 0.20, seeds [1, 2, 3, 4, 5] (RAW: no labels)
======================================================================================================================

rho 0  (CONTROL)
  D42 accounting per seed: draws attempted / failed / aggregated (draws with a late-onset RSU) / late-onset RSUs
    seed 1: attempted 50  failed 0  aggregated 36  late-onset RSUs 46  tau C-h 0.8832 D2-h 0.8562 C(S13) 0.8461
    seed 2: attempted 50  failed 0  aggregated 31  late-onset RSUs 42  tau C-h 0.8967 D2-h 0.9445 C(S13) 0.8602
    seed 3: attempted 50  failed 0  aggregated 33  late-onset RSUs 41  tau C-h 0.9270 D2-h 0.9510 C(S13) 0.8137
    seed 4: attempted 50  failed 0  aggregated 35  late-onset RSUs 44  tau C-h 0.8530 D2-h 0.9196 C(S13) 0.8598
    seed 5: attempted 50  failed 0  aggregated 31  late-onset RSUs 38  tau C-h 0.8680 D2-h 0.8969 C(S13) 0.8682
  detection latency (steps after own onset, late-onset RSUs, censored 399): C-h 95.7 +/- 7.8   D2-h 93.6 +/- 9.3   C(S13, next-task objective) 130.6 +/- 11.3
  detected fraction: C-h 1.000  D2-h 1.000  C(S13) 1.000
  PRIMARY  D2-h - C-h latency: -2.0 (sd 4.9) 95% [-8.1, +4.1]
  per-seed D2-h - C-h latency: [5.0, 1.2, -4.7, -5.8, -5.8]
  SECONDARY (b) (descriptive, no label)  C-h - C(S13) latency: -34.9 (sd 16.8) 95% [-55.8, -14.1]   per seed [-43.0, -24.4, -60.8, -20.9, -25.6]

rho 1  (TEST)
  D42 accounting per seed: draws attempted / failed / aggregated (draws with a late-onset RSU) / late-onset RSUs
    seed 1: attempted 50  failed 0  aggregated 50  late-onset RSUs 124  tau C-h 0.8832 D2-h 0.8562 C(S13) 0.8461
    seed 2: attempted 50  failed 0  aggregated 50  late-onset RSUs 134  tau C-h 0.8967 D2-h 0.9445 C(S13) 0.8602
    seed 3: attempted 50  failed 0  aggregated 50  late-onset RSUs 106  tau C-h 0.9270 D2-h 0.9510 C(S13) 0.8137
    seed 4: attempted 50  failed 0  aggregated 50  late-onset RSUs 141  tau C-h 0.8530 D2-h 0.9196 C(S13) 0.8598
    seed 5: attempted 50  failed 0  aggregated 50  late-onset RSUs 129  tau C-h 0.8680 D2-h 0.8969 C(S13) 0.8682
  detection latency (steps after own onset, late-onset RSUs, censored 399): C-h 92.3 +/- 4.6   D2-h 88.2 +/- 9.4   C(S13, next-task objective) 129.7 +/- 12.7
  detected fraction: C-h 1.000  D2-h 1.000  C(S13) 1.000
  PRIMARY  D2-h - C-h latency: -4.1 (sd 6.3) 95% [-12.0, +3.7]
  per-seed D2-h - C-h latency: [3.3, 0.6, -5.6, -12.9, -5.9]
  SECONDARY (b) (descriptive, no label)  C-h - C(S13) latency: -37.4 (sd 13.0) 95% [-53.5, -21.2]   per seed [-38.3, -34.8, -59.0, -28.6, -26.1]

(description) (D2-h - C-h latency at rho 1.0) minus (at rho 0), per-seed paired: -2.1 (sd 2.8) 95% [-5.6, +1.4]
```

Closed loop (SECONDARY (a)), rho 1.0, fraction 0.20, draws 300-349:

```
======================================================================================================================
S15 SECONDARY (a): closed-loop post-onset success, staggered, rho 1.0, fraction 0.20, draws 300-349, seeds [1, 2, 3, 4, 5] (RAW, descriptive: no labels)
======================================================================================================================
weights (alpha, beta, gamma): A (0.0, 0.25, 0.6)  C-h (0.25, 0.25, 0.3)  D2-h (0.5, 0.25, 0.6)
D42: attempted 250  failed 0  aggregated 250
  A     success 0.8970 +/- 0.0030
  C_h   success 0.9336 +/- 0.0093
  D2_h  success 0.9356 +/- 0.0085
  D2_h - C_h: +0.0020 (sd 0.0011) 95% [+0.0007, +0.0033]
  C_h - A: +0.0365 (sd 0.0078) 95% [+0.0268, +0.0463]
  D2_h - A: +0.0385 (sd 0.0071) 95% [+0.0298, +0.0473]
  n_av / n_win / never-avoided (mean +/- sd over seeds of per-seed means), always together (D60 Q4):
    A     n_av   589.6 +/-   90.9   n_win   903.7 +/-   19.9   never-avoided 0.468 +/- 0.129
    C_h   n_av   224.8 +/-   45.3   n_win   559.9 +/-   70.4   never-avoided 0.020 +/- 0.035
    D2_h  n_av   224.0 +/-   49.3   n_win   556.9 +/-   69.6   never-avoided 0.020 +/- 0.035
  D57-normalised (n_C-h - n_D2-h)/n_A on n_av: +0.001 [-0.013, +0.015]  (positive = D2-h sends fewer to degraded RSUs; descriptive)
  D57-normalised (n_C-h - n_D2-h)/n_A on n_win: +0.003 [-0.001, +0.008]  (positive = D2-h sends fewer to degraded RSUs; descriptive)
  detection latency (closed loop, D56 rule on each run's own trust, descriptive) C_h: 248.2 steps, detected fraction 0.945
  detection latency (closed loop, D56 rule on each run's own trust, descriptive) D2_h: 246.7 steps, detected fraction 0.947
```

### F49 - S15 verdicts against D72 and interpretation (written after F48 was committed in da1b039)
Date: 2026-10-15 | Rules exactly as in D72; no rule, window, floor, H, threshold, minimum or draw changed. Output: `scripts/s15_horizon.py eval --verdict`.

| item | quantity | mean [95% t, df 4] | D72 label |
|---|---|---|---|
| **PRIMARY**, staggered, horizon objective, rho 1.0 | D2-h - C-h latency of late-onset RSUs (steps) | -4.1 [-12.0, +3.7] (C-h 92.3, D2-h 88.2) | **NULL** (interval inside +-20; not WIN: hi = +3.7 is not < 0) |
| control, rho 0 | D2-h - C-h latency (steps) | -2.0 [-8.1, +4.1] (C-h 95.7, D2-h 93.6) | NULL |
| description | (rho 1.0) minus (rho 0) of D2-h - C-h | -2.1 [-5.6, +1.4] | none |
| SECONDARY (b), rho 1.0 | C-h - C(S13) latency (steps) | -37.4 [-53.5, -21.2] (C(S13) 129.7) | none (descriptive) |
| SECONDARY (b), rho 0 | C-h - C(S13) latency (steps) | -34.9 [-55.8, -14.1] (C(S13) 130.6) | none (descriptive) |
| SECONDARY (a), closed loop | success D2-h - C-h | +0.0020 [+0.0007, +0.0033] (C-h 0.9336, D2-h 0.9356, A 0.8970) | none (descriptive) |

Rho 0 reading (D72 section 4): the rho 1.0 label is NULL, so there is no structural reading. A difference of 20 steps either way between D2-h and C-h is ruled out at both rho (the intervals lie inside +-20). The NOISY extension to seeds 6-10 was **not triggered** (the label is not NOISY); none was run. Per-seed D2-h - C-h latency at rho 1.0: +3.3, +0.6, -5.6, -12.9, -5.9 (two seeds positive, three negative); at rho 0: +5.0, +1.2, -4.7, -5.8, -5.8.
D42 counts: rho 1.0 250 attempted / 0 failed / 250 aggregated (late-onset RSUs per seed 106-141); rho 0 250 / 0 / 166 aggregated (36, 31, 33, 35, 31; draws with no late-onset RSU are not aggregated); closed loop 250 / 0 / 250; detected fraction 1.000 for C-h, D2-h and C(S13) at both rho in the open-loop latency.
Closed loop, n_av / n_win / never-avoided together (F48): A 589.6 / 903.7 / 0.468, C-h 224.8 / 559.9 / 0.020, D2-h 224.0 / 556.9 / 0.020; D57-normalised (n_C-h - n_D2-h)/n_A +0.001 [-0.013, +0.015] on n_av and +0.003 [-0.001, +0.008] on n_win (descriptive, no label). Success differences against A (descriptive): C-h - A +0.0365 [+0.0268, +0.0463], D2-h - A +0.0385 [+0.0298, +0.0473]. The D2-h - C-h success interval lies above 0 and below the 0.01 floor of the D56 H3 rule; no label is applied to it (D72 section 5).
Neighbour-block probe (F46): D2-h output depended on the neighbour block (>= 0.01) in 4 of the 5 evaluation seeds and in the tuning seed; seed 1 was 0.0014, below the threshold (reported as D72 prescribes).
Interpretation (one line): trained to predict the coming 150-step window rather than the next task, a head that sees same-segment RSU neighbours (D2-h) did not detect later-onset RSUs earlier than the node-local head trained the same way (NULL at rho 1.0 and rho 0; 20 steps either way ruled out), so the A20 hypothesis that the next-task objective alone left neighbour evidence nothing to improve is not supported in this simulation for this one horizon objective, threshold and D2 construction. Separately and descriptively, the horizon objective itself moved the node-local head's detection earlier by 35-37 steps against the original C (the two latencies use each model's own pre-onset 1st-percentile threshold, so this is a statement about these models under the D65 rule, not about a threshold-free quantity), with no graph involved. This experiment does not revise F17, F26, F37 or F45 or any H2 verdict, and no mechanism is tested here for why the objective change helps C or why D2-h did not exceed C-h.

### F50 - S16 raw results: frozen C-h (S15, horizon objective) and C (S13, next-task objective), staggered onset, draws 350-399, seeds 1-5 (RAW - no labels, no verdict recorded here)
Date: 2026-10-15 | Commit: this entry | Pre-registration D76 = 96246a8 (2026-10-15 22:24:12 +0530, commit log; branch pushed before any number). Script: `scripts/s16_confirm.py` at efbe107 (`gen`, `check`, `trust`, `eval`, `closed`, `report`). No training, no tuning: C = checkpoints/s13/s3-seed{1..5}-C.pt, C-h = checkpoints/s15/s3-seed{1..5}-C.pt; weights A and C from weights_staggered.yaml (S13), C-h from weights_horizon.yaml (S15). Harness check before any number (seed 1, rho 1.0, draw 300, an S15 draw, not a result): the S16 open-loop path reproduces the stored S15 C-h and C trust arrays exactly and the S16 closed-loop path the stored S15 C-h closed-loop result exactly. Script-logged UTC times (results/s16/first_number.json): **first evaluation number (open-loop trust on draws 350-399) 2026-10-15T16:59:34Z (= 22:29:34 +0530)**; first closed-loop evaluation result 2026-10-15T17:04:03Z. Each number below was produced once; nothing was changed after the first number.
D42 counts are in the tables (every point: 250 draws attempted, 0 failed, 250 aggregated; 1000 degraded and 4000 healthy RSUs per rho; closed loop 250 / 0 / 250 per model). Inactive share of RSU-steps in [100, 400): 0 at every seed and rho.

Open loop (CO-PRIMARIES 1 and 2):

```
======================================================================================================================
S16 CO-PRIMARIES: staggered onset (S = 300), open-loop frozen C-h (horizon objective, S15) and C (next-task, S13), draws 350-399, fraction 0.20, seeds [1, 2, 3, 4, 5] (RAW: no labels)
======================================================================================================================

rho 1  (TEST)
  D42 accounting per seed: draws attempted / failed / aggregated (draws with a degraded RSU) / degraded RSUs / healthy RSUs
    seed 1: attempted 50  failed 0  aggregated 50  degraded RSUs 200  healthy RSUs 800  tau C-h 0.8832 C 0.8461  inactive share of RSU-steps in [100, 400) 0.0000
    seed 2: attempted 50  failed 0  aggregated 50  degraded RSUs 200  healthy RSUs 800  tau C-h 0.8967 C 0.8602  inactive share of RSU-steps in [100, 400) 0.0000
    seed 3: attempted 50  failed 0  aggregated 50  degraded RSUs 200  healthy RSUs 800  tau C-h 0.9270 C 0.8137  inactive share of RSU-steps in [100, 400) 0.0000
    seed 4: attempted 50  failed 0  aggregated 50  degraded RSUs 200  healthy RSUs 800  tau C-h 0.8530 C 0.8598  inactive share of RSU-steps in [100, 400) 0.0000
    seed 5: attempted 50  failed 0  aggregated 50  degraded RSUs 200  healthy RSUs 800  tau C-h 0.8680 C 0.8682  inactive share of RSU-steps in [100, 400) 0.0000
  detection latency, ALL degraded RSUs (steps after own onset, censored 399): C-h 90.9 +/- 6.8   C 129.1 +/- 9.1
  detected fraction: C-h 1.000  C 1.000
  CO-PRIMARY 1  C-h - C latency: -38.2 (sd 12.2) 95% [-53.4, -23.1]
     per seed: [-39.7, -42.2, -55.5, -30.3, -23.5]
  false-alarm rate, healthy RSUs, any crossing in [100, 1200): C-h 0.3162 +/- 0.1176   C 0.4428 +/- 0.1777
     (descriptive split) first crossing starting in [100, 400): C-h 0.1077  C 0.1103;   only inside [400, 1200): C-h 0.2085  C 0.3325
  CO-PRIMARY 2  C-h - C false-alarm rate: -0.1265 (sd 0.2688) 95% [-0.4602, +0.2072]
     per seed: [-0.0112, 0.0475, 0.0288, -0.1013, -0.5962]

rho 0  (CONTROL)
  D42 accounting per seed: draws attempted / failed / aggregated (draws with a degraded RSU) / degraded RSUs / healthy RSUs
    seed 1: attempted 50  failed 0  aggregated 50  degraded RSUs 200  healthy RSUs 800  tau C-h 0.8832 C 0.8461  inactive share of RSU-steps in [100, 400) 0.0000
    seed 2: attempted 50  failed 0  aggregated 50  degraded RSUs 200  healthy RSUs 800  tau C-h 0.8967 C 0.8602  inactive share of RSU-steps in [100, 400) 0.0000
    seed 3: attempted 50  failed 0  aggregated 50  degraded RSUs 200  healthy RSUs 800  tau C-h 0.9270 C 0.8137  inactive share of RSU-steps in [100, 400) 0.0000
    seed 4: attempted 50  failed 0  aggregated 50  degraded RSUs 200  healthy RSUs 800  tau C-h 0.8530 C 0.8598  inactive share of RSU-steps in [100, 400) 0.0000
    seed 5: attempted 50  failed 0  aggregated 50  degraded RSUs 200  healthy RSUs 800  tau C-h 0.8680 C 0.8682  inactive share of RSU-steps in [100, 400) 0.0000
  detection latency, ALL degraded RSUs (steps after own onset, censored 399): C-h 93.1 +/- 7.1   C 128.9 +/- 9.7
  detected fraction: C-h 1.000  C 1.000
  CO-PRIMARY 1  C-h - C latency: -35.7 (sd 14.2) 95% [-53.4, -18.1]
     per seed: [-36.7, -37.7, -57.6, -26.4, -20.3]
  false-alarm rate, healthy RSUs, any crossing in [100, 1200): C-h 0.3257 +/- 0.1284   C 0.4430 +/- 0.1842
     (descriptive split) first crossing starting in [100, 400): C-h 0.1100  C 0.1110;   only inside [400, 1200): C-h 0.2158  C 0.3320
  CO-PRIMARY 2  C-h - C false-alarm rate: -0.1173 (sd 0.2814) 95% [-0.4667, +0.2322]
     per seed: [0.0037, 0.0575, 0.0587, -0.0988, -0.6075]
```

Closed loop (SECONDARY), rho 1.0:

```
======================================================================================================================
S16 SECONDARY: closed-loop post-onset success, staggered, rho 1.0, fraction 0.20, draws 350-399, seeds [1, 2, 3, 4, 5] (RAW, descriptive: no labels)
======================================================================================================================
weights (alpha, beta, gamma): A (0.0, 0.25, 0.6)  C (2.0, 0.25, 0.15)  C-h (0.25, 0.25, 0.3)  (no new tuning)
D42: attempted 250 per model  failed 0  aggregated 250
  A    success 0.8972 +/- 0.0046
  C    success 0.9346 +/- 0.0065
  C_h  success 0.9338 +/- 0.0073
  C_h - C: -0.0008 (sd 0.0012) 95% [-0.0023, +0.0007]
  C - A: +0.0374 (sd 0.0040) 95% [+0.0324, +0.0424]
  C_h - A: +0.0366 (sd 0.0049) 95% [+0.0305, +0.0427]
  n_av / n_win / never-avoided (mean +/- sd over seeds of per-seed means), always together (D60 Q4):
    A    n_av   589.5 +/-  116.2   n_win   907.1 +/-   17.2   never-avoided 0.440 +/- 0.150
    C    n_av   245.4 +/-   29.8   n_win   576.5 +/-   42.0   never-avoided 0.024 +/- 0.036
    C_h  n_av   237.2 +/-   39.9   n_win   565.0 +/-   44.8   never-avoided 0.024 +/- 0.036
  D57-normalised (n_C - n_C-h)/n_A on n_av: +0.018 [-0.023, +0.058]  (positive = C-h sends fewer to degraded RSUs; descriptive)
  D57-normalised (n_C - n_C-h)/n_A on n_win: +0.013 [+0.001, +0.025]  (positive = C-h sends fewer to degraded RSUs; descriptive)
```

### F51 - S16 verdict against D76 and interpretation (written after F50 was committed in f2eb79d)
Date: 2026-10-15 | Rules exactly as in D76; no rule, threshold, hold, floor, bound, window, draw, model or weight changed. Output: `scripts/s16_confirm.py eval --verdict` (results/s16/report-eval-verdict-5seeds.txt).

| item | quantity (C-h minus C, paired per draw) | mean [95% t, df 4] | D76 label |
|---|---|---|---|
| **CO-PRIMARY 1, rho 1.0** | detection latency, all degraded RSUs (steps) | -38.2 [-53.4, -23.1] (C-h 90.9, C 129.1) | **EARLIER** (hi < 0 and mean <= -20) |
| **CO-PRIMARY 2, rho 1.0** | false-alarm rate, healthy RSUs | -0.1265 [-0.4602, +0.2072] (C-h 0.316, C 0.443) | **NOT ACCEPTABLE** (hi = +0.207 > 0.05) |
| **OVERALL, rho 1.0** | | | **NOT CONFIRMED** |
| control, rho 0, co-primary 1 | latency (steps) | -35.7 [-53.4, -18.1] (C-h 93.1, C 128.9) | EARLIER |
| control, rho 0, co-primary 2 | false-alarm rate | -0.1173 [-0.4667, +0.2322] (C-h 0.326, C 0.443) | NOT ACCEPTABLE |
| control, rho 0, overall | | | NOT CONFIRMED |
| SECONDARY, closed loop, rho 1.0 | post-onset success | -0.0008 [-0.0023, +0.0007] (C-h 0.9338, C 0.9346, A 0.8972) | none (descriptive) |

**Verdict (D76 section 6): NOT CONFIRMED.** Co-primary 1 is EARLIER at rho 1.0, but co-primary 2 is NOT ACCEPTABLE, so OBJECTIVE HELPS is not reached; the rho 0 control has the same two labels. No extension was run (D76 section 6 allows none).
Why co-primary 2 fails: its point estimate is negative (C-h raised fewer false alarms on average), but the interval is wide and its upper end, +0.207, is above the 0.05 bound, so up to about 21 percentage points more false alarms for C-h are not ruled out at 5 seeds. Per-seed C-h - C false-alarm rate at rho 1.0: -0.011, +0.048, +0.029, -0.101, -0.596 (seed 5 carries most of the mean; two seeds positive); at rho 0: +0.004, +0.058, +0.059, -0.099, -0.608. Per-seed C-h - C latency at rho 1.0: -39.7, -42.2, -55.5, -30.3, -23.5 (all five negative); at rho 0: -36.7, -37.7, -57.6, -26.4, -20.3.
D42 counts: every open-loop point 250 attempted / 0 failed / 250 aggregated, 1000 degraded and 4000 healthy RSUs per rho (4 degraded and 16 healthy per draw); detected fraction 1.000 for both models at both rho; closed loop 250 / 0 / 250 per model.
Closed loop, n_av / n_win / never-avoided together (F50): A 589.5 / 907.1 / 0.440, C 245.4 / 576.5 / 0.024, C-h 237.2 / 565.0 / 0.024; D57-normalised (n_C - n_C-h)/n_A +0.018 [-0.023, +0.058] on n_av and +0.013 [+0.001, +0.025] on n_win (descriptive, no label). C - A +0.0374 [+0.0324, +0.0424], C-h - A +0.0366 [+0.0305, +0.0427] (descriptive).
Descriptive facts that bear on reading the rule (no label): (i) under the D65/D76 rule (per-model pre-onset 1st percentile, 10-step hold) a large share of healthy RSUs cross at some step in [100, 1200): C 0.443 and C-h 0.316 at rho 1.0, of which about 0.11 for both models first cross before step 400. This is the first time the rule's false-alarm rate has been measured; it was not measured for D, D2, D2-h or H. (ii) Within a seed, both models' trust over steps 100-399 is identical across the 50 draws and both rho (checked after F50 on seeds 1 and 5, three draws per rho), so tau is a per-seed constant (equal to the F48 values) and the same healthy RSUs account for pre-onset crossings in every draw where they are healthy; the per-draw false-alarm rates within a seed are therefore strongly dependent, and the seed, as pre-registered, is the unit of analysis.
Interpretation (one line): on fresh draws and over all degraded RSUs, the horizon-trained node-local head again detected degradation earlier than the next-task head by a margin like A30's (38 steps at rho 1.0, 36 at rho 0, EARLIER at both), but the pre-registered condition that this comes without more false alarms is not met at 5 seeds (the false-alarm difference is between -0.46 and +0.21), so the confirmatory verdict is NOT CONFIRMED; the closed-loop success of the two heads did not differ. No mechanism is tested, and this does not revise F17, F26, F37, F45, F49 or any earlier verdict.

### F52 - S17 smoke run: the fault timeline of the spreading-fault and cold-start generators (harness draw 900, one seed; no variant, no trust, no comparison)
Date: 2026-10-17 | Commit: 508e8ca (code and D81 at 474230b) | Command: `python scripts/s17_smoke.py` (defaults: seed 1, draw 900, `configs/spread_L80.yaml` and `configs/restart_f20.yaml`, generated in memory, nothing written) | Draw 900 is a harness draw (D81 section 5): these numbers are not results and no S17 comparison number exists yet.

Spreading fault, L = 80, rho 1.0, fraction 0.20 (all four degraded RSUs on segment 0). `first_nonzero` is the first step at which the sealed fault level is above zero; `first_dropped_task` is the first task silently dropped on that RSU (realised, from the sealed file).

| rsu | hop | configured onset | first_nonzero | full severity | first dropped task | lag from seed (first_nonzero) |
|---|---|---|---|---|---|---|
| 7 (seed) | 0 | 400 | 401 | 700 | 459 | +0 |
| 5 | 1 | 480 | 481 | 780 | 621 | +80 |
| 10 | 1 | 480 | 481 | 780 | 620 | +80 |
| 19 | 1 | 480 | 481 | 780 | 629 | +80 |

Checks printed by the script: configured onset = onset_step + L x hop and the fault turns on one step later, for every RSU: True; healthy RSUs never faulted: True. The three downstream RSUs are each one hop from the seed, so this draw has two onset waves (400 and 480), not three.

Cold start, f = 0.20, restart step 400, 4 RSUs restarted (all on segment 3), 2 failing (RSUs 6 and 15, onset 420, ramp 100: first nonzero 421, full severity 520). `success_ewma` is the observable feature.

| rsu | failing | success_ewma at 399 | at 400 | at 401 | in service throughout |
|---|---|---|---|---|---|
| 0 | no | 1.0000 | 1.0000 | 1.0000 | True |
| 6 | yes | 1.0000 | 1.0000 | 1.0000 | True |
| 14 | no | 0.9831 | 1.0000 | 1.0000 | True |
| 15 | yes | 0.8956 | 1.0000 | 1.0000 | True |

RSUs not restarted: success_ewma at step 400 untouched (min 0.9408, max 1.0000).
Interpretation (one line): the spreading fault reaches each RSU at 400 + L x hop (the seed at 400, downstream RSUs 80 steps later at L = 80), and a restart resets a node's observable history at step 400 while keeping it in service; the generators do what D81 registers. Nothing here says anything about C, D or any detection or success rate.



### F53 - S18 raw results: spreading fault (L 0/20/40/80) and cold start (f 0.10/0.20/0.40), draws 400-449, seeds 1-5 (RAW - no labels, no verdict recorded here)
Date: 2026-10-17 | Session: S18 | Code commit: d044171 (D83 19dd46e, D84 1f2f37f; scripts/s18_run.py) | Configs: configs/spread_L{0,20,40,80}.yaml, configs/restart_f{10,20,40}.yaml | Seeds 1-5 | Draws 400-449 (evaluation), 600-603 and 640 (secondary-arm training and validation), 900 (harness check, not a result)
Setup: D81 as registered, amended by D83 (secondary arm, graph provenance, strict false-alarm reading printed) and D84 (implementation choices). PRIMARY ARM = frozen checkpoints (C, D: checkpoints/s3-seed{n}; D2: checkpoints/s14/shared), weights unchanged (weights.yaml, weights_h.yaml, weights_d2.yaml shared). SECONDARY ARM = C_r, D_r retrained per seed on draws 600-603 of each of the 7 conditions (validation draw 640), same objective, TrainSettings and parameter counts (C 1313, D 1614), primary C and D weights, no tuning. Harness check (draw 900): regenerated scenarios, trust and a closed-loop run repeat exactly. Process note: the first run of the secondary closed-loop report failed with a KeyError (it did not load the primary C and D for the C_r - C, D_r - D lines); fixed mechanically in d044171 and re-run; no result file changed. Times (UTC): {  "evaluation_first_trust_primary": "2026-10-17T08:52:44Z",  "closed_loop_first_result_primary": "2026-10-17T08:59:42Z",  "evaluation_first_trust_secondary": "2026-10-17T09:01:31Z",  "closed_loop_first_result_secondary": "2026-10-17T10:04:37Z" }
Secondary-arm training diagnostics (validation BCE on draw 640; not evaluation numbers):
seed 1: C params 1313 best epoch 21 stopped 26 (patience) val BCE 0.6838 -> 0.1542; D params 1614 best epoch 25 stopped 33 (patience) val BCE 0.9366 -> 0.1637 (train graphs 8400, val graphs 2100)
seed 2: C params 1313 best epoch 40 stopped 43 (patience) val BCE 0.7551 -> 0.1452; D params 1614 best epoch 105 stopped 113 (patience) val BCE 3.9588 -> 0.1329 (train graphs 8400, val graphs 2100)
seed 3: C params 1313 best epoch 35 stopped 42 (patience) val BCE 0.8014 -> 0.1677; D params 1614 best epoch 69 stopped 77 (patience) val BCE 2.5935 -> 0.1779 (train graphs 8400, val graphs 2100)
seed 4: C params 1313 best epoch 30 stopped 38 (patience) val BCE 0.6670 -> 0.1666; D params 1614 best epoch 75 stopped 83 (patience) val BCE 0.3212 -> 0.1557 (train graphs 8400, val graphs 2100)
seed 5: C params 1313 best epoch 11 stopped 15 (patience) val BCE 0.6406 -> 0.1639; D params 1614 best epoch 66 stopped 74 (patience) val BCE 0.4323 -> 0.1551 (train graphs 8400, val graphs 2100)
Reading guide: no labels in these reports (raw mode). "latency" = detection latency under the shared D76 rule (D78 caveat: a comparison under one rule, not a time-to-alarm at a stated false-alarm rate; the false-alarm rate is beside every latency). AUC = trust AUC in the first 100 steps (D84 item 3). "run-level never-avoided" = D56 rate over runs; "faulty RSUs never avoided" = per-RSU reading (D84 item 4). M = success_ewma monitor.

#### F53a - PRIMARY ARM, open loop (results/s18/report-open-primary-raw.txt)
```
============================================================================================================================
S18 OPEN LOOP (PRIMARY ARM): frozen heads (C, D, D2, M = success_ewma), draws 400-449, seeds [1, 2, 3, 4, 5], shared D76 rule (RAW: no labels)
============================================================================================================================

--- condition L0 (spreading fault, L = 0) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8393 D 0.5808 D2 0.8509 M 0.8453
    seed 2: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8396 D 0.6952 D2 0.8423 M 0.8366
    seed 3: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8396 D 0.5774 D2 0.8003 M 0.8719
    seed 4: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8580 D 0.5292 D2 0.8583 M 0.8533
    seed 5: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8577 D 0.6128 D2 0.8727 M 0.8534
  realised hop histogram of the degraded RSUs (hop 0, 1, 2, ...): [250, 507, 199, 40, 4]
  per model (degraded RSUs with hop >= 1); mean +/- sd over seeds of per-seed draw means
    C    latency  136.9 +/-   9.5  detected 1.000  | false-alarm rate 0.4730 +/- 0.1698 (starting pre-onset 0.1098, only after 0.3632)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6192 +/- 0.0397
    D    latency  214.7 +/-  26.4  detected 0.968  | false-alarm rate 0.1385 +/- 0.0739 (starting pre-onset 0.0407, only after 0.0978)
         trust AUC, first 100 steps after own onset (vs healthy): 0.5738 +/- 0.0443
    D2   latency  134.9 +/-   9.7  detected 1.000  | false-alarm rate 0.4385 +/- 0.1347 (starting pre-onset 0.0780, only after 0.3605)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6203 +/- 0.0475
    M    latency  138.9 +/-  13.5  detected 1.000  | false-alarm rate 0.4390 +/- 0.0944 (starting pre-onset 0.1120, only after 0.3270)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6510 +/- 0.0414
  D - C latency: +77.8 (sd 27.4) 95% [+43.8, +111.8]   per seed [55.5, 104.5, 99.6, 86.6, 42.7]
  D - C false-alarm rate: -0.3345 (sd 0.1310) 95% [-0.4972, -0.1718]   per seed [-0.3337, -0.2963, -0.1613, -0.355, -0.5262]
  D - C trust AUC (first 100 steps): -0.0454 (sd 0.0330) 95% [-0.0864, -0.0045]  (descriptive)
  D2 - C latency: -2.1 (sd 4.9) 95% [-8.2, +4.0]   per seed [-5.5, -4.8, 6.5, -4.1, -2.4]
  D2 - C false-alarm rate: -0.0345 (sd 0.0363) 95% [-0.0796, +0.0106]   per seed [-0.0037, -0.0462, 0.0025, -0.0375, -0.0875]
  M - C latency: +1.9 (sd 12.2) 95% [-13.2, +17.1]   per seed [-3.3, 10.1, -14.6, 0.6, 17.0]
  M - C false-alarm rate: -0.0340 (sd 0.2107) 95% [-0.2955, +0.2275]   per seed [0.0638, -0.155, 0.1138, 0.1525, -0.345]

--- condition L20 (spreading fault, L = 20) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8393 D 0.5808 D2 0.8509 M 0.8453
    seed 2: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8396 D 0.6952 D2 0.8423 M 0.8366
    seed 3: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8396 D 0.5774 D2 0.8003 M 0.8719
    seed 4: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8580 D 0.5292 D2 0.8583 M 0.8533
    seed 5: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8577 D 0.6128 D2 0.8727 M 0.8534
  realised hop histogram of the degraded RSUs (hop 0, 1, 2, ...): [250, 507, 199, 40, 4]
  per model (degraded RSUs with hop >= 1); mean +/- sd over seeds of per-seed draw means
    C    latency  130.6 +/-   9.3  detected 1.000  | false-alarm rate 0.4730 +/- 0.1698 (starting pre-onset 0.1098, only after 0.3632)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6294 +/- 0.0355
    D    latency  210.0 +/-  24.2  detected 0.980  | false-alarm rate 0.1390 +/- 0.0724 (starting pre-onset 0.0407, only after 0.0983)
         trust AUC, first 100 steps after own onset (vs healthy): 0.5789 +/- 0.0373
    D2   latency  127.7 +/-  10.3  detected 1.000  | false-alarm rate 0.4385 +/- 0.1347 (starting pre-onset 0.0780, only after 0.3605)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6214 +/- 0.0395
    M    latency  135.2 +/-  20.6  detected 1.000  | false-alarm rate 0.4390 +/- 0.0944 (starting pre-onset 0.1120, only after 0.3270)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6703 +/- 0.0365
  D - C latency: +79.4 (sd 27.0) 95% [+45.9, +112.9]   per seed [52.5, 104.5, 95.1, 97.1, 47.8]
  D - C false-alarm rate: -0.3340 (sd 0.1310) 95% [-0.4966, -0.1714]   per seed [-0.33, -0.2975, -0.1613, -0.355, -0.5262]
  D - C trust AUC (first 100 steps): -0.0505 (sd 0.0223) 95% [-0.0782, -0.0228]  (descriptive)
  D2 - C latency: -2.8 (sd 7.1) 95% [-11.6, +6.0]   per seed [-10.2, 0.3, 7.9, -7.1, -5.0]
  D2 - C false-alarm rate: -0.0345 (sd 0.0363) 95% [-0.0796, +0.0106]   per seed [-0.0037, -0.0462, 0.0025, -0.0375, -0.0875]
  M - C latency: +4.6 (sd 18.4) 95% [-18.3, +27.5]   per seed [-2.4, 29.5, -15.5, -6.1, 17.5]
  M - C false-alarm rate: -0.0340 (sd 0.2107) 95% [-0.2955, +0.2275]   per seed [0.0638, -0.155, 0.1138, 0.1525, -0.345]
  change of the D - C latency gap from L = 0 to this L (same draws): +1.6 95% [-6.1, +9.3]  (descriptive)

--- condition L40 (spreading fault, L = 40) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8393 D 0.5808 D2 0.8509 M 0.8453
    seed 2: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8396 D 0.6952 D2 0.8423 M 0.8366
    seed 3: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8396 D 0.5774 D2 0.8003 M 0.8719
    seed 4: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8580 D 0.5292 D2 0.8583 M 0.8533
    seed 5: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8577 D 0.6128 D2 0.8727 M 0.8534
  realised hop histogram of the degraded RSUs (hop 0, 1, 2, ...): [250, 507, 199, 40, 4]
  per model (degraded RSUs with hop >= 1); mean +/- sd over seeds of per-seed draw means
    C    latency  132.8 +/-  10.5  detected 1.000  | false-alarm rate 0.4730 +/- 0.1698 (starting pre-onset 0.1098, only after 0.3632)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6324 +/- 0.0271
    D    latency  212.8 +/-  17.2  detected 0.969  | false-alarm rate 0.1420 +/- 0.0744 (starting pre-onset 0.0407, only after 0.1012)
         trust AUC, first 100 steps after own onset (vs healthy): 0.5985 +/- 0.0318
    D2   latency  130.7 +/-  12.7  detected 1.000  | false-alarm rate 0.4383 +/- 0.1341 (starting pre-onset 0.0780, only after 0.3603)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6306 +/- 0.0374
    M    latency  135.3 +/-  21.7  detected 1.000  | false-alarm rate 0.4390 +/- 0.0944 (starting pre-onset 0.1120, only after 0.3270)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6870 +/- 0.0261
  D - C latency: +80.0 (sd 20.0) 95% [+55.2, +104.8]   per seed [54.7, 89.3, 91.1, 101.7, 63.3]
  D - C false-alarm rate: -0.3310 (sd 0.1309) 95% [-0.4935, -0.1685]   per seed [-0.325, -0.2838, -0.165, -0.355, -0.5262]
  D - C trust AUC (first 100 steps): -0.0339 (sd 0.0209) 95% [-0.0599, -0.0080]  (descriptive)
  D2 - C latency: -2.1 (sd 4.8) 95% [-8.0, +3.8]   per seed [-4.2, 2.0, 3.3, -3.4, -8.3]
  D2 - C false-alarm rate: -0.0347 (sd 0.0368) 95% [-0.0804, +0.0109]   per seed [-0.0037, -0.0462, 0.0025, -0.0375, -0.0887]
  M - C latency: +2.5 (sd 15.5) 95% [-16.7, +21.8]   per seed [-2.1, 16.7, -14.2, -8.6, 20.8]
  M - C false-alarm rate: -0.0340 (sd 0.2107) 95% [-0.2955, +0.2275]   per seed [0.0638, -0.155, 0.1138, 0.1525, -0.345]
  change of the D - C latency gap from L = 0 to this L (same draws): +2.2 95% [-16.8, +21.2]  (descriptive)

--- condition L80 (spreading fault, L = 80) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8393 D 0.5808 D2 0.8509 M 0.8453
    seed 2: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8396 D 0.6952 D2 0.8423 M 0.8366
    seed 3: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8396 D 0.5774 D2 0.8003 M 0.8719
    seed 4: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8580 D 0.5292 D2 0.8583 M 0.8533
    seed 5: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C 0.8577 D 0.6128 D2 0.8727 M 0.8534
  realised hop histogram of the degraded RSUs (hop 0, 1, 2, ...): [250, 507, 199, 40, 4]
  per model (degraded RSUs with hop >= 1); mean +/- sd over seeds of per-seed draw means
    C    latency  130.7 +/-  14.3  detected 1.000  | false-alarm rate 0.4730 +/- 0.1698 (starting pre-onset 0.1098, only after 0.3632)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6239 +/- 0.0470
    D    latency  214.0 +/-  16.6  detected 0.977  | false-alarm rate 0.1373 +/- 0.0779 (starting pre-onset 0.0407, only after 0.0965)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6136 +/- 0.0407
    D2   latency  130.5 +/-  12.4  detected 1.000  | false-alarm rate 0.4377 +/- 0.1331 (starting pre-onset 0.0780, only after 0.3598)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6244 +/- 0.0453
    M    latency  134.2 +/-  25.5  detected 1.000  | false-alarm rate 0.4390 +/- 0.0944 (starting pre-onset 0.1120, only after 0.3270)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6948 +/- 0.0418
  D - C latency: +83.4 (sd 22.7) 95% [+55.2, +111.5]   per seed [55.5, 77.6, 93.2, 116.1, 74.5]
  D - C false-alarm rate: -0.3357 (sd 0.1254) 95% [-0.4914, -0.1801]   per seed [-0.3337, -0.2863, -0.1787, -0.3563, -0.5238]
  D - C trust AUC (first 100 steps): -0.0103 (sd 0.0283) 95% [-0.0454, +0.0248]  (descriptive)
  D2 - C latency: -0.1 (sd 3.6) 95% [-4.6, +4.3]   per seed [-2.6, 0.9, -2.3, 5.7, -2.3]
  D2 - C false-alarm rate: -0.0352 (sd 0.0377) 95% [-0.0820, +0.0115]   per seed [-0.0037, -0.0462, 0.0025, -0.0375, -0.0912]
  M - C latency: +3.6 (sd 19.0) 95% [-20.0, +27.2]   per seed [-0.1, 22.8, -18.4, -9.8, 23.5]
  M - C false-alarm rate: -0.0340 (sd 0.2107) 95% [-0.2955, +0.2275]   per seed [0.0638, -0.155, 0.1138, 0.1525, -0.345]
  change of the D - C latency gap from L = 0 to this L (same draws): +5.6 95% [-25.4, +36.6]  (descriptive)

--- condition f10 (cold start, f = 0.1) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 50 healthy 950 restarted-healthy 50 | tau C 0.8393 D 0.5808 D2 0.8509 M 0.8453
    seed 2: attempted 50 failed 0 aggregated 50 | pop 50 healthy 950 restarted-healthy 50 | tau C 0.8396 D 0.6952 D2 0.8423 M 0.8366
    seed 3: attempted 50 failed 0 aggregated 50 | pop 50 healthy 950 restarted-healthy 50 | tau C 0.8396 D 0.5774 D2 0.8003 M 0.8719
    seed 4: attempted 50 failed 0 aggregated 50 | pop 50 healthy 950 restarted-healthy 50 | tau C 0.8580 D 0.5292 D2 0.8583 M 0.8533
    seed 5: attempted 50 failed 0 aggregated 50 | pop 50 healthy 950 restarted-healthy 50 | tau C 0.8577 D 0.6128 D2 0.8727 M 0.8534
  restarted-but-healthy RSUs sharing a backhaul segment with a failing RSU (sum over draws, per seed): [50, 50, 50, 50, 50]
  per model (failing RSUs, from onset 420); mean +/- sd over seeds of per-seed draw means
    C    latency   68.6 +/-   3.7  detected 1.000  | false-alarm rate 0.4764 +/- 0.1691 (starting pre-onset 0.1088, only after 0.3676)  restarted-healthy 0.4440  not restarted 0.4782
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7823 +/- 0.0501;  failing vs restarted-healthy only: 0.8920
    D    latency  129.0 +/-  32.3  detected 0.980  | false-alarm rate 0.1124 +/- 0.0792 (starting pre-onset 0.0396, only after 0.0728)  restarted-healthy 0.1760  not restarted 0.1089
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.6171 +/- 0.0430;  failing vs restarted-healthy only: 0.6440
    D2   latency   67.9 +/-   5.4  detected 1.000  | false-alarm rate 0.4432 +/- 0.1364 (starting pre-onset 0.0779, only after 0.3653)  restarted-healthy 0.4640  not restarted 0.4420
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7865 +/- 0.0359;  failing vs restarted-healthy only: 0.8920
    M    latency   70.3 +/-   4.3  detected 1.000  | false-alarm rate 0.4413 +/- 0.0946 (starting pre-onset 0.1118, only after 0.3295)  restarted-healthy 0.4240  not restarted 0.4422
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7749 +/- 0.0532;  failing vs restarted-healthy only: 0.8780
  D - C latency: +60.5 (sd 32.6) 95% [+20.0, +100.9]   per seed [48.2, 114.5, 66.3, 38.1, 35.1]
  D - C false-alarm rate: -0.3640 (sd 0.1080) 95% [-0.4981, -0.2299]   per seed [-0.3453, -0.3137, -0.2411, -0.3884, -0.5316]
  D - C trust AUC (first 100 steps): -0.1653 (sd 0.0611) 95% [-0.2412, -0.0894]  (descriptive)
  D2 - C latency: -0.7 (sd 2.0) 95% [-3.1, +1.8]   per seed [1.6, 0.6, -0.1, -3.3, -2.1]
  D2 - C false-alarm rate: -0.0333 (sd 0.0357) 95% [-0.0775, +0.0110]   per seed [0.0, -0.0347, 0.0, -0.0463, -0.0853]
  M - C latency: +1.7 (sd 2.8) 95% [-1.8, +5.2]   per seed [0.7, 2.1, -0.3, -0.4, 6.4]
  M - C false-alarm rate: -0.0352 (sd 0.2066) 95% [-0.2916, +0.2213]   per seed [0.0558, -0.1421, 0.1105, 0.1463, -0.3463]

--- condition f20 (cold start, f = 0.2) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 100 healthy 900 restarted-healthy 100 | tau C 0.8393 D 0.5808 D2 0.8509 M 0.8453
    seed 2: attempted 50 failed 0 aggregated 50 | pop 100 healthy 900 restarted-healthy 100 | tau C 0.8396 D 0.6952 D2 0.8423 M 0.8366
    seed 3: attempted 50 failed 0 aggregated 50 | pop 100 healthy 900 restarted-healthy 100 | tau C 0.8396 D 0.5774 D2 0.8003 M 0.8719
    seed 4: attempted 50 failed 0 aggregated 50 | pop 100 healthy 900 restarted-healthy 100 | tau C 0.8580 D 0.5292 D2 0.8583 M 0.8533
    seed 5: attempted 50 failed 0 aggregated 50 | pop 100 healthy 900 restarted-healthy 100 | tau C 0.8577 D 0.6128 D2 0.8727 M 0.8534
  restarted-but-healthy RSUs sharing a backhaul segment with a failing RSU (sum over draws, per seed): [96, 100, 91, 100, 100]
  per model (failing RSUs, from onset 420); mean +/- sd over seeds of per-seed draw means
    C    latency   69.5 +/-   3.1  detected 1.000  | false-alarm rate 0.4700 +/- 0.1689 (starting pre-onset 0.1091, only after 0.3609)  restarted-healthy 0.3860  not restarted 0.4805
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7799 +/- 0.0459;  failing vs restarted-healthy only: 0.8930
    D    latency  128.0 +/-  27.1  detected 0.976  | false-alarm rate 0.1311 +/- 0.0758 (starting pre-onset 0.0400, only after 0.0911)  restarted-healthy 0.2600  not restarted 0.1150
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.6443 +/- 0.0490;  failing vs restarted-healthy only: 0.7010
    D2   latency   68.7 +/-   3.4  detected 1.000  | false-alarm rate 0.4449 +/- 0.1397 (starting pre-onset 0.0791, only after 0.3658)  restarted-healthy 0.4540  not restarted 0.4437
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7809 +/- 0.0347;  failing vs restarted-healthy only: 0.8900
    M    latency   70.1 +/-   2.0  detected 1.000  | false-alarm rate 0.4400 +/- 0.0931 (starting pre-onset 0.1124, only after 0.3276)  restarted-healthy 0.4180  not restarted 0.4427
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7703 +/- 0.0376;  failing vs restarted-healthy only: 0.8585
  D - C latency: +58.5 (sd 29.2) 95% [+22.2, +94.8]   per seed [42.8, 107.2, 60.3, 51.0, 31.3]
  D - C false-alarm rate: -0.3389 (sd 0.1124) 95% [-0.4785, -0.1993]   per seed [-0.3333, -0.2956, -0.2056, -0.3456, -0.5144]
  D - C trust AUC (first 100 steps): -0.1356 (sd 0.0680) 95% [-0.2200, -0.0511]  (descriptive)
  D2 - C latency: -0.8 (sd 1.9) 95% [-3.1, +1.6]   per seed [0.5, 0.9, 0.2, -3.5, -1.9]
  D2 - C false-alarm rate: -0.0251 (sd 0.0297) 95% [-0.0620, +0.0118]   per seed [-0.01, -0.0333, 0.0011, -0.01, -0.0733]
  M - C latency: +0.7 (sd 2.5) 95% [-2.4, +3.7]   per seed [0.3, 3.4, -2.3, -1.1, 3.0]
  M - C false-alarm rate: -0.0300 (sd 0.2130) 95% [-0.2945, +0.2345]   per seed [0.0544, -0.1422, 0.1189, 0.1667, -0.3478]

--- condition f40 (cold start, f = 0.4) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 200 healthy 800 restarted-healthy 200 | tau C 0.8393 D 0.5808 D2 0.8509 M 0.8453
    seed 2: attempted 50 failed 0 aggregated 50 | pop 200 healthy 800 restarted-healthy 200 | tau C 0.8396 D 0.6952 D2 0.8423 M 0.8366
    seed 3: attempted 50 failed 0 aggregated 50 | pop 200 healthy 800 restarted-healthy 200 | tau C 0.8396 D 0.5774 D2 0.8003 M 0.8719
    seed 4: attempted 50 failed 0 aggregated 50 | pop 200 healthy 800 restarted-healthy 200 | tau C 0.8580 D 0.5292 D2 0.8583 M 0.8533
    seed 5: attempted 50 failed 0 aggregated 50 | pop 200 healthy 800 restarted-healthy 200 | tau C 0.8577 D 0.6128 D2 0.8727 M 0.8534
  restarted-but-healthy RSUs sharing a backhaul segment with a failing RSU (sum over draws, per seed): [184, 198, 183, 189, 190]
  per model (failing RSUs, from onset 420); mean +/- sd over seeds of per-seed draw means
    C    latency   71.9 +/-   4.5  detected 1.000  | false-alarm rate 0.4693 +/- 0.1682 (starting pre-onset 0.1087, only after 0.3605)  restarted-healthy 0.4250  not restarted 0.4840
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7728 +/- 0.0776;  failing vs restarted-healthy only: 0.8340
    D    latency  122.6 +/-  16.2  detected 0.989  | false-alarm rate 0.1885 +/- 0.0876 (starting pre-onset 0.0415, only after 0.1470)  restarted-healthy 0.3400  not restarted 0.1380
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.6326 +/- 0.0632;  failing vs restarted-healthy only: 0.6545
    D2   latency   71.0 +/-   5.4  detected 1.000  | false-alarm rate 0.4545 +/- 0.1476 (starting pre-onset 0.0773, only after 0.3772)  restarted-healthy 0.4740  not restarted 0.4480
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7733 +/- 0.0650;  failing vs restarted-healthy only: 0.8313
    M    latency   72.5 +/-   4.4  detected 1.000  | false-alarm rate 0.4410 +/- 0.1002 (starting pre-onset 0.1097, only after 0.3312)  restarted-healthy 0.4390  not restarted 0.4417
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7754 +/- 0.0570;  failing vs restarted-healthy only: 0.8530
  D - C latency: +50.7 (sd 19.1) 95% [+26.9, +74.5]   per seed [45.3, 79.2, 55.9, 46.5, 26.6]
  D - C false-alarm rate: -0.2807 (sd 0.1386) 95% [-0.4528, -0.1087]   per seed [-0.3312, -0.2213, -0.1087, -0.26, -0.4825]
  D - C trust AUC (first 100 steps): -0.1403 (sd 0.0558) 95% [-0.2095, -0.0710]  (descriptive)
  D2 - C latency: -0.9 (sd 2.3) 95% [-3.7, +2.0]   per seed [1.4, 0.8, 0.0, -4.0, -2.6]
  D2 - C false-alarm rate: -0.0147 (sd 0.0287) 95% [-0.0504, +0.0209]   per seed [-0.0225, -0.0238, 0.0013, 0.0238, -0.0525]
  M - C latency: +0.6 (sd 3.2) 95% [-3.4, +4.6]   per seed [-0.4, 3.5, -4.3, 1.0, 3.3]
  M - C false-alarm rate: -0.0282 (sd 0.2200) 95% [-0.3013, +0.2448]   per seed [0.0675, -0.1487, 0.1025, 0.19, -0.3525]
```

#### F53b - PRIMARY ARM, closed loop (results/s18/report-closed-primary-raw.txt)
```
============================================================================================================================
S18 CLOSED LOOP (PRIMARY ARM): dwell filter and handoff on, draws 400-449, seeds [1, 2, 3, 4, 5] (RAW: no labels)
============================================================================================================================
weights (alpha, beta, gamma): A (0.0, 0.25, 0.6)  M (0.5, 0.25, 0.3)  C (1.0, 0.25, 0.6)  D (2.0, 0.25, 0.6)  D2 (2.0, 0.25, 0.6)  B_oracle (0.5, 0.25, 0.6)  (existing tuned files; no new tuning)

--- condition L0 (spreading fault, L = 0) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  A        success [400,1200) 0.8724 +/- 0.0019  | n_av   579.0 +/-  106.9 n_win   899.2 +/-   21.4 run-level never-avoided 0.460 +/- 0.131 | faulty RSUs never avoided 0.066 (66/1000)
  M        success [400,1200) 0.9193 +/- 0.0108  | n_av   138.4 +/-   48.9 n_win   491.5 +/-   78.9 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.003 (3/1000)
  C        success [400,1200) 0.9231 +/- 0.0110  | n_av   139.2 +/-   46.2 n_win   488.6 +/-   82.1 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.015 (15/1000)
  D        success [400,1200) 0.9062 +/- 0.0104  | n_av   187.7 +/-   67.5 n_win   596.0 +/-   99.7 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.021 (21/1000)
  D2       success [400,1200) 0.9235 +/- 0.0108  | n_av   122.3 +/-   40.6 n_win   469.3 +/-   80.5 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.012 (12/1000)
  B_oracle success [400,1200) 0.9255 +/- 0.0108  | n_av    22.6 +/-   28.2 n_win   377.6 +/-   96.3 run-level never-avoided 0.012 +/- 0.027 | faulty RSUs never avoided 0.010 (10/1000)
  D - C: -0.0168 (sd 0.0097) 95% [-0.0288, -0.0048]   per seed [-0.0054, -0.0277, -0.026, -0.0135, -0.0114]
  D2 - C: +0.0004 (sd 0.0003) 95% [+0.0000, +0.0008]   per seed [0.0002, 0.0002, 0.0004, 0.0004, 0.0009]
  M - C: -0.0038 (sd 0.0006) 95% [-0.0045, -0.0030]   per seed [-0.0036, -0.0047, -0.0036, -0.0037, -0.0032]
  C - A: +0.0506 (sd 0.0093) 95% [+0.0391, +0.0622]   per seed [0.0533, 0.0536, 0.0626, 0.0379, 0.0456]
  D - A: +0.0338 (sd 0.0095) 95% [+0.0221, +0.0456]   per seed [0.048, 0.0259, 0.0365, 0.0245, 0.0343]
  D57-normalised (n_C - n_D)/n_A on n_av: -0.087 [-0.191, +0.016]  (positive = D sends fewer to the faulty RSUs; descriptive)
  D57-normalised (n_C - n_D)/n_A on n_win: -0.119 [-0.225, -0.012]  (positive = D sends fewer to the faulty RSUs; descriptive)

--- condition L20 (spreading fault, L = 20) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  A        success [400,1200) 0.8760 +/- 0.0024  | n_av   579.0 +/-  106.9 n_win   899.2 +/-   21.4 run-level never-avoided 0.460 +/- 0.131 | faulty RSUs never avoided 0.066 (66/1000)
  M        success [400,1200) 0.9206 +/- 0.0105  | n_av   156.4 +/-   47.2 n_win   506.2 +/-   73.9 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.006 (6/1000)
  C        success [400,1200) 0.9245 +/- 0.0106  | n_av   157.6 +/-   45.4 n_win   504.2 +/-   79.2 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.015 (15/1000)
  D        success [400,1200) 0.9081 +/- 0.0100  | n_av   203.6 +/-   62.7 n_win   606.2 +/-   95.5 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.025 (25/1000)
  D2       success [400,1200) 0.9249 +/- 0.0105  | n_av   142.4 +/-   44.4 n_win   484.4 +/-   77.6 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.021 (21/1000)
  B_oracle success [400,1200) 0.9268 +/- 0.0105  | n_av    27.1 +/-   29.2 n_win   392.9 +/-   92.1 run-level never-avoided 0.012 +/- 0.027 | faulty RSUs never avoided 0.010 (10/1000)
  D - C: -0.0164 (sd 0.0088) 95% [-0.0274, -0.0054]   per seed [-0.0057, -0.0265, -0.0245, -0.0135, -0.0118]
  D2 - C: +0.0004 (sd 0.0002) 95% [+0.0001, +0.0007]   per seed [0.0002, 0.0001, 0.0005, 0.0003, 0.0007]
  M - C: -0.0039 (sd 0.0006) 95% [-0.0047, -0.0031]   per seed [-0.0038, -0.005, -0.0035, -0.0037, -0.0035]
  C - A: +0.0485 (sd 0.0085) 95% [+0.0379, +0.0591]   per seed [0.0506, 0.0516, 0.0593, 0.0366, 0.0443]
  D - A: +0.0321 (sd 0.0087) 95% [+0.0213, +0.0429]   per seed [0.0449, 0.0251, 0.0348, 0.0231, 0.0326]
  D57-normalised (n_C - n_D)/n_A on n_av: -0.082 [-0.165, +0.001]  (positive = D sends fewer to the faulty RSUs; descriptive)
  D57-normalised (n_C - n_D)/n_A on n_win: -0.113 [-0.213, -0.013]  (positive = D sends fewer to the faulty RSUs; descriptive)

--- condition L40 (spreading fault, L = 40) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  A        success [400,1200) 0.8797 +/- 0.0030  | n_av   579.0 +/-  106.9 n_win   899.2 +/-   21.4 run-level never-avoided 0.460 +/- 0.131 | faulty RSUs never avoided 0.066 (66/1000)
  M        success [400,1200) 0.9222 +/- 0.0103  | n_av   175.8 +/-   42.9 n_win   521.2 +/-   69.9 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.005 (5/1000)
  C        success [400,1200) 0.9261 +/- 0.0104  | n_av   180.4 +/-   41.0 n_win   521.6 +/-   72.8 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.014 (14/1000)
  D        success [400,1200) 0.9103 +/- 0.0098  | n_av   219.3 +/-   60.6 n_win   618.0 +/-   90.1 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.026 (26/1000)
  D2       success [400,1200) 0.9264 +/- 0.0104  | n_av   165.1 +/-   37.0 n_win   500.7 +/-   71.2 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.023 (23/1000)
  B_oracle success [400,1200) 0.9282 +/- 0.0103  | n_av    40.5 +/-   29.5 n_win   408.1 +/-   88.2 run-level never-avoided 0.012 +/- 0.027 | faulty RSUs never avoided 0.010 (10/1000)
  D - C: -0.0158 (sd 0.0082) 95% [-0.0260, -0.0056]   per seed [-0.0057, -0.0252, -0.0232, -0.0133, -0.0116]
  D2 - C: +0.0003 (sd 0.0003) 95% [-0.0000, +0.0006]   per seed [0.0002, 0.0002, 0.0003, 0.0001, 0.0007]
  M - C: -0.0038 (sd 0.0006) 95% [-0.0046, -0.0031]   per seed [-0.0038, -0.0048, -0.0035, -0.0038, -0.0033]
  C - A: +0.0464 (sd 0.0078) 95% [+0.0368, +0.0560]   per seed [0.048, 0.0496, 0.0561, 0.0354, 0.043]
  D - A: +0.0306 (sd 0.0080) 95% [+0.0207, +0.0405]   per seed [0.0423, 0.0244, 0.0329, 0.022, 0.0313]
  D57-normalised (n_C - n_D)/n_A on n_av: -0.069 [-0.148, +0.009]  (positive = D sends fewer to the faulty RSUs; descriptive)
  D57-normalised (n_C - n_D)/n_A on n_win: -0.107 [-0.200, -0.013]  (positive = D sends fewer to the faulty RSUs; descriptive)

--- condition L80 (spreading fault, L = 80) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  A        success [400,1200) 0.8865 +/- 0.0038  | n_av   579.0 +/-  106.9 n_win   899.2 +/-   21.4 run-level never-avoided 0.460 +/- 0.131 | faulty RSUs never avoided 0.073 (73/1000)
  M        success [400,1200) 0.9253 +/- 0.0100  | n_av   207.2 +/-   38.2 n_win   554.3 +/-   60.7 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.008 (8/1000)
  C        success [400,1200) 0.9292 +/- 0.0100  | n_av   212.9 +/-   29.5 n_win   553.0 +/-   63.0 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.013 (13/1000)
  D        success [400,1200) 0.9145 +/- 0.0095  | n_av   244.7 +/-   56.4 n_win   640.4 +/-   81.4 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.018 (18/1000)
  D2       success [400,1200) 0.9295 +/- 0.0100  | n_av   201.1 +/-   30.5 n_win   533.7 +/-   60.2 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.022 (22/1000)
  B_oracle success [400,1200) 0.9310 +/- 0.0100  | n_av    84.4 +/-   35.6 n_win   441.5 +/-   78.9 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.006 (6/1000)
  D - C: -0.0147 (sd 0.0072) 95% [-0.0237, -0.0058]   per seed [-0.0056, -0.0231, -0.0209, -0.0129, -0.0112]
  D2 - C: +0.0003 (sd 0.0001) 95% [+0.0002, +0.0005]   per seed [0.0003, 0.0002, 0.0003, 0.0004, 0.0005]
  M - C: -0.0039 (sd 0.0006) 95% [-0.0047, -0.0032]   per seed [-0.0042, -0.0049, -0.0033, -0.0037, -0.0036]
  C - A: +0.0427 (sd 0.0066) 95% [+0.0344, +0.0509]   per seed [0.0434, 0.0463, 0.0505, 0.033, 0.04]
  D - A: +0.0279 (sd 0.0068) 95% [+0.0195, +0.0364]   per seed [0.0378, 0.0232, 0.0296, 0.0201, 0.0289]
  D57-normalised (n_C - n_D)/n_A on n_av: -0.058 [-0.139, +0.023]  (positive = D sends fewer to the faulty RSUs; descriptive)
  D57-normalised (n_C - n_D)/n_A on n_win: -0.097 [-0.182, -0.011]  (positive = D sends fewer to the faulty RSUs; descriptive)

--- condition f10 (cold start, f = 0.1) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  A        success [400,500) 0.9702 +/- 0.0090   after the window [500,1200) 0.9385 +/- 0.0019  | n_av    63.7 +/-   18.0 n_win   228.4 +/-   10.4 run-level never-avoided 0.052 +/- 0.046 | faulty RSUs never avoided 0.052 (13/250)
  M        success [400,500) 0.9690 +/- 0.0096   after the window [500,1200) 0.9647 +/- 0.0019  | n_av    12.5 +/-    1.3 n_win    41.8 +/-    2.9 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/250)
  C        success [400,500) 0.9725 +/- 0.0092   after the window [500,1200) 0.9682 +/- 0.0023  | n_av    11.7 +/-    2.1 n_win    38.1 +/-    3.8 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/250)
  D        success [400,500) 0.9674 +/- 0.0104   after the window [500,1200) 0.9629 +/- 0.0010  | n_av    16.1 +/-    2.2 n_win    58.0 +/-   15.5 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/250)
  D2       success [400,500) 0.9724 +/- 0.0097   after the window [500,1200) 0.9680 +/- 0.0024  | n_av    10.6 +/-    1.7 n_win    34.4 +/-    3.5 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/250)
  B_oracle success [400,500) 0.9750 +/- 0.0085   after the window [500,1200) 0.9693 +/- 0.0020  | n_av     0.2 +/-    0.2 n_win    13.5 +/-    3.8 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/250)
  D - C: -0.0051 (sd 0.0041) 95% [-0.0102, +0.0000]   per seed [-0.0121, -0.0016, -0.0033, -0.0042, -0.0042]
  D - C after the window [500,1200): -0.0053 95% [-0.0077, -0.0029]  (descriptive)
  D2 - C: -0.0001 (sd 0.0005) 95% [-0.0007, +0.0006]   per seed [0.0001, 0.0001, 0.0003, -0.001, 0.0001]
  M - C: -0.0035 (sd 0.0018) 95% [-0.0056, -0.0013]   per seed [-0.0055, -0.002, -0.0023, -0.0022, -0.0053]
  C - A: +0.0022 (sd 0.0005) 95% [+0.0016, +0.0029]   per seed [0.002, 0.0024, 0.0023, 0.0015, 0.0029]
  D - A: -0.0028 (sd 0.0043) 95% [-0.0082, +0.0025]   per seed [-0.0102, 0.0008, -0.0009, -0.0027, -0.0012]
  D57-normalised (n_C - n_D)/n_A on n_av: -0.076 [-0.142, -0.010]  (positive = D sends fewer to the faulty RSUs; descriptive)
  D57-normalised (n_C - n_D)/n_A on n_win: -0.088 [-0.183, +0.007]  (positive = D sends fewer to the faulty RSUs; descriptive)

--- condition f20 (cold start, f = 0.2) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  A        success [400,500) 0.9645 +/- 0.0090   after the window [500,1200) 0.9044 +/- 0.0031  | n_av   209.0 +/-   15.0 n_win   456.0 +/-   10.4 run-level never-avoided 0.244 +/- 0.093 | faulty RSUs never avoided 0.064 (32/500)
  M        success [400,500) 0.9645 +/- 0.0098   after the window [500,1200) 0.9521 +/- 0.0045  | n_av    25.9 +/-    2.4 n_win   131.7 +/-   16.9 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/500)
  C        success [400,500) 0.9680 +/- 0.0093   after the window [500,1200) 0.9556 +/- 0.0049  | n_av    26.5 +/-    5.9 n_win   128.0 +/-   22.2 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/500)
  D        success [400,500) 0.9622 +/- 0.0102   after the window [500,1200) 0.9439 +/- 0.0027  | n_av    36.9 +/-    8.1 n_win   185.5 +/-   34.0 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/500)
  D2       success [400,500) 0.9682 +/- 0.0097   after the window [500,1200) 0.9558 +/- 0.0049  | n_av    23.7 +/-    4.6 n_win   120.3 +/-   20.7 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/500)
  B_oracle success [400,500) 0.9717 +/- 0.0087   after the window [500,1200) 0.9573 +/- 0.0046  | n_av     2.4 +/-    1.6 n_win    86.5 +/-   20.6 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/500)
  D - C: -0.0057 (sd 0.0039) 95% [-0.0105, -0.0009]   per seed [-0.0125, -0.0025, -0.0043, -0.0044, -0.005]
  D - C after the window [500,1200): -0.0117 95% [-0.0203, -0.0031]  (descriptive)
  D2 - C: +0.0002 (sd 0.0005) 95% [-0.0004, +0.0009]   per seed [0.0004, 0.0003, 0.0007, -0.0006, 0.0004]
  M - C: -0.0034 (sd 0.0016) 95% [-0.0054, -0.0015]   per seed [-0.0055, -0.0019, -0.0026, -0.0025, -0.0047]
  C - A: +0.0035 (sd 0.0006) 95% [+0.0028, +0.0042]   per seed [0.0028, 0.0038, 0.0041, 0.0029, 0.0037]
  D - A: -0.0023 (sd 0.0043) 95% [-0.0076, +0.0030]   per seed [-0.0096, 0.0013, -0.0002, -0.0015, -0.0013]
  D57-normalised (n_C - n_D)/n_A on n_av: -0.050 [-0.098, -0.003]  (positive = D sends fewer to the faulty RSUs; descriptive)
  D57-normalised (n_C - n_D)/n_A on n_win: -0.126 [-0.259, +0.006]  (positive = D sends fewer to the faulty RSUs; descriptive)

--- condition f40 (cold start, f = 0.4) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  A        success [400,500) 0.9536 +/- 0.0088   after the window [500,1200) 0.8395 +/- 0.0018  | n_av   619.9 +/-   50.0 n_win   899.3 +/-   14.6 run-level never-avoided 0.508 +/- 0.074 | faulty RSUs never avoided 0.063 (63/1000)
  M        success [400,500) 0.9554 +/- 0.0093   after the window [500,1200) 0.9285 +/- 0.0060  | n_av    51.0 +/-    4.2 n_win   306.9 +/-   35.3 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.001 (1/1000)
  C        success [400,500) 0.9592 +/- 0.0089   after the window [500,1200) 0.9323 +/- 0.0063  | n_av    52.8 +/-    7.6 n_win   302.6 +/-   37.5 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.003 (3/1000)
  D        success [400,500) 0.9520 +/- 0.0102   after the window [500,1200) 0.9103 +/- 0.0092  | n_av    78.1 +/-   13.9 n_win   422.8 +/-   69.4 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.006 (6/1000)
  D2       success [400,500) 0.9599 +/- 0.0094   after the window [500,1200) 0.9329 +/- 0.0061  | n_av    45.4 +/-    5.4 n_win   288.1 +/-   35.3 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.004 (4/1000)
  B_oracle success [400,500) 0.9658 +/- 0.0090   after the window [500,1200) 0.9350 +/- 0.0061  | n_av     3.7 +/-    4.1 n_win   226.4 +/-   37.0 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.001 (1/1000)
  D - C: -0.0072 (sd 0.0035) 95% [-0.0116, -0.0028]   per seed [-0.0132, -0.0042, -0.0059, -0.0071, -0.0056]
  D - C after the window [500,1200): -0.0220 95% [-0.0389, -0.0050]  (descriptive)
  D2 - C: +0.0007 (sd 0.0004) 95% [+0.0002, +0.0012]   per seed [0.0009, 0.0012, 0.0008, 0.0, 0.0005]
  M - C: -0.0038 (sd 0.0008) 95% [-0.0048, -0.0027]   per seed [-0.0047, -0.0032, -0.0029, -0.0034, -0.0047]
  C - A: +0.0056 (sd 0.0009) 95% [+0.0045, +0.0068]   per seed [0.0044, 0.0065, 0.006, 0.0063, 0.0049]
  D - A: -0.0016 (sd 0.0042) 95% [-0.0068, +0.0037]   per seed [-0.0088, 0.0024, 0.0001, -0.0008, -0.0007]
  D57-normalised (n_C - n_D)/n_A on n_av: -0.042 [-0.083, -0.001]  (positive = D sends fewer to the faulty RSUs; descriptive)
  D57-normalised (n_C - n_D)/n_A on n_win: -0.134 [-0.256, -0.013]  (positive = D sends fewer to the faulty RSUs; descriptive)
```

#### F53c - SECONDARY ARM (C_r, D_r; descriptive, cannot change a verdict), open loop (results/s18/report-open-secondary-raw.txt)
```
============================================================================================================================
S18 OPEN LOOP (SECONDARY ARM): frozen heads (C_r, D_r retrained on draws 600-640, D83), draws 400-449, seeds [1, 2, 3, 4, 5], shared D76 rule (RAW: no labels)
============================================================================================================================

--- condition L0 (spreading fault, L = 0) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8309 D_r 0.8409
    seed 2: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8364 D_r 0.4930
    seed 3: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8132 D_r 0.6783
    seed 4: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8494 D_r 0.6406
    seed 5: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8745 D_r 0.7173
  realised hop histogram of the degraded RSUs (hop 0, 1, 2, ...): [250, 507, 199, 40, 4]
  per model (degraded RSUs with hop >= 1); mean +/- sd over seeds of per-seed draw means
    C_r  latency  135.7 +/-  10.7  detected 1.000  | false-alarm rate 0.4428 +/- 0.2096 (starting pre-onset 0.1190, only after 0.3237)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6332 +/- 0.0500
    D_r  latency  210.7 +/-  32.5  detected 0.975  | false-alarm rate 0.2605 +/- 0.1740 (starting pre-onset 0.0618, only after 0.1988)
         trust AUC, first 100 steps after own onset (vs healthy): 0.5717 +/- 0.0539
  D_r - C_r latency: +75.1 (sd 36.7) 95% [+29.6, +120.6]   per seed [29.1, 124.2, 95.3, 72.3, 54.5]
  D_r - C_r false-alarm rate: -0.1822 (sd 0.3005) 95% [-0.5554, +0.1909]   per seed [0.1737, -0.3025, 0.035, -0.2213, -0.5962]
  D_r - C_r trust AUC (first 100 steps): -0.0616 (sd 0.0290) 95% [-0.0976, -0.0255]  (descriptive)

--- condition L20 (spreading fault, L = 20) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8309 D_r 0.8409
    seed 2: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8364 D_r 0.4930
    seed 3: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8132 D_r 0.6783
    seed 4: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8494 D_r 0.6406
    seed 5: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8745 D_r 0.7173
  realised hop histogram of the degraded RSUs (hop 0, 1, 2, ...): [250, 507, 199, 40, 4]
  per model (degraded RSUs with hop >= 1); mean +/- sd over seeds of per-seed draw means
    C_r  latency  131.2 +/-  11.3  detected 1.000  | false-alarm rate 0.4428 +/- 0.2096 (starting pre-onset 0.1190, only after 0.3237)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6381 +/- 0.0356
    D_r  latency  210.7 +/-  36.0  detected 0.976  | false-alarm rate 0.2590 +/- 0.1782 (starting pre-onset 0.0618, only after 0.1973)
         trust AUC, first 100 steps after own onset (vs healthy): 0.5767 +/- 0.0408
  D_r - C_r latency: +79.5 (sd 35.3) 95% [+35.7, +123.3]   per seed [29.5, 115.5, 107.0, 85.4, 59.9]
  D_r - C_r false-alarm rate: -0.1837 (sd 0.3024) 95% [-0.5591, +0.1916]   per seed [0.1762, -0.3, 0.0387, -0.2375, -0.5962]
  D_r - C_r trust AUC (first 100 steps): -0.0614 (sd 0.0384) 95% [-0.1091, -0.0138]  (descriptive)
  change of the D_r - C_r latency gap from L = 0 to this L (same draws): +4.4 95% [-6.7, +15.4]  (descriptive)

--- condition L40 (spreading fault, L = 40) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8309 D_r 0.8409
    seed 2: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8364 D_r 0.4930
    seed 3: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8132 D_r 0.6783
    seed 4: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8494 D_r 0.6406
    seed 5: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8745 D_r 0.7173
  realised hop histogram of the degraded RSUs (hop 0, 1, 2, ...): [250, 507, 199, 40, 4]
  per model (degraded RSUs with hop >= 1); mean +/- sd over seeds of per-seed draw means
    C_r  latency  132.0 +/-  11.3  detected 1.000  | false-alarm rate 0.4428 +/- 0.2096 (starting pre-onset 0.1190, only after 0.3237)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6407 +/- 0.0295
    D_r  latency  208.4 +/-  41.5  detected 0.972  | false-alarm rate 0.2602 +/- 0.1834 (starting pre-onset 0.0618, only after 0.1985)
         trust AUC, first 100 steps after own onset (vs healthy): 0.5827 +/- 0.0463
  D_r - C_r latency: +76.5 (sd 37.1) 95% [+30.5, +122.5]   per seed [22.6, 109.4, 106.8, 88.1, 55.4]
  D_r - C_r false-alarm rate: -0.1825 (sd 0.3057) 95% [-0.5620, +0.1970]   per seed [0.185, -0.3013, 0.0425, -0.2437, -0.595]
  D_r - C_r trust AUC (first 100 steps): -0.0580 (sd 0.0357) 95% [-0.1023, -0.0137]  (descriptive)
  change of the D_r - C_r latency gap from L = 0 to this L (same draws): +1.4 95% [-14.3, +17.0]  (descriptive)

--- condition L80 (spreading fault, L = 80) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8309 D_r 0.8409
    seed 2: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8364 D_r 0.4930
    seed 3: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8132 D_r 0.6783
    seed 4: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8494 D_r 0.6406
    seed 5: attempted 50 failed 0 aggregated 50 | pop 150 healthy 800 restarted-healthy 0 | tau C_r 0.8745 D_r 0.7173
  realised hop histogram of the degraded RSUs (hop 0, 1, 2, ...): [250, 507, 199, 40, 4]
  per model (degraded RSUs with hop >= 1); mean +/- sd over seeds of per-seed draw means
    C_r  latency  132.0 +/-  12.4  detected 1.000  | false-alarm rate 0.4428 +/- 0.2096 (starting pre-onset 0.1190, only after 0.3237)
         trust AUC, first 100 steps after own onset (vs healthy): 0.6366 +/- 0.0528
    D_r  latency  204.1 +/-  45.8  detected 0.965  | false-alarm rate 0.2638 +/- 0.1884 (starting pre-onset 0.0618, only after 0.2020)
         trust AUC, first 100 steps after own onset (vs healthy): 0.5963 +/- 0.0399
  D_r - C_r latency: +72.1 (sd 38.2) 95% [+24.6, +119.5]   per seed [15.8, 97.5, 114.3, 75.4, 57.3]
  D_r - C_r false-alarm rate: -0.1790 (sd 0.3064) 95% [-0.5593, +0.2013]   per seed [0.1988, -0.3025, 0.0387, -0.2437, -0.5863]
  D_r - C_r trust AUC (first 100 steps): -0.0403 (sd 0.0375) 95% [-0.0868, +0.0063]  (descriptive)
  change of the D_r - C_r latency gap from L = 0 to this L (same draws): -3.0 95% [-24.7, +18.7]  (descriptive)

--- condition f10 (cold start, f = 0.1) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 50 healthy 950 restarted-healthy 50 | tau C_r 0.8309 D_r 0.8409
    seed 2: attempted 50 failed 0 aggregated 50 | pop 50 healthy 950 restarted-healthy 50 | tau C_r 0.8364 D_r 0.4930
    seed 3: attempted 50 failed 0 aggregated 50 | pop 50 healthy 950 restarted-healthy 50 | tau C_r 0.8132 D_r 0.6783
    seed 4: attempted 50 failed 0 aggregated 50 | pop 50 healthy 950 restarted-healthy 50 | tau C_r 0.8494 D_r 0.6406
    seed 5: attempted 50 failed 0 aggregated 50 | pop 50 healthy 950 restarted-healthy 50 | tau C_r 0.8745 D_r 0.7173
  restarted-but-healthy RSUs sharing a backhaul segment with a failing RSU (sum over draws, per seed): [50, 50, 50, 50, 50]
  per model (failing RSUs, from onset 420); mean +/- sd over seeds of per-seed draw means
    C_r  latency   68.9 +/-   5.8  detected 1.000  | false-alarm rate 0.4455 +/- 0.2101 (starting pre-onset 0.1192, only after 0.3263)  restarted-healthy 0.4000  not restarted 0.4480
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7935 +/- 0.0621;  failing vs restarted-healthy only: 0.8920
    D_r  latency  122.4 +/-  12.2  detected 1.000  | false-alarm rate 0.2215 +/- 0.1808 (starting pre-onset 0.0604, only after 0.1611)  restarted-healthy 0.2760  not restarted 0.2184
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.6179 +/- 0.0323;  failing vs restarted-healthy only: 0.6960
  D_r - C_r latency: +53.4 (sd 10.6) 95% [+40.2, +66.6]   per seed [42.9, 62.5, 59.2, 61.7, 40.9]
  D_r - C_r false-alarm rate: -0.2240 (sd 0.2921) 95% [-0.5867, +0.1387]   per seed [0.1284, -0.3358, -0.0126, -0.28, -0.62]
  D_r - C_r trust AUC (first 100 steps): -0.1756 (sd 0.0524) 95% [-0.2406, -0.1105]  (descriptive)

--- condition f20 (cold start, f = 0.2) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 100 healthy 900 restarted-healthy 100 | tau C_r 0.8309 D_r 0.8409
    seed 2: attempted 50 failed 0 aggregated 50 | pop 100 healthy 900 restarted-healthy 100 | tau C_r 0.8364 D_r 0.4930
    seed 3: attempted 50 failed 0 aggregated 50 | pop 100 healthy 900 restarted-healthy 100 | tau C_r 0.8132 D_r 0.6783
    seed 4: attempted 50 failed 0 aggregated 50 | pop 100 healthy 900 restarted-healthy 100 | tau C_r 0.8494 D_r 0.6406
    seed 5: attempted 50 failed 0 aggregated 50 | pop 100 healthy 900 restarted-healthy 100 | tau C_r 0.8745 D_r 0.7173
  restarted-but-healthy RSUs sharing a backhaul segment with a failing RSU (sum over draws, per seed): [96, 100, 91, 100, 100]
  per model (failing RSUs, from onset 420); mean +/- sd over seeds of per-seed draw means
    C_r  latency   70.0 +/-   5.7  detected 1.000  | false-alarm rate 0.4402 +/- 0.2105 (starting pre-onset 0.1204, only after 0.3198)  restarted-healthy 0.3660  not restarted 0.4495
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7924 +/- 0.0480;  failing vs restarted-healthy only: 0.8890
    D_r  latency  115.0 +/-  13.9  detected 0.998  | false-alarm rate 0.2404 +/- 0.1729 (starting pre-onset 0.0604, only after 0.1800)  restarted-healthy 0.4140  not restarted 0.2188
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.6547 +/- 0.0438;  failing vs restarted-healthy only: 0.7160
  D_r - C_r latency: +45.0 (sd 12.7) 95% [+29.2, +60.8]   per seed [30.6, 61.3, 48.9, 50.6, 33.7]
  D_r - C_r false-alarm rate: -0.1998 (sd 0.2794) 95% [-0.5466, +0.1470]   per seed [0.1367, -0.3033, -0.0056, -0.2411, -0.5856]
  D_r - C_r trust AUC (first 100 steps): -0.1378 (sd 0.0336) 95% [-0.1795, -0.0960]  (descriptive)

--- condition f40 (cold start, f = 0.4) ---
  D42 per seed: attempted / failed 0 / aggregated (draws with a latency population) | RSUs in population, healthy, restarted-healthy (sums over draws) | tau
    seed 1: attempted 50 failed 0 aggregated 50 | pop 200 healthy 800 restarted-healthy 200 | tau C_r 0.8309 D_r 0.8409
    seed 2: attempted 50 failed 0 aggregated 50 | pop 200 healthy 800 restarted-healthy 200 | tau C_r 0.8364 D_r 0.4930
    seed 3: attempted 50 failed 0 aggregated 50 | pop 200 healthy 800 restarted-healthy 200 | tau C_r 0.8132 D_r 0.6783
    seed 4: attempted 50 failed 0 aggregated 50 | pop 200 healthy 800 restarted-healthy 200 | tau C_r 0.8494 D_r 0.6406
    seed 5: attempted 50 failed 0 aggregated 50 | pop 200 healthy 800 restarted-healthy 200 | tau C_r 0.8745 D_r 0.7173
  restarted-but-healthy RSUs sharing a backhaul segment with a failing RSU (sum over draws, per seed): [184, 198, 183, 189, 190]
  per model (failing RSUs, from onset 420); mean +/- sd over seeds of per-seed draw means
    C_r  latency   71.9 +/-   6.6  detected 1.000  | false-alarm rate 0.4387 +/- 0.2116 (starting pre-onset 0.1202, only after 0.3185)  restarted-healthy 0.4130  not restarted 0.4473
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.7826 +/- 0.0869;  failing vs restarted-healthy only: 0.8380
    D_r  latency  113.0 +/-  16.4  detected 0.997  | false-alarm rate 0.3010 +/- 0.1604 (starting pre-onset 0.0590, only after 0.2420)  restarted-healthy 0.4940  not restarted 0.2367
         trust AUC, first 100 steps after restart [400, 500) (failing vs all non-failing): 0.6429 +/- 0.0655;  failing vs restarted-healthy only: 0.6535
  D_r - C_r latency: +41.1 (sd 17.0) 95% [+20.0, +62.2]   per seed [25.4, 65.8, 45.9, 43.8, 24.6]
  D_r - C_r false-alarm rate: -0.1377 (sd 0.2758) 95% [-0.4802, +0.2047]   per seed [0.1762, -0.2587, 0.0712, -0.1562, -0.5212]
  D_r - C_r trust AUC (first 100 steps): -0.1398 (sd 0.0391) 95% [-0.1883, -0.0912]  (descriptive)
```

#### F53d - SECONDARY ARM, closed loop (results/s18/report-closed-secondary-raw.txt)
```
============================================================================================================================
S18 CLOSED LOOP (SECONDARY ARM): dwell filter and handoff on, draws 400-449, seeds [1, 2, 3, 4, 5] (RAW: no labels)
============================================================================================================================
weights (alpha, beta, gamma): C_r (1.0, 0.25, 0.6)  D_r (2.0, 0.25, 0.6)  (existing tuned files; no new tuning)

--- condition L0 (spreading fault, L = 0) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  C_r      success [400,1200) 0.9232 +/- 0.0105  | n_av   139.4 +/-   44.9 n_win   489.3 +/-   78.2 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.015 (15/1000)
  D_r      success [400,1200) 0.9080 +/- 0.0062  | n_av   176.7 +/-   43.3 n_win   585.6 +/-   45.7 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.018 (18/1000)
  D_r - C_r: -0.0152 (sd 0.0060) 95% [-0.0227, -0.0078]   per seed [-0.0144, -0.0136, -0.0257, -0.0109, -0.0116]
  D_r - D: +0.0017 (sd 0.0081) 95% [-0.0083, +0.0118]   per seed [-0.0086, 0.0138, -0.0001, 0.0034, 0.0001]
  C_r - C: +0.0002 (sd 0.0005) 95% [-0.0005, +0.0008]   per seed [0.0005, -0.0003, -0.0005, 0.0008, 0.0002]

--- condition L20 (spreading fault, L = 20) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  C_r      success [400,1200) 0.9246 +/- 0.0102  | n_av   158.6 +/-   46.4 n_win   503.8 +/-   75.2 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.014 (14/1000)
  D_r      success [400,1200) 0.9096 +/- 0.0059  | n_av   201.9 +/-   39.5 n_win   598.5 +/-   43.0 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.022 (22/1000)
  D_r - C_r: -0.0150 (sd 0.0058) 95% [-0.0222, -0.0078]   per seed [-0.0144, -0.0133, -0.0251, -0.0108, -0.0114]
  D_r - D: +0.0014 (sd 0.0076) 95% [-0.0080, +0.0109]   per seed [-0.0083, 0.0126, -0.001, 0.0035, 0.0004]
  C_r - C: +0.0000 (sd 0.0006) 95% [-0.0007, +0.0008]   per seed [0.0004, -0.0006, -0.0004, 0.0008, 0.0]

--- condition L40 (spreading fault, L = 40) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  C_r      success [400,1200) 0.9261 +/- 0.0100  | n_av   181.8 +/-   45.8 n_win   521.0 +/-   69.5 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.016 (16/1000)
  D_r      success [400,1200) 0.9116 +/- 0.0060  | n_av   222.3 +/-   35.1 n_win   612.3 +/-   39.8 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.022 (22/1000)
  D_r - C_r: -0.0146 (sd 0.0054) 95% [-0.0213, -0.0078]   per seed [-0.0141, -0.0129, -0.024, -0.0107, -0.0112]
  D_r - D: +0.0013 (sd 0.0073) 95% [-0.0077, +0.0103]   per seed [-0.0081, 0.0119, -0.0012, 0.0033, 0.0006]
  C_r - C: +0.0000 (sd 0.0005) 95% [-0.0005, +0.0006]   per seed [0.0003, -0.0004, -0.0004, 0.0006, 0.0001]

--- condition L80 (spreading fault, L = 80) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  C_r      success [400,1200) 0.9293 +/- 0.0097  | n_av   217.1 +/-   34.0 n_win   552.3 +/-   60.2 run-level never-avoided 0.016 +/- 0.026 | faulty RSUs never avoided 0.018 (18/1000)
  D_r      success [400,1200) 0.9157 +/- 0.0064  | n_av   252.4 +/-   29.2 n_win   637.6 +/-   32.8 run-level never-avoided 0.020 +/- 0.024 | faulty RSUs never avoided 0.025 (25/1000)
  D_r - C_r: -0.0136 (sd 0.0047) 95% [-0.0194, -0.0078]   per seed [-0.0131, -0.0119, -0.0218, -0.0107, -0.0105]
  D_r - D: +0.0012 (sd 0.0066) 95% [-0.0070, +0.0095]   per seed [-0.0072, 0.0111, -0.0012, 0.0028, 0.0006]
  C_r - C: +0.0001 (sd 0.0004) 95% [-0.0004, +0.0006]   per seed [0.0004, -0.0001, -0.0004, 0.0006, -0.0001]

--- condition f10 (cold start, f = 0.1) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  C_r      success [400,500) 0.9724 +/- 0.0091   after the window [500,1200) 0.9681 +/- 0.0022  | n_av    12.3 +/-    1.7 n_win    38.6 +/-    3.1 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/250)
  D_r      success [400,500) 0.9682 +/- 0.0101   after the window [500,1200) 0.9636 +/- 0.0028  | n_av    16.4 +/-    1.1 n_win    55.9 +/-    5.4 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/250)
  D_r - C_r: -0.0042 (sd 0.0044) 95% [-0.0097, +0.0013]   per seed [-0.0119, -0.0013, -0.0022, -0.004, -0.0015]
  D_r - C_r after the window [500,1200): -0.0046 95% [-0.0064, -0.0027]  (descriptive)
  D_r - D: +0.0008 (sd 0.0010) 95% [-0.0004, +0.0020]   per seed [0.0005, -0.0001, 0.0009, 0.0004, 0.0025]
  C_r - C: -0.0001 (sd 0.0002) 95% [-0.0004, +0.0002]   per seed [0.0002, -0.0004, -0.0001, 0.0001, -0.0002]

--- condition f20 (cold start, f = 0.2) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  C_r      success [400,500) 0.9679 +/- 0.0093   after the window [500,1200) 0.9556 +/- 0.0045  | n_av    27.0 +/-    5.2 n_win   128.1 +/-   17.7 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/500)
  D_r      success [400,500) 0.9630 +/- 0.0100   after the window [500,1200) 0.9449 +/- 0.0033  | n_av    34.7 +/-    4.7 n_win   180.3 +/-    4.2 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.000 (0/500)
  D_r - C_r: -0.0048 (sd 0.0037) 95% [-0.0095, -0.0002]   per seed [-0.0113, -0.0026, -0.0031, -0.0049, -0.0023]
  D_r - C_r after the window [500,1200): -0.0107 95% [-0.0132, -0.0082]  (descriptive)
  D_r - D: +0.0008 (sd 0.0014) 95% [-0.0009, +0.0025]   per seed [0.0016, -0.0004, 0.0011, -0.0007, 0.0026]
  C_r - C: -0.0001 (sd 0.0003) 95% [-0.0004, +0.0003]   per seed [0.0004, -0.0003, -0.0001, -0.0002, -0.0001]

--- condition f40 (cold start, f = 0.4) ---
  D42: attempted 250 per model, failed 0, aggregated 250
  C_r      success [400,500) 0.9593 +/- 0.0089   after the window [500,1200) 0.9323 +/- 0.0059  | n_av    53.3 +/-    6.7 n_win   302.7 +/-   33.8 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.002 (2/1000)
  D_r      success [400,500) 0.9525 +/- 0.0105   after the window [500,1200) 0.9101 +/- 0.0051  | n_av    74.5 +/-    6.0 n_win   424.6 +/-   14.0 run-level never-avoided 0.000 +/- 0.000 | faulty RSUs never avoided 0.011 (11/1000)
  D_r - C_r: -0.0069 (sd 0.0048) 95% [-0.0128, -0.0009]   per seed [-0.0147, -0.0044, -0.0048, -0.0078, -0.0026]
  D_r - C_r after the window [500,1200): -0.0222 95% [-0.0323, -0.0122]  (descriptive)
  D_r - D: +0.0005 (sd 0.0015) 95% [-0.0014, +0.0024]   per seed [-0.0003, -0.0005, 0.0012, -0.0008, 0.0028]
  C_r - C: +0.0001 (sd 0.0006) 95% [-0.0006, +0.0009]   per seed [0.0012, -0.0003, 0.0, -0.0001, -0.0001]
```
Interpretation: not yet recorded (raw table first, Standing Rule 6); F54 gives the verdicts and the one-line interpretation.

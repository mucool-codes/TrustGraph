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

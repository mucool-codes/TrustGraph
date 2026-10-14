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

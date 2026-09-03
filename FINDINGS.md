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

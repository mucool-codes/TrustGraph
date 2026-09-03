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

# TrustGraph demo

Double-click **`Run_Demo.bat`** in the project root. The demo opens in your browser. It needs no server and no internet.

| File | What it is |
|---|---|
| `index.html` | The demo front end, with Network, Pipeline details, Results, Explain a decision and References tabs, and a pipeline dropdown (A, B, B-oracle, C, D, E, Gate-C, H, D2) |
| `data.js` | Data the page reads, built from saved outputs (`scenarios/s6`, `traces/demo-seed100.npz`, `results/s10`, `results/s8`) |
| `export_demo_data.py` | Builds `data.js`. Read-only towards the project: it writes only `demo/data.js` |
| `Rebuild_Data.bat` | Re-runs the export. Inside `.venv` (with PyTorch) it also re-runs the frozen C head, so C's per-step trust is exact rather than the logged values held between decisions |

The demo scenario is seed 100, draw 7: the tuning seed. It is an illustration, not an evaluation (CLAIMS X16).
Multi-seed numbers on the Results tab come from `results/s10/tables/`.

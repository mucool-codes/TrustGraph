#!/usr/bin/env bash
# Generate the S3 train / validation / held-out scenarios for one seed (all injection
# draws < 100, D43). Usage: bash scripts/s3_make_scenarios.sh configs/demo.yaml 1
# Requires the seed's trace: python scripts/generate_trace.py --config <cfg> --seed <n>
set -e
CFG=${1:-configs/demo.yaml}; SEED=${2:-1}
PY=${PYTHON:-.venv/Scripts/python.exe}
gen() { $PY scripts/generate_scenario.py --config "$CFG" --seed "$SEED" --rho "$1" --fraction "$2" --draw "$3" > /dev/null; }
for rho in 0.0 0.25 0.5 0.75 1.0; do for f in 0.1 0.2 0.3; do for d in 0 1; do gen $rho $f $d; done; done; done
for rho in 0.0 0.5 1.0; do for f in 0.1 0.2 0.3; do gen $rho $f 40; done; done
gen 1.0 0.2 90
echo done

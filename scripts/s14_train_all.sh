#!/bin/bash
# S14 (D68 section 2): train D2 for seeds 1-5 and 100 in both regimes, 3 torch threads each, 5 at a time.
cd "$(dirname "$0")/.."
export PY=.venv/Scripts/python.exe
mkdir -p results/s14/train
printf '%s\n' "shared 1" "shared 2" "shared 3" "shared 4" "shared 5" "shared 100" "stag 1" "stag 2" "stag 3" "stag 4" "stag 5" "stag 100" |
xargs -P 5 -L 1 bash -c '
regime=$0; seed=$1
if [ "$regime" = shared ]; then cfg=configs/demo.yaml; else cfg=configs/staggered.yaml; fi
$PY scripts/s3_train.py --config $cfg --seed $seed --variants D2 --threads 3 --ckpt-dir checkpoints/s14/$regime --out results/s14/$regime > results/s14/train/$regime-seed$seed.log 2>&1
'

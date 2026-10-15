#!/bin/bash
# S15 (D72 section 2): train C-h and D2-h (variants C and D2 on the horizon objective) for seeds 1-5 and 100,
# staggered regime, 3 torch threads each, 5 at a time.
cd "$(dirname "$0")/.."
export PY=.venv/Scripts/python.exe
mkdir -p results/s15/train
printf '%s\n' 1 2 3 4 5 100 |
xargs -P 5 -L 1 bash -c '
seed=$0
$PY scripts/s3_train.py --config configs/staggered.yaml --seed $seed --variants C D2 --objective horizon --threads 3 --ckpt-dir checkpoints/s15 --out results/s15 > results/s15/train/seed$seed.log 2>&1
'

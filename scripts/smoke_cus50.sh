#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVRPTW_DB_ROOT="${EVRPTW_DB_ROOT:-/data/Maojie/EVRPTW-DB}"
export EVRPTW_DB_ROOT
export PYTHONPATH="$ROOT:$EVRPTW_DB_ROOT:${PYTHONPATH:-}"

python -m offline2online.train \
  --config cus50_o2o_full.yaml \
  --seed 2005 \
  --epochs 2 \
  --num-envs-per-gpu 2 \
  --n-traj 2 \
  --rollout-steps 4 \
  --num-minibatches 1 \
  --eval-limit 2 \
  --eval-batch-size 2 \
  --eval-n-traj 2 \
  --device cpu \
  --debug

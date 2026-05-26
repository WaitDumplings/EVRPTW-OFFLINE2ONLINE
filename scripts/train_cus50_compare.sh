#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVRPTW_DB_ROOT="${EVRPTW_DB_ROOT:-/data/Maojie/Github2/EVRPTW-DB}"
export EVRPTW_DB_ROOT
export PYTHONPATH="$ROOT:$EVRPTW_DB_ROOT:${PYTHONPATH:-}"

mkdir -p "$ROOT/results/launch_logs"

O2O_COMMON=(
  -m offline2online.train
  --config cus50_o2o_full.yaml
  --epochs 500
)

TERRAN_COMMON=(
  -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train
  --config cus50_terran.yaml
  --epochs 500
  --eval-n-traj 50
)

CUDA_VISIBLE_DEVICES=0 python "${O2O_COMMON[@]}" --seed 2005 \
  > "$ROOT/results/launch_logs/o2o_full_seed_2005.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python "${O2O_COMMON[@]}" --seed 2006 \
  > "$ROOT/results/launch_logs/o2o_full_seed_2006.log" 2>&1 &

(
  cd "$EVRPTW_DB_ROOT"
  CUDA_VISIBLE_DEVICES=2 python "${TERRAN_COMMON[@]}" --seed 2005 \
    > "$ROOT/results/launch_logs/terran_seed_2005.log" 2>&1
) &

(
  cd "$EVRPTW_DB_ROOT"
  CUDA_VISIBLE_DEVICES=3 python "${TERRAN_COMMON[@]}" --seed 2006 \
    > "$ROOT/results/launch_logs/terran_seed_2006.log" 2>&1
) &

wait
python -m offline2online.summarize_cus50_comparison


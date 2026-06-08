#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O2O_ROOT="${O2O_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2005}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
EVRPTW_DB_ROOT="${EVRPTW_DB_ROOT:-/data/Maojie/EVRPTW-DB}"

mkdir -p "$O2O_ROOT/results/launch_logs"
export EVRPTW_DB_ROOT
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cus15-offline-methods}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$O2O_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" -m offline2online.train \
  --config "$O2O_ROOT/ablation/configs/cus15_o2o_dapg_u4_1000.yaml" \
  --seed "$SEED" \
  --device cuda \
  "$@" \
  > "$O2O_ROOT/results/launch_logs/cus15_dapg_u4_1000_seed_${SEED}.log" 2>&1

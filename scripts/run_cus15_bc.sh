#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O2O_ROOT="${O2O_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-3009}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CONFIG="${CONFIG:-configs/cus15_bc.yaml}"

mkdir -p "$O2O_ROOT/results/launch_logs"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$O2O_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" -m offline2online.train \
  --config "$CONFIG" \
  --seed "$SEED" \
  --device cuda \
  "$@" \
  > "$O2O_ROOT/results/launch_logs/cus15_bc_seed_${SEED}.log" 2>&1

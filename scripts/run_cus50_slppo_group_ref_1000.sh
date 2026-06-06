#!/usr/bin/env bash
set -euo pipefail

O2O_ROOT="/data/Maojie/EVRPTW-OFFLINE2ONLINE"
PYTHON_BIN="/home/exx/anaconda3/envs/maojie/bin/python"
SEED="${SEED:-2005}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

mkdir -p "$O2O_ROOT/results/launch_logs"
export EVRPTW_DB_ROOT="/data/Maojie/EVRPTW-DB"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/matplotlib-cus50-offline-methods"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "$O2O_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" -m offline2online.train \
  --config "$O2O_ROOT/configs/cus50_o2o_sl_ppo_group_ref_1000.yaml" \
  --seed "$SEED" \
  --device cuda \
  > "$O2O_ROOT/results/launch_logs/cus50_slppo_group_ref_1000_seed_${SEED}.log" 2>&1

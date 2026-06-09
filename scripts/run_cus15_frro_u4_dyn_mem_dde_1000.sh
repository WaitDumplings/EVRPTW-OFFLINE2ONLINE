#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O2O_ROOT="${O2O_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2005}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
EVRPTW_DB_ROOT="${EVRPTW_DB_ROOT:-/data/Maojie/EVRPTW-DB}"
FRRO_ALPHA="${FRRO_ALPHA:-0.10}"
FRRO_EXPERT_WEIGHT="${FRRO_EXPERT_WEIGHT:-2.0}"
FRRO_TAG="${FRRO_TAG:-A010_LE2}"
RUN_NAME="${RUN_NAME:-O2O_CUS15_FRRO_${FRRO_TAG}_DYN_MEM_DDE_KV_R40_U4_E1000}"

mkdir -p "$O2O_ROOT/results/launch_logs"
export EVRPTW_DB_ROOT
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cus15-offline-methods}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cmd=(
  "$PYTHON_BIN" -m offline2online.train
  --config "$O2O_ROOT/configs/cus15_o2o_frro_u4_dyn_mem_dde_1000.yaml"
  --run-name "$RUN_NAME"
  --seed "$SEED"
  --device cuda
  --sl-coef "$FRRO_ALPHA"
  --frro-coef 1.0
  --frro-expert-candidate-weight "$FRRO_EXPERT_WEIGHT"
)
if [[ -n "${TRAIN_DATASET_PATH:-}" ]]; then
  cmd+=(--train-dataset-path "$TRAIN_DATASET_PATH")
fi
if [[ -n "${EVAL_PATH:-}" ]]; then
  cmd+=(--eval-path "$EVAL_PATH")
fi
if [[ -n "${EXPERT_SOLUTION_PATH:-}" ]]; then
  cmd+=(--expert-solution-path "$EXPERT_SOLUTION_PATH")
fi
if [[ -n "${EXPERT_DATASET_PATH:-}" ]]; then
  cmd+=(--expert-dataset-path "$EXPERT_DATASET_PATH")
fi
if (($#)); then
  cmd+=("$@")
fi

cd "$O2O_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}" \
  > "$O2O_ROOT/results/launch_logs/cus15_frro_${FRRO_TAG}_u4_dyn_mem_dde_1000_seed_${SEED}.log" 2>&1

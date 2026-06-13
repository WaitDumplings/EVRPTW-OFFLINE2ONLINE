#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O2O_ROOT="${O2O_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2005}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DDE_VARIANT="${DDE_VARIANT:-dde0}"
EVRPTW_DB_ROOT="${EVRPTW_DB_ROOT:-/data/Maojie/Github2/EVRPTW-DB}"

case "$DDE_VARIANT" in
  dde0) CONFIG="$O2O_ROOT/configs/cus15_dde0_static_single_critic.yaml" ; LOG_TAG="cus15_dde0_static_seed_${SEED}" ;;
  dde1) CONFIG="$O2O_ROOT/configs/cus15_dde1_action_bias_only.yaml" ; LOG_TAG="cus15_dde1_action_bias_seed_${SEED}" ;;
  dde2) CONFIG="$O2O_ROOT/configs/cus15_dde2_action_key_bias.yaml" ; LOG_TAG="cus15_dde2_action_key_bias_seed_${SEED}" ;;
  dde3) CONFIG="$O2O_ROOT/configs/cus15_dde3_full_residual.yaml" ; LOG_TAG="cus15_dde3_full_residual_seed_${SEED}" ;;
  *) echo "Unknown DDE_VARIANT=$DDE_VARIANT; expected dde0|dde1|dde2|dde3" >&2; exit 2 ;;
esac

mkdir -p "$O2O_ROOT/results/launch_logs"
export EVRPTW_DB_ROOT
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cus15-dde}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cmd=(
  "$PYTHON_BIN" -m offline2online.train
  --config "$CONFIG"
  --seed "$SEED"
  --device cuda
)
if (($#)); then
  cmd+=("$@")
fi

cd "$O2O_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}" \
  > "$O2O_ROOT/results/launch_logs/${LOG_TAG}.log" 2>&1

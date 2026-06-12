#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O2O_ROOT="${O2O_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/npg/miniconda3/envs/maojie/bin/python}"
EVRPTW_DB_ROOT="${EVRPTW_DB_ROOT:-/data/Maojie/Github2/EVRPTW-DB}"
SEED="${SEED:-2005}"
OUT="$O2O_ROOT/results/launch_logs/cus15_ppo_dde_bafipo_e500_seed${SEED}"
mkdir -p "$OUT"

export EVRPTW_DB_ROOT
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

launch() {
  local gpu="$1"
  local cfg="$2"
  local tag="$3"
  local log="$OUT/${tag}_seed_${SEED}_gpu${gpu}.log"
  local pidf="$OUT/${tag}_seed_${SEED}_gpu${gpu}.pid"
  : > "$log"
  setsid env     EVRPTW_DB_ROOT="$EVRPTW_DB_ROOT"     PYTHONUNBUFFERED=1     PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"     CUDA_VISIBLE_DEVICES="$gpu"     "$PYTHON_BIN" -m offline2online.train       --config "$O2O_ROOT/$cfg"       --seed "$SEED"       --device cuda       > "$log" 2>&1 < /dev/null &
  echo $! > "$pidf"
  echo "$tag seed=$SEED gpu=$gpu pid=$(cat "$pidf") log=$log"
}

cd "$O2O_ROOT"
launch 0 configs/cus15_o2o_ppo_base_dyn_no_dde_u3_ne480_eb1000_en8_500.yaml ppo_base
launch 1 configs/cus15_o2o_ppo_dde_dyn_u3_ne480_eb1000_en8_500.yaml ppo_dde
launch 2 configs/cus15_o2o_bafipo_p005_mb8_dyn_dde_u3_ne480_eb1000_en8_500.yaml bafipo_p005
launch 3 configs/cus15_o2o_bafipo_p010_mb8_dyn_dde_u3_ne480_eb1000_en8_500.yaml bafipo_p010

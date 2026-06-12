#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O2O_ROOT="${O2O_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/npg/miniconda3/envs/maojie/bin/python}"
EVRPTW_DB_ROOT="${EVRPTW_DB_ROOT:-/data/Maojie/Github2/EVRPTW-DB}"
OUT="$O2O_ROOT/results/launch_logs/cus15_dapg_vs_bafipo_seed2005_2006"
mkdir -p "$OUT"

export EVRPTW_DB_ROOT
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

launch() {
  local gpu="$1"
  local seed="$2"
  local tag="bafipo_v1"
  local cfg="configs/cus15_o2o_bafipo_dyn_dde_u3_ne480_eb1000_en8_1000.yaml"
  local log="$OUT/${tag}_seed_${seed}_gpu${gpu}.log"
  local pidf="$OUT/${tag}_seed_${seed}_gpu${gpu}.pid"
  : > "$log"
  setsid env     EVRPTW_DB_ROOT="$EVRPTW_DB_ROOT"     PYTHONUNBUFFERED=1     PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"     CUDA_VISIBLE_DEVICES="$gpu"     "$PYTHON_BIN" -m offline2online.train       --config "$O2O_ROOT/$cfg"       --seed "$seed"       --device cuda       > "$log" 2>&1 < /dev/null &
  echo $! > "$pidf"
  echo "$tag seed=$seed gpu=$gpu pid=$(cat "$pidf") log=$log"
}

cd "$O2O_ROOT"
launch "${GPU_SEED_2005:-2}" 2005
launch "${GPU_SEED_2006:-3}" 2006

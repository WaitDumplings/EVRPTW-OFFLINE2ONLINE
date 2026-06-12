#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O2O_ROOT="${O2O_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/npg/miniconda3/envs/maojie/bin/python}"
EVRPTW_DB_ROOT="${EVRPTW_DB_ROOT:-/data/Maojie/Github2/EVRPTW-DB}"
SEEDS=("${SEEDS:-2005 2006}")
OUT="$O2O_ROOT/results/launch_logs/cus15_dapg_vs_gadapg_seed2005_2006"
mkdir -p "$OUT"

export EVRPTW_DB_ROOT
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

launch() {
  local gpu="$1"
  local cfg="$2"
  local seed="$3"
  local tag="$4"
  local log="$OUT/${tag}_seed_${seed}_gpu${gpu}.log"
  local pidf="$OUT/${tag}_seed_${seed}_gpu${gpu}.pid"
  : > "$log"
  setsid env     EVRPTW_DB_ROOT="$EVRPTW_DB_ROOT"     PYTHONUNBUFFERED=1     PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"     CUDA_VISIBLE_DEVICES="$gpu"     "$PYTHON_BIN" -m offline2online.train       --config "$O2O_ROOT/$cfg"       --seed "$seed"       --device cuda       > "$log" 2>&1 < /dev/null &
  echo $! > "$pidf"
  echo "$tag seed=$seed gpu=$gpu pid=$(cat "$pidf") log=$log"
}

cd "$O2O_ROOT"
launch 0 configs/cus15_o2o_dapg_dyn_dde_u3_ne480_eb1000_en8_1000.yaml 2005 dapg_base
launch 1 configs/cus15_o2o_dapg_dyn_dde_u3_ne480_eb1000_en8_1000.yaml 2006 dapg_base
launch 2 configs/cus15_o2o_gadapg_dyn_dde_u3_ne480_eb1000_en8_1000.yaml 2005 gadapg_ga03
launch 3 configs/cus15_o2o_gadapg_dyn_dde_u3_ne480_eb1000_en8_1000.yaml 2006 gadapg_ga03

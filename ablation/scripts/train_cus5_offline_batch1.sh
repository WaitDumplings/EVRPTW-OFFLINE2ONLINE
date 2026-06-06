#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/Maojie/Github2/EVRPTW-OFFLINE2ONLINE"
LOG_DIR="$REPO_ROOT/results/launch_logs/cus5_offline_batch1"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

launch_method() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    for seed in 2005 2006; do
      echo "[$(date '+%F %T')] START method=$name seed=$seed gpu=$gpu config=$config"
      conda run -n maojie python -m offline2online.train \
        --config "$config" \
        --seed "$seed" \
        --epochs 1000
      echo "[$(date '+%F %T')] DONE method=$name seed=$seed gpu=$gpu"
    done
  ) > "$LOG_DIR/${name}_gpu${gpu}.log" 2>&1 &
  echo $! > "$LOG_DIR/${name}_gpu${gpu}.pid"
  echo "launched $name on GPU $gpu pid=$(cat "$LOG_DIR/${name}_gpu${gpu}.pid")"
}

launch_method 0 O2O_PPO cus5_o2o_ppo.yaml
launch_method 1 O2O_BC_PPO ablation/configs/cus5_o2o_bc_ppo.yaml
launch_method 2 O2O_DAPG ablation/configs/cus5_o2o_dapg.yaml
launch_method 3 O2O_ROUTE_BC_PPO ablation/configs/cus5_o2o_route_bc_ppo.yaml

wait

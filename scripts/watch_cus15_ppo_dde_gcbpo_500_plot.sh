#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O2O_ROOT="${O2O_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/npg/miniconda3/envs/maojie/bin/python}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
LOG_DIR="$O2O_ROOT/results/launch_logs/cus15_ppo_dde_gcbpo_e500_seed2005"
mkdir -p "$LOG_DIR"

cd "$O2O_ROOT"
while true; do
  date '+[%F %T] update cus15 PPO/DDE/GCBPO plot'
  "$PYTHON_BIN" -m offline2online.plot_cus15_ppo_dde_gcbpo_500
  sleep "$INTERVAL_SECONDS"
done

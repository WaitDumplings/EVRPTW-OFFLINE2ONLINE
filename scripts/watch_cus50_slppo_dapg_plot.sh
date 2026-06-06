#!/usr/bin/env bash
set -euo pipefail

O2O_ROOT="/data/Maojie/EVRPTW-OFFLINE2ONLINE"
PYTHON_BIN="/home/exx/anaconda3/envs/maojie/bin/python"
POLL_SECONDS="${POLL_SECONDS:-300}"

export EVRPTW_DB_ROOT="/data/Maojie/EVRPTW-DB"
export MPLCONFIGDIR="/tmp/matplotlib-cus50-offline-methods"

cd "$O2O_ROOT"
while tmux has-session -t cus50_slppo_group_ref_1000 2>/dev/null || tmux has-session -t cus50_dapg_1000 2>/dev/null; do
  "$PYTHON_BIN" -m offline2online.plot_cus50_slppo_dapg || true
  sleep "$POLL_SECONDS"
done

"$PYTHON_BIN" -m offline2online.plot_cus50_slppo_dapg || true

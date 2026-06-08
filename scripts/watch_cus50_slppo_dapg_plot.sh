#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O2O_ROOT="${O2O_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
POLL_SECONDS="${POLL_SECONDS:-300}"
EVRPTW_DB_ROOT="${EVRPTW_DB_ROOT:-/data/Maojie/EVRPTW-DB}"

export EVRPTW_DB_ROOT
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cus50-offline-methods}"

cd "$O2O_ROOT"
while tmux has-session -t cus50_slppo_group_ref_1000 2>/dev/null \
  || tmux has-session -t cus50_slppo_group_ref_u4_1000 2>/dev/null \
  || tmux has-session -t cus50_dapg_1000 2>/dev/null \
  || tmux has-session -t cus50_dapg_u4_1000 2>/dev/null \
  || tmux has-session -t cus50_dapg_u4_dyn_1000 2>/dev/null; do
  "$PYTHON_BIN" -m offline2online.plot_cus50_slppo_dapg || true
  sleep "$POLL_SECONDS"
done

"$PYTHON_BIN" -m offline2online.plot_cus50_slppo_dapg || true

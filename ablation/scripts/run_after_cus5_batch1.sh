#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/Maojie/Github2/EVRPTW-OFFLINE2ONLINE"
BATCH_NAME="cus5_offline_batch1"
LOG_DIR="$REPO_ROOT/results/launch_logs/${BATCH_NAME}"
WATCH_LOG="$LOG_DIR/after_${BATCH_NAME}.log"
NEXT_SCRIPT="${1:-}"

mkdir -p "$LOG_DIR"
{
  echo "[$(date '+%F %T')] watcher started for $BATCH_NAME"
  if [[ -z "$NEXT_SCRIPT" ]]; then
    echo "[$(date '+%F %T')] no next script provided; watcher will only report completion"
  else
    echo "[$(date '+%F %T')] next script: $NEXT_SCRIPT"
  fi

  count_matching() {
    local pattern="$1"
    ps -ef | awk -v pat="$pattern" '$0 ~ pat && $0 !~ /awk -v pat/ {c++} END {print c + 0}'
  }

  while true; do
    running_count=$(count_matching "python -m offline2online.train --config .*cus5_o2o_(ppo|bc_ppo|dapg|route_bc_ppo|ppo_group_adv|ppo_ref_adv|sl_ppo_group_adv|sl_ppo_ref_adv)\.yaml")
    launcher_count=$(count_matching "train_cus5_offline_batch1\.sh")
    echo "[$(date '+%F %T')] running_train=$running_count launcher=$launcher_count"
    if [[ "$running_count" == "0" && "$launcher_count" == "0" ]]; then
      break
    fi
    sleep 60
  done

  echo "[$(date '+%F %T')] $BATCH_NAME finished"
  if [[ -n "$NEXT_SCRIPT" ]]; then
    if [[ ! -x "$NEXT_SCRIPT" ]]; then
      echo "[$(date '+%F %T')] ERROR: next script not executable or missing: $NEXT_SCRIPT"
      exit 2
    fi
    echo "[$(date '+%F %T')] launching next script"
    cd "$REPO_ROOT"
    bash "$NEXT_SCRIPT"
    echo "[$(date '+%F %T')] next script finished"
  fi
} >> "$WATCH_LOG" 2>&1

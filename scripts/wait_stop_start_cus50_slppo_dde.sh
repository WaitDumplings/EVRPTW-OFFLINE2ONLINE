#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O2O_ROOT="${O2O_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/exx/anaconda3/envs/maojie/bin/python}"
SEED="${SEED:-2005}"
TARGET_EPOCH="${TARGET_EPOCH:-400}"
POLL_SECONDS="${POLL_SECONDS:-300}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

DYN_SESSION="cus50_slppo_group_ref_u4_dyn_1000"
MEM_SESSION="cus50_slppo_group_ref_u4_dyn_mem_1000"
DDE_SESSION="cus50_slppo_group_ref_u4_dyn_mem_dde_1000"

DYN_EVAL="$O2O_ROOT/results/logs/Cus_50_CS_10/O2O_CUS50_SL_PPO_GROUP_REF_DYN_R70_U4_E1000/seed_${SEED}/eval_log.csv"
MEM_EVAL="$O2O_ROOT/results/logs/Cus_50_CS_10/O2O_CUS50_SL_PPO_GROUP_REF_MEM_DYN_R70_U4_E1000/seed_${SEED}/eval_log.csv"

latest_epoch() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  awk -F, 'NR > 1 && ($1 + 0) > max_epoch { max_epoch = $1 + 0 } END { print int(max_epoch) }' "$path"
}

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

cd "$O2O_ROOT"
log "waiting for ${DYN_SESSION} and ${MEM_SESSION} to reach epoch ${TARGET_EPOCH}"

while true; do
  dyn_epoch="$(latest_epoch "$DYN_EVAL")"
  mem_epoch="$(latest_epoch "$MEM_EVAL")"
  log "progress dyn=${dyn_epoch}, mem=${mem_epoch}, target=${TARGET_EPOCH}"

  if ((dyn_epoch >= TARGET_EPOCH && mem_epoch >= TARGET_EPOCH)); then
    log "target reached; stopping current dynamic experiments"
    tmux kill-session -t "$DYN_SESSION" 2>/dev/null || true
    tmux kill-session -t "$MEM_SESSION" 2>/dev/null || true
    sleep 10

    if tmux has-session -t "$DDE_SESSION" 2>/dev/null; then
      log "${DDE_SESSION} is already running"
    else
      log "starting ${DDE_SESSION} on CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}"
      tmux new-session -d -s "$DDE_SESSION" \
        "O2O_ROOT='$O2O_ROOT' CUDA_DEVICE='$CUDA_DEVICE' SEED='$SEED' PYTHON_BIN='$PYTHON_BIN' '$O2O_ROOT/scripts/run_cus50_slppo_group_ref_u4_dyn_mem_dde_1000.sh'"
    fi

    "$PYTHON_BIN" -m offline2online.plot_cus50_slppo_dapg || true
    log "handoff complete"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done

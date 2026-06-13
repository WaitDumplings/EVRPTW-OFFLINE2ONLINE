#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O2O_ROOT="${O2O_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INTERVAL="${INTERVAL:-300}"

cd "$O2O_ROOT"
while true; do
  "$PYTHON_BIN" -m offline2online.plot_cus15_dde_supplement || true
  sleep "$INTERVAL"
done

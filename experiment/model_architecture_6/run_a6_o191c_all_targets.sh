#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$SCRIPT_DIR"
RESULT_ROOT=$(PYTHONPATH="$PROJECT_ROOT" conda run -n sapien python -c 'from path_config import JOINTTRAIN_ARCH6_O191C_RESULT_ROOT; print(JOINTTRAIN_ARCH6_O191C_RESULT_ROOT)')
mkdir -p "$RESULT_ROOT/logs"

run_wave() {
  local wave=$1
  shift
  local pids=()
  local gpu=0
  for target_index in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu conda run -n sapien python run_a6_o191c_recovery_residual_seed_live.py \
      --max-calls 650 \
      --target-index "$target_index" \
      --execute-prefix 8 \
      >"$RESULT_ROOT/logs/target_${target_index}.log" 2>&1 &
    pids+=("$!")
    echo "started wave=$wave target=$target_index gpu=$gpu pid=$!"
    gpu=$((gpu + 1))
  done
  while true; do
    local active=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        active=$((active + 1))
      fi
    done
    echo "$(date -u +%FT%TZ) wave=$wave active_jobs=$active"
    if [[ $active -eq 0 ]]; then
      break
    fi
    sleep 30
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ $status -ne 0 ]]; then
    echo "O191C wave=$wave failed"
    exit "$status"
  fi
}

run_wave 1 0 1 2 3
run_wave 2 4 5 6 7
conda run -n sapien python run_a6_o191c_recovery_residual_seed_live_aggregate.py \
  >"$RESULT_ROOT/logs/aggregate.log" 2>&1
echo "O191C aggregate complete"

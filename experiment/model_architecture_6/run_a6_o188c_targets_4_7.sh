#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$SCRIPT_DIR"
RESULT_ROOT=$(PYTHONPATH="$PROJECT_ROOT" conda run -n sapien python -c 'from path_config import JOINTTRAIN_ARCH6_O188C_RESULT_ROOT; print(JOINTTRAIN_ARCH6_O188C_RESULT_ROOT)')
mkdir -p "$RESULT_ROOT/logs"

pids=()
for target_index in 4 5 6 7; do
  gpu=$((target_index - 4))
  CUDA_VISIBLE_DEVICES=$gpu conda run -n sapien python run_a6_o188c_fixed_budget_live.py \
    --max-calls 650 \
    --target-index "$target_index" \
    --execute-prefix 8 \
    >"$RESULT_ROOT/logs/target_${target_index}.log" 2>&1 &
  pids+=("$!")
  echo "started target=$target_index gpu=$gpu pid=$!"
done

while true; do
  active=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      active=$((active + 1))
    fi
  done
  echo "$(date -u +%FT%TZ) active_jobs=$active"
  if [[ $active -eq 0 ]]; then
    break
  fi
  sleep 30
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ $status -ne 0 ]]; then
  echo "one or more O188C target jobs failed"
  exit "$status"
fi

conda run -n sapien python run_a6_o188c_fixed_budget_live_aggregate.py \
  >"$RESULT_ROOT/logs/aggregate.log" 2>&1
echo "O188C aggregate complete"

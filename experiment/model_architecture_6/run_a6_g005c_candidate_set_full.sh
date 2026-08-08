#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RESULTS="$ROOT/jointTrain_new/experiment/model_architecture_6/results"
LOG="$RESULTS/a6_g005c_candidate_set_full.log"
exec > >(tee -a "$LOG") 2>&1

run_until_complete() {
  local gpu="$1"
  local runner="$2"
  local summary="$3"
  local attempt
  for attempt in $(seq 1 64); do
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n sapien env PYTHONPATH="$ROOT" \
      python "$runner" --targets 8 --resume --max-new-candidates 1
    if [[ -f "$summary" ]] && jq -e '.terminal == true and .complete == true' "$summary" >/dev/null; then
      return 0
    fi
  done
  echo "candidate process loop exhausted before terminal summary: $summary" >&2
  return 1
}

run_until_complete \
  0 \
  "$ROOT/jointTrain_new/experiment/model_architecture_6/run_a6_g005c_gt_traj_candidate_physical_pilot.py" \
  "$RESULTS/a6_g005c_gt_traj_candidate_physical_pilot_v2/summary.json"

run_until_complete \
  1 \
  "$ROOT/jointTrain_new/experiment/model_architecture_6/run_a6_g005c_gt_qpose_candidate_physical_pilot.py" \
  "$RESULTS/a6_g005c_gt_qpose_candidate_physical_pilot_v2/summary.json"

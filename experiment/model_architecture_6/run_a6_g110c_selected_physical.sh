#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
run_route() {
  local route="$1" gpu="$2"
  local log="jointTrain_new/experiment/model_architecture_6/results/a6_g110c_learned_selected_physical_v1/${route}.log"
  for index in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES="$gpu" python jointTrain_new/experiment/model_architecture_6/run_a6_g110c_learned_selected_physical.py --route "$route" --index "$index" >> "$log" 2>&1
  done
}
run_route traj 0 &
pid_traj=$!
run_route qpose 1 &
pid_qpose=$!
wait "$pid_traj"
wait "$pid_qpose"
python jointTrain_new/experiment/model_architecture_6/run_a6_g110c_learned_selected_physical.py --aggregate

#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUNNER="$ROOT/jointTrain_new/experiment/model_architecture_6/run_a6_a010_a020_affordance_train.py"
RESULT="$ROOT/jointTrain_new/experiment/model_architecture_6/results/a6_a020c_affordance_clean_train_v1"
mkdir -p "$RESULT"

pids=()
for spec in "20260806:1" "20260807:2" "20260808:3"; do
  seed="${spec%%:*}"
  gpu="${spec##*:}"
  python "$RUNNER" --mode full --steps 7000 --batch-size 48 --gradient-accumulation 2 --num-workers 4 --seed "$seed" --gpu "$gpu" >"$RESULT/seed_${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"

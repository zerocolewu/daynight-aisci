#!/usr/bin/env bash
# Run the full CPA-GAT (ORIG_BB, atomfeat35) matrix TWO more times (run2, run3)
# with the EXACT same config as run1, to measure pure run-to-run variance.
# Results land in results/run2/ and results/run3/ respectively (RUN_TAG routes them).
#
#   GPU0 : tg (6) + ea (3)   per run
#   GPU1 : tm (6) + ip (3)   per run
# run2 finishes on both GPUs, then run3 starts. After both, aggregate_runs.py
# computes mean +/- std of ens_mean RMSE over the 3 independent runs.
set -e
cd "$(dirname "$0")"
mkdir -p logs

for TAG in run2 run3; do
    echo "=== Independent run: $TAG (identical config to run1) ==="
    RUN_TAG=$TAG CUDA_VISIBLE_DEVICES=0 python run.py --datasets tg ea > logs/${TAG}_gpu0.log 2>&1 &
    PID0=$!
    RUN_TAG=$TAG CUDA_VISIBLE_DEVICES=1 python run.py --datasets tm ip > logs/${TAG}_gpu1.log 2>&1 &
    PID1=$!
    echo "$TAG  GPU0 pid=$PID0 (tg, ea) -> logs/${TAG}_gpu0.log"
    echo "$TAG  GPU1 pid=$PID1 (tm, ip) -> logs/${TAG}_gpu1.log"
    wait $PID0; S0=$?
    wait $PID1; S1=$?
    echo "$TAG  GPU0 exit=$S0  GPU1 exit=$S1"
done

echo "=== aggregating 3 independent runs ==="
python aggregate_runs.py --runs run1 run2 run3
echo "=== DONE ==="

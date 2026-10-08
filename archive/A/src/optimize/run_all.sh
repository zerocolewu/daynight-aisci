#!/usr/bin/env bash
# Launch the PTKGE matrix on TWO GPUs concurrently, then collect results.
#
#   GPU0 : tg (6) + ea (3)
#   GPU1 : tm (6) + ip (3)
# (tg/tm = 2 splits {42,71} x 3 init; ea/ip = official split x 3 init.)
# Runs whose result JSON already exists are skipped (--force re-runs).
# Rebalance by editing the --datasets lists below; the two lists must be disjoint
# and together cover all four datasets.
set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "=== PTKGE optimize: launching GPU0 and GPU1 ==="
CUDA_VISIBLE_DEVICES=0 python run.py --datasets tg ea > logs/gpu0.log 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python run.py --datasets tm ip > logs/gpu1.log 2>&1 &
PID1=$!

echo "GPU0 pid=$PID0 (tg, ea)  -> logs/gpu0.log"
echo "GPU1 pid=$PID1 (tm, ip) -> logs/gpu1.log"

wait $PID0; S0=$?
wait $PID1; S1=$?
echo "GPU0 exit=$S0  GPU1 exit=$S1"

echo "=== collecting results ==="
python collect_results.py
echo "=== DONE ==="

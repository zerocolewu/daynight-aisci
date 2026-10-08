#!/usr/bin/env bash
# Run ONE independent campaign (all datasets) on TWO GPUs concurrently, then
# collect per-run results into results/<RUN_ID>/.
#
# Usage:
#   bash run_one.sh run2
#   bash run_one.sh run3
#
# Env:
#   PTKGE_RUN_ID (required, set from $1) -- e.g. run2, run3
#   CUDA_VISIBLE_DEVICES is set per GPU inside this script.
#
# GPU0 : tg (6) + ea (3)
# GPU1 : tm (6) + ip (3)
# (tg/tm = 2 splits {42,71} x 3 init; ea/ip = official split x 3 init.)
# Runs whose result JSON already exists are skipped (--force re-runs).
set -e
cd "$(dirname "$0")"
mkdir -p logs

RUN_ID="${1:-}"
if [ -z "$RUN_ID" ]; then
    echo "Usage: bash run_one.sh <run_id>   (e.g. run2, run3)"; exit 1
fi
export PTKGE_RUN_ID="$RUN_ID"

echo "=== PTKGE optimize: run_id=$RUN_ID launching GPU0 and GPU1 ==="
CUDA_VISIBLE_DEVICES=0 python run.py --datasets tg ea > "logs/${RUN_ID}_gpu0.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python run.py --datasets tm ip > "logs/${RUN_ID}_gpu1.log" 2>&1 &
PID1=$!

echo "GPU0 pid=$PID0 (tg, ea)  -> logs/${RUN_ID}_gpu0.log"
echo "GPU1 pid=$PID1 (tm, ip)  -> logs/${RUN_ID}_gpu1.log"

wait $PID0; S0=$?
wait $PID1; S1=$?
echo "GPU0 exit=$S0  GPU1 exit=$S1"

echo "=== collecting results for $RUN_ID ==="
python collect_results.py
echo "=== DONE $RUN_ID ==="

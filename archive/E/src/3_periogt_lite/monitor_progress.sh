#!/bin/bash
# Monitor iteration 4 pipeline progress and estimate completion
# Run this script to get a quick status update

WORKDIR="/home/home_public/aisci/AIscientist/src/3_periogt_lite"
cd "$WORKDIR" || exit 1

echo "========================================"
echo "PerioGT-Lite Iteration 4 Pipeline Status"
echo "========================================"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check pipeline process
if ps -p 450016 > /dev/null 2>&1; then
    echo "Pipeline process: RUNNING (PID 450016)"
else
    echo "Pipeline process: NOT RUNNING (PID 450016)"
fi

# Check training process
TRAIN_PID=$(ps aux | grep "python.*pretrain.*edge_drop_feat_mask" | grep -v grep | awk '{print $2}')
if [ -n "$TRAIN_PID" ]; then
    echo "Training process: RUNNING (PID $TRAIN_PID)"
    ELAPSED=$(ps -p $TRAIN_PID -o etime= | tr -d ' ')
    echo "Training elapsed: $ELAPSED"
else
    echo "Training process: NOT FOUND"
fi

echo ""
echo "----------------------------------------"
echo "Progress by Corruption Rate:"
echo "----------------------------------------"

# Check each corruption rate
for rate in 40 50 60; do
    logfile="pretrain_edge_drop_feat_mask_${rate}.log"
    jsonfile="pretrain_log_edge_drop_feat_mask_${rate}.json"

    if [ -f "$logfile" ]; then
        # Count completed epochs
        epochs=$(grep -c "^Epoch [0-9]*:" "$logfile" 2>/dev/null || echo "0")
        echo "--- ${rate}% Corruption ---"

        if [ "$epochs" -gt 0 ]; then
            # Get last few epochs
            echo "Completed epochs: $epochs/50"
            grep "^Epoch [0-9]*:" "$logfile" | tail -3 | while read line; do
                echo "  $line"
            done

            # Get latest cos_sim if available
            if [ -f "$jsonfile" ]; then
                latest_cos=$(python3 -c "import json; f=open('$jsonfile'); d=json.load(f); print(d.get('epoch_metrics', [])[-1].get('cosine_sim', 'N/A') if d.get('epoch_metrics') else 'N/A')" 2>/dev/null)
                echo "  Latest cos_sim: $latest_cos"
            fi
        else
            echo "Status: Not started or in progress (no epochs completed yet)"
            # Show current progress if any
            latest_line=$(tail -1 "$logfile" 2>/dev/null | grep -oP '\d+%/.*?\| \d+/9375' | head -1)
            if [ -n "$latest_line" ]; then
                echo "  Current: $latest_line"
            fi
        fi
    else
        echo "--- ${rate}% Corruption ---"
        echo "Status: Logfile not found (not started yet)"
    fi
    echo ""
done

echo "----------------------------------------"
echo "Pipeline Log (last 10 lines):"
echo "----------------------------------------"
tail -10 pipeline_iter4.log

echo ""
echo "========================================"

#!/bin/bash
# Monitoring script for PerioGT-Lite Iteration 4 pipeline

WORKDIR="/home/home_public/aisci/AIscientist/src/3_periogt_lite"
PIPELINE_PID=450016

cd "$WORKDIR" || exit 1

check_status() {
    echo "=== $(date) ==="
    echo ""

    # Check if pipeline process is running
    if ps -p $PIPELINE_PID > /dev/null 2>&1; then
        echo "✓ Pipeline process (PID $PIPELINE_PID) is running"
    else
        echo "✗ Pipeline process (PID $PIPELINE_PID) NOT running"
    fi

    # Check for actual training process
    TRAIN_PID=$(ps aux | grep "python.*pretrain.*edge_drop_feat_mask" | grep -v grep | awk '{print $2}')
    if [ -n "$TRAIN_PID" ]; then
        echo "✓ Training process (PID $TRAIN_PID) is active"
    else
        echo "✗ No training process found"
    fi

    echo ""
    echo "=== Latest Epoch Progress ==="
    # Check each corruption rate's progress
    for rate in 40 50 60; do
        logfile="pretrain_edge_drop_feat_mask_${rate}.log"
        if [ -f "$logfile" ]; then
            echo "--- ${rate}% Corruption ---"
            grep "Epoch" "$logfile" | grep -v "it/s" | tail -3
            echo ""
        fi
    done

    echo "=== Pipeline Log (last 20 lines) ==="
    tail -20 pipeline_iter4.log

    echo ""
    echo "=========================================="
    echo ""
}

# Run status check
check_status

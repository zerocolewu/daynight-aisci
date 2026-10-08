#!/bin/bash
# Monitor training progress

echo "Monitoring HEPT v2 training..."
echo "================================"

while true; do
    if [ ! -f experiment_v2_training.log ]; then
        echo "Waiting for log file..."
        sleep 10
        continue
    fi

    # Check if training is still running
    if ! ps aux | grep -q "[p]ython -u run_experiment_v2.py"; then
        echo ""
        echo "================================"
        echo "Training completed!"
        echo "================================"
        echo ""
        echo "Final results:"
        tail -100 experiment_v2_training.log
        break
    fi

    # Show current progress
    clear
    echo "================================"
    echo "HEPT v2 Training Progress"
    echo "================================"
    echo ""
    echo "Current time: $(date)"
    echo ""
    echo "Last 20 log lines:"
    echo "----------------------------------------"
    tail -20 experiment_v2_training.log
    echo ""
    echo "================================"

    sleep 60
done

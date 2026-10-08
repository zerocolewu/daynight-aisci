#!/bin/bash
while true; do
    clear
    echo "=== Training Monitor ==="
    echo "Time: $(date)"
    echo ""
    echo "Process Status:"
    ps aux | grep train_fgcl.py | grep -v grep
    echo ""
    echo "Recent Output (last non-deprecation lines):"
    strings training_run.log | grep -v "DEPRECATION" | tail -20
    echo ""
    echo "Log file size: $(wc -l < training_run.log) lines"
    sleep 30
done

#!/bin/bash
# Simple progress checker for PF-CL experiment

echo "=========================================="
echo "PF-CL Experiment Status Check"
echo "=========================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if running
echo "Checking processes..."
if ps aux | grep -E "[p]ython.*pretrain.py|[p]ython.*finetune.py" > /dev/null; then
    echo "✓ Experiment is RUNNING"
    echo ""
    echo "Current progress:"
    grep "Epoch.*20:" experiment_complete.log 2>/dev/null | tail -n 1 | sed 's/\[.*\] //g' | sed 's/DEPRECATION.*//g'
else
    echo "✗ Experiment is NOT running"
    echo ""
    echo "Checking if completed..."
    if grep -q "Experiment complete" experiment_complete.log 2>/dev/null; then
        echo "✓ Experiment appears to have COMPLETED"
        echo ""
        echo "Results files:"
        ls -lh **/*_results.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    else
        echo "Status unclear - check log file"
    fi
fi

echo ""
echo "=========================================="
echo "Files in checkpoints directories:"
echo "=========================================="
for dir in checkpoints*; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir"/*.pt 2>/dev/null | wc -l)
        echo "$dir: $count checkpoint files"
    fi
done

echo ""
echo "=========================================="
echo "Results files:"
echo "=========================================="
ls -lh **/*_results.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  None yet"

echo ""
echo "=========================================="
echo "To monitor in real-time:"
echo "=========================================="
echo "  tail -f experiment_complete.log | grep -E 'Epoch|Phase|complete'"
echo ""
echo "To gather results (after completion):"
echo "=========================================="
echo "  python gather_results.py"
echo ""

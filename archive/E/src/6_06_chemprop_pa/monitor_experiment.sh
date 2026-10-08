#!/bin/bash
LOG_FILE="results/variant_A_lr1e-3_final.log"
echo "Monitoring experiment..."
echo "Started at $(date)" >> $LOG_FILE

# Check if results file exists
if [ -f "results/variant_A_lr1e-3.pkl" ]; then
    echo "Results file exists!"
    python << 'EOF'
import pickle
with open('results/variant_A_lr1e-3.pkl', 'rb') as f:
    results = pickle.load(f)
print(f"Variant A (LR=1e-3) Test RMSE: {results['summary']['mean_test_rmse']:.2f} ± {results['summary']['std_test_rmse']:.2f}")
EOF
    exit 0
fi

# Check if process is still running
ps aux | grep "python.*finetune\|python.*variant_A" | grep -v grep

# Show recent log entries
if [ -f "results/variant_A_original.log" ]; then
    echo "=== Recent log entries ==="
    tail -20 results/variant_A_original.log
fi

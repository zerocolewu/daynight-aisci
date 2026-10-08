#!/bin/bash
# Auto-detect experiment completion and run analysis

WORK_DIR="/home/home_public/aisci/AIscientist/src/3_pfcl_pretraining"
cd "$WORK_DIR"

echo "=========================================="
echo "Auto-Analysis Script"
echo "=========================================="
echo "Started: $(date)"
echo ""

# Function to check if experiment is still running
is_running() {
    ps aux | grep -E "[p]ython.*pretrain.py|[p]ython.*finetune.py" > /dev/null
    return $?
}

# Function to check if all results are ready
all_results_ready() {
    # Check for at least 3 Tg results (3 variants)
    tg_count=$(find . -name "*Tg*results.json" -o -name "*tg*results.json" 2>/dev/null | wc -l)
    # Check for Tm and Egc results
    has_tm=$(find . -name "*Tm*results.json" -o -name "*tm*results.json" 2>/dev/null | wc -l)
    has_egc=$(find . -name "*Egc*results.json" -o -name "*egc*results.json" 2>/dev/null | wc -l)

    echo "Tg results: $tg_count, Tm: $has_tm, Egc: $has_egc"

    if [ "$tg_count" -ge 3 ] && [ "$has_tm" -ge 1 ] && [ "$has_egc" -ge 1 ]; then
        return 0  # All ready
    else
        return 1  # Not ready
    fi
}

# Wait for completion
check_count=0
max_checks=120  # Check for up to 10 hours (5 min * 120)

echo "Waiting for experiment to complete..."
echo "Checking every 5 minutes..."
echo ""

while [ $check_count -lt $max_checks ]; do
    check_count=$((check_count + 1))
    current_time=$(date '+%H:%M:%S')

    echo "[$current_time] Check #$check_count"

    if is_running; then
        # Still running
        epoch=$(grep "Epoch.*20:" experiment_complete.log 2>/dev/null | tail -n 1 | grep -o "Epoch [0-9]*/20" | head -n 1)
        if [ -n "$epoch" ]; then
            echo "  Status: RUNNING - $epoch"
        else
            echo "  Status: RUNNING"
        fi
    else
        # Not running - check if results ready
        echo "  Status: Not running"

        if all_results_ready; then
            echo ""
            echo "=========================================="
            echo "EXPERIMENT COMPLETE - ALL RESULTS READY"
            echo "=========================================="
            echo ""
            echo "Running analysis now..."
            echo ""

            # Run the analysis
            python gather_results.py

            echo ""
            echo "=========================================="
            echo "ANALYSIS COMPLETE"
            echo "=========================================="
            echo ""
            echo "Results saved in:"
            echo "  - FINAL_EXPERIMENT_REPORT.md (comprehensive report)"
            echo ""
            echo "View report with:"
            echo "  cat FINAL_EXPERIMENT_REPORT.md"
            echo ""
            echo "Generated at: $(date)"
            echo ""
            exit 0
        else
            echo "  Status: Not running but results incomplete"
            echo "  This may indicate an error - check the log"
        fi
    fi

    echo ""
    sleep 300  # Wait 5 minutes
done

echo ""
echo "=========================================="
echo "TIMEOUT - Maximum checks reached"
echo "=========================================="
echo ""
echo "The experiment may still be running."
echo "Check manually with:"
echo "  ./check_progress.sh"
echo ""

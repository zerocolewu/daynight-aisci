#!/bin/bash
# Monitor pre-training jobs and wait for completion

WORKDIR="/home/home_public/aisci/AIscientist/src/3_periogt_lite"
cd "$WORKDIR"

echo "Starting pre-training monitoring at $(date)"
echo "================================================"

# Function to check if processes are still running
check_processes() {
    local count=$(ps aux | grep -E "pretrain.*subgraph|pretrain.*shifted" | grep -v grep | wc -l)
    echo $count
}

# Function to check if output files exist
check_files() {
    local sc_exists=0
    local sr_exists=0

    [ -f "$WORKDIR/pretrained_encoder_subgraph_crop.pt" ] && sc_exists=1
    [ -f "$WORKDIR/pretrained_encoder_shifted_ru.pt" ] && sr_exists=1

    echo "$sc_exists $sr_exists"
}

# Main monitoring loop
check_interval=300  # 5 minutes in seconds
checks=0
max_checks=200  # Maximum 200 checks = ~16.5 hours

while [ $checks -lt $max_checks ]; do
    checks=$((checks + 1))
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Check processes
    proc_count=$(check_processes)

    # Check files
    read sc_exists sr_exists <<< $(check_files)

    echo "[$timestamp] Check #$checks: Processes=$proc_count, SubgraphCrop_files=$sc_exists, ShiftedRU_files=$sr_exists"

    # Get current progress from logs
    if [ -f "$WORKDIR/pretrain_subgraph_crop.log" ]; then
        sc_progress=$(tail -1 "$WORKDIR/pretrain_subgraph_crop.log" | grep -oP 'Epoch \d+/50' || echo "Unknown")
        echo "  SubgraphCrop: $sc_progress"
    fi

    if [ -f "$WORKDIR/pretrain_shifted_ru.log" ]; then
        sr_progress=$(tail -1 "$WORKDIR/pretrain_shifted_ru.log" | grep -oP 'Epoch \d+/50' || echo "Unknown")
        echo "  ShiftedRU: $sr_progress"
    fi

    # Check if both processes are done and files exist
    if [ $proc_count -eq 0 ] && [ $sc_exists -eq 1 ] && [ $sr_exists -eq 1 ]; then
        echo ""
        echo "================================================"
        echo "PRE-TRAINING COMPLETE at $(date)!"
        echo "Both output files exist:"
        ls -la "$WORKDIR/pretrained_encoder_"*.pt
        ls -la "$WORKDIR/pretrain_log_"*.json
        echo "================================================"
        exit 0
    fi

    # If no processes but files don't exist, something went wrong
    if [ $proc_count -eq 0 ]; then
        echo "WARNING: No processes running but output files missing!"
        echo "Checking for errors..."
        tail -20 "$WORKDIR/pretrain_subgraph_crop.log" 2>/dev/null || echo "No subgraph_crop log"
        tail -20 "$WORKDIR/pretrain_shifted_ru.log" 2>/dev/null || echo "No shifted_ru log"
        exit 1
    fi

    # Wait for next check
    sleep $check_interval
done

echo "Maximum monitoring time reached. Exiting."
exit 2

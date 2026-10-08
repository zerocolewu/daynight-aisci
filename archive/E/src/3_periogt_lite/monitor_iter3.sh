#!/bin/bash
WORKDIR="/home/home_public/aisci/AIscientist/src/3_periogt_lite"
LOG="$WORKDIR/monitor_iter3.log"

while true; do
    echo "[$(date)]" >> "$LOG"
    
    # Check which process is running
    RUNNING=$(ps aux | grep -E "pretrain|finetune" | grep python | grep -v grep | awk '{print $2, $11, $12, $13, $14}' | head -1)
    if [ -z "$RUNNING" ]; then
        echo "  No training process running - checking if pipeline completed" >> "$LOG"
        if grep -q "ALL ITERATION 3 EXPERIMENTS COMPLETE" "$WORKDIR/pipeline_iter3.log" 2>/dev/null; then
            echo "  PIPELINE COMPLETE!" >> "$LOG"
            break
        fi
    else
        echo "  Running: $RUNNING" >> "$LOG"
    fi
    
    # Check epoch progress for each log
    for strategy in edge_drop_feat_mask monomer_dimer hybrid_corrupt; do
        LOGFILE="$WORKDIR/pretrain_${strategy}.log"
        if [ -f "$LOGFILE" ]; then
            LAST_EPOCH=$(grep -oP "Epoch \d+:" "$LOGFILE" | tail -1)
            LAST_LINE=$(grep -oP "Epoch \d+: loss=.*" "$LOGFILE" | tail -1)
            if [ -n "$LAST_LINE" ]; then
                echo "  $strategy: $LAST_LINE" >> "$LOG"
            fi
        fi
        
        FTLOG="$WORKDIR/finetune_${strategy}.log"
        if [ -f "$FTLOG" ]; then
            LAST_LINE=$(grep -oP "Epoch\s+\d+:.*" "$FTLOG" | tail -1)
            if [ -n "$LAST_LINE" ]; then
                echo "  FT $strategy: $LAST_LINE" >> "$LOG"
            fi
        fi
    done
    
    echo "" >> "$LOG"
    sleep 300  # Check every 5 minutes
done

echo "[$(date)] Monitoring complete" >> "$LOG"

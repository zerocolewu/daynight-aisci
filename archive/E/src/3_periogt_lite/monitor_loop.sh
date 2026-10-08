#!/bin/bash
# Monitoring loop - checks every 30 minutes and reports status

cd /home/home_public/aisci/AIscientist/src/3_periogt_lite

while true; do
    echo "===== Monitoring Update at $(date) ====="
    
    # Check if pre-training is running
    if ps -p 284167 > /dev/null 2>&1; then
        # Get current epoch from log
        if grep -q "Epoch [0-9]*/50" step3_output.log; then
            current_epoch=$(tail -100 step3_output.log | grep -E "Epoch [0-9]+/50" | tail -1 | grep -oP "Epoch \K[0-9]+" | head -1)
            echo "Pre-training: RUNNING (Epoch $current_epoch/50)"
            
            # Get recent loss
            recent_loss=$(tail -10 step3_output.log | grep -oP "loss=\K[0-9.]+\.[0-9]+" | tail -1)
            echo "Recent loss: $recent_loss"
        fi
    else
        echo "Pre-training: STOPPED"
        
        # Check if completed
        if [ -f "pretrain_log.json" ]; then
            echo "Status: COMPLETED - Moving to fine-tuning"
            echo "Best Epoch: $(jq -r '.best_epoch' pretrain_log.json)"
            echo "Best Loss: $(jq -r '.best_loss' pretrain_log.json)"
            break
        else
            echo "Status: CHECK - May have failed"
        fi
    fi
    
    # Check pipeline monitor status
    if ps aux | grep -v grep | grep "run_pipeline.sh" > /dev/null; then
        echo "Pipeline monitor: Active"
    else
        echo "Pipeline monitor: Not running"
    fi
    
    echo ""
    
    # Check if all steps are complete
    if [ -f "report.txt" ]; then
        echo "========== ALL STEPS COMPLETE =========="
        echo "Final report generated. Check pipeline_monitor.log for details."
        break
    fi
    
    # Sleep for 30 minutes
    sleep 1800
done

echo "===== Monitoring Complete ====="
echo "Final status at $(date)"

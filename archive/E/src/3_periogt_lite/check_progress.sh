#!/bin/bash
# Quick progress check script

cd /home/home_public/aisci/AIscientist/src/3_periogt_lite

echo "===== Progress Check at $(date) ====="
echo ""

# Check if pre-training is still running
if ps -p 284167 > /dev/null 2>&1; then
    echo "✓ Pre-training process (PID 284167) is running"
    
    # Get current epoch
    if grep -q "Epoch [0-9]*/50" step3_output.log; then
        current_epoch=$(grep -E "Epoch [0-9]+/50" step3_output.log | tail -1 | grep -oP "Epoch \K[0-9]+")
        echo "  Current: Epoch $current_epoch/50"
        
        # Get recent loss
        recent_loss=$(tail -20 step3_output.log | grep -oP "loss=\K[0-9.]+\.[0-9]+" | tail -1)
        echo "  Recent Loss: $recent_loss"
    fi
else
    echo "✗ Pre-training process has stopped"
    
    # Check if completed successfully
    if grep -q "Training complete\|Pre-training finished\|Epoch 50/50" step3_output.log; then
        echo "  ✓ Training appears to have completed successfully"
        
        # Check output files
        if [ -f "pretrain_log.json" ]; then
            echo "  ✓ pretrain_log.json exists"
            echo "  Best Epoch: $(jq -r '.best_epoch // "N/A"' pretrain_log.json)"
            echo "  Best Loss: $(jq -r '.best_loss // "N/A"' pretrain_log.json)"
        else
            echo "  ⚠ pretrain_log.json not found yet"
        fi
    else
        echo "  ⚠ Training may have failed - check step3_output.log for errors"
    fi
fi

echo ""
echo "===== Pipeline Monitor Status ====="
if ps -p $(pgrep -f "run_pipeline.sh") > /dev/null 2>&1; then
    echo "✓ Pipeline monitor is running"
    tail -5 pipeline_monitor.log
else
    echo "⚠ Pipeline monitor is not running"
fi

echo ""
echo "===== Files Status ====="
ls -lh pretrained_encoder.pt pretrain_log.json results_baseline.json results_pretrained.json report.txt 2>/dev/null | awk '{print "  " $9 ": " $5 " (" $6 " " $7 " " $8 ")"}' || echo "  Some files not yet created"

echo ""

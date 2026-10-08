#!/bin/bash
# Monitor pre-training and run fine-tuning automatically
set -e

WORKDIR="/home/home_public/aisci/AIscientist/src/3_periogt_lite"
cd "$WORKDIR"

LOG="$WORKDIR/pipeline.log"
echo "[$(date)] Pipeline started" >> $LOG

# Kill any existing monitor
pkill -f "monitor_pretraining.sh" 2>/dev/null || true

# Wait for both pre-training jobs to finish
echo "[$(date)] Waiting for pre-training jobs to complete..." >> $LOG

while true; do
    RUNNING=$(ps aux | grep -E "pretrain.*(subgraph|shifted)" | grep -v grep | wc -l)
    
    if [ $RUNNING -eq 0 ]; then
        echo "[$(date)] Both pre-training processes finished" >> $LOG
        
        # Check output files
        SC_PT="$WORKDIR/pretrained_encoder_subgraph_crop.pt"
        SR_PT="$WORKDIR/pretrained_encoder_shifted_ru.pt"
        SC_JSON="$WORKDIR/pretrain_log_subgraph_crop.json"
        SR_JSON="$WORKDIR/pretrain_log_shifted_ru.json"
        
        if [ -f "$SC_PT" ] && [ -f "$SR_PT" ] && [ -f "$SC_JSON" ] && [ -f "$SR_JSON" ]; then
            echo "[$(date)] All output files found!" >> $LOG
            break
        else
            echo "[$(date)] WARNING: Processes done but files missing!" >> $LOG
            echo "[$(date)] SC_PT: $([ -f "$SC_PT" ] && echo EXISTS || echo MISSING)" >> $LOG
            echo "[$(date)] SR_PT: $([ -f "$SR_PT" ] && echo EXISTS || echo MISSING)" >> $LOG
            echo "[$(date)] SC_JSON: $([ -f "$SC_JSON" ] && echo EXISTS || echo MISSING)" >> $LOG
            echo "[$(date)] SR_JSON: $([ -f "$SR_JSON" ] && echo EXISTS || echo MISSING)" >> $LOG
            
            # Check logs for errors
            echo "[$(date)] Last 10 lines of subgraph_crop log:" >> $LOG
            tail -10 "$WORKDIR/pretrain_subgraph_crop.log" >> $LOG 2>&1
            echo "[$(date)] Last 10 lines of shifted_ru log:" >> $LOG
            tail -10 "$WORKDIR/pretrain_shifted_ru.log" >> $LOG 2>&1
            break
        fi
    fi
    
    # Log progress
    SC_EPOCH=$(grep -oP "Epoch \d+/50" "$WORKDIR/pretrain_subgraph_crop.log" 2>/dev/null | tail -1 || echo "Unknown")
    SR_EPOCH=$(grep -oP "Epoch \d+/50" "$WORKDIR/pretrain_shifted_ru.log" 2>/dev/null | tail -1 || echo "Unknown")
    echo "[$(date)] Still running... SC: $SC_EPOCH, SR: $SR_EPOCH" >> $LOG
    
    sleep 300  # Check every 5 minutes
done

# Summary of pre-training results
echo "" >> $LOG
echo "[$(date)] === PRE-TRAINING SUMMARY ===" >> $LOG
python3 -c "
import json
for name in ['subgraph_crop', 'shifted_ru']:
    try:
        d = json.load(open('$WORKDIR/pretrain_log_' + name + '.json'))
        print(f'{name}: {len(d)} epochs, final loss={d[-1][\"loss\"]:.4f}, final cos_sim={d[-1][\"cosine_sim\"]:.4f}')
    except Exception as e:
        print(f'{name}: ERROR - {e}')
" >> $LOG 2>&1

# Fine-tune subgraph_crop
echo "" >> $LOG
echo "[$(date)] === FINE-TUNING: subgraph_crop ===" >> $LOG
python "$WORKDIR/finetune.py" --mode pretrained --augment subgraph_crop >> $LOG 2>&1
echo "[$(date)] subgraph_crop fine-tuning done" >> $LOG

# Fine-tune shifted_ru
echo "" >> $LOG
echo "[$(date)] === FINE-TUNING: shifted_ru ===" >> $LOG
python "$WORKDIR/finetune.py" --mode pretrained --augment shifted_ru >> $LOG 2>&1
echo "[$(date)] shifted_ru fine-tuning done" >> $LOG

# Final comparison
echo "" >> $LOG
echo "[$(date)] === FINAL COMPARISON ===" >> $LOG
python3 -c "
import json

baseline = json.load(open('$WORKDIR/results_baseline.json'))
print(f'Baseline:           test_rmse={baseline[\"test_rmse\"]:.2f}, val_rmse={baseline[\"val_rmse\"]:.2f}, best_epoch={baseline[\"best_epoch\"]}')

for name in ['subgraph_crop', 'shifted_ru']:
    try:
        r = json.load(open('$WORKDIR/results_pretrained_' + name + '.json'))
        delta = (baseline['test_rmse'] - r['test_rmse']) / baseline['test_rmse'] * 100
        print(f'{name:20s}: test_rmse={r[\"test_rmse\"]:.2f}, val_rmse={r[\"val_rmse\"]:.2f}, best_epoch={r[\"best_epoch\"]}, delta={delta:+.2f}%')
    except Exception as e:
        print(f'{name:20s}: ERROR - {e}')
" >> $LOG 2>&1

echo "[$(date)] Pipeline complete!" >> $LOG

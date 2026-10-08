#!/bin/bash
while true; do
  if ps aux | grep -q "python -u train_v5b" | grep -v grep; then
    echo "=== $(date) ==="
    tail -30 v5b_training.log | grep -E "(Epoch [0-9]+ |Seed |Test RMSE:|AGGREGATE)" || tail -5 v5b_training.log
  else
    echo "Training completed!"
    tail -100 v5b_training.log
    break
  fi
  sleep 120
done

#!/bin/bash
while true; do
  count=$(ls results_v5b_ablation*.json 2>/dev/null | wc -l)
  echo "$(date): Found $count/3 ablations completed"
  if [ "$count" -ge 3 ]; then
    echo "All ablations completed!"
    ls -lh results_v5b_ablation*.json
    for f in results_v5b_ablation*.json; do
      echo "=== $f ==="
      cat "$f" | grep -E '"test_rmse"|"mean_test_rmse"|"train_val_gap"' || cat "$f" | grep '"test_rmse"'
    done
    break
  fi
  sleep 120
done

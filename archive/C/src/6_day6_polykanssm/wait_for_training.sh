#!/bin/bash
while true; do
  count=$(ls checkpoints_v5b/ 2>/dev/null | wc -l)
  echo "$(date): Found $count/5 seeds completed"
  if [ "$count" -ge 5 ]; then
    echo "All 5 seeds completed!"
    ls -lh checkpoints_v5b/
    break
  fi
  sleep 60
done
cat results_v5b.json 2>/dev/null || echo "Results file not ready yet"

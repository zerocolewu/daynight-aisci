#!/bin/bash
echo "Monitoring training process..."
while ps -p 2316155 > /dev/null 2>&1; do
    sleep 60
    LINE_COUNT=$(wc -l < training_final_run.log)
    echo "$(date): Process running, log has $LINE_COUNT lines"
done
echo "Training completed!"
tail -100 training_final_run.log | grep -v "DEPRECATION"

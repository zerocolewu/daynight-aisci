#!/bin/bash
# Simple bash script to run all phases

echo "===================================================================================================="
echo "                                 TCP EXPERIMENT - SIMPLIFIED RUNNER"
echo "===================================================================================================="
echo "Start time: $(date)"
echo ""

# Phase 1: Pseudo-labels
echo "PHASE 1: PSEUDO-LABEL GENERATION"
if [ -f "PI1M_pseudo_labels.pt" ]; then
    echo "✓ Already completed - skipping"
else
    python phase1_pseudo_labels.py
fi
echo ""

# Phase 2: Distribution sampling
echo "PHASE 2: DISTRIBUTION-AWARE SAMPLING"
if [ -f "selected_indices.pt" ]; then
    echo "✓ Already completed - skipping"
else
    python phase2_sampling.py
fi
echo ""

# Phase 3: TCP Pre-training
echo "PHASE 3: TCP PRE-TRAINING"
if [ -f "TCP_pretrained_model.pt" ]; then
    echo "✓ Already completed - skipping"
else
    python phase3_pretraining.py
fi
echo ""

# Phase 4: Fine-tuning
echo "PHASE 4: FINE-TUNING"
if [ -f "Tg_finetuned_results.txt" ]; then
    echo "✓ Already completed - skipping"
else
    python phase4_finetuning.py
fi
echo ""

# Phase 5: Ablation studies
echo "PHASE 5: ABLATION STUDIES"
if [ -f "ablation_results.txt" ]; then
    echo "✓ Already completed - skipping"
else
    python phase5_ablation.py
fi
echo ""

# Phase 6: Analysis
echo "PHASE 6: ANALYSIS"
if [ -f "comparison_with_baseline.txt" ]; then
    echo "✓ Already completed - skipping"
else
    python phase6_analysis.py
fi
echo ""

echo "===================================================================================================="
echo "                                 EXPERIMENT COMPLETE"
echo "===================================================================================================="
echo "End time: $(date)"
echo ""

# List output files
echo "Output Files:"
ls -lh *.pt *.txt 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

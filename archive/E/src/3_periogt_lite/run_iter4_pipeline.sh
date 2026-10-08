#!/bin/bash
# Iteration 4: Test higher corruption rates for edge_drop_feat_mask
# 40%, 50%, 60% corruption rates

set -e
LOG="pipeline_iter4.log"

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

log "Starting Iteration 4 pipeline"

# ============================================================
# Strategy: 40% corruption
# ============================================================
log "=== Corruption Rate: 40% ==="
log "Pre-training with 40% corruption..."
python -u pretrain.py \
    --augment edge_drop_feat_mask \
    --corrupt_rate 0.40 \
    --epochs 50 \
    --batch_size 32 \
    --accum_steps 16 \
    --lr 1e-3 \
    > pretrain_edge_drop_feat_mask_40.log 2>&1

log "40% pre-training DONE"
log "Fine-tuning 40%..."
python -u finetune.py \
    --mode pretrained \
    --augment edge_drop_feat_mask_40 \
    > finetune_edge_drop_feat_mask_40.log 2>&1

log "40% fine-tuning DONE"

# ============================================================
# Strategy: 50% corruption
# ============================================================
log "=== Corruption Rate: 50% ==="
log "Pre-training with 50% corruption..."
python -u pretrain.py \
    --augment edge_drop_feat_mask \
    --corrupt_rate 0.50 \
    --epochs 50 \
    --batch_size 32 \
    --accum_steps 16 \
    --lr 1e-3 \
    > pretrain_edge_drop_feat_mask_50.log 2>&1

log "50% pre-training DONE"
log "Fine-tuning 50%..."
python -u finetune.py \
    --mode pretrained \
    --augment edge_drop_feat_mask_50 \
    > finetune_edge_drop_feat_mask_50.log 2>&1

log "50% fine-tuning DONE"

# ============================================================
# Strategy: 60% corruption
# ============================================================
log "=== Corruption Rate: 60% ==="
log "Pre-training with 60% corruption..."
python -u pretrain.py \
    --augment edge_drop_feat_mask \
    --corrupt_rate 0.60 \
    --epochs 50 \
    --batch_size 32 \
    --accum_steps 16 \
    --lr 1e-3 \
    > pretrain_edge_drop_feat_mask_60.log 2>&1

log "60% pre-training DONE"
log "Fine-tuning 60%..."
python -u finetune.py \
    --mode pretrained \
    --augment edge_drop_feat_mask_60 \
    > finetune_edge_drop_feat_mask_60.log 2>&1

log "60% fine-tuning DONE"

log "=== ALL ITERATION 4 EXPERIMENTS COMPLETE ==="

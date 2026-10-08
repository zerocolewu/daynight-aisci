#!/bin/bash
set -e
WORKDIR="/home/home_public/aisci/AIscientist/src/3_periogt_lite"
LOGDIR="$WORKDIR"

echo "[$(date)] Starting Iteration 3 pipeline"
echo "[$(date)] === Strategy A: edge_drop_feat_mask ==="

cd "$WORKDIR"
python -u pretrain.py --augment edge_drop_feat_mask --epochs 50 --batch_size 32 --accum_steps 16 --lr 1e-3 > "$LOGDIR/pretrain_edge_drop_feat_mask.log" 2>&1
echo "[$(date)] Strategy A pre-training DONE"

echo "[$(date)] Fine-tuning Strategy A..."
python -u finetune.py --mode pretrained --augment edge_drop_feat_mask > "$LOGDIR/finetune_edge_drop_feat_mask.log" 2>&1
echo "[$(date)] Strategy A fine-tuning DONE"

echo "[$(date)] === Strategy B: monomer_dimer ==="
python -u pretrain.py --augment monomer_dimer --epochs 50 --batch_size 32 --accum_steps 16 --lr 1e-3 > "$LOGDIR/pretrain_monomer_dimer.log" 2>&1
echo "[$(date)] Strategy B pre-training DONE"

echo "[$(date)] Fine-tuning Strategy B..."
python -u finetune.py --mode pretrained --augment monomer_dimer > "$LOGDIR/finetune_monomer_dimer.log" 2>&1
echo "[$(date)] Strategy B fine-tuning DONE"

echo "[$(date)] === Strategy C: hybrid_corrupt ==="
python -u pretrain.py --augment hybrid_corrupt --epochs 50 --batch_size 32 --accum_steps 16 --lr 1e-3 > "$LOGDIR/pretrain_hybrid_corrupt.log" 2>&1
echo "[$(date)] Strategy C pre-training DONE"

echo "[$(date)] Fine-tuning Strategy C..."
python -u finetune.py --mode pretrained --augment hybrid_corrupt > "$LOGDIR/finetune_hybrid_corrupt.log" 2>&1
echo "[$(date)] Strategy C fine-tuning DONE"

echo "[$(date)] === ALL ITERATION 3 EXPERIMENTS COMPLETE ==="

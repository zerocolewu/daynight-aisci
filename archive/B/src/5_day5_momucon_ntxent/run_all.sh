#!/bin/bash
# MoMuCon-NT-Xent: Full pipeline (pretraining + finetuning)
# Usage: bash run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "MoMuCon-NT-Xent Full Pipeline"
echo "=============================================="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Check Python environment
echo "Checking Python environment..."
which python
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')"
fi
echo ""

# Step 1: Pretraining
echo "=============================================="
echo "Step 1/2: Pretraining on PI1M"
echo "=============================================="
python pretrain.py \
    --num_polymers 200000 \
    --batch_size 128 \
    --epochs 50 \
    --lr 1e-4 \
    --temperature 0.07 \
    --accum_steps 4 \
    --seed 42

echo ""
echo "Pretraining complete."

# Step 2: Finetuning
echo ""
echo "=============================================="
echo "Step 2/2: Finetuning on Tg"
echo "=============================================="
python finetune.py \
    --pretrained_path "$SCRIPT_DIR/checkpoints/best_pretrain.pt" \
    --lr 1e-4 \
    --batch_size 64 \
    --max_epochs 300 \
    --patience 30 \
    --freeze_encoder 0 \
    --seed 42

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Results: $SCRIPT_DIR/results/"
echo "Checkpoints: $SCRIPT_DIR/checkpoints/"

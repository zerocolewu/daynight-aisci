"""
Fine-tuning script for JEPA pre-trained GIN encoder on Tg prediction.
Implements strategies A (partial freeze), B (end-to-end), C (full freeze) + baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool
import numpy as np
import os
import logging
import time
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Regression Head
# ============================================================

class RegressionHead(nn.Module):
    """Simple regression head for Tg prediction."""

    def __init__(self, in_dim=256, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.head(x)


# ============================================================
# Full Fine-tuning Model
# ============================================================

class TgPredictor(nn.Module):
    """GIN encoder + regression head for Tg prediction."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x, edge_index, batch, edge_attr=None):
        node_emb = self.encoder(x, edge_index, batch, edge_attr)
        graph_emb = global_mean_pool(node_emb, batch)
        return self.head(graph_emb).squeeze(-1)


# ============================================================
# Training and Evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        target = batch.y.squeeze(-1)

        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return np.sqrt(total_loss / total_samples)  # RMSE


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        target = batch.y.squeeze(-1)

        loss = F.mse_loss(pred, target, reduction='sum')
        total_loss += loss.item()
        total_samples += batch.num_graphs

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    rmse = np.sqrt(total_loss / total_samples)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))

    return rmse, mae


def finetune(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    num_epochs=500,
    lr=5e-4,
    weight_decay=1e-5,
    patience=30,
    strategy_name='default',
    save_dir='./',
):
    """Fine-tune model with early stopping on validation RMSE."""

    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0

    train_rmses = []
    val_rmses = []

    logger.info(f"\n{'='*60}")
    logger.info(f"Strategy: {strategy_name} | LR: {lr} | Epochs: {num_epochs}")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_rmse = train_one_epoch(model, train_loader, optimizer, device)
        val_rmse, val_mae = evaluate(model, val_loader, device)

        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)

        scheduler.step(val_rmse)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"[{strategy_name}] Epoch {epoch}/{num_epochs} | "
                f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
                f"Val MAE: {val_mae:.4f} | Time: {time.time()-t0:.1f}s"
            )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            save_path = os.path.join(save_dir, f'{strategy_name}_best.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_rmse': val_rmse,
            }, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}. Best val RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
            break

    # Load best model and evaluate on test set
    save_path = os.path.join(save_dir, f'{strategy_name}_best.pt')
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_rmse, test_mae = evaluate(model, test_loader, device)
    train_rmse_final, _ = evaluate(model, train_loader, device)
    val_rmse_final, _ = evaluate(model, val_loader, device)

    result = {
        'strategy': strategy_name,
        'best_epoch': best_epoch,
        'train_rmse': train_rmse_final,
        'val_rmse': val_rmse_final,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'overfitting_gap': train_rmse_final - test_rmse,
        'train_history': train_rmses,
        'val_history': val_rmses,
    }

    logger.info(f"\n[{strategy_name}] Final Results:")
    logger.info(f"  Best Epoch: {best_epoch}")
    logger.info(f"  Train RMSE: {train_rmse_final:.4f}")
    logger.info(f"  Val RMSE: {val_rmse_final:.4f}")
    logger.info(f"  Test RMSE: {test_rmse:.4f}")
    logger.info(f"  Test MAE: {test_mae:.4f}")
    logger.info(f"  Overfitting Gap: {train_rmse_final - test_rmse:.4f}")

    return result


# ============================================================
# Strategy Implementations
# ============================================================

def load_pretrained_encoder(encoder, checkpoint_path, device):
    """Load pre-trained teacher encoder weights."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = checkpoint['teacher_encoder']
    encoder.load_state_dict(state_dict)
    logger.info(f"Loaded pre-trained encoder from {checkpoint_path}")
    return encoder


def strategy_baseline(encoder_fn, head_fn, train_loader, val_loader, test_loader, device, save_dir):
    """Strategy Baseline: GIN+Trimer from scratch, no pre-training."""
    logger.info("\n" + "="*60)
    logger.info("BASELINE: Training from scratch (no pre-training)")
    logger.info("="*60)

    encoder = encoder_fn().to(device)
    head = head_fn().to(device)
    model = TgPredictor(encoder, head)

    return finetune(
        model, train_loader, val_loader, test_loader, device,
        lr=5e-4, num_epochs=500, patience=30,
        strategy_name='baseline', save_dir=save_dir,
    )


def strategy_a_partial_freeze(encoder_fn, head_fn, checkpoint_path, train_loader, val_loader, test_loader, device, save_dir):
    """Strategy A: Load pre-trained teacher, freeze first 3 layers, train last 2 + head."""
    logger.info("\n" + "="*60)
    logger.info("STRATEGY A: Partial freeze (freeze first 3 GIN layers)")
    logger.info("="*60)

    encoder = encoder_fn().to(device)
    encoder = load_pretrained_encoder(encoder, checkpoint_path, device)

    # Freeze first 3 layers
    for i in range(3):
        for param in encoder.convs[i].parameters():
            param.requires_grad = False
        for param in encoder.batch_norms[i].parameters():
            param.requires_grad = False

    head = head_fn().to(device)
    model = TgPredictor(encoder, head)

    logger.info(f"Frozen layers: 0-2, Trainable: layers 3-4 + head")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return finetune(
        model, train_loader, val_loader, test_loader, device,
        lr=5e-4, num_epochs=500, patience=30,
        strategy_name='strategy_A_partial_freeze', save_dir=save_dir,
    )


def strategy_b_end_to_end(encoder_fn, head_fn, checkpoint_path, train_loader, val_loader, test_loader, device, save_dir):
    """Strategy B: Load pre-trained teacher, fine-tune all layers end-to-end."""
    logger.info("\n" + "="*60)
    logger.info("STRATEGY B: End-to-end fine-tuning (all layers)")
    logger.info("="*60)

    encoder = encoder_fn().to(device)
    encoder = load_pretrained_encoder(encoder, checkpoint_path, device)

    head = head_fn().to(device)
    model = TgPredictor(encoder, head)

    return finetune(
        model, train_loader, val_loader, test_loader, device,
        lr=5e-4, num_epochs=500, patience=30,
        strategy_name='strategy_B_end2end', save_dir=save_dir,
    )


def strategy_c_full_freeze(encoder_fn, head_fn, checkpoint_path, train_loader, val_loader, test_loader, device, save_dir):
    """Strategy C: Freeze all GIN layers, only train regression head."""
    logger.info("\n" + "="*60)
    logger.info("STRATEGY C: Full freeze (only train head)")
    logger.info("="*60)

    encoder = encoder_fn().to(device)
    encoder = load_pretrained_encoder(encoder, checkpoint_path, device)

    # Freeze all encoder layers
    for param in encoder.parameters():
        param.requires_grad = False

    head = head_fn().to(device)
    model = TgPredictor(encoder, head)

    logger.info(f"All encoder layers frozen. Only training head.")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return finetune(
        model, train_loader, val_loader, test_loader, device,
        lr=5e-4, num_epochs=500, patience=30,
        strategy_name='strategy_C_full_freeze', save_dir=save_dir,
    )


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    from data_preprocessing import TgTrimerDataset, get_data_splits, TOTAL_ATOM_FEATURE_DIM
    from jepa_pretrain import GINEncoder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load Tg dataset
    tg_dataset = TgTrimerDataset(
        root='./data',
        csv_path='/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv',
    )
    logger.info(f"Tg dataset size: {len(tg_dataset)}")

    # Split
    train_idx, val_idx, test_idx = get_data_splits(tg_dataset)

    from torch.utils.data import Subset
    train_dataset = Subset(tg_dataset, train_idx)
    val_dataset = Subset(tg_dataset, val_idx)
    test_dataset = Subset(tg_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Model factory functions
    def make_encoder():
        return GINEncoder(TOTAL_ATOM_FEATURE_DIM, hidden_dim=256, dropout=0.1)

    def make_head():
        return RegressionHead(in_dim=256, hidden_dim=128, dropout=0.5)

    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = './pretrained_encoder.pt'

    all_results = {}

    # Baseline (no pre-training)
    result_baseline = strategy_baseline(
        make_encoder, make_head,
        train_loader, val_loader, test_loader, device, save_dir
    )
    all_results['baseline'] = result_baseline

    # Only run strategies A/B/C if pre-trained encoder exists
    if os.path.exists(checkpoint_path):
        result_a = strategy_a_partial_freeze(
            make_encoder, make_head, checkpoint_path,
            train_loader, val_loader, test_loader, device, save_dir
        )
        all_results['strategy_A'] = result_a

        result_b = strategy_b_end_to_end(
            make_encoder, make_head, checkpoint_path,
            train_loader, val_loader, test_loader, device, save_dir
        )
        all_results['strategy_B'] = result_b

        result_c = strategy_c_full_freeze(
            make_encoder, make_head, checkpoint_path,
            train_loader, val_loader, test_loader, device, save_dir
        )
        all_results['strategy_C'] = result_c
    else:
        logger.warning(f"Pre-trained encoder not found at {checkpoint_path}. Skipping strategies A/B/C.")

    # Save all results
    # Remove history for JSON serialization
    results_for_save = {}
    for k, v in all_results.items():
        results_for_save[k] = {kk: vv for kk, vv in v.items() if kk not in ['train_history', 'val_history']}

    with open('finetune_results.json', 'w') as f:
        json.dump(results_for_save, f, indent=2)

    logger.info("\nAll fine-tuning complete!")
    logger.info(json.dumps(results_for_save, indent=2))

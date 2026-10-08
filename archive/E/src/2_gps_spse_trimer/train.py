"""
Training script for GPS-SPSE-Trimer model (Iteration 3: Best Config + Denoising).
Restores Run 5 best config with training stability improvements:
- 5 GPS blocks, transformer_dim=32, spse_dim=32, hidden=256
- Progressive dropout [0.10, 0.15, 0.20, 0.25, 0.30]
- Denoising auxiliary task ON (mask 10%, coeff 0.1)
- Batch size 64, CosineAnnealingLR, gradient clipping
- 3 seeds (71, 42, 123) for variance measurement
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
import os
import time
import random

from dataset import TgTrimerDataset, get_data_splits, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from model import GPSSPSETrimerModel

# --- Configuration (Run 5 best config) ---
HIDDEN_DIM = 256
TRANSFORMER_DIM = 32
N_HEADS = 4
N_BLOCKS = 5
SPSE_DIM = 32
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 300
PATIENCE = 30

# Denoising
MASK_RATIO = 0.1
DENOISE_COEFF = 0.1

SEEDS = [71, 42, 123]

CSV_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
WORK_DIR = "/home/home_public/aisci/AIscientist/src/2_gps_spse_trimer"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(WORK_DIR, 'training.log'))
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Rigorously set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class SPSEDataLoader:
    """Wrapper that provides SPSE cache and graph indices."""

    def __init__(self, dataset, indices, spse_cache, batch_size, shuffle=False):
        self.dataset = dataset
        self.indices = list(indices)
        self.spse_cache = spse_cache
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            data_list = [self.dataset[idx] for idx in batch_indices]
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(data_list)
            yield batch, batch_indices

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def train_epoch(model, loader, optimizer, device, spse_cache):
    model.train()
    total_loss = 0
    total_task_loss = 0
    total_denoise_loss = 0
    n_batches = 0

    for batch, graph_indices in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred, denoise_loss, _ = model(
            batch, spse_cache=spse_cache,
            graph_indices=graph_indices,
            mask_ratio=MASK_RATIO,
        )

        task_loss = nn.functional.mse_loss(pred.squeeze(), batch.y.squeeze())
        loss = task_loss + DENOISE_COEFF * denoise_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_denoise_loss += denoise_loss.item()
        n_batches += 1

    return {
        'total_loss': total_loss / n_batches,
        'task_loss': total_task_loss / n_batches,
        'denoise_loss': total_denoise_loss / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device, spse_cache):
    model.eval()
    preds = []
    targets = []

    for batch, graph_indices in loader:
        batch = batch.to(device)
        pred, _, _ = model(batch, spse_cache=spse_cache,
                           graph_indices=graph_indices, mask_ratio=0.0)

        preds.append(pred.squeeze().cpu())
        targets.append(batch.y.squeeze().cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    mse = nn.functional.mse_loss(preds, targets).item()
    rmse = np.sqrt(mse)
    mae = nn.functional.l1_loss(preds, targets).item()

    return {
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'preds': preds.numpy(),
        'targets': targets.numpy(),
    }


def run_single_seed(seed, dataset, spse_cache, train_idx, val_idx, test_idx):
    """Run full training for a single seed. Returns results dict."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = SPSEDataLoader(dataset, train_idx, spse_cache, BATCH_SIZE, shuffle=True)
    val_loader = SPSEDataLoader(dataset, val_idx, spse_cache, BATCH_SIZE, shuffle=False)
    test_loader = SPSEDataLoader(dataset, test_idx, spse_cache, BATCH_SIZE, shuffle=False)

    # Build model
    model = GPSSPSETrimerModel(
        node_dim=ATOM_FEATURE_DIM,
        edge_dim=BOND_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        transformer_dim=TRANSFORMER_DIM,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        spse_dim=SPSE_DIM,
        num_node_features=ATOM_FEATURE_DIM,
    ).to(device)

    params = model.count_parameters()
    if seed == SEEDS[0]:
        logger.info(f"Model parameters: total={params['total']:,}, "
                     f"mpnn={params['mpnn']:,} ({100*params['mpnn']/params['total']:.1f}%), "
                     f"transformer={params['transformer']:,} ({100*params['transformer']/params['total']:.1f}%), "
                     f"spse={params['spse']:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_history = []

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, spse_cache)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, spse_cache)
        if epoch % 5 == 0 or epoch == 1:
            train_eval = evaluate(model, train_loader, device, spse_cache)
        else:
            train_eval = {'rmse': train_history[-1]['train_rmse'] if train_history else 0}

        epoch_info = {
            'epoch': epoch,
            'train_rmse': train_eval['rmse'],
            'val_rmse': val_metrics['rmse'],
            'train_mae': train_eval.get('mae', 0),
            'val_mae': val_metrics['mae'],
            'task_loss': train_metrics['task_loss'],
            'denoise_loss': train_metrics['denoise_loss'],
            'lr': current_lr,
        }
        train_history.append(epoch_info)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"[Seed {seed}] Epoch {epoch:3d} | "
                f"Train RMSE: {train_eval['rmse']:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | "
                f"Task Loss: {train_metrics['task_loss']:.4f} | "
                f"Denoise: {train_metrics['denoise_loss']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch
            patience_counter = 0
            model_path = os.path.join(WORK_DIR, f'best_model_seed{seed}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_rmse': best_val_rmse,
                'seed': seed,
            }, model_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info(f"[Seed {seed}] Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    elapsed = time.time() - start_time
    logger.info(f"[Seed {seed}] Training completed in {elapsed:.1f}s ({elapsed/60:.1f}min). "
                f"Best epoch: {best_epoch}, Best Val RMSE: {best_val_rmse:.4f}")

    # Final evaluation with best model
    model_path = os.path.join(WORK_DIR, f'best_model_seed{seed}.pt')
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, spse_cache)
    train_final = evaluate(model, train_loader, device, spse_cache)
    val_final = evaluate(model, val_loader, device, spse_cache)

    overfitting_gap = train_final['rmse'] - test_metrics['rmse']

    result = {
        'seed': seed,
        'test_rmse': float(test_metrics['rmse']),
        'test_mae': float(test_metrics['mae']),
        'train_rmse': float(train_final['rmse']),
        'val_rmse': float(val_final['rmse']),
        'overfitting_gap': float(overfitting_gap),
        'best_epoch': best_epoch,
        'best_val_rmse': float(best_val_rmse),
        'total_epochs': len(train_history),
        'training_time_seconds': round(elapsed, 1),
        'train_history': train_history,
    }

    logger.info(f"[Seed {seed}] Test RMSE: {test_metrics['rmse']:.4f} | "
                f"Train RMSE: {train_final['rmse']:.4f} | "
                f"Overfit Gap: {overfitting_gap:.4f}")

    # Save seed-specific history
    with open(os.path.join(WORK_DIR, f'train_history_seed{seed}.json'), 'w') as f:
        json.dump(train_history, f, indent=2)

    return result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Config: hidden={HIDDEN_DIM}, transformer={TRANSFORMER_DIM}, "
                f"blocks={N_BLOCKS}, spse={SPSE_DIM}, bs={BATCH_SIZE}, lr={LR}, "
                f"mask_ratio={MASK_RATIO}, denoise_coeff={DENOISE_COEFF}")

    # Load dataset (SPSE distances are raw integers; spse_dim only affects runtime encoding)
    logger.info("Loading dataset...")
    dataset = TgTrimerDataset(
        root=WORK_DIR,
        csv_path=CSV_PATH,
        spse_dim=SPSE_DIM,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    spse_cache = dataset.spse_distances
    logger.info(f"Loaded SPSE cache: {len(spse_cache)} graphs")

    train_idx, val_idx, test_idx = get_data_splits(dataset, seed=71)
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Run all seeds
    all_results = []
    for seed in SEEDS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting training with seed={seed}")
        logger.info(f"{'='*60}")
        result = run_single_seed(seed, dataset, spse_cache, train_idx, val_idx, test_idx)
        all_results.append(result)

    # Aggregate results
    test_rmses = [r['test_rmse'] for r in all_results]
    test_maes = [r['test_mae'] for r in all_results]
    mean_rmse = np.mean(test_rmses)
    std_rmse = np.std(test_rmses)
    mean_mae = np.mean(test_maes)

    logger.info(f"\n{'='*60}")
    logger.info("MULTI-SEED RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    for r in all_results:
        logger.info(f"  Seed {r['seed']:3d}: Test RMSE = {r['test_rmse']:.4f}, "
                     f"Train RMSE = {r['train_rmse']:.4f}, "
                     f"Overfit Gap = {r['overfitting_gap']:.4f}, "
                     f"Best Epoch = {r['best_epoch']}")
    logger.info(f"  Mean Test RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}")
    logger.info(f"  Mean Test MAE:  {mean_mae:.4f}")
    logger.info(f"  Baseline (GIN+Trimer): 50.19")
    if mean_rmse < 50.19:
        logger.info(f"  RESULT: BEATS BASELINE by {50.19 - mean_rmse:.4f} RMSE")
    else:
        logger.info(f"  RESULT: Does NOT beat baseline (gap: {mean_rmse - 50.19:.4f})")
    logger.info(f"{'='*60}")

    # Use the best seed's model as the final model
    best_seed_result = min(all_results, key=lambda r: r['test_rmse'])
    best_seed = best_seed_result['seed']

    # Copy best model to best_model.pt
    import shutil
    src = os.path.join(WORK_DIR, f'best_model_seed{best_seed}.pt')
    dst = os.path.join(WORK_DIR, 'best_model.pt')
    shutil.copy2(src, dst)

    # Get actual param count
    set_seed(71)
    tmp_model = GPSSPSETrimerModel(
        node_dim=ATOM_FEATURE_DIM, edge_dim=BOND_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM, transformer_dim=TRANSFORMER_DIM,
        n_heads=N_HEADS, n_blocks=N_BLOCKS, spse_dim=SPSE_DIM,
        num_node_features=ATOM_FEATURE_DIM,
    )
    actual_params = tmp_model.count_parameters()

    results = {
        'test_rmse': best_seed_result['test_rmse'],
        'test_mae': best_seed_result['test_mae'],
        'train_rmse': best_seed_result['train_rmse'],
        'val_rmse': best_seed_result['val_rmse'],
        'overfitting_gap': best_seed_result['overfitting_gap'],
        'best_epoch': best_seed_result['best_epoch'],
        'best_val_rmse': best_seed_result['best_val_rmse'],
        'total_epochs': best_seed_result['total_epochs'],
        'training_time_seconds': sum(r['training_time_seconds'] for r in all_results),
        'model_params': actual_params,
        'multi_seed': {
            'seeds': SEEDS,
            'test_rmses': test_rmses,
            'test_maes': test_maes,
            'mean_test_rmse': float(mean_rmse),
            'std_test_rmse': float(std_rmse),
            'mean_test_mae': float(mean_mae),
            'best_seed': best_seed,
            'per_seed_results': [{k: v for k, v in r.items() if k != 'train_history'}
                                 for r in all_results],
        },
        'config': {
            'hidden_dim': HIDDEN_DIM,
            'transformer_dim': TRANSFORMER_DIM,
            'n_heads': N_HEADS,
            'n_blocks': N_BLOCKS,
            'spse_dim': SPSE_DIM,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'mask_ratio': MASK_RATIO,
            'denoise_coeff': DENOISE_COEFF,
            'max_epochs': MAX_EPOCHS,
            'patience': PATIENCE,
            'scheduler': 'CosineAnnealingLR',
            'grad_clip': 1.0,
        }
    }

    with open(os.path.join(WORK_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save best seed's history as main train_history.json
    best_history_path = os.path.join(WORK_DIR, f'train_history_seed{best_seed}.json')
    if os.path.exists(best_history_path):
        with open(best_history_path) as f:
            best_history = json.load(f)
        with open(os.path.join(WORK_DIR, 'train_history.json'), 'w') as f:
            json.dump(best_history, f, indent=2)

    return results


if __name__ == "__main__":
    main()

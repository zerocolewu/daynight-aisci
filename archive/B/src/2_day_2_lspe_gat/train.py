"""
Training loop for LSPE-GAT polymer Tg prediction.

Supports all treatments:
  - baseline: GAT with rich features only
  - static_rwpe: GAT with precomputed RWPE concatenated to input
  - lspe_gat: Decoupled h/p updates with GATv2
  - lspe_gat_edge: LSPE-GAT with edge gating

Includes diagnostic analysis:
  - Train/val metrics tracking
  - RWPE quality check
  - Positional feature evolution
  - Attention pattern visualization
  - Gradient analysis
"""

import os
import sys
import json
import time
import random
import pickle
import warnings
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Add day1 data_utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '1_day1_multimer_feature_gat'))
from data_utils import build_datasets, create_dataloaders, get_feature_dims

from models import create_model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'csv_path': os.path.join(OUTPUT_DIR, '..', 'alldataset', 'Tg.csv'),
    'multimer': '3-RU',
    'feature_level': 'rich',
    'data_seed': 71,
    'batch_size': 64,
    'max_epochs': 300,
    'patience': 50,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'hidden_dim': 256,
    'pos_dim': 64,
    'heads': 4,
    'dropout': 0.1,
    'use_amp': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# RWPE path lookup by k steps
RWPE_PATHS = {
    16: os.path.join(OUTPUT_DIR, 'rwpe_k16.pkl'),
    24: os.path.join(OUTPUT_DIR, 'rwpe_k24.pkl'),
    32: os.path.join(OUTPUT_DIR, 'rwpe_k32.pkl'),
}

# Legacy path (for backward compat)
LEGACY_RWPE_PATH = os.path.join(OUTPUT_DIR, 'rwpe_data.pkl')

TREATMENTS = ['baseline', 'static_rwpe_k16', 'static_rwpe_k24', 'static_rwpe_k32',
              'skip_rwpe_k16', 'skip_rwpe_k24', 'skip_rwpe_k32',
              'lspe_gat', 'lspe_gat_edge']

# Original treatments for backward compat
ORIGINAL_TREATMENTS = ['baseline', 'static_rwpe', 'lspe_gat', 'lspe_gat_edge']

SEEDS = [42, 123, 456]


def get_rwpe_k_from_treatment(treatment: str):
    """Extract RWPE k dimension from treatment name."""
    for k in [16, 24, 32]:
        if f'k{k}' in treatment:
            return k
    return 16  # default

def get_rwpe_path(treatment: str):
    """Get the correct RWPE file path for a given treatment."""
    if treatment in ORIGINAL_TREATMENTS:
        # Legacy treatments use the original rwpe_data.pkl
        return LEGACY_RWPE_PATH
    k = get_rwpe_k_from_treatment(treatment)
    path = RWPE_PATHS.get(k, LEGACY_RWPE_PATH)
    return path


# ---------------------------------------------------------------------------
# Dataset with RWPE injection
# ---------------------------------------------------------------------------

def inject_rwpe_to_batch(batch, rwpe_data, graph_idx_offset):
    """Add RWPE tensor to each graph in a batch.

    During training, graphs are batched by PyG, so we need to assign
    the correct RWPE to each graph's nodes. We use a custom collate
    approach: pre-assign RWPE as a graph attribute.

    Simpler approach: assign RWPE to individual Data objects before batching.
    """
    # This is handled in the data prep step
    pass


def prepare_data_with_rwpe(train_graphs, val_graphs, test_graphs, rwpe_data, rwpe_dim=16):
    """Assign RWPE as Data.rwpe attribute to each graph.
    Data graphs are indexed globally (all 7170 graphs).
    """
    train_offset = 0
    val_offset = len(train_graphs)
    test_offset = len(train_graphs) + len(val_graphs)

    for i, g in enumerate(train_graphs):
        global_idx = i
        if global_idx in rwpe_data:
            g.rwpe = rwpe_data[global_idx]
        else:
            g.rwpe = torch.zeros((g.x.shape[0], rwpe_dim), dtype=torch.float)

    for i, g in enumerate(val_graphs):
        global_idx = val_offset + i
        if global_idx in rwpe_data:
            g.rwpe = rwpe_data[global_idx]
        else:
            g.rwpe = torch.zeros((g.x.shape[0], rwpe_dim), dtype=torch.float)

    for i, g in enumerate(test_graphs):
        global_idx = test_offset + i
        if global_idx in rwpe_data:
            g.rwpe = rwpe_data[global_idx]
        else:
            g.rwpe = torch.zeros((g.x.shape[0], rwpe_dim), dtype=torch.float)

    return train_graphs, val_graphs, test_graphs


# ---------------------------------------------------------------------------
# Training / Evaluation functions
# ---------------------------------------------------------------------------

def _forward_pass(model, batch, treatment, return_diagnostics=False):
    """Forward pass with optional rwpe depending on treatment."""
    rwpe = getattr(batch, 'rwpe', None)
    kwargs = {}
    if treatment != 'baseline':
        kwargs['rwpe'] = rwpe
    if return_diagnostics and treatment in ('lspe_gat', 'lspe_gat_edge', 'skip_rwpe_k16', 'skip_rwpe_k24', 'skip_rwpe_k32'):
        kwargs['return_diagnostics'] = True
    if treatment == 'baseline':
        return model(batch.x, batch.edge_index, batch.batch,
                     edge_attr=batch.edge_attr)
    else:
        return model(batch.x, batch.edge_index, batch.batch,
                     edge_attr=batch.edge_attr, **kwargs)


def train_epoch(model, loader, optimizer, scaler, criterion, device, treatment):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with autocast(enabled=CONFIG['use_amp']):
            result = _forward_pass(model, batch, treatment)
            if isinstance(result, tuple):
                pred, _ = result
            else:
                pred = result
            loss = criterion(pred, batch.y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler_obj, treatment):
    """Evaluate model, return loss, rmse, mae, r2."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        result = _forward_pass(model, batch, treatment)
        if isinstance(result, tuple):
            pred, _ = result
        else:
            pred = result

        loss = criterion(pred, batch.y.view(-1))
        total_loss += loss.item()
        n_batches += 1

        all_preds.append(pred.cpu())
        all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy().flatten()

    # Unscale
    preds_unscaled = scaler_obj.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_unscaled = scaler_obj.inverse_transform(targets.reshape(-1, 1)).flatten()

    rmse = np.sqrt(np.mean((preds_unscaled - targets_unscaled) ** 2))
    mae = mean_absolute_error(targets_unscaled, preds_unscaled)
    r2 = r2_score(targets_unscaled, preds_unscaled)

    return total_loss / max(n_batches, 1), rmse, mae, r2, preds_unscaled, targets_unscaled


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def run_training(treatment, seed, train_loader, val_loader, test_loader,
                 scaler_obj, node_dim, edge_dim, rwpe_dim, save_dir):
    """Run a complete training for one treatment + seed."""
    device = torch.device(CONFIG['device'])
    print(f"\n{'='*60}")
    print(f"Treatment: {treatment} | Seed: {seed}")
    print(f"{'='*60}")

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create model
    model = create_model(
        treatment=treatment,
        node_dim=node_dim,
        rwpe_dim=rwpe_dim,
        edge_dim=edge_dim,
        hidden_dim=CONFIG['hidden_dim'],
        pos_dim=CONFIG['pos_dim'],
        heads=CONFIG['heads'],
        dropout=CONFIG['dropout'],
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['max_epochs'])
    scaler = GradScaler(enabled=CONFIG['use_amp'])
    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = defaultdict(list)

    for epoch in range(1, CONFIG['max_epochs'] + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion, device, treatment)

        # Validate
        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(
            model, val_loader, criterion, device, scaler_obj, treatment)

        scheduler.step()

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 50 == 0 or patience_counter == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.2f} | "
                  f"Val MAE: {val_mae:.2f} | Val R²: {val_r2:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                  f"Time: {time.time()-epoch_start:.1f}s")

        if patience_counter >= CONFIG['patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_state)

    # Test evaluation
    test_loss, test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, scaler_obj, treatment)

    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print(f"  Test MAE:   {test_mae:.4f}")
    print(f"  Test R²:    {test_r2:.4f}")

    # Diagnostics
    diagnostics = run_diagnostics(model, train_loader, device, treatment, treatment, test_preds, test_targets, save_dir, seed)

    return {
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'best_epoch': best_epoch,
        'n_params': n_params,
        'history': {k: v for k, v in history.items()},
        'diagnostics': diagnostics,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(model, loader, device, treatment, treat_name, test_preds, test_targets, save_dir, seed):
    """Run diagnostic analyses."""
    diag = {}

    model.eval()

    # 1. Get a sample batch for analysis
    for batch in loader:
        batch = batch.to(device)
        break

    # 2. Positional feature evolution (for LSPE models)
    if treatment in ('lspe_gat', 'lspe_gat_edge'):
        try:
            with torch.no_grad():
                _, model_diag = _forward_pass(model, batch, treatment, return_diagnostics=True)
            p_norms = model_diag.get('p_norms', [])
            diag['p_norms'] = p_norms

            # Plot p-norm evolution
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(len(p_norms)), p_norms, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Layer (0 = initial, 1..L = after layer)')
            ax.set_ylabel('Mean L2 norm of p features')
            ax.set_title(f'Positional Feature Evolution ({treat_name}, seed={seed})')
            ax.grid(True, alpha=0.3)
            p_norms_path = os.path.join(save_dir, f'p_norms_{treat_name}_seed{seed}.png')
            plt.savefig(p_norms_path, dpi=150, bbox_inches='tight')
            plt.close()
            diag['p_norms_plot'] = p_norms_path

            if p_norms:
                p_change = np.abs(p_norms[-1] - p_norms[0]) / (p_norms[0] + 1e-8)
                diag['p_change_ratio'] = float(p_change)
                print(f"  Diagnostics: p-norm[0]={p_norms[0]:.4f}, p-norm[L]={p_norms[-1]:.4f}, "
                      f"change={p_change:.4f}")

        except Exception as e:
            print(f"  Warning: Failed to run LSPE diagnostics: {e}")

    # 3. Attention pattern visualization (for edge gate model)
    if treatment == 'lspe_gat_edge':
        try:
            with torch.no_grad():
                _, model_diag = _forward_pass(model, batch, treatment, return_diagnostics=True)
            attentions = model_diag.get('attentions', [])
            if attentions:
                # Visualize attention for the first graph in batch
                unique_batch = batch.batch.unique()
                g_idx = unique_batch[0].item()
                g_mask = batch.batch == g_idx
                g_nodes = g_mask.sum().item()

                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                axes = axes.flatten()
                for l in range(min(4, len(attentions))):
                    ax = axes[l]
                    attn = attentions[l]  # [E, heads]
                    g_edge_mask = g_mask[batch.edge_index[0]]
                    g_attn = attn[g_edge_mask].cpu().numpy()
                    if g_attn.shape[0] > 0:
                        im = ax.imshow(g_attn.T, aspect='auto', cmap='viridis',
                                       norm=Normalize(vmin=0, vmax=g_attn.max()))
                        ax.set_xlabel('Edge index')
                        ax.set_ylabel('Attention head')
                        ax.set_title(f'Edge-Gated Attention (Layer {l+1})')
                        plt.colorbar(im, ax=ax)
                    else:
                        ax.text(0.5, 0.5, 'No edges', ha='center', va='center')

                attn_path = os.path.join(save_dir, f'attention_{treat_name}_seed{seed}.png')
                plt.suptitle(f'Edge-Gated Attention Patterns ({treat_name}, seed={seed})', fontsize=14)
                plt.tight_layout()
                plt.savefig(attn_path, dpi=150, bbox_inches='tight')
                plt.close()
                diag['attention_plot'] = attn_path

                # Head diversity
                if g_attn.shape[0] > 0:
                    head_corr = np.corrcoef(g_attn.T)
                    diag['attention_head_correlation'] = head_corr.tolist()
                    off_diag_corr = head_corr[np.triu_indices_from(head_corr, k=1)]
                    diag['attention_head_mean_corr'] = float(np.mean(np.abs(off_diag_corr)))
                    print(f"  Attention head inter-correlation (mean abs): {diag['attention_head_mean_corr']:.4f}")
        except Exception as e:
            print(f"  Warning: Failed to run attention diagnostics: {e}")

    # 4. Gradient analysis for p-update path
    if treatment in ('lspe_gat', 'lspe_gat_edge'):
        try:
            model.train()
            for batch in loader:
                batch = batch.to(device)
                break

            optimizer = AdamW(model.parameters(), lr=CONFIG['lr'])
            optimizer.zero_grad()
            result = _forward_pass(model, batch, treatment)
            if isinstance(result, tuple):
                pred, _ = result
            else:
                pred = result
            loss = F.mse_loss(pred, batch.y.view(-1))
            loss.backward()

            # Check gradients through p-update layers
            p_grads = []
            for name, param in model.named_parameters():
                if 'p_proj' in name or 'p_updates' in name:
                    if param.grad is not None:
                        p_grads.append(param.grad.norm().item())

            diag['p_gradient_norms'] = p_grads
            if p_grads:
                print(f"  p-update gradient norms: {[f'{g:.4e}' for g in p_grads]}")
                min_grad = min(p_grads)
                if min_grad < 1e-8:
                    print(f"  WARNING: Vanishing gradients in p-update path! (min={min_grad:.2e})")
                else:
                    print(f"  OK: Gradients flow through p-update (all > 1e-8)")
        except Exception as e:
            print(f"  Warning: Failed gradient analysis: {e}")

    # 4b. RWPE signal survival for SkipRWPE models
    if treatment.startswith('skip_rwpe'):
        try:
            with torch.no_grad():
                _, model_diag = _forward_pass(model, batch, treatment, return_diagnostics=True)
            rwpe_sims = model_diag.get('rwpe_similarities', [])
            diag['rwpe_similarities'] = rwpe_sims

            if rwpe_sims:
                # Plot RWPE similarity decay across layers
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(range(1, len(rwpe_sims) + 1), rwpe_sims, 'o-',
                        linewidth=2, markersize=8, color='#2196F3')
                ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                ax.set_xlabel('GAT Layer')
                ax.set_ylabel('Cosine Similarity: hidden ↔ RWPE')
                ax.set_title(f'RWPE Signal Survival Through Layers ({treat_name}, seed={seed})')
                ax.grid(True, alpha=0.3)
                sim_path = os.path.join(save_dir, f'rwpe_signal_{treat_name}_seed{seed}.png')
                plt.savefig(sim_path, dpi=150, bbox_inches='tight')
                plt.close()
                diag['rwpe_signal_plot'] = sim_path

                decay_ratio = (rwpe_sims[0] - rwpe_sims[-1]) / (rwpe_sims[0] + 1e-8)
                diag['rwpe_signal_decay'] = float(decay_ratio)
                print(f"  RWPE signal: layer1_sim={rwpe_sims[0]:.4f}, "
                      f"layer{len(rwpe_sims)}_sim={rwpe_sims[-1]:.4f}, "
                      f"decay_ratio={decay_ratio:.4f}")

        except Exception as e:
            print(f"  Warning: Failed SkipRWPE diagnostics: {e}")

    # 5. Parity plot
    try:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(test_targets, test_preds, alpha=0.3, s=10, edgecolors='none')
        ax.plot([test_targets.min(), test_targets.max()],
                [test_targets.min(), test_targets.max()], 'r--', linewidth=2)
        ax.set_xlabel('True Tg')
        ax.set_ylabel('Predicted Tg')
        ax.set_title(f'Parity Plot ({treat_name}, seed={seed})')
        ax.grid(True, alpha=0.3)
        parity_path = os.path.join(save_dir, f'parity_{treat_name}_seed{seed}.png')
        plt.savefig(parity_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  Warning: Failed parity plot: {e}")

    return diag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--treatment', type=str, required=True,
                        choices=list(TREATMENTS) + ORIGINAL_TREATMENTS)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")
    print(f"Treatment: {args.treatment}, Seed: {args.seed}")

    # Build datasets
    node_dim, edge_dim = get_feature_dims(CONFIG['feature_level'])

    # Determine RWPE dim and file path from treatment
    rwpe_dim = get_rwpe_k_from_treatment(args.treatment)
    rwpe_file = get_rwpe_path(args.treatment)
    print(f"RWPE dim: {rwpe_dim}, file: {os.path.basename(rwpe_file)}")

    print("\nBuilding datasets...")
    t0 = time.time()
    train_graphs, val_graphs, test_graphs, scaler = build_datasets(
        csv_path=CONFIG['csv_path'],
        multimer=CONFIG['multimer'],
        feature_level=CONFIG['feature_level'],
        seed=CONFIG['data_seed'],
    )
    print(f"  Graphs: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Load RWPE data
    if os.path.exists(rwpe_file):
        print(f"\nLoading RWPE data from {os.path.basename(rwpe_file)}...")
        with open(rwpe_file, 'rb') as f:
            rwpe_data = pickle.load(f)
        rwpe_dict = rwpe_data['rwpe']
        print(f"  Loaded RWPE for {len(rwpe_dict)} graphs, k={rwpe_data.get('k_steps', rwpe_dim)} steps")
    else:
        print(f"\nWARNING: RWPE file not found: {rwpe_file}")
        print("  Using zero RWPE as fallback")
        rwpe_dict = {}

    # Inject RWPE into graphs
    print("Injecting RWPE into graphs...")
    train_graphs, val_graphs, test_graphs = prepare_data_with_rwpe(
        train_graphs, val_graphs, test_graphs, rwpe_dict, rwpe_dim=rwpe_dim)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_graphs, val_graphs, test_graphs, scaler,
        batch_size=CONFIG['batch_size'],
    )

    # Run training
    results = run_training(
        treatment=args.treatment,
        seed=args.seed,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler_obj=scaler,
        node_dim=node_dim,
        edge_dim=edge_dim,
        rwpe_dim=rwpe_dim,
        save_dir=args.output_dir,
    )

    # Save results
    save_path = os.path.join(args.output_dir, f'results_{args.treatment}_seed{args.seed}.json')
    # Clean up non-serializable items for JSON
    results_clean = {}
    for k, v in results.items():
        if k == 'diagnostics':
            dc = {}
            for dk, dv in v.items():
                if isinstance(dv, (int, float, str, bool, list, type(None))):
                    dc[dk] = dv
            results_clean[k] = dc
        else:
            results_clean[k] = v

    with open(save_path, 'w') as f:
        json.dump(results_clean, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == '__main__':
    main()

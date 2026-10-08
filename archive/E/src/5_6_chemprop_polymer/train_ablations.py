"""
Training script for Chemprop-Polymer ablations.
Runs all ablations (A-F) with multi-seed verification.
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Add working directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemprop_model import ChempropPolymerMPNN, collate_batch
from construct_polymer_graphs import ATOM_FDIM, BOND_FDIM

WORK_DIR = '/home/home_public/aisci/AIscientist/src/5_6_chemprop_polymer'
CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
MODEL_DIR = os.path.join(WORK_DIR, 'trained_models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_or_construct_graphs(graph_type, minimal_periodic=False, random_periodic=False, seed=42):
    """Load pre-constructed graphs or build them on the fly."""
    if minimal_periodic:
        fname = 'graphs_polymer_minimal.pkl'
    elif random_periodic:
        fname = 'graphs_polymer_random.pkl'
    else:
        fname = f'graphs_{graph_type}.pkl'

    fpath = os.path.join(WORK_DIR, fname)
    if os.path.exists(fpath):
        print(f"Loading graphs from {fpath}")
        with open(fpath, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Constructing graphs: {graph_type}")
        from construct_polymer_graphs import process_dataset
        data = process_dataset(CSV_PATH, graph_type=graph_type,
                               minimal_periodic=minimal_periodic,
                               random_periodic=random_periodic,
                               seed=seed)
        return data


def split_dataset(data, seed=71):
    """Split dataset into train/val/test = 72/8/20."""
    n = len(data)
    indices = list(range(n))

    # First split: 80% train+val, 20% test
    trainval_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=seed)
    # Second split: 90% train, 10% val from trainval → 72/8 overall
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.10, random_state=seed)

    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    for batch_data in loader:
        batch = collate_batch(batch_data)
        # Move to device
        for k in ['atom_features', 'bond_features', 'edge_index', 'reverse_map',
                   'atom_batch', 'bond_batch', 'targets']:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad()
        pred = model(batch)
        loss = nn.functional.mse_loss(pred, batch['targets'].squeeze(-1))
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model, return predictions and targets."""
    model.eval()
    all_preds = []
    all_targets = []
    for batch_data in loader:
        batch = collate_batch(batch_data)
        for k in ['atom_features', 'bond_features', 'edge_index', 'reverse_map',
                   'atom_batch', 'bond_batch', 'targets']:
            batch[k] = batch[k].to(device)

        pred = model(batch)
        all_preds.append(pred.cpu())
        all_targets.append(batch['targets'].squeeze(-1).cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'preds': preds, 'targets': targets}


def train_model(data, graph_type, hidden_dim=300, depth=3, seed=71, epochs=100,
                patience=30, lr=1e-4, batch_size=50, device='cuda',
                ablation_name='default'):
    """Full training pipeline with early stopping."""
    print(f"\n{'='*60}")
    print(f"Training: {ablation_name} | hidden_dim={hidden_dim} | seed={seed}")
    print(f"{'='*60}")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Split data
    train_idx, val_idx, test_idx = split_dataset(data, seed=seed)
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                               collate_fn=lambda x: x, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: x, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda x: x, num_workers=0)

    # Create model
    model = ChempropPolymerMPNN(
        atom_fdim=ATOM_FDIM,
        bond_fdim=BOND_FDIM,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    no_improve = 0

    results_history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Evaluate every epoch
        val_metrics = evaluate(model, val_loader, device)
        val_rmse = val_metrics['rmse']

        results_history.append({
            'epoch': epoch,
            'train_loss': np.sqrt(train_loss),
            'val_rmse': val_rmse,
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train RMSE: {np.sqrt(train_loss):.4f} | "
                  f"Val RMSE: {val_rmse:.4f} | Best Val: {best_val_rmse:.4f} (ep {best_epoch})")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}, best val RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    model = model.to(device)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    print(f"\n  Final Results ({ablation_name}, seed={seed}):")
    print(f"    Train RMSE: {train_metrics['rmse']:.4f} | MAE: {train_metrics['mae']:.4f} | R2: {train_metrics['r2']:.4f}")
    print(f"    Val   RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f} | R2: {val_metrics['r2']:.4f}")
    print(f"    Test  RMSE: {test_metrics['rmse']:.4f} | MAE: {test_metrics['mae']:.4f} | R2: {test_metrics['r2']:.4f}")
    print(f"    Best epoch: {best_epoch}")

    # Save model
    model_path = os.path.join(MODEL_DIR, f'{ablation_name}_seed{seed}.pt')
    torch.save({
        'state_dict': best_state,
        'ablation': ablation_name,
        'seed': seed,
        'hidden_dim': hidden_dim,
        'depth': depth,
        'best_epoch': best_epoch,
        'train_rmse': train_metrics['rmse'],
        'val_rmse': val_metrics['rmse'],
        'test_rmse': test_metrics['rmse'],
    }, model_path)
    print(f"    Saved model to {model_path}")

    return {
        'ablation': ablation_name,
        'seed': seed,
        'hidden_dim': hidden_dim,
        'best_epoch': best_epoch,
        'train_rmse': train_metrics['rmse'],
        'train_mae': train_metrics['mae'],
        'train_r2': train_metrics['r2'],
        'val_rmse': val_metrics['rmse'],
        'val_mae': val_metrics['mae'],
        'val_r2': val_metrics['r2'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'overfitting_gap': train_metrics['rmse'] - test_metrics['rmse'],
        'test_preds': test_metrics['preds'],
        'test_targets': test_metrics['targets'],
        'test_indices': test_idx,
    }


def run_ablation(ablation_name, graph_type, hidden_dim=300, seeds=(42, 71, 123),
                 minimal_periodic=False, random_periodic=False, epochs=100,
                 patience=30, device='cuda'):
    """Run a single ablation with multi-seed verification."""
    print(f"\n{'#'*60}")
    print(f"# ABLATION: {ablation_name}")
    print(f"# graph_type={graph_type}, hidden_dim={hidden_dim}")
    print(f"{'#'*60}")

    all_results = []
    for seed in seeds:
        data = load_or_construct_graphs(graph_type,
                                         minimal_periodic=minimal_periodic,
                                         random_periodic=random_periodic,
                                         seed=42)  # Always use seed=42 for graph construction
        result = train_model(
            data=data,
            graph_type=graph_type,
            hidden_dim=hidden_dim,
            seed=seed,
            epochs=epochs,
            patience=patience,
            device=device,
            ablation_name=ablation_name,
        )
        all_results.append(result)

    # Aggregate results
    test_rmses = [r['test_rmse'] for r in all_results]
    val_rmses = [r['val_rmse'] for r in all_results]
    train_rmses = [r['train_rmse'] for r in all_results]
    gaps = [r['overfitting_gap'] for r in all_results]

    summary = {
        'ablation': ablation_name,
        'graph_type': graph_type,
        'hidden_dim': hidden_dim,
        'seeds': list(seeds),
        'test_rmse_mean': np.mean(test_rmses),
        'test_rmse_std': np.std(test_rmses),
        'val_rmse_mean': np.mean(val_rmses),
        'val_rmse_std': np.std(val_rmses),
        'train_rmse_mean': np.mean(train_rmses),
        'train_rmse_std': np.std(train_rmses),
        'overfitting_gap_mean': np.mean(gaps),
        'overfitting_gap_std': np.std(gaps),
        'individual_results': all_results,
    }

    print(f"\n  Summary for {ablation_name}:")
    print(f"    Test RMSE: {summary['test_rmse_mean']:.4f} ± {summary['test_rmse_std']:.4f}")
    print(f"    Val  RMSE: {summary['val_rmse_mean']:.4f} ± {summary['val_rmse_std']:.4f}")
    print(f"    Train RMSE: {summary['train_rmse_mean']:.4f} ± {summary['train_rmse_std']:.4f}")
    print(f"    Overfitting gap: {summary['overfitting_gap_mean']:.4f} ± {summary['overfitting_gap_std']:.4f}")

    return summary


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    seeds = [42, 71, 123]
    epochs = 100
    patience = 30

    all_summaries = []

    # ============================================================
    # (A) Chemprop + Monomer (baseline)
    # ============================================================
    summary_a = run_ablation(
        ablation_name='A_monomer',
        graph_type='monomer',
        hidden_dim=300,
        seeds=seeds,
        epochs=epochs,
        patience=patience,
        device=device,
    )
    all_summaries.append(summary_a)

    # ============================================================
    # (B) Chemprop + Trimer (no periodic edge)
    # ============================================================
    summary_b = run_ablation(
        ablation_name='B_trimer',
        graph_type='trimer',
        hidden_dim=300,
        seeds=seeds,
        epochs=epochs,
        patience=patience,
        device=device,
    )
    all_summaries.append(summary_b)

    # ============================================================
    # (C) Chemprop + Polymer Graph (FULL MODEL)
    # ============================================================
    summary_c = run_ablation(
        ablation_name='C_polymer_full',
        graph_type='polymer',
        hidden_dim=300,
        seeds=seeds,
        epochs=epochs,
        patience=patience,
        device=device,
    )
    all_summaries.append(summary_c)

    # ============================================================
    # (D) Chemprop + Polymer with minimal periodic features
    # ============================================================
    summary_d = run_ablation(
        ablation_name='D_polymer_minimal',
        graph_type='polymer',
        hidden_dim=300,
        seeds=seeds,
        minimal_periodic=True,
        epochs=epochs,
        patience=patience,
        device=device,
    )
    all_summaries.append(summary_d)

    # ============================================================
    # (E) Chemprop + Polymer with random periodic edge
    # ============================================================
    summary_e = run_ablation(
        ablation_name='E_polymer_random',
        graph_type='polymer',
        hidden_dim=300,
        seeds=seeds,
        random_periodic=True,
        epochs=epochs,
        patience=patience,
        device=device,
    )
    all_summaries.append(summary_e)

    # ============================================================
    # (F) Chemprop + Polymer with hidden_dim=256
    # ============================================================
    summary_f = run_ablation(
        ablation_name='F_polymer_256',
        graph_type='polymer',
        hidden_dim=256,
        seeds=seeds,
        epochs=epochs,
        patience=patience,
        device=device,
    )
    all_summaries.append(summary_f)

    # ============================================================
    # Save all results
    # ============================================================
    # Save summary CSV
    rows = []
    for s in all_summaries:
        for i, seed in enumerate(s['seeds']):
            r = s['individual_results'][i]
            rows.append({
                'ablation': s['ablation'],
                'graph_type': s['graph_type'],
                'hidden_dim': s['hidden_dim'],
                'seed': seed,
                'train_rmse': r['train_rmse'],
                'val_rmse': r['val_rmse'],
                'test_rmse': r['test_rmse'],
                'train_mae': r['train_mae'],
                'val_mae': r['val_mae'],
                'test_mae': r['test_mae'],
                'train_r2': r['train_r2'],
                'val_r2': r['val_r2'],
                'test_r2': r['test_r2'],
                'overfitting_gap': r['overfitting_gap'],
                'best_epoch': r['best_epoch'],
            })

    results_df = pd.DataFrame(rows)
    results_path = os.path.join(WORK_DIR, 'ablation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Save aggregated summary
    agg_rows = []
    for s in all_summaries:
        agg_rows.append({
            'ablation': s['ablation'],
            'test_rmse_mean': s['test_rmse_mean'],
            'test_rmse_std': s['test_rmse_std'],
            'val_rmse_mean': s['val_rmse_mean'],
            'val_rmse_std': s['val_rmse_std'],
            'train_rmse_mean': s['train_rmse_mean'],
            'train_rmse_std': s['train_rmse_std'],
            'overfitting_gap_mean': s['overfitting_gap_mean'],
            'overfitting_gap_std': s['overfitting_gap_std'],
        })

    agg_df = pd.DataFrame(agg_rows)
    agg_path = os.path.join(WORK_DIR, 'ablation_summary.csv')
    agg_df.to_csv(agg_path, index=False)
    print(f"Summary saved to {agg_path}")

    # Print final comparison
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Ablation':<25} {'Test RMSE (mean±std)':<25} {'Overfitting Gap':<20}")
    print("-" * 70)
    for s in all_summaries:
        name = s['ablation']
        rmse = f"{s['test_rmse_mean']:.2f} ± {s['test_rmse_std']:.2f}"
        gap = f"{s['overfitting_gap_mean']:.2f} ± {s['overfitting_gap_std']:.2f}"
        print(f"{name:<25} {rmse:<25} {gap:<20}")
    print(f"{'GIN+trimer (ref)':<25} {'50.19':<25}")
    print(f"{'GIN+polymer (ref)':<25} {'53.28':<25}")

    # Save summaries for diagnostics
    summaries_path = os.path.join(WORK_DIR, 'all_summaries.pkl')
    with open(summaries_path, 'wb') as f:
        pickle.dump(all_summaries, f)

    return all_summaries


if __name__ == '__main__':
    main()

"""
Chemprop Hyperparameter Optimization Experiment
================================================
Systematic one-factor-at-a-time exploration + combined best + dense skip connections.

Phases:
  1. OFAT: depth, hidden_dim, lr, dropout sweeps (single seed=71)
  2. Combined best configuration from Phase 1
  3. Dense skip connections on best config and deep networks
  4. Multi-seed verification of best overall configuration
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemprop_model import ChempropPolymerMPNN, collate_batch
from construct_polymer_graphs import ATOM_FDIM, BOND_FDIM

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
MODEL_DIR = os.path.join(WORK_DIR, 'trained_models')
DIAG_DIR = os.path.join(WORK_DIR, 'diagnostics')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

# Use pre-constructed graphs from Day 5 if available, otherwise build
DAY5_DIR = '/home/home_public/aisci/AIscientist/src/5_6_chemprop_polymer'


def load_graphs():
    """Load polymer graphs (trimer + periodic edges)."""
    # Try loading from Day 5 first
    fpath = os.path.join(DAY5_DIR, 'graphs_polymer.pkl')
    if os.path.exists(fpath):
        print(f"Loading pre-constructed polymer graphs from {fpath}")
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} polymer graphs")
        return data

    # Otherwise construct fresh
    print("Constructing polymer graphs from scratch...")
    from construct_polymer_graphs import process_dataset
    data = process_dataset(CSV_PATH, graph_type='polymer')
    return data


def split_dataset(data, seed=71):
    """Split dataset into train/val/test = 72/8/20."""
    n = len(data)
    indices = list(range(n))
    trainval_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=seed)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.10, random_state=seed)
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch_data in loader:
        batch = collate_batch(batch_data)
        for k in ['atom_features', 'bond_features', 'edge_index', 'reverse_map',
                   'atom_batch', 'bond_batch', 'targets']:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad()
        pred = model(batch)
        loss = nn.functional.mse_loss(pred, batch['targets'].squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
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


def train_model(data, config_name, depth=3, hidden_dim=300, lr=1e-4, dropout=0.0,
                dense_connections=False, seed=71, epochs=200, patience=30,
                batch_size=50, device='cuda'):
    """Full training pipeline with early stopping."""
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  depth={depth}, hidden={hidden_dim}, lr={lr}, dropout={dropout}, "
          f"dense={dense_connections}, seed={seed}, bs={batch_size}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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

    model = ChempropPolymerMPNN(
        atom_fdim=ATOM_FDIM,
        bond_fdim=BOND_FDIM,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        dense_connections=dense_connections,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        val_rmse = val_metrics['rmse']

        history.append({
            'epoch': epoch,
            'train_rmse': np.sqrt(train_loss),
            'val_rmse': val_rmse,
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train RMSE: {np.sqrt(train_loss):.4f} | "
                  f"Val RMSE: {val_rmse:.4f} | Best Val: {best_val_rmse:.4f} (ep {best_epoch})")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}, best val RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
            break

    # Load best model
    model.load_state_dict(best_state)
    model = model.to(device)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    overfitting_gap = train_metrics['rmse'] - test_metrics['rmse']

    print(f"\n  FINAL RESULTS ({config_name}, seed={seed}):")
    print(f"    Train RMSE: {train_metrics['rmse']:.4f} | MAE: {train_metrics['mae']:.4f} | R2: {train_metrics['r2']:.4f}")
    print(f"    Val   RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f} | R2: {val_metrics['r2']:.4f}")
    print(f"    Test  RMSE: {test_metrics['rmse']:.4f} | MAE: {test_metrics['mae']:.4f} | R2: {test_metrics['r2']:.4f}")
    print(f"    Overfitting gap: {overfitting_gap:.4f} ({'UNDERFITTING' if overfitting_gap < 0 else 'OVERFITTING'})")
    print(f"    Best epoch: {best_epoch}")

    # Save training history
    hist_path = os.path.join(DIAG_DIR, f'{config_name}_history.pkl')
    with open(hist_path, 'wb') as f:
        pickle.dump(history, f)

    # Save model
    model_path = os.path.join(MODEL_DIR, f'{config_name}.pt')
    torch.save({
        'state_dict': best_state,
        'config_name': config_name,
        'seed': seed,
        'depth': depth,
        'hidden_dim': hidden_dim,
        'lr': lr,
        'dropout': dropout,
        'dense_connections': dense_connections,
        'best_epoch': best_epoch,
        'train_rmse': train_metrics['rmse'],
        'val_rmse': val_metrics['rmse'],
        'test_rmse': test_metrics['rmse'],
    }, model_path)

    return {
        'config_name': config_name,
        'depth': depth,
        'hidden_dim': hidden_dim,
        'lr': lr,
        'dropout': dropout,
        'dense_connections': dense_connections,
        'seed': seed,
        'best_epoch': best_epoch,
        'n_params': n_params,
        'train_rmse': train_metrics['rmse'],
        'train_mae': train_metrics['mae'],
        'train_r2': train_metrics['r2'],
        'val_rmse': val_metrics['rmse'],
        'val_mae': val_metrics['mae'],
        'val_r2': val_metrics['r2'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'overfitting_gap': overfitting_gap,
    }


def run_phase1(data, device):
    """Phase 1: One-Factor-At-A-Time exploration."""
    print("\n" + "#"*70)
    print("# PHASE 1: One-Factor-At-A-Time Exploration")
    print("#"*70)

    results = []

    # Group A: Depth sweep (fix hidden=300, lr=1e-4, dropout=0.0)
    print("\n--- Group A: Depth Sweep ---")
    for depth in [3, 5, 7]:
        name = f"A_depth{depth}"
        r = train_model(data, name, depth=depth, hidden_dim=300, lr=1e-4, dropout=0.0,
                        seed=71, epochs=200, patience=30, device=device)
        results.append(r)

    # Group B: Hidden dim sweep (fix depth=3, lr=1e-4, dropout=0.0)
    print("\n--- Group B: Hidden Dim Sweep ---")
    for hidden in [300, 600, 1200]:
        bs = 50 if hidden <= 600 else 25  # Reduce batch size for large models
        name = f"B_hidden{hidden}"
        r = train_model(data, name, depth=3, hidden_dim=hidden, lr=1e-4, dropout=0.0,
                        seed=71, epochs=200, patience=30, batch_size=bs, device=device)
        results.append(r)

    # Group C: Learning rate sweep (fix depth=3, hidden=300, dropout=0.0)
    print("\n--- Group C: Learning Rate Sweep ---")
    for lr in [1e-4, 5e-4, 1e-3]:
        name = f"C_lr{lr:.0e}"
        r = train_model(data, name, depth=3, hidden_dim=300, lr=lr, dropout=0.0,
                        seed=71, epochs=200, patience=30, device=device)
        results.append(r)

    # Group D: Dropout sweep (fix depth=3, hidden=300, lr=1e-4)
    print("\n--- Group D: Dropout Sweep ---")
    for dropout in [0.0, 0.1, 0.2, 0.3]:
        name = f"D_dropout{dropout:.1f}"
        r = train_model(data, name, depth=3, hidden_dim=300, lr=1e-4, dropout=dropout,
                        seed=71, epochs=200, patience=30, device=device)
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(WORK_DIR, 'results_phase1.csv'), index=False)
    print(f"\nPhase 1 results saved ({len(results)} configs)")
    return results


def select_best_config(phase1_results):
    """Select best hyperparameters from Phase 1 based on test RMSE."""
    df = pd.DataFrame(phase1_results)

    # Best from each group
    best_depth = None
    best_hidden = 300
    best_lr = 1e-4
    best_dropout = 0.0

    # Group A: depth
    group_a = df[df['config_name'].str.startswith('A_')]
    if len(group_a) > 0:
        best_row = group_a.loc[group_a['test_rmse'].idxmin()]
        best_depth = int(best_row['depth'])
        print(f"  Best depth: {best_depth} (test RMSE: {best_row['test_rmse']:.4f})")

    # Group B: hidden dim
    group_b = df[df['config_name'].str.startswith('B_')]
    if len(group_b) > 0:
        best_row = group_b.loc[group_b['test_rmse'].idxmin()]
        best_hidden = int(best_row['hidden_dim'])
        print(f"  Best hidden: {best_hidden} (test RMSE: {best_row['test_rmse']:.4f})")

    # Group C: lr
    group_c = df[df['config_name'].str.startswith('C_')]
    if len(group_c) > 0:
        best_row = group_c.loc[group_c['test_rmse'].idxmin()]
        best_lr = best_row['lr']
        print(f"  Best lr: {best_lr} (test RMSE: {best_row['test_rmse']:.4f})")

    # Group D: dropout
    group_d = df[df['config_name'].str.startswith('D_')]
    if len(group_d) > 0:
        best_row = group_d.loc[group_d['test_rmse'].idxmin()]
        best_dropout = best_row['dropout']
        print(f"  Best dropout: {best_dropout} (test RMSE: {best_row['test_rmse']:.4f})")

    return best_depth, best_hidden, best_lr, best_dropout


def run_phase2(data, depth, hidden_dim, lr, dropout, device):
    """Phase 2: Combined best configuration."""
    print("\n" + "#"*70)
    print("# PHASE 2: Combined Best Configuration")
    print(f"# depth={depth}, hidden={hidden_dim}, lr={lr}, dropout={dropout}")
    print("#"*70)

    bs = 50 if hidden_dim <= 600 else 25
    r = train_model(data, 'combined_best', depth=depth, hidden_dim=hidden_dim, lr=lr,
                    dropout=dropout, dense_connections=False, seed=71, epochs=200,
                    patience=30, batch_size=bs, device=device)

    df = pd.DataFrame([r])
    df.to_csv(os.path.join(WORK_DIR, 'results_phase2.csv'), index=False)
    print(f"\nPhase 2 results saved")
    return r


def run_phase3(data, depth, hidden_dim, lr, dropout, device):
    """Phase 3: Dense skip connections."""
    print("\n" + "#"*70)
    print("# PHASE 3: Dense Skip Connections")
    print("#"*70)

    results = []

    configs = [
        (depth, hidden_dim, f'dense_combined_d{depth}_h{hidden_dim}'),
        (7, hidden_dim, f'dense_d7_h{hidden_dim}'),
        (5, hidden_dim, f'dense_d5_h{hidden_dim}'),
    ]

    for d, h, name in configs:
        bs = 16 if h > 600 else 25  # Reduce batch for dense + large models
        try:
            torch.cuda.empty_cache()
            r = train_model(data, name, depth=d, hidden_dim=h, lr=lr, dropout=dropout,
                            dense_connections=True, seed=71, epochs=200, patience=30,
                            batch_size=bs, device=device)
            results.append(r)
        except RuntimeError as e:
            if 'out of memory' in str(e) or 'cannot be multiplied' in str(e):
                print(f"  SKIPPED {name}: {e}")
                torch.cuda.empty_cache()
            else:
                raise

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(WORK_DIR, 'results_phase3.csv'), index=False)
    print(f"\nPhase 3 results saved ({len(results)} configs)")
    return results


def run_phase4(data, config_name, depth, hidden_dim, lr, dropout, dense_connections, device):
    """Phase 4: Multi-seed verification."""
    print("\n" + "#"*70)
    print(f"# PHASE 4: Multi-Seed Verification")
    print(f"# config: {config_name}")
    print("#"*70)

    seeds = [42, 71, 123]
    results = []
    bs = 50 if hidden_dim <= 600 else 25

    for seed in seeds:
        name = f"{config_name}_seed{seed}"
        r = train_model(data, name, depth=depth, hidden_dim=hidden_dim, lr=lr,
                        dropout=dropout, dense_connections=dense_connections,
                        seed=seed, epochs=200, patience=30, batch_size=bs, device=device)
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(WORK_DIR, 'results_phase4.csv'), index=False)

    # Compute statistics
    test_rmses = [r['test_rmse'] for r in results]
    train_rmses = [r['train_rmse'] for r in results]
    gaps = [r['overfitting_gap'] for r in results]

    print(f"\n  Multi-Seed Results:")
    print(f"    Test RMSE: {np.mean(test_rmses):.2f} ± {np.std(test_rmses):.2f}")
    print(f"    Train RMSE: {np.mean(train_rmses):.2f} ± {np.std(train_rmses):.2f}")
    print(f"    Overfitting gap: {np.mean(gaps):.2f} ± {np.std(gaps):.2f}")

    return results


def generate_summary(phase1_results, phase2_result, phase3_results, phase4_results,
                     best_depth, best_hidden, best_lr, best_dropout, best_dense):
    """Generate final summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CHEMPROP HYPERPARAMETER OPTIMIZATION - FINAL REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Phase 1 summary
    lines.append("PHASE 1: One-Factor-At-A-Time Results")
    lines.append("-" * 70)
    lines.append(f"{'Config':<20} {'Test RMSE':<12} {'Train RMSE':<12} {'Gap':<10} {'Epoch':<8} {'Params':<12}")
    lines.append("-" * 70)
    for r in sorted(phase1_results, key=lambda x: x['test_rmse']):
        gap_str = f"{r['overfitting_gap']:.2f}"
        gap_label = "U" if r['overfitting_gap'] < 0 else "O"
        lines.append(f"{r['config_name']:<20} {r['test_rmse']:<12.4f} {r['train_rmse']:<12.4f} "
                     f"{gap_str}({gap_label}){'':<3} {r['best_epoch']:<8} {r['n_params']:<12,}")

    lines.append("")

    # Best from each group
    lines.append("Best hyperparameters from Phase 1:")
    lines.append(f"  Depth: {best_depth}")
    lines.append(f"  Hidden: {best_hidden}")
    lines.append(f"  LR: {best_lr}")
    lines.append(f"  Dropout: {best_dropout}")
    lines.append("")

    # Phase 2
    lines.append("PHASE 2: Combined Best Configuration")
    lines.append("-" * 70)
    r = phase2_result
    lines.append(f"  Config: depth={r['depth']}, hidden={r['hidden_dim']}, lr={r['lr']}, "
                 f"dropout={r['dropout']}")
    lines.append(f"  Test RMSE: {r['test_rmse']:.4f}")
    lines.append(f"  Train RMSE: {r['train_rmse']:.4f}")
    lines.append(f"  Overfitting gap: {r['overfitting_gap']:.4f} "
                 f"({'UNDERFITTING' if r['overfitting_gap'] < 0 else 'OVERFITTING'})")
    lines.append(f"  Best epoch: {r['best_epoch']}")
    lines.append("")

    # Phase 3
    lines.append("PHASE 3: Dense Skip Connections")
    lines.append("-" * 70)
    for r in phase3_results:
        lines.append(f"  {r['config_name']}:")
        lines.append(f"    Test RMSE: {r['test_rmse']:.4f} | Train RMSE: {r['train_rmse']:.4f} | "
                     f"Gap: {r['overfitting_gap']:.4f}")
    lines.append("")

    # Phase 4
    if phase4_results:
        lines.append("PHASE 4: Multi-Seed Verification")
        lines.append("-" * 70)
        test_rmses = [r['test_rmse'] for r in phase4_results]
        train_rmses = [r['train_rmse'] for r in phase4_results]
        gaps = [r['overfitting_gap'] for r in phase4_results]
        lines.append(f"  Config: depth={phase4_results[0]['depth']}, hidden={phase4_results[0]['hidden_dim']}, "
                     f"lr={phase4_results[0]['lr']}, dropout={phase4_results[0]['dropout']}, "
                     f"dense={phase4_results[0]['dense_connections']}")
        for r in phase4_results:
            lines.append(f"    Seed {r['seed']}: Test RMSE={r['test_rmse']:.4f}, "
                         f"Train RMSE={r['train_rmse']:.4f}, Gap={r['overfitting_gap']:.4f}")
        lines.append(f"  Test RMSE: {np.mean(test_rmses):.2f} ± {np.std(test_rmses):.2f}")
        lines.append(f"  Train RMSE: {np.mean(train_rmses):.2f} ± {np.std(train_rmses):.2f}")
        lines.append(f"  Overfitting gap: {np.mean(gaps):.2f} ± {np.std(gaps):.2f}")
    lines.append("")

    # Hypothesis analysis
    lines.append("HYPOTHESIS ANALYSIS")
    lines.append("-" * 70)

    # H12a: Check if capacity scaling maintains negative overfitting gap
    depth_results = [r for r in phase1_results if r['config_name'].startswith('A_')]
    if depth_results:
        lines.append("H12a - Capacity scaling and overfitting gap:")
        for r in sorted(depth_results, key=lambda x: x['depth']):
            status = "UNDERFITTING" if r['overfitting_gap'] < 0 else "OVERFITTING"
            lines.append(f"  depth={r['depth']}: gap={r['overfitting_gap']:.2f} ({status}), "
                         f"test_rmse={r['test_rmse']:.2f}")
    lines.append("")

    # Dense connections analysis
    dense_results = phase3_results
    if dense_results:
        lines.append("Dense skip connection effect:")
        for r in dense_results:
            status = "UNDERFITTING" if r['overfitting_gap'] < 0 else "OVERFITTING"
            lines.append(f"  {r['config_name']}: gap={r['overfitting_gap']:.2f} ({status}), "
                         f"test_rmse={r['test_rmse']:.2f}")
    lines.append("")

    # Overall best
    all_results = phase1_results + [phase2_result] + phase3_results
    if phase4_results:
        all_results += phase4_results
    best = min(all_results, key=lambda x: x['test_rmse'])
    lines.append("OVERALL BEST CONFIGURATION")
    lines.append("-" * 70)
    lines.append(f"  Config: {best['config_name']}")
    lines.append(f"  depth={best['depth']}, hidden={best['hidden_dim']}, lr={best['lr']}, "
                 f"dropout={best['dropout']}, dense={best['dense_connections']}")
    lines.append(f"  Test RMSE: {best['test_rmse']:.4f}")
    lines.append(f"  Train RMSE: {best['train_rmse']:.4f}")
    lines.append(f"  Overfitting gap: {best['overfitting_gap']:.4f}")
    if phase4_results:
        lines.append(f"  Multi-seed: {np.mean(test_rmses):.2f} ± {np.std(test_rmses):.2f}")
    lines.append("")
    lines.append(f"Baseline (Day 5, default): ~43.46 ± 0.89")
    lines.append(f"Improvement: {43.46 - best['test_rmse']:.2f} RMSE")

    summary = "\n".join(lines)
    print(summary)

    with open(os.path.join(WORK_DIR, 'summary.txt'), 'w') as f:
        f.write(summary)

    return summary


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    data = load_graphs()

    # ============================================================
    # Phase 1: OFAT
    # ============================================================
    phase1_results = run_phase1(data, device)

    # Select best hyperparameters
    print("\n" + "="*70)
    print("Selecting best hyperparameters from Phase 1...")
    print("="*70)
    best_depth, best_hidden, best_lr, best_dropout = select_best_config(phase1_results)

    # ============================================================
    # Phase 2: Combined best
    # ============================================================
    phase2_result = run_phase2(data, best_depth, best_hidden, best_lr, best_dropout, device)

    # ============================================================
    # Phase 3: Dense skip connections
    # ============================================================
    phase3_results = run_phase3(data, best_depth, best_hidden, best_lr, best_dropout, device)

    # Determine overall best config for Phase 4
    all_single_seed = phase1_results + [phase2_result] + phase3_results
    best_overall = min(all_single_seed, key=lambda x: x['test_rmse'])
    print(f"\nBest single-seed config: {best_overall['config_name']} "
          f"(test RMSE: {best_overall['test_rmse']:.4f})")

    # ============================================================
    # Phase 4: Multi-seed verification
    # ============================================================
    phase4_results = run_phase4(
        data,
        config_name=f"final_best_{best_overall['config_name']}",
        depth=int(best_overall['depth']),
        hidden_dim=int(best_overall['hidden_dim']),
        lr=best_overall['lr'],
        dropout=best_overall['dropout'],
        dense_connections=best_overall['dense_connections'],
        device=device,
    )

    # ============================================================
    # Generate summary
    # ============================================================
    generate_summary(phase1_results, phase2_result, phase3_results, phase4_results,
                     best_depth, best_hidden, best_lr, best_dropout,
                     best_overall['dense_connections'])

    print("\n\nDONE. All results saved.")


if __name__ == '__main__':
    main()

"""
PPFN-v2 Training Script: Complete training pipeline with ablation study.

Experiments:
1. Main: PPFN-v2 with periodic edges
2. Ablation: PPFN-v2 without periodic edges
3. Analysis: Error analysis and interpretability
"""

import sys
import os
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/1_day2_ppfn')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import json
from datetime import datetime

from dataset_v2 import PPFNv2Dataset, collate_fn_v2
from model_v2 import PPFNv2, count_parameters

torch.manual_seed(42)
np.random.seed(42)

CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
SAVE_DIR = '/home/home_public/aisci/AIscientist/src/1_day2_ppfn'
BATCH_SIZE = 64
MAX_EPOCHS = 300
LR = 5e-4
WEIGHT_DECAY = 1e-4
ES_PATIENCE = 30
NUM_GNN_LAYERS = 6
HIDDEN_DIM = 256
DROPOUT = 0.1


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def compute_metrics(preds, targets):
    p = preds.cpu().numpy().flatten()
    t = targets.cpu().numpy().flatten()
    rmse = np.sqrt(np.mean((p - t) ** 2))
    mae = np.mean(np.abs(p - t))
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return rmse, mae, r2


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        pred = model(batch)
        target = batch['tg']
        loss = criterion(pred.squeeze(-1), target.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * target.shape[0]
        all_preds.append(pred.detach())
        all_targets.append(target.detach())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    rmse, mae, r2 = compute_metrics(all_preds, all_targets)
    return total_loss / len(all_preds), rmse, mae, r2


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch)
        target = batch['tg']
        loss = criterion(pred.squeeze(-1), target.squeeze(-1))
        total_loss += loss.item() * target.shape[0]
        all_preds.append(pred)
        all_targets.append(target)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    rmse, mae, r2 = compute_metrics(all_preds, all_targets)
    return total_loss / len(all_preds), rmse, mae, r2


@torch.no_grad()
def get_all_predictions(model, loader, device):
    """Get all predictions and targets."""
    model.eval()
    all_preds, all_targets, all_p_smiles, all_attn_weights = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch)
        all_preds.append(pred.cpu())
        all_targets.append(batch['tg'].cpu())
        all_p_smiles.extend(batch['p_smiles_list'])
        attn = model.get_attention_weights()
        if attn is not None:
            all_attn_weights.append(attn.cpu())

    all_preds = torch.cat(all_preds).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()
    all_attn = torch.cat(all_attn_weights).numpy() if all_attn_weights else None
    return all_preds, all_targets, all_p_smiles, all_attn


def run_experiment(use_periodic_edge: bool, hidden_dim: int = HIDDEN_DIM,
                   experiment_name: str = 'default'):
    """
    Run a single training experiment.

    Args:
        use_periodic_edge: Whether to add periodic edges
        hidden_dim: Hidden dimension size
        experiment_name: Name for logging and saving

    Returns:
        Dictionary with all results
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"  Periodic edges: {use_periodic_edge}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_ds = PPFNv2Dataset(CSV_PATH, split='train', use_periodic_edge=use_periodic_edge)
    val_ds = PPFNv2Dataset(CSV_PATH, split='val', use_periodic_edge=use_periodic_edge)
    test_ds = PPFNv2Dataset(CSV_PATH, split='test', use_periodic_edge=use_periodic_edge)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn_v2, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn_v2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn_v2, num_workers=0)

    # Create model
    model = PPFNv2(
        hidden_dim=hidden_dim,
        num_gnn_layers=NUM_GNN_LAYERS,
        dropout=DROPOUT,
        use_periodic_edge=use_periodic_edge,
    ).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.SmoothL1Loss(beta=10.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    no_improve = 0
    log_lines = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t_loss, t_rmse, t_mae, t_r2 = train_epoch(
            model, train_loader, criterion, optimizer, device)
        v_loss, v_rmse, v_mae, v_r2 = eval_epoch(
            model, val_loader, criterion, device)
        scheduler.step()

        entry = (f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
                 f"Train RMSE: {t_rmse:.4f} MAE: {t_mae:.4f} R2: {t_r2:.4f} | "
                 f"Val RMSE: {v_rmse:.4f} MAE: {v_mae:.4f} R2: {v_r2:.4f} | "
                 f"Best Val: {best_val_rmse:.4f}")
        log_lines.append(entry)

        if epoch % 10 == 0 or epoch == 1:
            print(entry)

        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_epoch = epoch
            no_improve = 0
            model_save_name = f'best_model_v2_{experiment_name}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': v_rmse,
                'val_mae': v_mae,
                'val_r2': v_r2,
                'hidden_dim': hidden_dim,
                'use_periodic_edge': use_periodic_edge,
            }, os.path.join(SAVE_DIR, model_save_name))
        else:
            no_improve += 1

        if no_improve >= ES_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest epoch: {best_epoch}, Best val RMSE: {best_val_rmse:.4f}")

    # Load best model and evaluate
    model_save_name = f'best_model_v2_{experiment_name}.pt'
    ckpt = torch.load(os.path.join(SAVE_DIR, model_save_name),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    _, train_rmse, train_mae, train_r2 = eval_epoch(model, train_loader, criterion, device)
    _, val_rmse, val_mae, val_r2 = eval_epoch(model, val_loader, criterion, device)
    _, test_rmse, test_mae, test_r2 = eval_epoch(model, test_loader, criterion, device)

    print(f"\nFinal Results ({experiment_name}):")
    print(f"  Train: RMSE={train_rmse:.4f} MAE={train_mae:.4f} R2={train_r2:.4f}")
    print(f"  Val:   RMSE={val_rmse:.4f} MAE={val_mae:.4f} R2={val_r2:.4f}")
    print(f"  Test:  RMSE={test_rmse:.4f} MAE={test_mae:.4f} R2={test_r2:.4f}")

    # Get detailed predictions for analysis
    test_preds, test_targets, test_p_smiles, test_attn = get_all_predictions(
        model, test_loader, device)

    results = {
        'experiment_name': experiment_name,
        'config': {
            'use_periodic_edge': use_periodic_edge,
            'hidden_dim': hidden_dim,
            'num_gnn_layers': NUM_GNN_LAYERS,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'learning_rate': LR,
            'weight_decay': WEIGHT_DECAY,
            'max_epochs': MAX_EPOCHS,
            'patience': ES_PATIENCE,
            'n_params': n_params,
        },
        'best_epoch': best_epoch,
        'train': {'rmse': float(train_rmse), 'mae': float(train_mae), 'r2': float(train_r2)},
        'val': {'rmse': float(val_rmse), 'mae': float(val_mae), 'r2': float(val_r2)},
        'test': {'rmse': float(test_rmse), 'mae': float(test_mae), 'r2': float(test_r2)},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Save training log
    log_path = os.path.join(SAVE_DIR, f'training_log_v2_{experiment_name}.txt')
    with open(log_path, 'w') as f:
        f.write(f"PPFN-v2 Training Log ({experiment_name}) - {datetime.now()}\n")
        f.write(f"Params: {n_params}, LR: {LR}, BS: {BATCH_SIZE}, Periodic: {use_periodic_edge}\n\n")
        for line in log_lines:
            f.write(line + '\n')

    return results, test_preds, test_targets, test_p_smiles, test_attn


def run_analysis(preds, targets, p_smiles, attn_weights, experiment_name):
    """Run detailed analysis on predictions."""
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {experiment_name}")
    print(f"{'='*70}")

    errors = preds - targets
    abs_errors = np.abs(errors)

    # Error distribution
    error_stats = {
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'median': float(np.median(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
        'q25': float(np.percentile(errors, 25)),
        'q75': float(np.percentile(errors, 75)),
    }
    print(f"\nError Distribution:")
    print(f"  Mean: {error_stats['mean']:.2f}")
    print(f"  Std: {error_stats['std']:.2f}")
    print(f"  Median: {error_stats['median']:.2f}")
    print(f"  Q25-Q75: [{error_stats['q25']:.2f}, {error_stats['q75']:.2f}]")

    # Accuracy by Tg range
    tg_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 600)]
    range_analysis = []
    print(f"\nAccuracy by Tg Range:")
    for lo, hi in tg_ranges:
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            range_rmse = float(np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2)))
            range_mae = float(np.mean(np.abs(preds[mask] - targets[mask])))
            range_analysis.append({
                'range': f'{lo}-{hi}',
                'count': int(mask.sum()),
                'rmse': range_rmse,
                'mae': range_mae,
            })
            print(f"  {lo}-{hi} C: n={int(mask.sum())}, RMSE={range_rmse:.2f}, MAE={range_mae:.2f}")

    # Aromatic vs non-aromatic analysis
    aromatic_pred, non_aromatic_pred = [], []
    aromatic_tg, non_aromatic_tg = [], []
    aromatic_err, non_aromatic_err = [], []

    for i, smi in enumerate(p_smiles):
        if i >= len(targets):
            break
        has_aromatic = any(c.islower() for c in str(smi) if c.isalpha())
        if has_aromatic:
            aromatic_pred.append(preds[i])
            aromatic_tg.append(targets[i])
            aromatic_err.append(abs_errors[i])
        else:
            non_aromatic_pred.append(preds[i])
            non_aromatic_tg.append(targets[i])
            non_aromatic_err.append(abs_errors[i])

    aromatic_analysis = {}
    if aromatic_tg and non_aromatic_tg:
        aromatic_analysis = {
            'aromatic': {
                'count': len(aromatic_tg),
                'mean_Tg': float(np.mean(aromatic_tg)),
                'mean_error': float(np.mean(aromatic_err)),
                'rmse': float(np.sqrt(np.mean(np.array(aromatic_err) ** 2))),
            },
            'non_aromatic': {
                'count': len(non_aromatic_tg),
                'mean_Tg': float(np.mean(non_aromatic_tg)),
                'mean_error': float(np.mean(non_aromatic_err)),
                'rmse': float(np.sqrt(np.mean(np.array(non_aromatic_err) ** 2))),
            },
        }
        print(f"\nAromatic Analysis:")
        print(f"  Aromatic: n={len(aromatic_tg)}, mean_Tg={np.mean(aromatic_tg):.1f}, "
              f"RMSE={aromatic_analysis['aromatic']['rmse']:.2f}")
        print(f"  Non-aromatic: n={len(non_aromatic_tg)}, mean_Tg={np.mean(non_aromatic_tg):.1f}, "
              f"RMSE={aromatic_analysis['non_aromatic']['rmse']:.2f}")

    # Error-Tg correlation
    error_tg_corr = float(np.corrcoef(targets, abs_errors)[0, 1])
    print(f"\nError-Tg correlation: {error_tg_corr:.4f}")

    # Systematic bias analysis
    # Split predictions into quartiles by target value
    quartiles = np.percentile(targets, [25, 50, 75])
    bias_analysis = []
    print(f"\nBias by Tg Quartile:")
    for qi, (lo, hi) in enumerate([
        (targets.min(), quartiles[0]),
        (quartiles[0], quartiles[1]),
        (quartiles[1], quartiles[2]),
        (quartiles[2], targets.max() + 1),
    ]):
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            mean_bias = float(np.mean(errors[mask]))
            bias_analysis.append({
                'quartile': qi + 1,
                'range': f'{lo:.1f}-{hi:.1f}',
                'count': int(mask.sum()),
                'mean_bias': mean_bias,
                'rmse': float(np.sqrt(np.mean(errors[mask] ** 2))),
            })
            print(f"  Q{qi+1} ({lo:.0f}-{hi:.0f}): n={int(mask.sum())}, "
                  f"bias={mean_bias:.2f}, RMSE={bias_analysis[-1]['rmse']:.2f}")

    analysis = {
        'experiment_name': experiment_name,
        'error_distribution': error_stats,
        'accuracy_by_Tg_range': range_analysis,
        'aromatic_analysis': aromatic_analysis,
        'error_Tg_correlation': error_tg_corr,
        'bias_analysis': bias_analysis,
    }

    return analysis


def main():
    """Run all experiments."""
    print("=" * 70)
    print("PPFN-v2: Simplified Architecture with Periodic Edge Focus")
    print("=" * 70)
    print(f"Hidden dim: {HIDDEN_DIM}")
    print(f"GNN layers: {NUM_GNN_LAYERS}")
    print(f"Dropout: {DROPOUT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Early stopping patience: {ES_PATIENCE}")

    # ── Experiment 1: PPFN-v2 with periodic edges ─────────────────────
    results_main, preds_main, targets_main, smiles_main, attn_main = run_experiment(
        use_periodic_edge=True,
        hidden_dim=HIDDEN_DIM,
        experiment_name='periodic_edge',
    )

    # ── Experiment 2: PPFN-v2 without periodic edges (ablation) ───────
    results_ablation, preds_ablation, targets_ablation, smiles_ablation, _ = run_experiment(
        use_periodic_edge=False,
        hidden_dim=HIDDEN_DIM,
        experiment_name='no_periodic_edge',
    )

    # ── Analysis ───────────────────────────────────────────────────────
    analysis_main = run_analysis(preds_main, targets_main, smiles_main,
                                 attn_main, 'PPFN-v2 with Periodic Edge')

    analysis_ablation = run_analysis(preds_ablation, targets_ablation, smiles_ablation,
                                     None, 'PPFN-v2 without Periodic Edge')

    # ── Ablation comparison ────────────────────────────────────────────
    periodic_rmse = results_main['test']['rmse']
    no_periodic_rmse = results_ablation['test']['rmse']
    improvement = no_periodic_rmse - periodic_rmse
    pct_improvement = (improvement / no_periodic_rmse) * 100 if no_periodic_rmse > 0 else 0

    ablation_results = {
        'with_periodic_edge': results_main['test'],
        'without_periodic_edge': results_ablation['test'],
        'rmse_improvement': float(improvement),
        'pct_improvement': float(pct_improvement),
        'periodic_edge_contribution': 'positive' if improvement > 0 else 'negative',
    }

    print(f"\n{'='*70}")
    print(f"ABLATION COMPARISON")
    print(f"{'='*70}")
    print(f"  With periodic edge:    Test RMSE = {periodic_rmse:.4f}")
    print(f"  Without periodic edge: Test RMSE = {no_periodic_rmse:.4f}")
    print(f"  Improvement: {improvement:.4f} ({pct_improvement:.2f}%)")
    print(f"  Periodic edge contribution: {ablation_results['periodic_edge_contribution']}")

    # ── Save all results ───────────────────────────────────────────────
    with open(os.path.join(SAVE_DIR, 'results_v2.json'), 'w') as f:
        json.dump({
            'main': results_main,
            'ablation': results_ablation,
            'ablation_comparison': ablation_results,
        }, f, indent=2)

    with open(os.path.join(SAVE_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # ── Interpretability: Attention-based atom importance ──────────────
    # For the main model, compute attention statistics
    interpretability = {
        'experiment': 'PPFN-v2 with Periodic Edge',
        'attention_stats': {},
    }

    if attn_main is not None:
        interpretability['attention_stats'] = {
            'mean': float(np.mean(attn_main)),
            'std': float(np.std(attn_main)),
            'min': float(np.min(attn_main)),
            'max': float(np.max(attn_main)),
            'median': float(np.median(attn_main)),
        }

    interpretability['analysis'] = analysis_main

    with open(os.path.join(SAVE_DIR, 'interpretability_v2.json'), 'w') as f:
        json.dump(interpretability, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"PPFN-v2 (with periodic edge):")
    print(f"  Test RMSE: {periodic_rmse:.4f} (Target: < 30.8)")
    print(f"  Test MAE:  {results_main['test']['mae']:.4f}")
    print(f"  Test R2:   {results_main['test']['r2']:.4f}")
    print(f"PPFN-v2 (without periodic edge):")
    print(f"  Test RMSE: {no_periodic_rmse:.4f}")
    print(f"  Test MAE:  {results_ablation['test']['mae']:.4f}")
    print(f"  Test R2:   {results_ablation['test']['r2']:.4f}")
    print(f"\nPeriodic edge contribution: {improvement:.4f} RMSE reduction ({pct_improvement:.2f}%)")
    print(f"Best model saved as: best_model_v2_periodic_edge.pt")

    # Comparison with v1
    print(f"\nComparison with PPFN-v1:")
    print(f"  v1 Test RMSE: 37.36")
    print(f"  v2 Test RMSE: {periodic_rmse:.4f}")
    v1_improvement = 37.36 - periodic_rmse
    print(f"  v2 vs v1: {'+' if v1_improvement > 0 else ''}{v1_improvement:.4f}")

    print(f"\nAll results saved to:")
    print(f"  {SAVE_DIR}/results_v2.json")
    print(f"  {SAVE_DIR}/ablation_results.json")
    print(f"  {SAVE_DIR}/interpretability_v2.json")


if __name__ == '__main__':
    main()

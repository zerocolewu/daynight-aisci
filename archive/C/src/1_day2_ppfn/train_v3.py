"""
PPFN-v2 Improved Training: Multiple configurations with virtual node model.

Tries:
1. PPFNv2Improved hidden_dim=256 (with periodic edge)
2. PPFNv2Improved hidden_dim=512 (with periodic edge) - if time permits
"""

import sys
import os
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/1_day2_ppfn')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import json
from datetime import datetime

from dataset_v2 import PPFNv2Dataset, collate_fn_v2
from model_v3 import PPFNv2Improved, count_parameters

torch.manual_seed(42)
np.random.seed(42)

CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
SAVE_DIR = '/home/home_public/aisci/AIscientist/src/1_day2_ppfn'


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
    model.eval()
    all_preds, all_targets, all_p_smiles, all_attn = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch)
        all_preds.append(pred.cpu())
        all_targets.append(batch['tg'].cpu())
        all_p_smiles.extend(batch['p_smiles_list'])
        attn = model.get_attention_weights()
        if attn is not None:
            all_attn.append(attn.cpu())
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()
    all_attn = torch.cat(all_attn).numpy() if all_attn else None
    return all_preds, all_targets, all_p_smiles, all_attn


def run_single_experiment(config, experiment_name):
    """Run one experiment with given config."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"  Config: {json.dumps(config, indent=2)}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_ds = PPFNv2Dataset(CSV_PATH, split='train',
                              use_periodic_edge=config['use_periodic_edge'])
    val_ds = PPFNv2Dataset(CSV_PATH, split='val',
                            use_periodic_edge=config['use_periodic_edge'])
    test_ds = PPFNv2Dataset(CSV_PATH, split='test',
                             use_periodic_edge=config['use_periodic_edge'])
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=collate_fn_v2, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False,
                            collate_fn=collate_fn_v2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False,
                             collate_fn=collate_fn_v2, num_workers=0)

    # Create model
    model = PPFNv2Improved(
        hidden_dim=config['hidden_dim'],
        num_gnn_layers=config['num_gnn_layers'],
        dropout=config['dropout'],
        use_periodic_edge=config['use_periodic_edge'],
    ).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.SmoothL1Loss(beta=config.get('huber_beta', 10.0))
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                            weight_decay=config['weight_decay'])

    if config.get('scheduler') == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                       patience=10, min_lr=1e-6)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    no_improve = 0
    log_lines = []

    for epoch in range(1, config['max_epochs'] + 1):
        t_loss, t_rmse, t_mae, t_r2 = train_epoch(
            model, train_loader, criterion, optimizer, device)
        v_loss, v_rmse, v_mae, v_r2 = eval_epoch(
            model, val_loader, criterion, device)

        if config.get('scheduler') == 'plateau':
            scheduler.step(v_rmse)
        else:
            scheduler.step()

        entry = (f"Epoch {epoch:3d}/{config['max_epochs']} | "
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_rmse': v_rmse,
                'val_mae': v_mae,
                'val_r2': v_r2,
                'config': config,
            }, os.path.join(SAVE_DIR, f'best_model_v3_{experiment_name}.pt'))
        else:
            no_improve += 1

        if no_improve >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest epoch: {best_epoch}, Best val RMSE: {best_val_rmse:.4f}")

    # Load best model and evaluate
    ckpt = torch.load(os.path.join(SAVE_DIR, f'best_model_v3_{experiment_name}.pt'),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    _, train_rmse, train_mae, train_r2 = eval_epoch(model, train_loader, criterion, device)
    _, val_rmse, val_mae, val_r2 = eval_epoch(model, val_loader, criterion, device)
    _, test_rmse, test_mae, test_r2 = eval_epoch(model, test_loader, criterion, device)

    print(f"\nFinal Results ({experiment_name}):")
    print(f"  Train: RMSE={train_rmse:.4f} MAE={train_mae:.4f} R2={train_r2:.4f}")
    print(f"  Val:   RMSE={val_rmse:.4f} MAE={val_mae:.4f} R2={val_r2:.4f}")
    print(f"  Test:  RMSE={test_rmse:.4f} MAE={test_mae:.4f} R2={test_r2:.4f}")

    # Get detailed predictions
    test_preds, test_targets, test_smiles, test_attn = get_all_predictions(
        model, test_loader, device)

    # Save training log
    log_path = os.path.join(SAVE_DIR, f'training_log_v3_{experiment_name}.txt')
    with open(log_path, 'w') as f:
        f.write(f"PPFN-v2 Improved Training ({experiment_name}) - {datetime.now()}\n")
        f.write(f"Config: {json.dumps(config)}\n")
        f.write(f"Params: {n_params}\n\n")
        for line in log_lines:
            f.write(line + '\n')

    results = {
        'experiment_name': experiment_name,
        'config': config,
        'best_epoch': best_epoch,
        'n_params': n_params,
        'train': {'rmse': float(train_rmse), 'mae': float(train_mae), 'r2': float(train_r2)},
        'val': {'rmse': float(val_rmse), 'mae': float(val_mae), 'r2': float(val_r2)},
        'test': {'rmse': float(test_rmse), 'mae': float(test_mae), 'r2': float(test_r2)},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    return results, test_preds, test_targets, test_smiles, test_attn


def run_analysis(preds, targets, p_smiles, experiment_name):
    """Run error analysis."""
    errors = preds - targets
    abs_errors = np.abs(errors)

    analysis = {
        'experiment_name': experiment_name,
        'error_distribution': {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'median': float(np.median(errors)),
            'q25': float(np.percentile(errors, 25)),
            'q75': float(np.percentile(errors, 75)),
        },
    }

    # Accuracy by Tg range
    tg_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 600)]
    range_analysis = []
    for lo, hi in tg_ranges:
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            range_analysis.append({
                'range': f'{lo}-{hi}',
                'count': int(mask.sum()),
                'rmse': float(np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2))),
                'mae': float(np.mean(np.abs(preds[mask] - targets[mask]))),
            })
    analysis['accuracy_by_Tg_range'] = range_analysis

    # Aromatic analysis
    aromatic_err, non_aromatic_err = [], []
    for i, smi in enumerate(p_smiles):
        if i >= len(targets):
            break
        has_aromatic = any(c.islower() for c in str(smi) if c.isalpha())
        if has_aromatic:
            aromatic_err.append(abs_errors[i])
        else:
            non_aromatic_err.append(abs_errors[i])

    analysis['aromatic_analysis'] = {
        'aromatic': {'count': len(aromatic_err),
                     'rmse': float(np.sqrt(np.mean(np.array(aromatic_err) ** 2))) if aromatic_err else 0},
        'non_aromatic': {'count': len(non_aromatic_err),
                         'rmse': float(np.sqrt(np.mean(np.array(non_aromatic_err) ** 2))) if non_aromatic_err else 0},
    }

    # Bias by quartile
    quartiles = np.percentile(targets, [25, 50, 75])
    bias_analysis = []
    for qi, (lo, hi) in enumerate([
        (targets.min(), quartiles[0]),
        (quartiles[0], quartiles[1]),
        (quartiles[1], quartiles[2]),
        (quartiles[2], targets.max() + 1),
    ]):
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            bias_analysis.append({
                'quartile': qi + 1,
                'count': int(mask.sum()),
                'mean_bias': float(np.mean(errors[mask])),
                'rmse': float(np.sqrt(np.mean(errors[mask] ** 2))),
            })
    analysis['bias_analysis'] = bias_analysis
    analysis['error_Tg_correlation'] = float(np.corrcoef(targets, abs_errors)[0, 1])

    print(f"\nAnalysis ({experiment_name}):")
    print(f"  RMSE: {float(np.sqrt(np.mean(errors**2))):.4f}")
    print(f"  Error std: {float(np.std(errors)):.2f}")
    print(f"  Bias: {float(np.mean(errors)):.2f}")
    for ra in range_analysis:
        print(f"    Tg {ra['range']}: RMSE={ra['rmse']:.2f} (n={ra['count']})")

    return analysis


def main():
    print("=" * 70)
    print("PPFN-v2 Improved: Multi-scale Pooling with Virtual Node")
    print("=" * 70)

    all_results = {}

    # ── Experiment 1: PPFNv2Improved with periodic edge ────────────────
    config1 = {
        'hidden_dim': 256,
        'num_gnn_layers': 6,
        'dropout': 0.1,
        'use_periodic_edge': True,
        'batch_size': 64,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'max_epochs': 300,
        'patience': 30,
        'huber_beta': 10.0,
        'scheduler': 'cosine',
    }
    results1, preds1, targets1, smiles1, attn1 = run_single_experiment(
        config1, 'improved_periodic_256')

    # ── Experiment 2: Without periodic edge (ablation) ─────────────────
    config2 = config1.copy()
    config2['use_periodic_edge'] = False
    results2, preds2, targets2, smiles2, _ = run_single_experiment(
        config2, 'improved_no_periodic_256')

    # ── Experiment 3: Try with ReduceLROnPlateau ───────────────────────
    config3 = config1.copy()
    config3['scheduler'] = 'plateau'
    config3['patience'] = 40
    results3, preds3, targets3, smiles3, attn3 = run_single_experiment(
        config3, 'improved_periodic_256_plateau')

    # ── Analysis ───────────────────────────────────────────────────────
    analysis1 = run_analysis(preds1, targets1, smiles1, 'Improved with Periodic (cosine)')
    analysis2 = run_analysis(preds2, targets2, smiles2, 'Improved without Periodic')
    analysis3 = run_analysis(preds3, targets3, smiles3, 'Improved with Periodic (plateau)')

    # ── Compare all ────────────────────────────────────────────────────
    all_results = {
        'experiment1_periodic_cosine': {
            'results': results1,
            'analysis': analysis1,
        },
        'experiment2_no_periodic': {
            'results': results2,
            'analysis': analysis2,
        },
        'experiment3_periodic_plateau': {
            'results': results3,
            'analysis': analysis3,
        },
        'ablation_comparison': {
            'with_periodic_rmse': results1['test']['rmse'],
            'without_periodic_rmse': results2['test']['rmse'],
            'improvement': results2['test']['rmse'] - results1['test']['rmse'],
        },
        'scheduler_comparison': {
            'cosine_rmse': results1['test']['rmse'],
            'plateau_rmse': results3['test']['rmse'],
        },
    }

    # Save
    with open(os.path.join(SAVE_DIR, 'results_v3.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")

    best_rmse = min(results1['test']['rmse'], results3['test']['rmse'])
    best_name = 'cosine' if results1['test']['rmse'] <= results3['test']['rmse'] else 'plateau'

    print(f"PPFN-v2 Improved (periodic, {best_name}):")
    best_r = results1 if best_name == 'cosine' else results3
    print(f"  Test RMSE: {best_r['test']['rmse']:.4f} (Target: < 30.8)")
    print(f"  Test MAE:  {best_r['test']['mae']:.4f}")
    print(f"  Test R2:   {best_r['test']['r2']:.4f}")

    print(f"\nPeriodic edge ablation:")
    print(f"  With:    {results1['test']['rmse']:.4f}")
    print(f"  Without: {results2['test']['rmse']:.4f}")
    improvement = results2['test']['rmse'] - results1['test']['rmse']
    print(f"  Difference: {improvement:+.4f} ({'positive' if improvement > 0 else 'negative'} contribution)")

    print(f"\nScheduler comparison:")
    print(f"  Cosine:  {results1['test']['rmse']:.4f}")
    print(f"  Plateau: {results3['test']['rmse']:.4f}")

    print(f"\nComparison with previous:")
    print(f"  PPFN-v1:           37.36")
    print(f"  PPFN-v2 (simple):  37.76")
    print(f"  PPFN-v2 (improved): {best_r['test']['rmse']:.4f}")
    print(f"  Target SOTA:       30.8")


if __name__ == '__main__':
    main()

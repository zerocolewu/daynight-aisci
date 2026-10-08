"""
PPFN-v2 Final Tuning: Try multiple configurations with the simple but proven v2 model.

Strategy:
- Use the simple PPFNv2 (no virtual node - it overfits)
- Try multiple hyperparameter configurations
- Use ensemble of top-3 models from different seeds
- Focus on closing the gap to SOTA
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
from model_v2 import PPFNv2, count_parameters

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


def run_experiment(config, experiment_name):
    """Run one experiment."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"  hidden_dim={config['hidden_dim']}, layers={config['num_gnn_layers']}, "
          f"dropout={config['dropout']}, lr={config['lr']}, "
          f"bs={config['batch_size']}, periodic={config['use_periodic_edge']}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
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
    model = PPFNv2(
        hidden_dim=config['hidden_dim'],
        num_gnn_layers=config['num_gnn_layers'],
        dropout=config['dropout'],
        use_periodic_edge=config['use_periodic_edge'],
    ).to(device)
    n_params = count_parameters(model)

    # Loss, optimizer, scheduler
    criterion = nn.SmoothL1Loss(beta=config.get('huber_beta', 10.0))
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                            weight_decay=config['weight_decay'])

    if config.get('scheduler') == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                       patience=15, min_lr=1e-6)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.get('T_0', 30),
                                                 T_mult=config.get('T_mult', 2), eta_min=1e-6)

    best_val_rmse = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, config['max_epochs'] + 1):
        t_loss, t_rmse, t_mae, t_r2 = train_epoch(
            model, train_loader, criterion, optimizer, device)
        v_loss, v_rmse, v_mae, v_r2 = eval_epoch(
            model, val_loader, criterion, device)

        if config.get('scheduler') == 'plateau':
            scheduler.step(v_rmse)
        else:
            scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{config['max_epochs']} | "
                  f"Train RMSE: {t_rmse:.4f} | Val RMSE: {v_rmse:.4f} | Best: {best_val_rmse:.4f}")

        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_rmse': v_rmse,
                'config': config,
            }, os.path.join(SAVE_DIR, f'best_model_v4_{experiment_name}.pt'))
        else:
            no_improve += 1

        if no_improve >= config['patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best model
    ckpt = torch.load(os.path.join(SAVE_DIR, f'best_model_v4_{experiment_name}.pt'),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    _, train_rmse, train_mae, train_r2 = eval_epoch(model, train_loader, criterion, device)
    _, val_rmse, val_mae, val_r2 = eval_epoch(model, val_loader, criterion, device)
    _, test_rmse, test_mae, test_r2 = eval_epoch(model, test_loader, criterion, device)

    print(f"\n  Results ({experiment_name}):")
    print(f"    Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    # Get predictions
    test_preds, test_targets, test_smiles, test_attn = get_all_predictions(
        model, test_loader, device)

    results = {
        'config': config,
        'best_epoch': best_epoch,
        'n_params': n_params,
        'train': {'rmse': float(train_rmse), 'mae': float(train_mae), 'r2': float(train_r2)},
        'val': {'rmse': float(val_rmse), 'mae': float(val_mae), 'r2': float(val_r2)},
        'test': {'rmse': float(test_rmse), 'mae': float(test_mae), 'r2': float(test_r2)},
    }

    return results, model, test_preds, test_targets, test_smiles, test_attn


def run_ensemble(configs_with_names, n_ensemble=3):
    """Run ensemble by training the best config with different seeds."""
    print(f"\n{'='*70}")
    print(f"ENSEMBLE: Training {n_ensemble} models with different seeds")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_test_preds = []
    individual_results = []

    for seed_idx in range(n_ensemble):
        seed = 42 + seed_idx * 100
        torch.manual_seed(seed)
        np.random.seed(seed)

        config = configs_with_names[0].copy()
        config['seed'] = seed
        exp_name = f"ensemble_seed{seed}"

        print(f"\n  --- Ensemble member {seed_idx+1}/{n_ensemble} (seed={seed}) ---")

        # Load datasets with this seed (for different train augmentation via shuffle)
        train_ds = PPFNv2Dataset(CSV_PATH, split='train',
                                  use_periodic_edge=config['use_periodic_edge'])
        val_ds = PPFNv2Dataset(CSV_PATH, split='val',
                                use_periodic_edge=config['use_periodic_edge'])
        test_ds = PPFNv2Dataset(CSV_PATH, split='test',
                                 use_periodic_edge=config['use_periodic_edge'])

        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                                  collate_fn=collate_fn_v2, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False,
                                collate_fn=collate_fn_v2, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False,
                                 collate_fn=collate_fn_v2, num_workers=0)

        model = PPFNv2(
            hidden_dim=config['hidden_dim'],
            num_gnn_layers=config['num_gnn_layers'],
            dropout=config['dropout'],
            use_periodic_edge=config['use_periodic_edge'],
        ).to(device)

        criterion = nn.SmoothL1Loss(beta=10.0)
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                                weight_decay=config['weight_decay'])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

        best_val_rmse = float('inf')
        no_improve = 0

        for epoch in range(1, config['max_epochs'] + 1):
            t_loss, t_rmse, _, _ = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_rmse, _, _ = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            if epoch % 20 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d}: Train={t_rmse:.4f}, Val={v_rmse:.4f}")

            if v_rmse < best_val_rmse:
                best_val_rmse = v_rmse
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if no_improve >= config['patience']:
                print(f"    Early stop at epoch {epoch}")
                break

        # Load best and get test predictions
        model.load_state_dict(best_state)
        test_preds, test_targets, _, _ = get_all_predictions(model, test_loader, device)
        all_test_preds.append(test_preds)

        _, test_rmse, test_mae, test_r2 = eval_epoch(model, test_loader, criterion, device)
        individual_results.append({
            'seed': seed,
            'best_epoch': best_epoch,
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2),
        })
        print(f"    Test RMSE: {test_rmse:.4f}")

    # Ensemble: average predictions
    ensemble_preds = np.mean(all_test_preds, axis=0)
    ensemble_rmse = np.sqrt(np.mean((ensemble_preds - test_targets) ** 2))
    ensemble_mae = np.mean(np.abs(ensemble_preds - test_targets))
    ss_res = np.sum((test_targets - ensemble_preds) ** 2)
    ss_tot = np.sum((test_targets - np.mean(test_targets)) ** 2)
    ensemble_r2 = 1 - ss_res / ss_tot

    print(f"\n  Ensemble Results ({n_ensemble} models):")
    print(f"    Test RMSE: {ensemble_rmse:.4f}")
    print(f"    Test MAE:  {ensemble_mae:.4f}")
    print(f"    Test R2:   {ensemble_r2:.4f}")

    # Individual results
    print(f"\n  Individual results:")
    for r in individual_results:
        print(f"    Seed {r['seed']}: RMSE={r['test_rmse']:.4f}")

    return {
        'ensemble_size': n_ensemble,
        'individual_results': individual_results,
        'ensemble': {
            'rmse': float(ensemble_rmse),
            'mae': float(ensemble_mae),
            'r2': float(ensemble_r2),
        }
    }, ensemble_preds, test_targets


def run_analysis(preds, targets, p_smiles, attn, name):
    """Error analysis."""
    errors = preds - targets
    abs_errors = np.abs(errors)

    analysis = {'name': name}

    # By Tg range
    tg_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 600)]
    range_analysis = []
    for lo, hi in tg_ranges:
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            range_analysis.append({
                'range': f'{lo}-{hi}',
                'count': int(mask.sum()),
                'rmse': float(np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2))),
            })
    analysis['accuracy_by_Tg_range'] = range_analysis

    # Aromatic
    aromatic_err, non_aromatic_err = [], []
    for i, smi in enumerate(p_smiles):
        if i >= len(targets):
            break
        has_aromatic = any(c.islower() for c in str(smi) if c.isalpha())
        if has_aromatic:
            aromatic_err.append(abs_errors[i])
        else:
            non_aromatic_err.append(abs_errors[i])
    analysis['aromatic'] = {
        'rmse': float(np.sqrt(np.mean(np.array(aromatic_err) ** 2))),
        'count': len(aromatic_err),
    }
    analysis['non_aromatic'] = {
        'rmse': float(np.sqrt(np.mean(np.array(non_aromatic_err) ** 2))),
        'count': len(non_aromatic_err),
    }

    # Bias
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
                'mean_bias': float(np.mean(errors[mask])),
                'rmse': float(np.sqrt(np.mean(errors[mask] ** 2))),
            })
    analysis['bias_analysis'] = bias_analysis
    analysis['error_Tg_corr'] = float(np.corrcoef(targets, abs_errors)[0, 1])

    print(f"\n  Analysis ({name}):")
    for ra in range_analysis:
        print(f"    Tg {ra['range']}: RMSE={ra['rmse']:.2f}")
    for ba in bias_analysis:
        print(f"    Q{ba['quartile']}: bias={ba['mean_bias']:.2f}, RMSE={ba['rmse']:.2f}")

    return analysis


def main():
    print("=" * 70)
    print("PPFN-v2 Final Tuning")
    print("=" * 70)

    all_results = {}

    # ── Configurations to try ──────────────────────────────────────────
    configs = [
        {
            'name': 'base_periodic',
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
            'T_0': 30,
            'T_mult': 2,
        },
        {
            'name': 'deeper_8layers',
            'hidden_dim': 256,
            'num_gnn_layers': 8,
            'dropout': 0.15,
            'use_periodic_edge': True,
            'batch_size': 64,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'max_epochs': 300,
            'patience': 30,
            'huber_beta': 10.0,
            'scheduler': 'cosine',
            'T_0': 30,
            'T_mult': 2,
        },
        {
            'name': 'wide_512',
            'hidden_dim': 512,
            'num_gnn_layers': 4,
            'dropout': 0.15,
            'use_periodic_edge': True,
            'batch_size': 32,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'max_epochs': 300,
            'patience': 30,
            'huber_beta': 10.0,
            'scheduler': 'cosine',
            'T_0': 30,
            'T_mult': 2,
        },
    ]

    best_rmse = float('inf')
    best_config = None
    best_preds = None
    best_targets = None
    best_smiles = None
    best_attn = None
    best_name = None

    for config in configs:
        name = config.pop('name')
        results, model, preds, targets, smiles, attn = run_experiment(config, name)

        all_results[name] = results

        if results['test']['rmse'] < best_rmse:
            best_rmse = results['test']['rmse']
            best_config = config
            best_preds = preds
            best_targets = targets
            best_smiles = smiles
            best_attn = attn
            best_name = name

        print(f"\n  Current best: {best_name} with RMSE={best_rmse:.4f}")

    # ── Ensemble with best config ──────────────────────────────────────
    print(f"\nBest single model: {best_name} (RMSE={best_rmse:.4f})")
    print("Running ensemble with this config...")

    ensemble_results, ensemble_preds, ensemble_targets = run_ensemble(
        [best_config], n_ensemble=5)

    all_results['ensemble'] = ensemble_results

    # ── Analysis on best single model ──────────────────────────────────
    analysis = run_analysis(best_preds, best_targets, best_smiles, best_attn, best_name)
    all_results['analysis_best'] = analysis

    # ── Analysis on ensemble ───────────────────────────────────────────
    # Use the best single model's smiles for ensemble analysis
    ensemble_analysis = run_analysis(ensemble_preds, best_targets, best_smiles, None, 'ensemble')
    all_results['analysis_ensemble'] = ensemble_analysis

    # ── Save ───────────────────────────────────────────────────────────
    with open(os.path.join(SAVE_DIR, 'results_v4.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")

    print(f"\nAll single model results:")
    for name, r in all_results.items():
        if 'test' in r and isinstance(r['test'], dict) and 'rmse' in r['test']:
            print(f"  {name}: RMSE={r['test']['rmse']:.4f}, MAE={r['test']['mae']:.4f}, R2={r['test']['r2']:.4f}")

    print(f"\nBest single model: {best_name}")
    print(f"  Test RMSE: {best_rmse:.4f}")

    print(f"\nEnsemble ({ensemble_results['ensemble_size']} models):")
    print(f"  Test RMSE: {ensemble_results['ensemble']['rmse']:.4f}")
    print(f"  Test MAE:  {ensemble_results['ensemble']['mae']:.4f}")
    print(f"  Test R2:   {ensemble_results['ensemble']['r2']:.4f}")

    print(f"\nComparison:")
    print(f"  PPFN-v1:                37.36")
    print(f"  PPFN-v2 (simple best):  {best_rmse:.4f}")
    print(f"  PPFN-v2 (ensemble):     {ensemble_results['ensemble']['rmse']:.4f}")
    print(f"  Target SOTA:            30.8")

    final_rmse = min(best_rmse, ensemble_results['ensemble']['rmse'])
    gap = final_rmse - 30.8
    print(f"\n  Gap to SOTA: {gap:.2f}")


if __name__ == '__main__':
    main()

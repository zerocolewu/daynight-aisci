"""
Run all PIPGA experiments with proper results saving.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, 'src')

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data import PolymerDataset, collate_fn
from model import PIPGA, count_parameters


def run_single_experiment(config, csv_path, save_dir, device):
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {config['model_name']}")
    print(f"{'='*60}")

    # Load data
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    tg_values = df['Tg'].tolist()

    # Setup cross-validation
    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=config['seed'])

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'model'), exist_ok=True)

    fold_results = []
    all_predictions = []
    all_targets = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(smiles_list)):
        print(f"\n--- Fold {fold_idx + 1}/{config['n_folds']} ---")

        # Split data
        train_val_smiles = [smiles_list[i] for i in train_val_idx]
        train_val_tg = [tg_values[i] for i in train_val_idx]

        n_val = int(len(train_val_smiles) * 0.111)  # ~10% for validation
        val_smiles = train_val_smiles[:n_val]
        val_tg = train_val_tg[:n_val]
        train_smiles = train_val_smiles[n_val:]
        train_tg = train_val_tg[n_val:]

        test_smiles = [smiles_list[i] for i in test_idx]
        test_tg = [tg_values[i] for i in test_idx]

        # Normalize targets
        scaler = StandardScaler()
        train_tg_norm = scaler.fit_transform(np.array(train_tg).reshape(-1, 1)).flatten().tolist()
        val_tg_norm = scaler.transform(np.array(val_tg).reshape(-1, 1)).flatten().tolist()
        test_tg_norm = scaler.transform(np.array(test_tg).reshape(-1, 1)).flatten().tolist()

        # Create datasets
        train_dataset = PolymerDataset(train_smiles, train_tg_norm)
        val_dataset = PolymerDataset(val_smiles, val_tg_norm)
        test_dataset = PolymerDataset(test_smiles, test_tg_norm)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

        # Create model
        model = PIPGA(
            node_input_dim=23,
            node_dim=config['node_dim'],
            edge_dim=config['edge_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            use_softmax=config['use_softmax'],
            use_fingerprint=config['use_fingerprint'],
            use_self_loop=config['use_self_loop'],
            use_multi_edge=config['use_multi_edge']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config['scheduler_patience'], factor=config['scheduler_factor'])
        criterion = torch.nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(config['max_epochs']):
            # Train
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                optimizer.zero_grad()
                pred = model(batch['atom_features'], batch['edge_index'], batch['edge_offset'],
                            batch['self_loop_features'], batch['fingerprint'], batch['batch'], batch['num_atoms']).squeeze(-1)
                loss = criterion(pred, batch['tg'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    pred = model(batch['atom_features'], batch['edge_index'], batch['edge_offset'],
                                batch['self_loop_features'], batch['fingerprint'], batch['batch'], batch['num_atoms']).squeeze(-1)
                    val_loss += criterion(pred, batch['tg']).item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, os.path.join(save_dir, 'model', f'best_model_fold{fold_idx}.pt'))
            else:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

        # Load best model and evaluate on test set
        checkpoint = torch.load(os.path.join(save_dir, 'model', f'best_model_fold{fold_idx}.pt'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_preds = []
        test_targets = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                pred = model(batch['atom_features'], batch['edge_index'], batch['edge_offset'],
                            batch['self_loop_features'], batch['fingerprint'], batch['batch'], batch['num_atoms']).squeeze(-1)
                test_preds.extend(pred.cpu().numpy())
                test_targets.extend(batch['tg'].cpu().numpy())

        # Inverse transform
        test_preds = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
        test_targets = scaler.inverse_transform(np.array(test_targets).reshape(-1, 1)).flatten()

        # Compute metrics
        rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
        mae = np.mean(np.abs(test_preds - test_targets))
        r2 = 1 - np.sum((test_preds - test_targets) ** 2) / np.sum((test_targets - np.mean(test_targets)) ** 2)

        print(f"Test RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")

        fold_results.append({
            'fold': fold_idx,
            'test_rmse': float(rmse),
            'test_mae': float(mae),
            'test_r2': float(r2),
            'best_epoch': best_epoch
        })
        all_predictions.extend(test_preds.tolist())
        all_targets.extend(test_targets.tolist())

    # Aggregate results
    test_rmses = [r['test_rmse'] for r in fold_results]
    test_maes = [r['test_mae'] for r in fold_results]
    test_r2s = [r['test_r2'] for r in fold_results]

    print(f"\n{'='*60}")
    print(f"Cross-validation Results for {config['model_name']}")
    print(f"{'='*60}")
    print(f"Test RMSE: {np.mean(test_rmses):.2f} ± {np.std(test_rmses):.2f}")
    print(f"Test MAE: {np.mean(test_maes):.2f} ± {np.std(test_maes):.2f}")
    print(f"Test R2: {np.mean(test_r2s):.4f} ± {np.std(test_r2s):.4f}")

    # Save results
    results = {
        'model_name': config['model_name'],
        'config': config,
        'test_rmse_mean': float(np.mean(test_rmses)),
        'test_rmse_std': float(np.std(test_rmses)),
        'test_mae_mean': float(np.mean(test_maes)),
        'test_mae_std': float(np.std(test_maes)),
        'test_r2_mean': float(np.mean(test_r2s)),
        'test_r2_std': float(np.std(test_r2s)),
        'fold_results': fold_results
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions
    pd.DataFrame({'target': all_targets, 'prediction': all_predictions}).to_csv(
        os.path.join(save_dir, 'predictions.csv'), index=False)

    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    csv_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    base_dir = "/home/home_public/aisci/AIscientist/src/3_03_pippga/results"

    # Base configuration
    base_config = {
        'node_dim': 128,
        'edge_dim': 64,
        'num_layers': 4,
        'num_heads': 4,
        'distance_threshold': 3,
        'batch_size': 32,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'max_epochs': 200,
        'early_stopping_patience': 20,
        'scheduler_patience': 10,
        'scheduler_factor': 0.5,
        'n_folds': 5,
        'seed': 42
    }

    # Experiment configurations
    experiments = [
        {'name': 'full_model', 'config': {**base_config, 'model_name': 'PIPGA_Full', 'use_softmax': False, 'use_fingerprint': True, 'use_self_loop': True, 'use_multi_edge': True}},
        {'name': 'no_fingerprint', 'config': {**base_config, 'model_name': 'PIPGA_NoFingerprint', 'use_softmax': False, 'use_fingerprint': False, 'use_self_loop': True, 'use_multi_edge': True}},
        {'name': 'softmax_attention', 'config': {**base_config, 'model_name': 'PIPGA_Softmax', 'use_softmax': True, 'use_fingerprint': True, 'use_self_loop': True, 'use_multi_edge': True}},
        {'name': 'no_self_loop', 'config': {**base_config, 'model_name': 'PIPGA_NoSelfLoop', 'use_softmax': False, 'use_fingerprint': True, 'use_self_loop': False, 'use_multi_edge': True}},
        {'name': 'single_edge', 'config': {**base_config, 'model_name': 'PIPGA_SingleEdge', 'use_softmax': False, 'use_fingerprint': True, 'use_self_loop': True, 'use_multi_edge': False}}
    ]

    all_results = {}

    for exp in experiments:
        save_dir = os.path.join(base_dir, exp['name'])
        result = run_single_experiment(exp['config'], csv_path, save_dir, device)
        all_results[exp['name']] = {
            'rmse_mean': result['test_rmse_mean'],
            'rmse_std': result['test_rmse_std']
        }

    # Create summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    summary_lines = [
        "PIPGA Experiment Results Summary",
        "=" * 60,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "CROSS-VALIDATION RESULTS (5 folds)",
        "=" * 60,
        "",
        f"{'Model':<25} {'Test RMSE':<20}",
        "-" * 45
    ]

    for name, res in all_results.items():
        summary_lines.append(f"{name:<25} {res['rmse_mean']:.2f} ± {res['rmse_std']:.2f}")
        print(f"{name:<25} {res['rmse_mean']:.2f} ± {res['rmse_std']:.2f}")

    # Ablation analysis
    full_rmse = all_results['full_model']['rmse_mean']
    summary_lines.extend(["", "ABLATION ANALYSIS", "=" * 60])

    ablations = [
        ('no_fingerprint', 'No Fingerprint', 0.5),
        ('softmax_attention', 'Softmax Attention', 1.5),
        ('no_self_loop', 'No Self-loop', 0.5),
        ('single_edge', 'Single-edge', 0.5)
    ]

    for key, name, threshold in ablations:
        degradation = all_results[key]['rmse_mean'] - full_rmse
        meets = "YES" if degradation > threshold else "NO"
        summary_lines.append(f"Full vs {name}: {degradation:+.2f} RMSE degradation (target >{threshold}: {meets})")
        print(f"Full vs {name}: {degradation:+.2f} RMSE degradation (target >{threshold}: {meets})")

    # Save summary
    with open(os.path.join(base_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"\nResults saved to {base_dir}")


if __name__ == "__main__":
    main()

"""
Training pipeline for PIPGA ablation studies with 5-fold cross-validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import sys
import json
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import PolymerDataset, collate_fn, load_tg_data
from model import PIPGA, count_parameters


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        # Move to device
        atom_features = batch['atom_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_offset = batch['edge_offset'].to(device)
        self_loop_features = batch['self_loop_features'].to(device)
        fingerprint = batch['fingerprint'].to(device)
        tg = batch['tg'].to(device)
        batch_idx = batch['batch'].to(device)
        num_atoms = batch['num_atoms']

        # Forward pass
        optimizer.zero_grad()
        predictions = model(atom_features, edge_index, edge_offset,
                           self_loop_features, fingerprint, batch_idx, num_atoms)
        predictions = predictions.squeeze(-1)

        # Compute loss
        loss = criterion(predictions, tg)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, loader, criterion, device, scaler=None):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            # Move to device
            atom_features = batch['atom_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_offset = batch['edge_offset'].to(device)
            self_loop_features = batch['self_loop_features'].to(device)
            fingerprint = batch['fingerprint'].to(device)
            tg = batch['tg'].to(device)
            batch_idx = batch['batch'].to(device)
            num_atoms = batch['num_atoms']

            # Forward pass
            predictions = model(atom_features, edge_index, edge_offset,
                               self_loop_features, fingerprint, batch_idx, num_atoms)
            predictions = predictions.squeeze(-1)

            # Compute loss
            loss = criterion(predictions, tg)

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(tg.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Inverse transform if scaler provided
    if scaler is not None:
        all_predictions = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
        all_targets = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()

    # Compute metrics
    mse = np.mean((all_predictions - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_predictions - all_targets))
    r2 = 1 - np.sum((all_predictions - all_targets) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)

    metrics = {
        'loss': total_loss / num_batches,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    return metrics, all_predictions, all_targets


def run_fold(fold_idx, train_data, val_data, test_data, config, device, save_dir):
    """Run training for a single fold."""
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}")
    print(f"{'='*60}")

    # Normalize targets
    scaler = StandardScaler()
    train_tg_normalized = scaler.fit_transform(np.array(train_data['tg']).reshape(-1, 1)).flatten()
    val_tg_normalized = scaler.transform(np.array(val_data['tg']).reshape(-1, 1)).flatten()
    test_tg_normalized = scaler.transform(np.array(test_data['tg']).reshape(-1, 1)).flatten()

    # Create datasets
    train_dataset = PolymerDataset(train_data['smiles'], train_tg_normalized.tolist(),
                                   distance_threshold=config['distance_threshold'])
    val_dataset = PolymerDataset(val_data['smiles'], val_tg_normalized.tolist(),
                                 distance_threshold=config['distance_threshold'])
    test_dataset = PolymerDataset(test_data['smiles'], test_tg_normalized.tolist(),
                                  distance_threshold=config['distance_threshold'])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, collate_fn=collate_fn, num_workers=0)

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

    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      patience=config['scheduler_patience'],
                                                      factor=config['scheduler_factor'])

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config['max_epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, scaler)

        # Update scheduler
        scheduler.step(val_metrics['rmse'])

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val RMSE: {val_metrics['rmse']:.2f} | Val R2: {val_metrics['r2']:.4f}")

        # Early stopping
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': best_val_rmse,
                'config': config
            }, os.path.join(save_dir, f'best_model_fold{fold_idx}.pt'))
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nBest validation RMSE: {best_val_rmse:.2f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    checkpoint = torch.load(os.path.join(save_dir, f'best_model_fold{fold_idx}.pt'),
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics, test_predictions, test_targets = evaluate(model, test_loader, criterion,
                                                            device, scaler)

    print(f"Test RMSE: {test_metrics['rmse']:.2f} | Test MAE: {test_metrics['mae']:.2f} | "
          f"Test R2: {test_metrics['r2']:.4f}")

    return {
        'fold': fold_idx,
        'best_val_rmse': best_val_rmse,
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
        'predictions': test_predictions,
        'targets': test_targets
    }


def run_cross_validation(config, csv_path, save_dir):
    """Run 5-fold cross-validation."""
    print(f"\n{'='*60}")
    print(f"Running {config['model_name']}")
    print(f"{'='*60}")

    # Load data
    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    tg_values = df['Tg'].tolist()

    # Setup cross-validation
    kfold = KFold(n_splits=config['n_folds'], shuffle=True,
                  random_state=config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'model'), exist_ok=True)

    fold_results = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(smiles_list)):
        # Split train_val into train and val
        train_val_smiles = [smiles_list[i] for i in train_val_idx]
        train_val_tg = [tg_values[i] for i in train_val_idx]

        # Use 80/20 split for train/val within train_val
        n_val = int(len(train_val_smiles) * 0.1 / 0.9)  # 10% of total
        val_smiles = train_val_smiles[:n_val]
        val_tg = train_val_tg[:n_val]
        train_smiles = train_val_smiles[n_val:]
        train_tg = train_val_tg[n_val:]

        test_smiles = [smiles_list[i] for i in test_idx]
        test_tg = [tg_values[i] for i in test_idx]

        train_data = {'smiles': train_smiles, 'tg': train_tg}
        val_data = {'smiles': val_smiles, 'tg': val_tg}
        test_data = {'smiles': test_smiles, 'tg': test_tg}

        # Run fold
        result = run_fold(fold_idx, train_data, val_data, test_data,
                         config, device, os.path.join(save_dir, 'model'))
        fold_results.append(result)

    # Aggregate results
    test_rmses = [r['test_metrics']['rmse'] for r in fold_results]
    test_maes = [r['test_metrics']['mae'] for r in fold_results]
    test_r2s = [r['test_metrics']['r2'] for r in fold_results]

    print(f"\n{'='*60}")
    print("Cross-validation results:")
    print(f"{'='*60}")
    print(f"Test RMSE: {np.mean(test_rmses):.2f} ± {np.std(test_rmses):.2f}")
    print(f"Test MAE: {np.mean(test_maes):.2f} ± {np.std(test_maes):.2f}")
    print(f"Test R2: {np.mean(test_r2s):.4f} ± {np.std(test_r2s):.4f}")

    # Save results
    results_summary = {
        'model_name': config['model_name'],
        'config': config,
        'test_rmse_mean': float(np.mean(test_rmses)),
        'test_rmse_std': float(np.std(test_rmses)),
        'test_mae_mean': float(np.mean(test_maes)),
        'test_mae_std': float(np.std(test_maes)),
        'test_r2_mean': float(np.mean(test_r2s)),
        'test_r2_std': float(np.std(test_r2s)),
        'fold_results': [{
            'fold': int(r['fold']),
            'test_rmse': float(r['test_metrics']['rmse']),
            'test_mae': float(r['test_metrics']['mae']),
            'test_r2': float(r['test_metrics']['r2']),
            'best_epoch': int(r['best_epoch'])
        } for r in fold_results]
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Save predictions
    all_predictions = []
    all_targets = []
    for r in fold_results:
        for pred, target in zip(r['predictions'], r['targets']):
            all_predictions.append(pred)
            all_targets.append(target)

    predictions_df = pd.DataFrame({
        'target': all_targets,
        'prediction': all_predictions
    })
    predictions_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

    return results_summary


def plot_results(results_summary, save_dir):
    """Plot prediction results."""
    predictions_df = pd.read_csv(os.path.join(save_dir, 'predictions.csv'))

    plt.figure(figsize=(8, 8))
    plt.scatter(predictions_df['target'], predictions_df['prediction'],
                alpha=0.5, s=20)

    # Plot diagonal
    min_val = min(predictions_df['target'].min(), predictions_df['prediction'].min())
    max_val = max(predictions_df['target'].max(), predictions_df['prediction'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.xlabel('True Tg (°C)', fontsize=14)
    plt.ylabel('Predicted Tg (°C)', fontsize=14)
    plt.title(f"{results_summary['model_name']}\n"
              f"RMSE: {results_summary['test_rmse_mean']:.2f} ± {results_summary['test_rmse_std']:.2f}",
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    # Get ablation type from command line
    ablation_type = sys.argv[1] if len(sys.argv) > 1 else "no_fingerprint"

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
        'seed': 42,
        'use_softmax': False,
        'use_fingerprint': True,
        'use_self_loop': True,
        'use_multi_edge': True
    }

    # Configure ablation
    if ablation_type == "no_fingerprint":
        config = base_config.copy()
        config['model_name'] = 'PIPGA_NoFingerprint'
        config['use_fingerprint'] = False
        save_dir = "/home/home_public/aisci/AIscientist/src/3_03_pippga/results/no_fingerprint"

    elif ablation_type == "softmax_attention":
        config = base_config.copy()
        config['model_name'] = 'PIPGA_SoftmaxAttention'
        config['use_softmax'] = True
        save_dir = "/home/home_public/aisci/AIscientist/src/3_03_pippga/results/softmax_attention"

    elif ablation_type == "no_self_loop":
        config = base_config.copy()
        config['model_name'] = 'PIPGA_NoSelfLoop'
        config['use_self_loop'] = False
        save_dir = "/home/home_public/aisci/AIscientist/src/3_03_pippga/results/no_self_loop"

    elif ablation_type == "single_edge":
        config = base_config.copy()
        config['model_name'] = 'PIPGA_SingleEdge'
        config['use_multi_edge'] = False
        save_dir = "/home/home_public/aisci/AIscientist/src/3_03_pippga/results/single_edge"

    else:
        print(f"Unknown ablation type: {ablation_type}")
        print("Valid options: no_fingerprint, softmax_attention, no_self_loop, single_edge")
        sys.exit(1)

    csv_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"

    # Run cross-validation
    results = run_cross_validation(config, csv_path, save_dir)

    # Plot results
    plot_results(results, save_dir)

    print(f"\nResults saved to {save_dir}")

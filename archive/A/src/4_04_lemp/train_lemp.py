"""
Training script for LEMP model with 5-fold cross-validation and ablation studies
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lemp_model import LEMP
from dataset import load_tg_dataset


class ListDataset(TorchDataset):
    """Simple wrapper for list of PyG Data objects"""
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def custom_collate(data_list):
    """Custom collate for DataLoader - returns tuple for model forward"""
    # Extract fingerprints, positions and atom types before batching
    fp_list = []
    positions_list = []
    atom_types_list = []

    for data in data_list:
        fp_list.append(data.fp)
        positions_list.append(data.positions_3d)
        atom_types_list.append(data.atom_types)

    # Stack fingerprints manually
    fp_batch = torch.stack(fp_list, dim=0)

    # Make copies of data objects without special attributes for PyG batching
    from torch_geometric.data import Data
    clean_data_list = []
    for data in data_list:
        clean_data = Data()
        clean_data.x = data.x
        clean_data.edge_index = data.edge_index
        clean_data.edge_attr = getattr(data, 'edge_attr', None)
        clean_data.y = data.y
        clean_data_list.append(clean_data)

    # Create PyG batch
    batch = Batch.from_data_list(clean_data_list)

    # Add fingerprint batch
    batch.fp = fp_batch

    return batch, positions_list, atom_types_list


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch, positions_list, atom_types_list in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(
            fp=batch.fp,
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            positions_list=positions_list,
            atom_types_list=atom_types_list
        )

        loss = criterion(predictions, batch.y.squeeze())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch, positions_list, atom_types_list in dataloader:
            batch = batch.to(device)

            predictions = model(
                fp=batch.fp,
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                positions_list=positions_list,
                atom_types_list=atom_types_list
            )

            loss = criterion(predictions, batch.y.squeeze())

            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(batch.y.squeeze().cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return {
        'loss': total_loss / max(len(dataloader), 1),
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets
    }


def run_fold(train_data, val_data, config, device, fold_id):
    """Run training for a single fold"""
    print(f"\n{'='*50}")
    print(f"Fold {fold_id + 1}")
    print(f"{'='*50}")

    # Create data loaders with ListDataset wrapper
    train_dataset = ListDataset(train_data)
    val_dataset = ListDataset(val_data)

    train_loader = TorchDataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, collate_fn=custom_collate, num_workers=0
    )
    val_loader = TorchDataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, collate_fn=custom_collate, num_workers=0
    )

    # Initialize model
    model = LEMP(
        fp_size=config['fp_size'],
        node_dim=25,  # From dataset.py: 10 atom types + 6 degree + 5 hybridization + 4 other = 25
        hidden_dim=config['hidden_dim'],
        grid_size=config['grid_size'],
        voxel_channels=config['voxel_channels'],
        num_kpoints=config['num_kpoints'],
        cutoff=config['cutoff'],
        num_gin_layers=config['num_gin_layers'],
        use_long_range=config['use_long_range'],
        use_fingerprint=config['use_fingerprint'],
        use_gin=config['use_gin']
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config['max_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_results = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_results['rmse'])

        if val_results['rmse'] < best_val_rmse:
            best_val_rmse = val_results['rmse']
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(),
                      os.path.join(config['output_dir'], f'fold_{fold_id}_best.pt'))
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val RMSE={val_results['rmse']:.2f}, "
                  f"Val MAE={val_results['mae']:.2f}, "
                  f"Val R²={val_results['r2']:.4f}")

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Fold {fold_id + 1} Best Val RMSE: {best_val_rmse:.2f} (epoch {best_epoch + 1})")

    # Load best model and return results
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], f'fold_{fold_id}_best.pt')))
    final_results = evaluate(model, val_loader, criterion, device)

    return {
        'best_val_rmse': best_val_rmse,
        'best_epoch': best_epoch,
        'final_rmse': final_results['rmse'],
        'final_mae': final_results['mae'],
        'final_r2': final_results['r2'],
        'predictions': final_results['predictions'],
        'targets': final_results['targets']
    }


def run_cross_validation(data_list, config, device):
    """Run 5-fold cross-validation"""
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    all_predictions = []
    all_targets = []

    for fold_id, (train_idx, val_idx) in enumerate(kfold.split(data_list)):
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        result = run_fold(train_data, val_data, config, device, fold_id)
        fold_results.append(result)

        all_predictions.extend(result['predictions'])
        all_targets.extend(result['targets'])

    # Aggregate results
    avg_rmse = np.mean([r['best_val_rmse'] for r in fold_results])
    std_rmse = np.std([r['best_val_rmse'] for r in fold_results])
    avg_mae = np.mean([r['final_mae'] for r in fold_results])
    avg_r2 = np.mean([r['final_r2'] for r in fold_results])

    overall_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))

    print(f"\n{'='*60}")
    print(f"Cross-Validation Results ({config['experiment_name']})")
    print(f"{'='*60}")
    print(f"Average Val RMSE: {avg_rmse:.2f} ± {std_rmse:.2f}")
    print(f"Average Val MAE: {avg_mae:.2f}")
    print(f"Average Val R²: {avg_r2:.4f}")
    print(f"Overall RMSE (all predictions): {overall_rmse:.2f}")

    return {
        'fold_results': fold_results,
        'avg_rmse': avg_rmse,
        'std_rmse': std_rmse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'overall_rmse': overall_rmse,
        'all_predictions': all_predictions,
        'all_targets': all_targets
    }


def run_ablation_studies(data_list, base_config, device):
    """Run ablation studies to measure component contributions"""

    ablation_configs = {
        'Full LEMP': {
            'use_long_range': True,
            'use_fingerprint': True,
            'use_gin': True,
            'experiment_name': 'Full LEMP'
        },
        'No Long-Range': {
            'use_long_range': False,
            'use_fingerprint': True,
            'use_gin': True,
            'experiment_name': 'No Long-Range (baseline)'
        },
        'No Fingerprint': {
            'use_long_range': True,
            'use_fingerprint': False,
            'use_gin': True,
            'experiment_name': 'No Fingerprint'
        },
        'No GIN': {
            'use_long_range': True,
            'use_fingerprint': True,
            'use_gin': False,
            'experiment_name': 'No GIN'
        }
    }

    # Cutoff ablation (only for full model)
    cutoff_configs = {
        'Cutoff 5Å': {'cutoff': 5.0, 'experiment_name': 'Cutoff 5Å'},
        'Cutoff 10Å': {'cutoff': 10.0, 'experiment_name': 'Cutoff 10Å'},
        'Cutoff 15Å': {'cutoff': 15.0, 'experiment_name': 'Cutoff 15Å'}
    }

    all_results = {}

    # Run component ablations
    print("\n" + "="*80)
    print("ABLATION STUDY: Component Contributions")
    print("="*80)

    for name, ablation_config in ablation_configs.items():
        config = {**base_config, **ablation_config}
        print(f"\n>>> Running: {name}")
        results = run_cross_validation(data_list, config, device)
        all_results[name] = results

    # Run cutoff ablations (full model only)
    print("\n" + "="*80)
    print("ABLATION STUDY: Long-Range Cutoff")
    print("="*80)

    for name, cutoff_config in cutoff_configs.items():
        config = {**base_config, **cutoff_config}
        config['use_long_range'] = True
        config['use_fingerprint'] = True
        config['use_gin'] = True
        print(f"\n>>> Running: {name}")
        results = run_cross_validation(data_list, config, device)
        all_results[name] = results

    return all_results


def analyze_results(ablation_results):
    """Analyze ablation results to quantify component contributions"""

    print("\n" + "="*80)
    print("ABLATION ANALYSIS: Component Contributions")
    print("="*80)

    # Component contributions
    full_rmse = ablation_results['Full LEMP']['overall_rmse']

    baseline_rmse = ablation_results['No Long-Range']['overall_rmse']
    no_fp_rmse = ablation_results['No Fingerprint']['overall_rmse']
    no_gin_rmse = ablation_results['No GIN']['overall_rmse']

    print(f"\nBaseline (No Long-Range): {baseline_rmse:.2f} RMSE")
    print(f"Full LEMP: {full_rmse:.2f} RMSE")
    print(f"\nComponent Contributions (RMSE improvement when included):")
    print(f"  Long-Range: {baseline_rmse - full_rmse:.2f} RMSE")
    print(f"  Fingerprint: {no_fp_rmse - full_rmse:.2f} RMSE")
    print(f"  Local GIN: {no_gin_rmse - full_rmse:.2f} RMSE")

    # Cutoff analysis
    print(f"\nLong-Range Cutoff Analysis:")
    for cutoff_name in ['Cutoff 5Å', 'Cutoff 10Å', 'Cutoff 15Å']:
        if cutoff_name in ablation_results:
            rmse = ablation_results[cutoff_name]['overall_rmse']
            print(f"  {cutoff_name}: {rmse:.2f} RMSE")

    # Summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY TABLE")
    print("="*80)
    print(f"{'Configuration':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-"*55)

    for name, results in ablation_results.items():
        print(f"{name:<25} {results['overall_rmse']:<10.2f} {results['avg_mae']:<10.2f} {results['avg_r2']:<10.4f}")

    return {
        'long_range_contribution': baseline_rmse - full_rmse,
        'fingerprint_contribution': no_fp_rmse - full_rmse,
        'gin_contribution': no_gin_rmse - full_rmse,
        'baseline_rmse': baseline_rmse,
        'full_rmse': full_rmse
    }


def main():
    # Configuration
    config = {
        'fp_size': 2048,
        'hidden_dim': 256,
        'grid_size': 16,
        'voxel_channels': 8,
        'num_kpoints': 32,
        'cutoff': 5.0,
        'num_gin_layers': 2,
        'batch_size': 32,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'max_epochs': 200,
        'patience': 20,
        'output_dir': '/home/home_public/aisci/AIscientist/src/4_04_lemp',
        'experiment_name': 'Full LEMP'
    }

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Load dataset
    print("\n" + "="*80)
    print("Loading Tg Dataset")
    print("="*80)

    csv_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
    data_list = load_tg_dataset(csv_path, fp_size=config['fp_size'], force_reprocess=True)

    print(f"Loaded {len(data_list)} samples")

    # Run ablation studies
    ablation_results = run_ablation_studies(data_list, config, device)

    # Analyze results
    analysis = analyze_results(ablation_results)

    # Save results
    results_path = os.path.join(config['output_dir'], 'ablation_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for name, results in ablation_results.items():
            serializable_results[name] = {
                'avg_rmse': float(results['avg_rmse']),
                'std_rmse': float(results['std_rmse']),
                'avg_mae': float(results['avg_mae']),
                'avg_r2': float(results['avg_r2']),
                'overall_rmse': float(results['overall_rmse'])
            }
        json.dump(serializable_results, f, indent=2)

    analysis_path = os.path.join(config['output_dir'], 'analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nResults saved to {config['output_dir']}")

    # Final report
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    print(f"\nTarget: <34.0 RMSE")
    print(f"Achieved: {analysis['full_rmse']:.2f} RMSE")
    print(f"\nLong-Range Contribution: {analysis['long_range_contribution']:.2f} RMSE")
    print(f"  (Target: >1.5 RMSE improvement)")

    if analysis['long_range_contribution'] > 1.5:
        print("\n✓ Long-range hypothesis VALIDATED")
    else:
        print("\n✗ Long-range hypothesis NOT validated")

    if analysis['full_rmse'] < 34.0:
        print("✓ RMSE target ACHIEVED")
    else:
        print("✗ RMSE target NOT achieved")

    if analysis['fingerprint_contribution'] > 0:
        print("✓ Fingerprint contribution confirmed (additive, not replacing)")
    else:
        print("✗ Fingerprint contribution negative")


if __name__ == "__main__":
    main()

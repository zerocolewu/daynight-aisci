#!/usr/bin/env python3
"""
Generate Tm results from existing checkpoints.
This script loads the trained Tm models and generates predictions/metrics.
"""

import os
import json
import pickle
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from torch_geometric.loader import DataLoader

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors

# Import from training script
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_ensemble_v2 import (
    GINWithFeatures, compute_148_features, process_psmiles,
    mol_to_pyg_data, set_seed, SEEDS, device
)

# Set seeds
SEEDS = [42, 123, 456, 789, 1024]
N_FOLDS = 5


def load_data_and_prepare_splits(csv_path: str):
    """Load data and prepare train/val/test splits."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    # Determine column names
    if 'psmiles' in df.columns:
        psmiles_col = 'psmiles'
    elif 'p_smiles' in df.columns:
        psmiles_col = 'p_smiles'
    elif 'Smiles' in df.columns:
        psmiles_col = 'Smiles'
    else:
        psmiles_col = df.columns[0]

    target_col = [c for c in df.columns if c.lower() in ['tm', 'egc', 'tg']][0]
    print(f"Using columns: psmiles={psmiles_col}, target={target_col}")

    # Process molecules
    print("Processing molecules...")
    data_list = []
    failed = 0

    for idx, row in df.iterrows():
        psmiles = row[psmiles_col]
        target = row[target_col]

        mol, smiles = process_psmiles(psmiles)
        if mol is None:
            failed += 1
            continue

        features = compute_148_features(mol)
        data = mol_to_pyg_data(mol, target, features)

        if data is not None:
            data_list.append(data)
        else:
            failed += 1

    print(f"Successfully processed {len(data_list)} molecules, {failed} failed")
    return data_list


def generate_tm_results_from_checkpoints(
    data_list: List,
    checkpoint_dir: str,
    property_name: str = "Tm"
) -> Dict:
    """Generate results by loading checkpoints and evaluating."""

    print(f"\n{'='*80}")
    print(f"Generating {property_name} results from checkpoints")
    print(f"{'='*80}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    all_results = {
        'property': property_name,
        'n_samples': len(data_list),
        'n_folds': 5,
        'n_seeds': len(SEEDS),
        'seeds': SEEDS,
        'fold_results': [],
    }

    all_fold_predictions = {}

    for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(data_list), 1):
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx}/5")
        print(f"{'='*80}")

        train_val_data = [data_list[i] for i in train_val_idx]
        test_data = [data_list[i] for i in test_idx]

        split_idx = int(len(train_val_data) * 0.8)
        train_data = train_val_data[:split_idx]
        val_data = train_val_data[split_idx:]

        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        fold_seed_results = []
        fold_predictions = []

        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        num_node_features = test_data[0].x.shape[1]

        for seed in SEEDS:
            print(f"\n  Loading seed {seed}...")

            checkpoint_path = os.path.join(checkpoint_dir, f"{property_name}_fold{fold_idx}_seed{seed}.pt")
            if not os.path.exists(checkpoint_path):
                print(f"  WARNING: Checkpoint not found: {checkpoint_path}")
                continue

            # Load model
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = GINWithFeatures(num_node_features=num_node_features, num_features=148, hidden_dim=256, num_layers=5)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            # Evaluate
            predictions = []
            targets = []

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    predictions.extend(output.cpu().numpy())
                    targets.extend(batch.y.view(-1).cpu().numpy())

            predictions = np.array(predictions)
            targets = np.array(targets)

            rmse = np.sqrt(mean_squared_error(targets, predictions))
            mae = mean_absolute_error(targets, predictions)
            r2 = r2_score(targets, predictions)

            fold_seed_results.append({
                'seed': seed,
                'test_rmse': float(rmse),
                'test_mae': float(mae),
                'test_r2': float(r2),
                'best_epoch': checkpoint.get('best_epoch', 0),
                'best_val_loss': checkpoint.get('best_val_loss', 0.0),
            })

            fold_predictions.append(predictions)

            print(f"    Seed {seed}: Test RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        # Compute ensemble metrics
        fold_predictions = np.array(fold_predictions)
        ensemble_pred = np.mean(fold_predictions, axis=0)
        ensemble_std = np.std(fold_predictions, axis=0)
        targets = targets  # Same for all seeds

        ensemble_rmse = np.sqrt(mean_squared_error(targets, ensemble_pred))
        ensemble_mae = mean_absolute_error(targets, ensemble_pred)
        ensemble_r2 = r2_score(targets, ensemble_pred)

        # Uncertainty-error correlation
        abs_errors = np.abs(targets - ensemble_pred)
        correlation, p_value = stats.pearsonr(ensemble_std, abs_errors)

        # High/low uncertainty RMSE
        threshold = np.percentile(ensemble_std, 80)
        high_uncertainty_mask = ensemble_std >= threshold

        high_unc_rmse = np.sqrt(mean_squared_error(
            targets[high_uncertainty_mask],
            ensemble_pred[high_uncertainty_mask]
        )) if np.sum(high_uncertainty_mask) > 0 else 0

        low_unc_rmse = np.sqrt(mean_squared_error(
            targets[~high_uncertainty_mask],
            ensemble_pred[~high_uncertainty_mask]
        )) if np.sum(~high_uncertainty_mask) > 0 else 0

        fold_result = {
            'fold': fold_idx,
            'seed_results': fold_seed_results,
            'individual_seed_rmse': [r['test_rmse'] for r in fold_seed_results],
            'individual_seed_rmse_mean': float(np.mean([r['test_rmse'] for r in fold_seed_results])),
            'individual_seed_rmse_std': float(np.std([r['test_rmse'] for r in fold_seed_results])),
            'ensemble_rmse': float(ensemble_rmse),
            'ensemble_mae': float(ensemble_mae),
            'ensemble_r2': float(ensemble_r2),
            'uncertainty_error_correlation': float(correlation),
            'uncertainty_error_pvalue': float(p_value),
            'high_uncertainty_rmse': float(high_unc_rmse),
            'low_uncertainty_rmse': float(low_unc_rmse),
            'n_high_uncertainty': int(np.sum(high_uncertainty_mask)),
        }

        all_results['fold_results'].append(fold_result)

        all_fold_predictions[fold_idx] = {
            'predictions': fold_predictions,
            'ensemble_pred': ensemble_pred,
            'ensemble_std': ensemble_std,
            'targets': targets,
        }

        print(f"\n  Fold {fold_idx} Results:")
        print(f"    Individual seeds RMSE: {fold_result['individual_seed_rmse_mean']:.4f} ± {fold_result['individual_seed_rmse_std']:.4f}")
        print(f"    Ensemble RMSE: {ensemble_rmse:.4f}")
        print(f"    Uncertainty-Error correlation: r={correlation:.4f} (p={p_value:.4e})")

    # Cross-fold statistics
    ensemble_rmses = [f['ensemble_rmse'] for f in all_results['fold_results']]
    individual_means = [f['individual_seed_rmse_mean'] for f in all_results['fold_results']]

    all_results['cross_fold_ensemble'] = {
        'mean_rmse': float(np.mean(ensemble_rmses)),
        'std_rmse': float(np.std(ensemble_rmses)),
        'best_rmse': float(np.min(ensemble_rmses)),
        'best_fold': int(np.argmin(ensemble_rmses) + 1),
        'worst_rmse': float(np.max(ensemble_rmses)),
        'worst_fold': int(np.argmax(ensemble_rmses) + 1),
        'variance_reduction': float((np.std(individual_means) - np.std(ensemble_rmses)) / np.std(individual_means) * 100) if np.std(individual_means) > 0 else 0.0,
    }

    print(f"\n{'='*80}")
    print(f"{property_name} Cross-Fold Results:")
    print(f"  Ensemble RMSE: {all_results['cross_fold_ensemble']['mean_rmse']:.4f} ± {all_results['cross_fold_ensemble']['std_rmse']:.4f}")
    print(f"  Best fold: {all_results['cross_fold_ensemble']['best_fold']} (RMSE: {all_results['cross_fold_ensemble']['best_rmse']:.4f})")
    print(f"  Worst fold: {all_results['cross_fold_ensemble']['worst_fold']} (RMSE: {all_results['cross_fold_ensemble']['worst_rmse']:.4f})")
    print(f"  Variance reduction: {all_results['cross_fold_ensemble']['variance_reduction']:.2f}%")
    print(f"{'='*80}\n")

    # Save predictions
    predictions_path = os.path.join(checkpoint_dir, f"{property_name}_predictions.pkl")
    with open(predictions_path, 'wb') as f:
        pickle.dump(all_fold_predictions, f)

    return all_results


def main():
    base_dir = Path(__file__).parent
    checkpoint_dir = base_dir / "checkpoints"
    dataset_dir = Path(__file__).parent.parent / "alldataset"

    # Load and prepare Tm data
    data_list = load_data_and_prepare_splits(str(dataset_dir / "Tm.csv"))

    # Generate Tm results from checkpoints
    tm_results = generate_tm_results_from_checkpoints(
        data_list=data_list,
        checkpoint_dir=str(checkpoint_dir),
        property_name="Tm"
    )

    # Save results
    results_path = base_dir / "tm_results.json"
    with open(results_path, 'w') as f:
        json.dump(tm_results, f, indent=2)

    print(f"\nTm results saved to: {results_path}")


if __name__ == "__main__":
    main()

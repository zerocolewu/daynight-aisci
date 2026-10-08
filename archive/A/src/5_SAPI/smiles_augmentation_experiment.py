#!/usr/bin/env python
"""
SMILES Augmentation Experiment for Tg Prediction
Testing whether PerioGT's advantage comes from data augmentation rather than architecture
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import json
import os
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_path': '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv',
    'output_dir': '/home/home_public/aisci/AIscientist/src/5_SAPI/results',
    'n_folds': 5,
    'n_epochs': 500,
    'batch_size': 64,
    'learning_rate': 0.001,
    'patience': 20,
    'hidden_dim': 128,
    'n_bits': 2048,
    'radius': 2,
    'k_values': [1, 3, 5, 10],  # Augmentation factors
    'use_test_time_aug': [False, True],  # Test both
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# SMILES AUGMENTATION FUNCTIONS
# ============================================================================

def generate_random_smiles(mol, k=5):
    """
    Generate k random SMILES for a molecule using RDKit's random SMILES generation.
    This randomizes atom ordering before canonicalization.
    """
    if mol is None:
        return []

    smiles_list = []
    for _ in range(k):
        try:
            # doRandom=True generates random SMILES by randomizing atom order
            smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            if smiles:
                smiles_list.append(smiles)
        except:
            continue

    return smiles_list


def generate_rotated_smiles(mol, k=5):
    """
    Generate rotated SMILES for cyclic structures by changing starting atom position.
    For polymers, this rotates the representation of the repeat unit.
    """
    if mol is None:
        return []

    smiles_list = []
    num_atoms = mol.GetNumAtoms()

    if num_atoms == 0:
        return []

    # Try different starting atoms
    start_atoms = list(range(min(k, num_atoms)))

    for start_idx in start_atoms:
        try:
            smiles = Chem.MolToSmiles(mol, rootedAtAtom=start_idx, canonical=False)
            if smiles:
                smiles_list.append(smiles)
        except:
            continue

    return smiles_list


def generate_combined_augmentation(mol, k=5):
    """
    Combine random and rotated SMILES generation.
    """
    random_smiles = generate_random_smiles(mol, k=(k+1)//2)
    rotated_smiles = generate_rotated_smiles(mol, k=k//2)
    return random_smiles + rotated_smiles


# ============================================================================
# FINGERPRINT GENERATION
# ============================================================================

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """
    Convert SMILES to ECFP4 fingerprint (Morgan fingerprint with radius 2).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None


# ============================================================================
# PYTORCH MODEL
# ============================================================================

class TgPredictor(nn.Module):
    """
    Simple MLP for Tg prediction from fingerprints.
    Architecture: Input -> Linear(2048, 128) -> ReLU -> Linear(128, 1)
    """
    def __init__(self, input_dim=2048, hidden_dim=128):
        super(TgPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PolymerDataset(Dataset):
    """Dataset class for polymer Tg prediction."""
    def __init__(self, fingerprints, targets):
        self.fingerprints = torch.FloatTensor(fingerprints)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.fingerprints[idx], self.targets[idx]


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for fingerprints, targets in dataloader:
        fingerprints = fingerprints.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(fingerprints)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for fingerprints, targets in dataloader:
            fingerprints = fingerprints.to(device)
            targets = targets.to(device)

            outputs = model(fingerprints)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    return total_loss / len(dataloader), np.array(predictions), np.array(actuals)


def train_model(train_fps, train_targets, val_fps, val_targets, config, verbose=False):
    """
    Train the MLP model with early stopping.
    """
    device = config['device']

    # Create datasets and dataloaders
    train_dataset = PolymerDataset(train_fps, train_targets)
    val_dataset = PolymerDataset(val_fps, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    model = TgPredictor(input_dim=config['n_bits'], hidden_dim=config['hidden_dim']).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(config['n_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{config['n_epochs']}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, best_val_loss


# ============================================================================
# MAIN EXPERIMENT FUNCTIONS
# ============================================================================

def augment_dataset(df, k, augmentation_type='random'):
    """
    Create augmented dataset with k SMILES per polymer.

    Args:
        df: DataFrame with 'Smiles' and 'Tg' columns
        k: Number of augmented SMILES per polymer
        augmentation_type: 'random', 'rotated', or 'combined'

    Returns:
        augmented_fps: List of fingerprints
        augmented_targets: List of Tg values
    """
    augmented_fps = []
    augmented_targets = []

    print(f"Generating {k} augmented SMILES per polymer using {augmentation_type} strategy...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row['Smiles']
        target = row['Tg']

        # Parse molecule
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # If parsing fails, use canonical SMILES only
            fp = smiles_to_fingerprint(smiles, CONFIG['radius'], CONFIG['n_bits'])
            if fp is not None:
                augmented_fps.append(fp)
                augmented_targets.append(target)
            continue

        # Generate augmented SMILES
        if k == 1:
            # No augmentation - use canonical SMILES
            aug_smiles_list = [Chem.MolToSmiles(mol, canonical=True)]
        else:
            if augmentation_type == 'random':
                aug_smiles_list = generate_random_smiles(mol, k)
            elif augmentation_type == 'rotated':
                aug_smiles_list = generate_rotated_smiles(mol, k)
            else:  # combined
                aug_smiles_list = generate_combined_augmentation(mol, k)

        # Convert to fingerprints
        for aug_smiles in aug_smiles_list:
            fp = smiles_to_fingerprint(aug_smiles, CONFIG['radius'], CONFIG['n_bits'])
            if fp is not None:
                augmented_fps.append(fp)
                augmented_targets.append(target)

    print(f"Generated {len(augmented_fps)} samples from {len(df)} original polymers")

    return np.array(augmented_fps), np.array(augmented_targets)


def predict_with_test_time_augmentation(model, test_df, k, scaler, config):
    """
    Make predictions using test-time augmentation.
    For each test polymer, generate k augmented SMILES, get k predictions, and average.
    """
    device = config['device']
    model.eval()

    all_predictions = []

    print(f"Test-time augmentation with k={k}...")

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        smiles = row['Smiles']
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # Use canonical if parsing fails
            fp = smiles_to_fingerprint(smiles, CONFIG['radius'], CONFIG['n_bits'])
            if fp is not None:
                fp_tensor = torch.FloatTensor(fp).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(fp_tensor).item()
                all_predictions.append(pred)
            continue

        # Generate k augmented SMILES
        aug_smiles_list = generate_random_smiles(mol, k)

        if len(aug_smiles_list) == 0:
            aug_smiles_list = [Chem.MolToSmiles(mol, canonical=True)]

        # Get predictions for each augmented SMILES
        predictions = []
        for aug_smiles in aug_smiles_list:
            fp = smiles_to_fingerprint(aug_smiles, CONFIG['radius'], CONFIG['n_bits'])
            if fp is not None:
                fp_tensor = torch.FloatTensor(fp).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(fp_tensor).item()
                predictions.append(pred)

        # Average predictions
        if len(predictions) > 0:
            avg_pred = np.mean(predictions)
            all_predictions.append(avg_pred)

    # Inverse transform
    all_predictions = np.array(all_predictions).reshape(-1, 1)
    all_predictions = scaler.inverse_transform(all_predictions).flatten()

    return all_predictions


def run_single_experiment(k, use_test_time_aug, config, verbose=False):
    """
    Run a single experiment with given augmentation parameters.

    Args:
        k: Number of augmented SMILES per polymer
        use_test_time_aug: Whether to use test-time augmentation
        config: Configuration dictionary
        verbose: Print detailed progress

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: K={k}, Test-time Aug={use_test_time_aug}")
    print(f"{'='*80}\n")

    # Load data
    df = pd.read_csv(config['data_path'])
    print(f"Loaded {len(df)} polymer samples")

    # 5-fold cross-validation
    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=SEED)

    fold_results = []
    all_train_curves = []
    all_val_curves = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df)):
        print(f"\n--- Fold {fold_idx + 1}/{config['n_folds']} ---")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Augment training data
        train_fps, train_targets = augment_dataset(train_df, k, augmentation_type='random')

        # For validation/test, use canonical SMILES (no augmentation)
        val_fps, val_targets = augment_dataset(val_df, k=1, augmentation_type='random')

        # Normalize targets
        scaler = StandardScaler()
        train_targets_norm = scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
        val_targets_norm = scaler.transform(val_targets.reshape(-1, 1)).flatten()

        # Train model
        if verbose:
            print(f"Training on {len(train_fps)} samples, validating on {len(val_fps)} samples")

        model, train_losses, val_losses, best_val_loss = train_model(
            train_fps, train_targets_norm, val_fps, val_targets_norm, config, verbose=verbose
        )

        all_train_curves.append(train_losses)
        all_val_curves.append(val_losses)

        # Make predictions
        if use_test_time_aug and k > 1:
            # Test-time augmentation
            predictions = predict_with_test_time_augmentation(model, val_df, k, scaler, config)
            actuals = val_targets
        else:
            # Standard prediction
            val_dataset = PolymerDataset(val_fps, val_targets_norm)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

            device = config['device']
            criterion = nn.MSELoss()
            _, preds_norm, actuals_norm = evaluate(model, val_loader, criterion, device)

            # Inverse transform
            predictions = scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()
            actuals = scaler.inverse_transform(actuals_norm.reshape(-1, 1)).flatten()

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = 1 - np.sum((actuals - predictions)**2) / np.sum((actuals - np.mean(actuals))**2)

        print(f"Fold {fold_idx + 1} Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        fold_results.append({
            'fold': fold_idx + 1,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'actuals': actuals,
            'train_size': len(train_fps),
            'val_size': len(val_fps),
            'n_params': model.count_parameters(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        })

    # Aggregate results
    rmses = [r['rmse'] for r in fold_results]
    maes = [r['mae'] for r in fold_results]
    r2s = [r['r2'] for r in fold_results]

    results = {
        'k': k,
        'test_time_aug': use_test_time_aug,
        'fold_results': fold_results,
        'mean_rmse': np.mean(rmses),
        'std_rmse': np.std(rmses),
        'mean_mae': np.mean(maes),
        'std_mae': np.std(maes),
        'mean_r2': np.mean(r2s),
        'std_r2': np.std(r2s),
        'all_train_curves': all_train_curves,
        'all_val_curves': all_val_curves
    }

    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY: K={k}, Test-time Aug={use_test_time_aug}")
    print(f"Mean RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
    print(f"Mean MAE: {results['mean_mae']:.4f} ± {results['std_mae']:.4f}")
    print(f"Mean R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    print(f"{'='*80}\n")

    return results


def plot_training_curves(results_list, output_dir):
    """Plot training curves for all experiments."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, results in enumerate(results_list):
        k = results['k']
        test_aug = results['test_time_aug']

        # Average across folds
        avg_train_curves = []
        avg_val_curves = []

        max_epochs = max(len(curve) for curve in results['all_train_curves'])

        for i in range(max_epochs):
            train_vals = [curve[i] if i < len(curve) else np.nan
                         for curve in results['all_train_curves']]
            val_vals = [curve[i] if i < len(curve) else np.nan
                       for curve in results['all_val_curves']]

            avg_train_curves.append(np.nanmean(train_vals))
            avg_val_curves.append(np.nanmean(val_vals))

        row = idx // 3
        col = idx % 3

        axes[row, col].plot(avg_train_curves, label='Train Loss', alpha=0.7)
        axes[row, col].plot(avg_val_curves, label='Val Loss', alpha=0.7)
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel('Loss (MSE)')
        axes[row, col].set_title(f'K={k}, Test-Aug={test_aug}\nRMSE: {results["mean_rmse"]:.2f}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_dir}/training_curves.png")
    plt.close()


def plot_results_summary(results_list, output_dir):
    """Plot summary of results across all experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Extract data
    ks = [r['k'] for r in results_list]
    test_augs = [r['test_time_aug'] for r in results_list]
    rmses = [r['mean_rmse'] for r in results_list]
    rmse_stds = [r['std_rmse'] for r in results_list]
    maes = [r['mean_mae'] for r in results_list]
    mae_stds = [r['std_mae'] for r in results_list]
    r2s = [r['mean_r2'] for r in results_list]
    r2_stds = [r['std_r2'] for r in results_list]

    # Create labels
    labels = [f'K={k}\nTest-Aug={ta}' for k, ta in zip(ks, test_augs)]

    x = np.arange(len(labels))

    # RMSE plot
    axes[0].bar(x, rmses, yerr=rmse_stds, capsize=5, alpha=0.7, color='steelblue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE Comparison')
    axes[0].axhline(y=38.78, color='red', linestyle='--', label='Baseline (K=1)')
    axes[0].axhline(y=30.8, color='green', linestyle='--', label='PerioGT Target')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # MAE plot
    axes[1].bar(x, maes, yerr=mae_stds, capsize=5, alpha=0.7, color='coral')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')

    # R² plot
    axes[2].bar(x, r2s, yerr=r2_stds, capsize=5, alpha=0.7, color='lightgreen')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² Comparison')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results_summary.png'), dpi=300, bbox_inches='tight')
    print(f"Saved results summary to {output_dir}/results_summary.png")
    plt.close()


def main():
    """Main experiment runner."""
    print(f"\n{'='*80}")
    print("SMILES AUGMENTATION EXPERIMENT FOR TG PREDICTION")
    print(f"{'='*80}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Save configuration
    with open(os.path.join(CONFIG['output_dir'], 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)

    # Run all experiments
    all_results = []

    # Experiment conditions:
    # 1. K=1 (no augmentation) - baseline
    # 2. K=3 random SMILES
    # 3. K=5 random SMILES
    # 4. K=10 random SMILES
    # 5. K=5 with test-time augmentation

    experiments = [
        (1, False),   # Baseline
        (3, False),   # K=3
        (5, False),   # K=5
        (10, False),  # K=10
        (5, True),    # K=5 with test-time aug
    ]

    for k, test_aug in experiments:
        results = run_single_experiment(k, test_aug, CONFIG, verbose=True)
        all_results.append(results)

        # Save intermediate results
        with open(os.path.join(CONFIG['output_dir'], f'results_k{k}_testaug{test_aug}.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_to_save = {
                'k': results['k'],
                'test_time_aug': results['test_time_aug'],
                'mean_rmse': float(results['mean_rmse']),
                'std_rmse': float(results['std_rmse']),
                'mean_mae': float(results['mean_mae']),
                'std_mae': float(results['std_mae']),
                'mean_r2': float(results['mean_r2']),
                'std_r2': float(results['std_r2']),
            }
            json.dump(results_to_save, f, indent=2)

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(all_results, CONFIG['output_dir'])
    plot_results_summary(all_results, CONFIG['output_dir'])

    # Create final summary table
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'K':<5} {'Test-Aug':<10} {'RMSE':<15} {'MAE':<15} {'R²':<15}")
    print(f"{'-'*80}")

    for results in all_results:
        print(f"{results['k']:<5} {str(results['test_time_aug']):<10} "
              f"{results['mean_rmse']:.4f}±{results['std_rmse']:.4f}  "
              f"{results['mean_mae']:.4f}±{results['std_mae']:.4f}  "
              f"{results['mean_r2']:.4f}±{results['std_r2']:.4f}")

    print(f"{'='*80}\n")

    # Save final results
    final_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': [
            {
                'k': r['k'],
                'test_time_aug': r['test_time_aug'],
                'mean_rmse': float(r['mean_rmse']),
                'std_rmse': float(r['std_rmse']),
                'mean_mae': float(r['mean_mae']),
                'std_mae': float(r['std_mae']),
                'mean_r2': float(r['mean_r2']),
                'std_r2': float(r['std_r2']),
            }
            for r in all_results
        ]
    }

    with open(os.path.join(CONFIG['output_dir'], 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nAll results saved to: {CONFIG['output_dir']}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results


if __name__ == '__main__':
    results = main()

"""
Training script for Repetition-Invariant Transformer (RIT) polymer Tg prediction.
Includes ensemble training, evaluation, ablation studies, and comprehensive metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from dataset import PolymerDataset, collate_fn, get_tg_range
from model import RITModel, create_model

# Set random seeds for reproducibility
def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_path: str) -> Tuple[List[str], List[float]]:
    """Load polymer data from CSV file."""
    df = pd.read_csv(data_path)

    # Check columns
    if 'Smiles' not in df.columns or 'Tg' not in df.columns:
        raise ValueError("CSV must have 'Smiles' and 'Tg' columns")

    smiles_list = df['Smiles'].tolist()
    tg_values = df['Tg'].tolist()

    print(f"Loaded {len(smiles_list)} samples from {data_path}")
    print(f"Tg range: {min(tg_values):.1f} - {max(tg_values):.1f} K")

    return smiles_list, tg_values


def create_datasets(smiles_list: List[str], tg_values: List[float],
                    train_ratio: float = 0.72, val_ratio: float = 0.08,
                    test_ratio: float = 0.20, seed: int = 42,
                    num_ru: int = 3) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with stratified splitting.

    Args:
        smiles_list: List of p-SMILES strings
        tg_values: List of Tg values
        train_ratio: Fraction of training data
        val_ratio: Fraction of validation data
        test_ratio: Fraction of test data
        seed: Random seed for splitting
        num_ru: Number of repeat unit copies

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = PolymerDataset(smiles_list, tg_values, num_ru=num_ru)

    # Stratified split by Tg range
    indices = list(range(len(full_dataset)))
    ranges = [get_tg_range(full_dataset.data_list[i].y.item()) for i in indices]

    # Split indices by range
    range_indices = {0: [], 1: [], 2: []}
    for idx, r in enumerate(ranges):
        range_indices[r].append(idx)

    # Split each range
    train_indices, val_indices, test_indices = [], [], []

    for r in range_indices:
        idxs = range_indices[r]
        np.random.seed(seed)
        np.random.shuffle(idxs)

        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_indices.extend(idxs[:n_train])
        val_indices.extend(idxs[n_train:n_train + n_val])
        test_indices.extend(idxs[n_train + n_val:])

    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    print(f"\nData split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader


def compute_metrics(predictions: np.ndarray, targets: np.ndarray,
                    range_probs: np.ndarray = None) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Predicted Tg values
        targets: True Tg values
        range_probs: Predicted range probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    # RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # MAE
    mae = np.mean(np.abs(predictions - targets))

    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    # Per-range metrics
    ranges = [get_tg_range(t) for t in targets]
    range_names = ['low (<100)', 'medium (100-200)', 'high (>200)']

    for r in range(3):
        mask = np.array(ranges) == r
        if mask.sum() > 0:
            pred_r = predictions[mask]
            targ_r = targets[mask]
            rmse_r = np.sqrt(np.mean((pred_r - targ_r) ** 2))
            mae_r = np.mean(np.abs(pred_r - targ_r))
            metrics[f'rmse_range_{r}'] = rmse_r
            metrics[f'mae_range_{r}'] = mae_r
            metrics[f'count_range_{r}'] = mask.sum()

    # Range classification accuracy
    if range_probs is not None:
        pred_ranges = np.argmax(range_probs, axis=1)
        true_ranges = np.array(ranges)
        range_acc = np.mean(pred_ranges == true_ranges)
        metrics['range_accuracy'] = range_acc

    return metrics


def train_epoch(model: nn.Module, train_loader: DataLoader,
                optimizer: optim.Optimizer, device: torch.device,
                ce_weight: float = 0.1) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_mse = 0
    total_ce = 0
    count = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        batch = batch.to(device)

        # Forward pass
        tg_pred, range_logits = model(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.conformer_stats, batch.batch
        )

        # Compute losses
        targets = batch.y
        mse_loss = F.mse_loss(tg_pred.squeeze(), targets.squeeze())

        # Range classification loss
        true_ranges = torch.tensor([get_tg_range(t.item()) for t in targets],
                                   dtype=torch.long, device=device)
        ce_loss = F.cross_entropy(range_logits, true_ranges)

        # Combined loss
        loss = mse_loss + ce_weight * ce_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_ce += ce_loss.item()
        count += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'mse': f'{mse_loss.item():.2f}',
                          'ce': f'{ce_loss.item():.4f}'})

    return total_loss / count, total_mse / count, total_ce / count


def validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    predictions = []
    targets = []
    range_probs_list = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            batch = batch.to(device)

            tg_pred, range_logits = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.conformer_stats, batch.batch
            )

            predictions.extend(tg_pred.squeeze().cpu().numpy())
            targets.extend(batch.y.squeeze().cpu().numpy())

            range_probs = F.softmax(range_logits, dim=-1)
            range_probs_list.extend(range_probs.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)
    range_probs = np.array(range_probs_list)

    metrics = compute_metrics(predictions, targets, range_probs)

    return metrics


def test(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Test model."""
    model.eval()

    predictions = []
    targets = []
    range_probs_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            batch = batch.to(device)

            tg_pred, range_logits = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.conformer_stats, batch.batch
            )

            predictions.extend(tg_pred.squeeze().cpu().numpy())
            targets.extend(batch.y.squeeze().cpu().numpy())

            range_probs = F.softmax(range_logits, dim=-1)
            range_probs_list.extend(range_probs.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)
    range_probs = np.array(range_probs_list)

    metrics = compute_metrics(predictions, targets, range_probs)

    return metrics, predictions, targets


def train_single_model(train_loader: DataLoader, val_loader: DataLoader,
                       device: torch.device, seed: int = 42,
                       save_path: str = None) -> Tuple[nn.Module, Dict]:
    """
    Train a single model with early stopping.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        seed: Random seed
        save_path: Path to save best model

    Returns:
        Trained model and training history
    """
    set_seed(seed)

    # Create model
    model = create_model(
        node_dim=30,
        edge_dim=7,
        global_dim=16,
        hidden_dim=256,
        num_gnn_layers=5,
        num_transformer_layers=4,
        num_heads=8,
        num_ru=3,
        dropout=0.1
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training loop
    best_val_rmse = float('inf')
    patience_counter = 0
    max_patience = 30
    max_epochs = 300

    history = {
        'train_loss': [],
        'train_mse': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': [],
        'lr': []
    }

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        print("-" * 50)

        # Train
        train_loss, train_mse, train_ce = train_epoch(
            model, train_loader, optimizer, device, ce_weight=0.1
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_mse'].append(train_mse)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | MSE: {train_mse:.2f}")
        print(f"Val RMSE: {val_metrics['rmse']:.2f} | MAE: {val_metrics['mae']:.2f} | R²: {val_metrics['r2']:.4f}")
        print(f"Per-range RMSE: Low={val_metrics.get('rmse_range_0', 0):.2f}, "
              f"Mid={val_metrics.get('rmse_range_1', 0):.2f}, "
              f"High={val_metrics.get('rmse_range_2', 0):.2f}")

        # Learning rate scheduling
        scheduler.step(val_metrics['rmse'])

        # Early stopping and model saving
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            patience_counter = 0

            # Save best model
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_rmse': float(best_val_rmse),
                    'seed': seed
                }, save_path)
                print(f"✓ Saved best model (RMSE: {best_val_rmse:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    print(f"\nBest validation RMSE: {best_val_rmse:.2f}")

    # Load best model
    if save_path and os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model, history


def train_ensemble(train_loader: DataLoader, val_loader: DataLoader,
                   test_loader: DataLoader, device: torch.device,
                   save_dir: str, num_seeds: int = 5) -> Dict:
    """
    Train ensemble of models.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to train on
        save_dir: Directory to save models
        num_seeds: Number of ensemble members

    Returns:
        Dictionary with ensemble results
    """
    os.makedirs(save_dir, exist_ok=True)

    ensemble_predictions = []
    ensemble_results = []

    for i, seed in enumerate(range(42, 42 + num_seeds)):
        print(f"\n{'='*70}")
        print(f"Training ensemble member {i+1}/{num_seeds} (seed={seed})")
        print(f"{'='*70}")

        save_path = os.path.join(save_dir, f'best_model_seed{i}.pt')

        # Skip if already trained
        if os.path.exists(save_path):
            print(f"Found existing checkpoint, loading...")
            model = create_model().to(device)
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            history = {'val_rmse': [checkpoint['val_rmse']]}
        else:
            # Train model
            model, history = train_single_model(
                train_loader, val_loader, device, seed=seed, save_path=save_path
            )

        # Test
        test_metrics, predictions, targets = test(model, test_loader, device)

        ensemble_predictions.append(predictions)
        ensemble_results.append(test_metrics)

        print(f"\nTest Results (Seed {seed}):")
        print(f"  RMSE: {test_metrics['rmse']:.2f}")
        print(f"  MAE: {test_metrics['mae']:.2f}")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  Range Acc: {test_metrics.get('range_accuracy', 0):.4f}")

    # Ensemble prediction (average)
    ensemble_pred = np.mean(ensemble_predictions, axis=0)
    ensemble_metrics = compute_metrics(ensemble_pred, targets)

    print(f"\n{'='*70}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*70}")
    print(f"RMSE: {ensemble_metrics['rmse']:.2f}")
    print(f"MAE: {ensemble_metrics['mae']:.2f}")
    print(f"R²: {ensemble_metrics['r2']:.4f}")
    print(f"\nPer-range RMSE:")
    print(f"  Low (<100): {ensemble_metrics.get('rmse_range_0', 0):.2f} (n={ensemble_metrics.get('count_range_0', 0)})")
    print(f"  Medium (100-200): {ensemble_metrics.get('rmse_range_1', 0):.2f} (n={ensemble_metrics.get('count_range_1', 0)})")
    print(f"  High (>200): {ensemble_metrics.get('rmse_range_2', 0):.2f} (n={ensemble_metrics.get('count_range_2', 0)})")

    # Compile results
    results = {
        'ensemble': ensemble_metrics,
        'individual_members': ensemble_results,
        'num_members': num_seeds
    }

    return results


def plot_results(results: Dict, save_dir: str):
    """Generate diagnostic plots."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Parity plot (ensemble)
    # Need to re-compute ensemble predictions
    # For now, just plot individual member results

    # 2. Per-range RMSE comparison
    ranges = ['Low\n(<100)', 'Medium\n(100-200)', 'High\n(>200)']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Individual members
    ax = axes[0]
    x = np.arange(len(results['individual_members']))
    rmse_values = [m['rmse'] for m in results['individual_members']]

    bars = ax.bar(x, rmse_values, color='steelblue', alpha=0.7)
    ax.axhline(y=results['ensemble']['rmse'], color='red', linestyle='--',
               linewidth=2, label=f"Ensemble: {results['ensemble']['rmse']:.2f}")
    ax.set_xlabel('Ensemble Member', fontsize=12)
    ax.set_ylabel('RMSE (K)', fontsize=12)
    ax.set_title('Test RMSE by Ensemble Member', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seed {42+i}' for i in range(len(results['individual_members']))])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Per-range RMSE
    ax = axes[1]
    x = np.arange(len(ranges))
    width = 0.15

    for i, member_result in enumerate(results['individual_members']):
        rmse_ranges = [
            member_result.get('rmse_range_0', 0),
            member_result.get('rmse_range_1', 0),
            member_result.get('rmse_range_2', 0)
        ]
        ax.bar(x + i * width, rmse_ranges, width, label=f'Member {i+1}', alpha=0.7)

    # Ensemble
    ensemble_rmse_ranges = [
        results['ensemble'].get('rmse_range_0', 0),
        results['ensemble'].get('rmse_range_1', 0),
        results['ensemble'].get('rmse_range_2', 0)
    ]
    ax.bar(x + len(results['individual_members']) * width, ensemble_rmse_ranges,
           width, label='Ensemble', color='red', alpha=0.9)

    ax.set_xlabel('Tg Range', fontsize=12)
    ax.set_ylabel('RMSE (K)', fontsize=12)
    ax.set_title('RMSE by Tg Range', fontsize=14)
    ax.set_xticks(x + width * len(results['individual_members']) / 2)
    ax.set_xticklabels(ranges)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rmse_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plots to {save_dir}")


def main():
    """Main training function."""

    # Configuration
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = '/home/home_public/aisci/AIscientist/archive/ablation4archive/src/alldataset/Tg.csv'
    save_dir = os.path.join(project_dir, 'results')
    num_seeds = 5

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    smiles_list, tg_values = load_data(data_path)

    # Create datasets
    print("\n" + "="*70)
    print("CREATING DATASETS")
    print("="*70)
    train_loader, val_loader, test_loader = create_datasets(
        smiles_list, tg_values, train_ratio=0.72, val_ratio=0.08,
        test_ratio=0.20, seed=42, num_ru=3
    )

    # Train ensemble
    print("\n" + "="*70)
    print("TRAINING ENSEMBLE")
    print("="*70)
    results = train_ensemble(
        train_loader, val_loader, test_loader, device, save_dir, num_seeds=num_seeds
    )

    # Save results
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    print(f"\nSaved results to {results_path}")

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    plot_results(results, save_dir)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Target SOTA RMSE: 30.8 K")
    print(f"Achieved RMSE: {results['ensemble']['rmse']:.2f} K")
    print(f"Improvement over target: {30.8 - results['ensemble']['rmse']:.2f} K" if results['ensemble']['rmse'] < 30.8 else f"Gap to target: {results['ensemble']['rmse'] - 30.8:.2f} K")
    print(f"Previous best (ensemble): RMSE 34.14 K")
    print(f"Improvement over previous: {34.14 - results['ensemble']['rmse']:.2f} K")
    print("="*70)


if __name__ == '__main__':
    main()

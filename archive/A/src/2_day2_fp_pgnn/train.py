"""
Training script for FP-PGNN with 5-fold cross-validation and ablation studies
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dataset import PolymerDataset
from model import FP_PGNN, BaselineGAT, BaselineMLP


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        output = model(batch)
        loss = criterion(output, batch.y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / num_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model and return predictions, targets, and metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        loss = criterion(output, batch.y.view(-1))

        total_loss += loss.item() * batch.num_graphs
        all_preds.extend(output.cpu().numpy())
        all_targets.extend(batch.y.view(-1).cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    avg_loss = total_loss / len(all_targets)

    return avg_loss, rmse, r2, all_preds, all_targets


def train_model(model, train_loader, val_loader, config, device, verbose=True):
    """
    Train model with early stopping.

    Returns:
        best_model_state, training_history
    """
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_r2': []}

    best_model_state = None

    epoch_iter = range(config['epochs']) if not verbose else tqdm(range(config['epochs']), desc="Training")

    for epoch in epoch_iter:
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)

        # Validate
        val_loss, val_rmse, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)

        # Learning rate scheduling
        scheduler.step(val_rmse)

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break

        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val RMSE={val_rmse:.4f}, Val R²={val_r2:.4f}")

    if verbose:
        print(f"  Best Val RMSE: {best_val_rmse:.4f} at epoch {best_epoch + 1}")

    return best_model_state, history, best_val_rmse


def run_cross_validation(dataset, config, model_type='fp_pgnn',
                         use_periodic=True, use_fingerprint=True, device='cuda'):
    """
    Run 5-fold cross-validation.

    Returns:
        Dictionary with fold results and summary statistics
    """
    n_samples = len(dataset)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(n_samples))):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/5")
        print(f"{'='*60}")

        # Create data loaders
        train_dataset = dataset[train_idx.tolist()]
        val_dataset = dataset[val_idx.tolist()]

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # Create model
        if model_type == 'fp_pgnn':
            model = FP_PGNN(
                node_dim=28,
                fp_dim=2048,
                hidden_dim=config['hidden_dim'],
                edge_dim=5,
                num_gat_layers=config['num_layers'],
                heads=config['heads'],
                dropout=config['dropout'],
                use_periodic=use_periodic,
                use_fingerprint=use_fingerprint
            ).to(device)
        elif model_type == 'gat':
            model = BaselineGAT(
                node_dim=28,
                hidden_dim=config['hidden_dim'],
                edge_dim=5,
                num_layers=config['num_layers'],
                heads=config['heads'],
                dropout=config['dropout']
            ).to(device)
        elif model_type == 'mlp':
            model = BaselineMLP(
                fp_dim=2048,
                hidden_dim=config['hidden_dim'] * 2,
                dropout=config['dropout']
            ).to(device)

        # Train
        best_state, history, best_val_rmse = train_model(
            model, train_loader, val_loader, config, device, verbose=True
        )

        # Load best model and evaluate
        model.load_state_dict(best_state)
        criterion = nn.MSELoss()
        _, final_rmse, final_r2, preds, targets = evaluate(model, val_loader, criterion, device)

        fold_results.append({
            'fold': fold + 1,
            'val_rmse': float(final_rmse),
            'val_r2': float(final_r2),
            'best_epoch': len(history['val_rmse']) - 20,  # Approximate
            'predictions': preds.tolist(),
            'targets': targets.tolist()
        })

        # Save model checkpoint
        torch.save(best_state, f'checkpoint_fold{fold + 1}_{model_type}.pt')

    # Compute summary statistics
    all_rmse = [r['val_rmse'] for r in fold_results]
    all_r2 = [r['val_r2'] for r in fold_results]

    summary = {
        'model_type': model_type,
        'use_periodic': use_periodic,
        'use_fingerprint': use_fingerprint,
        'mean_rmse': float(np.mean(all_rmse)),
        'std_rmse': float(np.std(all_rmse)),
        'mean_r2': float(np.mean(all_r2)),
        'std_r2': float(np.std(all_r2)),
        'fold_results': fold_results
    }

    print(f"\n{'='*60}")
    print(f"Cross-validation Summary for {model_type}")
    print(f"{'='*60}")
    print(f"Mean RMSE: {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}")
    print(f"Mean R²: {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")

    return summary


def main():
    # Configuration
    config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'heads': 4,
        'dropout': 0.1,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'epochs': 200,
        'patience': 20
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    csv_path = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"

    all_results = {}

    # ========== ABLATION STUDY 1: Full FP-PGNN ==========
    print("\n" + "="*80)
    print("ABLATION STUDY 1: Full FP-PGNN (Periodic + Fingerprint)")
    print("="*80)

    dataset_full = PolymerDataset(csv_path, use_periodic=True, use_fingerprint=True)
    results_full = run_cross_validation(
        dataset_full, config, model_type='fp_pgnn',
        use_periodic=True, use_fingerprint=True, device=device
    )
    all_results['fp_pgnn_full'] = results_full

    # ========== ABLATION STUDY 2: No Periodic Edges ==========
    print("\n" + "="*80)
    print("ABLATION STUDY 2: FP-PGNN without Periodic Edges")
    print("="*80)

    dataset_no_periodic = PolymerDataset(csv_path, use_periodic=False, use_fingerprint=True)
    results_no_periodic = run_cross_validation(
        dataset_no_periodic, config, model_type='fp_pgnn',
        use_periodic=False, use_fingerprint=True, device=device
    )
    all_results['fp_pgnn_no_periodic'] = results_no_periodic

    # ========== ABLATION STUDY 3: No Fingerprint ==========
    print("\n" + "="*80)
    print("ABLATION STUDY 3: FP-PGNN without Fingerprint (Periodic GAT only)")
    print("="*80)

    dataset_no_fp = PolymerDataset(csv_path, use_periodic=True, use_fingerprint=False)
    results_no_fp = run_cross_validation(
        dataset_no_fp, config, model_type='fp_pgnn',
        use_periodic=True, use_fingerprint=False, device=device
    )
    all_results['fp_pgnn_no_fingerprint'] = results_no_fp

    # ========== ABLATION STUDY 4: Pure GAT (No Periodic, No Fingerprint) ==========
    print("\n" + "="*80)
    print("ABLATION STUDY 4: Pure GAT (No Periodic, No Fingerprint)")
    print("="*80)

    dataset_gat = PolymerDataset(csv_path, use_periodic=False, use_fingerprint=False)
    results_gat = run_cross_validation(
        dataset_gat, config, model_type='gat',
        use_periodic=False, use_fingerprint=False, device=device
    )
    all_results['gat_baseline'] = results_gat

    # ========== BASELINE: MLP with ECFP4 ==========
    print("\n" + "="*80)
    print("BASELINE: MLP with ECFP4 Fingerprint")
    print("="*80)

    dataset_mlp = PolymerDataset(csv_path, use_periodic=False, use_fingerprint=True)
    results_mlp = run_cross_validation(
        dataset_mlp, config, model_type='mlp',
        use_periodic=False, use_fingerprint=True, device=device
    )
    all_results['mlp_baseline'] = results_mlp

    # Save all results
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    # Print comparison table
    print(f"\n{'Model':<30} {'RMSE':<15} {'R²':<15}")
    print("-" * 60)
    for name, results in all_results.items():
        print(f"{name:<30} {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}   "
              f"{results['mean_r2']:.4f} ± {results['std_r2']:.4f}")

    # Compute ablation effects
    full_rmse = all_results['fp_pgnn_full']['mean_rmse']

    print(f"\n--- Ablation Analysis ---")
    print(f"Full FP-PGNN: {full_rmse:.4f}")
    print(f"Without Periodic: {all_results['fp_pgnn_no_periodic']['mean_rmse']:.4f} "
          f"(Δ = {all_results['fp_pgnn_no_periodic']['mean_rmse'] - full_rmse:+.4f})")
    print(f"Without Fingerprint: {all_results['fp_pgnn_no_fingerprint']['mean_rmse']:.4f} "
          f"(Δ = {all_results['fp_pgnn_no_fingerprint']['mean_rmse'] - full_rmse:+.4f})")
    print(f"Pure GAT: {all_results['gat_baseline']['mean_rmse']:.4f} "
          f"(Δ = {all_results['gat_baseline']['mean_rmse'] - full_rmse:+.4f})")
    print(f"MLP Baseline: {all_results['mlp_baseline']['mean_rmse']:.4f} "
          f"(Δ = {all_results['mlp_baseline']['mean_rmse'] - full_rmse:+.4f})")

    # Check success criteria
    print(f"\n--- Success Criteria ---")
    if full_rmse < 37.0:
        print(f"✓ Primary: RMSE < 37.0 (Achieved: {full_rmse:.4f})")
    else:
        print(f"✗ Primary: RMSE < 37.0 (Achieved: {full_rmse:.4f})")

    periodic_contrib = all_results['fp_pgnn_no_periodic']['mean_rmse'] - full_rmse
    fp_contrib = all_results['fp_pgnn_no_fingerprint']['mean_rmse'] - full_rmse

    if periodic_contrib > 1.0 and fp_contrib > 1.0:
        print(f"✓ Secondary: Both components contribute >1 RMSE")
        print(f"  Periodic contribution: {periodic_contrib:+.4f}")
        print(f"  Fingerprint contribution: {fp_contrib:+.4f}")
    else:
        print(f"✗ Secondary: Both components contribute >1 RMSE")
        print(f"  Periodic contribution: {periodic_contrib:+.4f}")
        print(f"  Fingerprint contribution: {fp_contrib:+.4f}")

    print(f"\nResults saved to results.json")


if __name__ == "__main__":
    main()

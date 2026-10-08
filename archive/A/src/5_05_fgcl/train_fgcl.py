"""
Training script for FGCL with comprehensive ablation studies
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from fgcl_model import FGCLModel, info_nce_loss, count_parameters
from data_utils import load_tg_dataset, get_kfold_splits


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_contrastive_epoch(model, loader, optimizer, device, temp=0.1):
    """
    Train one epoch with contrastive learning only

    Returns:
        Average contrastive loss
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        # Get projection embeddings
        projections = model(batch, return_projection=True)

        # Compute InfoNCE loss
        loss = info_nce_loss(projections, temp=temp)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def train_supervised_epoch(model, loader, optimizer, device, lambda_contrastive=0.0, temp=0.1):
    """
    Train one epoch with supervised loss + optional contrastive regularization

    Returns:
        (avg_supervised_loss, avg_contrastive_loss, avg_total_loss)
    """
    model.train()
    total_supervised = 0
    total_contrastive = 0
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        # Supervised prediction
        predictions = model(batch, return_projection=False)
        supervised_loss = nn.MSELoss()(predictions, batch.y.squeeze())

        # Contrastive loss (optional)
        if lambda_contrastive > 0:
            projections = model(batch, return_projection=True)
            contrastive_loss = info_nce_loss(projections, temp=temp)
            loss = supervised_loss + lambda_contrastive * contrastive_loss
            total_contrastive += contrastive_loss.item()
        else:
            loss = supervised_loss
            total_contrastive += 0.0

        loss.backward()
        optimizer.step()

        total_supervised += supervised_loss.item()
        total_loss += loss.item()
        num_batches += 1

    avg_supervised = total_supervised / num_batches if num_batches > 0 else 0.0
    avg_contrastive = total_contrastive / num_batches if num_batches > 0 else 0.0
    avg_total = total_loss / num_batches if num_batches > 0 else 0.0

    return avg_supervised, avg_contrastive, avg_total


def evaluate(model, loader, device):
    """
    Evaluate model on test set

    Returns:
        dict with RMSE, MAE, R2, and predictions
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, return_projection=False)

            predictions.extend(pred.cpu().numpy())
            targets.extend(batch.y.squeeze().cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }


def get_embeddings(model, loader, device):
    """
    Extract graph embeddings for visualization
    """
    model.eval()
    embeddings = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model.get_graph_embedding(batch)

            embeddings.extend(emb.cpu().numpy())
            targets.extend(batch.y.squeeze().cpu().numpy())

    return np.array(embeddings), np.array(targets)


def train_fgcl_fold(
    fold_idx,
    train_dataset,
    test_dataset,
    config,
    device,
    ablation_mode='full',
    lambda_contrastive=0.5,
    pretrain_epochs=10,
    finetune_epochs=100,
    batch_size=32,
    lr=1e-3,
    temp=0.1,
    patience=20,
    seed=42
):
    """
    Train FGCL for one fold with specified ablation mode

    Args:
        fold_idx: Fold index
        train_dataset: Training dataset
        test_dataset: Test dataset
        config: Model configuration
        device: Device to use
        ablation_mode: One of ['supervised_only', 'pretrain_only', 'regularization_only', 'full']
        lambda_contrastive: Weight for contrastive loss during fine-tuning
        pretrain_epochs: Number of pre-training epochs
        finetune_epochs: Number of fine-tuning epochs
        batch_size: Batch size
        lr: Learning rate
        temp: Temperature for InfoNCE
        patience: Early stopping patience
        seed: Random seed

    Returns:
        dict with results
    """
    set_seed(seed)

    print(f"\n{'='*80}")
    print(f"Fold {fold_idx + 1} - Ablation: {ablation_mode}, Lambda: {lambda_contrastive}")
    print(f"{'='*80}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = FGCLModel(config).to(device)

    # Results tracking
    results = {
        'fold': fold_idx + 1,
        'ablation_mode': ablation_mode,
        'lambda_contrastive': lambda_contrastive,
        'pretrain_losses': [],
        'finetune_supervised_losses': [],
        'finetune_contrastive_losses': [],
        'finetune_total_losses': [],
        'val_rmses': [],
        'best_val_rmse': float('inf'),
        'test_rmse': None,
        'test_mae': None,
        'test_r2': None
    }

    # Split train into train/val for early stopping
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size

    train_subset = train_dataset[:train_size]
    val_subset = train_dataset[train_size:]

    train_loader_split = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Phase 1: Contrastive Pre-training (if applicable)
    if ablation_mode in ['pretrain_only', 'full']:
        print(f"\n--- Phase 1: Contrastive Pre-training ({pretrain_epochs} epochs) ---")
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(pretrain_epochs):
            loss = train_contrastive_epoch(model, train_loader_split, optimizer, device, temp)
            results['pretrain_losses'].append(loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{pretrain_epochs} - Contrastive Loss: {loss:.4f}")

        print(f"Pre-training completed. Final loss: {results['pretrain_losses'][-1]:.4f}")
    else:
        print(f"\n--- Skipping contrastive pre-training (ablation: {ablation_mode}) ---")

    # Phase 2: Fine-tuning
    print(f"\n--- Phase 2: Fine-tuning ({finetune_epochs} epochs) ---")

    # Reset optimizer for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Determine lambda for this mode
    if ablation_mode == 'supervised_only':
        lambda_finetune = 0.0
    elif ablation_mode == 'pretrain_only':
        lambda_finetune = 0.0
    elif ablation_mode == 'regularization_only':
        lambda_finetune = lambda_contrastive
    elif ablation_mode == 'full':
        lambda_finetune = lambda_contrastive
    else:
        lambda_finetune = lambda_contrastive

    print(f"Lambda for fine-tuning: {lambda_finetune}")

    best_val_rmse = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(finetune_epochs):
        # Train
        sup_loss, cont_loss, total_loss = train_supervised_epoch(
            model, train_loader_split, optimizer, device, lambda_finetune, temp
        )

        results['finetune_supervised_losses'].append(sup_loss)
        results['finetune_contrastive_losses'].append(cont_loss)
        results['finetune_total_losses'].append(total_loss)

        # Validate
        val_results = evaluate(model, val_loader, device)
        val_rmse = val_results['rmse']
        results['val_rmses'].append(val_rmse)

        scheduler.step(val_rmse)

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{finetune_epochs} - "
                  f"Sup Loss: {sup_loss:.4f}, "
                  f"Cont Loss: {cont_loss:.4f}, "
                  f"Val RMSE: {val_rmse:.4f}, "
                  f"Best: {best_val_rmse:.4f} @ epoch {best_epoch}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    print(f"\n--- Final Evaluation on Test Set ---")
    test_results = evaluate(model, test_loader, device)

    results['best_val_rmse'] = best_val_rmse
    results['best_epoch'] = best_epoch
    results['test_rmse'] = test_results['rmse']
    results['test_mae'] = test_results['mae']
    results['test_r2'] = test_results['r2']
    results['predictions'] = test_results['predictions']
    results['targets'] = test_results['targets']

    print(f"Best Val RMSE: {best_val_rmse:.4f} @ epoch {best_epoch}")
    print(f"Test RMSE: {test_results['rmse']:.4f}")
    print(f"Test MAE: {test_results['mae']:.4f}")
    print(f"Test R2: {test_results['r2']:.4f}")

    return results, model


def run_ablation_studies(
    dataset,
    config,
    device,
    output_dir='results',
    n_folds=5,
    pretrain_epochs=10,
    finetune_epochs=100,
    batch_size=32,
    lr=1e-3,
    temp=0.1,
    seed=42
):
    """
    Run all ablation studies across 5 folds
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get fold splits
    splits = get_kfold_splits(dataset, n_splits=n_folds, random_seed=seed)

    # Ablation modes
    ablation_modes = [
        'supervised_only',
        'pretrain_only',
        'regularization_only',
        'full'
    ]

    # Lambda values to test
    lambda_values = [0.1, 0.5, 1.0, 2.0]

    # Results storage
    all_results = []

    print(f"\n{'#'*80}")
    print(f"# Starting FGCL Ablation Studies")
    print(f"# Dataset size: {len(dataset)}")
    print(f"# Folds: {n_folds}")
    print(f"# Ablation modes: {ablation_modes}")
    print(f"# Lambda values: {lambda_values}")
    print(f"{'#'*80}\n")

    # Run ablation modes (a-d)
    for mode in ablation_modes:
        print(f"\n{'='*80}")
        print(f"Ablation Mode: {mode}")
        print(f"{'='*80}")

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_dataset = [dataset[i] for i in train_idx]
            test_dataset = [dataset[i] for i in test_idx]

            # Use lambda=0.5 for modes that need it
            lambda_val = 0.5 if mode in ['regularization_only', 'full'] else 0.0

            results, _ = train_fgcl_fold(
                fold_idx=fold_idx,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                config=config,
                device=device,
                ablation_mode=mode,
                lambda_contrastive=lambda_val,
                pretrain_epochs=pretrain_epochs,
                finetune_epochs=finetune_epochs,
                batch_size=batch_size,
                lr=lr,
                temp=temp,
                seed=seed
            )

            all_results.append(results)

    # Run lambda ablation (e)
    print(f"\n{'='*80}")
    print(f"Lambda Ablation Study")
    print(f"{'='*80}")

    for lambda_val in lambda_values:
        print(f"\n--- Lambda = {lambda_val} ---")

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_dataset = [dataset[i] for i in train_idx]
            test_dataset = [dataset[i] for i in test_idx]

            results, _ = train_fgcl_fold(
                fold_idx=fold_idx,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                config=config,
                device=device,
                ablation_mode='full',
                lambda_contrastive=lambda_val,
                pretrain_epochs=pretrain_epochs,
                finetune_epochs=finetune_epochs,
                batch_size=batch_size,
                lr=lr,
                temp=temp,
                seed=seed
            )

            # Mark as lambda ablation
            results['ablation_mode'] = f'full_lambda_{lambda_val}'
            all_results.append(results)

    # Save results
    results_df = pd.DataFrame([
        {
            'fold': r['fold'],
            'ablation_mode': r['ablation_mode'],
            'lambda_contrastive': r['lambda_contrastive'],
            'best_epoch': r['best_epoch'],
            'best_val_rmse': r['best_val_rmse'],
            'test_rmse': r['test_rmse'],
            'test_mae': r['test_mae'],
            'test_r2': r['test_r2']
        }
        for r in all_results
    ])

    results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)

    # Compute summary statistics
    summary = results_df.groupby('ablation_mode').agg({
        'test_rmse': ['mean', 'std', 'min', 'max'],
        'test_mae': ['mean', 'std'],
        'test_r2': ['mean', 'std']
    }).round(4)

    summary.to_csv(os.path.join(output_dir, 'summary.csv'))

    print(f"\n{'='*80}")
    print(f"Summary of Results")
    print(f"{'='*80}")
    print(summary)

    # Save detailed results as JSON
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_for_json = []
        for r in all_results:
            r_copy = r.copy()
            r_copy['predictions'] = r_copy['predictions'].tolist()
            r_copy['targets'] = r_copy['targets'].tolist()
            results_for_json.append(r_copy)

        json.dump(results_for_json, f, indent=2)

    return all_results, summary


def main():
    """Main training function"""

    # Configuration
    config = {
        'num_layers': 2,
        'hidden_dim': 128,
        'node_dim': 32,
        'projection_dim': 32,
        'ecfp_dim': 2048
    }

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_tg_dataset('/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv')

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample: {dataset[0]}")

    # Run ablation studies
    all_results, summary = run_ablation_studies(
        dataset=dataset,
        config=config,
        device=device,
        output_dir='results',
        n_folds=5,
        pretrain_epochs=10,
        finetune_epochs=100,
        batch_size=32,
        lr=1e-3,
        temp=0.1,
        seed=42
    )

    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Results saved to: results/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

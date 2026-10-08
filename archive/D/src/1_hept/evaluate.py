"""
Evaluation and diagnostic analysis for HEPT model.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from data_loader import get_dataloaders, PolymerTgDataset, stratified_split
from model import HEPT, HEPT_NoEquivariant, HEPT_NoPeriodic, HEPT_NoLongRange, SimpleGNN


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Pearson correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]

    # Bias (mean error)
    bias = np.mean(predictions - targets)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mse": float(mse),
        "correlation": float(correlation),
        "bias": float(bias)
    }


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Get predictions and targets from data loader."""
    model.eval()

    all_predictions = []
    all_targets = []
    all_smiles = []

    for batch in data_loader:
        batch = batch.to(device)

        predictions = model(batch.x, batch.pos, batch.edge_index, batch.batch)

        all_predictions.append(predictions.squeeze().cpu().numpy())
        all_targets.append(batch.y.squeeze().cpu().numpy())
        all_smiles.extend(batch.smiles)

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return predictions, targets, all_smiles


# ============================================================================
# Diagnostic Analysis
# ============================================================================

def plot_predicted_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    title: str = "Predicted vs Actual Tg"
):
    """Plot predicted vs actual Tg values."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(targets, predictions, alpha=0.5, s=20, edgecolors='none')

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    # Compute metrics
    metrics = compute_metrics(predictions, targets)

    # Add metrics text
    textstr = f'RMSE: {metrics["rmse"]:.2f}\nMAE: {metrics["mae"]:.2f}\nR²: {metrics["r2"]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ax.set_xlabel('Actual Tg (°C)', fontsize=12)
    ax.set_ylabel('Predicted Tg (°C)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {output_path}")


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    title: str = "Error Distribution"
):
    """Plot error distribution."""
    errors = predictions - targets

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of errors
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2,
                    label=f'Mean: {np.mean(errors):.2f}')
    axes[0].set_xlabel('Prediction Error (°C)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Error Histogram', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error vs actual value
    axes[1].scatter(targets, errors, alpha=0.5, s=20, edgecolors='none')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Actual Tg (°C)', fontsize=12)
    axes[1].set_ylabel('Prediction Error (°C)', fontsize=12)
    axes[1].set_title('Error vs Actual Tg', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {output_path}")


def analyze_by_tg_range(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str
):
    """Analyze performance by Tg range."""
    # Define ranges
    ranges = [
        (-200, -50, "Very Low (< -50°C)"),
        (-50, 0, "Low (-50 to 0°C)"),
        (0, 50, "Medium (0 to 50°C)"),
        (50, 100, "Medium-High (50 to 100°C)"),
        (100, 150, "High (100 to 150°C)"),
        (150, 300, "Very High (> 150°C)")
    ]

    results = []

    for low, high, label in ranges:
        mask = (targets >= low) & (targets < high)
        if mask.sum() > 0:
            preds_range = predictions[mask]
            targets_range = targets[mask]
            metrics = compute_metrics(preds_range, targets_range)
            results.append({
                "range": label,
                "count": mask.sum(),
                **metrics
            })

    # Create bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ranges_labels = [r["range"] for r in results]
    counts = [r["count"] for r in results]
    rmses = [r["rmse"] for r in results]

    # Count per range
    axes[0].bar(range(len(ranges_labels)), counts, color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(ranges_labels)))
    axes[0].set_xticklabels(ranges_labels, rotation=45, ha='right')
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Samples per Tg Range', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')

    # RMSE per range
    axes[1].bar(range(len(ranges_labels)), rmses, color='coral', edgecolor='black')
    axes[1].set_xticks(range(len(ranges_labels)))
    axes[1].set_xticklabels(ranges_labels, rotation=45, ha='right')
    axes[1].set_ylabel('RMSE (°C)', fontsize=12)
    axes[1].set_title('RMSE per Tg Range', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save detailed results
    df = pd.DataFrame(results)
    df.to_csv(output_path.replace('.png', '.csv'), index=False)

    print(f"Saved analysis to {output_path}")
    return results


def analyze_aromatic_vs_aliphatic(
    predictions: np.ndarray,
    targets: np.ndarray,
    smiles_list: List[str],
    output_path: str
):
    """Analyze performance on aromatic vs aliphatic polymers."""
    from rdkit import Chem

    aromatic_mask = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                is_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
                aromatic_mask.append(is_aromatic)
            else:
                aromatic_mask.append(False)
        except:
            aromatic_mask.append(False)

    aromatic_mask = np.array(aromatic_mask)
    aliphatic_mask = ~aromatic_mask

    results = {}

    # Aromatic
    if aromatic_mask.sum() > 0:
        results["aromatic"] = {
            "count": int(aromatic_mask.sum()),
            **compute_metrics(predictions[aromatic_mask], targets[aromatic_mask])
        }

    # Aliphatic
    if aliphatic_mask.sum() > 0:
        results["aliphatic"] = {
            "count": int(aliphatic_mask.sum()),
            **compute_metrics(predictions[aliphatic_mask], targets[aliphatic_mask])
        }

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    categories = list(results.keys())
    counts = [results[c]["count"] for c in categories]
    rmses = [results[c]["rmse"] for c in categories]
    r2s = [results[c]["r2"] for c in categories]

    x = range(len(categories))

    # Count
    axes[0].bar(x, counts, color=['steelblue', 'coral'], edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Aromatic', 'Aliphatic'])
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Sample Count', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')

    # RMSE
    axes[1].bar(x, rmses, color=['steelblue', 'coral'], edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Aromatic', 'Aliphatic'])
    axes[1].set_ylabel('RMSE (°C)', fontsize=12)
    axes[1].set_title('RMSE Comparison', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    # R²
    axes[2].bar(x, r2s, color=['steelblue', 'coral'], edgecolor='black')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['Aromatic', 'Aliphatic'])
    axes[2].set_ylabel('R²', fontsize=12)
    axes[2].set_title('R² Comparison', fontsize=14)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Aromatic vs Aliphatic Polymer Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved analysis to {output_path}")
    return results


def identify_outliers(
    predictions: np.ndarray,
    targets: np.ndarray,
    smiles_list: List[str],
    output_path: str,
    threshold: float = 50.0
):
    """Identify prediction outliers."""
    errors = np.abs(predictions - targets)

    outlier_mask = errors > threshold
    outlier_indices = np.where(outlier_mask)[0]

    results = []
    for idx in outlier_indices:
        results.append({
            "smiles": smiles_list[idx],
            "actual_tg": float(targets[idx]),
            "predicted_tg": float(predictions[idx]),
            "error": float(errors[idx])
        })

    # Sort by error
    results = sorted(results, key=lambda x: x["error"], reverse=True)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"Found {len(results)} outliers (error > {threshold}°C)")
    print(f"Saved outliers to {output_path}")

    return results


def plot_training_history(
    history: Dict[str, List],
    output_path: str
):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Training and validation loss
    axes[0, 0].plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation RMSE
    axes[0, 1].plot(epochs, history["val_rmse"], 'g-')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('RMSE (°C)', fontsize=12)
    axes[0, 1].set_title('Validation RMSE', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # Validation MAE
    axes[1, 0].plot(epochs, history["val_mae"], 'm-')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('MAE (°C)', fontsize=12)
    axes[1, 0].set_title('Validation MAE', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, history["lr"], 'c-')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved training history plot to {output_path}")


# ============================================================================
# Main Evaluation Function
# ============================================================================

def run_full_evaluation(
    checkpoint_path: str,
    output_dir: str = "analysis",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Run full evaluation pipeline."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint_path}")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint.get("config", {})

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    # Get input dimension
    for batch in train_loader:
        input_dim = batch.x.size(-1)
        break

    # Create model
    model = HEPT(
        input_dim=input_dim,
        hidden_dim=config_dict.get("hidden_dim", 256),
        output_dim=1,
        num_equivariant_layers=config_dict.get("num_equivariant_layers", 3),
        num_periodic_layers=config_dict.get("num_periodic_layers", 2),
        num_heads=config_dict.get("num_heads", 8),
        dropout=config_dict.get("dropout", 0.1)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Get predictions
    print("Getting predictions...")
    predictions, targets, smiles_list = get_predictions(model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Correlation: {metrics['correlation']:.4f}")

    # Generate plots
    print("\nGenerating diagnostic plots...")

    # 1. Predicted vs Actual
    plot_predicted_vs_actual(
        predictions, targets,
        output_dir / "predicted_vs_actual.png",
        title="HEPT: Predicted vs Actual Tg"
    )

    # 2. Error distribution
    plot_error_distribution(
        predictions, targets,
        output_dir / "error_distribution.png",
        title="HEPT: Error Distribution"
    )

    # 3. Analysis by Tg range
    analyze_by_tg_range(
        predictions, targets,
        output_dir / "tg_range_analysis.png"
    )

    # 4. Aromatic vs Aliphatic
    analyze_aromatic_vs_aliphatic(
        predictions, targets, smiles_list,
        output_dir / "aromatic_vs_aliphatic.png"
    )

    # 5. Identify outliers
    identify_outliers(
        predictions, targets, smiles_list,
        output_dir / "outliers.csv",
        threshold=50.0
    )

    # Save final results
    results = {
        "model": "HEPT",
        "test_metrics": metrics,
        "num_test_samples": len(targets),
        "checkpoint_path": str(checkpoint_path),
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    predictions_df = pd.DataFrame({
        "smiles": smiles_list,
        "actual_tg": targets,
        "predicted_tg": predictions,
        "error": predictions - targets
    })
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"\nEvaluation complete. Results saved to {output_dir}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hept_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="analysis",
                        help="Output directory for analysis")
    args = parser.parse_args()

    run_full_evaluation(args.checkpoint, args.output)

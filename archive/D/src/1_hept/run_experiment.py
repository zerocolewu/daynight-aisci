"""
Complete experiment runner for HEPT polymer Tg prediction.
Runs data loading, training, evaluation, and ablation studies.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
import torch

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Import local modules
from data_loader import get_dataloaders, PolymerTgDataset, stratified_split, collate_fn
from model import HEPT, HEPT_NoEquivariant, HEPT_NoPeriodic, HEPT_NoLongRange, SimpleGNN
from train import Config, train, count_parameters
from evaluate import run_full_evaluation, compute_metrics, get_predictions


def main():
    """Main experiment function."""

    print("="*80)
    print("HEPT Experiment: Hierarchical Equivariant Periodic Transformer")
    print("Target: Beat PerioGT baseline RMSE=30.8 on Tg dataset")
    print("="*80)

    # Configuration
    config = Config()
    config.max_epochs = 200
    config.patience = 25
    config.batch_size = 32
    config.learning_rate = 1e-3
    config.hidden_dim = 256
    config.num_equivariant_layers = 3
    config.num_periodic_layers = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Configuration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Patience: {config.patience}")

    # Create output directories
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = Path("analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Train main HEPT model
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING HEPT MODEL")
    print("="*80)

    start_time = time.time()
    model, results = train(config, model_name="hept")
    training_time = time.time() - start_time

    print(f"\nHEPT Training completed in {training_time/60:.1f} minutes")
    print(f"Best validation RMSE: {results['best_val_rmse']:.2f}")
    print(f"Test RMSE: {results['test_rmse']:.2f}")
    print(f"Test MAE: {results['test_mae']:.2f}")
    print(f"Test R²: {results['test_r2']:.4f}")

    # =========================================================================
    # Run ablation studies
    # =========================================================================
    ablation_results = {}

    # Ablation 1: Without equivariant module
    print("\n" + "="*80)
    print("ABLATION 1: WITHOUT EQUIVARIANT MODULE")
    print("="*80)

    config_ablation = Config()
    config_ablation.max_epochs = 150
    config_ablation.patience = 20
    config_ablation.batch_size = 32
    config_ablation.hidden_dim = 256

    try:
        _, ablation_results['no_equivariant'] = train(config_ablation, model_name="hept_no_equiv")
    except Exception as e:
        print(f"Ablation failed: {e}")
        ablation_results['no_equivariant'] = {"test_rmse": float('nan'), "error": str(e)}

    # Ablation 2: Without periodic attention
    print("\n" + "="*80)
    print("ABLATION 2: WITHOUT PERIODIC ATTENTION")
    print("="*80)

    try:
        _, ablation_results['no_periodic'] = train(config_ablation, model_name="hept_no_periodic")
    except Exception as e:
        print(f"Ablation failed: {e}")
        ablation_results['no_periodic'] = {"test_rmse": float('nan'), "error": str(e)}

    # Ablation 3: Without long-range module
    print("\n" + "="*80)
    print("ABLATION 3: WITHOUT LONG-RANGE MODULE")
    print("="*80)

    try:
        _, ablation_results['no_longrange'] = train(config_ablation, model_name="hept_no_longrange")
    except Exception as e:
        print(f"Ablation failed: {e}")
        ablation_results['no_longrange'] = {"test_rmse": float('NaN'), "error": str(e)}

    # Ablation 4: Simple GNN baseline
    print("\n" + "="*80)
    print("ABLATION 4: SIMPLE GNN BASELINE")
    print("="*80)

    try:
        _, ablation_results['simple_gnn'] = train(config_ablation, model_name="simple_gnn")
    except Exception as e:
        print(f"Ablation failed: {e}")
        ablation_results['simple_gnn'] = {"test_rmse": float('NaN'), "error": str(e)}

    # =========================================================================
    # Compile final results
    # =========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    final_results = {
        "timestamp": datetime.now().isoformat(),
        "model": "HEPT",
        "main_model": {
            "test_rmse": results["test_rmse"],
            "test_mae": results["test_mae"],
            "test_r2": results["test_r2"],
            "best_val_rmse": results["best_val_rmse"],
            "best_epoch": results["best_epoch"],
            "training_time_minutes": training_time / 60,
            "num_parameters": results["num_parameters"]
        },
        "ablation_studies": {
            name: {
                "test_rmse": r.get("test_rmse", float('NaN')),
                "test_mae": r.get("test_mae", float('NaN')),
                "test_r2": r.get("test_r2", float('NaN'))
            }
            for name, r in ablation_results.items()
        },
        "baseline_comparison": {
            "GAT_baseline_rmse": 50.5,
            "PerioGT_baseline_rmse": 30.8,
            "HEPT_rmse": results["test_rmse"],
            "improvement_vs_PerioGT": 30.8 - results["test_rmse"],
            "percent_improvement": (30.8 - results["test_rmse"]) / 30.8 * 100
        },
        "config": {
            "hidden_dim": config.hidden_dim,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_epochs": config.max_epochs,
            "num_equivariant_layers": config.num_equivariant_layers,
            "num_periodic_layers": config.num_periodic_layers,
            "num_heads": config.num_heads,
            "dropout": config.dropout
        }
    }

    # Save results
    with open("results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    print("\n" + "-"*60)
    print("SUMMARY")
    print("-"*60)
    print(f"HEPT Test RMSE: {results['test_rmse']:.2f}")
    print(f"HEPT Test MAE: {results['test_mae']:.2f}")
    print(f"HEPT Test R²: {results['test_r2']:.4f}")
    print()
    print("Baseline Comparison:")
    print(f"  GAT baseline:      RMSE = 50.5")
    print(f"  PerioGT baseline:  RMSE = 30.8")
    print(f"  HEPT:              RMSE = {results['test_rmse']:.2f}")
    print()
    print(f"Improvement vs PerioGT: {30.8 - results['test_rmse']:.2f} ({(30.8 - results['test_rmse'])/30.8*100:.1f}%)")
    print()
    print("Ablation Results:")
    for name, r in ablation_results.items():
        rmse = r.get("test_rmse", float('NaN'))
        if not np.isnan(rmse):
            print(f"  {name}: RMSE = {rmse:.2f}")

    # =========================================================================
    # Generate analysis plots
    # =========================================================================
    print("\n" + "-"*60)
    print("GENERATING ANALYSIS PLOTS")
    print("-"*60)

    try:
        run_full_evaluation("checkpoints/hept_best.pt", "analysis")
    except Exception as e:
        print(f"Analysis generation failed: {e}")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print(f"Results saved to results.json")
    print(f"Checkpoints saved to checkpoints/")
    print(f"Analysis saved to analysis/")
    print("="*80)

    # Determine success
    success = results['test_rmse'] < 30.8
    if success:
        print("\n*** SUCCESS: HEPT beats PerioGT baseline! ***")
    else:
        print(f"\n*** Target not reached. Gap: {results['test_rmse'] - 30.8:.2f} ***")

    return final_results


if __name__ == "__main__":
    results = main()

"""
HEPT Iteration 2 - Improved Training Experiment
This script runs the full experiment with improved data loading and model architecture.
"""

import os
import sys
import json
import time
import copy
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch
from rdkit import RDLogger

# Suppress warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# Local imports
from data_loader import get_dataloaders, stratified_split, collate_fn
from model import HEPT
from train import Config, count_parameters, train
from evaluate import run_full_evaluation


def run_iteration2():
    """
    Run improved HEPT training with:
    1. MMFF-optimized conformers
    2. Proper periodic images (1x1x1 supercell)
    3. Enhanced molecular features
    4. Better regularization
    """
    print("=" * 80)
    print("HEPT Iteration 2 - Improved Training")
    print("Target: Beat PerioGT baseline RMSE=30.8 on Tg dataset")
    print("=" * 80)

    # Configuration with improved settings
    config = Config()
    config.hidden_dim = 384  # Increased from 256
    config.num_equivariant_layers = 4  # More equivariant layers
    config.num_periodic_layers = 3  # More periodic layers
    config.gradient_clip = 0.5  # Tighter gradient clipping
    config.num_periodic_images = 1  # 1x1x1 supercell (reduced for memory)
    config.dropout = 0.15  # Increased dropout
    config.weight_decay = 1e-4  # Increased weight decay
    config.max_epochs = 300  # More training epochs
    config.batch_size = 24  # Reduced batch size for memory
    config.learning_rate = 5e-4  # Lower learning rate
    config.patience = 25  # More patience
    config.num_workers = 0
    config.seed = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Configuration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Patience: {config.patience}")
    print(f"  Gradient clip: {config.gradient_clip}")

    # Create directories
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = Path("analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Train
    print("\nStarting training...")
    start_time = time.time()

    model, results = train(config, model_name="hept")

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    print(f"Best validation RMSE: {results['best_val_rmse']:.2f}")
    print(f"Test RMSE: {results['test_rmse']:.2f}")

    # Save results
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "model": "HEPT_v2_Iteration2",
        "experiment": "Polymer Tg Prediction",
        "dataset": {
            "path": "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv",
            "total_samples": 7170,
        },
        "model_config": {
            "hidden_dim": config.hidden_dim,
            "num_equivariant_layers": config.num_equivariant_layers,
            "num_periodic_layers": config.num_periodic_layers,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
            "weight_decay": config.weight_decay,
            "gradient_clip": config.gradient_clip,
            "num_parameters": results["num_parameters"]
        },
        "training": {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "best_epoch": results["best_epoch"],
            "early_stopped_at_epoch": len(results["history"]["train_loss"]),
            "training_time_minutes": training_time / 60
        },
        "test_metrics": {
            "rmse": results["test_rmse"],
            "mae": results["test_mae"],
            "mse": results["test_rmse"] ** 2,
            "r2": results["test_r2"],
            "correlation": np.corrcoef(results["test_targets"], results["test_predictions"])[0] if len(results["test_predictions"]) > 1 else 0.0,
        },
        "baseline_comparison": {
            "GAT_baseline_rmse": 50.5,
            "PerioGT_baseline_rmse": 30.8,
            "HEPT_iter1_rmse": 49.58,
            "HEPT_iter2_rmse": results["test_rmse"],
            "improvement_vs_iter1": 49.58 - results["test_rmse"],
            "improvement_vs_PerioGT": 30.8 - results["test_rmse"],
            "percent_improvement_vs_PerioGT": (30.8 - results["test_rmse"]) / 30.8 * 100
        }
    }

    with open("results_v2.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print("\nResults saved to results_v2.json")

    # Run evaluation
    print("\nRunning evaluation...")
    try:
        run_full_evaluation("checkpoints/hept_best.pt", "analysis")
    except Exception as e:
        print(f"Evaluation failed: {e}")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Test RMSE: {results['test_rmse']:.2f}")
    print(f"Test MAE: {results['test_mae']:.2f}")
    print(f"Test R²: {results['test_r2']:.4f}")
    print()
    print("Baseline Comparison:")
    print(f"  GAT baseline:      RMSE = 50.5")
    print(f"  PerioGT baseline:  RMSE = 30.8")
    print(f"  HEPT (Iter 2):    RMSE = {results['test_rmse']:.2f}")
    print()

    improvement = results['test_rmse'] - 49.58  # vs Iteration 1
    print(f"Improvement vs Iteration 1 (49.58): {improvement:.2f}")

    improvement_vs_target = 30.8 - results['test_rmse']
    if improvement_vs_target > 0:
        print(f"SUCCESS: Beat PerioGT baseline by {improvement_vs_target:.2f}!")
    else:
        print(f"Gap to PerioGT baseline: {-improvement_vs_target:.2f}")

    return results


if __name__ == "__main__":
    results = run_iteration2()

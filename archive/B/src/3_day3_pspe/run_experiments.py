"""
Run all 4 PSPE experimental conditions.
"""

import sys
import os
import json
import time
import warnings
import numpy as np
import torch

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pspe_data import (
    build_datasets_pspe,
    load_rwpe,
    assign_rwpe_to_graphs_normalized,
    create_dataloaders,
)
from train import run_training, CONFIG, SEED

warnings.filterwarnings("ignore")


def main():
    csv_path = CONFIG['csv_path']
    rwpe_path = CONFIG['rwpe_path']
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    batch_size = CONFIG['batch_size']
    device = CONFIG['device']

    print("=" * 70)
    print("PSPE Experiment: Polymer-Specific Positional Encodings for Tg Prediction")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Model seed: {SEED}")

    # =========================================================================
    # Step 1: Build datasets with PSPE features
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Building datasets with PSPE features")
    print("=" * 70)

    train_graphs, val_graphs, test_graphs, scaler, metadata = build_datasets_pspe(
        csv_path=csv_path,
        seed=71,
    )

    print(f"Train: {metadata['num_train']}, Val: {metadata['num_val']}, Test: {metadata['num_test']}")
    print(f"PSPE-dce stats: mean={metadata['pspe_dce']['mean']:.4f}, std={metadata['pspe_dce']['std']:.4f}")
    print(f"PSPE-mi stats: mean={metadata['pspe_mi']['mean']:.4f}, std={metadata['pspe_mi']['std']:.4f}")
    print(f"Backbone length: mean={metadata['backbone_lengths']['mean']:.1f}, "
          f"min={metadata['backbone_lengths']['min']}, max={metadata['backbone_lengths']['max']}")

    # Save metadata
    with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # =========================================================================
    # Step 2: Load and assign RWPE
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Loading RWPE k=32")
    print("=" * 70)

    rwpe_data = load_rwpe(rwpe_path)

    # Assign RWPE by list position (same as Day 2 approach)
    # train: offset 0, val: offset len(train), test: offset len(train)+len(val)
    n1_train, n0_train, _, _ = assign_rwpe_to_graphs_normalized(train_graphs, rwpe_data, offset=0)
    n1_val, n0_val, _, _ = assign_rwpe_to_graphs_normalized(val_graphs, rwpe_data, offset=len(train_graphs))
    n1_test, n0_test, _, _ = assign_rwpe_to_graphs_normalized(
        test_graphs, rwpe_data, offset=len(train_graphs) + len(val_graphs)
    )
    print(f"RWPE assigned: train={n1_train}/{n0_train}, val={n1_val}/{n0_val}, test={n1_test}/{n0_test} (ok/fallback)")

    # =========================================================================
    # Step 3: Create dataloaders
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Creating dataloaders")
    print("=" * 70)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_graphs, val_graphs, test_graphs, scaler, batch_size=batch_size
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")

    # =========================================================================
    # Step 4: Run experiments
    # =========================================================================
    conditions = ['baseline', 'pspe_dce', 'pspe_mi', 'pspe_combined']
    all_results = {}

    for condition in conditions:
        print("\n" + "=" * 70)
        print(f"Condition: {condition}")
        print("=" * 70)

        results = run_training(
            condition=condition,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            scaler=scaler,
            save_dir=save_dir,
        )
        all_results[condition] = results

    # =========================================================================
    # Step 5: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary of Results")
    print("=" * 70)
    print(f"{'Condition':<20} {'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<10} {'Best Epoch':<12}")
    print("-" * 66)

    baseline_rmse = all_results['baseline']['test_rmse']

    for condition in conditions:
        r = all_results[condition]
        delta = r['test_rmse'] - baseline_rmse
        delta_str = f"{delta:+.2f}K" if delta != 0 else " baseline"
        print(f"{condition:<20} {r['test_rmse']:>8.2f}K    {r['test_mae']:>8.2f}K    "
              f"{r['test_r2']:>8.4f}  {r['best_epoch']:>6d}     {delta_str:>10s}")

    # Save summary
    summary = {
        'conditions': conditions,
        'baseline_rmse': baseline_rmse,
        'results': {c: {
            'test_rmse': r['test_rmse'],
            'test_mae': r['test_mae'],
            'test_r2': r['test_r2'],
            'best_epoch': r['best_epoch'],
            'n_params': r['n_params'],
            'delta_rmse': r['test_rmse'] - baseline_rmse,
        } for c, r in all_results.items()},
        'metadata': metadata,
    }
    with open(os.path.join(save_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {save_dir}/")

    return all_results


if __name__ == '__main__':
    main()

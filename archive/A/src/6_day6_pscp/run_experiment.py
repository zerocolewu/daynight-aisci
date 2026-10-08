#!/usr/bin/env python3
"""
Main experiment runner for PSCP with proper error handling and checkpointing
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

# Import local modules
from train_pscp import (
    PSCPTrainer, FineTuner,
    run_ablation_study, run_scale_analysis, generate_visualizations
)


def check_gpu():
    """Check GPU availability"""
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return torch.cuda.is_available()


def verify_datasets(pi1m_path, tg_path):
    """Verify datasets exist and are readable"""
    print("\nVerifying datasets...")
    assert os.path.exists(pi1m_path), f"PI1M not found: {pi1m_path}"
    assert os.path.exists(tg_path), f"Tg dataset not found: {tg_path}"

    # Check PI1M
    pi1m_df = pd.read_csv(pi1m_path)
    print(f"PI1M: {len(pi1m_df)} samples")
    print(f"  Columns: {pi1m_df.columns.tolist()}")

    # Check Tg
    tg_df = pd.read_csv(tg_path)
    print(f"Tg: {len(tg_df)} samples")
    print(f"  Columns: {tg_df.columns.tolist()}")
    print(f"  Tg range: {tg_df['Tg'].min():.1f} to {tg_df['Tg'].max():.1f}")

    return True


def main():
    """Main execution with error handling"""
    print("=" * 70)
    print("PSCP EXPERIMENT: Polymer-Specific Contrastive Pretraining")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Check GPU
        has_gpu = check_gpu()

        # Paths
        pi1m_path = '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv'
        tg_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
        output_dir = '/home/home_public/aisci/AIscientist/src/6_day6_pscp'

        # Verify datasets
        verify_datasets(pi1m_path, tg_path)

        # Configuration
        config = {
            'num_node_features': 159,  # Correct: 119 atomic_num + 8 degree + 8 formal_charge + 5 chiral + 6 num_Hs + 6 hybridization + 3 aromatic + 3 is_star + 1 mass
            'hidden_dim': 256,
            'num_layers': 3,
            'edge_dim': 13,  # Correct: 5 bond_types + 1 conjugated + 1 in_ring + 6 stereo = 13
            'proj_dim': 128,
            'lr': 1e-4,
            'weight_decay': 0.01,
            'batch_size': 32,  # Reduced for memory
            'pretrain_epochs': 5,  # 5 epochs as per requirements
            'pretrain_samples': 100000,  # Use 100K for ablation
            'num_negatives': 5,
            'task_weights': {'fgm': 1.0, 'cri': 0.5, 'egcl': 1.0},
            'ecfp_bits': 2048,
            'log_dir': output_dir
        }

        print(f"\nConfiguration:")
        print(json.dumps(config, indent=2))

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save config
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # Run ablation study
        print("\n" + "=" * 70)
        print("PHASE 1: ABLATION STUDY")
        print("=" * 70)

        ablation_start = time.time()
        ablation_df = run_ablation_study(config, pi1m_path, tg_path, output_dir)
        ablation_time = time.time() - ablation_start

        print(f"\nAblation study completed in {ablation_time/3600:.2f} hours")

        # Run scale analysis
        print("\n" + "=" * 70)
        print("PHASE 2: SCALE ANALYSIS")
        print("=" * 70)

        scale_start = time.time()
        scale_df = run_scale_analysis(config, pi1m_path, tg_path, output_dir)
        scale_time = time.time() - scale_start

        print(f"\nScale analysis completed in {scale_time/3600:.2f} hours")

        # Generate visualizations
        print("\n" + "=" * 70)
        print("PHASE 3: VISUALIZATION")
        print("=" * 70)

        try:
            generate_visualizations(output_dir, ablation_df, scale_df)
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            traceback.print_exc()

        # Print final report
        print("\n" + "=" * 70)
        print("FINAL REPORT")
        print("=" * 70)

        mean_results = ablation_df[ablation_df['fold'] == 'mean']
        baseline_rmse = 38.78

        print("\n[ABLATION RESULTS]")
        print("-" * 70)
        print(f"{'Condition':<15} | {'RMSE':>8} | {'MAE':>8} | {'R2':>8} | {'Improvement':>12}")
        print("-" * 70)
        for _, row in mean_results.iterrows():
            improvement = baseline_rmse - row['rmse']
            print(f"{row['condition']:<15} | {row['rmse']:8.2f} | {row['mae']:8.2f} | {row['r2']:8.4f} | +{improvement:8.2f}")

        print("\n[SCALE ANALYSIS]")
        print("-" * 50)
        print(f"{'Scale':>10} | {'RMSE':>8} | {'± Std':>8} | {'R2':>8}")
        print("-" * 50)
        for _, row in scale_df.iterrows():
            print(f"{int(row['scale']):10} | {row['mean_rmse']:8.2f} | {row['std_rmse']:8.2f} | {row['mean_r2']:8.4f}")

        # Find best result
        best_rmse = mean_results['rmse'].min()
        best_condition = mean_results.loc[mean_results['rmse'].idxmin(), 'condition']
        best_r2 = mean_results.loc[mean_results['rmse'].idxmin(), 'r2']

        print("\n[KEY FINDINGS]")
        print("-" * 50)
        print(f"Best RMSE:       {best_rmse:.2f} ({best_condition})")
        print(f"Best R2:         {best_r2:.4f}")
        print(f"Baseline RMSE:   {baseline_rmse:.2f}")
        print(f"Improvement:     +{baseline_rmse - best_rmse:.2f} ({(baseline_rmse - best_rmse)/baseline_rmse*100:.1f}%)")
        print(f"Target (< 35.0): {'✓ ACHIEVED' if best_rmse < 35.0 else '✗ FAILED'}")

        # Check each pretraining task
        print("\n[TASK CONTRIBUTIONS]")
        print("-" * 50)
        fgm_rmse = mean_results.loc[mean_results['condition'] == 'fgm_only', 'rmse'].values[0]
        cri_rmse = mean_results.loc[mean_results['condition'] == 'cri_only', 'rmse'].values[0]
        egcl_rmse = mean_results.loc[mean_results['condition'] == 'egcl_only', 'rmse'].values[0]
        full_rmse = mean_results.loc[mean_results['condition'] == 'full_pscp', 'rmse'].values[0]
        no_pretrain_rmse = mean_results.loc[mean_results['condition'] == 'no_pretrain', 'rmse'].values[0]

        print(f"FGM only:        {fgm_rmse:.2f} (vs no pretrain: {no_pretrain_rmse - fgm_rmse:+.2f})")
        print(f"CRI only:        {cri_rmse:.2f} (vs no pretrain: {no_pretrain_rmse - cri_rmse:+.2f})")
        print(f"EGCL only:       {egcl_rmse:.2f} (vs no pretrain: {no_pretrain_rmse - egcl_rmse:+.2f})")
        print(f"Full PSCP:       {full_rmse:.2f} (vs no pretrain: {no_pretrain_rmse - full_rmse:+.2f})")

        # Save summary to file
        summary = {
            'best_rmse': float(best_rmse),
            'best_condition': best_condition,
            'best_r2': float(best_r2),
            'baseline_rmse': baseline_rmse,
            'improvement': float(baseline_rmse - best_rmse),
            'target_achieved': bool(best_rmse < 35.0),
            'ablation_results': mean_results.to_dict('records'),
            'scale_results': scale_df.to_dict('records')
        }

        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nSummary saved to {output_dir}/summary.json")

        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {(ablation_time + scale_time)/3600:.2f} hours")

        return 0

    except Exception as e:
        print(f"\n{'!' * 70}")
        print(f"EXPERIMENT FAILED: {e}")
        print(f"{'!' * 70}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

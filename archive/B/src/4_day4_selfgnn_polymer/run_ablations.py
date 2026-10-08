#!/usr/bin/env python3
"""
Run all SelfGNN-Polymer ablation experiments.

Ablations:
  A: No pretraining baseline (random-init GATv2)
  B: SelfGNN with Split-only augmentation
  C: SelfGNN with Standardize-only augmentation
  D: SelfGNN with Split+Standardize (PRIMARY)
  G: Frozen encoder vs full finetuning (post-primary pretraining)
  H: No BatchNorm pretraining → finetuning (collapse check, 50 epochs)
"""

import os
import sys
import subprocess
import json
import time

WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_day4_selfgnn_polymer"
PYTHON = "/home/home_public/aisci/miniconda3/envs/aisci/bin/python"


def run_cmd(cmd, desc):
    """Run a command and print output."""
    print(f"\n{'='*70}")
    print(f"  Running: {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=WORK_DIR, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Command failed with return code {result.returncode}")
    else:
        print(f"  SUCCESS: {desc}")
    return result.returncode


def main():
    print("=" * 70)
    print("  SelfGNN-Polymer: Full Ablation Experiment Suite")
    print("=" * 70)

    results_summary = {}

    # ============================================================
    # Ablation D (PRIMARY): Split + Standardize pretraining
    # ============================================================
    print("\n" + "="*70)
    print("  ABLATION D: SelfGNN Split+Standardize (PRIMARY)")
    print("="*70)

    primary_pretrain_output = "selfgnn_primary"
    run_cmd([
        PYTHON, "pretrain.py",
        "--use_standardize", "--use_split",
        "--output_name", primary_pretrain_output,
        "--epochs", "200",
        "--batch_size", "64",
        "--patience", "30",
    ], "Ablation D: SelfGNN Pretraining (Split+Std)")

    # Finetune with 3 seeds
    pretrained_path = f"results/{primary_pretrain_output}_encoder.pt"
    run_cmd([
        PYTHON, "finetune.py",
        "--pretrained_path", pretrained_path,
        "--output_name", "abl_d_split_std",
        "--seeds", "71", "42", "123",
        "--epochs", "200",
        "--patience", "50",
    ], "Ablation D: Finetuning (3 seeds)")

    # Load results
    with open(os.path.join(WORK_DIR, "results/abl_d_split_std_results.json")) as f:
        results_d = json.load(f)
    results_summary['SelfGNN Split+Std'] = {
        'test_rmse': results_d['mean_test_rmse'],
        'test_rmse_std': results_d['std_test_rmse'],
        'test_mae': results_d['mean_test_mae'],
    }

    # ============================================================
    # Ablation A: No pretraining baseline
    # ============================================================
    print("\n" + "="*70)
    print("  ABLATION A: No Pretraining Baseline")
    print("="*70)

    run_cmd([
        PYTHON, "finetune.py",
        "--output_name", "abl_a_baseline",
        "--encoder_type", "gatv2",
        "--seeds", "71", "42", "123",
        "--epochs", "200",
        "--patience", "50",
    ], "Ablation A: Baseline (random init, 3 seeds)")

    with open(os.path.join(WORK_DIR, "results/abl_a_baseline_results.json")) as f:
        results_a = json.load(f)
    results_summary['No Pretraining'] = {
        'test_rmse': results_a['mean_test_rmse'],
        'test_rmse_std': results_a['std_test_rmse'],
        'test_mae': results_a['mean_test_mae'],
    }

    # Compute improvement
    delta_d = results_a['mean_test_rmse'] - results_d['mean_test_rmse']
    print(f"\n  *** ΔRMSE (Split+Std vs Baseline): {delta_d:.2f}K ***")

    # ============================================================
    # Ablation B: Split-only pretraining
    # ============================================================
    print("\n" + "="*70)
    print("  ABLATION B: SelfGNN Split-Only")
    print("="*70)

    run_cmd([
        PYTHON, "pretrain.py",
        "--use_split", "--no_standardize",
        "--output_name", "selfgnn_split_only",
        "--epochs", "200",
        "--batch_size", "64",
        "--patience", "30",
    ], "Ablation B: SelfGNN Pretraining (Split only)")

    # Check if pretraining completed
    split_ckpt = os.path.join(WORK_DIR, "results/selfgnn_split_only_encoder.pt")
    if os.path.exists(split_ckpt):
        run_cmd([
            PYTHON, "finetune.py",
            "--pretrained_path", split_ckpt,
            "--output_name", "abl_b_split_only",
            "--seeds", "71", "42", "123",
            "--epochs", "200",
            "--patience", "50",
        ], "Ablation B: Finetuning (3 seeds)")

        with open(os.path.join(WORK_DIR, "results/abl_b_split_only_results.json")) as f:
            results_b = json.load(f)
        results_summary['Split Only'] = {
            'test_rmse': results_b['mean_test_rmse'],
            'test_rmse_std': results_b['std_test_rmse'],
            'test_mae': results_b['mean_test_mae'],
        }

    # ============================================================
    # Ablation C: Standardize-only pretraining
    # ============================================================
    print("\n" + "="*70)
    print("  ABLATION C: SelfGNN Standardize-Only")
    print("="*70)

    run_cmd([
        PYTHON, "pretrain.py",
        "--no_split", "--use_standardize",
        "--output_name", "selfgnn_std_only",
        "--epochs", "200",
        "--batch_size", "64",
        "--patience", "30",
    ], "Ablation C: SelfGNN Pretraining (Std only)")

    std_ckpt = os.path.join(WORK_DIR, "results/selfgnn_std_only_encoder.pt")
    if os.path.exists(std_ckpt):
        run_cmd([
            PYTHON, "finetune.py",
            "--pretrained_path", std_ckpt,
            "--output_name", "abl_c_std_only",
            "--seeds", "71", "42", "123",
            "--epochs", "200",
            "--patience", "50",
        ], "Ablation C: Finetuning (3 seeds)")

        with open(os.path.join(WORK_DIR, "results/abl_c_std_only_results.json")) as f:
            results_c = json.load(f)
        results_summary['Std Only'] = {
            'test_rmse': results_c['mean_test_rmse'],
            'test_rmse_std': results_c['std_test_rmse'],
            'test_mae': results_c['mean_test_mae'],
        }

    # ============================================================
    # Ablation H: No BatchNorm pretraining (collapse check, 50 epochs)
    # ============================================================
    print("\n" + "="*70)
    print("  ABLATION H: No BatchNorm Pretraining (Collapse Check)")
    print("="*70)

    run_cmd([
        PYTHON, "pretrain.py",
        "--use_standardize", "--use_split", "--no_batchnorm",
        "--output_name", "selfgnn_no_bn",
        "--epochs", "50",
        "--batch_size", "64",
        "--patience", "50",  # Don't stop early, check for collapse
    ], "Ablation H: SelfGNN Pretraining (No BN, 50 epochs)")

    no_bn_ckpt = os.path.join(WORK_DIR, "results/selfgnn_no_bn_encoder.pt")
    if os.path.exists(no_bn_ckpt):
        run_cmd([
            PYTHON, "finetune.py",
            "--pretrained_path", no_bn_ckpt,
            "--output_name", "abl_h_no_bn",
            "--seeds", "71", "42", "123",
            "--epochs", "200",
            "--patience", "50",
        ], "Ablation H: Finetuning (3 seeds)")

        with open(os.path.join(WORK_DIR, "results/abl_h_no_bn_results.json")) as f:
            results_h = json.load(f)
        results_summary['No BatchNorm'] = {
            'test_rmse': results_h['mean_test_rmse'],
            'test_rmse_std': results_h['std_test_rmse'],
            'test_mae': results_h['mean_test_mae'],
        }

    # ============================================================
    # Ablation G: Frozen encoder finetuning
    # ============================================================
    print("\n" + "="*70)
    print("  ABLATION G: Frozen Encoder vs Full Finetuning")
    print("="*70)

    if os.path.exists(pretrained_path):
        run_cmd([
            PYTHON, "finetune.py",
            "--pretrained_path", pretrained_path,
            "--freeze_encoder",
            "--output_name", "abl_g_frozen",
            "--seeds", "71", "42", "123",
            "--epochs", "200",
            "--patience", "50",
        ], "Ablation G: Frozen encoder finetuning (3 seeds)")

        with open(os.path.join(WORK_DIR, "results/abl_g_frozen_results.json")) as f:
            results_g = json.load(f)
        results_summary['Frozen Encoder'] = {
            'test_rmse': results_g['mean_test_rmse'],
            'test_rmse_std': results_g['std_test_rmse'],
            'test_mae': results_g['mean_test_mae'],
        }

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("  FINAL RESULTS SUMMARY")
    print("="*70)

    for name, res in results_summary.items():
        print(f"  {name:25s}: RMSE={res['test_rmse']:.4f}K ± {res['test_rmse_std']:.4f}K")

    # Compute improvements
    if 'No Pretraining' in results_summary:
        baseline = results_summary['No Pretraining']['test_rmse']
        print(f"\n  Improvements over baseline ({baseline:.2f}K):")
        for name, res in results_summary.items():
            if name != 'No Pretraining':
                delta = baseline - res['test_rmse']
                print(f"    {name:25s}: ΔRMSE = {delta:+.2f}K")

    # Save summary
    summary_path = os.path.join(WORK_DIR, "results/ablations_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()

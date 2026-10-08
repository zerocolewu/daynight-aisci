"""
Master script for PolymerBipartite v2 Iteration 2 experiment.
Runs all training, diagnostics, and produces results summary.

Changes from iter1:
- Wide baseline (hidden_dim matching) instead of extra depth
- Updated model/training API with separate LR, gate penalty, etc.
"""

import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import PolymerBipartiteDataset, get_data_splits
from train import (
    train_bipartite_v2, train_baseline, train_random_ru,
    prepare_batch, count_parameters
)
from model import create_wide_baseline
from diagnostics import (
    diagnostic_ru_discriminability, diagnostic_random_ru_summary,
    diagnostic_overfitting_gap, diagnostic_multi_seed,
    analyze_gate_values, write_results_summary
)


def main():
    start_time = time.time()

    # Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(current_dir), 'alldataset', 'Tg.csv')
    save_dir = current_dir
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CSV path: {csv_path}")

    # Load dataset
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    dataset = PolymerBipartiteDataset(csv_path, cache_dir=save_dir)

    # Data splits
    train_idx, val_idx, test_idx = get_data_splits(len(dataset))
    print(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    all_results = {}

    # =========================================================================
    # Step 1: Train PolymerBipartite v2 iter2 (seed=71)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Training PolymerBipartite v2 Iteration 2 (seed=71)")
    print("=" * 80)
    bp_model, bp_results = train_bipartite_v2(
        dataset, train_idx, val_idx, test_idx,
        seed=71, epochs=300, device=device, save_dir=save_dir
    )
    all_results['PolymerBipartiteV2'] = bp_results

    # =========================================================================
    # Step 2: Train GIN+Trimer baseline (seed=71, hidden_dim=256)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Training GIN+Trimer Baseline (seed=71)")
    print("=" * 80)
    bl_model, bl_results = train_baseline(
        dataset, train_idx, val_idx, test_idx,
        seed=71, epochs=300, device=device, save_dir=save_dir,
        save_name='baseline_h256_seed71.pt'
    )
    all_results['GIN+Trimer'] = bl_results

    # =========================================================================
    # Step 3: Diagnostic 1 — RU Discriminability
    # =========================================================================
    disc1 = diagnostic_ru_discriminability(bp_model, dataset, test_idx, device)
    all_results['ru_discriminability'] = disc1

    # =========================================================================
    # Step 4: Diagnostic 2 — Random RU Assignment
    # =========================================================================
    _, random_results = train_random_ru(
        dataset, train_idx, val_idx, test_idx,
        seed=71, epochs=300, device=device, save_dir=save_dir
    )
    all_results['RandomRU'] = random_results
    disc2 = diagnostic_random_ru_summary(bp_results, random_results)
    all_results['random_ru'] = disc2

    # =========================================================================
    # Step 5: Diagnostic 3 — Parameter-Matched Baseline (WIDE)
    # =========================================================================
    bp_params = bp_results['n_params']
    print(f"\nPolymerBipartite v2 params: {bp_params:,}")
    print(f"Standard baseline params: {bl_results['n_params']:,}")

    if bp_params > bl_results['n_params']:
        print("Training WIDE parameter-matched baseline...")
        wide_model = create_wide_baseline(bp_params)
        wide_dim = wide_model.hidden_dim
        print(f"Wide baseline hidden_dim: {wide_dim}, params: {count_parameters(wide_model):,}")

        _, matched_results = train_baseline(
            dataset, train_idx, val_idx, test_idx,
            seed=71, epochs=300, device=device, save_dir=save_dir,
            hidden_dim=wide_dim,
            save_name=f'baseline_h{wide_dim}_seed71.pt'
        )
        all_results['GIN+Trimer_matched'] = matched_results
        all_results['param_matched'] = {
            'bipartite_params': bp_params,
            'baseline_params': bl_results['n_params'],
            'matched_params': matched_results['n_params'],
            'baseline_test_rmse': bl_results['test_rmse'],
            'matched_test_rmse': matched_results['test_rmse'],
            'bipartite_test_rmse': bp_results['test_rmse'],
        }
        print(f"Wide parameter-matched baseline test RMSE: {matched_results['test_rmse']:.2f}")
        print(f"PolymerBipartite v2 iter2 test RMSE: {bp_results['test_rmse']:.2f}")
        if bp_results['test_rmse'] < matched_results['test_rmse']:
            print("PASS: Bipartite v2 iter2 outperforms wide parameter-matched baseline")
        else:
            print("FAIL: Wide parameter-matched baseline matches or beats Bipartite v2")

    # =========================================================================
    # Step 6: Diagnostic 4 — Overfitting Gap
    # =========================================================================
    gap_models = {k: v for k, v in all_results.items()
                  if k in ['PolymerBipartiteV2', 'GIN+Trimer', 'RandomRU', 'GIN+Trimer_matched']}
    overfitting_results = diagnostic_overfitting_gap(gap_models)
    all_results['overfitting'] = overfitting_results

    # =========================================================================
    # Step 7: Gate Value Analysis
    # =========================================================================
    gate_results = analyze_gate_values(bp_model, dataset, test_idx, device)
    all_results['gate_values'] = gate_results

    # =========================================================================
    # Step 8: Diagnostic 5 — Multi-Seed Verification
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: Multi-Seed Verification (seeds: 42, 71, 123)")
    print("=" * 80)

    multi_seed = {
        'PolymerBipartiteV2': {},
        'GIN+Trimer': {},
    }

    # Seed 71 already done
    multi_seed['PolymerBipartiteV2'][71] = bp_results
    multi_seed['GIN+Trimer'][71] = bl_results

    for seed in [42, 123]:
        print(f"\n--- Seed {seed} ---")
        _, bp_r = train_bipartite_v2(
            dataset, train_idx, val_idx, test_idx,
            seed=seed, epochs=300, device=device, save_dir=save_dir
        )
        multi_seed['PolymerBipartiteV2'][seed] = bp_r

        _, bl_r = train_baseline(
            dataset, train_idx, val_idx, test_idx,
            seed=seed, epochs=300, device=device, save_dir=save_dir,
            save_name=f'baseline_h256_seed{seed}.pt'
        )
        multi_seed['GIN+Trimer'][seed] = bl_r

    diagnostic_multi_seed(multi_seed)
    all_results['multi_seed'] = {
        model: {s: {'test_rmse': r['test_rmse']} for s, r in seeds.items()}
        for model, seeds in multi_seed.items()
    }

    # =========================================================================
    # Final Summary
    # =========================================================================
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    summary_path = os.path.join(save_dir, 'results_summary_iter2.txt')
    write_results_summary(summary_path, all_results)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    main()

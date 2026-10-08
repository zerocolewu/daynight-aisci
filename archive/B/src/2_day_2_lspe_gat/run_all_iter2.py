"""
Orchestration script for LSPE-GAT Iteration 2 experiments.

Treatments:
  D1: static_rwpe_k24 - Static RWPE with k=24 concatenated to input
  D2: static_rwpe_k32 - Static RWPE with k=32 concatenated to input
  E1: skip_rwpe_k16  - SkipRWPE (layer-wise RWPE injection) with k=16
  E2: skip_rwpe_k32  - SkipRWPE with k=32
  F:  skip_rwpe_k32  - Combined: k=32 with layer-wise injection (same as E2)

Pipeline:
  1. Precompute extended RWPE (k=24, k=32) if not done
  2. Train all new treatments with 3 seeds each
  3. Collect and report results with cross-comparison
"""
import os
import sys
import json
import subprocess
import time
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# New treatments for this iteration
NEW_TREATMENTS = [
    'static_rwpe_k24',
    'static_rwpe_k32',
    'skip_rwpe_k16',
    'skip_rwpe_k32',
]

SEEDS = [42, 123, 456]

# Baseline results from previous run (for comparison)
PREVIOUS_RESULTS = {
    'baseline': {'test_rmse': 38.51, 'test_mae': 24.96, 'test_r2': 0.8777},
    'static_rwpe_k16': {'test_rmse': 36.29, 'test_mae': 24.01, 'test_r2': 0.8914},
    'lspe_gat': {'test_rmse': 37.96, 'test_mae': 24.54, 'test_r2': 0.8812},
    'lspe_gat_edge': {'test_rmse': 37.85, 'test_mae': 24.34, 'test_r2': 0.8819},
}


def run_command(cmd, description):
    """Run a command and stream output."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=OUTPUT_DIR)

    elapsed = time.time() - t0
    print(result.stdout)
    if result.stderr:
        stderr_str = result.stderr[:3000]
        print("STDERR:", stderr_str)

    print(f"{'='*70}")
    print(f"{description} completed in {elapsed:.1f}s (exit code: {result.returncode})")
    print(f"{'='*70}")

    return result.returncode == 0


def main():
    print("=" * 70)
    print("LSPE-GAT Iteration 2: Higher-k RWPE + Layer-wise Injection")
    print("=" * 70)
    print(f"New treatments: {NEW_TREATMENTS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(NEW_TREATMENTS) * len(SEEDS)}")
    print(f"Output dir: {OUTPUT_DIR}")

    overall_start = time.time()

    # Step 1: Precompute extended RWPE
    rwpe_files_ok = all(
        os.path.exists(os.path.join(OUTPUT_DIR, f'rwpe_k{k}.pkl'))
        for k in [16, 24, 32]
    )

    if not rwpe_files_ok:
        print("\n[Step 1/3] Precomputing extended RWPE (k=16,24,32)...")
        ok = run_command(
            [sys.executable, os.path.join(OUTPUT_DIR, 'extend_rwpe_precompute.py')],
            "Extended RWPE Precomputation"
        )
        if not ok:
            print("ERROR: Extended RWPE precomputation failed!")
            return
    else:
        print(f"\n[Step 1/3] All RWPE files already exist:")
        for k in [16, 24, 32]:
            p = os.path.join(OUTPUT_DIR, f'rwpe_k{k}.pkl')
            if os.path.exists(p):
                print(f"  rwpe_k{k}.pkl: {os.path.getsize(p)/(1024*1024):.1f} MB")

    # Step 2: Train new treatments
    print("\n[Step 2/3] Training all new treatments...")
    all_results = {}

    for treatment in NEW_TREATMENTS:
        treatment_results = {}
        for seed in SEEDS:
            name = f"{treatment}_seed{seed}"
            result_file = os.path.join(OUTPUT_DIR, f'results_{treatment}_seed{seed}.json')

            if os.path.exists(result_file):
                print(f"\n  Results already exist for {name}, skipping...")
                with open(result_file) as f:
                    treatment_results[str(seed)] = json.load(f)
                continue

            ok = run_command(
                [sys.executable, os.path.join(OUTPUT_DIR, 'train.py'),
                 '--treatment', treatment, '--seed', str(seed),
                 '--output_dir', OUTPUT_DIR],
                f"Training {name}"
            )

            if ok and os.path.exists(result_file):
                with open(result_file) as f:
                    treatment_results[str(seed)] = json.load(f)
            else:
                print(f"  WARNING: Training failed or results not found for {name}")
                treatment_results[str(seed)] = {'error': 'Training failed'}

        all_results[treatment] = treatment_results

    # Step 3: Aggregate and report
    print(f"\n{'='*70}")
    print(f"[Step 3/3] Aggregating Results")
    print(f"{'='*70}")

    summary = {}
    for treatment in NEW_TREATMENTS:
        tr = all_results.get(treatment, {})
        metrics = {'test_rmse': [], 'test_mae': [], 'test_r2': [], 'best_epoch': [], 'n_params': []}
        for seed_str, result in tr.items():
            for key in metrics:
                val = result.get(key)
                if val is not None:
                    metrics[key].append(float(val))

        if metrics['test_rmse']:
            summary[treatment] = {
                'test_rmse': {'mean': np.mean(metrics['test_rmse']), 'std': np.std(metrics['test_rmse'])},
                'test_mae': {'mean': np.mean(metrics['test_mae']), 'std': np.std(metrics['test_mae'])},
                'test_r2': {'mean': np.mean(metrics['test_r2']), 'std': np.std(metrics['test_r2'])},
                'best_epoch_mean': np.mean(metrics['best_epoch']),
                'n_params': metrics['n_params'][0] if metrics['n_params'] else 'N/A',
                'seeds_completed': len(metrics['test_rmse']),
            }
        else:
            summary[treatment] = {'error': 'No results', 'seeds_completed': 0}

    # --- Full results table ---
    print(f"\n{'='*70}")
    print("FULL RESULTS (Including Previous Run Baselines)")
    print(f"{'='*70}")
    print(f"\n{'Treatment':<22} {'Test RMSE':<18} {'Test MAE':<18} {'Test R²':<18} {'ΔRMSE':<12} {'Seeds':<8}")
    print("-" * 96)

    # Previous results first
    for treat, vals in PREVIOUS_RESULTS.items():
        rmse = f"{vals['test_rmse']:.2f}"
        mae = f"{vals['test_mae']:.2f}"
        r2 = f"{vals['test_r2']:.4f}"
        delta = "—"
        seeds_str = "—"
        print(f"  {treat:<20} {rmse:<18} {mae:<18} {r2:<18} {delta:<12} {seeds_str:<8}")

    # New results
    bl_rmse = PREVIOUS_RESULTS['baseline']['test_rmse']
    for treatment in NEW_TREATMENTS:
        s = summary.get(treatment, {})
        if 'error' in s:
            print(f"  {treatment:<20} {'ERROR: ' + s['error']:<18}")
        else:
            rmse_str = f"{s['test_rmse']['mean']:.2f}±{s['test_rmse']['std']:.2f}"
            mae_str = f"{s['test_mae']['mean']:.2f}±{s['test_mae']['std']:.2f}"
            r2_str = f"{s['test_r2']['mean']:.4f}±{s['test_r2']['std']:.4f}"
            delta = s['test_rmse']['mean'] - bl_rmse
            sign = "+" if delta > 0 else ""
            delta_str = f"{sign}{delta:.2f}"
            seeds_str = f"{s['seeds_completed']}/{len(SEEDS)}"
            print(f"  {treatment:<20} {rmse_str:<18} {mae_str:<18} {r2_str:<18} {delta_str:<12} {seeds_str:<8}")

    # Parameter counts
    print(f"\nModel Parameters:")
    for treatment in NEW_TREATMENTS:
        s = summary.get(treatment, {})
        n = s.get('n_params', 'N/A')
        print(f"  {treatment:<22}: {n}")

    # Comparative analysis
    print(f"\n{'='*70}")
    print("Comparative Analysis (vs baseline {:.2f}K RMSE)".format(bl_rmse))
    print(f"{'='*70}")

    best_rmse = bl_rmse
    best_treatment = 'baseline'
    for treatment in NEW_TREATMENTS:
        s = summary.get(treatment, {})
        t_rmse = s.get('test_rmse', {}).get('mean', None)
        t_r2 = s.get('test_r2', {}).get('mean', None)
        if t_rmse is not None:
            delta_rmse = t_rmse - bl_rmse
            delta_rmse_pct = (delta_rmse / bl_rmse) * 100
            sign = "+" if delta_rmse > 0 else ""
            print(f"  {treatment:<22}: ΔRMSE={sign}{delta_rmse:.2f}K ({sign}{delta_rmse_pct:.1f}%), "
                  f"R²={t_r2:.4f}" if t_r2 else f"  {treatment:<22}: ΔRMSE={sign}{delta_rmse:.2f}K")
            if t_rmse < best_rmse:
                best_rmse = t_rmse
                best_treatment = treatment

    # Also compare against previous best (static_rwpe_k16: 36.29)
    prev_best_rmse = PREVIOUS_RESULTS['static_rwpe_k16']['test_rmse']
    print(f"\n  Comparison vs previous best (static_rwpe_k16: {prev_best_rmse:.2f}K):")
    for treatment in NEW_TREATMENTS:
        s = summary.get(treatment, {})
        t_rmse = s.get('test_rmse', {}).get('mean', None)
        if t_rmse is not None:
            delta = t_rmse - prev_best_rmse
            sign = "+" if delta > 0 else ""
            print(f"    {treatment:<22}: Δ={sign}{delta:.2f}K")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, 'iter2_results_summary.json')
    # Merge with previous results for complete picture
    full_summary = {'previous': PREVIOUS_RESULTS, 'new': summary}
    with open(summary_path, 'w') as f:
        json.dump(full_summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print(f"\n{'='*70}")
    print(f"BEST: {best_treatment} with RMSE={best_rmse:.2f}K")
    print(f"{'='*70}")

    overall_elapsed = time.time() - overall_start
    print(f"\nTotal time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print("DONE.")


if __name__ == '__main__':
    main()

"""
Run all LSPE-GAT experiments for polymer Tg prediction.

Pipeline:
  1. Precompute RWPE (if not already done)
  2. Train all treatments with 3 seeds each
  3. Collect and report results
"""

import os
import sys
import json
import subprocess
import time
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RWPE_PATH = os.path.join(OUTPUT_DIR, 'rwpe_data.pkl')

TREATMENTS = ['baseline', 'static_rwpe', 'lspe_gat', 'lspe_gat_edge']
SEEDS = [42, 123, 456]


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
        print("STDERR:", result.stderr[:2000])

    print(f"{'='*70}")
    print(f"{description} completed in {elapsed:.1f}s (exit code: {result.returncode})")
    print(f"{'='*70}")

    return result.returncode == 0


def main():
    print("=" * 70)
    print("LSPE-GAT Experiment Suite")
    print("=" * 70)
    print(f"Treatments: {TREATMENTS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(TREATMENTS) * len(SEEDS)}")
    print(f"Output dir: {OUTPUT_DIR}")

    overall_start = time.time()

    # Step 1: Precompute RWPE
    if not os.path.exists(RWPE_PATH):
        print("\n[Step 1/3] Precomputing RWPE...")
        ok = run_command(
            [sys.executable, os.path.join(OUTPUT_DIR, 'rwpe_precompute.py')],
            "RWPE Precomputation"
        )
        if not ok:
            print("ERROR: RWPE precomputation failed!")
            return
    else:
        print(f"\n[Step 1/3] RWPE data already exists at {RWPE_PATH}")
        print(f"  File size: {os.path.getsize(RWPE_PATH) / (1024*1024):.1f} MB")

    # Step 2: Train all treatments
    print("\n[Step 2/3] Training all treatments...")
    all_results = {}

    for treatment in TREATMENTS:
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
    for treatment in TREATMENTS:
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

    # Print results table
    print(f"\n{'Treatment':<20} {'Test RMSE':<18} {'Test MAE':<18} {'Test R²':<18} {'Seeds':<8}")
    print("-" * 82)

    for treatment in TREATMENTS:
        s = summary.get(treatment, {})
        if 'error' in s:
            print(f"{treatment:<20} {'ERROR: ' + s['error']:<18}")
        else:
            rmse = f"{s['test_rmse']['mean']:.4f}±{s['test_rmse']['std']:.4f}"
            mae = f"{s['test_mae']['mean']:.4f}±{s['test_mae']['std']:.4f}"
            r2 = f"{s['test_r2']['mean']:.4f}±{s['test_r2']['std']:.4f}"
            seeds = f"{s['seeds_completed']}/{len(SEEDS)}"
            print(f"{treatment:<20} {rmse:<18} {mae:<18} {r2:<18} {seeds:<8}")

    # Parameter counts
    print(f"\nModel Parameters:")
    for treatment in TREATMENTS:
        s = summary.get(treatment, {})
        n = s.get('n_params', 'N/A')
        print(f"  {treatment:<20}: {n}")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, 'all_results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Comparative analysis
    print(f"\n{'='*70}")
    print("Comparative Analysis")
    print(f"{'='*70}")

    baseline = summary.get('baseline', {})
    bl_rmse = baseline.get('test_rmse', {}).get('mean', None)
    bl_r2 = baseline.get('test_r2', {}).get('mean', None)

    if bl_rmse is not None:
        print(f"\n  Baseline (GAT): RMSE={bl_rmse:.4f}, R²={bl_r2:.4f}")
        for treatment in ['static_rwpe', 'lspe_gat', 'lspe_gat_edge']:
            ts = summary.get(treatment, {})
            t_rmse = ts.get('test_rmse', {}).get('mean', None)
            t_r2 = ts.get('test_r2', {}).get('mean', None)
            if t_rmse is not None:
                delta_rmse = t_rmse - bl_rmse
                delta_r2 = t_r2 - bl_r2 if t_r2 is not None else float('nan')
                sign = "+" if delta_rmse > 0 else ""
                print(f"    {treatment}: ΔRMSE={sign}{delta_rmse:.4f}, ΔR²={delta_r2:+.4f}")
    else:
        print("  Baseline results not available for comparison")

    overall_elapsed = time.time() - overall_start
    print(f"\nTotal time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print("DONE.")


if __name__ == '__main__':
    main()

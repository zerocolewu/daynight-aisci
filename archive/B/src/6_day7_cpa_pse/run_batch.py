#!/usr/bin/env python3
"""
Script to run CPA-PSE experiments with proper logging.
Runs conditions A, B sequentially (5 seeds each).
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import CPAGAT, NonCPAGAT
from train import train_model

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "run_log.txt")

SEEDS = [71, 42, 123, 888, 2024]

ALL_CONDITIONS = [
    ('A', CPAGAT, 'cpa_baseline', 'CPA-GAT baseline (42-dim)'),
    ('B', CPAGAT, 'cpa_pse', 'CPA-PSE full (45-dim)'),
    ('C', CPAGAT, 'cpa_modpos', 'CPA + modular_position only (43-dim)'),
    ('D', CPAGAT, 'cpa_perdist', 'CPA + periodic_distance only (43-dim)'),
    ('E', CPAGAT, 'cpa_backbone', 'CPA + is_backbone only (43-dim)'),
    ('F', CPAGAT, 'cpa_pse_nopspe', 'CPA + PSE without PSPE (43-dim)'),
    ('H', NonCPAGAT, 'noncpa_pse', 'Non-CPA GAT + full PSE (45-dim)'),
]


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')


def run_batch(condition_ids):
    log(f"\n{'='*60}")
    log(f"CPA-PSE EXPERIMENT RUNNER")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Conditions: {condition_ids}")
    log(f"Seeds: {SEEDS}")
    log(f"{'='*60}")

    total_start = time.time()

    for cond_id in condition_ids:
        for (c_id, model_cls, model_name, desc) in ALL_CONDITIONS:
            if c_id == cond_id:
                break
        else:
            continue

        log(f"\n## CONDITION {c_id}: {desc}")
        cond_start = time.time()
        cond_rmse = []

        for seed in SEEDS:
            seed_start = time.time()
            log(f"  Seed {seed} starting...")

            try:
                result = train_model(model_cls, model_name, seed, c_id)
                rmse = result['test_rmse']
                cond_rmse.append(rmse)
                elapsed = time.time() - seed_start
                log(f"  Seed {seed}: Test RMSE = {rmse:.4f} K, Time = {elapsed:.0f}s")
            except Exception as e:
                log(f"  Seed {seed}: FAILED - {e}")
                import traceback
                log(traceback.format_exc())

        if cond_rmse:
            import numpy as np
            mean_rmse = np.mean(cond_rmse)
            std_rmse = np.std(cond_rmse, ddof=1)
            log(f"  CONDITION {c_id} SUMMARY: {mean_rmse:.4f} ± {std_rmse:.4f} K "
                f"(per-seed: {[f'{r:.2f}' for r in cond_rmse]})")

        cond_elapsed = time.time() - cond_start
        log(f"  Condition time: {cond_elapsed:.0f}s")

    total_elapsed = time.time() - total_start
    log(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    log(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditions', nargs='+', default=['A', 'B'])
    args = parser.parse_args()
    run_batch(args.conditions)

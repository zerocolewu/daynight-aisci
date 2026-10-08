"""
Master script that runs all phases of PerioGT-Lite.
"""

import os
import sys
import json
import subprocess
import time

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'#'*60}")
    print(f"# {description}")
    print(f"{'#'*60}")
    start = time.time()
    result = subprocess.run(cmd, shell=True, cwd=OUTPUT_DIR,
                          capture_output=False, text=True)
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s (exit code: {result.returncode})")
    if result.returncode != 0:
        print(f"ERROR: {description} failed with exit code {result.returncode}")
        return False
    return True


def main():
    python = sys.executable

    # Phase 0a: Preprocess PI1M
    if not run_command(f"{python} preprocess_pi1m.py", "Phase 0a: Preprocess PI1M"):
        return

    # Phase 0b: Preprocess Tg
    if not run_command(f"{python} preprocess_tg.py", "Phase 0b: Preprocess Tg"):
        return

    # Phase 1: Contrastive pre-training
    if not run_command(f"{python} pretrain.py", "Phase 1: Contrastive Pre-training"):
        return

    # Phase 2a: Fine-tune baseline (no pre-training)
    if not run_command(f"{python} finetune.py --mode baseline",
                      "Phase 2a: Fine-tune Baseline"):
        return

    # Phase 2b: Fine-tune with pre-trained encoder
    if not run_command(f"{python} finetune.py --mode pretrained",
                      "Phase 2b: Fine-tune Pre-trained"):
        return

    # Phase 3: Diagnostics
    if not run_command(f"{python} diagnostics.py", "Phase 3: Diagnostics"):
        return

    # Print summary
    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print("=" * 60)

    # Load and display results
    for name, path in [('Baseline', 'results_baseline.json'),
                       ('Pre-trained', 'results_pretrained.json')]:
        fpath = os.path.join(OUTPUT_DIR, path)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            r = data.get('results', {})
            print(f"\n{name}:")
            print(f"  Train RMSE: {r.get('train_rmse', 'N/A'):.4f}")
            print(f"  Val RMSE:   {r.get('val_rmse', 'N/A'):.4f}")
            print(f"  Test RMSE:  {r.get('test_rmse', 'N/A'):.4f}")
            print(f"  Best epoch: {r.get('best_epoch', 'N/A')}")
            print(f"  Overfit gap: {r.get('overfitting_gap', 'N/A'):.4f}")


if __name__ == '__main__':
    main()

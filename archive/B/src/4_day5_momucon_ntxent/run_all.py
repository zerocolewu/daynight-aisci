#!/usr/bin/env python3
"""
MoMuCon-NT-Xent: Full Pipeline Orchestrator

Steps:
  1. Prepare data (PI1M pretraining pairs + Tg finetuning graphs)
  2. Pretrain Siamese GAT with NT-Xent on PI1M
  3. Finetune on Tg prediction (random baseline, linear probe, full finetuning)
  4. Run diagnostics (loss curves, collapse monitoring, t-SNE, comparison plots)
  5. Generate summary report
"""

import os
import sys
import json
import time
import subprocess

WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_day5_momucon_ntxent"
os.chdir(WORK_DIR)

RESULTS_DIR = os.path.join(WORK_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

STAGE_TIMES = {}


def run_step(name, script_name):
    """Run a Python script and record timing."""
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=WORK_DIR,
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0
    STAGE_TIMES[name] = elapsed

    if result.returncode != 0:
        print(f"\nERROR: {name} failed with return code {result.returncode}")
        return False

    print(f"\n{name} completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    return True


def generate_report():
    """Generate a comprehensive summary report."""
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("MoMuCon-NT-Xent: Monomer-vs-Multimer Contrastive Pretraining")
    report_lines.append("Experiment Report")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Timing
    report_lines.append("## Execution Timeline")
    report_lines.append("")
    total_time = 0
    for name, elapsed in STAGE_TIMES.items():
        report_lines.append(f"  {name}: {elapsed:.1f}s ({elapsed/60:.1f}m)")
        total_time += elapsed
    report_lines.append(f"  TOTAL: {total_time:.1f}s ({total_time/60:.1f}m)")
    report_lines.append("")

    # Pretraining results
    config_path = os.path.join(RESULTS_DIR, "pretrain_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pt_config = json.load(f)
        report_lines.append("## Pretraining")
        report_lines.append(f"  PI1M polymers: {pt_config.get('n_pairs', 'N/A')}")
        report_lines.append(f"  Training pairs: {pt_config.get('n_train_pairs', 'N/A')}")
        report_lines.append(f"  Epochs: {pt_config.get('epochs', 'N/A')}")
        report_lines.append(f"  Batch size: {pt_config.get('batch_size', 'N/A')}")
        report_lines.append(f"  LR: {pt_config.get('lr', 'N/A')}")
        report_lines.append(f"  Temperature: {pt_config.get('tau', 'N/A')}")
        report_lines.append(f"  Best epoch: {pt_config.get('best_epoch', 'N/A')}")
        report_lines.append(f"  Best loss: {pt_config.get('best_loss', 'N/A'):.6f}" if isinstance(pt_config.get('best_loss'), float) else f"  Best loss: {pt_config.get('best_loss', 'N/A')}")
        report_lines.append(f"  Final loss: {pt_config.get('final_loss', 'N/A'):.6f}" if isinstance(pt_config.get('final_loss'), float) else f"  Final loss: {pt_config.get('final_loss', 'N/A')}")
        report_lines.append(f"  Trainable params: {pt_config.get('n_trainable_params', 'N/A'):,}" if isinstance(pt_config.get('n_trainable_params'), int) else f"  Trainable params: {pt_config.get('n_trainable_params', 'N/A')}")
        report_lines.append("")

    # Finetuning results
    finetune_path = os.path.join(RESULTS_DIR, "finetune_results.json")
    if os.path.exists(finetune_path):
        with open(finetune_path, 'r') as f:
            ft_results = json.load(f)
        report_lines.append("## Finetuning Results")
        report_lines.append("")
        report_lines.append(f"  {'Mode':<25} {'Test RMSE':>10} {'Test MAE':>10} {'Test R²':>10}")
        report_lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        for mode, r in sorted(ft_results.items()):
            report_lines.append(f"  {mode:<25} {r['test_rmse']:>10.4f} {r['test_mae']:>10.4f} {r['test_r2']:>10.4f}")
        report_lines.append("")

    # Comparison with historical baselines
    report_lines.append("## Comparison with Historical Baselines")
    report_lines.append("")
    historical = [
        ("RWPE-only (Day 2)", 35.94),
        ("PSPE-combined (Day 3)", 35.44),
        ("SelfGNN-Polymer (Day 4)", 34.89),
    ]
    for name, val in historical:
        report_lines.append(f"  {name}: {val:.2f}K RMSE")

    if os.path.exists(finetune_path):
        for mode, r in ft_results.items():
            report_lines.append(f"  MoMuCon-NT-Xent ({mode}): {r['test_rmse']:.2f}K RMSE")
    report_lines.append("")

    # Diagnostics
    diag_path = os.path.join(RESULTS_DIR, "pretrain_diagnostics.json")
    if os.path.exists(diag_path):
        with open(diag_path, 'r') as f:
            diag = json.load(f)
        report_lines.append("## Collapse Diagnostics")
        if diag:
            first = diag[0]
            last = diag[-1]
            report_lines.append(f"  Epoch {first['epoch']}: MeanCos={first.get('mean_pairwise', 0):.4f}, "
                               f"PosSim={first.get('pos_sim', 0):.4f}, NegSim={first.get('neg_sim', 0):.4f}")
            report_lines.append(f"  Epoch {last['epoch']}: MeanCos={last.get('mean_pairwise', 0):.4f}, "
                               f"PosSim={last.get('pos_sim', 0):.4f}, NegSim={last.get('neg_sim', 0):.4f}")
            report_lines.append(f"  Final Pos-Neg Gap: {last.get('pos_neg_gap', 0):.4f}")
        report_lines.append("")

    # Figures
    report_lines.append("## Generated Figures")
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    if os.path.exists(figures_dir):
        for f in sorted(os.listdir(figures_dir)):
            report_lines.append(f"  - {f}")
    report_lines.append("")

    # Write report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(RESULTS_DIR, "report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")


def main():
    print("=" * 70)
    print("MoMuCon-NT-Xent: Full Pipeline")
    print("Working directory:", WORK_DIR)
    print("=" * 70)

    steps = [
        ("Data Preparation (PI1M)", "data_utils_prepare_pi1m.py"),
        ("Data Preparation (Tg)", "data_utils_prepare_tg.py"),
        ("Pretraining", "pretrain.py"),
        ("Finetuning", "finetune.py"),
        ("Diagnostics", "diagnostics.py"),
    ]

    for name, script in steps:
        # Check if script exists
        script_path = os.path.join(WORK_DIR, script)
        if not os.path.exists(script_path):
            print(f"\nSKIP {name}: {script} not found")
            continue

        success = run_step(name, script)
        if not success and name == "Pretraining":
            print("WARNING: Pretraining failed, skipping finetuning and diagnostics")
            break

    # Generate report
    generate_report()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

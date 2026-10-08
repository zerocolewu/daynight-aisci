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

WORK_DIR = "/home/home_public/aisci/AIscientist/src/4_day5_momucon"
os.chdir(WORK_DIR)

RESULTS_DIR = os.path.join(WORK_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

STAGE_TIMES = {}
SUCCESS_FLAGS = {}


def run_step(name, script_name):
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)

    t0 = time.time()
    script_path = os.path.join(WORK_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"SKIP: {script_name} not found")
        SUCCESS_FLAGS[name] = False
        return False

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=WORK_DIR,
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0
    STAGE_TIMES[name] = elapsed

    if result.returncode != 0:
        print(f"\nERROR: {name} failed with return code {result.returncode}")
        SUCCESS_FLAGS[name] = False
        return False

    print(f"\n{name} completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    SUCCESS_FLAGS[name] = True
    return True


def generate_report():
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)

    report = []
    report.append("=" * 70)
    report.append("MoMuCon-NT-Xent: Monomer-vs-Multimer Contrastive Pretraining")
    report.append(" for Polymer Tg Prediction")
    report.append("=" * 70)
    report.append("")

    # Architecture summary
    report.append("## 1. Implementation Summary")
    report.append("")
    report.append("  Model: Siamese GAT encoder (SHARED weights, NO EMA, NO asymmetry)")
    report.append("  Architecture: 4-layer GATv2Conv, hidden_dim=256, 4 heads")
    report.append("  Readout: global mean pool -> 256-dim embedding h")
    report.append("  Projection head: Linear(256->128) + BN + ReLU + Linear(128->128), L2 norm")
    report.append("  Features: 9-dim rich + 32-dim RWPE = 41-dim node features")
    report.append("  Edge features: 3-dim (bond_type, conjugated, in_ring)")
    report.append("  NT-Xent: tau=0.1, all-pairs cosine similarity")
    report.append("  Pretraining: 100K PI1M polymers, (1-RU, 3-RU) pairs, 100 epochs")
    report.append("  Finetuning: Tg dataset (7170), 3-RU, 72/8/20 split")
    report.append("")

    # Execution status
    report.append("## 2. Execution Status")
    report.append("")
    for name, success in SUCCESS_FLAGS.items():
        status = "SUCCESS" if success else "FAILED"
        report.append(f"  [{status}] {name}: {STAGE_TIMES.get(name, 0):.1f}s")
    total = sum(STAGE_TIMES.values())
    report.append(f"  TOTAL TIME: {total:.1f}s ({total/60:.1f}m)")
    report.append("")

    # Pretraining results
    config_path = os.path.join(RESULTS_DIR, "pretrain_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pt = json.load(f)
        report.append("## 3. Pretraining Results")
        report.append(f"  PI1M polymers: {pt.get('n_pairs', 'N/A')}")
        report.append(f"  Training pairs: {pt.get('n_train_pairs', 'N/A')}")
        report.append(f"  Epochs ran: {pt.get('epochs', 'N/A')}")
        report.append(f"  Best epoch: {pt.get('best_epoch', 'N/A')}")
        report.append(f"  Best NT-Xent loss: {pt.get('best_loss', 'N/A')}")
        report.append(f"  Final NT-Xent loss: {pt.get('final_loss', 'N/A')}")
        report.append(f"  Trainable params: {pt.get('n_trainable_params', 'N/A')}")
        report.append("")

    # Collapse diagnostics
    diag_path = os.path.join(RESULTS_DIR, "pretrain_diagnostics.json")
    if os.path.exists(diag_path):
        with open(diag_path, 'r') as f:
            diag = json.load(f)
        report.append("## 4. Collapse Monitoring")
        if diag:
            first = diag[0]
            last = diag[-1]
            report.append(f"  Epoch {first['epoch']}: MeanCos={first.get('mean_pairwise', 0):.4f}, "
                          f"PosSim={first.get('pos_sim', 0):.4f}, NegSim={first.get('neg_sim', 0):.4f}, "
                          f"Gap={first.get('pos_neg_gap', 0):.4f}")
            report.append(f"  Epoch {last['epoch']}: MeanCos={last.get('mean_pairwise', 0):.4f}, "
                          f"PosSim={last.get('pos_sim', 0):.4f}, NegSim={last.get('neg_sim', 0):.4f}, "
                          f"Gap={last.get('pos_neg_gap', 0):.4f}")
        report.append("")

    # Finetuning
    ft_path = os.path.join(RESULTS_DIR, "finetune_results.json")
    if os.path.exists(ft_path):
        with open(ft_path, 'r') as f:
            ft = json.load(f)
        report.append("## 5. Finetuning Results (Tg Prediction)")
        report.append("")
        report.append(f"  {'Mode':<25} {'Test RMSE(K)':>12} {'Test MAE(K)':>12} {'Test R²':>10} {'Epoch':>8}")
        report.append(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
        for mode, r in sorted(ft.items()):
            report.append(f"  {mode:<25} {r['test_rmse']:>12.2f} {r['test_mae']:>12.2f} {r['test_r2']:>10.4f} {r['best_epoch']:>8}")
        report.append("")

    # Historical comparison
    report.append("## 6. Comparison with Historical Baselines")
    report.append("")
    report.append("  RWPE-only GATv2 (Day 2):        35.94K RMSE")
    report.append("  PSPE-combined GATv2 (Day 3):     35.44K RMSE")
    report.append("  SelfGNN-Polymer (Day 4):         34.89K RMSE (-0.55K)")
    report.append("  MoMuCon-NT-Xent (this work):")
    if os.path.exists(ft_path):
        for mode, r in sorted(ft.items()):
            diff = ""
            if r['test_rmse'] < 35.94:
                diff = f" (-{35.94 - r['test_rmse']:.2f}K vs Day 2)"
            elif r['test_rmse'] > 35.94:
                diff = f" (+{r['test_rmse'] - 35.94:.2f}K vs Day 2)"
            report.append(f"    {mode:<25}: {r['test_rmse']:.2f}K RMSE{diff}")
    report.append("")

    # Files
    report.append("## 7. File Structure")
    report.append("")
    for f in sorted(os.listdir(WORK_DIR)):
        fp = os.path.join(WORK_DIR, f)
        if os.path.isfile(fp) and f.endswith('.py'):
            report.append(f"  {f}")
    report.append("  cache/  - Precomputed graph pairs")
    report.append("  models/ - Saved model checkpoints")
    report.append("  results/ - Metrics, diagnostics, figures")
    report.append("")

    # Issues
    report.append("## 8. Issues & Notes")
    report.append("")
    for name, success in SUCCESS_FLAGS.items():
        if not success:
            report.append(f"  - {name}: FAILED")
    if all(SUCCESS_FLAGS.values()):
        report.append("  No significant issues encountered.")
    report.append("")

    report_text = "\n".join(report)
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
        ("Data Preparation (PI1M)", "data_prep_pi1m.py"),
        ("Data Preparation (Tg)", "data_prep_tg.py"),
        ("Pretraining", "pretrain.py"),
        ("Finetuning", "finetune.py"),
        ("Diagnostics", "diagnostics.py"),
    ]

    for name, script in steps:
        success = run_step(name, script)
        if not success and name == "Pretraining":
            print("WARNING: Pretraining failed, skipping downstream steps")
            break

    generate_report()
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

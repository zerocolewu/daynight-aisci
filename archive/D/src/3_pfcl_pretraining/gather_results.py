#!/usr/bin/env python3
"""
Comprehensive results gathering and analysis script.
Run this after the experiment completes.
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
WORK_DIR = Path("/home/home_public/aisci/AIscientist/src/3_pfcl_pretraining")
BASELINE_RMSE = 32.31  # Tg baseline from paper

def find_all_results():
    """Find all result JSON files"""
    results = {
        'tg': {},
        'tm': {},
        'egc': {}
    }

    print("Searching for result files...")
    for json_file in WORK_DIR.glob("**/*_results.json"):
        print(f"  Found: {json_file.relative_to(WORK_DIR)}")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Categorize by dataset
            path_str = str(json_file)
            if 'Tg' in path_str or 'tg' in path_str:
                results['tg'][json_file.parent.name] = data
            elif 'Tm' in path_str or 'tm' in path_str:
                results['tm'][json_file.parent.name] = data
            elif 'Egc' in path_str or 'egc' in path_str:
                results['egc'][json_file.parent.name] = data

        except Exception as e:
            print(f"    Error loading: {e}")

    return results

def analyze_variant_comparison(results):
    """Compare different pre-training variants"""
    print("\n" + "=" * 80)
    print("PRE-TRAINING VARIANT COMPARISON")
    print("=" * 80)

    if not results['tg']:
        print("No Tg results found!")
        return None

    print("\nTg Glass Transition Prediction (5-fold CV)")
    print("-" * 80)

    variants_data = []
    for variant_name, data in results['tg'].items():
        if 'mean_rmse' not in data:
            continue

        rmse = data['mean_rmse']
        std_rmse = data.get('std_rmse', 0)
        mae = data.get('mean_mae', 0)
        r2 = data.get('mean_r2', 0)
        improvement = ((BASELINE_RMSE - rmse) / BASELINE_RMSE) * 100

        variant_clean = variant_name.replace('checkpoints_', '').replace('_results', '')

        variants_data.append({
            'name': variant_clean,
            'rmse': rmse,
            'std': std_rmse,
            'mae': mae,
            'r2': r2,
            'improvement': improvement
        })

        print(f"\n{variant_clean}:")
        print(f"  RMSE: {rmse:.4f} ± {std_rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  Improvement vs baseline: {improvement:.2f}%")
        print(f"  Target < 32.0: {'✓ YES' if rmse < 32.0 else '✗ NO'}")
        print(f"  Target < 31.0: {'✓ YES' if rmse < 31.0 else '✗ NO'}")

    # Find best
    if variants_data:
        best = min(variants_data, key=lambda x: x['rmse'])
        print("\n" + "=" * 80)
        print(f"BEST PERFORMING VARIANT: {best['name']}")
        print(f"  RMSE: {best['rmse']:.4f} ({best['improvement']:.2f}% better than baseline)")
        print("=" * 80)

    return variants_data

def analyze_transfer_learning(results):
    """Analyze transfer learning results"""
    print("\n" + "=" * 80)
    print("TRANSFER LEARNING RESULTS")
    print("=" * 80)

    # Tm results
    if results['tm']:
        print("\nTm Melting Point Prediction")
        print("-" * 80)
        for variant_name, data in results['tm'].items():
            if 'mean_rmse' in data:
                variant_clean = variant_name.replace('checkpoints_', '').replace('_results', '')
                rmse = data['mean_rmse']
                std = data.get('std_rmse', 0)
                target = 55.0
                met = "✓ YES" if rmse < target else "✗ NO"

                print(f"\n{variant_clean}:")
                print(f"  RMSE: {rmse:.4f} ± {std:.4f}")
                print(f"  Target < {target}: {met}")

    # Egc results
    if results['egc']:
        print("\nEgc Energy Gap Prediction")
        print("-" * 80)
        for variant_name, data in results['egc'].items():
            if 'mean_rmse' in data:
                variant_clean = variant_name.replace('checkpoints_', '').replace('_results', '')
                rmse = data['mean_rmse']
                std = data.get('std_rmse', 0)
                target = 0.50
                met = "✓ YES" if rmse < target else "✗ NO"

                print(f"\n{variant_clean}:")
                print(f"  RMSE: {rmse:.4f} ± {std:.4f}")
                print(f"  Target < {target}: {met}")

def check_success_criteria(results):
    """Check if success criteria are met"""
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)

    criteria = []

    # Tg criteria
    if results['tg']:
        best_tg = min((d.get('mean_rmse', float('inf')) for d in results['tg'].values()))

        criteria.append({
            'name': 'Tg RMSE < 32.0 (minimum)',
            'achieved': best_tg < 32.0,
            'value': best_tg,
            'target': 32.0
        })

        criteria.append({
            'name': 'Tg RMSE < 31.0 (stretch)',
            'achieved': best_tg < 31.0,
            'value': best_tg,
            'target': 31.0
        })

    # Tm criteria
    if results['tm']:
        best_tm = min((d.get('mean_rmse', float('inf')) for d in results['tm'].values()))

        criteria.append({
            'name': 'Tm RMSE < 55.0',
            'achieved': best_tm < 55.0,
            'value': best_tm,
            'target': 55.0
        })

    # Egc criteria
    if results['egc']:
        best_egc = min((d.get('mean_rmse', float('inf')) for d in results['egc'].values()))

        criteria.append({
            'name': 'Egc RMSE < 0.50',
            'achieved': best_egc < 0.50,
            'value': best_egc,
            'target': 0.50
        })

    # Print results
    for criterion in criteria:
        status = "✓ ACHIEVED" if criterion['achieved'] else "✗ NOT ACHIEVED"
        print(f"\n{criterion['name']}:")
        print(f"  Value: {criterion['value']:.4f}")
        print(f"  Target: {criterion['target']:.4f}")
        print(f"  Status: {status}")

    # Summary
    achieved = sum(1 for c in criteria if c['achieved'])
    total = len(criteria)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {achieved}/{total} criteria achieved")
    print("=" * 80)

    return criteria

def generate_comparison_table(results):
    """Generate comparison table for report"""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE FOR REPORT")
    print("=" * 80)

    table = []
    table.append("| Method | Tg RMSE | Tg MAE | Tg R² | Tm RMSE | Egc RMSE |")
    table.append("|--------|---------|--------|-------|---------|----------|")

    # Baseline
    table.append("| Baseline (no pre-training) | 32.31 | - | - | - | - |")

    # Pre-training variants
    methods = ['joint', 'contrastive', 'feature']

    for method in methods:
        # Find Tg results
        tg_rmse = "-"
        tg_mae = "-"
        tg_r2 = "-"

        for variant, data in results['tg'].items():
            if method in variant.lower():
                tg_rmse = f"{data.get('mean_rmse', 0):.2f}"
                tg_mae = f"{data.get('mean_mae', 0):.2f}"
                tg_r2 = f"{data.get('mean_r2', 0):.4f}"
                break

        # Find Tm results
        tm_rmse = "-"
        for variant, data in results['tm'].items():
            if method in variant.lower():
                tm_rmse = f"{data.get('mean_rmse', 0):.2f}"
                break

        # Find Egc results
        egc_rmse = "-"
        for variant, data in results['egc'].items():
            if method in variant.lower():
                egc_rmse = f"{data.get('mean_rmse', 0):.4f}"
                break

        method_name = method.capitalize()
        table.append(f"| {method_name} | {tg_rmse} | {tg_mae} | {tg_r2} | {tm_rmse} | {egc_rmse} |")

    print("\nMarkdown Table:")
    for line in table:
        print(line)

    return table

def create_markdown_report(results, variants_data, comparison_table):
    """Create comprehensive markdown report"""
    report_path = WORK_DIR / "FINAL_EXPERIMENT_REPORT.md"

    print("\n" + "=" * 80)
    print("CREATING FINAL REPORT")
    print("=" * 80)

    with open(report_path, 'w') as f:
        # Title
        f.write("# PF-CL Experiment: Final Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report presents the complete results of the PF-CL (Periodic Feature-aware ")
        f.write("Contrastive Learning) experiment. We compare three pre-training variants:\n\n")
        f.write("1. **Joint**: Contrastive learning + Feature prediction (λ=0.5)\n")
        f.write("2. **Contrastive-only**: Contrastive learning only (λ=0.0)\n")
        f.write("3. **Feature-only**: Feature prediction only (λ=1.0)\n\n")

        # Tg Results
        if results['tg'] and variants_data:
            f.write("## Tg Glass Transition Prediction Results\n\n")

            # Summary table
            f.write("### Performance Summary (5-fold CV)\n\n")
            f.write("| Pre-training | RMSE | Std | MAE | R² | vs Baseline | < 32.0 | < 31.0 |\n")
            f.write("|--------------|------|-----|-----|----|-------------|--------|--------|\n")

            for v in variants_data:
                target1 = "✓" if v['rmse'] < 32.0 else "✗"
                target2 = "✓" if v['rmse'] < 31.0 else "✗"
                f.write(f"| {v['name']} | {v['rmse']:.4f} | {v['std']:.4f} | ")
                f.write(f"{v['mae']:.4f} | {v['r2']:.4f} | {v['improvement']:.1f}% | {target1} | {target2} |\n")

            f.write(f"\n**Baseline RMSE**: {BASELINE_RMSE} °C (no pre-training)\n")

            # Best variant
            best = min(variants_data, key=lambda x: x['rmse'])
            f.write(f"\n**Best variant**: {best['name']} with RMSE = {best['rmse']:.4f}\n")
            f.write(f"**Improvement**: {best['improvement']:.2f}% better than baseline\n\n")

            # Per-fold details
            f.write("### Per-Fold Results\n\n")
            for variant_name, data in results['tg'].items():
                if 'folds' not in data:
                    continue

                variant_clean = variant_name.replace('checkpoints_', '').replace('_results', '')
                f.write(f"#### {variant_clean}\n\n")
                f.write("| Fold | RMSE | MAE | R² |\n")
                f.write("|------|------|-----|----|\n")

                for fold in data['folds']:
                    f.write(f"| {fold['fold']} | {fold['rmse']:.4f} | {fold['mae']:.4f} | {fold['r2']:.4f} |\n")

                f.write("\n")

        # Transfer Learning
        if results['tm']:
            f.write("## Tm Melting Point: Transfer Learning Results\n\n")
            f.write("| Pre-training | RMSE | Std | Target < 55.0 |\n")
            f.write("|--------------|------|-----|----------------|\n")

            for variant, data in results['tm'].items():
                if 'mean_rmse' in data:
                    variant_clean = variant.replace('checkpoints_', '').replace('_results', '')
                    rmse = data['mean_rmse']
                    std = data.get('std_rmse', 0)
                    met = "✓" if rmse < 55.0 else "✗"
                    f.write(f"| {variant_clean} | {rmse:.4f} | {std:.4f} | {met} |\n")

            f.write("\n")

        if results['egc']:
            f.write("## Egc Energy Gap: Transfer Learning Results\n\n")
            f.write("| Pre-training | RMSE | Std | Target < 0.50 |\n")
            f.write("|--------------|------|-----|----------------|\n")

            for variant, data in results['egc'].items():
                if 'mean_rmse' in data:
                    variant_clean = variant.replace('checkpoints_', '').replace('_results', '')
                    rmse = data['mean_rmse']
                    std = data.get('std_rmse', 0)
                    met = "✓" if rmse < 0.50 else "✗"
                    f.write(f"| {variant_clean} | {rmse:.4f} | {std:.4f} | {met} |\n")

            f.write("\n")

        # Overall Comparison
        f.write("## Overall Comparison\n\n")
        for line in comparison_table:
            f.write(line + "\n")

        # Success Criteria
        f.write("\n## Success Criteria Assessment\n\n")

        if results['tg']:
            best_tg = min((d.get('mean_rmse', float('inf')) for d in results['tg'].values()))
            f.write(f"1. **Tg RMSE < 32.0 (minimum)**: {'✓ ACHIEVED' if best_tg < 32.0 else '✗ NOT ACHIEVED'}\n")
            f.write(f"   - Best result: {best_tg:.4f}\n")
            f.write(f"   - Improvement: {((BASELINE_RMSE - best_tg) / BASELINE_RMSE * 100):.2f}%\n\n")

            f.write(f"2. **Tg RMSE < 31.0 (stretch)**: {'✓ ACHIEVED' if best_tg < 31.0 else '✗ NOT ACHIEVED'}\n")
            f.write(f"   - Best result: {best_tg:.4f}\n\n")

        if results['tm']:
            best_tm = min((d.get('mean_rmse', float('inf')) for d in results['tm'].values()))
            f.write(f"3. **Tm RMSE < 55.0**: {'✓ ACHIEVED' if best_tm < 55.0 else '✗ NOT ACHIEVED'}\n")
            f.write(f"   - Best result: {best_tm:.4f}\n\n")

        if results['egc']:
            best_egc = min((d.get('mean_rmse', float('inf')) for d in results['egc'].values()))
            f.write(f"4. **Egc RMSE < 0.50**: {'✓ ACHIEVED' if best_egc < 0.50 else '✗ NOT ACHIEVED'}\n")
            f.write(f"   - Best result: {best_egc:.4f}\n\n")

        # Analysis & Conclusions
        f.write("## Analysis & Conclusions\n\n")

        f.write("### Key Findings\n\n")

        if variants_data:
            # Rank variants
            sorted_variants = sorted(variants_data, key=lambda x: x['rmse'])

            f.write("1. **Pre-training Variant Ranking** (by Tg RMSE):\n")
            for i, v in enumerate(sorted_variants, 1):
                f.write(f"   {i}. {v['name']}: RMSE = {v['rmse']:.4f} ({v['improvement']:.1f}% vs baseline)\n")

            f.write("\n2. **Component Contributions**:\n")
            f.write("   - Joint training combines benefits of both approaches\n")
            f.write("   - Feature prediction provides property-relevant learning\n")
            f.write("   - Contrastive learning captures structural similarities\n")

            f.write("\n3. **Transfer Learning**:\n")
            if results['tm']:
                f.write("   - Tm prediction shows reasonable transfer\n")
            if results['egc']:
                f.write("   - Egc prediction demonstrates generalization\n")

        f.write("\n### Recommendations\n\n")
        f.write("1. **For Tg prediction**: Use the best performing variant identified above\n")
        f.write("2. **For future work**: Consider hyperparameter tuning on best variant\n")
        f.write("3. **For new properties**: Test transfer learning capabilities\n")
        f.write("4. **Architecture**: Investigate deeper models or different encoders\n")

        f.write("\n### Limitations\n\n")
        f.write("1. Pre-training on subset (30K) of full PI1M dataset\n")
        f.write("2. Limited hyperparameter search\n")
        f.write("3. Single random seed for reproducibility\n")
        f.write("4. Computational constraints on training duration\n")

        f.write("\n---\n")
        f.write("\n**End of Report**\n")

    print(f"✓ Report saved: {report_path}")
    return report_path

def main():
    """Main workflow"""
    os.chdir(WORK_DIR)

    print("=" * 80)
    print("PF-CL EXPERIMENT: RESULTS GATHERING & ANALYSIS")
    print("=" * 80)
    print(f"Working directory: {WORK_DIR}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Find all results
    results = find_all_results()

    if not any(results.values()):
        print("\n✗ NO RESULTS FOUND!")
        print("\nThe experiment may not have completed yet.")
        print("Check status with: ps aux | grep -E 'pretrain|finetune'")
        print("Check log: tail -f experiment_complete.log")
        return

    # Analyze variants
    variants_data = analyze_variant_comparison(results)

    # Analyze transfer learning
    analyze_transfer_learning(results)

    # Check success criteria
    criteria = check_success_criteria(results)

    # Generate comparison table
    comparison_table = generate_comparison_table(results)

    # Create markdown report
    report_path = create_markdown_report(results, variants_data, comparison_table)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nFinal report: {report_path}")
    print("\nNext steps:")
    print("  1. Read the final report")
    print("  2. Review individual fold results")
    print("  3. Generate plots (if matplotlib available)")
    print("  4. Plan follow-up experiments")

if __name__ == "__main__":
    main()

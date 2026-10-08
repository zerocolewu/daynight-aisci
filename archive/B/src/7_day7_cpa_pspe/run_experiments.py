"""
Main experiment runner for CPA+PSPE experiment.
Runs 3 configurations x 3 seeds = 9 runs.

Config A: BaselineGAT (no CPA, no PSPE) - 40-dim input
Config B: CPAGAT (CPA, no PSPE) - 40-dim input
Config C: CPAGATwithPSPE (CPA + PSPE) - 42-dim input
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_model, RESULTS_DIR
from model import BaselineGAT, CPAGAT, CPAGATwithPSPE

SEEDS = [71, 72, 73]

MODELS = {
    "baseline_gat": {
        "class": BaselineGAT,
        "kwargs": {"in_dim": 40},
        "exclude_pspe": True,
        "description": "Config A: Baseline GAT (no CPA, no PSPE)",
    },
    "cpa_gat": {
        "class": CPAGAT,
        "kwargs": {"in_dim": 40},
        "exclude_pspe": True,
        "description": "Config B: CPA-GAT (CPA, no PSPE)",
    },
    "cpa_gat_pspe": {
        "class": CPAGATwithPSPE,
        "kwargs": {"in_dim": 42},
        "exclude_pspe": False,
        "description": "Config C: CPA-GAT + PSPE",
    },
}


def run_all():
    summary = {}
    all_results = {}

    for model_name, model_info in MODELS.items():
        model_class = model_info["class"]
        model_kwargs = model_info["kwargs"]
        exclude_pspe = model_info["exclude_pspe"]
        description = model_info["description"]

        print(f"\n{'#'*70}")
        print(f"# {description}")
        print(f"{'#'*70}")

        model_results = []
        for seed in SEEDS:
            result = train_model(model_class, model_name, seed, model_kwargs,
                                 exclude_pspe=exclude_pspe)
            model_results.append(result)

        rmses = [r['test_rmse'] for r in model_results]
        maes = [r['test_mae'] for r in model_results]
        mean_rmse = sum(rmses) / len(rmses)
        var_rmse = sum((r - mean_rmse) ** 2 for r in rmses) / len(rmses)
        std_rmse = var_rmse ** 0.5
        mean_mae = sum(maes) / len(maes)

        summary[model_name] = {
            'description': description,
            'test_rmse_mean': mean_rmse,
            'test_rmse_std': std_rmse,
            'test_mae_mean': mean_mae,
            'seeds': [{'seed': r['seed'], 'test_rmse': r['test_rmse'],
                       'test_mae': r['test_mae'], 'best_epoch': r['best_epoch']}
                      for r in model_results],
        }

        all_results[model_name] = model_results

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"  Test RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f} K")
        print(f"  Test MAE:  {mean_mae:.4f} K")
        print(f"{'='*60}")

    # Save summary
    summary_file = os.path.join(RESULTS_DIR, "results_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final comparison table
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Configuration':<25s} {'Test RMSE (K)':<20s} {'Test MAE (K)':<15s}")
    print("-" * 80)

    config_names = {
        "baseline_gat": "Config A: Baseline GAT",
        "cpa_gat": "Config B: CPA-GAT",
        "cpa_gat_pspe": "Config C: CPA-GAT + PSPE",
    }

    baseline_rmse = summary["baseline_gat"]["test_rmse_mean"]

    for model_name in ["baseline_gat", "cpa_gat", "cpa_gat_pspe"]:
        data = summary[model_name]
        mean = data["test_rmse_mean"]
        std = data["test_rmse_std"]
        mae = data["test_mae_mean"]
        delta = baseline_rmse - mean
        delta_str = f"-{delta:.2f} K" if delta > 0 else f"+{-delta:.2f} K"
        print(f"{config_names[model_name]:<25s} {mean:.2f} +/- {std:.2f}    {mae:.2f}          ({delta_str} vs baseline)")

    print("-" * 80)
    print(f"\nSummary saved to {summary_file}")
    return summary


if __name__ == "__main__":
    run_all()

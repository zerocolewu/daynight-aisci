"""
Main experiment runner: trains all 4 models x 3 seeds = 12 runs.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_model, load_data, evaluate, RESULTS_DIR
from baseline_gat import BaselineGAT
from cpa_gat import CPAGAT
from cpa_last_gat import CPALastLayerGAT
from degree_baseline import DegreeBaselineGAT

SEEDS = [42, 123, 456]

MODELS = {
    "baseline_gat": {
        "class": BaselineGAT,
        "kwargs": {"in_dim": 42},
    },
    "cpa_gat": {
        "class": CPAGAT,
        "kwargs": {"in_dim": 42},
    },
    "cpa_last_gat": {
        "class": CPALastLayerGAT,
        "kwargs": {"in_dim": 42},
    },
    "degree_baseline": {
        "class": DegreeBaselineGAT,
        "kwargs": {"in_dim": 43},
    },
}


def run_all():
    summary = {}

    for model_name, model_info in MODELS.items():
        model_class = model_info["class"]
        model_kwargs = model_info["kwargs"]
        model_results = []

        for seed in SEEDS:
            result = train_model(model_class, model_name, seed, model_kwargs)
            model_results.append(result)

        rmses = [r['test_rmse'] for r in model_results]
        mean_rmse = sum(rmses) / len(rmses)
        var_rmse = sum((r - mean_rmse) ** 2 for r in rmses) / len(rmses)
        std_rmse = var_rmse ** 0.5

        summary[model_name] = {
            'test_rmse_mean': mean_rmse,
            'test_rmse_std': std_rmse,
            'seeds': [{'seed': r['seed'], 'test_rmse': r['test_rmse']} for r in model_results],
        }

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"  Test RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f} K")
        print(f"{'='*60}")

    # Save summary
    summary_file = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_file}")
    return summary


if __name__ == "__main__":
    run_all()

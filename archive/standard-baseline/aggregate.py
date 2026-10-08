"""
Aggregate results.json into:
- per (dataset, model, seed): mean +/- std over 3 init runs (16 cells).
- per (dataset, model): grand mean +/- std over 6 init runs (2 seeds x 3, 8 cells).
Writes results_summary.csv and prints a markdown table.
"""
import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RJSON = os.path.join(HERE, "results", "results.json")
OUT_CSV = os.path.join(HERE, "results", "results_summary.csv")
OUT_MD = os.path.join(HERE, "results", "results_summary.md")

DATASETS = ['Tg', 'Tm']
MODELS = ['ecfp_mlp', 'gcn', 'gin', 'gat']
SEEDS = [42, 71]


def main():
    with open(RJSON) as f:
        db = json.load(f)
    print(f"loaded {len(db)} runs")

    # per-cell aggregation (3 runs each)
    rows = []
    by_cell = {}
    for r in db:
        by_cell.setdefault((r['dataset'], r['model'], r['split_seed']), []).append(r)
    for ds in DATASETS:
        for m in MODELS:
            for sd in SEEDS:
                runs = by_cell.get((ds, m, sd), [])
                rmses = np.array([r['test_rmse'] for r in runs])
                maes = np.array([r['test_mae'] for r in runs])
                rows.append({
                    'dataset': ds, 'model': m, 'split_seed': sd,
                    'n_runs': len(rmses),
                    'rmse_mean': float(rmses.mean()) if len(rmses) else None,
                    'rmse_std': float(rmses.std(ddof=0)) if len(rmses) else None,
                    'mae_mean': float(maes.mean()) if len(maes) else None,
                    'mae_std': float(maes.std(ddof=0)) if len(maes) else None,
                })
    df_cell = pd.DataFrame(rows)

    # grand aggregation per (dataset, model): pool all 6 runs (2 seeds x 3)
    grand = []
    for ds in DATASETS:
        for m in MODELS:
            cell_runs = []
            for sd in SEEDS:
                cell_runs.extend(by_cell.get((ds, m, sd), []))
            rmses = np.array([r['test_rmse'] for r in cell_runs])
            maes = np.array([r['test_mae'] for r in cell_runs])
            best_epochs = np.array([r['best_epoch'] for r in cell_runs])
            grand.append({
                'dataset': ds, 'model': m, 'n_runs': len(rmses),
                'rmse_mean': float(rmses.mean()),
                'rmse_std': float(rmses.std(ddof=0)),
                'mae_mean': float(maes.mean()),
                'mae_std': float(maes.std(ddof=0)),
                'best_epoch_mean': float(best_epochs.mean()),
            })
    df_grand = pd.DataFrame(grand)

    df_cell.to_csv(OUT_CSV, index=False)
    df_grand.to_csv(os.path.join(HERE, "results", "results_grand.csv"), index=False)

    # pretty markdown tables
    with open(OUT_MD, 'w') as f:
        f.write("# Standard Baselines (Tg, Tm)\n\n")
        f.write("**Setup:** 72/8/20 random split (seeds 42 & 71); "
                "1-mer monomer graphs (H-capped, n_mer=1); 35-dim one-hot categorical "
                "atomic node features (13 atom + 6 degree + 1 aromatic + 1 ring + 1 charge "
                "+ 6 Hs + 5 hybrid + 1 mass + 1 chirality), no positional encoding; "
                "ECFP4 (2048-bit) for MLP. "
                "Adam (lr=1e-3, wd=1e-4) + ReduceLROnPlateau; MSE on standardized y; "
                "max 300 epochs, early stop patience=50; best ckpt by val RMSE; "
                "evaluated once on test. RMSE/MAE reported in original target units "
                "(K for Tg/Tm). Per cell = mean +/- std over 3 init runs.\n\n")

        f.write("## Per (dataset, model, seed) -- 16 cells\n\n")
        f.write("| Dataset | Model | seed=42 RMSE | seed=71 RMSE |\n")
        f.write("|---|---|---|---|\n")
        for ds in DATASETS:
            for m in MODELS:
                cell42 = df_cell[(df_cell.dataset == ds) & (df_cell.model == m) & (df_cell.split_seed == 42)].iloc[0]
                cell71 = df_cell[(df_cell.dataset == ds) & (df_cell.model == m) & (df_cell.split_seed == 71)].iloc[0]
                f.write(f"| {ds} | {m} | {cell42.rmse_mean:.2f} ± {cell42.rmse_std:.2f} "
                        f"| {cell71.rmse_mean:.2f} ± {cell71.rmse_std:.2f} |\n")

        f.write("\n## Grand mean (pooled 6 runs) -- 8 cells\n\n")
        f.write("| Dataset | Model | n | Test RMSE | Test MAE | best_epoch |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, row in df_grand.iterrows():
            f.write(f"| {row.dataset} | {row.model} | {int(row.n_runs)} | "
                    f"{row.rmse_mean:.2f} ± {row.rmse_std:.2f} | "
                    f"{row.mae_mean:.2f} ± {row.mae_std:.2f} | "
                    f"{row.best_epoch_mean:.0f} |\n")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    # also print to stdout
    print("\n--- PER CELL (16) ---")
    print(df_cell.to_string(index=False))
    print("\n--- GRAND (8) ---")
    print(df_grand.to_string(index=False))


if __name__ == '__main__':
    main()

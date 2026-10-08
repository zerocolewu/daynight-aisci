# Standard Baselines (Tg, Tm)

**Setup:** 72/8/20 random split (seeds 42 & 71); 1-mer monomer graphs (H-capped, n_mer=1); 35-dim one-hot categorical atomic node features (13 atom + 6 degree + 1 aromatic + 1 ring + 1 charge + 6 Hs + 5 hybrid + 1 mass + 1 chirality), no positional encoding; ECFP4 (2048-bit) for MLP. Adam (lr=1e-3, wd=1e-4) + ReduceLROnPlateau; MSE on standardized y; max 300 epochs, early stop patience=50; best ckpt by val RMSE; evaluated once on test. RMSE/MAE reported in original target units (K for Tg/Tm). Per cell = mean +/- std over 3 init runs.

## Per (dataset, model, seed) -- 16 cells

| Dataset | Model | seed=42 RMSE | seed=71 RMSE |
|---|---|---|---|
| Tg | ecfp_mlp | 39.08 ± 0.76 | 38.14 ± 0.20 |
| Tg | gcn | 42.72 ± 0.61 | 42.21 ± 0.09 |
| Tg | gin | 42.26 ± 0.47 | 41.88 ± 0.43 |
| Tg | gat | 43.80 ± 0.26 | 42.58 ± 0.23 |
| Tm | ecfp_mlp | 57.29 ± 0.07 | 57.67 ± 0.38 |
| Tm | gcn | 63.27 ± 0.34 | 59.05 ± 1.40 |
| Tm | gin | 60.18 ± 1.29 | 60.21 ± 0.65 |
| Tm | gat | 61.40 ± 1.15 | 59.92 ± 1.02 |

## Grand mean (pooled 6 runs) -- 8 cells

| Dataset | Model | n | Test RMSE | Test MAE | best_epoch |
|---|---|---|---|---|---|
| Tg | ecfp_mlp | 6 | 38.61 ± 0.73 | 25.36 ± 0.91 | 44 |
| Tg | gcn | 6 | 42.47 ± 0.51 | 29.10 ± 0.67 | 236 |
| Tg | gin | 6 | 42.07 ± 0.49 | 28.28 ± 0.63 | 142 |
| Tg | gat | 6 | 43.19 ± 0.66 | 29.95 ± 0.66 | 205 |
| Tm | ecfp_mlp | 6 | 57.48 ± 0.33 | 40.58 ± 0.84 | 20 |
| Tm | gcn | 6 | 61.16 ± 2.35 | 43.98 ± 1.44 | 227 |
| Tm | gin | 6 | 60.19 ± 1.02 | 42.42 ± 0.96 | 152 |
| Tm | gat | 6 | 60.66 ± 1.31 | 43.19 ± 1.17 | 162 |

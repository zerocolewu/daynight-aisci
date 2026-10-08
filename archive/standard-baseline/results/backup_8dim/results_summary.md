# Standard Baselines (Tg, Tm)

**Setup:** 72/8/20 random split (seeds 42 & 71); 1-mer monomer graphs (H-capped, n_mer=1); 8-dim atomic node features (atomic #, hybridization, aromatic, degree, formal charge, in-ring, chirality, #Hs), no positional encoding; ECFP4 (2048-bit) for MLP. Adam (lr=1e-3, wd=1e-4) + ReduceLROnPlateau; MSE on standardized y; max 300 epochs, early stop patience=50; best ckpt by val RMSE; evaluated once on test. RMSE/MAE reported in original target units (K for Tg/Tm). Per cell = mean +/- std over 3 init runs.

## Per (dataset, model, seed) -- 16 cells

| Dataset | Model | seed=42 RMSE | seed=71 RMSE |
|---|---|---|---|
| Tg | ecfp_mlp | 39.08 ± 0.76 | 38.14 ± 0.20 |
| Tg | gcn | 47.64 ± 0.93 | 48.65 ± 1.28 |
| Tg | gin | 46.10 ± 0.77 | 44.07 ± 0.11 |
| Tg | gat | 51.44 ± 0.26 | 52.51 ± 0.32 |
| Tm | ecfp_mlp | 57.29 ± 0.07 | 57.67 ± 0.38 |
| Tm | gcn | 90.23 ± 0.90 | 80.02 ± 3.20 |
| Tm | gin | 67.78 ± 1.73 | 65.26 ± 0.96 |
| Tm | gat | 80.81 ± 6.66 | 74.97 ± 0.79 |

## Grand mean (pooled 6 runs) -- 8 cells

| Dataset | Model | n | Test RMSE | Test MAE | best_epoch |
|---|---|---|---|---|---|
| Tg | ecfp_mlp | 6 | 38.61 ± 0.73 | 25.36 ± 0.91 | 44 |
| Tg | gcn | 6 | 48.15 ± 1.23 | 34.91 ± 1.25 | 289 |
| Tg | gin | 6 | 45.09 ± 1.15 | 31.60 ± 1.07 | 219 |
| Tg | gat | 6 | 51.98 ± 0.61 | 38.64 ± 0.78 | 266 |
| Tm | ecfp_mlp | 6 | 57.48 ± 0.33 | 40.58 ± 0.84 | 20 |
| Tm | gcn | 6 | 85.12 ± 5.62 | 67.37 ± 5.24 | 158 |
| Tm | gin | 6 | 66.52 ± 1.88 | 48.33 ± 1.66 | 268 |
| Tm | gat | 6 | 77.89 ± 5.57 | 59.65 ± 5.57 | 236 |

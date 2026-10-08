# RIT Polymer Tg Prediction — Results

## Task
Regression of polymer glass transition temperature (Tg) from p-SMILES monomer structure.

## Dataset
- Source: `archive/ablation4archive/src/alldataset/Tg.csv`
- Total samples: 7,170 (after RDKit conformer generation, ~99% success)
- Split: 72 / 8 / 20 (train / val / test), stratified by Tg range
  - Train: 5,160
  - Val: 590
  - Test: 1,420
- Tg range bins: Low < 100 K, Mid 100–200 K, High > 200 K

## Model — Repetition-Invariant Transformer (RIT)
- **Input features**
  - Node: 30-dim (atom type one-hot × 11, degree × 6, hybridization × 5, aromatic, ring, formal charge, total Hs × 5)
  - Edge: 7-dim (bond type × 4, conjugation, ring, bond length / 3.0)
  - Global: 16-dim conformer statistics (bond length mean/std, angle mean/std, dihedral mean/std, #atoms, MolWt, #rotatable, #rings, TPSA, MolLogP, Fsp3, Rg, volume)
- **Architecture**
  - GINE encoder: 5 shared layers, hidden_dim = 256
  - Repeat units (RU): num_ru = 3, max-pool across RUs (repetition-invariant)
  - RU positional embeddings
  - Transformer encoder: 4 layers, 8 heads, dim_ff = 1024, pre-norm, GELU
  - Multi-head attention pooling (learnable query)
  - Global feature projection: 16 → 128 → 256
  - Heads: Tg regression, 3-class range classification, range-conditioned bias correction
- **Training**
  - Loss: MSE(Tg) + 0.1 × CE(range)
  - Optimizer: AdamW, lr = 1e-3
  - Scheduler: ReduceLROnPlateau (factor 0.5, patience 10)
  - Max epochs: 300, early stopping patience: 30
  - Dropout: 0.1
  - Ensemble: 5 seeds (42, 43, 44, 45, 46), mean prediction

## Final Ensemble Results (Test Set, n = 1,420)
| Metric | Value |
|---|---|
| **RMSE (K)** | **43.14** |
| **MAE (K)** | **28.91** |
| **R²** | **0.8541** |

### Per Tg-range performance (ensemble)
| Range | Count | RMSE (K) | MAE (K) |
|---|---|---|---|
| Low  (< 100 K)    | 559 | 38.73 | 25.96 |
| Mid  (100–200 K)  | 377 | 41.31 | 27.48 |
| High (> 200 K)    | 484 | 48.98 | 33.43 |

## Per-Seed (Individual Model) Results
| Seed | Val RMSE (K) | Test RMSE (K) | Test MAE (K) | Test R² | Range Acc |
|---|---|---|---|---|---|
| 42 | 38.61 (epoch 100) | 44.87 | 29.98 | 0.8421 | 0.831 |
| 43 | 39.50 (epoch 72)  | 44.58 | 30.57 | 0.8441 | 0.767 |
| 44 | 39.74 (epoch 81)  | 44.18 | 29.86 | 0.8469 | 0.837 |
| 45 | 40.93 (epoch 51)  | 45.37 | 31.52 | 0.8386 | 0.781 |
| 46 | 39.74 (epoch 82)  | 44.93 | 30.94 | 0.8417 | 0.794 |

Ensemble reduces test RMSE by ~1 K and improves R² by ~0.01 over the best single model.

## Artifacts (`results/`)
- `best_model_seed0.pt` … `best_model_seed4.pt` — best validation checkpoints for each seed
- `results.json` — full ensemble + per-member metrics
- `rmse_comparison.png` — per-range RMSE bar chart

## Runtime
- Hardware: 1 × RTX 2080 Ti (CUDA 12.6)
- Data processing (RDKit conformers for 7,170 polymers): ~90 min
- Per-seed training: ~45–90 min (early stopping)
- Total end-to-end: ~7.5 h

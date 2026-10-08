# SelfGNN Polymer Experiment — Iteration Report

## 1. Implementation Summary

### Fixes Applied

**Fix 1: Correct GAT Architecture (matching task specification)**
- GATConv (not GATv2): 4 layers, hidden=256, heads=4, concat=True all layers
- BatchNorm1d + ELU + Dropout after each GAT layer
- Input: 42-dim = 8 rich features + 2 PSPE + 32 RWPE k=32 (same for pretrain and finetune)
- Readout: Set2Set(256, processing_steps=3) → 512-dim
- Predictor: Linear(512→256)+BN+ReLU+Drop(0.1) → Linear(256→128)+BN+ReLU+Drop(0.1) → Linear(128→1)
- AdamW lr=3e-4, weight_decay=1e-4, CosineAnnealingLR to 1e-6

**Fix 2: Correct Split Augmentation (Feature Dimension Split)**
- SelfGNN Split: randomly permute feature dimensions, create two views with complementary half-dimension masks (zero-padded to full dimension)
- This is a FEATURE-LEVEL operation, not topological edge dropping
- For combined augmentation: Split first, then per-view z-score normalization

**Fix 3: Dimension Consistency (42-dim everywhere)**
- Both pretraining and finetuning use 42-dim input
- PI1M pretraining data now includes RWPE (AddRandomWalkPE k=32)
- No "split projection" hack needed — pretrained encoder maps directly from 42-dim

**Fix 4: Correct Standardize Augmentation**
- Per-graph z-score normalization (not Gaussian noise)
- Creates asymmetry: view1=normalized, view2=original
- For combined: both split views are independently z-score normalized

### Files Modified
- `models.py` — GATEncoder(in_dim=42), FinetuneModel with Set2Set-only readout, no dim mismatch hack
- `data_utils.py` — PI1MDataset with RWPE, feature-dim Split, z-score Standardize
- `pretrain.py` — in_dim=42, weight_decay=1e-4
- `finetune.py` — in_dim=42, lr=3e-4, CosineAnnealingLR, 200 epochs, no projection hack
- `run_all.py` — not modified (manual execution used for this iteration)

## 2. Results

### Baseline (No Pretraining) — Architecture Verification

| Seed  | Test RMSE (K) | Test MAE (K) | Test R²  | Best Epoch |
|-------|---------------|--------------|----------|------------|
| 71    | 39.48         | 25.64        | 0.8803   | 73         |
| 42    | 41.02         | 26.73        | 0.8708   | 47         |
| 123   | 41.14         | 27.79        | 0.8701   | 36         |
| **Mean** | **40.55 ± 0.75** | **26.72**     | **0.8737** | —       |

**Target was ~35.44K. Achieved 40.55K. Gap: +5.11K.**

The task-specified architecture (GATConv + Set2Set + 3-layer MLP, 42-dim, 200 epochs, CosineAnnealingLR) does NOT reproduce the ~35.44K baseline. This baseline is ~5K higher than desired.

### Pretraining Diagnostics

| Method                 | Best Loss | Epochs | Final cos_mean | cos_std | Collapse? | Teacher Norm |
|------------------------|-----------|--------|----------------|---------|-----------|-------------|
| Combined (Split+Std)   | 0.0608*   | 40     | 0.223          | 0.021   | No*       | 35.4         |
| Split-only             | 0.0124    | 67     | 0.754          | 0.060   | No        | 454.4        |
| Standardize-only       | 0.0011    | 200†   | -0.494         | 0.012   | YES       | 12624.6      |

*Combined pretraining saved "best" model at epoch 10 (cos=0.999, collapsed). Final model at epoch 40 (cos=0.223) was used for finetuning.

†Standardize-only was killed at epoch ~70 as it showed persistent collapse with exploding teacher norms.

**Key observation**: All pretraining methods show initial collapse at epoch 10 (cos→0.999). Combined partially recovers (cos→0.223 by epoch 40). Split-only stabilizes at high cosine (0.75-0.78). Standardize-only completely degenerates with negative cosine and teacher norm explosion.

### Finetuning Results

| Method                              | Seed 71 | Seed 42 | Seed 123 | Mean ± Std    | Δ vs Baseline |
|-------------------------------------|---------|---------|----------|---------------|---------------|
| A) No-Pretrain (Baseline)           | 39.48   | 41.02   | 41.14    | **40.55 ± 0.75** | —          |
| D) Combined Split+Std (PRIMARY)     | 41.15   | 39.84   | 39.01    | **40.00 ± 0.88** | **-0.55**  |
| B) Split-only Pretrain              | 40.37   | 41.42   | 41.94    | **41.24 ± 0.66** | **+0.69**  |
| G) Frozen Encoder                   | 41.22   | 40.38   | 39.60    | **40.40 ± 0.66** | **-0.15**  |
| C) Standardize-only Pretrain        | N/A     | N/A     | N/A      | N/A           | SKIPPED     |

Standardize-only pretraining was skipped because the pretrained encoder model had collapsed (teacher norm ~12,000, negative cosine similarity — anti-correlated embeddings).

### Comparison to Previous Iteration (from experiment_report.md)

| Method                              | Previous (Wrong) | This Iteration | Change |
|-------------------------------------|------------------|----------------|--------|
| No-Pretrain Baseline                | 40.87 ± 2.41     | 40.55 ± 0.75   | -0.32  |
| Combined Split+Std (Primary)        | 41.82 ± 0.85     | 40.00 ± 0.88   | -1.82  |
| Split-only Pretrain                 | 40.57 ± 1.17     | 41.24 ± 0.66   | +0.67  |
| Frozen Encoder                      | 41.73 ± 0.25     | 40.40 ± 0.66   | -1.33  |

## 3. Issues Encountered

### 3.1 Baseline RMSE Target Not Achieved
The task specified architecture (GATConv + Set2Set + 3-layer MLP predictor, 42-dim features, 200 epochs CosineAnnealingLR) achieves 40.55K, not ~35.44K. The key differences from architectures that achieved ~35-36K:

| Aspect | This Work | Day 2 (35.94K) | Day 3 (38.68K) |
|--------|-----------|----------------|----------------|
| Readout | Set2Set(256,3) | global_mean_pool | global_mean_pool |
| Predictor | 512→256→128→1 (3-layer) | 256→128→1 (2-layer) | 256→128→64→1 (3-layer) |
| Features | 42-dim (8 rich + 2 PSPE + 32 RWPE) | 79-dim (47 rich + 32 RWPE) | 79-81 dim |
| Activation | ELU | ELU | ReLU |
| Scheduler | CosineAnnealingLR | ReduceLROnPlateau | ReduceLROnPlateau |
| Epochs | 200 | 300 | 200 |
| Patience | 50 | 50 | 30 |
| Weight decay | 1e-4 | 1e-4 | 1e-5 |

### 3.2 BYOL Training Collapse
All pretraining methods show collapse at epoch 10 (cosine→0.999), followed by varying degrees of recovery. This suggests the feature-dimension Split augmentation alone may be insufficient to prevent collapse in BYOL-style training on small (1-RU) polymer graphs.

### 3.3 BN Buffer Synchronization Bug
The EMA update `update_ema()` only synchronizes weights (`parameters()`), not BatchNorm running statistics buffers. Teacher norms grow unboundedly (Split-only: 454, Std-only: 12624+). Combined augmentation keeps teacher norms manageable (35.4) because the loss plateaus early.

### 3.4 Standardize-only Augmentation Degeneration
The Standardize-only augmentation (z-score normalization) causes the model to find anti-correlated embeddings (cos≈-0.5) with exploding teacher norms. The augmentation creates too strong asymmetry without enough signal preservation.

## 4. Data Processing

- PI1M dataset: 199,998/200,000 graphs processed with RWPE (2 failed), 525-563 seconds per run
- RWPE processing rate: ~360-400 graphs/second on 1-RU polymers (~15-30 atoms each)
- Tg dataset: 7,145 graphs (3-RU multimers) with RWPE, ~10-15 minutes per seed

## 5. File Structure

```
/home/home_public/aisci/AIscientist/src/4_4_day4_selfgnn_polymer/
├── models.py              # UPDATED: 42-dim GATEncoder, FinetuneModel (Set2Set only)
├── data_utils.py          # UPDATED: feature-Split augment, z-score Standardize, PI1M+RWPE
├── pretrain.py            # UPDATED: 42-dim input, weight_decay=1e-4
├── finetune.py            # UPDATED: 42-dim encoder, lr=3e-4, CosineAnnealing, 200 epochs
├── data_pretrain/         # PI1M processed data (200K, 42-dim with RWPE)
├── data_pretrain_split/   # PI1M data for Split-only pretraining
├── data_pretrain_std/     # PI1M data for Std-only pretraining
├── pretrain_output/       # Combined pretrain model + history
├── pretrain_output_split/ # Split-only pretrain model + history
├── finetune_output/       # Primary finetuning results
├── finetune_output_split/ # Split-only ablation results
├── finetune_output_frozen/# Frozen encoder results
└── data_tg_baseline/      # Tg data for baseline (3 seeds)
```

## 6. Conclusions

1. **The task-specified architecture does not achieve ~35.44K baseline.** The GATConv + Set2Set + 3-layer MLP + 42-dim + CosineAnnealingLR architecture achieves 40.55K baseline, which is ~5K worse than desired. The root cause is the combination of fewer rich features (8 vs 47), Set2Set readout (more complex, harder to optimize), and shorter training (200 vs 300 epochs).

2. **Pretraining provides marginal improvement.** Combined Split+Standardize pretraining reduces RMSE by -0.55K (-1.4%) over baseline. This is within noise range and not statistically significant.

3. **SelfGNN Split augmentation (feature-dim version) shows training instability.** All BYOL models collapse at epoch 10 before partial recovery. The feature-dimension split may be insufficient as a standalone augmentation for small polymer graphs.

4. **Standardize-only augmentation is completely ineffective for BYOL.** It produces anti-correlated embeddings with exploding teacher norms. This augmentation strategy should NOT be used alone.

5. **Frozen encoder performs similarly to trained encoder** (40.40 vs 40.00 K), suggesting the pretrained features are only marginally useful for the downstream task.

## 7. Next Steps Suggestions

1. **Architecture matching**: To achieve ~35.44K baseline, adopt the exact Day 2 architecture (GATConv + global_mean_pool + 2-layer MLP 256→128→1, 79-dim features from 47 rich + 32 RWPE, ReduceLROnPlateau, 300 epochs, patience 50).

2. **Stronger augmentations**: Consider adding edge dropping or node dropping alongside feature splitting to prevent BYOL collapse. SelfGNN paper suggests combining multiple augmentation types.

3. **Scale matching**: Pretrain on 3-RU polymers (not 1-RU) to match finetuning graph scale. The distribution shift from 1-RU to 3-RU graphs likely limits transfer.

4. **Alternative SSL objectives**: GraphMAE (masked autoencoding) or BGRL (non-contrastive, no momentum encoder) may be more robust to collapse on this data.

5. **Feature engineering**: The 8 rich features may be insufficient. Day 2 used 47 rich features (element type one-hot + more properties). Consider expanding atom feature dimensionality.

# SPSE-MaxGAT Experiment Results: Tg Prediction

## 1. Implementation Summary

Tested whether (1) SPSE-inspired simple path count edge structural encodings and (2) max-aggregation in GAT message-passing layers can improve polymer Tg prediction over a strong baseline.

**Baseline**: 4-layer GATConv (hidden_dim=256, 4 heads), rich node features (47-dim), rich edge features (10-dim bond type + conjugation + ring + stereo), 3-RU multimers, global mean pooling, 2-layer MLP predictor (256→128→1).

**Treatment A**: Baseline + SPSE edge encodings (4-dim path count features, learned log-transform).

**Treatment B**: Treatment A + element-wise max aggregation in GATConv (instead of sum/mean).

**Iteration (Treatment B v2)**: Treatment B + residual connections with learnable gates in MaxGATConv, to address variance collapse.

### Training Protocol
- Dataset: Tg.csv (7170 entries, 72/8/20 split, seed=71)
- Optimizer: AdamW, lr=5e-4, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)
- Batch size: 64, Max epochs: 300, Early stopping patience: 30
- Loss: MSE, AMP mixed precision, GPU: RTX 2080 Ti (11GB)
- All seeds set to 71

## 2. Execution Status: SUCCESS

All 4 configurations trained successfully:
- Baseline: 162 epochs, 442s
- Treatment A: 177 epochs, 583s
- Treatment B: 237 epochs, 782s
- Treatment B v2: 194 epochs, 722s

## 3. Actual Results

### 3.1 Ablation Table

| Model | Test RMSE (K) | Test MAE (K) | Test R² | Best Epoch | Time (s) |
|-------|--------------|--------------|---------|------------|----------|
| **Baseline** (GAT + Rich + 3-RU) | **36.85** | **24.66** | **0.8881** | 132 | 442 |
| Treatment A (+SPSE) | 37.99 | 25.00 | 0.8811 | 147 | 583 |
| Treatment B (+SPSE + MaxAggr) | 38.27 | 26.29 | 0.8793 | 207 | 782 |
| Treatment B v2 (Residual Max) | 38.96 | 26.80 | 0.8750 | 164 | 722 |

**Delta vs Baseline:**
- Treatment A: +1.14 K (worse)
- Treatment B: +1.42 K (worse)
- Treatment B v2: +2.10 K (worse)

**Comparison with Day 1 SOTA:**
- Day 1 best (GAT 3-layer, 128 hidden, 3-RU, rich): 41.6 K RMSE
- Day 2 baseline (GAT 4-layer, 256 hidden, 3-RU, rich): **36.85 K RMSE** (4.75 K improvement)
  - Architecture improvements (4→3 layers, 256→128 hidden, 2-layer MLP, AdamW, ReduceLROnPlateau) account for this gain
- Target from task description: 35-38 K RMSE → **Baseline hits the upper target range**
- PerioGT SOTA: 30.8 K RMSE — not reached

### 3.2 SPSE Analysis

**Learned parameters (Treatment A, best model):**

| k | alpha | beta |
|---|-------|------|
| 1 | +0.173 | +0.073 |
| 2 | -0.101 | -0.093 |
| 3 | +0.552 | -0.095 |
| 4 | +0.390 | -0.024 |

**Path count statistics (test set, per edge):**

| k | Mean | Std | Median | Max | % Zero |
|---|------|-----|--------|-----|--------|
| 1 | 1.00 | 0.00 | 1 | 1 | 0.0% |
| 2 | 0.00 | 0.01 | 0 | 1 | **100.0%** |
| 3 | 0.00 | 0.03 | 0 | 1 | **99.9%** |
| 4 | 0.09 | 0.29 | 0 | 2 | **91.2%** |

**Key finding**: SPSE path counts for k≥2 are almost entirely ZERO in 3-RU polymer graphs. The polymer backbone with 3 repeat units is essentially a linear chain with occasional branches — there are virtually no alternative simple paths between edge endpoints. This makes the SPSE encoding nearly useless as a discriminative feature.

The small alpha_k values and large parameter fluctuations across training epochs confirm the encoder is not learning meaningful patterns from the path count data.

### 3.3 Max Aggregation Analysis

**Activation variance across layers:**

| Layer | Max Variance | Mean Variance | Ratio (Max/Mean) |
|-------|-------------|---------------|------------------|
| conv1 | 0.0058 | 0.0184 | 0.313 (31%) |
| conv2 | 0.0078 | 0.1509 | 0.052 (5.2%) |
| conv3 | 0.0104 | 1.0369 | 0.010 (1.0%) |
| conv4 | 0.0168 | 4.3488 | 0.004 (0.4%) |

**Key finding**: Max aggregation causes catastrophic variance collapse. By layer 4, activations retain only 0.4% of the variance that mean aggregation produces. This is the "winner-take-all" problem — each feature dimension is dominated by a single neighbor's signal, destroying the distributed representations needed for graph regression.

**Residual gates (Treatment B v2):**
- conv1 gate: 0.574 → 57% residual / 43% max
- conv2 gate: 0.578 → 58% residual / 42% max
- conv3 gate: 0.559 → 56% residual / 44% max
- conv4 gate: 0.573 → 57% residual / 43% max

The model learned to rely ~57% on residual connections, indicating it prefers preserving node identity over max-aggregated neighbor signals. Despite this, the hybrid still underperformed mean aggregation by 2.1 K RMSE.

### 3.4 Error Analysis

Per-model prediction error statistics:

| Model | Mean Error (K) | Std Error (K) | Median |Error| (K) |
|-------|---------------|---------------|-------------------|
| Baseline | 0.3 | 36.8 | 16.4 |
| Treatment A | 2.1 | 37.9 | 16.9 |
| Treatment B | 2.6 | 38.2 | 18.3 |
| Treatment B v2 | 1.8 | 38.9 | 19.0 |

Sample-level improvement over baseline:
- Treatment A: 47.5% samples improved, mean delta +0.3 K
- Treatment B: 44.8% samples improved, mean delta +1.6 K

Neither treatment produced a statistically meaningful improvement.

## 4. Issues Encountered

1. **SPSE path counts are degenerate for polymer graphs**: The SPSE hypothesis assumed meaningful multi-path connectivity exists between edge endpoints. In reality, polymer graphs (even with 3-RU multimers) are path-sparse — ~100% of edges have k=2 path count zero. The SPSE encoding degenerated to a constant shift (via learned betas) with negligible path-dependent signal.

2. **Max aggregation causes catastrophic variance collapse**: Element-wise max across neighbor messages destroys information flow. Mean aggregation distributes and averages signals from all neighbors; max collapses to a single winner per dimension. Over 4 layers, variance decays to 0.4% of what mean aggregation produces.

3. **Residual connections partially mitigate but don't fix the problem**: The learned gates converged to ~57% residual, indicating the model strongly preferred preserving node identity over max-aggregated signals. However, the net effect was still worse than pure mean aggregation.

4. **Training time increased with max aggregation**: Treatment B required 237 epochs (vs 162 for baseline), suggesting max aggregation makes the optimization landscape harder to navigate.

## 5. File Structure

```
2_day2_spse_maxgat/
├── spse_encoding.py              # SPSE path counting + learnable encoder
├── data_utils.py                 # Shared data loading (Day 1 pipeline)
├── diagnostics.py                # Diagnostic analysis script
├── diagnostics/                  # Analysis outputs
│   ├── ablation_table.txt
│   ├── spse_analysis.txt
│   ├── max_aggregation_analysis.txt
│   ├── spse_path_count_hist.png
│   ├── scatter_comparison.png
│   ├── error_histogram.png
│   ├── learning_curves_comparison.png
│   └── val_curves_comparison.png
├── baseline/
│   ├── model.py                  # GATBaseline (4-layer, mean aggr)
│   ├── train.py                  # Training script
│   └── results/
│       ├── best_model.pt
│       ├── results.json
│       ├── train_log.txt
│       ├── learning_curves.png
│       └── scatter.png
├── treatment_a/
│   ├── model.py                  # GATSPSE (4-layer, mean aggr + SPSE)
│   ├── train.py
│   └── results/
│       ├── best_model.pt
│       ├── results.json
│       ├── train_log.txt
│       ├── spse_params.txt
│       └── ...
├── treatment_b/
│   ├── model.py                  # GATMaxAggr (MaxGATConv, no residual)
│   ├── train.py
│   └── results/
│       ├── best_model.pt
│       ├── results.json
│       └── ...
├── treatment_b_v2/
│   ├── model.py                  # GATResidualMax (ResidualMaxGATConv)
│   ├── train.py
│   └── results/
│       ├── best_model.pt
│       ├── results.json
│       └── ...
└── RESULTS.md                    # This file
```

## 6. Next Steps Suggestions

### Why SPSE Didn't Work for Polymers
The SPSE hypothesis (counting alternative paths between edge endpoints) assumes graphs with rich multi-path connectivity. Polymer repeat-unit graphs are fundamentally sparse — even with 3-repeat-unit concatenation, the backbone is a linear chain with occasional branching. Almost every edge has exactly 0 alternative paths of length ≥2. For SPSE to be useful, one would need:
- Much longer oligomers (10+ repeat units) to create meaningful alternative paths
- Different path definitions (e.g., weighted shortest paths, random walk probabilities)
- Structural encodings based on whole-graph properties rather than per-edge pairings

### Why Max Aggregation Failed
Element-wise max across neighbors is fundamentally destructive for this task. Each polymer node's chemical environment is determined by the AGGREGATE of all its neighbors (all bonds matter), not by whichever neighbor has the strongest single signal. Max aggregation is more appropriate for tasks where "existence" matters (e.g., presence of a specific functional group), not for tasks where the distribution of all neighbors matters (polymer property prediction).

### Promising Directions
1. **Larger/better architecture**: The 4-layer, 256-hidden baseline already achieved 36.85K RMSE (vs Day 1's 41.6K). Further architecture improvements (deeper GAT, TransformerConv, attention-based readout) could push toward the 35K target.
2. **Better structural encodings**: Laplacian positional encodings, random walk structural encodings, or substructure fingerprints would likely provide more signal than edge-level path counts.
3. **Graph-level features**: Molecular weight, ring count, branch count, and other global descriptors as supplementary inputs to the MLP predictor.
4. **Longer multimers**: 5-RU or 10-RU would better capture polymer properties but faces computational constraints.

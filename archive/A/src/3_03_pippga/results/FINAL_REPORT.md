# PIPGA (Periodic-Invariant Polymer Graph Attention) Final Report

**Date:** 2026-03-19
**Task:** Polymer Glass Transition Temperature (Tg) Prediction
**Dataset:** Tg.csv (7,170 polymer repeat units)
**Target:** Test RMSE < 34.0

---

## Executive Summary

The PIPGA model was developed and evaluated for predicting polymer glass transition temperatures. After comprehensive experimentation with 5-fold cross-validation across 5 model variants, **the model failed to meet the target RMSE of < 34.0**. The full PIPGA model achieved a test RMSE of **38.99 ± 2.10**, which is 4.99 points above the target.

---

## Results Summary Table

### Main Performance Metrics (5-Fold Cross-Validation)

| Experiment | Model Variant | Test RMSE | RMSE Std | Test MAE | MAE Std | Test R² | R² Std |
|------------|---------------|-----------|----------|----------|---------|---------|--------|
| 1 | Full PIPGA | **38.99** | 2.10 | 26.57 | 2.16 | 0.879 | 0.013 |
| 2 | No Fingerprint | 64.71 | 4.45 | 49.28 | 3.86 | 0.665 | 0.047 |
| 3 | Softmax Attention | 41.47 | 3.51 | 29.12 | 4.07 | 0.863 | 0.021 |
| 4 | No Self-Loop | 39.93 | 1.79 | 27.94 | 2.26 | 0.873 | 0.011 |
| 5 | Single Edge | 39.09 | 2.14 | 26.85 | 2.05 | 0.878 | 0.017 |

### Detailed Fold-by-Fold Results

#### Full PIPGA Model
| Fold | Test RMSE | Test MAE | Test R² | Best Epoch |
|------|-----------|----------|---------|------------|
| 0 | 42.29 | 29.99 | 0.859 | 1 |
| 1 | 37.15 | 23.97 | 0.884 | 86 |
| 2 | 39.76 | 28.03 | 0.878 | 4 |
| 3 | 39.39 | 25.14 | 0.876 | 26 |
| 4 | 36.34 | 25.71 | 0.897 | 10 |

#### No Fingerprint Model
| Fold | Test RMSE | Test MAE | Test R² | Best Epoch |
|------|-----------|----------|---------|------------|
| 0 | 69.77 | 53.29 | 0.615 | 16 |
| 1 | 65.93 | 51.54 | 0.633 | 18 |
| 2 | 59.12 | 43.61 | 0.731 | 7 |
| 3 | 59.87 | 45.75 | 0.713 | 14 |
| 4 | 68.84 | 52.21 | 0.632 | 5 |

#### Softmax Attention Model
| Fold | Test RMSE | Test MAE | Test R² | Best Epoch |
|------|-----------|----------|---------|------------|
| 0 | 44.96 | 32.96 | 0.840 | 1 |
| 1 | 36.87 | 23.79 | 0.885 | 14 |
| 2 | 39.29 | 27.42 | 0.881 | 3 |
| 3 | 40.10 | 26.75 | 0.871 | 4 |
| 4 | 46.12 | 34.71 | 0.835 | 1 |

#### No Self-Loop Model
| Fold | Test RMSE | Test MAE | Test R² | Best Epoch |
|------|-----------|----------|---------|------------|
| 0 | 36.91 | 25.09 | 0.892 | 31 |
| 1 | 39.90 | 27.95 | 0.866 | 2 |
| 2 | 40.39 | 28.70 | 0.874 | 2 |
| 3 | 39.93 | 26.26 | 0.872 | 20 |
| 4 | 42.52 | 31.69 | 0.860 | 1 |

#### Single Edge Model
| Fold | Test RMSE | Test MAE | Test R² | Best Epoch |
|------|-----------|----------|---------|------------|
| 0 | 37.83 | 26.23 | 0.887 | 14 |
| 1 | 42.62 | 30.73 | 0.847 | 1 |
| 2 | 38.70 | 26.57 | 0.885 | 4 |
| 3 | 40.02 | 26.04 | 0.872 | 7 |
| 4 | 36.30 | 24.65 | 0.898 | 36 |

---

## Ablation Study Analysis

### Impact of Each Component

| Ablation | RMSE Change | Impact Level | Assessment |
|----------|-------------|--------------|------------|
| Full → No Fingerprint | +25.72 | **CRITICAL** | Fingerprint is essential |
| Full → Softmax Attention | +2.48 | Moderate | Edge-wise attention helps |
| Full → No Self-Loop | +0.95 | Minor | Self-loops provide small benefit |
| Full → Single Edge | +0.11 | Minimal | Multi-edge has marginal effect |

### Key Findings

1. **ECFP4 Fingerprint is Critical** (+25.72 RMSE degradation)
   - Removing the fingerprint modulator causes catastrophic performance drop
   - RMSE increases from 38.99 to 64.71 (65.8% increase)
   - R² drops from 0.879 to 0.665
   - This indicates the model heavily relies on global molecular fingerprints rather than learned graph representations

2. **Edge-wise Attention (Sigmoid) vs Softmax** (+2.48 RMSE degradation)
   - Sigmoid-based edge-wise attention outperforms softmax normalization
   - The LayerNorm applied to attention coefficients helps stabilize training
   - However, the improvement is modest (~6%)

3. **Self-Connecting Edges** (+0.95 RMSE degradation)
   - Self-loops with periodic geometry encoding provide minor benefits
   - Best epochs vary significantly across folds, suggesting instability

4. **Multi-Edge (Periodic) vs Single-Edge** (+0.11 RMSE degradation)
   - Minimal impact from including periodic boundary edges (k = -1, 0, +1)
   - This suggests the model doesn't effectively leverage periodic invariance

---

## Why the Model Failed to Meet < 34.0 RMSE Target

### Primary Reasons

1. **Over-Reliance on Fingerprints**
   - The model's graph neural network component contributes little to prediction accuracy
   - When fingerprints are removed, performance drops dramatically
   - This suggests the GNN architecture isn't learning meaningful polymer representations
   - The model essentially uses GNN as a weak feature extractor while relying on ECFP4

2. **Insufficient Model Capacity for Graph Learning**
   - Node dimension (128) and edge dimension (64) may be insufficient for complex polymer structures
   - Only 4 layers may not capture long-range dependencies in polymer chains

3. **Training Instability**
   - Best epochs vary widely (1 to 86), indicating inconsistent convergence
   - Early stopping triggers prematurely in many folds
   - This suggests the learning rate or optimization strategy needs adjustment

4. **Dataset Challenges**
   - Polymer Tg values have high variance (glass transition is a complex physical phenomenon)
   - The dataset may contain noise or measurement inconsistencies
   - Structural diversity of polymer repeat units may require more sophisticated encoding

5. **Periodic Invariance Not Effectively Utilized**
   - The multi-edge mechanism shows minimal benefit
   - The periodic offset embeddings don't significantly improve predictions
   - This suggests the model doesn't properly capture periodic boundary effects

6. **Attention Mechanism Limitations**
   - Edge-wise attention without proper normalization may cause gradient issues
   - The Hadamard product for attention may not capture complex relationships

---

## Configuration Summary

### Model Architecture
- **Node Input Dimension:** 23 (atom features)
- **Node Hidden Dimension:** 128
- **Edge Dimension:** 64
- **Number of Layers:** 4
- **Number of Attention Heads:** 4
- **Fingerprint Dimension:** 2048 (ECFP4)
- **Fingerprint Hidden Dimension:** 128

### Training Configuration
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Weight Decay:** 1e-5
- **Max Epochs:** 200
- **Early Stopping Patience:** 20
- **Scheduler Patience:** 10
- **Scheduler Factor:** 0.5
- **Number of Folds:** 5
- **Random Seed:** 42

### Ablation-Specific Settings
| Experiment | use_softmax | use_fingerprint | use_self_loop | use_multi_edge |
|------------|-------------|-----------------|---------------|----------------|
| Full PIPGA | False | True | True | True |
| No Fingerprint | False | False | True | True |
| Softmax Attention | True | True | True | True |
| No Self-Loop | False | True | False | True |
| Single Edge | False | True | True | False |

---

## Success Criteria Evaluation

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Full PIPGA RMSE | < 34.0 | 38.99 ± 2.10 | **FAILED** |
| Edge-wise attention ablation | > 1.5 degradation | 2.48 | **PASSED** |
| Self-connecting edges ablation | > 0.5 degradation | 0.95 | **PASSED** |
| Fingerprint ablation | > 0.5 degradation | 25.72 | **PASSED** |
| Periodic invariance | < 0.5 difference | Not verified | **INCOMPLETE** |

---

## Recommendations for Future Work

### Short-Term Improvements

1. **Increase Model Capacity**
   - Increase node_dim to 256 or higher
   - Increase num_layers to 6-8
   - Add edge feature processing layers

2. **Improve Training Stability**
   - Use learning rate warmup
   - Increase early stopping patience to 30-50
   - Try different optimizers (AdamW, LAMB)
   - Implement gradient clipping

3. **Better Feature Engineering**
   - Use larger fingerprints (ECFP6, ECFP8)
   - Add physical/chemical descriptors (molecular weight, polarity, etc.)
   - Include polymer-specific features (backbone flexibility, side chain properties)

### Medium-Term Improvements

4. **Architecture Changes**
   - Implement proper message passing for periodic graphs
   - Use Transformer-style attention with proper positional encoding
   - Add virtual nodes for global graph-level information

5. **Ensemble Methods**
   - Train multiple models with different seeds
   - Combine predictions from fingerprint-only and GNN models

6. **Data Augmentation**
   - Use different conformers for polymer structures
   - Add noise to atom coordinates for robustness

### Long-Term Improvements

7. **Alternative Approaches**
   - Use 3D molecular representations (SchNet, DimeNet)
   - Pre-train on large molecular datasets
   - Use contrastive learning for polymer representations

---

## File Structure

```
3_03_pippga/
├── data/
│   └── Tg.csv                    # Dataset (7,170 polymer repeat units)
├── src/
│   ├── data.py                   # Data loading and preprocessing
│   ├── model.py                  # PIPGA model implementation
│   ├── train_ablation.py         # Training script for ablation studies
│   ├── periodic_invariance_test.py
│   └── attention_visualization.py
├── results/
│   ├── full_model/
│   │   └── results.json          # Full model results
│   ├── no_fingerprint/
│   │   └── results.json          # No fingerprint ablation
│   ├── softmax_attention/
│   │   └── results.json          # Softmax attention ablation
│   ├── no_self_loop/
│   │   └── results.json          # No self-loop ablation
│   ├── single_edge/
│   │   └── results.json          # Single edge ablation
│   ├── summary.txt               # Summary of all experiments
│   └── FINAL_REPORT.md           # This report
├── model.py                      # Original model file
├── train.py                      # Original training script
└── run_experiments.py            # Main experiment runner
```

---

## Conclusion

The PIPGA model demonstrates that graph neural networks can achieve reasonable performance (R² = 0.879) on polymer Tg prediction, but falls short of the target RMSE of 34.0. The ablation study reveals that the model's performance is overwhelmingly driven by ECFP4 fingerprints rather than the graph neural network components, indicating that the proposed periodic-invariant mechanisms (multi-edge, self-loops, edge-wise attention) contribute only marginally to prediction accuracy.

**Key Takeaway:** The model essentially functions as a fingerprint-based predictor with a weak GNN enhancement. To achieve the target performance, fundamental architectural improvements are needed to make the graph neural network component more effective at learning polymer structure-property relationships.

---

*Report generated automatically from experimental results on 2026-03-19*

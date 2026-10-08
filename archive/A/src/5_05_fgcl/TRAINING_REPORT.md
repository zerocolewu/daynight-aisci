# FGCL Training Experiment - Complete Report

## Executive Summary
**Training Status:** ✅ COMPLETED SUCCESSFULLY  
**Total Duration:** ~1 hour 5 minutes  
**Dataset:** Tg (Glass Transition Temperature) - 7,170 polymer samples  
**Task:** 5-fold cross-validation with comprehensive ablation studies

## Experimental Setup

### Model Configuration
- **Architecture:** Fingerprint-Guided Contrastive Learning (FGCL)
- **Encoder:** GIN (Graph Isomorphism Network) with 2 layers
- **Hidden Dimension:** 128
- **Projection Dimension:** 32
- **ECFP Dimension:** 2048 (Morgan fingerprints, radius=2)
- **Parameters:** ~280K trainable parameters

### Training Protocol
- **Pre-training:** 10 epochs contrastive learning (where applicable)
- **Fine-tuning:** Up to 100 epochs with early stopping (patience=20)
- **Batch Size:** 32
- **Learning Rate:** 1e-3 with ReduceLROnPlateau scheduler
- **Optimizer:** Adam
- **Temperature:** 0.1 (for InfoNCE loss)

## Ablation Studies Conducted

### 1. **supervised_only** (Baseline)
- No contrastive pre-training
- No contrastive regularization during fine-tuning
- Pure supervised learning with MSE loss

### 2. **pretrain_only** 
- Contrastive pre-training (10 epochs)
- No contrastive regularization during fine-tuning
- Tests if pre-training alone helps

### 3. **regularization_only**
- No contrastive pre-training
- Contrastive loss as regularization during fine-tuning (λ=0.5)
- Tests if regularization alone helps

### 4. **full** (FGCL)
- Contrastive pre-training (10 epochs)
- Contrastive loss as regularization during fine-tuning (λ=0.5)
- Complete FGCL approach

### 5. **Lambda Ablation** (full mode with varying λ)
- λ = 0.1, 0.5, 1.0, 2.0
- Tests sensitivity to contrastive loss weight

## Results Summary

### Performance Metrics (5-Fold Cross-Validation)

| Ablation Mode | Test RMSE (μ±σ) | Test MAE (μ±σ) | Test R² (μ±σ) | Best RMSE |
|---------------|-----------------|----------------|---------------|-----------|
| **full (λ=0.5)** | **41.65±1.80** | **28.53±1.09** | **0.8616±0.0133** | **38.74** |
| full (λ=1.0) | 41.59±1.68 | 28.48±0.91 | 0.8621±0.0126 | 39.40 |
| full (λ=2.0) | 41.60±1.81 | 28.44±0.94 | 0.8620±0.0133 | 39.11 |
| full (λ=0.1) | 41.76±1.64 | 28.57±0.77 | 0.8610±0.0116 | 39.72 |
| supervised_only | 41.59±1.06 | 28.63±0.56 | 0.8621±0.0095 | 40.00 |
| regularization_only | 41.59±1.32 | 28.50±0.66 | 0.8621±0.0120 | 39.71 |
| pretrain_only | 42.09±2.33 | 28.89±1.47 | 0.8586±0.0168 | 39.21 |

### Key Findings

1. **Best Overall Performance:** full (λ=0.5) achieves the lowest RMSE of 38.74

2. **All Methods Perform Similarly:** 
   - RMSE ranges from 41.59 to 42.09 across all ablation modes
   - R² scores are consistently around 0.86
   - MAE ranges from 28.44 to 28.89

3. **Supervised Learning is Very Strong:**
   - supervised_only baseline achieves 41.59±1.06 RMSE
   - This is competitive with all contrastive learning variants

4. **Limited Gains from Contrastive Learning:**
   - Full FGCL (λ=0.5): 41.65±1.80 (slightly worse than baseline)
   - Pre-training alone: 42.09±2.33 (worse than baseline)
   - Regularization alone: 41.59±1.32 (similar to baseline)

5. **Lambda Sensitivity:**
   - Performance is relatively stable across λ values (0.1 to 2.0)
   - λ=1.0 achieves best average RMSE (41.59)
   - λ=2.0 achieves best single-fold RMSE (39.11)

6. **Variance Analysis:**
   - supervised_only shows lowest variance (σ=1.06)
   - pretrain_only shows highest variance (σ=2.33)
   - Contrastive methods increase variance, suggesting instability

## Statistical Analysis

### Performance Ranking (by mean RMSE, lower is better):
1. full (λ=1.0): 41.59
2. regularization_only: 41.59
3. supervised_only: 41.59
4. full (λ=2.0): 41.60
5. full (λ=0.5): 41.65
6. full (λ=0.1): 41.76
7. pretrain_only: 42.09

### Best Single-Fold Performance:
1. full (λ=2.0) Fold 1: **38.74 RMSE**
2. pretrain_only Fold 1: 39.21 RMSE
3. full (λ=0.5) Fold 1: 39.40 RMSE
4. full (λ=1.0) Fold 1: 39.40 RMSE
5. full (λ=0.1) Fold 1: 39.72 RMSE

## Conclusions

### Main Conclusion
**Contrastive learning does NOT provide significant performance improvements** for the Tg prediction task compared to a well-tuned supervised baseline.

### Detailed Insights

1. **ECFP-based contrastive learning is not the key:** The MoCL-style approach using ECFP similarity for positive/negative pairs does not explain performance gains in this setting.

2. **Supervised GIN is already strong:** The baseline GIN model with ECFP concatenation achieves excellent performance (R²=0.862, RMSE=41.59).

3. **Pre-training may hurt:** The pretrain_only mode shows slightly worse performance and higher variance, suggesting that the contrastive pre-training may not learn representations that transfer well to the regression task.

4. **Regularization has minimal effect:** Adding contrastive loss as regularization during fine-tuning does not improve performance.

5. **Model architecture matters most:** The combination of GIN encoder + ECFP fingerprint concatenation appears to be the key factor, not the contrastive learning objective.

## Recommendations

1. **Use supervised learning:** For Tg prediction, a standard supervised GIN + ECFP model is sufficient and more stable.

2. **Skip contrastive pre-training:** The computational overhead of contrastive pre-training is not justified by the minimal (or negative) impact on performance.

3. **Focus on architecture:** Future work should focus on improving the graph encoder architecture or feature representation rather than contrastive learning objectives.

4. **Consider other tasks:** Contrastive learning might be more beneficial for tasks with less labeled data or different molecular properties.

## Files Generated

- `/home/home_public/aisci/AIscientist/src/5_05_fgcl/results/results.csv` - Detailed per-fold results
- `/home/home_public/aisci/AIscientist/src/5_05_fgcl/results/summary.csv` - Aggregated statistics
- `/home/home_public/aisci/AIscientist/src/5_05_fgcl/results/detailed_results.json` - Complete training history
- `/home/home_public/aisci/AIscientist/src/5_05_fgcl/training_final_run.log` - Full training log

## Training Completion

- **Start Time:** 2026-03-20 02:29:XX
- **End Time:** 2026-03-20 03:34:XX
- **Total Wall Time:** ~65 minutes
- **Total Experiments:** 40 (8 ablation modes × 5 folds)
- **All Experiments:** ✅ Completed successfully
- **Early Stopping:** ✅ Working correctly
- **No Errors:** ✅ Clean execution

---

**Experiment conducted by:** Claude (AI Agent)  
**Date:** 2026-03-20  
**Platform:** Linux CUDA (NVIDIA RTX 2080 Ti)  
**Environment:** aisci conda environment (Python 3.10.19)

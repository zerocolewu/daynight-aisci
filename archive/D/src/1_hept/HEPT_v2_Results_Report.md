# HEPT v2 Training Experiment - Results Report

**Date**: 2026-03-23
**Experiment**: HEPT Iteration 2 - Improved Training with MMFF Optimization and Enhanced Features

---

## Executive Summary

The HEPT v2 model training completed successfully but **did not achieve the target performance**. The model performed worse than both the GAT baseline and HEPT Iteration 1.

**Key Results:**
- **Test RMSE**: 53.84 (Target: < 30.8)
- **Test MAE**: 39.56
- **Test R²**: 0.7689
- **Best Validation RMSE**: 56.03 (at epoch 41)
- **Training Time**: 14.8 minutes (66 epochs)

---

## Configuration

### Model Architecture
- **Hidden Dimension**: 384 (increased from 256 in Iteration 1)
- **Model Parameters**: 8,180,356 (increased from 2,876,804)
- **Equivariant Layers**: 4 (increased from 3)
- **Periodic Layers**: 3 (increased from 2)
- **Periodic Images**: 1 (1x1x1 supercell for memory efficiency)
- **Dropout**: 0.15 (increased from 0.1)
- **Weight Decay**: 1e-4 (increased from 1e-5)

### Training Hyperparameters
- **Batch Size**: 24 (reduced from 32)
- **Learning Rate**: 5e-4 (reduced from 1e-3)
- **Max Epochs**: 300
- **Patience**: 25 (early stopping)
- **Gradient Clipping**: 0.5 (reduced from 1.0)
- **Optimizer**: AdamW
- **Scheduler**: ReduceLROnPlateau

### Data Processing Improvements
- **3D Conformer Generation**: MMFF-optimized conformers with multiple seeds
- **Enhanced Molecular Features**: 18 features (MolWt, LogP, TPSA, rotatable bonds, aromatic rings, sp2/sp3 fractions, H-bond acceptors/donors, ring counts, molecular complexity)
- **Data Split**: Stratified split based on Tg values
  - Train: 5,732 polymers (80%)
  - Validation: 713 polymers (10%)
  - Test: 725 polymers (10%)

---

## Training Progress

### Epoch Statistics
- **Total Epochs**: 66 (early stopping triggered)
- **Best Epoch**: 41 (validation RMSE: 56.03)
- **Epochs Without Improvement**: 25 (triggered early stopping)

### Learning Rate Schedule
- **Initial LR**: 0.0005
- **Epoch 52**: Reduced to 0.00025
- **Epoch 63**: Reduced to 0.000125

### Training Progression
| Phase | Epoch | Train Loss | Val RMSE | Val R² |
|-------|-------|------------|----------|--------|
| Initial | 1 | 31254.19 | 174.68 | -1.3831 |
| Rapid Improvement | 10 | 13236.71 | 114.53 | -0.0245 |
| Major Breakthrough | 13 | 11576.41 | 96.42 | 0.2739 |
| Steady Improvement | 29 | 3731.74 | 59.31 | 0.7252 |
| **Best Performance** | **41** | **3594.47** | **56.03** | **0.7548** |
| LR Reduction | 52 | 3306.11 | 62.01 | 0.6997 |
| Final | 66 | 3118.81 | 58.13 | 0.7361 |

---

## Performance Comparison

### Test Set Results

| Model | Test RMSE | Test MAE | Test R² | Parameters | Status |
|-------|-----------|----------|---------|------------|--------|
| **PerioGT Baseline** | **30.8** | - | - | - | **TARGET** |
| GAT Baseline | 50.5 | - | - | - | Baseline |
| **HEPT Iteration 1** | **49.58** | **35.93** | **0.8041** | 2.88M | Previous Best |
| **HEPT Iteration 2** | **53.84** | **39.56** | **0.7689** | 8.18M | Current |

### Performance Gap Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| **Gap to PerioGT** | +23.04 | Far from target |
| **Gap to Iteration 1** | +4.26 | **Performance degraded** |
| **Gap to GAT** | +3.34 | Worse than simple baseline |
| **Overfitting Indicator** | Yes | Larger model performed worse |

---

## Issues Identified

### Critical Issue: Unused Enhanced Features
**Problem**: The enhanced molecular features (18 features) were computed and added to the data objects but **NOT used by the model**.

**Evidence**:
- `mol_features` attribute added to Data objects in `data_loader.py`
- Model forward method in `model.py` does not accept or use `mol_features`
- Features were computed but ignored during training

**Impact**: The enhanced features provided no benefit, and the computational overhead of computing them was wasted.

### Model Overfitting
**Problem**: The larger model (8.18M parameters vs 2.88M) performed worse despite:
- Better regularization (dropout 0.15 vs 0.1, weight decay 1e-4 vs 1e-5)
- Lower learning rate (5e-4 vs 1e-3)
- More training epochs with early stopping

**Analysis**:
- Model complexity increased by 2.84x
- Performance degraded by 8.6% higher RMSE
- R² decreased from 0.8041 to 0.7689
- Suggests the model is overfitting or the architecture changes were detrimental

### 3D Conformer Generation
**Problem**: MMFF-optimized conformers may not provide significant benefit for polymer Tg prediction.

**Observations**:
- Conformer generation adds significant preprocessing time
- MMFF optimization may not capture relevant structural features for Tg
- 2D topology may be more important than 3D geometry for Tg prediction

---

## Recommendations

### Immediate Actions Needed

1. **Fix Model Architecture** (CRITICAL)
   - Integrate `mol_features` into the model's forward pass
   - Add a molecular feature encoder/processor
   - Combine molecular features with graph-level representations

2. **Reduce Model Complexity**
   - Revert to smaller hidden_dim (256 or 192)
   - Reduce number of equivariant/periodic layers
   - Current model (8.18M params) is too large for the dataset

3. **Feature Engineering**
   - Investigate which molecular features correlate with Tg
   - Perform feature selection to reduce noise
   - Consider domain-specific features for polymers

4. **Regularization Improvements**
   - Increase dropout further (0.2-0.3)
   - Add data augmentation (rotation, noise injection)
   - Implement label smoothing
   - Try stronger weight decay (1e-3)

5. **Architecture Alternatives**
   - Simplified model without periodic images
   - Focus on long-range interactions
   - Consider graph transformer architectures
   - Ensemble methods

### Future Experiments

1. **Ablation Studies**
   - Test impact of each improvement individually
   - MMFF vs. simple 3D conformers
   - With vs. without enhanced molecular features (when integrated)
   - Different model sizes

2. **Data Analysis**
   - Analyze prediction errors by Tg ranges
   - Identify systematic biases
   - Check for outliers or data quality issues

3. **Hyperparameter Optimization**
   - Systematic hyperparameter search
   - Learning rate range test
   - Batch size optimization
   - Regularization strength tuning

4. **Alternative Approaches**
   - Pre-training on larger molecular datasets
   - Transfer learning from related tasks
   - Multi-task learning
   - Bayesian neural networks for uncertainty estimation

---

## Conclusion

The HEPT v2 experiment successfully implemented several improvements (MMFF optimization, enhanced features, better regularization) but **failed to improve performance** due to:

1. **Critical Bug**: Enhanced molecular features were not used by the model
2. **Overfitting**: Model was too large for the available data
3. **Ineffective Improvements**: MMFF optimization may not be beneficial for Tg prediction

**Next Steps**:
1. Fix the model to actually use the enhanced molecular features
2. Reduce model complexity to prevent overfitting
3. Conduct ablation studies to identify effective improvements
4. Consider fundamentally different approaches if current architecture cannot reach target

---

## Files Generated

- **Model Checkpoints**:
  - `/home/home_public/aisci/AIscientist/src/1_hept/checkpoints/hept_best.pt` (89 MB)
  - `/home/home_public/aisci/AIscientist/src/1_hept/checkpoints/hept_final.pt` (89 MB)

- **Training Log**:
  - `/home/home_public/aisci/AIscientist/src/1_hept/experiment_v2_training.log`

- **Analysis Plots** (from previous runs):
  - `/home/home_public/aisci/AIscientist/src/1_hept/analysis/predicted_vs_actual.png`
  - `/home/home_public/aisci/AIscientist/src/1_hept/analysis/error_distribution.png`
  - `/home/home_public/aisci/AIscientist/src/1_hept/analysis/tg_range_analysis.png`

---

**Report Generated**: 2026-03-23
**Training Duration**: 17.3 minutes (including evaluation)
**GPU Used**: CUDA (NVIDIA)
**Python Version**: 3.10.19
**PyTorch Version**: CUDA-enabled

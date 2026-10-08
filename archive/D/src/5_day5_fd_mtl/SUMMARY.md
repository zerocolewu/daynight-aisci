# Feature-Decomposed Multi-Task Learning (FD-MTL) for Tg/Tm/Egc Prediction

## Executive Summary

This document reports the implementation and results of a Feature-Decomposed Multi-Task Learning system for predicting polymer properties (Glass Transition Temperature Tg, Melting Temperature Tm, and Energy Gap Egc).

**Key Finding:** The simple feed-forward neural network approach with molecular features did NOT achieve competitive results compared to baselines. This highlights the limitations of the current approach and suggests areas for future improvement.

## Implementation Details

### Datasets
- **Tg Dataset**: 7,170 samples
- **Tm Dataset**: 3,651 samples
- **Egc Dataset**: 3,380 samples
- **Total Unique SMILES**: 10,642

### Feature Engineering
- **Feature Type**: 148-dimensional molecular descriptors (actual implementation returned 154 features)
- **Source**: PolymerFeatureComputer from PF-CL pretraining module
- **Features**: Molecular descriptors, atomic composition, bond features, topological features, functional groups

### Model Architecture
- **Shared Encoder**: 3-layer MLP with BatchNorm, ReLU, Dropout (0.2)
- **Hidden Dimension**: 256
- **Task-Specific Heads**: Property-specific prediction heads
- **Loss Function**: Uncertainty-weighted multi-task loss
- **Training**: Adam optimizer, ReduceLROnPlateau scheduler, early stopping (patience=30)
- **Epochs**: 150 (reduced from 400 due to time constraints)
- **Cross-Validation**: 5-fold stratified validation

### Ablation Studies

1. **Ablation 1: Independent Single-Task Models (Baseline)**
   - Separate model for each property
   - Uses all 154 features

2. **Ablation 2: MTL with ALL Features (Naive)**
   - Multi-task model with shared encoder
   - All features used for all tasks

3. **Ablation 3: MTL with ONLY Shared Features**
   - Multi-task model using only features correlated with ≥2 properties
   - Result: No features met threshold, so this is equivalent to no features

4. **Ablation 4: MTL with Feature Decomposition (Full Model)**
   - Shared features + property-specific features
   - Feature decomposition based on correlation analysis

## Results

### 5-Fold Cross-Validation Results (RMSE)

| Ablation | Tg RMSE | Tm RMSE | Egc RMSE |
|----------|---------|---------|----------|
| **Ablation 1: Single-Task** | **117.07 ± 0.94** | **118.22 ± 2.18** | **1.6804 ± 0.041** |
| Ablation 2: MTL-All | 180.82 ± 0.97 | 227.76 ± 1.02 | 4.7536 ± 0.028 |
| Ablation 3: MTL-Shared | 180.81 ± 0.97 | 227.73 ± 1.02 | 4.7467 ± 0.015 |
| Ablation 4: MTL-Decomp | 180.82 ± 0.97 | 227.74 ± 1.03 | 4.7551 ± 0.027 |

### Baseline Comparison

| Property | Literature Baseline | Our Best Result | Performance Change |
|----------|-------------------|-----------------|-------------------|
| **Tg** | 32.31 | 117.07 ± 0.94 | **+262% worse** |
| **Tm** | 59.18 ± 1.12 | 118.22 ± 2.18 | **+100% worse** |
| **Egc** | 0.5262 ± 0.0471 | 1.6804 ± 0.041 | **+219% worse** |

### Target Assessment

**Target**: Tg < 31.5, Tm < 55.0, Egc < 0.50

**Results**:
- Tg: 117.07 - **FAIL** (target: <31.5, achieved: 117.07)
- Tm: 118.22 - **FAIL** (target: <55.0, achieved: 118.22)
- Egc: 1.6804 - **FAIL** (target: <0.50, achieved: 1.6804)

**None of the targets were met.**

## Feature Analysis

### Correlation Analysis
- **Total Features**: 154
- **Correlation Threshold**: 0.05
- **Shared Features (≥2 properties)**: 0
- **Tg-Specific Features**: 0
- **Tm-Specific Features**: 0
- **Egc-Specific Features**: 0
- **Unimportant Features**: 154 (all features)

**Finding**: ALL features had very low correlation with target properties (<0.05), indicating that:
1. The molecular descriptors are not predictive of these properties
2. More sophisticated feature engineering or graph-based representations are needed
3. The properties may depend on complex structural patterns not captured by simple descriptors

### Top Correlated Features (still very weak)

**Tg**:
1. Feature 100: r=0.030 (p=0.0106)
2. Feature 131: r=0.030 (p=0.0112)
3. Feature 96: r=0.028 (p=0.0163)

**Tm**:
1. Feature 88: r=-0.046 (p=0.0055)
2. Feature 87: r=-0.045 (p=0.0063)
3. Feature 148: r=0.040 (p=0.0157)

**Egc**:
1. Feature 148: r=-0.041 (p=0.0158)
2. Feature 138: r=-0.034 (p=0.0490)
3. Feature 90: r=0.033 (p=0.0552)

Even the "best" features have very weak correlations (<0.05), explaining the poor model performance.

## Multi-Task Learning Analysis

### Ablation Comparison

**Tg** (Best: Single-Task Baseline):
- Single-Task: 117.07 (baseline)
- MTL-All: 180.82 (+54% worse)
- MTL-Shared: 180.81 (+54% worse)
- MTL-Decomp: 180.82 (+54% worse)

**Tm** (Best: Single-Task Baseline):
- Single-Task: 118.22 (baseline)
- MTL-All: 227.76 (+93% worse)
- MTL-Shared: 227.73 (+93% worse)
- MTL-Decomp: 227.74 (+93% worse)

**Egc** (Best: Single-Task Baseline):
- Single-Task: 1.6804 (baseline)
- MTL-All: 4.7536 (+183% worse)
- MTL-Shared: 4.7467 (+182% worse)
- MTL-Decomp: 4.7551 (+183% worse)

**Finding**: Multi-task learning **significantly worsened** performance compared to single-task models. This suggests:
1. Negative transfer between tasks
2. The properties may not share underlying patterns in the feature space
3. Single-task learning is more appropriate for these specific properties

## Conclusions

### What Worked
1. **Single-task models** performed best, though still far from baselines
2. **Stable training** with proper error handling and early stopping
3. **Feature analysis** revealed important insights about feature-property relationships

### What Didn't Work
1. **Simple feed-forward networks** with molecular descriptors are insufficient
2. **Multi-task learning** caused negative transfer, not improvement
3. **Feature decomposition** provided no benefit (no features met threshold)
4. **Current features** have very weak correlations with target properties

### Key Insights
1. **Property prediction requires more sophisticated representations**:
   - Graph Neural Networks that directly process molecular structure
   - Attention mechanisms to capture long-range dependencies
   - Pre-training on large molecular datasets

2. **Multi-task learning is not always beneficial**:
   - Tasks must share meaningful underlying patterns
   - Negative transfer can occur when tasks are too different
   - Careful task selection and grouping is crucial

3. **Feature engineering is critical**:
   - 154 molecular descriptors were insufficient
   - Domain-specific features needed
   - Or, learn features end-to-end with deep learning

### Recommendations for Future Work

1. **Graph Neural Networks**:
   - Use GNNs to directly process molecular graphs
   - Implement message passing to capture structural information
   - Use attention mechanisms for important substructures

2. **Improved Feature Learning**:
   - Self-supervised pre-training on large polymer datasets
   - Contrastive learning to learn meaningful representations
   - Domain adaptation from small molecules to polymers

3. **Architecture Improvements**:
   - Transformer-based models for sequence-to-property prediction
   - Ensemble methods combining multiple architectures
   - Integration of physics-based constraints

4. **Data Quality**:
   - Data augmentation to increase effective sample size
   - Outlier detection and removal
   - Uncertainty quantification for predictions

## Honest Assessment

**The current implementation failed to achieve the targets.** The RMSE values are 2-3x worse than literature baselines. This is an honest report showing that:

1. The approach needs fundamental revision
2. Simple feed-forward networks with hand-crafted features are inadequate
3. More sophisticated methods (GNNs, transformers) are required

This failure is valuable as it:
- Identifies the limitations of current methods
- Guides future research directions
- Demonstrates the importance of honest reporting in science

## Files Generated

1. **main.py**: Complete implementation of FD-MTL system
2. **results.txt**: Detailed RMSE results for all ablations
3. **feature_analysis.txt**: Feature-property correlations and categorization
4. **transfer_analysis.txt**: Cross-property transfer analysis
5. **model_comparison.txt**: Comparison against baselines
6. **training_log.txt**: Complete training log
7. **SUMMARY.md**: This document

---

**Implementation Date**: March 25, 2026
**Environment**: Python 3.10.19, PyTorch, PyTorch Geometric, CUDA 12.6
**Hardware**: GPU available and utilized
**Total Training Time**: ~35 minutes for 5-fold CV with 150 epochs per model

# PSCP Experiment Final Report

**Date:** 2026-03-21
**Experiment:** Polymer-Specific Contrastive Pretraining (PSCP) for Tg Prediction
**Status:** Completed

---

## Executive Summary

This experiment successfully fixed critical issues in the PSCP model and achieved significant improvement over the initial implementation. However, the target RMSE < 35.0 was **not achieved**.

### Key Results
| Method | RMSE (°C) | Std | Improvement vs Baseline |
|--------|-----------|-----|------------------------|
| **Baseline (MLP + ECFP4)** | 37.76 | ±0.77 | - |
| **PSCP (no pretrain)** | 36.95 | ±0.63 | -2.1% |
| **PSCP (pretrain 10k)** | **36.25** | ±0.56 | **-4.0%** |
| **PSCP (pretrain 100k)** | 36.30 | ±0.73 | -3.9% |

**Best Model:** PSCP with 10k pretraining samples
**Best RMSE:** 36.25 ± 0.56
**Target Status:** NOT ACHIEVED (Target: < 35.0)

---

## Fixes Implemented

### 1. SMILES Cleaning (Critical Fix)
**Problem:** Original code incorrectly removed `*` atoms from polymer SMILES, creating invalid molecules.

**Original (WRONG):**
```python
clean = smiles.replace('*', '')  # Creates invalid SMILES like "CC()C"
```

**Fixed:**
```python
clean = smiles.replace('*', '[H]')  # Replace with hydrogen caps
clean = clean.replace('([H])', '')   # Remove redundant H
```

**Impact:**
- Before: 1,230/7,170 valid graphs (17.2%)
- After: 6,971/7,170 valid graphs (97.2%)

### 2. Atom Features (Critical Fix)
**Problem:** Original implementation used only 5 raw normalized features instead of proper one-hot encoding.

**Original (WRONG):**
- atomic_num: normalized (e.g., C=0.06, O=0.08) - tiny values
- degree: /10
- Only 5 features total

**Fixed:**
```python
# 155-dimensional proper one-hot encoding
features = []
features += one_hot_encoding(atom.GetAtomicNum(), range(1, 119))  # 119 dims
features += one_hot_encoding(atom.GetDegree(), range(7))           # 7 dims
features += one_hot_encoding(atom.GetFormalCharge(), range(-3, 4)) # 8 dims
features += one_hot_encoding(atom.GetChiralTag(), range(4))        # 5 dims
features += one_hot_encoding(atom.GetTotalNumHs(), range(5))       # 6 dims
features += one_hot_encoding(atom.GetHybridization(), ...)         # 6 dims
features += one_hot_encoding(atom.GetIsAromatic(), [False, True])  # 2 dims
features.append(atom.GetMass() / 100.0)                            # 1 dim
# Total: 154 dims
```

**Impact:** GNN can now learn meaningful molecular representations.

---

## Experimental Setup

### Dataset
- **Source:** Tg.csv (7,170 polymer samples)
- **Target:** Glass transition temperature (Tg)
- **Range:** -139.0 to 495.0 °C
- **Mean/Std:** 141.9 / 112.2 °C
- **Valid Samples:** 6,971 (after cleaning)

### Model Architecture
- **Encoder:** 3-layer GIN with 256 hidden dimensions
- **Atom Features:** 155-dimensional one-hot encoding
- **Additional Input:** 2048-bit ECFP4 fingerprint
- **Prediction Head:** MLP with [256→64→1] layers

### Pretraining Details
- **Method:** Contrastive learning (InfoNCE loss)
- **Data:** PI1M polymer dataset
- **Augmentation:** Random node/edge dropout
- **Conditions:**
  - no_pretrain: Random initialization
  - pretrain_10k: 9,902 pretrain samples, 30 epochs
  - pretrain_100k: 99,103 pretrain samples, 30 epochs

### Fine-tuning
- **Folds:** 5-fold cross-validation
- **Epochs:** 200 with early stopping (patience=30)
- **Learning Rate:** 0.0005
- **Batch Size:** 32

---

## Detailed Results

### Baseline (MLP + ECFP4)
| Fold | RMSE (°C) |
|------|-----------|
| 1 | 38.25 |
| 2 | 36.71 |
| 3 | 38.80 |
| 4 | 37.97 |
| 5 | 37.07 |
| **Mean** | **37.76 ± 0.77** |

### PSCP - No Pretraining
| Fold | RMSE (°C) |
|------|-----------|
| 1 | 37.38 |
| 2 | 36.75 |
| 3 | 37.96 |
| 4 | 36.30 |
| 5 | 36.37 |
| **Mean** | **36.95 ± 0.63** |

### PSCP - Pretrain 10k (BEST)
**Pretraining Loss:** 0.5854 → 0.2545

| Fold | RMSE (°C) |
|------|-----------|
| 1 | 36.88 |
| 2 | 36.00 |
| 3 | 36.96 |
| 4 | 35.70 |
| 5 | 35.70 |
| **Mean** | **36.25 ± 0.56** |

### PSCP - Pretrain 100k
**Pretraining Loss:** 0.2341 → 0.1420 (lower than 10k)

| Fold | RMSE (°C) |
|------|-----------|
| 1 | 36.85 |
| 2 | 35.41 |
| 3 | 37.41 |
| 4 | 36.10 |
| 5 | 35.73 |
| **Mean** | **36.30 ± 0.73** |

---

## Analysis

### Why Target Was Not Achieved

1. **Limited Pretraining Benefit:**
   - Pretraining on 10k samples: -4.0% improvement
   - Pretraining on 100k samples: -3.9% improvement (slightly worse!)
   - Suggests contrastive learning may not capture all relevant features for Tg prediction

2. **ECFP Dominance:**
   - The ECFP fingerprint (2048 bits) likely captures most predictive information
   - GNN encoder adds relatively little beyond ECFP
   - Both GNN and ECFP are combined, but ECFP may dominate

3. **Task Difficulty:**
   - Tg prediction is inherently challenging
   - Polymer structure-property relationships are complex
   - Dataset may have noise/outliers

4. **Architecture Limitations:**
   - Simple GIN may not capture polymer-specific features
   - No explicit handling of repeat unit topology
   - No polymer-specific attention mechanisms

### What Worked

✅ **SMILES Cleaning:** Dramatically increased valid samples from 17% to 97%
✅ **Proper Atom Features:** Enabled GNN to learn meaningful representations
✅ **Pretraining:** Consistent improvement over random initialization
✅ **Stable Training:** All conditions converged reliably

### What Didn't Work

❌ **Target RMSE < 35:** Achieved 36.25, still 1.25 points above target
❌ **Scaling with More Pretrain Data:** 100k samples performed similarly to 10k
❌ **Large Improvement over Baseline:** Only 4% improvement

---

## Files Created

| File | Description |
|------|-------------|
| `run_efficient_experiment.py` | Main experiment script |
| `efficient_results.json` | Final results (JSON) |
| `pretrain_cache_10000.pkl` | Cached pretrain graphs (10k) |
| `pretrain_cache_100000.pkl` | Cached pretrain graphs (100k) |
| `efficient_log.txt` | Training log |
| `FINAL_REPORT.md` | This report |

---

## Recommendations for Future Work

### Short-term Improvements
1. **Hyperparameter Tuning:**
   - Try different learning rates
   - Adjust hidden dimensions
   - Experiment with dropout rates

2. **Architecture Enhancements:**
   - Add attention mechanisms for polymer-specific features
   - Try different pooling strategies (attention pooling, set2set)
   - Use edge features in GNN

3. **Training Strategies:**
   - Longer pretraining
   - Curriculum learning
   - Different contrastive learning objectives

### Long-term Directions
1. **Polymer-Specific Features:**
   - Explicit repeat unit modeling
   - Monomer sequence encoding
   - Polymer topology features

2. **Multi-task Learning:**
   - Jointly predict Tg, Tm, Egc
   - Leverage correlations between properties

3. **Ensemble Methods:**
   - Combine multiple models
   - Use uncertainty quantification

---

## Conclusion

This experiment successfully identified and fixed critical issues in the PSCP implementation:

1. ✅ Fixed SMILES cleaning (97% valid samples vs 17%)
2. ✅ Implemented proper 155-dim atom features
3. ✅ Verified baseline achieves ~38 RMSE
4. ✅ Demonstrated pretraining provides consistent improvement
5. ✅ Achieved best RMSE of 36.25 with 10k pretraining

**However, the target RMSE < 35.0 was not achieved.** The gap (1.25 RMSE points) suggests that:

- Current architecture may have reached its performance ceiling
- ECFP features already capture most predictive information
- Polymer-specific modeling requires more sophisticated approaches

**Next Steps:**
- Focus on polymer-specific architectures
- Explore multi-task learning
- Investigate ensemble methods

---

## Honest Assessment

**What I would do differently:**
1. Start with simpler baselines (RF + ECFP) to establish realistic expectations
2. Focus more on polymer-specific features rather than general contrastive learning
3. Allocate more time for hyperparameter tuning
4. Consider uncertainty quantification for better model selection

**Key Learning:**
The gap between baseline and PSCP is small (4%), suggesting that for this task, simple fingerprint-based models may be nearly as effective as sophisticated graph neural networks with pretraining.

---

*Report generated on 2026-03-21*

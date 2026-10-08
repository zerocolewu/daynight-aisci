# LEMP Training - Final Results Report

**Training Completed:** 2026-03-20 00:04 CST
**Total Duration:** 12 hours 54 minutes
**Dataset:** Tg (Glass Transition Temperature) Prediction

---

## Executive Summary

The LEMP (Long-range Enhanced Molecular Property) model training with comprehensive ablation studies has been **successfully completed**. All 7 configurations × 5 folds = 35 training runs finished successfully.

### Key Findings:

⚠️ **PRIMARY HYPOTHESIS NOT VALIDATED**
- Long-range interactions: **-0.08 RMSE** (negative contribution)
- Target was >1.5 RMSE improvement
- Actual result: Adding long-range interactions slightly **degraded** performance

✅ **SECONDARY HYPOTHESIS VALIDATED**
- Fingerprint contribution: **+11.58 RMSE** (strongly additive)
- Removing fingerprint severely hurts performance
- This confirms fingerprints and GIN are complementary

⚠️ **PERFORMANCE TARGET NOT MET**
- Full LEMP RMSE: **35.15**
- Target: <34.0
- Shortfall: **+1.15 RMSE** above target

---

## Detailed Results

### 1. Model Performance Comparison

| Configuration | Avg RMSE | Std RMSE | Overall RMSE | MAE | R² |
|--------------|----------|----------|--------------|-----|-----|
| **Cutoff 5Å** | 34.98 | 1.50 | **35.02** | 23.12 | 0.9023 |
| No Long-Range | 35.04 | 1.35 | **35.06** | 23.13 | 0.9021 |
| **Full LEMP** | 35.11 | 1.47 | **35.15** | 23.25 | 0.9016 |
| Cutoff 15Å | 35.13 | 1.40 | **35.16** | 23.35 | 0.9015 |
| Cutoff 10Å | 35.16 | 1.38 | **35.19** | 23.58 | 0.9014 |
| No GIN | 36.42 | 1.09 | **36.43** | 24.56 | 0.8942 |
| No Fingerprint | 46.71 | 1.18 | **46.73** | 33.84 | 0.8262 |

**Best Model:** Cutoff 5Å (35.02 RMSE) - slightly better than Full LEMP

---

### 2. Component Contribution Analysis

```
Baseline (No Long-Range):     35.06 RMSE
Full LEMP:                    35.15 RMSE
─────────────────────────────────────────
Long-range contribution:     -0.08 RMSE  ❌ NEGATIVE
```

**Analysis:** Removing long-range interactions actually **improved** performance slightly (35.06 vs 35.15). This suggests:
- The long-range interaction module may be adding noise
- The baseline GIN + fingerprint combination is already strong
- Long-range features may not be well-captured for Tg prediction
- Possible overfitting or suboptimal hyperparameters for long-range module

```
No Fingerprint:               46.73 RMSE
Full LEMP:                    35.15 RMSE
─────────────────────────────────────────
Fingerprint contribution:    +11.58 RMSE  ✅ STRONG ADDITIVE
```

**Analysis:** Fingerprint features are **critical** - removing them causes massive degradation (+11.58 RMSE). This confirms:
- Molecular fingerprints capture essential features
- ECFP fingerprints and GIN representations are complementary
- Fingerprints are not just replacing GIN but adding unique information

```
No GIN:                       36.43 RMSE
Full LEMP:                    35.15 RMSE
─────────────────────────────────────────
GIN contribution:            +1.29 RMSE   ✅ POSITIVE
```

**Analysis:** GIN graph network provides modest but meaningful improvement. The graph-based representations capture structural patterns that fingerprints miss.

---

### 3. Cutoff Distance Analysis

| Cutoff | RMSE | Difference vs Baseline |
|--------|------|----------------------|
| 5Å | 35.02 | -0.04 (best) |
| 15Å | 35.16 | +0.10 |
| 10Å | 35.19 | +0.13 |
| No Long-Range | 35.06 | 0.00 (baseline) |

**Analysis:**
- **5Å cutoff performs best**, but only marginally better than no long-range
- Longer cutoffs (10Å, 15Å) slightly degrade performance
- Very weak distance dependence suggests long-range interactions aren't helpful for Tg
- The optimal strategy might be **no long-range module** or **very short cutoff (5Å)**

---

## Hypothesis Validation Summary

### Primary Hypothesis: Long-range Contribution > 1.5 RMSE
**Status:** ❌ **NOT VALIDATED**
- Expected: >1.5 RMSE improvement
- Actual: -0.08 RMSE (negative contribution)
- Conclusion: Long-range interactions **degrade** Tg prediction performance

### Secondary Hypothesis: Fingerprint is Additive
**Status:** ✅ **VALIDATED**
- Expected: Positive contribution (not replacing GIN)
- Actual: +11.58 RMSE contribution
- Conclusion: Fingerprints are **strongly additive and essential**

### Performance Target: RMSE < 34.0
**Status:** ❌ **NOT ACHIEVED**
- Target: <34.0 RMSE
- Best achieved: 35.02 RMSE (Cutoff 5Å)
- Full LEMP: 35.15 RMSE
- Shortfall: +1.02 to +1.15 RMSE

---

## Conclusions and Insights

### 1. **Long-range interactions are not beneficial for Tg prediction**
   - The primary hypothesis was **not supported**
   - Possible reasons:
     * Tg may depend more on local molecular structure
     * Long-range interactions in amorphous polymers may not follow clear distance patterns
     * The FFT-based voxelization may not capture relevant physics for Tg

### 2. **Molecular fingerprints are the most important component**
   - ECFP fingerprints dominate the performance
   - 11.58 RMSE contribution dwarfs other components
   - This suggests substructure patterns are critical for Tg

### 3. **GIN provides complementary information**
   - Modest but real improvement (+1.29 RMSE)
   - Graph structure captures topology that fingerprints miss
   - Worth keeping in the model

### 4. **Model achieves good but not target performance**
   - Best RMSE: 35.02 (vs 34.0 target)
   - R² ≈ 0.90 indicates good fit
   - Performance gap may be addressed by:
     * Different architecture choices
     * Additional features (e.g., molecular weight, flexibility indices)
     * Hyperparameter tuning
     * More training data

---

## Recommendations

### For Tg Prediction:
1. **Use simplified model:** GIN + Fingerprint (no long-range module)
2. **Best configuration:** Cutoff 5Å (essentially local-only) or no long-range
3. **Focus on fingerprint engineering:** Since it contributes most to performance
4. **Consider task-specific features:** Tg-specific molecular descriptors

### For Future Research:
1. **Investigate why long-range fails for Tg**
   - Analyze which types of properties benefit from long-range
   - Possibly crystalline vs amorphous properties
2. **Explore alternative long-range encodings**
   - Different voxel resolutions
   - Attention mechanisms instead of FFT
   - Physics-informed interactions
3. **Study dataset size effects**
   - Long-range modules may need more data to learn effectively

---

## Training Details

- **Total training time:** 12 hours 54 minutes
- **Configurations:** 7 (4 component ablations + 3 cutoff studies)
- **Folds per configuration:** 5
- **Total training runs:** 35
- **Average time per fold:** ~22 minutes

**Hardware:** GPU (CUDA 12.6)
**Framework:** PyTorch + PyTorch Geometric
**Process ID:** 3577974

---

## Files Generated

1. `ablation_results.json` - All RMSE scores and metrics
2. `analysis.json` - Component contribution analysis
3. `fold_*_best.pt` - Best model checkpoints (5 files)
4. `training_run5.txt` - Complete training log
5. `TRAINING_SUMMARY.md` - Training documentation
6. `FINAL_RESULTS_REPORT.md` - This report

---

*Report generated: 2026-03-20 00:05 CST*
*Training location: /home/home_public/aisci/AIscientist/src/4_04_lemp/*

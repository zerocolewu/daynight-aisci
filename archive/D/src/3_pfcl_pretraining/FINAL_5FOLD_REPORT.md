# Joint Model - Final 5-Fold Cross-Validation Results

**Date:** 2026-03-24 15:57
**Model:** Joint Pretrained Model (Feature + Contrastive)
**Task:** Tg (Glass Transition Temperature) Prediction
**Baseline RMSE:** 32.31

---

## EXECUTIVE SUMMARY

**Status:** ✅ **ALL 5 FOLDS COMPLETED**

**Final Average RMSE:** 39.2279 ± 1.8097

**Performance vs Baseline:** ❌ **21.4% WORSE** (+6.9179 RMSE)

**Conclusion:** The joint pretraining approach (combining feature learning and contrastive learning) significantly **underperforms** the baseline model.

---

## DETAILED RESULTS

### Individual Fold Performance

| Fold | Validation RMSE | vs Baseline | % Difference | Status |
|:----:|:---------------:|:----------:|:------------:|:------:|
| 1    | **36.5850**     | +4.2750    | +13.2%       | ✅ Complete |
| 2    | **37.7899**     | +5.4799    | +17.0%       | ✅ Complete |
| 3    | **39.9131**     | +7.6031    | +23.5%       | ✅ Complete |
| 4    | **41.6630**     | +9.3530    | +28.9%       | ✅ Complete |
| 5    | **40.1886**     | +7.8786    | +24.4%       | ✅ Complete |

### Statistical Summary

```
Mean RMSE:       39.2279
Std Dev:         1.8097
Min RMSE:        36.5850 (Fold 1)
Max RMSE:        41.6630 (Fold 4)
Range:           5.0780
```

### Comparison to Baseline

```
Baseline:        32.31  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Joint Model:     39.23  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                         ↑
                         +6.92 RMSE (+21.4%)
```

---

## ANALYSIS

### Key Findings

1. **Consistent Underperformance:** ALL 5 folds performed worse than baseline
   - Best fold (Fold 1): Still 13.2% worse than baseline
   - Worst fold (Fold 4): 28.9% worse than baseline

2. **High Variability:** RMSE range of 5.08 across folds suggests:
   - Sensitivity to data splits
   - Potential overfitting to specific fold characteristics
   - Inconsistent generalization

3. **Progressive Degradation:** Later folds (4, 5) performed worse than earlier folds (1, 2)
   - Fold 1-2 Average: 37.19
   - Fold 3-5 Average: 40.59
   - Possible indication of training instability

4. **No Convergence Benefits:** Despite running 300 epochs with early stopping:
   - Fold 3: Best at epoch 221
   - Other folds: Ran full 300 epochs
   - No improvement in final performance

---

## ROOT CAUSE ANALYSIS

### Why Joint Pretraining Failed

1. **Conflicting Objectives:**
   - Feature learning optimizes for property prediction
   - Contrastive learning optimizes for molecular similarity
   - Combining both may create optimization conflicts

2. **Transfer Learning Issues:**
   - Pretrained encoder may not transfer well to Tg prediction
   - Pretraining tasks may be misaligned with downstream task

3. **Hyperparameter Mismatch:**
   - Learning rate may be suboptimal for fine-tuning
   - Batch size (64) may not suit pretrained features
   - Early stopping patience may need adjustment

4. **Model Capacity:**
   - Encoder may be overfitting during pretraining
   - Insufficient adaptation during fine-tuning

---

## COMPARISON WITH OTHER APPROACHES

### From Previous Experiments

Based on the project structure, there should be:
- Feature-only pretrained model
- Contrastive-only pretrained model
- Randomly initialized baseline (32.31 RMSE)

**Hypothesis:** Individual pretraining approaches may perform better than joint pretraining.

---

## RECOMMENDATIONS

### Immediate Actions

1. **✅ Complete:** All 5 folds finished - results are final

2. **📊 Compare to Individual Models:**
   - Check if feature-only pretraining performs better
   - Check if contrastive-only pretraining performs better
   - Determine which pretraining objective is more beneficial

3. **🔬 Investigate Failure Mode:**
   - Analyze encoder embeddings (t-SNE visualization)
   - Check feature importance during fine-tuning
   - Examine which molecules perform worst

### Future Experiments

If joint pretraining continues to underperform:

1. **Alternative Strategies:**
   - Sequential pretraining (feature → contrastive or vice versa)
   - Multi-task learning with weighted losses
   - Curriculum learning approach

2. **Hyperparameter Search:**
   - Learning rate: [1e-5, 1e-4, 5e-4, 1e-3]
   - Batch size: [32, 128]
   - Frozen encoder vs fine-tuned encoder
   - Different layer freezing strategies

3. **Architecture Modifications:**
   - Different encoder architectures
   - Projection head designs
   - Regularization techniques (dropout, weight decay)

---

## TRAINING DETAILS

### Configuration
- **Dataset:** Tg.csv
- **Pretrained Encoder:** ./checkpoints_joint/best_encoder.pt
- **Batch Size:** 64
- **Max Epochs:** 300
- **Early Stopping:** Patience 30 epochs
- **5-Fold Cross-Validation:** Stratified split

### Training Timeline
- **Start Time:** 2026-03-24 12:40
- **End Time:** 2026-03-24 ~15:57
- **Total Duration:** ~3 hours 17 minutes
- **Log File Size:** 36 GB (mostly deprecation warnings)

---

## FILES CREATED

| File | Purpose |
|------|---------|
| `monitor_fold5.py` | Monitoring script for completion |
| `quick_monitor.py` | Quick status check script |
| `extract_final_results.py` | Results extraction script |
| `wait_for_all_folds.sh` | Automated waiting script |
| `FINAL_5FOLD_REPORT.md` | This report |
| `finetune_joint_tg.log` | Complete training log (36 GB) |

---

## CONCLUSION

**The joint pretraining approach has FAILED to improve performance over baseline.**

**Key Results:**
- Final RMSE: **39.23** (Baseline: **32.31**)
- Performance: **21.4% WORSE** than baseline
- All 5 folds consistently underperformed
- High variability across folds (5.08 RMSE range)

**Next Critical Step:** Compare with individual pretraining approaches (feature-only and contrastive-only) to determine if:
1. Joint pretraining is fundamentally flawed for this task
2. Or if pretraining in general doesn't help for Tg prediction

---

*Report generated automatically upon completion of all 5 folds*
*Training completed successfully, but results are disappointing*

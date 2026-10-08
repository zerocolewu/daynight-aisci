# HEPT v2 Experiment - Executive Summary

## Quick Facts

**Status**: ❌ **FAILED** - Did not achieve target performance

**Test RMSE**: 53.84 (Target: < 30.8)
**Gap to Target**: +23.04 (74.8% above target)
**Comparison to Iteration 1**: +4.26 RMSE (8.6% worse)

---

## Key Findings

### What Worked
✅ Successfully implemented MMFF-optimized 3D conformer generation
✅ Added 18 enhanced molecular features
✅ Increased model capacity to 8.18M parameters
✅ Improved regularization (dropout, weight decay)
✅ Training completed without errors

### What Didn't Work
❌ **Critical Bug**: Enhanced molecular features were computed but NOT used by the model
❌ Model overfitting: Larger model (8.18M) performed worse than smaller model (2.88M)
❌ MMFF optimization provided no measurable benefit
❌ R² degraded from 0.8041 → 0.7689
❌ Performance worse than simple GAT baseline

---

## Performance Comparison

| Model | RMSE | MAE | R² | Params | Verdict |
|-------|------|-----|----|---------|---------|
| PerioGT | **30.8** | - | - | - | 🎯 Target |
| GAT | 50.5 | - | - | - | Baseline |
| HEPT v1 | **49.58** | 35.93 | 0.8041 | 2.88M | ✅ Best |
| HEPT v2 | 53.84 | 39.56 | 0.7689 | 8.18M | ❌ Failed |

---

## Root Cause Analysis

### Primary Issue: Unused Features
The 18 enhanced molecular features were added to the data pipeline but the model's forward method doesn't accept or use them. This is a critical integration bug.

### Secondary Issue: Model Overfitting
- Model size increased by 2.84x (2.88M → 8.18M parameters)
- Performance degraded by 8.6%
- Clear sign of overfitting despite increased regularization

### Tertiary Issue: Wrong Improvements
- MMFF optimization may not be relevant for Tg prediction
- 3D geometry less important than 2D topology for Tg
- Computational overhead without benefit

---

## Critical Next Steps

### 1. Fix the Model (CRITICAL - Do First)
```python
# Current forward method signature:
def forward(self, x, pos, edge_index, batch)

# Needs to be:
def forward(self, x, pos, edge_index, batch, mol_features)

# And mol_features must be integrated into the prediction pipeline
```

### 2. Reduce Model Complexity
- Revert to hidden_dim=256 or go lower to 192
- Reduce layers: equivariant=3, periodic=2
- Target: ~3-4M parameters max

### 3. Ablation Studies
Test each improvement individually:
- MMFF vs. simple 3D
- With/without enhanced features (after fixing bug)
- Different regularization strengths
- Different model sizes

### 4. Consider Alternative Approaches
If current architecture can't reach target after fixes:
- Simpler architectures (GraphSAGE, GIN)
- Pre-training on larger datasets
- Ensemble methods
- Domain-specific features for polymers

---

## Training Details

**Duration**: 14.8 minutes (66 epochs)
**Best Epoch**: 41 (Val RMSE: 56.03)
**Early Stopping**: Triggered at epoch 66
**Learning Rate Schedule**:
- Epochs 1-51: 0.0005
- Epochs 52-62: 0.00025
- Epochs 63-66: 0.000125

**Data Split**:
- Train: 5,732 polymers (80%)
- Val: 713 polymers (10%)
- Test: 725 polymers (10%)

---

## Files

- **Report**: `HEPT_v2_Results_Report.md` (detailed analysis)
- **Checkpoints**: `checkpoints/hept_best.pt`, `checkpoints/hept_final.pt`
- **Training Log**: `experiment_v2_training.log`
- **Comparison Chart**: `analysis/v2_comparison.png`

---

## Conclusion

The HEPT v2 experiment was **technically successful** (all improvements implemented, training completed) but **scientifically unsuccessful** (performance degraded).

**Primary blocker**: Enhanced molecular features not integrated into model architecture.

**Recommendation**: Fix the model integration bug, reduce model complexity, and re-run before making architectural changes.

**Probability of success after fixes**: Moderate - still challenging to reach 30.8 RMSE target

---

*Generated: 2026-03-23*
*Experiment: HEPT v2 - Improved Training*
*Status: Complete - Needs Fixes*

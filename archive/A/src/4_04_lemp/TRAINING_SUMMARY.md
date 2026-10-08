# LEMP Training Experiment - Comprehensive Summary

## Experiment Overview

**Objective:** Train LEMP (Long-range Enhanced Molecular Property prediction) model with comprehensive ablation studies

**Dataset:** Tg (Glass Transition Temperature) prediction
**Training Method:** 5-fold cross-validation
**Date:** 2026-03-19

## Training Configuration

### Model Architecture:
- **Molecular Fingerprint:** 2048-bit ECFP
- **GIN Network:** 2 layers
- **Long-range Interactions:** Grid-based voxelization with FFT
  - Grid size: 16×16×16
  - Voxel channels: 8
  - K-space points: 32
  - Cutoff: 5Å

### Training Parameters:
- **Epochs:** 200 max (with early stopping, patience=20)
- **Batch size:** 32
- **Learning rate:** 1e-3
- **Optimizer:** Adam
- **Loss function:** MSE

## Ablation Study Design

### 7 Configurations × 5 Folds = 35 Total Training Runs

1. **Full LEMP** - All components enabled
2. **No Long-Range** - Baseline without long-range interactions
3. **No Fingerprint** - Remove molecular fingerprint features
4. **No GIN** - Remove graph isomorphism network
5. **Cutoff 5Å** - Full model with 5Å cutoff
6. **Cutoff 10Å** - Full model with 10Å cutoff
7. **Cutoff 15Å** - Full model with 15Å cutoff

## Progress Timeline

- **11:10** - Training started
- **13:09** - Full LEMP completed (5 folds)
- **15:09** - No Long-Range completed (5 folds)
- **17:22** - No Fingerprint completed (5 folds)
- **19:10** - No GIN completed (5 folds)
- **23:33** - Cutoff studies in progress
- **~00:30** - Expected completion

## Hypotheses to Validate

### Primary Hypothesis:
Long-range molecular interactions contribute >1.5 RMSE improvement
- **Test:** Compare Full LEMP vs No Long-Range
- **Expected:** Full LEMP RMSE - No Long-Range RMSE > 1.5

### Secondary Hypothesis:
Fingerprint features are additive (not replacing GIN representations)
- **Test:** Compare Full LEMP vs No Fingerprint
- **Expected:** Positive contribution (RMSE increase when removed)

### Performance Target:
Full model should achieve RMSE < 34.0 on Tg prediction

## Results Files

When training completes, results will be in:
1. **ablation_results.json** - All RMSE scores for each configuration
2. **analysis.json** - Component contribution analysis
3. **fold_*_best.pt** - Best model checkpoints for each fold

## Key Metrics to Check

```python
# Expected results structure:
{
  "Full LEMP": {"overall_rmse": X.XX},
  "No Long-Range": {"overall_rmse": Y.YY},
  "No Fingerprint": {"overall_rmse": Z.ZZ},
  "No GIN": {"overall_rmse": W.WW},
  "Cutoff 5Å": {"overall_rmse": V.VV},
  "Cutoff 10Å": {"overall_rmse": U.UU},
  "Cutoff 15Å": {"overall_rmse": T.TT}
}

# Component contributions:
long_range_contribution = No_LongRange_RMSE - Full_LEMP_RMSE
fingerprint_contribution = No_Fingerprint_RMSE - Full_LEMP_RMSE
gin_contribution = No_GIN_RMSE - Full_LEMP_RMSE
```

## Monitoring Commands

```bash
# Check if training is still running
ps -p 3577974

# Check latest model updates
ls -lt /home/home_public/aisci/AIscientist/src/4_04_lemp/fold_*_best.pt | head -5

# Check for results
ls -lah /home/home_public/aisci/AIscientist/src/4_04_lemp/*.json

# View results when ready
cat /home/home_public/aisci/AIscientist/src/4_04_lemp/ablation_results.json
cat /home/home_public/aisci/AIscientist/src/4_04_lemp/analysis.json
```

## Training Status (as of 23:34 CST)

- **Duration:** 12 hours 24 minutes
- **Configurations completed:** 5/7
- **Folds completed:** ~30/35 (86%)
- **Status:** Final cutoff studies in progress
- **Expected completion:** ~00:30 CST

## Next Steps

1. ⏳ Wait for ablation_results.json to appear
2. ✓ Read and analyze results
3. ✓ Validate hypotheses
4. ✓ Generate comprehensive report
5. ✓ Document findings

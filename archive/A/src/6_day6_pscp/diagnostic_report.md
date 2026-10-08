# DIAGNOSTIC REPORT: PSCP Model RMSE=103.18 vs Baseline=38.78

## Executive Summary

The PSCP (Polymer-Specific Contrastive Pretraining) model is achieving RMSE=103.18, which is **2.6x worse** than the expected baseline of ~38.78. This diagnostic investigation has identified **two critical issues** causing this poor performance.

---

## Investigation Findings

### 1. Dataset Analysis

| Property | Value |
|----------|-------|
| Total samples | 7,170 |
| Tg range | -139 to 495 °C |
| Tg mean | 141.9 °C |
| Tg std | 112.2 °C |
| Units | **Celsius** (not Kelvin) |

**Finding**: The dataset units are correct. The Tg values are in Celsius, which is standard for polymer glass transition temperatures. This is NOT a unit conversion issue.

---

### 2. Baseline Performance

| Method | RMSE (°C) | Notes |
|--------|-----------|-------|
| Random Forest + ECFP4 | **40.64 ± 2.1** | Good baseline |
| Gradient Boosting + ECFP4 | 41.32 ± 1.2 | Good baseline |
| MLP + ECFP4 (proper training) | **40.44 ± 0.7** | Good with correct setup |
| PSCP Model | **103.18 ± 1.6** | **2.6x worse** |

**Finding**: The baseline of ~38.78 K mentioned in the task is achievable. Both RF and properly-trained MLP with ECFP4 achieve ~40 RMSE.

---

### 3. CRITICAL ISSUE #1: Polymer SMILES Cleaning

**Problem**: The code incorrectly handles polymer SMILES notation.

Polymer SMILES use `*` to mark connection points:
- Original: `*CC(*)C`
- Current code: `CC()C` ← **INVALID** (82.8% of samples!)
- Proper handling: `[H]CC([H])C` or `CCC` ← **VALID**

**Impact**:
- **5,940 out of 7,170 samples (82.8%)** produce invalid SMILES
- These samples get placeholder/zero features
- Graph construction fails for most data
- ECFP generation may work by RDKit's graceful handling

**Evidence from logs**:
```
Current approach valid graphs: 1,230/7,170 (17.2%)
Proper approach valid graphs: 6,971/7,170 (97.2%)
```

---

### 4. CRITICAL ISSUE #2: Poor Atom Features

**Problem**: The GNN uses inadequate atom features.

| Aspect | Current Implementation | Standard Practice |
|--------|----------------------|-------------------|
| Feature dimension | **5** | **~159** |
| Encoding | Raw normalized values | One-hot encoding |
| atomic_num | /100 (e.g., C=0.06, O=0.08) | 119-dim one-hot |
| degree | /10 | 7-dim one-hot |
| charge | /10 | 8-dim one-hot |
| Other | is_aromatic, num_Hs | chiral, hybridization, mass |

**Impact**:
- Features like `atomic_num/100` produce tiny values (0.01-0.12)
- GNN cannot learn meaningful molecular representations
- Pretraining provides no benefit (all conditions give similar RMSE)

**Evidence**: All three pretraining conditions (no_pretrain, pretrain_10k, pretrain_100k) achieve nearly identical RMSE (~103), indicating the encoder features are not useful regardless of pretraining.

---

### 5. Training Analysis

From the experiment log:
```
Epoch 25: Loss=21925.09, Val RMSE=147.59, R2=-0.7426
Epoch 50: Loss=12942.20, Val RMSE=114.30, R2=-0.0452
Epoch 75: Loss=11094.25, Val RMSE=103.37, R2=0.1452
Epoch 100: Loss=10729.12, Val RMSE=103.51, R2=0.1428
```

**Observations**:
- Loss decreases but RMSE plateaus at ~103
- R² is very low (~0.15), indicating poor fit
- Model converges to a poor local minimum

---

## Root Cause Summary

The RMSE of 103.18 vs baseline 38.78 is caused by:

1. **Invalid SMILES handling (82.8% of data)**
   - Polymer markers (`*`) are removed incorrectly
   - Creates invalid SMILES like `CC()C`
   - Graph construction fails

2. **Poor atom feature representation**
   - Only 5 raw features vs standard 159-dim one-hot
   - Tiny normalized values hurt gradient flow
   - GNN cannot learn meaningful representations

3. **Combined architecture issues**
   - Poor GNN features may overwhelm ECFP signal
   - Model doesn't benefit from pretraining

---

## Recommendations

### Immediate Fixes

1. **Fix polymer SMILES cleaning**:
   ```python
   # Current (WRONG):
   clean = smiles.replace('*', '')

   # Proper:
   clean = smiles.replace('*', '[H]').replace('([H])', '')
   # Or use RDKit's ReplaceSubstructs
   ```

2. **Use proper atom features**:
   ```python
   # Use one-hot encoding (159-dim) instead of 5 raw features
   # Or use ECFP-only model which already works well
   ```

### Alternative Approach

Since RF+ECFP4 achieves ~40 RMSE, consider:
- Using ECFP-only model (simpler, works well)
- Only adding GNN if it provides clear benefit
- Proper ablation: compare ECFP-only vs ECFP+GNN

---

## Files Created

| File | Description |
|------|-------------|
| `diagnostic.py` | Initial diagnostic script |
| `diagnostic_detailed.py` | Detailed feature analysis |
| `diagnostic_final.py` | Final baseline comparison |
| `diagnostic_investigation.py` | MLP vs RF investigation |
| `diagnostic_smiles.py` | SMILES cleaning analysis |
| `diagnostic_report.md` | This report |

---

## Conclusion

**The PSCP model's poor performance (RMSE=103.18) is NOT due to unit conversion issues.**

The dataset is correctly in Celsius, and the baseline of ~38.78 is achievable with proper ECFP features.

The root causes are:
1. **82.8% of polymer SMILES are incorrectly processed**, creating invalid molecules
2. **Atom features are inadequate** (5 raw vs 159 one-hot), preventing the GNN from learning

Fixing these issues should bring performance close to the baseline of ~40 RMSE.

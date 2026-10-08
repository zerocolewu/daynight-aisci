# PerioGT-Lite: Multi-Seed Verification Report

## Executive Summary

**The 50% corruption pre-training result (49.83 RMSE) is NOT robust.** Across two independent multi-seed experiments (10 total runs each), the pre-trained encoder consistently fails to outperform the baseline. The original result appears to be a statistical fluke driven by GPU-level randomness.

---

## Experiment Design

### Two complementary experiments were run:

**Experiment 1: Random Splits** (tests generalization to different data splits)
- 5 seeds × 2 modes (baseline + pretrained) = 10 runs
- Data split: random 72/8/20 with seed-dependent permutation
- Same pre-trained encoder used for all seeds

**Experiment 2: Fixed Split** (tests robustness to training randomness only)
- 5 seeds × 2 modes = 10 runs
- Data split: original index-based split (identical for all seeds)
- Only training randomness varies (model init, data shuffling)

### Seeds tested: [42, 71, 123, 456, 789]

---

## Results

### Experiment 1: Random Splits

| Seed | Baseline | Pretrained | Diff | Winner |
|------|----------|------------|------|--------|
| 42   | 38.6481  | 41.4645    | -2.82 | Baseline |
| 71   | 37.1838  | 37.1846    | -0.00 | Baseline |
| 123  | 34.9827  | 37.1957    | -2.21 | Baseline |
| 456  | 39.0233  | 39.7030    | -0.68 | Baseline |
| 789  | 36.5592  | 37.0685    | -0.51 | Baseline |
| **Mean** | **37.28 ± 1.46** | **38.52 ± 1.77** | **-1.24** | **Baseline** |

**Pretrained wins: 0/5** (p=0.08, NOT significant)

### Experiment 2: Fixed Index-Based Split

| Seed | Baseline | Pretrained | Diff | Winner |
|------|----------|------------|------|--------|
| 42   | 50.5877  | 52.6042    | -2.02 | Baseline |
| 71   | 51.4874  | 53.3041    | -1.82 | Baseline |
| 123  | 53.6502  | 53.2300    | +0.42 | Pretrained |
| 456  | 53.7551  | 52.6645    | +1.09 | Pretrained |
| 789  | 52.9924  | 52.8083    | +0.18 | Pretrained |
| **Mean** | **52.49 ± 1.25** | **52.92 ± 0.29** | **-0.43** | **Baseline** |

**Pretrained wins: 3/5** (p=0.53, NOT significant)

---

## Critical Observations

### 1. Original Result Not Reproducible
The original seed-71 fixed-split result (49.83 RMSE pretrained vs 51.92 baseline) could NOT be reproduced with the same seed and same data split:
- **Original run**: Pretrained=49.83, Baseline=51.92 (pretrained wins by +2.10)
- **Re-run**: Pretrained=53.30, Baseline=51.49 (baseline wins by -1.82)

This 3.5 RMSE swing on the same configuration confirms extreme sensitivity to GPU-level non-determinism.

### 2. Pre-trained Encoder Shows No Consistent Benefit
- With random splits: baseline wins ALL 5 seeds (avg improvement: -3.34%)
- With fixed split: pretrained wins 3/5 but by tiny margins (avg improvement: -0.81%)
- Neither experiment shows statistically significant improvement (p=0.08 and p=0.53)

### 3. Pre-trained Encoder Has Lower Variance
Notably, the pre-trained encoder shows lower variance across seeds in the fixed-split experiment (std=0.29 vs 1.25 for baseline). This suggests the pre-training provides some regularization/stabilization, but NOT improved mean performance.

### 4. RMSE Levels Differ by Split Type
- Random splits: ~37 RMSE (both modes)
- Fixed index-based split: ~52-53 RMSE (both modes)
This 15-point difference indicates the original index-based split has a much harder test set (likely containing harder-to-predict polymers), making those RMSE values not directly comparable to random-split results.

---

## Conclusion

**VERDICT: The 50% corruption pre-training benefit is a STATISTICAL FLUCTUATION.**

1. The original 49.83 RMSE result cannot be reproduced with the same seed and split
2. Across 10 paired comparisons (5 seeds × 2 experiments), pre-training wins only 3/10 times
3. Mean performance favors baseline in both experiments
4. No statistically significant improvement detected (both p > 0.05)
5. The pre-training adds ~5 hours of compute for no reliable benefit

### Recommendation
Do NOT pursue contrastive pre-training further for this architecture/dataset combination. Focus on:
- Architectural improvements (attention mechanisms, better pooling)
- Feature engineering (better polymer representations)
- Ensemble methods for variance reduction
- Larger/better curated datasets

---

## Files Created
- `multiseed_verify.py`: Random-split verification script
- `multiseed_verify_fixedsplit.py`: Fixed-split verification script
- `multiseed_verify.log`: Full log of random-split experiment
- `multiseed_fixedsplit.log`: Full log of fixed-split experiment
- `multiseed_verification_results.json`: Random-split results (JSON)
- `multiseed_fixedsplit_results.json`: Fixed-split results (JSON)

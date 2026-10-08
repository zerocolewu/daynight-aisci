# BlochSpec Experiment: Final Report

## 1. Implementation Summary

### What was implemented
The complete BlochSpec experiment as specified, with 5 code files:

1. **compute_spectral.py** - Pre-computes Bloch band spectral features from polymer p-SMILES:
   - Chemistry-weighted Laplacian construction (electronegativity + bond type weighting)
   - Coupling matrix for periodic boundary conditions
   - Bloch Hamiltonian H(k) = L_UC + exp(ik)*L_coupling + exp(-ik)*L_coupling^T at 24 k-points
   - 12 compact spectral summary features extracted

2. **dataset.py** - Custom PyG InMemoryDataset combining trimer graphs + spectral features

3. **model.py** - Three model architectures:
   - `BlochSpecGIN`: GIN (3 layers, 256 hidden) + spectral features in prediction head
   - `PureSpectralMLP`: MLP on spectral features only (ablation)
   - `BaselineGIN`: Standard GIN with no spectral features

4. **train.py** - Full training pipeline with 5 experiments across 3 seeds (42, 71, 123)

5. **diagnostics.py** - Correlation analysis and single-feature importance experiments

### Key implementation details
- **GIN architecture**: EdgeAwareGINConv (same as Day 1 baseline), 3 layers, hidden_dim=256
- **Pooling**: global_add_pool (as specified in task), differs from Day 1's global_mean_pool
- **Prediction head**: Linear(268, 128) -> ReLU -> Dropout(0.1) -> Linear(128, 1)
- **Training**: LR=1e-3, cosine annealing, batch_size=32, early stopping patience=30
- **Split**: 72/8/20 with seed=71 (same as Day 1)

## 2. Execution Status: SUCCESS

All experiments completed without critical errors:
- Spectral features computed for 7169/7170 polymers (99.99% success rate)
- 15 training runs completed (5 experiments x 3 seeds)
- 12 single-feature importance experiments completed
- All diagnostic analyses completed

Minor issues:
- RDKit kekulization warnings during trimer generation (non-critical, handled by sanitization fallback)
- PyG UserWarning about accessing InMemoryDataset internal storage (non-critical)

## 3. Actual Results

### 3.1 Main Results (Multi-seed, mean +/- std)

| Experiment | Test RMSE | Train RMSE | Overfitting Gap |
|---|---|---|---|
| **(a) Baseline GIN+Trimer** | 66.15 +/- 0.27 | 26.84 +/- 3.05 | 39.32 +/- 3.27 |
| **(b) BlochSpec (GIN+Bloch)** | 61.52 +/- 5.99 | 26.34 +/- 6.73 | 35.18 +/- 3.24 |
| **(e) Pure Spectral MLP** | 76.71 +/- 1.26 | 72.46 +/- 1.63 | 4.25 +/- 0.69 |
| **(f) GIN + Random Features** | 58.68 +/- 1.41 | 28.80 +/- 6.20 | 29.88 +/- 5.43 |
| **(g) GIN + Unweighted Lap** | 58.45 +/- 1.09 | 21.11 +/- 0.57 | 37.34 +/- 1.24 |

### 3.2 Individual Seed Results

**Baseline GIN+Trimer:**
- seed=42: test=65.78, train=29.50, val=58.76, gap=36.29
- seed=71: test=66.42, train=22.56, val=55.64, gap=43.85
- seed=123: test=66.25, train=28.45, val=60.63, gap=37.81

**BlochSpec (GIN+Bloch):**
- seed=42: test=69.84, train=35.42, val=53.93, gap=34.42
- seed=71: test=58.79, train=19.32, val=48.80, gap=39.48
- seed=123: test=55.94, train=24.29, val=49.12, gap=31.66

**GIN + Random Features:**
- seed=42: test=58.81, train=35.99, val=54.85, gap=22.83
- seed=71: test=60.34, train=29.55, val=56.81, gap=30.79
- seed=123: test=56.90, train=20.86, val=51.68, gap=36.03

**GIN + Unweighted Lap:**
- seed=42: test=59.22, train=21.81, val=51.34, gap=37.42
- seed=71: test=56.90, train=21.11, val=53.87, gap=35.79
- seed=123: test=59.22, train=20.40, val=51.58, gap=38.82

### 3.3 Spectral Feature Correlations with Tg

| Feature | Pearson r | p-value | |r|>0.2 |
|---|---|---|---|
| spectral_gap | -0.4683 | 0 | *** |
| spectral_radius | +0.2040 | 3.1e-68 | *** |
| spectral_centroid | +0.5768 | 0 | *** |
| spectral_variance | +0.1538 | 3.2e-39 | |
| low_freq_density | -0.6366 | 0 | *** |
| band_width_max | -0.4473 | 0 | *** |
| band_width_mean | -0.3742 | 3.7e-237 | *** |
| band_flatness | -0.0555 | 2.6e-06 | |
| spectral_gap_kpi | -0.1793 | 7.4e-53 | |
| density_at_Fermi | -0.3037 | 7.2e-153 | *** |
| spectral_skew | -0.3548 | 1.0e-211 | *** |
| spectral_kurtosis | -0.2628 | 1.5e-113 | *** |

**9 out of 12 features** have |r| > 0.2 with Tg. Strongest: low_freq_density (r=-0.64), spectral_centroid (r=+0.58).

### 3.4 Feature Importance (Single Feature, ranked by test RMSE)

| Rank | Feature | Test RMSE |
|---|---|---|
| 1 | band_flatness | 62.60 |
| 2 | band_width_mean | 63.63 |
| 3 | spectral_kurtosis | 63.94 |
| 4 | spectral_radius | 64.56 |
| 5 | band_width_max | 65.15 |
| 6 | spectral_gap | 66.72 |
| 7 | spectral_centroid | 67.70 |
| 8 | spectral_variance | 67.94 |
| 9 | spectral_gap_kpi | 68.25 |
| 10 | low_freq_density | 68.96 |
| 11 | density_at_Fermi | 69.53 |
| 12 | spectral_skew | 70.68 |

Baseline without any spectral features: 66.42 (seed=71).

## 4. Hypothesis Evaluation

### H1: Input-space periodicity bypasses representation-usage gap
**PARTIALLY SUPPORTED.** BlochSpec (61.52) improved over baseline (66.15) by ~4.6 RMSE on average, but with high variance (std=5.99). However, this improvement was NOT unique to physics-derived features - random features (58.68) and unweighted features (58.45) performed equally well or better. The improvement appears to come from having ANY additional features rather than specifically Bloch features.

### H2: Bloch band structure encodes Tg-relevant physics
**SUPPORTED.** The strong Pearson correlations confirm the spectral features carry Tg-relevant signal. The most correlated features - low_freq_density (r=-0.64), spectral_centroid (r=+0.58), spectral_gap (r=-0.47) - have clear physical interpretations (chain flexibility, average coupling strength, chain connectivity). The Pure Spectral MLP achieved 76.71 RMSE from just 12 features, far below random expectation (~140), confirming the physics is informative.

### H3: Chemistry-weighted Laplacian captures both topology and chemistry
**NOT SUPPORTED.** The unweighted Laplacian (58.45) performed essentially the same as the weighted version (61.52), and actually slightly better. The chemistry weighting via electronegativity does not appear to add value beyond simple topological connectivity.

### Key unexpected finding
The most striking result is that **random features** (58.68) performed comparably to physics-derived features (61.52) and unweighted features (58.45). This suggests the improvement from adding spectral features comes primarily from the additional input dimensions providing the model with extra fitting capacity, rather than from the specific physics content. However, the spectral features ARE informative (strong correlations with Tg), so their value may be limited by the GNN already capturing similar information.

## 5. Caveats and Methodological Notes

1. **Pooling method**: This experiment used global_add_pool as specified, while the Day 1 baseline used global_mean_pool. The baseline RMSE of 66.15 (add_pool) vs the reference 50.19 (mean_pool) differs significantly. All relative comparisons between experiments remain valid since they use the same pooling.

2. **High variance in BlochSpec**: The full BlochSpec model showed std=5.99 across seeds, suggesting training instability. Seed 42 gave 69.84 while seed 123 gave 55.94.

3. **Feature importance paradox**: Features with the strongest correlations (low_freq_density r=-0.64) were NOT the most useful when combined with GNN (ranked 10th). This suggests the GNN already captures some of this information, making the additional features redundant.

## 6. File Structure

```
src/4_4_blochspec/
├── compute_spectral.py    # Pre-compute Bloch spectral features
├── dataset.py             # Custom PyG dataset with spectral features
├── model.py               # Model architectures (BlochSpecGIN, PureSpectralMLP, BaselineGIN)
├── train.py               # Training loop with multi-seed support
├── diagnostics.py         # Correlation analysis, feature importance
├── FINAL_REPORT.md        # This report
├── spectral_data/
│   ├── spectral_features.npy           # Weighted Bloch features (7169 x 12)
│   ├── spectral_features_unweighted.npy # Unweighted Bloch features (7169 x 12)
│   ├── spectral_features_random.npy     # Random features (7169 x 12)
│   ├── feature_indices.npy              # Valid polymer indices
│   ├── feature_tgs.npy                  # Tg values for valid polymers
│   ├── feature_names.txt                # Feature name list
│   └── correlation_analysis.txt         # Pearson correlations
├── data/
│   └── processed/
│       └── blochspec_trimer.pt          # Processed PyG dataset
├── models/                              # Trained model checkpoints (27 files)
└── results/
    ├── results_summary.txt              # Multi-seed RMSE results
    ├── feature_importance.txt           # Single-feature analysis
    └── diagnostics.txt                  # Full diagnostic report
```

## 7. Next Steps Suggestions

1. **Use global_mean_pool**: Switch from add_pool to mean_pool to match the reference baseline and get more comparable absolute numbers. The relative trends should persist.

2. **Feature normalization**: Standardize spectral features before concatenation. The features have very different scales (e.g., spectral_radius ~10 vs spectral_gap_kpi ~0.02).

3. **Spectral feature gating**: Instead of simple concatenation, use a learned gating mechanism to selectively use spectral information.

4. **Investigate the random feature anomaly**: The strong performance of random features warrants deeper investigation. Consider matching the dimensionality of random features to test if it's purely a dimensionality effect.

5. **Larger k-point sampling**: Try more k-points (48 or 96) for finer spectral resolution.

6. **Direct eigenvalue features**: Instead of summary statistics, try using a subset of raw eigenvalues (e.g., at k=0 and k=pi) as features.

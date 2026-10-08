# PolyKAN-SSM v4: Final Experimental Report

## Executive Summary

PolyKAN-SSM v4 successfully reduced the overfitting problem that plagued v3, achieving a **train-val gap of only 0.55** compared to v3's gap of ~11.2. However, the absolute test RMSE increased from ~40 to ~54, suggesting that while overfitting was fixed, the model capacity may have been reduced too much.

## Main Experiment Results (5 seeds)

### Performance Metrics
- **Mean Test RMSE**: 54.54 ± 1.56
- **Mean Test MAE**: 40.82 ± 0.97
- **Mean Test R²**: 0.762 ± 0.014
- **Mean Train RMSE**: 53.56 ± 2.38
- **Mean Val RMSE**: 54.11 ± 2.43
- **Train-Val Gap**: 0.55 (major improvement from v3's ~11.2!)
- **Best Single Seed**: 52.25

### Per-Seed Results
| Seed | Train RMSE | Val RMSE | Test RMSE | Test MAE | Test R² |
|------|------------|----------|-----------|----------|---------|
| 42   | 50.40      | 50.20    | 52.25     | 38.94    | 0.785   |
| 123  | 55.78      | 56.99    | 53.70     | 41.36    | 0.765   |
| 456  | 56.58      | 54.90    | 54.98     | 40.92    | 0.754   |
| 789  | 51.51      | 55.86    | 56.97     | 41.68    | 0.744   |
| 2024 | 53.55      | 52.60    | 54.82     | 41.21    | 0.763   |

### Key Observations
1. **Overfitting Fixed**: Train-val gap of 0.55 is excellent, showing the SSM-based architecture successfully controls gradient flow and prevents memorization.
2. **Consistent Performance**: All 5 seeds perform similarly, indicating good stability.
3. **Not Yet SOTA**: Mean RMSE of 54.54 is still above the SOTA target of 30.8.

## Ablation Studies (seed=42)

### Results Summary
| Ablation | Test RMSE | Test MAE | Test R² | Notes |
|----------|-----------|----------|---------|-------|
| **No Fourier-KAN** | 50.13 | 36.23 | 0.802 | Replaced FourierKANLayer with Linear+GELU |
| **No GNN-SSM** | 45.56 | 30.06 | 0.837 | Replaced GNNSSMLayer with GINConv |
| **No non-covalent edges** | 48.34 | 34.83 | 0.816 | Covalent bonds only |
| **Full v4 Model** | 52.25 | 38.94 | 0.785 | Complete architecture |

### Surprising Findings

1. **No GNN-SSM performed best** (45.56 RMSE):
   - Standard GINConv actually outperformed the custom SSM layer
   - Suggests the SSM formulation may be overly constraining
   - Train-val gap: 8.3 (still reasonable)

2. **No Fourier-KAN also better** (50.13 RMSE):
   - Simple Linear+GELU outperformed Fourier-KAN
   - Train-val gap: 2.2 (excellent)
   - Fourier basis may not be adding value

3. **No non-covalent edges** (48.34 RMSE):
   - Removing non-covalent edges improved performance
   - Suggests the non-covalent edges may be adding noise
   - Train-val gap: -1.8 (actually underfitting slightly)

## Architecture Changes in v4

### Key Modifications from v3
1. **FourierKANLayer**: Improved formulation with learnable frequencies initialized to small values
2. **GNNSSMLayer**: New SSM-style GNN with multi-head attention message aggregation
3. **No Virtual Node**: Removed to reduce parameters and overfitting
4. **Smaller Architecture**: hidden_dim=192 (vs 256), num_layers=4 (vs 5)
5. **Dual Pooling**: Attention-weighted + mean pooling concatenated
6. **Smaller Head**: Reduced prediction head size to prevent overfitting

### What Worked
- Dramatic reduction in overfitting (0.55 gap vs 11.2 in v3)
- Stable training across all seeds
- No early stopping issues

### What Didn't Work
- Absolute performance degraded (54 vs 40 in v3)
- SSM layer may be too restrictive
- Fourier-KAN not providing clear benefit
- Non-covalent edges not helping performance

## Recommendations for Future Iterations

1. **Try v5 with GINConv backbone**: The ablation shows standard GIN works best
2. **Simplify further**: Remove Fourier-KAN, use standard activations
3. **Reconsider non-covalent edges**: May need better feature engineering or distance cutoff
4. **Increase capacity**: Now that overfitting is controlled, can afford larger model
5. **Hyperparameter tuning**: The ablation models used same hyperparams as v4

## Conclusion

PolyKAN-SSM v4 successfully solved the overfitting problem but at the cost of absolute performance. The ablation studies reveal valuable insights:
- The SSM formulation may be overly complex
- Standard GINConv with proper regularization works well
- Fourier-KAN needs rethinking or removal

The best path forward is likely a v5 that combines the best elements:
- GINConv backbone (from ablation 2)
- No Fourier-KAN (from ablation 1)
- Strong regularization (from v4)
- Moderate capacity increase

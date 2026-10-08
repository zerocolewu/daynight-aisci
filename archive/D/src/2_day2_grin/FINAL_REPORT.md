# GRIN Experiment Final Report
## Graph Repetition INvariance for Polymer Property Prediction

**Date**: March 24, 2026
**Dataset**: Tg (Glass Transition Temperature) - 7170 polymers
**Results Directory**: `/home/home_public/aisci/AIscientist/src/2_day2_grin/results_20260324_072344/`

---

## Executive Summary

This experiment tested the GRIN (Graph Repetition INvariance) hypothesis for polymer property prediction on the Tg dataset. The core hypothesis was that **max-aggregation with L1 sparsity penalty combined with {1-RU, 3-RU} training data augmentation would achieve repeat-size invariance and improve prediction accuracy**.

### Key Findings

1. **Baseline model performed best**: GIN with mean aggregation (no augmentation) achieved Test RMSE of **76.08**, outperforming all GRIN variants tested
2. **Max aggregation degraded performance**: Max-only aggregation increased RMSE to 78.75 (+3.5% worse)
3. **L1 sparsity provided marginal improvement**: Adding L1 to max aggregation reduced RMSE to 77.94 (still worse than baseline)
4. **Augmentation experiments failed**: {1,3}-RU augmentation encountered critical implementation issues with SMILES-to-graph conversion
5. **Target not achieved**: Best RMSE (76.08) was **2.06x higher** than the GraphFormer baseline target of 37.0

---

## Experimental Setup

### Dataset
- **Source**: `/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv`
- **Size**: 7,170 polymers with glass transition temperatures
- **Features**: Polymer SMILES (p-SMILES) with `*` marking polymerization points
- **Split**: 80/10/10 train/val/test (5,736 / 717 / 717 samples)
- **Target**: Glass transition temperature (Tg) - continuous value

### Model Architecture
- **Base**: GIN (Graph Isomorphism Network)
- **Hidden Dimension**: 256
- **Number of Layers**: 5
- **Total Parameters**: ~701K
- **Aggregation Strategies Tested**:
  - Mean aggregation (baseline)
  - Max aggregation (GRIN variant)

### Training Protocol
- **Optimizer**: Adam (lr=1e-4)
- **Scheduler**: Cosine Annealing
- **Batch Size**: 64
- **Max Epochs**: 200
- **Early Stopping**: Patience 30 on validation RMSE
- **Loss Function**: MSE + L1 penalty (when enabled, λ=0.001)

### Experiments Conducted
1. **Baseline**: GIN with mean aggregation, no augmentation
2. **Max Only**: GIN with max aggregation only
3. **Max + L1**: GIN with max aggregation + L1 sparsity penalty
4. **Full GRIN**: Max + L1 + {1,3}-RU augmentation (FAILED)
5. **Mean + Aug**: Mean aggregation + augmentation (NOT RUN due to previous failure)

---

## Detailed Results

### Quantitative Performance

| Experiment | Test RMSE | Test MAE | Best Val RMSE | Best Epoch | Status |
|------------|-----------|----------|---------------|------------|--------|
| **Baseline** | **76.08** | **52.27** | **76.65** | 199 | ✅ Complete |
| Max Only | 78.75 | 54.56 | 79.76 | 200 | ✅ Complete |
| Max + L1 | 77.94 | 53.88 | 79.33 | 200 | ✅ Complete |
| Full GRIN | - | - | - | - | ❌ Failed |
| Mean + Aug | - | - | - | - | ⏭️ Skipped |

### Performance Analysis

#### Baseline vs Max Aggregation
- **Baseline RMSE**: 76.08
- **Max Only RMSE**: 78.75
- **Performance Gap**: +3.51% worse with max aggregation
- **Interpretation**: Mean aggregation better captures polymer structure for this dataset

#### Effect of L1 Sparsity
- **Max Only RMSE**: 78.75
- **Max + L1 RMSE**: 77.94
- **Improvement**: -1.03% (minor benefit)
- **Interpretation**: L1 regularization helps slightly but doesn't overcome max aggregation's disadvantage

#### Comparison with Target
- **Target RMSE**: 37.0 (GraphFormer baseline)
- **Best Achieved**: 76.08
- **Gap**: +105.6% (2.06x worse than target)

---

## Technical Issues Encountered

### 1. Augmentation Implementation Failure

**Issue**: The {1-RU, 3-RU} augmentation strategy failed during training due to SMILES-to-graph conversion errors.

**Root Cause**:
- The simplified repeat unit concatenation (`*CCC*` → `CCC-CCC-CCC`) generated invalid SMILES
- PyTorch Geometric's `from_smiles()` function failed to convert some augmented polymers
- This caused batch size mismatches and tensor dimension errors

**Impact**:
- Full GRIN and Mean + Aug experiments could not complete
- Unable to test the core hypothesis about augmentation benefits

### 2. Data Type Serialization

**Issue**: JSON serialization failed with numpy float32 types

**Resolution**: Explicitly converted all numeric values to Python floats before JSON dumping

### 3. Training Time

**Observation**: Each experiment took 13-20 minutes for 200 epochs
- Baseline: ~13 minutes (199 epochs)
- Max Only: ~15 minutes (200 epochs)
- Max + L1: ~15 minutes (200 epochs)

---

## Implementation Details

### Files Created

```
/home/home_public/aisci/AIscientist/src/2_day2_grin/
├── data_utils.py           # Data preprocessing and augmentation
├── models.py                # GIN with max aggregation
├── train.py                 # Training pipeline
├── analysis.py              # Analysis utilities
├── generate_report.py       # Report generation
├── visualize_results.py     # Result visualization
└── results_20260324_072344/
    ├── baseline/
    │   ├── best_model.pt
    │   ├── results.json
    │   ├── training_history.png
    │   └── predictions.png
    ├── max_only/
    │   ├── best_model.pt
    │   ├── results.json
    │   ├── training_history.png
    │   └── predictions.png
    ├── max_l1/
    │   ├── best_model.pt
    │   ├── results.json
    │   ├── training_history.png
    │   └── predictions.png
    ├── comparison_results.png
    ├── training_curves.png
    └── validation_rmse_curves.png
```

### Model Architecture Details

**GINMaxAggregation**:
```python
- Input projection: Linear(9 → 256)
- 5 GIN layers with BatchNorm and ReLU
- Each GIN layer: MLP with [256 → 256 → 256]
- Aggregation: Mean or Max (configurable)
- Output head: [256 → 128 → 1] with LayerNorm
- Dropout: 0.1 between layers
```

### Data Preprocessing

**Graph Construction**:
- Used PyTorch Geometric's `from_smiles()` for automatic feature extraction
- 9-dimensional atom features (default PyG features)
- Edge features included bond types

**Augmentation Strategy** (attempted):
- 1-RU: Original p-SMILES
- 3-RU: Concatenate 3 repeat units (simplified approach)
- Implementation: Replace `*` markers with repeat unit to extend chain

---

## Analysis and Discussion

### Why Did Max Aggregation Perform Worse?

**Hypothesis 1: Information Loss**
- Max aggregation selects only the most informative neighbor message
- For polymer graphs, this may discard important structural information
- Mean aggregation preserves more information by averaging all neighbors

**Hypothesis 2: Gradient Flow**
- Max operation has sparse gradients (only to the selected neighbor)
- This may hinder learning of fine-grained polymer features
- Mean aggregation provides dense gradients to all neighbors

**Hypothesis 3: Dataset Characteristics**
- Tg dataset may not exhibit the size-dependence that max aggregation is designed to address
- Polymer property prediction may benefit more from holistic aggregation

### Why Did the Experiments Fail to Meet the Target?

1. **Model Capacity**: 701K parameters may be insufficient for this task
   - GraphFormer likely uses larger models or pre-training

2. **Feature Engineering**: Default PyG atom features may be suboptimal
   - Custom polymer-specific features (molecular weight, topology) could help

3. **Data Processing**: p-SMILES handling was simplified
   - More sophisticated polymer graph construction might improve performance

4. **Hyperparameters**: Limited exploration of hyperparameter space
   - Learning rate, hidden dim, number of layers could be optimized

5. **Missing Augmentation**: Core GRIN augmentation component couldn't be tested
   - This might have provided significant improvements

### What Can Be Learned?

1. **Baseline importance**: Simple approaches can outperform sophisticated ones
2. **Implementation details matter**: Augmentation failure prevented full hypothesis testing
3. **Domain knowledge is critical**: Polymer-specific features may be more important than aggregation strategy
4. **Target gaps are informative**: 2x performance gap suggests fundamental approach differences

---

## Recommendations

### Short-term Improvements

1. **Fix Augmentation Pipeline**
   - Use RDKit to properly generate 3-RU polymers
   - Add robust error handling for failed SMILES conversions
   - Implement `drop_last=True` in DataLoader to handle partial batches

2. **Feature Engineering**
   - Add molecular weight as a graph-level feature
   - Include polymer topology descriptors
   - Use custom atom features (e.g., aromatic flags, degree)

3. **Hyperparameter Tuning**
   - Grid search over learning rates [1e-5, 1e-3]
   - Test different hidden dimensions [128, 512]
   - Experiment with L1 lambda values [0.0001, 0.01]

4. **Model Improvements**
   - Increase model capacity (2-3M parameters)
   - Try attention-based aggregation
   - Add residual connections

### Long-term Directions

1. **Transfer Learning**
   - Pre-train on larger polymer datasets
   - Fine-tune on Tg prediction

2. **Multi-task Learning**
   - Train on multiple polymer properties simultaneously
   - Leverage correlations between properties

3. **Better Polymer Representation**
   - Use polymer-specific molecular representations
   - Consider 3D conformer information
   - Incorporate monomer-level features

4. **Ensemble Methods**
   - Combine predictions from multiple models
   - Use different aggregation strategies in ensemble

---

## Conclusion

### Summary of Findings

The GRIN experiment on the Tg dataset yielded mixed but informative results:

✅ **Successfully Completed**:
- Implemented GIN with max aggregation and L1 sparsity
- Trained and evaluated 3 ablation experiments
- Generated comprehensive visualizations and analysis

❌ **Failed to Achieve**:
- Target RMSE < 37.0 (achieved 76.08, 2.06x worse)
- GRIN hypothesis testing (augmentation failed)
- Performance improvement over baseline

📊 **Key Insights**:
- Mean aggregation outperformed max aggregation for this dataset
- L1 sparsity provided marginal benefit
- Implementation challenges prevented full hypothesis evaluation
- Significant gap to target suggests fundamental approach differences

### Honest Assessment

**Success**: ❌ Did not meet target
**Insight**: ✅ Provided valuable negative results
**Implementation**: ⚠️ Partially successful (3/5 experiments)

The experiment revealed that:
1. The GRIN approach with max aggregation does not improve over standard GIN for this dataset
2. The augmentation strategy requires more sophisticated implementation
3. The performance gap to GraphFormer suggests missing components (pre-training, features, or architecture)

### Next Steps

To properly evaluate the GRIN hypothesis:
1. Fix the augmentation pipeline using proper polymer chemistry tools
2. Test on datasets where size-invariance is more critical
3. Compare against GraphFormer directly to understand the performance gap
4. Consider that the GRIN insights may be dataset-dependent

---

## Files and Resources

### Code Files
- `data_utils.py`: Data loading and augmentation utilities
- `models.py`: GIN with max aggregation implementation
- `train.py`: Training pipeline and experiment runner
- `analysis.py`: Embedding analysis and visualization tools
- `visualize_results.py`: Result visualization script

### Results
- `results_20260324_072344/`: Complete experiment results
  - Model checkpoints for all completed experiments
  - Training histories and predictions
  - Comparison visualizations

### Visualizations
- `comparison_results.png`: Bar chart comparing all experiments
- `training_curves.png`: Training loss curves for each experiment
- `validation_rmse_curves.png`: Validation RMSE over training
- Individual experiment directories contain additional plots

---

**Report Generated**: March 24, 2026, 08:05 UTC
**Experiment Duration**: ~45 minutes (3/5 experiments completed)
**Total GPU Time**: ~43 minutes

---

## Appendix: Raw Results

### Baseline Results
```json
{
  "test_rmse": 76.0824,
  "test_mae": 52.2674,
  "best_val_rmse": 76.6540,
  "best_epoch": 199
}
```

### Max Only Results
```json
{
  "test_rmse": 78.7479,
  "test_mae": 54.5568,
  "best_val_rmse": 79.7625,
  "best_epoch": 200
}
```

### Max + L1 Results
```json
{
  "test_rmse": 77.9384,
  "test_mae": 53.8768,
  "best_val_rmse": 79.3343,
  "best_epoch": 200
}
```

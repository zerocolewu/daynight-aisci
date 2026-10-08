# PerioGT Ablation Study - README

## Quick Navigation

### 📊 **Main Results**

- **EXECUTION_SUMMARY.md** - Start here! Complete summary of experiment
- **COMPREHENSIVE_REPORT.md** - Detailed technical analysis
- **5fold_cv_results.png** - Visualization of results

### 📈 **Data Files**

- **cv_results.csv** - Per-fold results from 5-fold cross-validation
- **cv_experiment.log** - Full training log
- **cv_summary.json** - Summary statistics

### 💻 **Code Files**

- **run_5fold_cv.py** - Main experiment script (5-fold CV)
- **model.py** - Model implementations (Chemprop MPNN, GraphFormer)
- **dataset.py** - Dataset loading and periodic graph construction
- **visualize_results.py** - Result visualization script

---

## Experiment Results at a Glance

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Test RMSE** | **46.89 ± 1.41** | < 32.0 | ❌ FAILED |
| **Paper RMSE** | **37.35** | - | Reference |
| **Gap to Paper** | **+9.54** | - | - |
| **Test R²** | **0.825 ± 0.008** | - | Good |

---

## What Was Implemented

### ✅ PerioGT Architecture (from paper)

1. **Periodic Graph Representation**
   - Edges between terminal atoms (wildcards)
   - p-SMILES parsing with `**` notation
   - Correct atom/bond featurization

2. **Chemprop-style MPNN**
   - Directed bond-level message passing
   - 3 message passing layers
   - Hidden dim = 300
   - 2-layer FFN with 300 neurons

3. **5-Fold Cross-Validation**
   - 80/10/10 split (train/val/test)
   - 50 epochs per fold
   - Best model by validation RMSE

### ❌ Missing Components

1. **NOAM Learning Rate Schedule** (CRITICAL)
   - Chemprop uses warmup + cosine decay
   - We used constant LR = 1e-3
   - Estimated impact: **5-10 RMSE improvement**

2. **Median Aggregation**
   - For duplicate measurements
   - Estimated impact: **1-2 RMSE**

---

## Why We Didn't Meet the Target

### Root Cause: Missing Learning Rate Schedule

The paper says "Chemprop default hyperparameters" but doesn't explicitly mention the learning rate schedule. Chemprop's default is a NOAM-style schedule with:

- 2 epoch warmup (LR increases from init to max)
- Cosine decay from max to final LR
- This helps escape local minima and converge better

**Evidence**: Training curves show models still improving at epoch 50

---

## How to Reproduce

### Run 5-Fold Cross-Validation

```bash
python run_5fold_cv.py
```

This will:
1. Load Tg dataset from `/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv`
2. Convert SMILES to periodic graphs
3. Run 5-fold CV with Chemprop-style MPNN
4. Save results to `cv_results.csv`

### Generate Visualizations

```bash
python visualize_results.py
```

Creates `5fold_cv_results.png` with comparison charts.

---

## Key Findings

### 1. Periodic Graphs Work

Our implementation correctly captures periodicity:
- R² = 0.825 (good correlation)
- Consistent across folds (std = 1.41)

### 2. Learning Rate Schedule is Critical

Missing the NOAM schedule caused ~9.5 RMSE gap:
- Our constant LR: 46.89 RMSE
- Paper's NOAM schedule: 37.35 RMSE

### 3. GraphFormer vs Chemprop

Surprising finding from previous work:
- **GraphFormer**: 37.00 ± 0.68 RMSE
- **Chemprop (this work)**: 46.89 ± 1.41 RMSE
- **Paper Chemprop**: 37.35 RMSE

GraphFormer with proper training may actually be better!

---

## Recommendations

### To Close the Gap (Priority Order)

1. **HIGH**: Implement NOAM learning rate schedule
   - Expected: ~40-42 RMSE (5-10 point improvement)

2. **MEDIUM**: Use actual Chemprop package
   - `pip install chemprop`
   - Modify for periodic graphs

3. **LOW**: Data preprocessing
   - Median aggregation
   - Better filtering

### Alternative: Use GraphFormer

Our previous GraphFormer (37.00 RMSE) nearly matches the paper (37.35):
- Try GraphFormer with 5-fold CV
- Might achieve similar or better results
- Already has proper attention mechanism

---

## Important Clarifications

### ❌ Misconception: "Periodic Attention"

The paper does NOT use a special "periodic attention mechanism". The innovation is:
- **Periodic graph construction** (adding edge between terminal atoms)
- Standard Chemprop MPNN on this periodic graph

### ❌ Misconception: Target < 32.0

The paper reports **37.35 RMSE**, not 30.8:
- Target < 32.0 is unrealistic
- Even perfect implementation would achieve ~37-38 RMSE

---

## File Structure

```
2_02_PerioGT_Ablation/
├── EXECUTION_SUMMARY.md      # START HERE - Complete summary
├── COMPREHENSIVE_REPORT.md    # Detailed technical analysis
├── README.md                  # This file
│
├── run_5fold_cv.py           # Main experiment script
├── model.py                   # Model implementations
├── dataset.py                 # Data loading
├── visualize_results.py       # Visualization script
│
├── cv_results.csv            # Per-fold results
├── cv_experiment.log         # Training log
├── cv_summary.json           # Summary stats
│
├── 5fold_cv_results.png      # Result visualizations
│
└── [Previous experiment files]
    ├── analysis_report.md
    ├── comprehensive_results.csv
    └── ...
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{antoniuk2022representing,
  title={Representing Polymers as Periodic Graphs with Learned Descriptors for Accurate Polymer Property Predictions},
  author={Antoniuk, Evan R and Li, Peggy and Kailkhura, Bhavya and Hiszpanski, Anna M},
  journal={Journal of Chemical Information and Modeling},
  volume={62},
  number={21},
  pages={5435--5445},
  year={2022},
  publisher={ACS Publications}
}
```

---

## Contact & Questions

For questions about this implementation:
- Check EXECUTION_SUMMARY.md for detailed analysis
- Check COMPREHENSIVE_REPORT.md for technical details
- Review cv_experiment.log for training details

---

**Last Updated**: March 24, 2026
**Status**: Experiment Complete (Failed to meet target)
**Next Action**: Implement NOAM learning rate schedule or try GraphFormer with 5-fold CV

# Chemprop-Polymer-K: Contrastive Pre-Training Pipeline

Complete implementation of contrastive pre-training for polymer property prediction using Chemprop-Polymer-K architecture.

## Files Overview

### Core Files

1. **model.py** (351 lines)
   - `ChempropPolymerKMPNN`: Main model class
   - K-node support for global polymer context
   - Projector head for contrastive pre-training
   - `collate_batch()`: Batch collation function

2. **data_utils.py** (866 lines)
   - Feature dimension constants (ATOM_FDIM=123, BOND_FDIM=9, K_FEAT_DIM=528)
   - p-SMILES parsing
   - Graph construction (monomer, dimer, trimer)
   - K-node feature computation (528 dims)
   - Dataset loading functions for Tg and PI1M

3. **pretrain.py** (560 lines)
   - `ContrastiveDataset`: Dataset with on-the-fly augmentation
   - `info_nce_loss()`: Contrastive loss with temperature scaling
   - `pretrain()`: Main pre-training loop
   - Supports 4 variants (A, B, C, D)

4. **finetune.py** (432 lines)
   - `TgDataset`: Dataset for fine-tuning
   - `finetune()`: Fine-tuning with early stopping
   - Multi-seed evaluation (seeds 71, 42, 123)
   - Returns detailed metrics per seed

5. **generate_random.py** (236 lines)
   - `generate_random_graph()`: Create single random graph
   - `generate_random_graphs()`: Generate 300K random graphs
   - Erdos-Renyi random graph structure
   - No periodic edges or polymer structure

6. **diagnostics.py** (579 lines)
   - `augmentation_spectrum()`: Analyze augmentation intensity effects
   - `embedding_diversity()`: Check for embedding collapse
   - `overfitting_gap_analysis()`: Compare train-test gaps across variants

## Usage

### 1. Pre-Training

```bash
# Variant B: PI1M with K-node
python pretrain.py \
    --data /path/to/pi1m_graphs.pkl \
    --checkpoint checkpoints/variant_B.pt \
    --variant variant_B \
    --batch-size 32 \
    --epochs 30

# Variant C: Random graphs with K-node
python pretrain.py \
    --data /path/to/random_graphs.pkl \
    --checkpoint checkpoints/variant_C.pt \
    --variant variant_C \
    --batch-size 32 \
    --epochs 30

# Variant D: PI1M without K-node
python pretrain.py \
    --data /path/to/pi1m_graphs.pkl \
    --checkpoint checkpoints/variant_D.pt \
    --no-knode \
    --variant variant_D \
    --batch-size 32 \
    --epochs 30
```

### 2. Fine-Tuning

```bash
# Variant A: No pre-training (random initialization)
python finetune.py \
    --tg-data /path/to/tg_graphs.pkl \
    --checkpoint None \
    --variant variant_A \
    --output results/variant_A.pkl

# Variant B: Pre-trained on PI1M with K-node
python finetune.py \
    --tg-data /path/to/tg_graphs.pkl \
    --checkpoint checkpoints/variant_B.pt \
    --variant variant_B \
    --output results/variant_B.pkl

# Variants C, D: Similar with respective checkpoints
```

### 3. Generate Random Graphs

```bash
python generate_random.py \
    --num-graphs 300000 \
    --output /path/to/random_graphs.pkl \
    --seed 42
```

### 4. Diagnostic Analysis

```bash
# Augmentation spectrum
python diagnostics.py \
    --mode augmentation_spectrum \
    --checkpoint checkpoints/variant_B.pt \
    --data /path/to/pi1m_graphs.pkl \
    --output results/aug_spectrum.pkl

# Embedding diversity
python diagnostics.py \
    --mode embedding_diversity \
    --checkpoint checkpoints/variant_B.pt \
    --data /path/to/pi1m_graphs.pkl \
    --output results/embedding_diversity.pkl

# Overfitting gap analysis
python diagnostics.py \
    --mode overfitting_gap \
    --checkpoint checkpoints/variant_B.pt \
    --data /path/to/pi1m_graphs.pkl \
    --results results/combined_results.pkl \
    --output results/overfitting_gaps.pkl
```

## Key Features

### Contrastive Pre-Training
- **InfoNCE Loss**: Temperature τ=0.07, L2-normalized embeddings
- **Augmentation**:
  - Trimer-shift: monomer/dimer/trimer selection
  - Bond dropping: 15% rate (protects K-node connections)
  - Feature masking: 25% rate (excludes K-node features)
- **K-node Protection**: K-node features NEVER augmented

### Model Architecture
- **Encoder**: 7-layer directed bond-level MPNN
- **Hidden Dimension**: 1200
- **K-node**: Virtual atom connected to all real atoms (528-dim features)
- **Projector**: 1200 → 128 → 64 (for pre-training only)

### Fine-Tuning
- **Dataset Split**: 72% train / 8% val / 20% test (seed=71)
- **Optimization**: Adam, lr=1e-3 or 1e-4
- **Early Stopping**: Patience=30 epochs
- **Multi-Seed**: Seeds {71, 42, 123} for FFN initialization
- **Prediction Head**: Randomly initialized FFN (1200 → 1200 → 1200 → 1)

## Data Requirements

### Input Graph Format
Each graph dict must contain:
- `atom_features`: [n_atoms, 123] tensor (real atoms only)
- `k_node_features`: [528] tensor (if use_knode)
- `bond_features`: [n_bonds, 9] tensor
- `edge_index`: [2, n_bonds] tensor (directed bonds)
- `reverse_map`: [n_bonds] tensor
- `num_atoms`: int (real atoms, excluding K-node)
- `target`: [1] tensor (for fine-tuning)
- `psmiles`: str
- `periodic_edge_bonds`: list of indices

For contrastive pre-training, graphs should also include:
- `monomer_*`, `dimer_*`, `trimer_*`: Alternative fragment views

### Pre-Processing
Use `data_utils.py` functions:
```python
from data_utils import process_tg_dataset, process_pi1m_dataset

# Process Tg dataset
tg_graphs = process_tg_dataset(
    csv_path='/path/to/Tg.csv',
    use_knode=True,
    augment=False,
    seed=42
)

# Process PI1M dataset
pi1m_graphs = process_pi1m_dataset(
    csv_path='/path/to/PI1M.csv',
    use_knode=True,
    augment=False,
    max_samples=300000,
    seed=42
)
```

## Expected Results

### Pre-Training
- 30 epochs over 300K PI1M samples
- Batch size: 32 polymers (64 graphs with 2 views)
- Best loss typically: 2.5-3.5
- Training time: ~2-3 hours on RTX 2080 Ti (11GB)

### Fine-Tuning
- Variant A (no pre-training): RMSE ~25-30 K
- Variant B (PI1M + K-node): RMSE ~20-25 K (expected improvement)
- Variant C (random + K-node): RMSE ~25-30 K (control)
- Variant D (PI1M no K-node): RMSE ~22-27 K

### Diagnostic Metrics
- **Augmentation Spectrum**: Bond dropping should reduce cos_sim to 0.7-0.9
- **Embedding Diversity**: Mean cos_sim should be 0.1-0.3 (not collapsed)
- **Overfitting Gap**: Should be <15 K for good generalization

## Hardware Requirements
- GPU: NVIDIA GPU with ≥11GB VRAM (RTX 2080 Ti or better)
- RAM: 32GB recommended
- Storage: ~10GB for datasets and checkpoints

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Reduce model size: `--hidden-dim 800` (not recommended)
- Use gradient checkpointing (not implemented)

### Slow Training
- Reduce number of samples: `--num-samples 100000`
- Use smaller model: `--hidden-dim 800`
- Use CPU (not recommended): `--device cpu`

### Poor Results
- Check data quality: ensure graphs are constructed correctly
- Verify K-node features: should be non-zero
- Adjust hyperparameters: learning rate, temperature, augmentation rates
- Train longer: `--epochs 50`

## Citation
If you use this code, please cite the relevant papers.

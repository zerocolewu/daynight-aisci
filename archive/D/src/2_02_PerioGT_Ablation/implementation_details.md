# PerioGT Implementation Details - Literature Review

## Paper Reference
**"Representing Polymers as Periodic Graphs with Learned Descriptors for Accurate Polymer Property Predictions"**
Antoniuk et al., J. Chem. Inf. Model. 2022, 62, 5435-5445

## Key Implementation Details from Paper

### 1. Model Architecture (Chemprop MPNN)

| Component | Paper Value | Notes |
|-----------|-------------|-------|
| Message passing type | Directed bond-level messages | Not atom-level |
| Bond message vector size | 300 | Hidden dimension |
| Number of message passing steps | 3 | T iterations |
| FFN hidden layers | 2 | After message passing |
| FFN hidden neurons | 300 | Per layer |
| Activation function | ReLU | Throughout network |
| Aggregation | Sum incoming edges | For atom embeddings |

### 2. Graph Representation (Periodic Polymer Graph)

| Component | Description |
|-----------|-------------|
| Base graph | Monomer graph |
| Periodic edge | Connects terminal atoms |
| Edge features | Same as other edges (bond type features) |
| Purpose | Ensures correct bonding environment, captures periodicity |

### 3. Training Protocol

| Parameter | Paper Value |
|-----------|-------------|
| Cross-validation | 5-fold |
| Train/val/test split | 80/10/10 |
| Epochs | 50 |
| Early stopping | Best model by lowest validation RMSE |
| Framework | Chemprop default hyperparameters |

### 4. Data Preprocessing

| Parameter | Description |
|-----------|-------------|
| Input format | Monomer SMILES with `**` notation |
| Example | `*CCO*` for `(-CH2-CH2-O-)n` |
| Data source | PoLyInfo database (Tg) |
| Multiple measurements | Median value used |
| Exclusions | Compound/composite samples excluded |

### 5. Reported Performance

| Property | MPNN/Periodic Graph RMSE | MPNN/Monomer Graph RMSE |
|----------|-------------------------|------------------------|
| Tg (°C) | 37.35 | 39.83 |

## Critical Implementation Details to Ablate

### A. Data Preprocessing
1. Normalization: None mentioned (raw values)
2. Split method: Random split (5-fold CV)
3. Median aggregation for multiple measurements

### B. Training Protocol
1. Epochs: 50 (vs our 500)
2. No explicit learning rate mentioned (Chemprop default)
3. No weight decay mentioned (Chemprop default)
4. Early stopping: Select best validation model

### C. Model Hyperparameters
1. Hidden dim: 300 (vs our 256)
2. Message passing steps: 3 (vs our 4 layers)
3. FFN structure: 2 layers x 300 neurons

### D. Architecture Details
1. **Chemprop uses directed bond-level messages** - NOT standard atom-level GNN
2. Aggregation: Node embeddings from sum of incoming edges
3. Node features: Concatenated with atomic features, multiplied by learned matrix, ReLU

## Gap Analysis

| Factor | Current Implementation | Paper | Potential Impact |
|--------|----------------------|-------|------------------|
| Hidden dim | 256 | 300 | Medium |
| Num layers | 4 | 3 | Medium |
| Epochs | 500 | 50 | High (overfitting?) |
| Model type | Custom GraphTransformer | Chemprop MPNN | High |
| LR scheduler | CosineAnnealing | Unknown (default) | Medium |
| Message type | Atom-level | Bond-level | High |

## Hypothesis
The 5.6 RMSE gap may be due to:
1. **Over-training**: 500 epochs vs 50 epochs may cause overfitting
2. **Architecture mismatch**: Our GraphTransformer vs Chemprop's bond-level MPNN
3. **Hidden dimension**: 256 vs 300
4. **Training dynamics**: Different LR schedule/optimization

## Ablation Plan

### Phase 1: Quick Wins (Training Protocol)
- Reduce epochs to 50-100
- Test paper's hidden_dim (300)
- Test paper's num_layers (3)

### Phase 2: Architecture Alignment
- Implement bond-level message passing
- Match FFN structure exactly

### Phase 3: Fine-tuning
- Learning rate search
- Batch size search
- Regularization tuning

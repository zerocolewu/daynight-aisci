# PPFN: Polymer Periodic Fragment Network

Hierarchical graph neural network for predicting polymer glass transition temperature (Tg).

## Model Architecture

The PPFN model integrates three hierarchical levels of representation:

### 1. Atom-Level GNN (GINE)
- **Input**: Trimer graph with periodic edges
- **Architecture**: 4 layers of Graph Isomorphism Network with Edge features
- **Features**: Captures local atomic environments and bonding patterns
- **Dimension**: 27 atom features, 6 edge features → 256 hidden

### 2. Fragment-Level GNN (GAT)
- **Input**: Fragment graph (BRICS fragments as nodes)
- **Architecture**: 2 layers of Graph Attention Network with 4 heads
- **Features**: Models relationships between chemical fragments
- **Dimension**: 31 fragment features → 256 hidden

### 3. Long-Range Bipartite Module
- **Input**: Bipartite graph between all atoms and fragments
- **Architecture**: 2 layers of transformer-style cross-attention (4 heads each)
- **Features**: Captures non-local dependencies across the polymer
- **Dimension**: 256 → 256 hidden

### 4. Prediction Head
- Global mean pooling of atom, fragment, and long-range representations
- MLP: 768 → 512 → 256 → 1
- Output: Glass transition temperature (Tg)

## Model Statistics

- **Total Parameters**: 3,641,349
- **Hidden Dimension**: 256
- **CUDA Compatible**: Yes
- **Batch Processing**: Yes

## Feature Dimensions

### Atom Features (27)
- 18 atom types (one-hot): C, N, O, S, F, Si, P, Cl, Br, I, B, Se, Te, Ge, As, Sb, Bi, *
- 1 degree (integer)
- 5 hybridization types (one-hot): SP, SP2, SP3, SP3D, SP3D2
- 1 aromatic flag
- 1 formal charge
- 1 ring flag

### Edge Features (6)
- 4 bond types (one-hot): SINGLE, DOUBLE, TRIPLE, AROMATIC
- 1 conjugated flag
- 1 ring flag

### Fragment Features (31)
- 27 (mean of atom features in fragment)
- 1 fragment size (normalized by total atoms)
- 1 aromatic flag (from fragment SMILES)
- 1 OH group flag (from fragment SMILES)
- 1 NH group flag (from fragment SMILES)

## Usage

### Basic Training

```python
from model import PPFN
from dataset import PPFNDataset, collate_fn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

# Load data
train_dataset = PPFNDataset('path/to/Tg.csv', split='train')
train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, collate_fn=collate_fn)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PPFN(hidden_dim=256).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Training loop
model.train()
for batch in train_loader:
    # Move to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    # Forward pass
    predictions = model(batch)

    # Compute loss
    loss = F.mse_loss(predictions.squeeze(), batch['tg'])

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Interpretability

```python
# Get fragment attention weights
model.eval()
with torch.no_grad():
    predictions = model(batch)
    interp = model.interpret(batch)

# Access interpretability information
fragment_attention = interp['fragment_attention']  # (N_frags,)
fragment_importance = interp['fragment_importance']  # (batch_size, max_frags)

# For each sample in batch, see which fragments are most important
for i in range(batch_size):
    sample_frags = fragment_importance[i]
    print(f"Sample {i} fragment importance: {sample_frags}")
```

### Evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in val_loader:
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Forward pass
        predictions = model(batch)

        all_preds.extend(predictions.squeeze().cpu().numpy())
        all_targets.extend(batch['tg'].cpu().numpy())

# Compute metrics
mse = mean_squared_error(all_targets, all_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
```

## File Structure

```
1_day2_ppfn/
├── dataset.py              # Data loading and preprocessing
├── model.py                # PPFN model architecture
├── train.py                # Training script (to be created)
├── test_model_dataset.py   # Test model with real data
├── test_edge_cases.py      # Test edge cases
├── test_training.py        # Training simulation test
└── model_summary.py        # Architecture documentation
```

## Key Features

1. **Hierarchical Design**: Captures multi-scale polymer structure from atoms to fragments
2. **Periodic Edges**: Special handling for polymer repeat units via periodic boundary edges
3. **Attention Mechanisms**: Multiple attention layers for capturing important features
4. **Interpretability**: Built-in attention weight extraction for analysis
5. **Robust**: Handles edge cases (empty graphs, varying sizes, etc.)

## Model Components

### MultiHeadAttention
Standard multi-head attention for fragment-level processing.

### AtomToFragmentAggregation
Attention-based aggregation from atom representations to fragment representations using bipartite graph structure.

### LongRangeBipartiteModule
Cross-attention module capturing long-range dependencies between all atoms and fragments within each polymer.

### GINE Layers (Atom-level)
Graph Isomorphism Network with Edge features for processing local atomic structure.

### GAT Layers (Fragment-level)
Graph Attention Network for processing fragment-level relationships.

## Testing

All tests pass successfully:
- ✓ Model creation with 3.64M parameters
- ✓ CPU and CUDA forward pass
- ✓ Edge case handling (empty graphs, single atoms, etc.)
- ✓ Training loop with backward pass
- ✓ Interpretability method
- ✓ Real dataset integration

## Next Steps

1. Create `train.py` with full training pipeline
2. Implement early stopping and model checkpointing
3. Add hyperparameter tuning
4. Perform extensive evaluation on test set
5. Analyze attention weights for chemical insights
6. Visualize learned representations

## References

- GINE: Graph Isomorphism Network with Edge features
- GAT: Graph Attention Network
- BRICS: Fragmentation method for chemical compounds
- Periodic boundary conditions for polymer modeling

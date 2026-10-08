"""
PPFN Model Architecture Summary
================================

This file documents the complete architecture of the Polymer Periodic Fragment Network.

Model Statistics:
-----------------
Total Parameters: 3,641,349
Hidden Dimension: 256

Architecture Components:
-----------------------

1. INPUT EMBEDDINGS
   - Atom embedding: Linear(27 -> 256)
   - Edge embedding: Linear(6 -> 256)
   - Fragment embedding: Linear(31 -> 256)

2. ATOM-LEVEL GNN (4 layers of GINE)
   - Graph Isomorphism Network with Edge features
   - Each layer: Linear(256 -> 256) -> BatchNorm -> ReLU -> Dropout(0.1)
   - Processes trimer graph with periodic edges
   - Captures local atomic environments and bonding patterns

3. FRAGMENT-LEVEL GNN (2 layers of GAT)
   - Graph Attention Network with 4 heads
   - Each layer: GATConv(256 -> 256) -> BatchNorm -> ReLU -> Dropout(0.1)
   - Processes fragment graph (fragments as nodes)
   - Captures relationships between chemical fragments

4. ATOM-TO-FRAGMENT AGGREGATION
   - Attention-based message passing from atoms to fragments
   - Multi-head attention (4 heads) for aggregating atom information
   - Uses atom_frag_edge_index for bipartite connectivity
   - Output: Updated fragment representations with atom-level context

5. LONG-RANGE BIPARTITE MODULE
   - Transformer-style cross-attention between atoms and fragments
   - 2 layers, 4 attention heads each
   - Fragments query all atoms in the same polymer
   - Captures non-local dependencies and long-range interactions
   - Output: Fragment representations enriched with global atom information

6. PREDICTION HEAD
   - Global mean pooling:
     * Atom representations: (batch_size, 256)
     * Fragment representations: (batch_size, 256)
     * Long-range representations: (batch_size, 256)
   - Concatenate: (batch_size, 768)
   - MLP: 768 -> 512 -> 256 -> 1
   - Dropout(0.2) between layers, ReLU activation
   - Output: Glass transition temperature prediction

Feature Dimensions:
------------------
- NUM_ATOM_FEATURES = 27
  - 18 atom types (one-hot)
  - 1 degree
  - 5 hybridization types (one-hot)
  - 1 aromatic flag
  - 1 formal charge
  - 1 ring flag

- NUM_EDGE_FEATURES = 6
  - 4 bond types (one-hot)
  - 1 conjugated flag
  - 1 ring flag

- NUM_FRAG_FEATURES = 31
  - 27 (mean of atom features in fragment)
  - 1 fragment size (normalized)
  - 1 aromatic flag
  - 1 OH group flag
  - 1 NH group flag

Input Batch Format:
------------------
The model expects a batch dictionary with the following keys:

- 'atom_features': (N_total_atoms, 27)
  - Atom features for all atoms across all samples

- 'edge_index': (2, N_total_edges)
  - COO format edge indices for atom-level graph

- 'edge_features': (N_total_edges, 6)
  - Edge features for atom-level bonds

- 'frag_features': (N_total_frags, 31)
  - Fragment features for all fragments

- 'frag_edge_index': (2, N_total_frag_edges)
  - Edge indices for fragment-level graph

- 'frag_edge_features': (N_total_frag_edges, 6)
  - Edge features for fragment-level connections

- 'atom_frag_edge_index': (2, N_atom_frag_edges)
  - Bipartite edges: row 0 = atom indices, row 1 = fragment indices

- 'batch_atom': (N_total_atoms,)
  - Batch assignment for each atom

- 'batch_frag': (N_total_frags,)
  - Batch assignment for each fragment

Model Usage:
------------
```python
from model import PPFN
from dataset import PPFNDataset, collate_fn
from torch.utils.data import DataLoader

# Load data
dataset = PPFNDataset('path/to/Tg.csv', split='train')
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Create model
model = PPFN(hidden_dim=256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop
for batch in loader:
    # Move to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    # Forward pass
    predictions = model(batch)

    # Compute loss
    loss = F.mse_loss(predictions, batch['tg'])

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Interpretability
interp = model.interpret(batch)
fragment_attention = interp['fragment_attention']  # Per-fragment importance
```

Interpretability:
-----------------
The model provides interpretability through fragment attention weights:

1. Fragment-level attention: Which fragments contribute most to prediction
2. Atom-to-fragment attention: How atoms influence fragment representations
3. Long-range attention: Cross-attention weights for global dependencies

Use model.interpret(batch) to extract attention weights for analysis.

Key Design Choices:
------------------
1. GINE for atom-level: Preserves graph isomorphism while using edge features
2. GAT for fragment-level: Attention mechanism weights fragment relationships
3. Bipartite design: Explicitly models atom-fragment hierarchy
4. Long-range module: Captures non-local interactions beyond local bonds
5. Periodic edges: Special handling for polymer repeat units

Hierarchical Representation:
---------------------------
The model builds representations at three levels:

1. Local (atoms): Chemical structure, bonds, functional groups
2. Mesoscale (fragments): Chemical moieties, building blocks
3. Global (polymer): Long-range interactions, overall properties

This hierarchical design mirrors the multi-scale nature of polymers,
from atomic structure to bulk material properties.
"""

if __name__ == '__main__':
    print(__doc__)

# Supplementary Information: Baseline Model Configurations

This section documents the architectures of the four baseline models
(ECFP+MLP, GCN, GIN, GAT) used for polymer glass-transition temperature
(Tg) and melting temperature (Tm) regression. All graph neural networks
operate on the same monomer-level molecular graph (n_mer = 1, * atoms
capped with H) and share an identical 35-dimensional categorical atomic
node feature vector and global-mean pooling, so that observed performance
differences reflect the message-passing operator rather than input
representation.

## Table S1. Atomic node features (35-dimensional, one-hot categorical)

All GNN baselines (GCN, GIN, GAT) use the same per-atom feature vector
below. Categorical descriptors (element, degree, hydrogen count,
hybridization) are encoded as one-hot vectors; the two continuous
descriptors (formal charge, atomic mass) are range-scaled. Bonds are
encoded as a bidirectional edge index (no edge attributes are used by
the GNN baselines). Features are computed with RDKit.

| Block | Dims | Descriptor | Encoding scheme |
|---|---|---|---|
| 1 | 13 | Atomic number Z | One-hot over {H, B, C, N, O, F, Si, P, S, Cl, Br, I} + 1 "other" bucket |
| 2 | 6 | Atom degree | One-hot over {0, 1, 2, 3, 4, ≥5} |
| 3 | 1 | Aromaticity | Binary {0, 1} |
| 4 | 1 | Ring membership | Binary {0, 1} |
| 5 | 1 | Formal charge | q / 10 (range-scaled) |
| 6 | 6 | Total Hs (implicit) | One-hot over {0, 1, 2, 3, 4, ≥5} |
| 7 | 5 | Hybridization | One-hot over {S, SP, SP2, SP3, SP3D} |
| 8 | 1 | Atomic mass | m / 100 (range-scaled) |
| 9 | 1 | Chirality possible | Binary {0, 1} (`atom.HasProp('_ChiralityPossible')`) |
| **Total** | **35** | | |

## Table S2. Molecular fingerprint (ECFP+MLP baseline)

The ECFP+MLP baseline does not consume a graph; instead it uses a
fixed-length molecular fingerprint computed on the same H-capped
monomer structure.

| Property | Value |
|---|---|
| Algorithm | Morgan / ECFP4 (radius = 2) |
| Bit-vector length | 2048 |
| Implementation | `rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator` |

## Table S3. Model architectures

All GNN models use 3 message-passing layers followed by global mean
pooling and a 2-layer regression head
(`Linear(d, d/2) → ReLU → Dropout(p) → Linear(d/2, 1)`).
The ECFP+MLP is a feed-forward MLP with three hidden layers and a final
linear regression layer. Dropout = 0.1 for all GNNs, 0.2 for ECFP+MLP.

| Model | Input dim | Message-passing body (3 layers) | Pooling | Regression head | Body activation |
|---|---|---|---|---|---|
| **ECFP+MLP** | 2048 | `Linear(2048→512)` → `Linear(512→512)` × 2 | — | `Linear(512→1)` | ReLU + Dropout(0.2) |
| **GCN** | 35 | `GCNConv(35→128)` → `GCNConv(128→128)` × 2 | global mean | `Linear(128→64) → ReLU → Dropout(0.1) → Linear(64→1)` | ReLU + Dropout(0.1) |
| **GIN** | 35 | `GINConv{MLP: Linear(35→128)→ReLU→Linear(128→128)}` → `GINConv{MLP(128→128→128)}` × 2 | global mean | `Linear(128→64) → ReLU → Dropout(0.1) → Linear(64→1)` | (MLP-internal ReLU, ε = 0 fixed) + Dropout(0.1) |
| **GAT** | 35 | `GATConv(35→64, heads=4, concat)` → `GATConv(256→64, heads=4, concat)` → `GATConv(256→64, heads=4, avg)` | global mean | `Linear(64→32) → ReLU → Dropout(0.1) → Linear(32→1)` | ELU + Dropout(0.1) |

Notes on the GNN building blocks:
- **GCNConv** follows Kipf & Welling (2017); propagation uses the
  symmetrically normalized adjacency with self-loops.
- **GINConv** follows Xu et al. (2019), `h' = MLP((1+ε)·h + Σ_{u∈N(v)} h_u)`
  with ε = 0 (fixed) and `MLP = Linear → ReLU → Linear`. No external
  non-linearity is applied between GIN layers (the MLP carries it).
- **GATConv** follows Veličković et al. (2018) with additive attention.
  The first two layers concatenate the 4 heads
  (output dim = `hidden × heads = 256`); the last layer **averages**
  the heads (output dim = `hidden = 64`), as in the original paper.

## Table S4. Trainable parameter counts

| Model | Body parameters | Head parameters | Total parameters |
|---|---|---|---|
| ECFP+MLP | 1,574,400 | 513 | **1,574,913** |
| GCN | 37,632 | 8,321 | **45,953** |
| GIN | 87,168 | 8,321 | **95,489** |
| GAT | 142,144 | 2,113 | **144,257** |

For ECFP+MLP, "body" = three hidden Linear layers and "head" = the final
Linear(512→1). For the GNNs, "body" = the three message-passing layers
and "head" = the 2-layer regression MLP. Counts were obtained with
`sum(p.numel() for p in model.parameters() if p.requires_grad)`.

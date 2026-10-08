"""
Day 4 Experiment 1: Prompt-Conditioned Fine-Tuning (PCFT)
Tests H4: The representation-usage gap can be bridged by making pre-trained
representations an active conditioning input (prompt) rather than just weight initialization.

Architecture: Dual-Encoder with Frozen Prompt
- Encoder P (frozen): Pre-trained GIN on monomer graphs -> h_pre
- Encoder T (trainable): GIN on trimer graphs -> h_task
- Prediction Head: MLP(h_task || h_pre) -> Tg

Phases:
1. Pre-training on PI1M (reuse existing checkpoint from Day 3)
2. Compute h_pre lookup table for all Tg monomers
3-4. Fine-tuning with PCFT + all ablations (multi-seed)
5. Diagnostic analyses
"""

import os
import sys
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from tqdm import tqdm
from copy import deepcopy
import pandas as pd

# ---- Configuration ----
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), 'alldataset')
PERIOGT_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), '3_periogt_lite')

# Architecture params (matching Day 3/Day 1 baseline: 3 layers, 256 hidden)
NODE_DIM = 30
EDGE_DIM = 6
HIDDEN_DIM = 256
NUM_LAYERS = 3  # Same as existing pre-trained checkpoint
DROPOUT = 0.5   # Dropout for Encoder T during fine-tuning

# Training params
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 30
MAX_EPOCHS = 500
SPLIT_SEED = 71
TRAINING_SEEDS = [42, 71, 123]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'experiment_log.txt'))
    ]
)
logger = logging.getLogger(__name__)


# ===================================================================
# Feature Extraction (consistent with Day 1-3)
# ===================================================================
ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'Si', 'B']
DEGREES = [0, 1, 2, 3, 4, 5]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
]
H_COUNTS = [0, 1, 2, 3, 4, 5]
CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
]
ATOM_FEATURE_DIM = 30
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
BOND_FEATURE_DIM = 6


def one_hot(value, choices):
    vec = [0] * len(choices)
    try:
        idx = choices.index(value)
        vec[idx] = 1
    except ValueError:
        pass
    return vec


def get_atom_features(atom):
    symbol = atom.GetSymbol()
    atom_type = one_hot(symbol, ATOM_TYPES)
    degree = one_hot(atom.GetTotalDegree(), DEGREES)
    hyb = atom.GetHybridization()
    hybridization = one_hot(hyb, HYBRIDIZATIONS)
    aromatic = [int(atom.GetIsAromatic())]
    ring = [int(atom.IsInRing())]
    h_count = one_hot(atom.GetTotalNumHs(), H_COUNTS)
    chirality = one_hot(atom.GetChiralTag(), CHIRALITY)
    return atom_type + degree + hybridization + aromatic + ring + h_count + chirality


def get_bond_features(bond):
    bond_type = one_hot(bond.GetBondType(), BOND_TYPES)
    conjugated = [int(bond.GetIsConjugated())]
    ring = [int(bond.IsInRing())]
    return bond_type + conjugated + ring


# ===================================================================
# Graph Utilities (p-SMILES -> monomer/trimer graphs)
# ===================================================================
def safe_sanitize(mol):
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return True
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
        return True
    except Exception:
        return False


def find_star_atoms(mol):
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]


def get_star_neighbors(mol, star_indices):
    star_to_neighbor = {}
    for star_idx in star_indices:
        atom = mol.GetAtomWithIdx(star_idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() != 0:
                star_to_neighbor[star_idx] = nbr.GetIdx()
                break
    return star_to_neighbor


def psmiles_to_monomer_mol(p_smiles):
    """Convert p-SMILES to monomer (remove star atoms)."""
    try:
        mol = Chem.MolFromSmiles(p_smiles)
        if mol is None:
            return None
        star_indices = find_star_atoms(mol)
        if len(star_indices) != 2:
            return None
        star_to_neighbor = get_star_neighbors(mol, star_indices)
        if len(star_to_neighbor) != 2:
            return None
        rw_mol = Chem.RWMol(mol)
        remove_order = sorted(star_indices, reverse=True)
        for idx in remove_order:
            rw_mol.RemoveAtom(idx)
        result = rw_mol.GetMol()
        if not safe_sanitize(result):
            return None
        return result
    except Exception:
        return None


def psmiles_to_trimer_mol(p_smiles):
    """Convert p-SMILES to trimer (3 repeat units)."""
    try:
        mol = Chem.MolFromSmiles(p_smiles)
        if mol is None:
            return None
        star_indices = find_star_atoms(mol)
        if len(star_indices) != 2:
            return None
        star_to_neighbor = get_star_neighbors(mol, star_indices)
        if len(star_to_neighbor) != 2:
            return None

        stars = list(star_to_neighbor.keys())
        head_star, tail_star = stars[0], stars[1]
        head_nbr_orig = star_to_neighbor[head_star]
        tail_nbr_orig = star_to_neighbor[tail_star]

        rw_mol = Chem.RWMol(mol)
        remove_order = sorted([head_star, tail_star], reverse=True)
        for idx in remove_order:
            rw_mol.RemoveAtom(idx)

        atoms_removed = sorted([head_star, tail_star])

        def update_idx(idx):
            new_idx = idx
            for removed_idx in atoms_removed:
                if idx > removed_idx:
                    new_idx -= 1
            return new_idx

        monomer_head = update_idx(head_nbr_orig)
        monomer_tail = update_idx(tail_nbr_orig)
        monomer = rw_mol.GetMol()
        n_monomer_atoms = monomer.GetNumAtoms()

        combined = Chem.RWMol(Chem.Mol())
        for unit in range(3):
            offset = combined.GetNumAtoms()
            for atom in monomer.GetAtoms():
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                new_atom.SetIsAromatic(atom.GetIsAromatic())
                new_atom.SetChiralTag(atom.GetChiralTag())
                new_atom.SetHybridization(atom.GetHybridization())
                new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
                combined.AddAtom(new_atom)
            for bond in monomer.GetBonds():
                begin = bond.GetBeginAtomIdx() + offset
                end = bond.GetEndAtomIdx() + offset
                combined.AddBond(begin, end, bond.GetBondType())

        for i in range(2):
            tail_of_i = i * n_monomer_atoms + monomer_tail
            head_of_next = (i + 1) * n_monomer_atoms + monomer_head
            combined.AddBond(tail_of_i, head_of_next, Chem.BondType.SINGLE)

        result = combined.GetMol()
        if not safe_sanitize(result):
            return None
        return result
    except Exception:
        return None


def mol_to_data(mol, target=None):
    """Convert RDKit Mol to PyG Data object."""
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat = get_bond_features(bond)
        edge_indices.append([begin, end])
        edge_indices.append([end, begin])
        edge_attrs.append(feat)
        edge_attrs.append(feat)

    if len(edge_indices) == 0:
        edge_indices = [[0, 0]]
        edge_attrs = [[0.0] * BOND_FEATURE_DIM]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_atoms)
    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float)
    return data


# ===================================================================
# Model Architecture
# ===================================================================
class EdgeAwareGINConv(MessagePassing):
    """GIN convolution with edge feature awareness."""

    def __init__(self, nn_func, edge_dim, hidden_dim, train_eps=True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.nn = nn_func
        self.edge_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([0.0]))
        else:
            self.register_buffer('eps', torch.Tensor([0.0]))

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out + (1 + self.eps) * x

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr)
            return x_j + edge_feat
        return x_j

    def update(self, aggr_out):
        return self.nn(aggr_out)


class GINEncoder(nn.Module):
    """
    GIN Encoder matching Day 1/3 baseline:
    - 3 layers, hidden_dim=256
    - BatchNorm inside MLP + BatchNorm after conv + ReLU
    - train_eps=True
    """

    def __init__(self, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            conv = EdgeAwareGINConv(
                nn_func=mlp,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                train_eps=True,
            )
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_embedding(x)
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            h = torch.relu(h)
            if self.dropout_layer is not None:
                h = self.dropout_layer(h)
        h_graph = global_mean_pool(h, batch)
        return h_graph


class PCFTModel(nn.Module):
    """
    PCFT: Dual-Encoder with Frozen Prompt.
    Encoder P (frozen) processes monomer -> h_pre
    Encoder T (trainable) processes trimer -> h_task
    Prediction Head: MLP(h_task || h_pre) -> Tg
    """

    def __init__(self, encoder_p_state_dict=None, freeze_p=True,
                 dropout_t=DROPOUT, head_dropout=0.3):
        super().__init__()

        # Encoder P: Frozen pre-trained encoder for monomer (no dropout)
        self.encoder_p = GINEncoder(dropout=0.0)
        if encoder_p_state_dict is not None:
            self.encoder_p.load_state_dict(encoder_p_state_dict)
        if freeze_p:
            for param in self.encoder_p.parameters():
                param.requires_grad = False

        # Encoder T: Fine-tuning encoder for trimer (with dropout)
        self.encoder_t = GINEncoder(dropout=dropout_t)
        if encoder_p_state_dict is not None:
            self.encoder_t.load_state_dict(deepcopy(encoder_p_state_dict))

        # Prediction Head: 512 -> 256 -> 128 -> 1
        self.pred_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_tri, edge_index_tri, edge_attr_tri, batch_tri,
                h_pre_batch):
        """
        Args:
            x_tri, edge_index_tri, edge_attr_tri, batch_tri: trimer graph batch
            h_pre_batch: pre-computed h_pre vectors for this batch [batch_size, 256]
        Returns:
            Tg prediction [batch_size]
        """
        h_task = self.encoder_t(x_tri, edge_index_tri, edge_attr_tri, batch_tri)
        h_combined = torch.cat([h_task, h_pre_batch], dim=-1)
        pred = self.pred_head(h_combined)
        return pred.squeeze(-1)

    def get_encoder_p_repr(self, x, edge_index, edge_attr, batch):
        """Compute h_pre using frozen encoder P."""
        with torch.no_grad():
            return self.encoder_p(x, edge_index, edge_attr, batch)


class BaselineGINModel(nn.Module):
    """Standard single-encoder GIN for ablation (a) and (d)."""

    def __init__(self, pretrained_state_dict=None, freeze_encoder=False, dropout=0.0):
        super().__init__()
        self.encoder = GINEncoder(dropout=dropout)
        if pretrained_state_dict is not None:
            self.encoder.load_state_dict(pretrained_state_dict)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x, edge_index, edge_attr, batch)
        return self.head(h).squeeze(-1)


# ===================================================================
# Data Processing
# ===================================================================
def load_tg_dataset():
    """Load and preprocess Tg dataset: monomer + trimer graphs."""
    logger.info("Loading Tg dataset...")
    csv_path = os.path.join(DATA_DIR, 'Tg.csv')
    df = pd.read_csv(csv_path)
    smiles_col = df.columns[0]
    tg_col = df.columns[1]

    # Check for cached data
    cache_path = os.path.join(OUTPUT_DIR, 'tg_processed_data.pt')
    if os.path.exists(cache_path):
        logger.info("Loading cached processed data...")
        cached = torch.load(cache_path, weights_only=False)
        return cached['monomers'], cached['trimers'], cached['targets'], cached['smiles']

    monomers = []
    trimers = []
    targets = []
    smiles_list = []
    n_fail_mono = 0
    n_fail_tri = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Tg"):
        smi = row[smiles_col]
        tg = row[tg_col]

        mono_mol = psmiles_to_monomer_mol(smi)
        tri_mol = psmiles_to_trimer_mol(smi)

        if mono_mol is None:
            n_fail_mono += 1
            continue
        if tri_mol is None:
            n_fail_tri += 1
            continue

        mono_data = mol_to_data(mono_mol)
        tri_data = mol_to_data(tri_mol, target=tg)

        if mono_data is None or tri_data is None:
            continue

        monomers.append(mono_data)
        trimers.append(tri_data)
        targets.append(tg)
        smiles_list.append(smi)

    logger.info(f"Tg dataset: {len(monomers)} polymers loaded")
    logger.info(f"  Failed monomers: {n_fail_mono}, Failed trimers: {n_fail_tri}")

    # Cache
    torch.save({
        'monomers': monomers,
        'trimers': trimers,
        'targets': targets,
        'smiles': smiles_list,
    }, cache_path)

    return monomers, trimers, targets, smiles_list


def get_data_splits(n_data, seed=SPLIT_SEED, train_ratio=0.72, val_ratio=0.08):
    """Standard 72/8/20 split."""
    indices = list(range(n_data))
    n_train = int(n_data * train_ratio)
    n_val = int(n_data * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ===================================================================
# Training & Evaluation
# ===================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, h_pre_lookup=None,
                    is_dual_encoder=True, h_pre_tensor=None):
    """Train for one epoch."""
    model.train()
    # Ensure encoder P stays frozen
    if is_dual_encoder and hasattr(model, 'encoder_p'):
        model.encoder_p.eval()

    total_loss = 0
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if is_dual_encoder and h_pre_lookup is not None:
            # Get h_pre for this batch
            batch_indices = batch.polymer_idx.squeeze().tolist()
            if isinstance(batch_indices, int):
                batch_indices = [batch_indices]
            h_pre_batch = h_pre_tensor[batch_indices].to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, h_pre_batch)
        else:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        loss = criterion(pred, batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs

    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, device, h_pre_tensor=None, is_dual_encoder=True):
    """Evaluate model, return RMSE and predictions."""
    model.eval()
    preds = []
    targets = []

    for batch in loader:
        batch = batch.to(device)

        if is_dual_encoder and h_pre_tensor is not None:
            batch_indices = batch.polymer_idx.squeeze().tolist()
            if isinstance(batch_indices, int):
                batch_indices = [batch_indices]
            h_pre_batch = h_pre_tensor[batch_indices].to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, h_pre_batch)
        else:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        preds.append(pred.cpu())
        targets.append(batch.y.squeeze().cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    rmse = torch.sqrt(F.mse_loss(preds, targets)).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    return rmse, mae, preds, targets


def run_training(model, train_loader, val_loader, test_loader, device,
                 h_pre_tensor=None, is_dual_encoder=True,
                 seed=42, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 lr=LR, tag="model"):
    """Full training loop with early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            h_pre_lookup=is_dual_encoder, is_dual_encoder=is_dual_encoder,
            h_pre_tensor=h_pre_tensor
        )
        scheduler.step()

        train_rmse, _, _, _ = evaluate(model, train_loader, device, h_pre_tensor, is_dual_encoder)
        val_rmse, _, _, _ = evaluate(model, val_loader, device, h_pre_tensor, is_dual_encoder)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"[{tag}] Epoch {epoch}: train_rmse={train_rmse:.4f}, "
                       f"val_rmse={val_rmse:.4f}, best_val={best_val_rmse:.4f}")

        if patience_counter >= patience:
            logger.info(f"[{tag}] Early stopping at epoch {epoch}, best epoch={best_epoch}")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    train_rmse, train_mae, _, _ = evaluate(model, train_loader, device, h_pre_tensor, is_dual_encoder)
    val_rmse, val_mae, _, _ = evaluate(model, val_loader, device, h_pre_tensor, is_dual_encoder)
    test_rmse, test_mae, test_preds, test_targets = evaluate(
        model, test_loader, device, h_pre_tensor, is_dual_encoder)

    result = {
        'seed': seed,
        'best_epoch': best_epoch,
        'total_epochs': epoch,
        'train_rmse': float(train_rmse),
        'val_rmse': float(val_rmse),
        'test_rmse': float(test_rmse),
        'train_mae': float(train_mae),
        'val_mae': float(val_mae),
        'test_mae': float(test_mae),
        'overfitting_gap': float(train_rmse - test_rmse),
        'history': history,
    }

    logger.info(f"[{tag}] Seed={seed}: test_rmse={test_rmse:.4f}, "
                f"train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, "
                f"gap={train_rmse - test_rmse:.4f}")

    return result, model


# ===================================================================
# Custom Data class for polymer index tracking
# ===================================================================
class IndexedData(Data):
    """PyG Data subclass that properly handles polymer index for batching."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.polymer_idx = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'polymer_idx':
            return 0  # Don't increment - these are global indices
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'polymer_idx':
            return 0  # Concatenate along dim 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def wrap_data_with_index(data, idx):
    """Wrap a PyG Data object with a polymer index."""
    indexed = IndexedData(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        num_nodes=data.num_nodes,
    )
    if hasattr(data, 'y'):
        indexed.y = data.y
    indexed.polymer_idx = torch.tensor([idx], dtype=torch.long)
    return indexed


def make_loader(data_list, indices, batch_size=BATCH_SIZE, shuffle=False):
    """Create DataLoader with indexed dataset."""
    indexed_list = []
    for i, (data, idx) in enumerate(zip(data_list, indices)):
        indexed_list.append(wrap_data_with_index(data, idx))
    return DataLoader(indexed_list, batch_size=batch_size, shuffle=shuffle,
                      follow_batch=[])


# ===================================================================
# Phase 2: Compute h_pre Lookup Table
# ===================================================================
def compute_hpre_lookup(encoder_p_state_dict, monomer_data_list, device):
    """Forward all monomers through frozen Encoder P to get h_pre vectors."""
    logger.info("Computing h_pre lookup table...")

    encoder = GINEncoder(dropout=0.0).to(device)
    encoder.load_state_dict(encoder_p_state_dict)
    encoder.eval()

    loader = DataLoader(monomer_data_list, batch_size=64, shuffle=False)
    all_hpre = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing h_pre"):
            batch = batch.to(device)
            h_pre = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_hpre.append(h_pre.cpu())

    h_pre_tensor = torch.cat(all_hpre, dim=0)
    logger.info(f"h_pre lookup table shape: {h_pre_tensor.shape}")

    # Save
    save_path = os.path.join(OUTPUT_DIR, 'h_pre_lookup.pt')
    torch.save(h_pre_tensor, save_path)
    logger.info(f"h_pre lookup saved to {save_path}")

    return h_pre_tensor


# ===================================================================
# Phase 5: Diagnostic Analyses
# ===================================================================
def diagnostic_hpre_quality(h_pre_tensor):
    """Analyze h_pre quality via pairwise cosine similarities."""
    logger.info("Diagnostic: h_pre quality analysis...")

    # Subsample for pairwise computation (full pairwise is O(n^2) for 7170)
    n = h_pre_tensor.shape[0]
    sample_size = min(1000, n)
    indices = np.random.choice(n, sample_size, replace=False)
    h_sample = h_pre_tensor[indices]

    # L2 normalize
    h_norm = F.normalize(h_sample, p=2, dim=-1)
    sim_matrix = torch.mm(h_norm, h_norm.t())

    # Remove diagonal
    mask = ~torch.eye(sample_size, dtype=torch.bool)
    sims = sim_matrix[mask]

    results = {
        'mean_cos_sim': float(sims.mean()),
        'std_cos_sim': float(sims.std()),
        'min_cos_sim': float(sims.min()),
        'max_cos_sim': float(sims.max()),
        'median_cos_sim': float(sims.median()),
        'pct_above_095': float((sims > 0.95).float().mean() * 100),
        'pct_above_090': float((sims > 0.90).float().mean() * 100),
        'pct_below_050': float((sims < 0.50).float().mean() * 100),
        'sample_size': sample_size,
        'total_polymers': n,
    }

    logger.info(f"  h_pre cos_sim: mean={results['mean_cos_sim']:.4f}, "
                f"std={results['std_cos_sim']:.4f}, "
                f"min={results['min_cos_sim']:.4f}, max={results['max_cos_sim']:.4f}")

    save_path = os.path.join(OUTPUT_DIR, 'diagnostic_hpre_quality.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def diagnostic_htask_hpre_divergence(pcft_model, train_loader, val_loader,
                                      device, h_pre_tensor, n_epochs=20):
    """Track cos_sim(h_task, h_pre) divergence across training."""
    logger.info("Diagnostic: h_task vs h_pre divergence tracking...")

    # We'll track divergence for a fixed sample of the validation set
    divergences = {f'epoch_{e}': [] for e in range(0, n_epochs + 1)}

    # Get a batch from val set
    sample_batch = None
    for batch in val_loader:
        sample_batch = batch.to(device)
        break

    batch_indices = sample_batch.polymer_idx.squeeze().tolist()
    if isinstance(batch_indices, int):
        batch_indices = [batch_indices]
    h_pre_sample = h_pre_tensor[batch_indices].to(device)

    # Record initial divergence
    pcft_model.eval()
    with torch.no_grad():
        h_task = pcft_model.encoder_t(
            sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr, sample_batch.batch)
        cos_sim = F.cosine_similarity(h_task, h_pre_sample, dim=1)
        divergences['epoch_0'] = {
            'mean': float(cos_sim.mean()),
            'std': float(cos_sim.std()),
            'min': float(cos_sim.min()),
            'max': float(cos_sim.max()),
        }
        logger.info(f"  Epoch 0: cos_sim(h_task, h_pre) = {cos_sim.mean():.4f}")

    # Continue training and track
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, pcft_model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        pcft_model.train()
        pcft_model.encoder_p.eval()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            batch_idx = batch.polymer_idx.squeeze().tolist()
            if isinstance(batch_idx, int):
                batch_idx = [batch_idx]
            h_pre_batch = h_pre_tensor[batch_idx].to(device)

            pred = pcft_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, h_pre_batch)
            loss = criterion(pred, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pcft_model.parameters(), 1.0)
            optimizer.step()

        # Measure divergence
        pcft_model.eval()
        with torch.no_grad():
            h_task = pcft_model.encoder_t(
                sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr, sample_batch.batch)
            cos_sim = F.cosine_similarity(h_task, h_pre_sample, dim=1)
            divergences[f'epoch_{epoch}'] = {
                'mean': float(cos_sim.mean()),
                'std': float(cos_sim.std()),
                'min': float(cos_sim.min()),
                'max': float(cos_sim.max()),
            }

        if epoch % 5 == 0:
            logger.info(f"  Epoch {epoch}: cos_sim(h_task, h_pre) = {cos_sim.mean():.4f}")

    save_path = os.path.join(OUTPUT_DIR, 'diagnostic_htask_hpre_divergence.json')
    with open(save_path, 'w') as f:
        json.dump(divergences, f, indent=2)

    return divergences


def diagnostic_prediction_head_weights(pcft_model):
    """Analyze prediction head weight reliance on h_task vs h_pre."""
    logger.info("Diagnostic: Prediction head weight analysis...")

    # First linear layer: weight shape [256, 512]
    # First 256 input dims = h_task, last 256 = h_pre
    first_layer = pcft_model.pred_head[0]  # Linear(512, 256)

    weight = first_layer.weight.data  # [256, 512]
    w_task = weight[:, :256]  # weights for h_task
    w_pre = weight[:, 256:]  # weights for h_pre

    results = {
        'w_task_norm': float(torch.norm(w_task).item()),
        'w_pre_norm': float(torch.norm(w_pre).item()),
        'w_task_mean_abs': float(torch.abs(w_task).mean().item()),
        'w_pre_mean_abs': float(torch.abs(w_pre).mean().item()),
        'w_task_max_abs': float(torch.abs(w_task).max().item()),
        'w_pre_max_abs': float(torch.abs(w_pre).max().item()),
        'ratio_pre_to_task': float(torch.norm(w_pre).item() / (torch.norm(w_task).item() + 1e-8)),
    }

    logger.info(f"  ||W_task||={results['w_task_norm']:.4f}, ||W_pre||={results['w_pre_norm']:.4f}, "
                f"ratio={results['ratio_pre_to_task']:.4f}")

    save_path = os.path.join(OUTPUT_DIR, 'diagnostic_prediction_head_weights.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ===================================================================
# Main Experiment Runner
# ===================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    all_results = {}

    # ---- Load dataset ----
    monomers, trimers, targets, smiles_list = load_tg_dataset()
    n_data = len(monomers)
    train_idx, val_idx, test_idx = get_data_splits(n_data)
    logger.info(f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ---- Phase 1: Load pre-trained encoder ----
    ckpt_path = os.path.join(PERIOGT_DIR, 'pretrained_encoder_edge_drop_feat_mask_50.pt')
    if os.path.exists(ckpt_path):
        logger.info(f"Loading pre-trained encoder from {ckpt_path}")
        encoder_p_state_dict = torch.load(ckpt_path, map_location='cpu')
    else:
        logger.warning("No pre-trained checkpoint found! Using random initialization.")
        encoder_p_state_dict = None

    # Copy checkpoint to our output dir
    if encoder_p_state_dict is not None:
        save_path = os.path.join(OUTPUT_DIR, 'pretrain_checkpoint.pt')
        torch.save(encoder_p_state_dict, save_path)
        logger.info(f"Checkpoint copied to {save_path}")

    # ---- Phase 2: Compute h_pre lookup table ----
    h_pre_path = os.path.join(OUTPUT_DIR, 'h_pre_lookup.pt')
    if os.path.exists(h_pre_path):
        logger.info("Loading cached h_pre lookup table...")
        h_pre_tensor = torch.load(h_pre_path, weights_only=False)
    else:
        if encoder_p_state_dict is not None:
            h_pre_tensor = compute_hpre_lookup(encoder_p_state_dict, monomers, device)
        else:
            # Random encoder -> random h_pre
            h_pre_tensor = torch.randn(n_data, HIDDEN_DIM)

    assert h_pre_tensor.shape[0] == n_data, \
        f"h_pre size mismatch: {h_pre_tensor.shape[0]} vs {n_data}"

    # ---- Prepare data loaders ----
    train_tri = [trimers[i] for i in train_idx]
    val_tri = [trimers[i] for i in val_idx]
    test_tri = [trimers[i] for i in test_idx]

    train_mono = [monomers[i] for i in train_idx]
    val_mono = [monomers[i] for i in val_idx]
    test_mono = [monomers[i] for i in test_idx]

    # ---- Phase 3-4: Ablation Experiments ----
    # =========================================================
    # ABLATION (a): GIN+Trimer Baseline (no pre-training, single encoder)
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION (a): GIN+Trimer Baseline (random init, single encoder, mean_pool)")
    logger.info("=" * 60)

    ablation_a_results = []
    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = make_loader(train_tri, train_idx, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_tri, val_idx, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = make_loader(test_tri, test_idx, batch_size=BATCH_SIZE, shuffle=False)

        model = BaselineGINModel(pretrained_state_dict=None, freeze_encoder=False).to(device)
        result, _ = run_training(
            model, train_loader, val_loader, test_loader, device,
            h_pre_tensor=None, is_dual_encoder=False,
            seed=seed, tag=f"ablation_a_seed{seed}"
        )
        ablation_a_results.append(result)

    all_results['ablation_a'] = ablation_a_results
    save_results(ablation_a_results, 'results_ablation_a.json')

    # =========================================================
    # ABLATION (b): PCFT with Random-Init Encoder P
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION (b): PCFT with Random-Init Encoder P")
    logger.info("=" * 60)

    # Compute h_pre from random encoder
    random_encoder = GINEncoder(dropout=0.0)
    random_state = random_encoder.state_dict()
    h_pre_random = compute_hpre_lookup(random_state, monomers, device)

    ablation_b_results = []
    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = make_loader(train_tri, train_idx, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_tri, val_idx, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = make_loader(test_tri, test_idx, batch_size=BATCH_SIZE, shuffle=False)

        # Random init for BOTH encoders
        random_seed_state = GINEncoder(dropout=0.0).state_dict()
        model = PCFTModel(
            encoder_p_state_dict=random_seed_state,
            freeze_p=True,
            dropout_t=DROPOUT,
        ).to(device)

        result, _ = run_training(
            model, train_loader, val_loader, test_loader, device,
            h_pre_tensor=h_pre_random, is_dual_encoder=True,
            seed=seed, tag=f"ablation_b_seed{seed}"
        )
        ablation_b_results.append(result)

    all_results['ablation_b'] = ablation_b_results
    save_results(ablation_b_results, 'results_ablation_b.json')

    # =========================================================
    # ABLATION (c): PCFT with PA Pre-trained Encoder P [PRIMARY]
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION (c): PCFT with PA Pre-trained Encoder P [PRIMARY]")
    logger.info("=" * 60)

    ablation_c_results = []
    trained_models_c = []
    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = make_loader(train_tri, train_idx, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_tri, val_idx, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = make_loader(test_tri, test_idx, batch_size=BATCH_SIZE, shuffle=False)

        model = PCFTModel(
            encoder_p_state_dict=encoder_p_state_dict,
            freeze_p=True,
            dropout_t=DROPOUT,
        ).to(device)

        result, trained_model = run_training(
            model, train_loader, val_loader, test_loader, device,
            h_pre_tensor=h_pre_tensor, is_dual_encoder=True,
            seed=seed, tag=f"ablation_c_seed{seed}"
        )
        ablation_c_results.append(result)
        trained_models_c.append(trained_model)

    all_results['ablation_c'] = ablation_c_results
    save_results(ablation_c_results, 'results_ablation_c.json')

    # =========================================================
    # ABLATION (d): Standard Fine-tuning (Negative Control)
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION (d): Standard Fine-tuning (pre-trained, end-to-end, no prompt)")
    logger.info("=" * 60)

    ablation_d_results = []
    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = make_loader(train_tri, train_idx, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_tri, val_idx, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = make_loader(test_tri, test_idx, batch_size=BATCH_SIZE, shuffle=False)

        model = BaselineGINModel(
            pretrained_state_dict=encoder_p_state_dict,
            freeze_encoder=False,
            dropout=0.0,
        ).to(device)

        result, _ = run_training(
            model, train_loader, val_loader, test_loader, device,
            h_pre_tensor=None, is_dual_encoder=False,
            seed=seed, tag=f"ablation_d_seed{seed}"
        )
        ablation_d_results.append(result)

    all_results['ablation_d'] = ablation_d_results
    save_results(ablation_d_results, 'results_ablation_d.json')

    # =========================================================
    # ABLATION (e): PCFT with Both Encoders Frozen
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION (e): PCFT with Both Encoders Frozen (only head trainable)")
    logger.info("=" * 60)

    ablation_e_results = []
    for seed in TRAINING_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = make_loader(train_tri, train_idx, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_tri, val_idx, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = make_loader(test_tri, test_idx, batch_size=BATCH_SIZE, shuffle=False)

        # Both encoders frozen, initialized from pre-trained weights
        model = PCFTModel(
            encoder_p_state_dict=encoder_p_state_dict,
            freeze_p=True,
            dropout_t=0.0,  # No dropout since frozen
        ).to(device)

        # Also freeze Encoder T
        for param in model.encoder_t.parameters():
            param.requires_grad = False

        # Re-initialize Encoder T from pre-trained (it will stay frozen)
        model.encoder_t.load_state_dict(deepcopy(encoder_p_state_dict))

        result, _ = run_training(
            model, train_loader, val_loader, test_loader, device,
            h_pre_tensor=h_pre_tensor, is_dual_encoder=True,
            seed=seed, tag=f"ablation_e_seed{seed}"
        )
        ablation_e_results.append(result)

    all_results['ablation_e'] = ablation_e_results
    save_results(ablation_e_results, 'results_ablation_e.json')

    # ---- Phase 5: Diagnostic Analyses ----
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 5: Diagnostic Analyses")
    logger.info("=" * 60)

    # 1. h_pre quality
    hpre_diag = diagnostic_hpre_quality(h_pre_tensor)

    # 2. h_task vs h_pre divergence
    # Use the best model from ablation (c) seed 42
    best_model_c = trained_models_c[0]  # seed=42 model
    train_loader_diag = make_loader(train_tri, train_idx, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_diag = make_loader(val_tri, val_idx, batch_size=BATCH_SIZE, shuffle=False)
    divergence_diag = diagnostic_htask_hpre_divergence(
        best_model_c, train_loader_diag, val_loader_diag, device, h_pre_tensor, n_epochs=20)

    # 3. Prediction head weight analysis (from best model)
    weight_diag = diagnostic_prediction_head_weights(best_model_c)

    # ---- Generate Summary Report ----
    generate_summary_report(all_results, hpre_diag, divergence_diag, weight_diag)

    logger.info("\nExperiment complete!")
    return all_results


def save_results(results, filename):
    """Save results to JSON."""
    # Remove history to keep file size manageable
    results_clean = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != 'history'}
        results_clean.append(r_copy)

    save_path = os.path.join(OUTPUT_DIR, filename)
    with open(save_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    logger.info(f"Results saved to {save_path}")


def summarize_seeds(results):
    """Compute mean±std across seeds for a set of results."""
    test_rmses = [r['test_rmse'] for r in results]
    train_rmses = [r['train_rmse'] for r in results]
    val_rmses = [r['val_rmse'] for r in results]
    gaps = [r['overfitting_gap'] for r in results]

    return {
        'test_rmse_mean': float(np.mean(test_rmses)),
        'test_rmse_std': float(np.std(test_rmses)),
        'train_rmse_mean': float(np.mean(train_rmses)),
        'train_rmse_std': float(np.std(train_rmses)),
        'val_rmse_mean': float(np.mean(val_rmses)),
        'val_rmse_std': float(np.std(val_rmses)),
        'gap_mean': float(np.mean(gaps)),
        'gap_std': float(np.std(gaps)),
        'seeds': [r['seed'] for r in results],
    }


def generate_summary_report(all_results, hpre_diag, divergence_diag, weight_diag):
    """Generate comprehensive summary report."""
    logger.info("Generating summary report...")

    # Summarize each ablation
    summaries = {}
    for key in ['ablation_a', 'ablation_b', 'ablation_c', 'ablation_d', 'ablation_e']:
        if key in all_results:
            summaries[key] = summarize_seeds(all_results[key])

    # Statistical significance tests (paired t-test)
    from scipy import stats as scipy_stats

    def paired_ttest(results1, results2):
        vals1 = [r['test_rmse'] for r in results1]
        vals2 = [r['test_rmse'] for r in results2]
        if len(vals1) >= 2 and len(vals2) >= 2:
            t_stat, p_value = scipy_stats.ttest_rel(vals1, vals2)
            return float(t_stat), float(p_value)
        return None, None

    comparisons = {}
    pairs = [
        ('c_vs_a', 'ablation_c', 'ablation_a'),
        ('c_vs_d', 'ablation_c', 'ablation_d'),
        ('b_vs_a', 'ablation_b', 'ablation_a'),
        ('c_vs_b', 'ablation_c', 'ablation_b'),
        ('c_vs_e', 'ablation_c', 'ablation_e'),
    ]
    for comp_name, key1, key2 in pairs:
        if key1 in all_results and key2 in all_results:
            t, p = paired_ttest(all_results[key1], all_results[key2])
            comparisons[comp_name] = {
                't_statistic': t,
                'p_value': p,
                'significant_005': p < 0.05 if p is not None else None,
            }

    # Build report text
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PCFT Experiment Summary Report")
    report_lines.append("Day 4 Experiment 1: Prompt-Conditioned Fine-Tuning")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append("ABLATION RESULTS (Test RMSE, mean ± std across 3 seeds)")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Ablation':<45} {'Test RMSE':>12} {'Train RMSE':>12} {'Gap':>10}")
    report_lines.append("-" * 80)

    labels = {
        'ablation_a': '(a) GIN+Trimer Baseline (no pretrain)',
        'ablation_b': '(b) PCFT Random-Init Encoder P',
        'ablation_c': '(c) PCFT PA Pre-trained Encoder P [PRIMARY]',
        'ablation_d': '(d) Standard Fine-tuning (no prompt)',
        'ablation_e': '(e) PCFT Both Encoders Frozen',
    }

    for key, label in labels.items():
        if key in summaries:
            s = summaries[key]
            report_lines.append(
                f"{label:<45} {s['test_rmse_mean']:.2f}±{s['test_rmse_std']:.2f} "
                f"{s['train_rmse_mean']:.2f}±{s['train_rmse_std']:.2f} "
                f"{s['gap_mean']:.2f}±{s['gap_std']:.2f}"
            )

    report_lines.append("-" * 80)
    report_lines.append("")

    report_lines.append("STATISTICAL SIGNIFICANCE (Paired t-test on test RMSE)")
    report_lines.append("-" * 60)
    for comp_name, comp in comparisons.items():
        sig = "YES" if comp.get('significant_005') else "NO"
        report_lines.append(
            f"  {comp_name}: t={comp['t_statistic']:.4f}, "
            f"p={comp['p_value']:.4f}, significant={sig}"
        )
    report_lines.append("")

    report_lines.append("DIAGNOSTIC: h_pre QUALITY")
    report_lines.append("-" * 60)
    report_lines.append(f"  Mean pairwise cosine similarity: {hpre_diag['mean_cos_sim']:.4f}")
    report_lines.append(f"  Std pairwise cosine similarity:  {hpre_diag['std_cos_sim']:.4f}")
    report_lines.append(f"  Min cosine similarity:           {hpre_diag['min_cos_sim']:.4f}")
    report_lines.append(f"  Max cosine similarity:           {hpre_diag['max_cos_sim']:.4f}")
    report_lines.append(f"  % pairs with cos_sim > 0.95:     {hpre_diag['pct_above_095']:.2f}%")
    report_lines.append(f"  % pairs with cos_sim > 0.90:     {hpre_diag['pct_above_090']:.2f}%")
    report_lines.append(f"  % pairs with cos_sim < 0.50:     {hpre_diag['pct_below_050']:.2f}%")
    report_lines.append("")

    report_lines.append("DIAGNOSTIC: h_task vs h_pre DIVERGENCE")
    report_lines.append("-" * 60)
    for key in sorted(divergence_diag.keys()):
        d = divergence_diag[key]
        report_lines.append(f"  {key}: mean_cos_sim={d['mean']:.4f}, std={d['std']:.4f}")
    report_lines.append("")

    report_lines.append("DIAGNOSTIC: PREDICTION HEAD WEIGHT ANALYSIS")
    report_lines.append("-" * 60)
    report_lines.append(f"  ||W_task|| = {weight_diag['w_task_norm']:.4f}")
    report_lines.append(f"  ||W_pre||  = {weight_diag['w_pre_norm']:.4f}")
    report_lines.append(f"  Ratio (pre/task) = {weight_diag['ratio_pre_to_task']:.4f}")
    report_lines.append(f"  Mean |W_task| = {weight_diag['w_task_mean_abs']:.6f}")
    report_lines.append(f"  Mean |W_pre|  = {weight_diag['w_pre_mean_abs']:.6f}")
    report_lines.append("")

    report_lines.append("CONCLUSIONS")
    report_lines.append("-" * 60)

    # Auto-generate conclusions
    if 'ablation_c' in summaries and 'ablation_a' in summaries:
        c_rmse = summaries['ablation_c']['test_rmse_mean']
        a_rmse = summaries['ablation_a']['test_rmse_mean']
        d_rmse = summaries['ablation_d']['test_rmse_mean'] if 'ablation_d' in summaries else None
        b_rmse = summaries['ablation_b']['test_rmse_mean'] if 'ablation_b' in summaries else None

        report_lines.append(f"1. PCFT (c) test RMSE: {c_rmse:.2f}")
        report_lines.append(f"   Baseline (a) test RMSE: {a_rmse:.2f}")
        if c_rmse < a_rmse:
            report_lines.append(f"   -> PCFT IMPROVES over baseline by {a_rmse - c_rmse:.2f} RMSE")
            report_lines.append(f"   -> H4a SUPPORTED: Making periodicity active conditioning bridges usage gap")
        else:
            report_lines.append(f"   -> PCFT DOES NOT improve over baseline (diff = {c_rmse - a_rmse:.2f})")
            report_lines.append(f"   -> H4a NOT SUPPORTED: Prompt mechanism alone insufficient")

        if d_rmse is not None:
            report_lines.append(f"   Standard FT (d) test RMSE: {d_rmse:.2f}")
            if c_rmse < d_rmse:
                report_lines.append(f"   -> PCFT IMPROVES over standard FT by {d_rmse - c_rmse:.2f} RMSE")
            else:
                report_lines.append(f"   -> PCFT DOES NOT improve over standard FT (diff = {c_rmse - d_rmse:.2f})")

        if b_rmse is not None:
            report_lines.append(f"   Random-Init PCFT (b) test RMSE: {b_rmse:.2f}")
            if c_rmse < b_rmse:
                report_lines.append(f"   -> PA pre-training helps: {b_rmse - c_rmse:.2f} improvement over random prompt")
            else:
                report_lines.append(f"   -> PA pre-training does NOT help (diff = {c_rmse - b_rmse:.2f})")

        if b_rmse is not None and a_rmse is not None:
            if b_rmse < a_rmse:
                report_lines.append(f"   -> Dual-encoder structure alone helps even without pre-training (b vs a: {a_rmse - b_rmse:.2f})")
                report_lines.append(f"   -> H4b PARTIALLY SUPPORTED: Complementary information from dual channels")
            else:
                report_lines.append(f"   -> Dual-encoder structure alone does NOT help (b vs a: {b_rmse - a_rmse:.2f})")

    report_lines.append("")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    save_path = os.path.join(OUTPUT_DIR, 'summary_report.txt')
    with open(save_path, 'w') as f:
        f.write(report_text)

    logger.info(f"\n{report_text}")
    logger.info(f"Summary report saved to {save_path}")


if __name__ == '__main__':
    results = main()

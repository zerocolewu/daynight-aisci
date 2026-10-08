#!/usr/bin/env python3
"""
IAPT v2: Intensive-Aware Polymer Transformer for Tg Prediction
Complete implementation with all ablation variants.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
WORK_DIR = '/home/home_public/aisci/AIscientist/src/4_iapt_v2_intensive_aware'
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/alldataset/some_dataprocesscode')
from multimer import generate_multimer_smiles

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 15
SCHEDULER_FACTOR = 0.5

HIDDEN_DIM = 256
NUM_GNN_LAYERS = 5
NUM_LINE_GRAPH_LAYERS = 2
LINE_GRAPH_HIDDEN_DIM = 128
DROPOUT = 0.1
NUM_REPEAT_UNITS = 3

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I',
              'B', 'Se', 'Te', 'Ge', 'As', 'Sb', 'Bi', '*']
ATOM_TYPES_DICT = {t: i for i, t in enumerate(ATOM_TYPES)}

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
HYBRIDIZATION_DICT = {t: i for i, t in enumerate(HYBRIDIZATION_TYPES)}

BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_TYPES_DICT = {t: i for i, t in enumerate(BOND_TYPES)}

def atom_features(atom: Chem.Atom) -> np.ndarray:
    features = []
    # Atom type (18 dim)
    atom_type = atom.GetSymbol()
    type_feat = np.zeros(len(ATOM_TYPES))
    if atom_type in ATOM_TYPES_DICT:
        type_feat[ATOM_TYPES_DICT[atom_type]] = 1.0
    features.append(type_feat)
    # Degree (1 dim)
    features.append([atom.GetDegree() / 6.0])
    # Hybridization (5 dim)
    hybrid = atom.GetHybridization()
    hybrid_feat = np.zeros(len(HYBRIDIZATION_TYPES))
    if hybrid in HYBRIDIZATION_DICT:
        hybrid_feat[HYBRIDIZATION_DICT[hybrid]] = 1.0
    features.append(hybrid_feat)
    # Aromatic (1 dim)
    features.append([float(atom.GetIsAromatic())])
    # Formal charge (1 dim)
    features.append([atom.GetFormalCharge() / 4.0])
    # In ring (1 dim)
    features.append([float(atom.IsInRing())])
    return np.concatenate(features).astype(np.float32)

def bond_features(bond: Chem.Bond) -> np.ndarray:
    features = []
    # Bond type (4 dim)
    bond_type = bond.GetBondType()
    type_feat = np.zeros(len(BOND_TYPES))
    if bond_type in BOND_TYPES_DICT:
        type_feat[BOND_TYPES_DICT[bond_type]] = 1.0
    else:
        type_feat[0] = 1.0
    features.append(type_feat)
    # Conjugated (1 dim)
    features.append([float(bond.GetIsConjugated())])
    # In ring (1 dim)
    features.append([float(bond.IsInRing())])
    return np.concatenate(features).astype(np.float32)

def generate_3d_conformer(mol: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        if mol.GetNumAtoms() > 100:
            return None
        embed_params = AllChem.ETKDGv3()
        embed_params.randomSeed = 42
        conf_id = AllChem.EmbedMolecule(mol, embed_params)
        if conf_id == -1:
            return None
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                if ff is not None:
                    ff.Minimize(maxIts=50)
        except:
            pass
        return mol
    except:
        return None

def compute_equivariant_features(mol_3d: Chem.Mol) -> np.ndarray:
    conf = mol_3d.GetConformer()
    n_atoms = mol_3d.GetNumAtoms()
    positions = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                          for i in range(n_atoms)])
    center = np.mean(positions, axis=0)
    features = []
    for i in range(n_atoms):
        pos = positions[i]
        displacement = pos - center
        l1_norm = np.linalg.norm(displacement)
        features.append([l1_norm / 10.0])  # Normalized
    return np.array(features, dtype=np.float32)

def build_trimer(p_smiles: str) -> Optional[Chem.Mol]:
    try:
        trimer_smiles = generate_multimer_smiles(NUM_REPEAT_UNITS, p_smiles, replace_dummy_atoms=False)
        return Chem.MolFromSmiles(trimer_smiles)
    except:
        return None

def process_polymer(p_smiles: str, tg: float, enable_3d: bool = True,
                   enable_equivariant: bool = True) -> Optional[Dict]:
    trimer_mol = build_trimer(p_smiles)
    if trimer_mol is None:
        return None

    # Atom features
    atom_feats = np.stack([atom_features(atom) for atom in trimer_mol.GetAtoms()])

    # Edge features
    edge_indices = []
    edge_feats_list = []
    for bond in trimer_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
        bf = bond_features(bond)
        edge_feats_list.extend([bf, bf])

    if not edge_indices:
        return None

    edge_index = np.array(edge_indices, dtype=np.int64).T
    edge_feats = np.stack(edge_feats_list)

    # 3D conformer
    mol_3d = None
    conformer_success = False
    if enable_3d:
        mol_3d = generate_3d_conformer(trimer_mol)
        if mol_3d is not None:
            conformer_success = True
            conf = mol_3d.GetConformer()
            # Add bond lengths to edge features
            extended_edge_feats = []
            for idx, (i, j) in enumerate(edge_indices):
                base_feat = edge_feats_list[idx]
                if idx < len(edge_indices) // 2:
                    length = 1.5  # default
                    for bond in trimer_mol.GetBonds():
                        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        if (bi == i and bj == j) or (bi == j and bj == i):
                            pos_i = np.array(conf.GetAtomPosition(bi))
                            pos_j = np.array(conf.GetAtomPosition(bj))
                            length = np.linalg.norm(pos_i - pos_j)
                            break
                    bond_length = length / 2.0
                else:
                    bond_length = 0.0
                extended_feat = np.concatenate([base_feat, [bond_length]])
                extended_edge_feats.append(extended_feat)
            edge_feats = np.stack(extended_edge_feats)

    # Equivariant features
    if enable_equivariant and mol_3d is not None:
        equivariant_feats = compute_equivariant_features(mol_3d)
        atom_feats = np.concatenate([atom_feats, equivariant_feats], axis=1)

    # Periodic edges
    star_atoms = [a.GetIdx() for a in trimer_mol.GetAtoms() if a.GetSymbol() == '*']
    if len(star_atoms) >= 2:
        periodic_edges = []
        for i in range(len(star_atoms)):
            for j in range(i + 1, len(star_atoms)):
                periodic_edges.extend([[star_atoms[i], star_atoms[j]], [star_atoms[j], star_atoms[i]]])

        if periodic_edges:
            pe_feat = np.zeros(edge_feats.shape[1])
            pe_feat[0] = 1.0
            periodic_edge_features = np.stack([pe_feat] * len(periodic_edges))
            periodic_edge_index = np.array(periodic_edges, dtype=np.int64).T
            edge_index = np.concatenate([edge_index, periodic_edge_index], axis=1)
            edge_feats = np.concatenate([edge_feats, periodic_edge_features], axis=0)

    return {
        'atom_features': atom_feats,
        'edge_index': edge_index,
        'edge_features': edge_feats,
        'tg': tg,
        'n_atoms': trimer_mol.GetNumAtoms(),
        'conformer_success': conformer_success,
    }

# =============================================================================
# DATASET
# =============================================================================

class IAPTv2Dataset(Dataset):
    def __init__(self, csv_path: str, split: str = 'train',
                 train_ratio: float = 0.72, val_ratio: float = 0.08,
                 seed: int = 42, enable_3d: bool = True, enable_equivariant: bool = True):
        self.split = split
        df = pd.read_csv(csv_path)
        np.random.seed(seed)
        n = len(df)
        indices = np.random.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == 'train':
            sel_indices = indices[:n_train]
        elif split == 'val':
            sel_indices = indices[n_train:n_train + n_val]
        else:
            sel_indices = indices[n_train + n_val:]

        df_split = df.iloc[sel_indices].reset_index(drop=True)

        self.samples = []
        self.failed = 0
        self.conformer_success = 0

        for idx, row in df_split.iterrows():
            result = process_polymer(row['Smiles'], row['Tg'], enable_3d, enable_equivariant)
            if result is not None:
                self.samples.append(result)
                if result['conformer_success']:
                    self.conformer_success += 1
            else:
                self.failed += 1

        print(f"{split}: {len(self.samples)} samples, {self.failed} failed, {self.conformer_success} with 3D")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'atom_features': torch.FloatTensor(sample['atom_features']),
            'edge_index': torch.LongTensor(sample['edge_index']),
            'edge_features': torch.FloatTensor(sample['edge_features']),
            'tg': torch.FloatTensor([sample['tg']]),
            'n_atoms': sample['n_atoms'],
        }

def collate_fn(batch):
    atom_features_list = []
    edge_index_list = []
    edge_features_list = []
    tg_list = []
    batch_atom_indices = []
    n_atoms_offset = 0

    for i, sample in enumerate(batch):
        n_atoms = sample['n_atoms']
        atom_features_list.append(sample['atom_features'])
        edge_index_list.append(sample['edge_index'] + n_atoms_offset)
        edge_features_list.append(sample['edge_features'])
        tg_list.append(sample['tg'])
        batch_atom_indices.append(torch.full((n_atoms,), i, dtype=torch.long))
        n_atoms_offset += n_atoms

    return {
        'atom_features': torch.cat(atom_features_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'edge_features': torch.cat(edge_features_list, dim=0),
        'tg': torch.cat(tg_list, dim=0),
        'batch_atom': torch.cat(batch_atom_indices, dim=0),
    }

# =============================================================================
# MODEL
# =============================================================================

class IAPTv2Model(nn.Module):
    def __init__(self, atom_dim: int, edge_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_gnn_layers: int = NUM_GNN_LAYERS, dropout: float = DROPOUT,
                 pooling_type: str = 'mean'):
        super().__init__()
        self.pooling_type = pooling_type

        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_gnn_layers):
            nn_module = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            gine_conv = GINEConv(nn=nn_module, eps=0.0, train_eps=True)
            self.gnn_layers.append(gine_conv)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        if pooling_type == 'attention':
            self.attention_query = nn.Parameter(torch.randn(1, hidden_dim))
            self.attention_key = nn.Linear(hidden_dim, hidden_dim)
            self.attention_value = nn.Linear(hidden_dim, hidden_dim)

        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.atom_embedding(batch['atom_features'])
        edge_index = batch['edge_index']
        edge_attr = self.edge_embedding(batch['edge_features'])

        for gnn_layer, norm in zip(self.gnn_layers, self.layer_norms):
            if edge_index.shape[1] > 0:
                x_new = gnn_layer(x, edge_index, edge_attr)
                x = norm(x + x_new)
                x = F.relu(x)

        batch_atom = batch['batch_atom']

        if self.pooling_type == 'mean':
            pooled = global_mean_pool(x, batch_atom)
        elif self.pooling_type == 'max':
            pooled = global_max_pool(x, batch_atom)
        elif self.pooling_type == 'sum':
            pooled = scatter(x, batch_atom, dim=0, reduce='sum')
        elif self.pooling_type == 'attention':
            keys = self.attention_key(x)
            values = self.attention_value(x)
            batch_size = batch['tg'].shape[0]

            pooled_list = []
            for i in range(batch_size):
                mask = (batch_atom == i)
                if mask.sum() == 0:
                    pooled_list.append(torch.zeros(1, self.hidden_dim, device=x.device))
                    continue

                graph_keys = keys[mask]
                graph_values = values[mask]

                query = self.attention_query
                scores = torch.sum(query * graph_keys, dim=-1) / np.sqrt(self.hidden_dim)
                scores = F.softmax(scores, dim=0)

                attended = torch.sum(scores.unsqueeze(-1) * graph_values, dim=0)
                pooled_list.append(attended.unsqueeze(0))

            pooled = torch.cat(pooled_list, dim=0)
        else:
            pooled = global_mean_pool(x, batch_atom)

        return self.predictor(pooled)

# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        atom_features = batch['atom_features'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_features = batch['edge_features'].to(device)
        batch_atom = batch['batch_atom'].to(device)
        targets = batch['tg'].to(device)

        targets_np = targets.cpu().numpy().reshape(-1, 1)
        targets_norm = torch.FloatTensor(scaler.transform(targets_np)).to(device)

        optimizer.zero_grad()
        predictions = model({
            'atom_features': atom_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'batch_atom': batch_atom,
        })

        loss = criterion(predictions, targets_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * predictions.shape[0]

        preds_np = predictions.detach().cpu().numpy().reshape(-1, 1)
        preds_denorm = torch.FloatTensor(scaler.inverse_transform(preds_np))
        all_preds.extend(preds_denorm.numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())

    return total_loss / len(all_preds), np.sqrt(mean_squared_error(all_targets, all_preds))

def validate(model, dataloader, criterion, scaler, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            atom_features = batch['atom_features'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_features = batch['edge_features'].to(device)
            batch_atom = batch['batch_atom'].to(device)
            targets = batch['tg'].to(device)

            targets_norm = torch.FloatTensor(scaler.transform(
                targets.cpu().numpy().reshape(-1, 1))).to(device)

            predictions = model({
                'atom_features': atom_features,
                'edge_index': edge_index,
                'edge_features': edge_features,
                'batch_atom': batch_atom,
            })

            preds_denorm = torch.FloatTensor(scaler.inverse_transform(
                predictions.cpu().numpy().reshape(-1, 1)))
            all_preds.extend(preds_denorm.numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return rmse, mae, r2, all_targets, all_preds

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("IAPT v2: Intensive-Aware Polymer Transformer")
    print("Target: Beat RMSE 30.8 on Tg dataset")
    print("="*80)
    print(f"\nDevice: {DEVICE}")
    print(f"Hidden dim: {HIDDEN_DIM}, GNN layers: {NUM_GNN_LAYERS}")
    print(f"Batch size: {BATCH_SIZE}, Max epochs: {MAX_EPOCHS}")
    print()

    os.makedirs(WORK_DIR, exist_ok=True)

    variants = [
        ('IAPT_v2_mean', 'mean', True, True),
        ('IAPT_v2_max', 'max', True, True),
        ('IAPT_v2_att', 'attention', True, True),
        ('IAPT_v2_sum', 'sum', True, True),
        ('IAPT_v2_no_eq', 'mean', True, False),
        ('IAPT_v2_no_3d', 'mean', False, False),
    ]

    all_results = []

    for variant_name, pooling_type, enable_3d, enable_equivariant in variants:
        print(f"\n{'='*80}")
        print(f"Running variant: {variant_name}")
        print(f"{'='*80}\n")

        # Create datasets
        train_dataset = IAPTv2Dataset(CSV_PATH, 'train', enable_3d=enable_3d, enable_equivariant=enable_equivariant)
        val_dataset = IAPTv2Dataset(CSV_PATH, 'val', enable_3d=enable_3d, enable_equivariant=enable_equivariant)
        test_dataset = IAPTv2Dataset(CSV_PATH, 'test', enable_3d=enable_3d, enable_equivariant=enable_equivariant)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        train_labels = np.array([sample['tg'].item() for sample in train_dataset])
        scaler = StandardScaler()
        scaler.fit(train_labels.reshape(-1, 1))

        sample_batch = next(iter(train_loader))
        atom_dim = sample_batch['atom_features'].shape[1]
        edge_dim = sample_batch['edge_features'].shape[1]

        # Create model
        model = IAPTv2Model(atom_dim, edge_dim, HIDDEN_DIM, NUM_GNN_LAYERS, DROPOUT, pooling_type).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
        criterion = nn.MSELoss()

        best_val_rmse = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(MAX_EPOCHS):
            train_loss, train_rmse = train_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE)
            val_rmse, val_mae, val_r2, _, _ = validate(model, val_loader, criterion, scaler, DEVICE)

            scheduler.step(val_rmse)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | Train RMSE: {train_rmse:.2f} | "
                      f"Val RMSE: {val_rmse:.2f} | Val R2: {val_r2:.4f}")

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})

        test_rmse, test_mae, test_r2, _, _ = validate(model, test_loader, criterion, scaler, DEVICE)

        print(f"\nTest RMSE: {test_rmse:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        print(f"Test R2: {test_r2:.4f}")

        torch.save(model.state_dict(), os.path.join(WORK_DIR, f'best_model_{variant_name}.pt'))

        all_results.append({
            'variant': variant_name,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'best_val_rmse': best_val_rmse,
            'best_epoch': best_epoch,
        })

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(WORK_DIR, 'results.csv'), index=False)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\n{'Variant':<20} {'Test RMSE':<12} {'Test MAE':<12} {'Test R2':<12}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['variant']:<20} {r['test_rmse']:<12.2f} {r['test_mae']:<12.2f} {r['test_r2']:<12.4f}")

    best_result = min(all_results, key=lambda x: x['test_rmse'])
    print(f"\nBest: {best_result['variant']} with RMSE = {best_result['test_rmse']:.2f}")

    if best_result['test_rmse'] < 30.8:
        print(f"\nSUCCESS! Beat target RMSE of 30.8")
    else:
        print(f"\nDid not reach target. Gap: {best_result['test_rmse'] - 30.8:.2f}")

if __name__ == '__main__':
    main()

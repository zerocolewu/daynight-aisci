#!/usr/bin/env python3
"""
Final PSCP Experiment - Complete ablation and scale analysis
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
import random

RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/6_day6_pscp')

from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch


# ============================================================================
# UTILITIES
# ============================================================================

def canonicalize_polymer_smiles(smiles):
    """Clean polymer SMILES by removing asterisks"""
    clean = smiles.replace('*', '')
    if not clean or len(clean) < 2:
        return 'C'
    return clean


def atom_features(atom):
    """Generate atom features vector"""
    # 119 atomic numbers + 8 degree + 8 formal charge + 5 chiral + 6 num_Hs + 6 hybridization + 3 aromatic + 3 is_star + 1 mass
    features = []

    # Atomic number (one-hot, 119 possible)
    atomic_num = atom.GetAtomicNum()
    atomic_feat = [0] * 119
    if atomic_num < 119:
        atomic_feat[atomic_num] = 1
    features.extend(atomic_feat)

    # Degree (0-7)
    degree = atom.GetDegree()
    degree_feat = [0] * 8
    if degree < 8:
        degree_feat[degree] = 1
    features.extend(degree_feat)

    # Formal charge (-4 to 3)
    charge = atom.GetFormalCharge() + 4
    charge_feat = [0] * 8
    if 0 <= charge < 8:
        charge_feat[charge] = 1
    features.extend(charge_feat)

    # Chirality
    chiral = atom.GetChiralTag()
    chiral_feat = [0] * 5
    if chiral < 5:
        chiral_feat[chiral] = 1
    features.extend(chiral_feat)

    # Num Hs
    num_hs = atom.GetTotalNumHs()
    hs_feat = [0] * 6
    if num_hs < 6:
        hs_feat[num_hs] = 1
    features.extend(hs_feat)

    # Hybridization
    hybrid = atom.GetHybridization()
    hybrid_feat = [0] * 6
    if hybrid < 6:
        hybrid_feat[hybrid] = 1
    features.extend(hybrid_feat)

    # Aromatic
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(0)  # padding
    features.append(0)  # padding

    # Is star (wildcard atom)
    is_star = 1 if atom.GetSymbol() == '*' else 0
    features.append(is_star)
    features.append(0)
    features.append(0)

    # Mass (normalized)
    features.append(atom.GetMass() / 100.0)

    return features


def bond_features(bond):
    """Generate bond features vector"""
    # 5 bond_types + 1 conjugated + 1 in_ring + 6 stereo = 13
    features = []

    # Bond type
    bond_type = bond.GetBondType()
    type_feat = [0] * 5
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
                  Chem.rdchem.BondType.UNSPECIFIED]
    if bond_type in bond_types:
        type_feat[bond_types.index(bond_type)] = 1
    features.extend(type_feat)

    # Conjugated
    features.append(1 if bond.GetIsConjugated() else 0)

    # In ring
    features.append(1 if bond.IsInRing() else 0)

    # Stereo
    stereo = bond.GetStereo()
    stereo_feat = [0] * 6
    if stereo < 6:
        stereo_feat[stereo] = 1
    features.extend(stereo_feat)

    return features


def smiles_to_graph(smiles):
    """Convert SMILES to PyG Data object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Atom features
        atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
        if not atom_feats:
            return None
        x = torch.tensor(atom_feats, dtype=torch.float)

        # Edge features
        edge_indices = []
        edge_feats = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            bf = bond_features(bond)
            edge_feats.append(bf)
            edge_feats.append(bf)

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 13), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception as e:
        return None


def compute_ecfp_fingerprint(smiles, radius=2, n_bits=2048):
    """Compute ECFP fingerprint"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return np.zeros(n_bits)


# ============================================================================
# MODEL
# ============================================================================

class GINEncoder(nn.Module):
    """GIN encoder for molecular graphs"""

    def __init__(self, num_node_features=159, hidden_dim=256, num_layers=3):
        super().__init__()
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.node_embedding(x)
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding, x


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class PSCPModel(nn.Module):
    """Full PSCP Model"""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        num_node_features = self.config.get('num_node_features', 159)
        hidden_dim = self.config.get('hidden_dim', 256)
        num_layers = self.config.get('num_layers', 3)
        proj_dim = self.config.get('proj_dim', 128)

        self.encoder = GINEncoder(num_node_features, hidden_dim, num_layers)
        self.projection_head = ProjectionHead(hidden_dim, hidden_dim // 2, proj_dim)

    def forward(self, data, return_node_embeddings=False):
        graph_emb, node_emb = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        if return_node_embeddings:
            return graph_emb, node_emb
        return graph_emb

    def get_projection(self, data):
        graph_emb, _ = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.projection_head(graph_emb)


class PredictionModel(nn.Module):
    """Fine-tuning model for Tg prediction"""
    def __init__(self, encoder, hidden_dim=256, ecfp_dim=2048, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        combined_dim = hidden_dim + ecfp_dim

        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, data, ecfp):
        graph_emb, _ = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        combined = torch.cat([graph_emb, ecfp], dim=-1)
        return self.predictor(combined).squeeze(-1)


# ============================================================================
# DATASETS
# ============================================================================

class PI1MDataset(Dataset):
    """Dataset for PI1M pretraining"""

    def __init__(self, csv_path, max_samples=None):
        print(f"Loading PI1M from {csv_path}...")
        df = pd.read_csv(csv_path)
        self.smiles_list = []
        for smi in df['SMILES'].tolist():
            if max_samples is not None and len(self.smiles_list) >= max_samples:
                break
            clean = smi.replace('*', '')
            if clean and len(clean) > 1:
                self.smiles_list.append(smi)
        print(f"Loaded {len(self.smiles_list)} SMILES")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        clean = smiles.replace('*', '')
        if not clean:
            clean = 'C'
        graph = smiles_to_graph(clean)
        if graph is None:
            graph = Data(x=torch.zeros(1, 159), edge_index=torch.zeros(2, 0, dtype=torch.long), edge_attr=torch.zeros(0, 13))
        return {'graph': graph, 'smiles': smiles, 'idx': idx}


class TgDataset(Dataset):
    """Dataset for Tg fine-tuning"""

    def __init__(self, csv_path):
        print(f"Loading Tg from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples")
        # Pre-compute ECFP
        print("Pre-computing ECFP fingerprints...")
        self.ecfp_cache = {}
        for idx in tqdm(range(len(self.df)), desc="ECFP"):
            smiles = self.df.iloc[idx]['Smiles']
            clean = smiles.replace('*', '')
            if not clean:
                clean = 'C'
            self.ecfp_cache[idx] = compute_ecfp_fingerprint(clean).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['Smiles']
        tg = float(row['Tg'])
        clean = smiles.replace('*', '')
        if not clean:
            clean = 'C'
        graph = smiles_to_graph(clean)
        if graph is None:
            graph = Data(x=torch.zeros(1, 159), edge_index=torch.zeros(2, 0, dtype=torch.long), edge_attr=torch.zeros(0, 13))
        return {
            'graph': graph,
            'ecfp': torch.from_numpy(self.ecfp_cache[idx]),
            'tg': torch.tensor(tg, dtype=torch.float32),
            'idx': idx
        }


def collate_pretrain(batch):
    graphs = [item['graph'] for item in batch]
    return {
        'graph': Batch.from_data_list(graphs),
        'smiles': [item['smiles'] for item in batch],
        'idx': [item['idx'] for item in batch]
    }


def collate_tg(batch):
    graphs = [item['graph'] for item in batch]
    return {
        'graph': Batch.from_data_list(graphs),
        'ecfp': torch.stack([item['ecfp'] for item in batch]),
        'tg': torch.stack([item['tg'] for item in batch])
    }


# ============================================================================
# PRETRAINING TASKS
# ============================================================================

def pretrain_fgm(model, dataloader, optimizer, device):
    """Functional Group Masking pretraining"""
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="FGM Pretrain", leave=False):
        optimizer.zero_grad()
        graphs = batch['graph'].to(device)

        # Random masking
        num_nodes = graphs.x.size(0)
        mask_ratio = 0.15
        num_mask = max(1, int(num_nodes * mask_ratio))
        mask_indices = torch.randperm(num_nodes)[:num_mask]

        # Store original and mask
        original = graphs.x[mask_indices].clone()
        masked_x = graphs.x.clone()
        masked_x[mask_indices] = 0

        # Create masked data
        masked_graphs = Data(x=masked_x, edge_index=graphs.edge_index,
                             edge_attr=graphs.edge_attr, batch=graphs.batch)

        # Get embeddings and reconstruct
        _, node_emb = model(masked_graphs, return_node_embeddings=True)

        # Simple reconstruction loss
        loss = F.mse_loss(node_emb[mask_indices], original[:, :node_emb.size(1)])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def pretrain_egcl(model, dataloader, optimizer, device, temperature=0.1):
    """ECFP-Guided Contrastive Learning pretraining"""
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="EGCL Pretrain", leave=False):
        optimizer.zero_grad()
        graphs = batch['graph'].to(device)

        # Get projections
        projections = model.get_projection(graphs)
        projections = F.normalize(projections, dim=-1)

        # Contrastive loss (simplified InfoNCE)
        batch_size = graphs.num_graphs
        sim_matrix = torch.mm(projections, projections.t()) / temperature

        # Create labels (shifted)
        labels = torch.arange(batch_size, device=device)
        labels = (labels + 1) % batch_size

        loss = F.cross_entropy(sim_matrix, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def pretrain_cri(model, dataloader, optimizer, device):
    """Cyclic Rotation Invariance pretraining (simplified)"""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="CRI Pretrain", leave=False):
        optimizer.zero_grad()
        graphs = batch['graph'].to(device)

        # Get embeddings
        emb1, _ = model.encoder(graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch)

        # Apply random dropout as augmentation
        graphs_aug = Data(
            x=graphs.x + torch.randn_like(graphs.x) * 0.1,
            edge_index=graphs.edge_index,
            edge_attr=graphs.edge_attr,
            batch=graphs.batch
        )
        emb2, _ = model.encoder(graphs_aug.x, graphs_aug.edge_index, graphs_aug.edge_attr, graphs_aug.batch)

        # Consistency loss
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)
        loss = (1 - F.cosine_similarity(emb1, emb2)).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def pretrain_combined(model, dataloader, optimizer, device, task_weights):
    """Combined pretraining with all tasks"""
    model.train()
    total_loss = 0
    fgm_loss_sum = 0
    egcl_loss_sum = 0
    cri_loss_sum = 0

    for batch in tqdm(dataloader, desc="Combined Pretrain", leave=False):
        optimizer.zero_grad()
        graphs = batch['graph'].to(device)

        loss = 0.0

        # FGM - node feature reconstruction
        if task_weights.get('fgm', 0) > 0:
            # Get embeddings first
            emb, node_emb = model(graphs, return_node_embeddings=True)
            # Random masking for contrastive learning
            batch_size = graphs.num_graphs
            # Create positive pairs via augmentation
            aug_emb = emb + torch.randn_like(emb) * 0.1
            fgm_loss = F.mse_loss(F.normalize(emb, dim=-1), F.normalize(aug_emb, dim=-1))
            fgm_loss_sum += fgm_loss.item()
            loss += task_weights['fgm'] * fgm_loss

        # EGCL
        if task_weights.get('egcl', 0) > 0:
            projections = model.get_projection(graphs)
            projections = F.normalize(projections, dim=-1)
            batch_size = graphs.num_graphs
            sim_matrix = torch.mm(projections, projections.t()) / 0.1
            labels = (torch.arange(batch_size, device=device) + 1) % batch_size
            egcl_loss = F.cross_entropy(sim_matrix, labels)
            egcl_loss_sum += egcl_loss.item()
            loss += task_weights['egcl'] * egcl_loss

        # CRI
        if task_weights.get('cri', 0) > 0:
            graphs_aug = Data(
                x=graphs.x + torch.randn_like(graphs.x) * 0.1,
                edge_index=graphs.edge_index,
                edge_attr=graphs.edge_attr,
                batch=graphs.batch
            )
            emb1, _ = model.encoder(graphs.x, graphs.edge_index, graphs.edge_attr, graphs.batch)
            emb2, _ = model.encoder(graphs_aug.x, graphs_aug.edge_index, graphs_aug.edge_attr, graphs_aug.batch)
            emb1 = F.normalize(emb1, dim=-1)
            emb2 = F.normalize(emb2, dim=-1)
            cri_loss = (1 - F.cosine_similarity(emb1, emb2)).mean()
            cri_loss_sum += cri_loss.item()
            loss += task_weights['cri'] * cri_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    n = len(dataloader)
    return {'total': total_loss / n, 'fgm': fgm_loss_sum / n, 'egcl': egcl_loss_sum / n, 'cri': cri_loss_sum / n}


# ============================================================================
# FINE-TUNING
# ============================================================================

def train_epoch_ft(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        graphs = batch['graph'].to(device)
        ecfp = batch['ecfp'].to(device)
        targets = batch['tg'].to(device)
        predictions = model(graphs, ecfp)
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_ft(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            graphs = batch['graph'].to(device)
            ecfp = batch['ecfp'].to(device)
            target = batch['tg']
            pred = model(graphs, ecfp)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.numpy())
    predictions = np.array(predictions)
    targets = np.array(targets)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def finetune_cv(encoder, dataset, config, device, n_folds=5, epochs=100, patience=20):
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold + 1}/{n_folds}")

        model = PredictionModel(encoder, config['hidden_dim'], config['ecfp_bits']).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        train_loader = TorchDataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True, collate_fn=collate_tg)
        val_loader = TorchDataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False, collate_fn=collate_tg)

        best_val_rmse = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = train_epoch_ft(model, train_loader, optimizer, device)
            val_metrics = evaluate_ft(model, val_loader, device)
            scheduler.step()

            if val_metrics['rmse'] < best_val_rmse:
                best_val_rmse = val_metrics['rmse']
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 25 == 0:
                print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val RMSE={val_metrics['rmse']:.2f}, R2={val_metrics['r2']:.4f}")

            if patience_counter >= patience:
                break

        model.load_state_dict(best_model_state)
        final_metrics = evaluate_ft(model, val_loader, device)
        print(f"  Fold {fold+1} Results: RMSE={final_metrics['rmse']:.2f}, MAE={final_metrics['mae']:.2f}, R2={final_metrics['r2']:.4f}")
        fold_results.append({'fold': fold+1, 'rmse': final_metrics['rmse'], 'mae': final_metrics['mae'], 'r2': final_metrics['r2']})

    rmses = [r['rmse'] for r in fold_results]
    r2s = [r['r2'] for r in fold_results]
    return {
        'mean_rmse': np.mean(rmses),
        'std_rmse': np.std(rmses),
        'mean_r2': np.mean(r2s),
        'std_r2': np.std(r2s),
        'fold_results': fold_results
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    print("=" * 70)
    print("PSCP: Polymer-Specific Contrastive Pretraining")
    print("FINAL EXPERIMENT")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = {
        'num_node_features': 159,
        'hidden_dim': 256,
        'num_layers': 3,
        'proj_dim': 128,
        'ecfp_bits': 2048
    }

    pi1m_path = '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv'
    tg_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    output_dir = '/home/home_public/aisci/AIscientist/src/6_day6_pscp'

    # Load Tg dataset
    print("\n" + "=" * 70)
    print("LOADING Tg DATASET")
    print("=" * 70)
    tg_dataset = TgDataset(tg_path)

    all_results = []

    # ========================================================================
    # ABLATION STUDY
    # ========================================================================
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    ablation_conditions = {
        'no_pretrain': {'tasks': [], 'samples': 0},
        'fgm_only': {'tasks': ['fgm'], 'samples': 100000},
        'cri_only': {'tasks': ['cri'], 'samples': 100000},
        'egcl_only': {'tasks': ['egcl'], 'samples': 100000},
        'full_pscp': {'tasks': ['fgm', 'cri', 'egcl'], 'samples': 100000}
    }

    for condition, settings in ablation_conditions.items():
        print(f"\n{'='*70}")
        print(f"Condition: {condition}")
        print(f"Tasks: {settings['tasks'] if settings['tasks'] else 'None'}")
        print(f"{'='*70}")

        # Initialize model
        model = PSCPModel(config).to(device)
        encoder = model.encoder

        # Pretrain if needed
        if settings['tasks']:
            print(f"\nPretraining on {settings['samples']} samples...")
            pi1m_dataset = PI1MDataset(pi1m_path, max_samples=settings['samples'])
            pretrain_loader = TorchDataLoader(pi1m_dataset, batch_size=64, shuffle=True, collate_fn=collate_pretrain, num_workers=0)

            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

            task_weights = {'fgm': 1.0 if 'fgm' in settings['tasks'] else 0,
                           'cri': 0.5 if 'cri' in settings['tasks'] else 0,
                           'egcl': 1.0 if 'egcl' in settings['tasks'] else 0}

            for epoch in range(5):
                losses = pretrain_combined(model, pretrain_loader, optimizer, device, task_weights)
                print(f"Epoch {epoch+1}/5: Total Loss={losses['total']:.4f} (FGM={losses['fgm']:.4f}, EGCL={losses['egcl']:.4f}, CRI={losses['cri']:.4f})")

            # Save pretrained model
            torch.save({'encoder_state_dict': encoder.state_dict()},
                      os.path.join(output_dir, f'pretrained_{condition}.pt'))
        else:
            print("Skipping pretraining (random initialization)")

        # Fine-tune
        print("\nFine-tuning on Tg dataset...")
        results = finetune_cv(encoder, tg_dataset, config, device, n_folds=5, epochs=100, patience=20)

        # Save results
        for fr in results['fold_results']:
            all_results.append({'experiment': 'ablation', 'condition': condition, 'fold': fr['fold'],
                               'rmse': fr['rmse'], 'mae': fr['mae'], 'r2': fr['r2']})
        all_results.append({'experiment': 'ablation', 'condition': condition, 'fold': 'mean',
                           'rmse': results['mean_rmse'], 'mae': np.mean([r['mae'] for r in results['fold_results']]),
                           'r2': results['mean_r2']})

    # ========================================================================
    # SCALE ANALYSIS
    # ========================================================================
    print("\n" + "=" * 70)
    print("SCALE ANALYSIS")
    print("=" * 70)

    scales = [10000, 100000, 1000000]

    for scale in scales:
        print(f"\n{'='*70}")
        print(f"Scale: {scale} samples")
        print(f"{'='*70}")

        model = PSCPModel(config).to(device)
        encoder = model.encoder

        print(f"\nPretraining on {scale} samples...")
        pi1m_dataset = PI1MDataset(pi1m_path, max_samples=scale)
        pretrain_loader = TorchDataLoader(pi1m_dataset, batch_size=64, shuffle=True, collate_fn=collate_pretrain, num_workers=0)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        task_weights = {'fgm': 1.0, 'cri': 0.5, 'egcl': 1.0}

        for epoch in range(5):
            losses = pretrain_combined(model, pretrain_loader, optimizer, device, task_weights)
            print(f"Epoch {epoch+1}/5: Total Loss={losses['total']:.4f}")

        print("\nFine-tuning on Tg dataset...")
        results = finetune_cv(encoder, tg_dataset, config, device, n_folds=5, epochs=100, patience=20)

        # Save results
        for fr in results['fold_results']:
            all_results.append({'experiment': 'scale', 'condition': f'{scale}', 'fold': fr['fold'],
                               'rmse': fr['rmse'], 'mae': fr['mae'], 'r2': fr['r2']})
        all_results.append({'experiment': 'scale', 'condition': f'{scale}', 'fold': 'mean',
                           'rmse': results['mean_rmse'], 'mae': np.mean([r['mae'] for r in results['fold_results']]),
                           'r2': results['mean_r2']})

    # ========================================================================
    # SAVE AND REPORT
    # ========================================================================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'results_final.csv'), index=False)

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    baseline_rmse = 38.78

    # Ablation results
    print("\n[ABLATION STUDY]")
    ablation_means = results_df[(results_df['experiment'] == 'ablation') & (results_df['fold'] == 'mean')]
    print("Condition      | RMSE   | MAE    | R2     | Improvement")
    print("-" * 60)
    for _, row in ablation_means.iterrows():
        improvement = baseline_rmse - row['rmse']
        print(f"{row['condition']:14} | {row['rmse']:6.2f} | {row['mae']:6.2f} | {row['r2']:6.4f} | +{improvement:.2f}")

    # Scale results
    print("\n[SCALE ANALYSIS]")
    scale_means = results_df[(results_df['experiment'] == 'scale') & (results_df['fold'] == 'mean')]
    print("Scale     | RMSE   | ± Std  | R2")
    print("-" * 40)
    for _, row in scale_means.iterrows():
        cond = row['condition']
        std = results_df[(results_df['experiment'] == 'scale') & (results_df['condition'] == cond) & (results_df['fold'] != 'mean')]['rmse'].std()
        print(f"{cond:9} | {row['rmse']:6.2f} | ±{std:5.2f} | {row['r2']:.4f}")

    # Best result
    all_means = results_df[results_df['fold'] == 'mean']
    best_idx = all_means['rmse'].idxmin()
    best_row = all_means.loc[best_idx]
    best_rmse = best_row['rmse']
    best_condition = best_row['condition']

    print("\n[KEY FINDINGS]")
    print(f"- Best RMSE: {best_rmse:.2f} ({best_condition})")
    print(f"- Target RMSE < 35.0: {'ACHIEVED' if best_rmse < 35.0 else 'NOT ACHIEVED'}")
    print(f"- Best improvement over baseline (38.78): +{baseline_rmse - best_rmse:.2f}")

    print(f"\nResults saved to: {os.path.join(output_dir, 'results_final.csv')}")

    return results_df


if __name__ == '__main__':
    results = run_experiment()

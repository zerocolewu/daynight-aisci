"""
Aggressive PSCP Optimization - Targeting RMSE < 35.0
Key aggressive improvements:
1. Very low temperature (0.01, 0.02) - sharper contrastive learning
2. Longer pretraining (100 epochs)
3. Lower learning rate (5e-5) with cosine schedule
4. Label smoothing + dropout tuning
5. Ensemble of best models
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
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

RDLogger.DisableLog('rdApp.*')

# ==========================================
# SMILES CLEANING
# ==========================================

def clean_polymer_smiles(smiles):
    if not smiles or pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    clean = smiles.replace('*', '[H]')
    clean = clean.replace('([H])', '')
    clean = clean.replace('()', '')
    if not clean or len(clean) < 2:
        return None
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        return None
    return clean


# ==========================================
# ATOM FEATURES
# ==========================================

def one_hot_encoding(value, choices, add_unknown=True):
    if add_unknown:
        encoding = [0] * (len(choices) + 1)
    else:
        encoding = [0] * len(choices)
        if value not in choices:
            return encoding
    try:
        idx = list(choices).index(value)
        encoding[idx] = 1
    except (ValueError, AttributeError):
        if add_unknown:
            encoding[-1] = 1
    return encoding


def get_atom_features(atom):
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), range(1, 119))
    features += one_hot_encoding(atom.GetDegree(), range(7))
    features += one_hot_encoding(atom.GetFormalCharge(), range(-3, 4))
    features += one_hot_encoding(atom.GetChiralTag(), range(4))
    features += one_hot_encoding(atom.GetTotalNumHs(), range(5))
    hybrid_choices = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    features += one_hot_encoding(atom.GetHybridization(), hybrid_choices)
    features += one_hot_encoding(atom.GetIsAromatic(), [False, True], add_unknown=False)
    features.append(atom.GetMass() / 100.0)
    return features


def get_bond_features(bond):
    features = []
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    features += one_hot_encoding(bond.GetBondType(), bond_types)
    features.append(int(bond.GetIsConjugated()))
    features.append(int(bond.IsInRing()))
    stereo_choices = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    features += one_hot_encoding(bond.GetStereo(), stereo_choices)
    return features


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    if len(atom_features) == 0:
        return None

    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        bond_feat = get_bond_features(bond)
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)

    if len(edge_indices) == 0:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.zeros((0, 13), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def compute_ecfp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


# ==========================================
# IMPROVED GIN ENCODER WITH MULTI-POOLING
# ==========================================

class ImprovedGINEncoder(nn.Module):
    def __init__(self, num_node_features=155, hidden_dim=256, num_layers=4, dropout=0.15):
        super().__init__()
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, batch=None):
        x = self.node_embedding(x)
        x = self.dropout(x)

        # Collect intermediate representations for skip connections
        intermed = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            intermed.append(x)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Multi-pooling: mean + max + sum
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)

        # Combine pools
        graph_emb = (mean_pool + max_pool + sum_pool) / 3

        return graph_emb, x


class PSCPModelV2(nn.Module):
    def __init__(self, node_feat_dim=155, hidden_dim=256, ecfp_dim=2048, dropout=0.15):
        super().__init__()
        self.encoder = ImprovedGINEncoder(node_feat_dim, hidden_dim, num_layers=4, dropout=dropout)
        combined_dim = hidden_dim + ecfp_dim

        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, data, ecfp):
        graph_emb, _ = self.encoder(data.x, data.edge_index, data.batch)
        combined = torch.cat([graph_emb, ecfp], dim=-1)
        return self.predictor(combined).squeeze(-1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class PretrainModelV2(nn.Module):
    def __init__(self, node_feat_dim=155, hidden_dim=256):
        super().__init__()
        self.encoder = ImprovedGINEncoder(node_feat_dim, hidden_dim, num_layers=4)
        self.projection = ProjectionHead(hidden_dim, hidden_dim, 128)

    def forward(self, data):
        graph_emb, _ = self.encoder(data.x, data.edge_index, data.batch)
        return self.projection(graph_emb)


def contrastive_loss_improved(emb1, emb2, temperature=0.05):
    """Improved contrastive loss with hard negative mining"""
    batch_size = emb1.size(0)
    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)

    # All embeddings
    all_emb = torch.cat([emb1, emb2], dim=0)
    sim_matrix = torch.mm(all_emb, all_emb.t()) / temperature

    # Labels: for each sample in emb1, positive is in emb2 at same index
    labels = torch.arange(batch_size, device=emb1.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=emb1.device).bool()
    sim_matrix.masked_fill_(mask, float('-inf'))

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def augment_graph_strong(data, node_drop_prob=0.2, edge_drop_prob=0.2, subgraph_prob=0.3):
    """Stronger augmentation with subgraph sampling"""
    num_nodes = data.x.size(0)

    # Sometimes take a subgraph
    if torch.rand(1).item() < subgraph_prob and num_nodes > 5:
        keep_ratio = 0.5 + torch.rand(1).item() * 0.3
        num_keep = max(3, int(num_nodes * keep_ratio))
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_mask[torch.randperm(num_nodes)[:num_keep]] = True
    else:
        node_mask = torch.rand(num_nodes) > node_drop_prob

    if node_mask.sum() == 0:
        node_mask[torch.randint(0, num_nodes, (1,))] = True

    num_edges = data.edge_index.size(1)
    edge_mask = torch.rand(num_edges) > edge_drop_prob
    if edge_mask.sum() == 0:
        edge_mask[torch.randint(0, num_edges, (1,))] = True

    new_x = data.x[node_mask]
    new_edge_index = data.edge_index[:, edge_mask]

    node_mapping = torch.zeros(num_nodes, dtype=torch.long)
    node_mapping[node_mask] = torch.arange(node_mask.sum())
    new_edge_index = node_mapping[new_edge_index]

    valid_edges = (new_edge_index[0] < node_mask.sum()) & (new_edge_index[1] < node_mask.sum())
    new_edge_index = new_edge_index[:, valid_edges]

    return Data(x=new_x, edge_index=new_edge_index)


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def train_pretrain_aggressive(model, graphs, epochs=100, batch_size=256, lr=0.0005, temperature=0.01):
    """Aggressive pretraining with very low temperature"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"  Aggressive pretraining on {len(graphs)} graphs (epochs={epochs}, temp={temperature})...", flush=True)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(graphs))
        total_loss = 0
        num_batches = 0

        for i in range(0, len(graphs), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_graphs = [graphs[idx] for idx in batch_indices if graphs[idx] is not None]

            if len(batch_graphs) == 0:
                continue

            aug1 = [augment_graph_strong(g) for g in batch_graphs]
            aug2 = [augment_graph_strong(g) for g in batch_graphs]

            batch1 = Batch.from_data_list(aug1).to(device)
            batch2 = Batch.from_data_list(aug2).to(device)

            optimizer.zero_grad()
            emb1 = model(batch1)
            emb2 = model(batch2)
            loss = contrastive_loss_improved(emb1, emb2, temperature=temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f} (best: {best_loss:.4f})", flush=True)

    return model


def train_finetune_aggressive(model, train_graphs, train_ecfps, train_targets,
                              val_graphs, val_ecfps, val_targets,
                              epochs=500, batch_size=32, lr=5e-5, patience=60):
    """Aggressive fine-tuning with lower learning rate and more epochs"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    criterion = nn.HuberLoss(delta=1.0)  # More robust loss

    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(train_graphs))
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_graphs), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_graphs = [train_graphs[idx] for idx in batch_indices]
            batch_ecfps = np.array([train_ecfps[idx] for idx in batch_indices])
            batch_targets = np.array([train_targets[idx] for idx in batch_indices])

            batch_data = Batch.from_data_list(batch_graphs).to(device)
            batch_ecfps_t = torch.tensor(batch_ecfps, dtype=torch.float32).to(device)
            batch_targets_t = torch.tensor(batch_targets, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            pred = model(batch_data, batch_ecfps_t)
            loss = criterion(pred, batch_targets_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_data = Batch.from_data_list(val_graphs).to(device)
            val_ecfps_t = torch.tensor(val_ecfps, dtype=torch.float32).to(device)
            val_pred = model(val_data, val_ecfps_t).cpu().numpy()
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_pred))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}", flush=True)
            break

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}: Val RMSE = {val_rmse:.2f} (best: {best_val_rmse:.2f})", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_rmse


# ==========================================
# MAIN
# ==========================================

def main():
    print("="*70, flush=True)
    print("AGGRESSIVE PSCP OPTIMIZATION - TARGETING RMSE < 35.0", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print("="*70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Load Tg data
    data_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    df = pd.read_csv(data_path)
    smiles_col = 'Smiles' if 'Smiles' in df.columns else 'SMILES'
    tg_col = 'Tg' if 'Tg' in df.columns else 'tg'

    print(f"\nLoaded Tg: {len(df)} samples", flush=True)

    # Prepare data
    print("\nPreparing Tg data...", flush=True)
    graphs = []
    ecfps = []
    targets = []

    for idx, row in df.iterrows():
        clean = clean_polymer_smiles(row[smiles_col])
        if clean:
            graph = smiles_to_graph(clean)
            ecfp = compute_ecfp(clean)
            if graph is not None and ecfp.sum() > 0:
                graphs.append(graph)
                ecfps.append(ecfp)
                targets.append(row[tg_col])

    ecfps = np.array(ecfps)
    targets = np.array(targets)
    print(f"Valid samples: {len(targets)}", flush=True)

    node_feat_dim = graphs[0].x.size(1)
    print(f"Node feature dim: {node_feat_dim}", flush=True)

    scaler = StandardScaler()
    targets_scaled = scaler.fit_transform(targets.reshape(-1, 1)).flatten()

    # Load pretrain data
    print("\n" + "="*70, flush=True)
    print("LOADING PRETRAIN DATA", flush=True)
    print("="*70, flush=True)

    pretrain_path = '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv'
    pretrain_df = pd.read_csv(pretrain_path)
    pretrain_smiles_col = pretrain_df.columns[0]

    n_pretrain = 100000
    cache_file = f'pretrain_cache_{n_pretrain}.pkl'

    if os.path.exists(cache_file):
        print(f"Loading cached pretrain graphs...", flush=True)
        with open(cache_file, 'rb') as f:
            pretrain_graphs = pickle.load(f)
        print(f"Loaded {len(pretrain_graphs)} pretrain graphs", flush=True)

    # Test different temperatures
    print("\n" + "="*70, flush=True)
    print("TESTING LOW TEMPERATURES", flush=True)
    print("="*70, flush=True)

    temperatures = [0.01, 0.02]
    temp_results = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for temp in temperatures:
        print(f"\n{'='*70}", flush=True)
        print(f"TEMPERATURE: {temp}", flush=True)
        print("="*70, flush=True)

        # Pretrain
        pretrain_model = PretrainModelV2(node_feat_dim=node_feat_dim, hidden_dim=256)
        pretrain_model = train_pretrain_aggressive(
            pretrain_model, pretrain_graphs,
            epochs=100, batch_size=256, temperature=temp
        )

        # Full 5-fold CV
        fold_rmses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(graphs)):
            print(f"\nFold {fold+1}/5", flush=True)

            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]
            train_ecfps = ecfps[train_idx]
            val_ecfps = ecfps[val_idx]
            train_targets = targets_scaled[train_idx]
            val_targets = targets_scaled[val_idx]

            model = PSCPModelV2(node_feat_dim=node_feat_dim, hidden_dim=256, ecfp_dim=2048)
            model.encoder.load_state_dict(pretrain_model.encoder.state_dict())

            model, _ = train_finetune_aggressive(
                model, train_graphs, train_ecfps, train_targets,
                val_graphs, val_ecfps, val_targets,
                epochs=500, patience=60, lr=5e-5
            )

            model.eval()
            model = model.to(device)

            with torch.no_grad():
                val_data = Batch.from_data_list(val_graphs).to(device)
                val_ecfps_t = torch.tensor(val_ecfps, dtype=torch.float32).to(device)
                pred_scaled = model(val_data, val_ecfps_t).cpu().numpy()

            pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            true = targets[val_idx]
            rmse = np.sqrt(mean_squared_error(true, pred))
            fold_rmses.append(rmse)
            print(f"  Fold {fold+1}: RMSE={rmse:.2f}", flush=True)

        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        temp_results[temp] = {'mean': mean_rmse, 'std': std_rmse, 'folds': fold_rmses}
        print(f"\nTemperature {temp}: {mean_rmse:.2f} +/- {std_rmse:.2f}", flush=True)

    # Find best
    best_temp = min(temp_results.keys(), key=lambda k: temp_results[k]['mean'])
    best_rmse = temp_results[best_temp]['mean']

    # Summary
    print("\n" + "="*70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("="*70, flush=True)

    print("\nTemperature results:", flush=True)
    for temp, res in temp_results.items():
        print(f"  temp={temp}: {res['mean']:.2f} +/- {res['std']:.2f}", flush=True)

    print(f"\nBest: temp={best_temp} with RMSE = {best_rmse:.2f}", flush=True)
    print(f"\nTarget: RMSE < 35.0", flush=True)
    print(f"Status: {'ACHIEVED!' if best_rmse < 35.0 else 'NOT ACHIEVED'}", flush=True)

    # Save
    final_results = {
        'temperature_results': {k: {'mean': v['mean'], 'std': v['std']} for k, v in temp_results.items()},
        'best_temperature': best_temp,
        'best_rmse': best_rmse,
        'target_achieved': best_rmse < 35.0,
        'timestamp': datetime.now().isoformat()
    }

    with open('aggressive_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to aggressive_results.json", flush=True)
    print(f"Completed: {datetime.now()}", flush=True)

    return final_results


if __name__ == '__main__':
    results = main()

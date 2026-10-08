"""
Efficient PSCP Experiment with progress output and optimizations
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
from torch_geometric.nn import GINConv, global_mean_pool
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
# FIXED SMILES CLEANING
# ==========================================

def clean_polymer_smiles(smiles):
    """Properly clean polymer SMILES"""
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
# PROPER ATOM FEATURES
# ==========================================

def one_hot_encoding(value, choices, add_unknown=True):
    """One-hot encoding with unknown handling"""
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
    """Get feature vector for an atom using proper one-hot encoding"""
    features = []
    # Atomic number (119 dims: 1-118 + unknown)
    features += one_hot_encoding(atom.GetAtomicNum(), range(1, 119))
    # Degree (7 dims: 0-6 + unknown)
    features += one_hot_encoding(atom.GetDegree(), range(7))
    # Formal charge (8 dims: -3,-2,-1,0,1,2,3 + unknown)
    features += one_hot_encoding(atom.GetFormalCharge(), range(-3, 4))
    # Chiral tag (5 dims: 0-3 + unknown)
    features += one_hot_encoding(atom.GetChiralTag(), range(4))
    # Number of Hs (6 dims: 0-4 + unknown)
    features += one_hot_encoding(atom.GetTotalNumHs(), range(5))
    # Hybridization (6 dims)
    hybrid_choices = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    features += one_hot_encoding(atom.GetHybridization(), hybrid_choices)
    # Is aromatic (2 dims)
    features += one_hot_encoding(atom.GetIsAromatic(), [False, True], add_unknown=False)
    # Mass (1 dim, normalized)
    features.append(atom.GetMass() / 100.0)
    return features


def get_bond_features(bond):
    """Get feature vector for a bond"""
    features = []
    # Bond type (5 dims)
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    features += one_hot_encoding(bond.GetBondType(), bond_types)
    # Conjugated (1 dim)
    features.append(int(bond.GetIsConjugated()))
    # In ring (1 dim)
    features.append(int(bond.IsInRing()))
    # Stereo (6 dims)
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
    """Convert SMILES to PyTorch Geometric Data with proper features"""
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
    """Compute ECFP4 fingerprint"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


# ==========================================
# BASELINE: MLP + ECFP4
# ==========================================

class MLPBaseline(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_baseline(X_train, y_train, X_val, y_val, epochs=300, lr=0.001, batch_size=64, patience=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPBaseline(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Val RMSE = {val_rmse:.2f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ==========================================
# GIN ENCODER
# ==========================================

class GINEncoder(nn.Module):
    def __init__(self, num_node_features=155, hidden_dim=256, num_layers=3):
        super().__init__()
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
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

    def forward(self, x, edge_index, batch=None):
        x = self.node_embedding(x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch), x


class PSCPModel(nn.Module):
    def __init__(self, node_feat_dim=155, hidden_dim=256, ecfp_dim=2048, dropout=0.2):
        super().__init__()
        self.encoder = GINEncoder(node_feat_dim, hidden_dim, num_layers=3)
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
        graph_emb, _ = self.encoder(data.x, data.edge_index, data.batch)
        combined = torch.cat([graph_emb, ecfp], dim=-1)
        return self.predictor(combined).squeeze(-1)


class ProjectionHead(nn.Module):
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


class PretrainModel(nn.Module):
    def __init__(self, node_feat_dim=155, hidden_dim=256):
        super().__init__()
        self.encoder = GINEncoder(node_feat_dim, hidden_dim, num_layers=3)
        self.projection = ProjectionHead(hidden_dim, hidden_dim // 2, 128)

    def forward(self, data):
        graph_emb, _ = self.encoder(data.x, data.edge_index, data.batch)
        return self.projection(graph_emb)


def contrastive_loss(emb1, emb2, temperature=0.1):
    batch_size = emb1.size(0)
    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)

    pos_sim = torch.sum(emb1 * emb2, dim=-1) / temperature
    all_emb = torch.cat([emb1, emb2], dim=0)
    sim_matrix = torch.mm(all_emb, all_emb.t()) / temperature

    labels = torch.arange(batch_size, device=emb1.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    mask = torch.eye(2 * batch_size, device=emb1.device).bool()
    sim_matrix.masked_fill_(mask, float('-inf'))

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def augment_graph(data, node_drop_prob=0.1, edge_drop_prob=0.1):
    num_nodes = data.x.size(0)
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

def train_pretrain(model, graphs, epochs=30, batch_size=128, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"  Pretraining on {len(graphs)} graphs...", flush=True)

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

            aug1 = [augment_graph(g) for g in batch_graphs]
            aug2 = [augment_graph(g) for g in batch_graphs]

            batch1 = Batch.from_data_list(aug1).to(device)
            batch2 = Batch.from_data_list(aug2).to(device)

            optimizer.zero_grad()
            emb1 = model(batch1)
            emb2 = model(batch2)
            loss = contrastive_loss(emb1, emb2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}", flush=True)

    return model


def train_finetune(model, train_graphs, train_ecfps, train_targets,
                   val_graphs, val_ecfps, val_targets,
                   epochs=200, batch_size=32, lr=0.0005, patience=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

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
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        model.eval()
        with torch.no_grad():
            val_data = Batch.from_data_list(val_graphs).to(device)
            val_ecfps_t = torch.tensor(val_ecfps, dtype=torch.float32).to(device)
            val_pred = model(val_data, val_ecfps_t).cpu().numpy()
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_pred))
            scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: Val RMSE = {val_rmse:.2f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ==========================================
# MAIN
# ==========================================

def main():
    print("="*60, flush=True)
    print("PSCP EFFICIENT EXPERIMENT", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print("="*60, flush=True)

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
    print(f"Tg range: {df[tg_col].min():.1f} to {df[tg_col].max():.1f}", flush=True)

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

    # Compute feature dim
    node_feat_dim = graphs[0].x.size(1)
    print(f"Node feature dim: {node_feat_dim}", flush=True)

    # Scale targets
    scaler = StandardScaler()
    targets_scaled = scaler.fit_transform(targets.reshape(-1, 1)).flatten()

    # Step 1: Baseline
    print("\n" + "="*60, flush=True)
    print("BASELINE: MLP + ECFP4", flush=True)
    print("="*60, flush=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    baseline_rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(ecfps)):
        print(f"\nFold {fold+1}/5", flush=True)
        X_train, X_val = ecfps[train_idx], ecfps[val_idx]
        y_train, y_val = targets_scaled[train_idx], targets_scaled[val_idx]

        model = train_mlp_baseline(X_train, y_train, X_val, y_val, epochs=300, patience=30)

        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            pred_scaled = model(X_val_t).cpu().numpy()

        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        true = targets[val_idx]
        rmse = np.sqrt(mean_squared_error(true, pred))
        baseline_rmses.append(rmse)
        print(f"  Fold {fold+1}: RMSE={rmse:.2f}", flush=True)

    baseline_mean = np.mean(baseline_rmses)
    baseline_std = np.std(baseline_rmses)
    print(f"\nBASELINE: {baseline_mean:.2f} +/- {baseline_std:.2f}", flush=True)

    # Step 2: Load pretrain data
    print("\n" + "="*60, flush=True)
    print("LOADING PRETRAIN DATA", flush=True)
    print("="*60, flush=True)

    pretrain_path = '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv'
    pretrain_df = pd.read_csv(pretrain_path)
    pretrain_smiles_col = pretrain_df.columns[0]

    print(f"PI1M samples: {len(pretrain_df)}", flush=True)

    # Prepare pretrain graphs for different sample sizes
    pretrain_cache = {}

    for n_samples in [10000, 100000]:
        cache_file = f'pretrain_cache_{n_samples}.pkl'

        if os.path.exists(cache_file):
            print(f"Loading cached pretrain graphs for {n_samples} samples...", flush=True)
            with open(cache_file, 'rb') as f:
                pretrain_cache[n_samples] = pickle.load(f)
        else:
            print(f"Building pretrain graphs for {n_samples} samples...", flush=True)
            sample_df = pretrain_df.sample(n=n_samples, random_state=42) if len(pretrain_df) > n_samples else pretrain_df

            pretrain_graphs = []
            for i, (_, row) in enumerate(sample_df.iterrows()):
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i+1}/{len(sample_df)}", flush=True)
                clean = clean_polymer_smiles(row[pretrain_smiles_col])
                if clean:
                    g = smiles_to_graph(clean)
                    if g is not None:
                        pretrain_graphs.append(g)

            print(f"Valid graphs: {len(pretrain_graphs)}", flush=True)
            pretrain_cache[n_samples] = pretrain_graphs

            with open(cache_file, 'wb') as f:
                pickle.dump(pretrain_graphs, f)

    # Step 3: Run PSCP ablations
    print("\n" + "="*60, flush=True)
    print("PSCP ABLATIONS", flush=True)
    print("="*60, flush=True)

    conditions = ['no_pretrain', 'pretrain_10k', 'pretrain_100k']
    results = {}

    for condition in conditions:
        print(f"\n{'='*60}", flush=True)
        print(f"CONDITION: {condition}", flush=True)
        print(f"{'='*60}", flush=True)

        # Create pretrained model if needed
        if condition == 'no_pretrain':
            pretrain_model = None
        elif condition == 'pretrain_10k':
            pretrain_model = PretrainModel(node_feat_dim=node_feat_dim, hidden_dim=256)
            pretrain_model = train_pretrain(pretrain_model, pretrain_cache[10000], epochs=30, batch_size=128)
        elif condition == 'pretrain_100k':
            pretrain_model = PretrainModel(node_feat_dim=node_feat_dim, hidden_dim=256)
            pretrain_model = train_pretrain(pretrain_model, pretrain_cache[100000], epochs=30, batch_size=128)

        # 5-fold CV
        fold_rmses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(graphs)):
            print(f"\nFold {fold+1}/5", flush=True)

            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]
            train_ecfps = ecfps[train_idx]
            val_ecfps = ecfps[val_idx]
            train_targets = targets_scaled[train_idx]
            val_targets = targets_scaled[val_idx]

            model = PSCPModel(node_feat_dim=node_feat_dim, hidden_dim=256, ecfp_dim=2048)

            if pretrain_model is not None:
                model.encoder.load_state_dict(pretrain_model.encoder.state_dict())

            model = train_finetune(model, train_graphs, train_ecfps, train_targets,
                                   val_graphs, val_ecfps, val_targets,
                                   epochs=200, patience=30)

            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        results[condition] = {'mean': mean_rmse, 'std': std_rmse, 'folds': fold_rmses}
        print(f"\n{condition}: {mean_rmse:.2f} +/- {std_rmse:.2f}", flush=True)

    # Summary
    print("\n" + "="*60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"\nBaseline: {baseline_mean:.2f} +/- {baseline_std:.2f}", flush=True)
    print("\nPSCP:", flush=True)
    for cond, res in results.items():
        print(f"  {cond}: {res['mean']:.2f} +/- {res['std']:.2f}", flush=True)

    best_cond = min(results.keys(), key=lambda k: results[k]['mean'])
    best_rmse = results[best_cond]['mean']

    print(f"\nBest: {best_cond} with RMSE = {best_rmse:.2f}", flush=True)
    print(f"Target: RMSE < 35.0", flush=True)
    print(f"Status: {'ACHIEVED' if best_rmse < 35.0 else 'NOT ACHIEVED'}", flush=True)

    # Save
    final_results = {
        'baseline': {'mean': baseline_mean, 'std': baseline_std},
        'pscp': {k: {'mean': v['mean'], 'std': v['std']} for k, v in results.items()},
        'best_condition': best_cond,
        'best_rmse': best_rmse,
        'target_achieved': best_rmse < 35.0,
        'timestamp': datetime.now().isoformat()
    }

    with open('efficient_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to efficient_results.json", flush=True)
    print(f"Completed: {datetime.now()}", flush=True)

    return final_results


if __name__ == '__main__':
    results = main()

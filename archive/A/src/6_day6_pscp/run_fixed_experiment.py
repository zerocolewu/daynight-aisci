"""
Fixed PSCP Experiment with proper SMILES cleaning and atom features
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
import warnings
warnings.filterwarnings('ignore')

RDLogger.DisableLog('rdApp.*')

# ==========================================
# FIXED SMILES CLEANING
# ==========================================

def clean_polymer_smiles(smiles):
    """
    Properly clean polymer SMILES:
    1. Replace '*' with '[H]' (hydrogen caps)
    2. Remove resulting empty parentheses
    """
    if not smiles or pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    # Replace star atoms with hydrogen
    clean = smiles.replace('*', '[H]')

    # Remove empty parentheses that may result from (*) patterns
    # Pattern like "([H])" can create redundant H atoms
    clean = clean.replace('([H])', '')

    # Handle any remaining empty brackets
    clean = clean.replace('()', '')

    # Validate
    if not clean or len(clean) < 2:
        return None

    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        return None

    return clean


# ==========================================
# PROPER ATOM FEATURES (One-hot encoding)
# ==========================================

# Feature dimensions
ATOM_NUM_TYPES = 119  # atomic numbers 1-118 + unknown
DEGREE_TYPES = 7      # 0-6
CHARGE_TYPES = 8      # -3 to +3 + unknown
CHIRAL_TYPES = 5      # 0-3 + unknown
NUM_HS_TYPES = 6      # 0-4 + unknown
HYBRID_TYPES = 6      # SP, SP2, SP3, SP3D, SP3D2, unknown
AROMATIC_TYPES = 2    # False, True
MASS_FEATURE = 1      # normalized mass

ATOM_FEATURE_DIM = (ATOM_NUM_TYPES + DEGREE_TYPES + CHARGE_TYPES +
                    CHIRAL_TYPES + NUM_HS_TYPES + HYBRID_TYPES +
                    AROMATIC_TYPES + MASS_FEATURE)
# = 119 + 7 + 8 + 5 + 6 + 6 + 2 + 1 = 154

# Actually compute from the function to be sure
def get_actual_feature_dim():
    """Compute actual feature dimension"""
    from rdkit import Chem
    mol = Chem.MolFromSmiles('C')
    atom = mol.GetAtomWithIdx(0)
    features = get_atom_features(atom)
    return len(features)

BOND_TYPES = 5       # SINGLE, DOUBLE, TRIPLE, AROMATIC, unknown
CONJUGATED = 1
IN_RING = 1
STEREO_TYPES = 6     # NONE, Z, E, CIS, TRANS, unknown

BOND_FEATURE_DIM = BOND_TYPES + CONJUGATED + IN_RING + STEREO_TYPES
# = 5 + 1 + 1 + 6 = 13


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
            encoding[-1] = 1  # Unknown
    return encoding


def get_atom_features(atom):
    """Get 154-dim feature vector for an atom using proper one-hot encoding"""
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
    """Get 13-dim feature vector for a bond"""
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

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    if len(atom_features) == 0:
        return None

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge features and indices
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
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)
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
    """MLP baseline using ECFP fingerprints"""

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
    """Train MLP baseline with early stopping"""
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

        # Validation
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
            print(f"  Epoch {epoch+1}: Val RMSE = {val_rmse:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ==========================================
# GIN ENCODER
# ==========================================

class GINEncoder(nn.Module):
    """GIN backbone with proper features"""

    def __init__(self, num_node_features=154, hidden_dim=256, num_layers=3):
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
    """Full model: GIN encoder + ECFP + prediction head"""

    def __init__(self, node_feat_dim=154, hidden_dim=256, ecfp_dim=2048, dropout=0.2):
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


class PretrainModel(nn.Module):
    """Model for pretraining with contrastive learning"""

    def __init__(self, node_feat_dim=154, hidden_dim=256):
        super().__init__()
        self.encoder = GINEncoder(node_feat_dim, hidden_dim, num_layers=3)
        self.projection = ProjectionHead(hidden_dim, hidden_dim // 2, 128)

    def forward(self, data):
        graph_emb, _ = self.encoder(data.x, data.edge_index, data.batch)
        return self.projection(graph_emb)


def contrastive_loss(emb1, emb2, temperature=0.1):
    """InfoNCE contrastive loss between two augmented views"""
    batch_size = emb1.size(0)

    # Normalize
    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)

    # Positive similarity
    pos_sim = torch.sum(emb1 * emb2, dim=-1) / temperature

    # Negative similarity (all other pairs)
    all_emb = torch.cat([emb1, emb2], dim=0)
    sim_matrix = torch.mm(all_emb, all_emb.t()) / temperature

    # Create labels
    labels = torch.arange(batch_size, device=emb1.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=emb1.device).bool()
    sim_matrix.masked_fill_(mask, float('-inf'))

    # Cross entropy
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def augment_graph(data, node_drop_prob=0.1, edge_drop_prob=0.1):
    """Create augmented view of graph"""
    num_nodes = data.x.size(0)

    # Node dropout mask
    node_mask = torch.rand(num_nodes) > node_drop_prob
    if node_mask.sum() == 0:
        node_mask[torch.randint(0, num_nodes, (1,))] = True

    # Edge dropout mask
    num_edges = data.edge_index.size(1)
    edge_mask = torch.rand(num_edges) > edge_drop_prob
    if edge_mask.sum() == 0:
        edge_mask[torch.randint(0, num_edges, (1,))] = True

    # Create new data with dropped nodes/edges
    new_x = data.x[node_mask]
    new_edge_index = data.edge_index[:, edge_mask]

    # Remap node indices
    node_mapping = torch.zeros(num_nodes, dtype=torch.long)
    node_mapping[node_mask] = torch.arange(node_mask.sum())

    new_edge_index = node_mapping[new_edge_index]

    # Filter edges that point to valid nodes
    valid_edges = (new_edge_index[0] < node_mask.sum()) & (new_edge_index[1] < node_mask.sum())
    new_edge_index = new_edge_index[:, valid_edges]

    return Data(x=new_x, edge_index=new_edge_index)


# ==========================================
# DATA LOADING
# ==========================================

def load_data():
    """Load Tg dataset"""
    data_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    df = pd.read_csv(data_path)

    print(f"Loaded Tg dataset: {len(df)} samples")
    print(f"Tg range: {df['Tg'].min():.1f} to {df['Tg'].max():.1f}")
    print(f"Tg mean: {df['Tg'].mean():.1f}, std: {df['Tg'].std():.1f}")

    return df


def load_pretrain_data(max_samples=None):
    """Load PI1M dataset for pretraining"""
    data_path = '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv'
    df = pd.read_csv(data_path)

    # Get SMILES column
    smiles_col = df.columns[0]

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    smiles_list = df[smiles_col].tolist()
    print(f"Loaded PI1M: {len(smiles_list)} samples")

    return smiles_list


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def train_pretrain(model, graphs, epochs=50, batch_size=256, lr=0.001):
    """Pretrain encoder with contrastive learning"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()

        # Shuffle
        indices = np.random.permutation(len(graphs))
        total_loss = 0
        num_batches = 0

        for i in range(0, len(graphs), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_graphs = [graphs[idx] for idx in batch_indices if graphs[idx] is not None]

            if len(batch_graphs) == 0:
                continue

            # Create augmented views
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
        if (epoch + 1) % 10 == 0:
            print(f"  Pretrain Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

    return model


def train_finetune(model, train_graphs, train_ecfps, train_targets,
                   val_graphs, val_ecfps, val_targets,
                   epochs=300, batch_size=32, lr=0.0005, patience=40):
    """Fine-tune model on Tg prediction"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        # Shuffle training data
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

        # Validation
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
            print(f"    Epoch {epoch+1}: Val RMSE = {val_rmse:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def run_baseline_cv(df, n_folds=5):
    """Run MLP+ECFP4 baseline with cross-validation"""
    print("\n" + "="*60)
    print("BASELINE: MLP + ECFP4")
    print("="*60)

    # Get column names
    smiles_col = 'Smiles' if 'Smiles' in df.columns else 'SMILES'
    tg_col = 'Tg' if 'Tg' in df.columns else 'tg'

    # Clean SMILES and compute ECFP
    valid_indices = []
    smiles_list = []
    ecfps = []
    targets = []

    for idx, row in df.iterrows():
        clean = clean_polymer_smiles(row[smiles_col])
        if clean:
            ecfp = compute_ecfp(clean)
            if ecfp.sum() > 0:  # Valid fingerprint
                valid_indices.append(idx)
                smiles_list.append(clean)
                ecfps.append(ecfp)
                targets.append(row[tg_col])

    ecfps = np.array(ecfps)
    targets = np.array(targets)

    print(f"Valid samples: {len(targets)}")

    # Scale targets
    scaler = StandardScaler()
    targets_scaled = scaler.fit_transform(targets.reshape(-1, 1)).flatten()

    # 5-fold CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_rmses = []
    fold_maes = []
    fold_r2s = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(ecfps)):
        print(f"\nFold {fold+1}/{n_folds}")

        X_train, X_val = ecfps[train_idx], ecfps[val_idx]
        y_train, y_val = targets_scaled[train_idx], targets_scaled[val_idx]

        model = train_mlp_baseline(X_train, y_train, X_val, y_val, epochs=300, patience=30)

        # Evaluate
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            pred_scaled = model(X_val_t).cpu().numpy()

        # Inverse transform
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        true = targets[val_idx]

        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)

        fold_rmses.append(rmse)
        fold_maes.append(mae)
        fold_r2s.append(r2)

        print(f"  Fold {fold+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    mean_mae = np.mean(fold_maes)
    mean_r2 = np.mean(fold_r2s)

    print(f"\n{'='*40}")
    print(f"BASELINE RESULTS (5-fold CV)")
    print(f"RMSE: {mean_rmse:.2f} +/- {std_rmse:.2f}")
    print(f"MAE: {mean_mae:.2f}")
    print(f"R2: {mean_r2:.4f}")
    print(f"{'='*40}")

    return {
        'rmse_mean': mean_rmse,
        'rmse_std': std_rmse,
        'mae_mean': mean_mae,
        'r2_mean': mean_r2,
        'fold_rmses': fold_rmses
    }


def run_pscp_ablation(df, pretrain_conditions, n_folds=5):
    """Run PSCP with different pretraining conditions"""

    # First prepare data
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)

    # Get column names
    smiles_col = 'Smiles' if 'Smiles' in df.columns else 'SMILES'
    tg_col = 'Tg' if 'Tg' in df.columns else 'tg'

    valid_indices = []
    smiles_list = []
    graphs = []
    ecfps = []
    targets = []

    for idx, row in df.iterrows():
        clean = clean_polymer_smiles(row[smiles_col])
        if clean:
            graph = smiles_to_graph(clean)
            ecfp = compute_ecfp(clean)
            if graph is not None and ecfp.sum() > 0:
                valid_indices.append(idx)
                smiles_list.append(clean)
                graphs.append(graph)
                ecfps.append(ecfp)
                targets.append(row[tg_col])

    ecfps = np.array(ecfps)
    targets = np.array(targets)

    print(f"Valid samples with graphs: {len(targets)}")

    # Scale targets
    scaler = StandardScaler()
    targets_scaled = scaler.fit_transform(targets.reshape(-1, 1)).flatten()

    results = {}

    for condition in pretrain_conditions:
        print("\n" + "="*60)
        print(f"CONDITION: {condition}")
        print("="*60)

        # Load or create pretrained encoder
        if condition == 'no_pretrain':
            pretrain_model = None
        else:
            # Parse condition
            if condition == 'pretrain_10k':
                n_samples = 10000
            elif condition == 'pretrain_100k':
                n_samples = 100000
            elif condition == 'pretrain_1m':
                n_samples = None  # Use all
            else:
                n_samples = 10000

            # Load pretrain data
            pretrain_smiles = load_pretrain_data(max_samples=n_samples)

            # Build graphs
            print(f"Building pretrain graphs...")
            pretrain_graphs = []
            for s in pretrain_smiles:
                clean = clean_polymer_smiles(s)
                if clean:
                    g = smiles_to_graph(clean)
                    if g is not None:
                        pretrain_graphs.append(g)

            print(f"Valid pretrain graphs: {len(pretrain_graphs)}")

            if len(pretrain_graphs) > 0:
                # Pretrain
                pretrain_model = PretrainModel(node_feat_dim=ATOM_FEATURE_DIM, hidden_dim=256)
                print(f"Pretraining on {len(pretrain_graphs)} samples...")
                pretrain_model = train_pretrain(pretrain_model, pretrain_graphs, epochs=50, batch_size=256)
            else:
                pretrain_model = None

        # 5-fold CV
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_rmses = []
        fold_maes = []
        fold_r2s = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(graphs)):
            print(f"\nFold {fold+1}/{n_folds}")

            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]
            train_ecfps = ecfps[train_idx]
            val_ecfps = ecfps[val_idx]
            train_targets = targets_scaled[train_idx]
            val_targets = targets_scaled[val_idx]

            # Create model
            model = PSCPModel(node_feat_dim=ATOM_FEATURE_DIM, hidden_dim=256, ecfp_dim=2048)

            # Load pretrained weights if available
            if pretrain_model is not None:
                model.encoder.load_state_dict(pretrain_model.encoder.state_dict())

            # Fine-tune
            model = train_finetune(model, train_graphs, train_ecfps, train_targets,
                                   val_graphs, val_ecfps, val_targets,
                                   epochs=300, patience=40)

            # Evaluate
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                val_data = Batch.from_data_list(val_graphs).to(device)
                val_ecfps_t = torch.tensor(val_ecfps, dtype=torch.float32).to(device)
                pred_scaled = model(val_data, val_ecfps_t).cpu().numpy()

            # Inverse transform
            pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            true = targets[val_idx]

            rmse = np.sqrt(mean_squared_error(true, pred))
            mae = mean_absolute_error(true, pred)
            r2 = r2_score(true, pred)

            fold_rmses.append(rmse)
            fold_maes.append(mae)
            fold_r2s.append(r2)

            print(f"  Fold {fold+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        mean_mae = np.mean(fold_maes)
        mean_r2 = np.mean(fold_r2s)

        print(f"\n{'='*40}")
        print(f"{condition} RESULTS (5-fold CV)")
        print(f"RMSE: {mean_rmse:.2f} +/- {std_rmse:.2f}")
        print(f"MAE: {mean_mae:.2f}")
        print(f"R2: {mean_r2:.4f}")
        print(f"{'='*40}")

        results[condition] = {
            'rmse_mean': mean_rmse,
            'rmse_std': std_rmse,
            'mae_mean': mean_mae,
            'r2_mean': mean_r2,
            'fold_rmses': fold_rmses
        }

    return results


def main():
    """Run complete experiment"""
    print("="*60)
    print("PSCP FIXED EXPERIMENT")
    print(f"Started: {datetime.now()}")
    print("="*60)

    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Compute actual atom feature dimension
    global ATOM_FEATURE_DIM
    ATOM_FEATURE_DIM = get_actual_feature_dim()
    print(f"Atom feature dimension: {ATOM_FEATURE_DIM}")

    # Load data
    df = load_data()

    # Step 1: Verify baseline
    baseline_results = run_baseline_cv(df, n_folds=5)

    # Step 2: Run PSCP with ablations
    pretrain_conditions = ['no_pretrain', 'pretrain_10k', 'pretrain_100k']
    pscp_results = run_pscp_ablation(df, pretrain_conditions, n_folds=5)

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\nBaseline (MLP+ECFP4): {baseline_results['rmse_mean']:.2f} +/- {baseline_results['rmse_std']:.2f}")
    print("\nPSCP Results:")
    for cond, res in pscp_results.items():
        print(f"  {cond}: {res['rmse_mean']:.2f} +/- {res['rmse_std']:.2f}")

    # Find best
    best_condition = min(pscp_results.keys(), key=lambda k: pscp_results[k]['rmse_mean'])
    best_rmse = pscp_results[best_condition]['rmse_mean']

    print(f"\nBest PSCP: {best_condition} with RMSE = {best_rmse:.2f}")
    print(f"Target: RMSE < 35.0")
    print(f"Status: {'ACHIEVED' if best_rmse < 35.0 else 'NOT ACHIEVED'}")

    # Save results
    all_results = {
        'baseline': baseline_results,
        'pscp': pscp_results,
        'best_condition': best_condition,
        'best_rmse': best_rmse,
        'target_achieved': best_rmse < 35.0,
        'timestamp': datetime.now().isoformat()
    }

    with open('fixed_pscp_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to fixed_pscp_results.json")
    print(f"Completed: {datetime.now()}")

    return all_results


if __name__ == '__main__':
    results = main()

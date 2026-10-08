"""
Final Push - Ensemble approach combining best strategies
Focus on achieving RMSE < 35.0 through:
1. Simple, proven architecture (no over-engineering)
2. Best hyperparameters from previous experiments
3. Multiple seeds for ensemble
4. Proper early stopping
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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
RDLogger.DisableLog('rdApp.*')


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


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    if len(atom_features) == 0:
        return None
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
    if len(edge_indices) == 0:
        edge_index = torch.tensor([[], []], dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def compute_ecfp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


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
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + ecfp_dim, 256),
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


class PretrainModel(nn.Module):
    def __init__(self, node_feat_dim=155, hidden_dim=256):
        super().__init__()
        self.encoder = GINEncoder(node_feat_dim, hidden_dim, num_layers=3)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, data):
        graph_emb, _ = self.encoder(data.x, data.edge_index, data.batch)
        return self.projection(graph_emb)


def contrastive_loss(emb1, emb2, temperature=0.05):
    batch_size = emb1.size(0)
    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)
    all_emb = torch.cat([emb1, emb2], dim=0)
    sim_matrix = torch.mm(all_emb, all_emb.t()) / temperature
    labels = torch.arange(batch_size, device=emb1.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    mask = torch.eye(2 * batch_size, device=emb1.device).bool()
    sim_matrix.masked_fill_(mask, float('-inf'))
    return F.cross_entropy(sim_matrix, labels)


def augment_graph(data, node_drop_prob=0.15, edge_drop_prob=0.15):
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


def train_pretrain(model, graphs, epochs=80, batch_size=256, lr=0.001, temperature=0.05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  Pretraining (epochs={epochs}, temp={temperature})...", flush=True)

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
            loss = contrastive_loss(emb1, emb2, temperature)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {total_loss/max(num_batches,1):.4f}", flush=True)

    return model


def train_finetune(model, train_graphs, train_ecfps, train_targets,
                   val_graphs, val_ecfps, val_targets,
                   epochs=400, batch_size=32, lr=1e-4, patience=50):
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

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}: Val RMSE = {val_rmse:.2f} (best: {best_val_rmse:.2f})", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def main():
    print("="*70, flush=True)
    print("FINAL PUSH - ENSEMBLE APPROACH", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print("="*70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Load data
    data_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    df = pd.read_csv(data_path)
    smiles_col = 'Smiles' if 'Smiles' in df.columns else 'SMILES'
    tg_col = 'Tg' if 'Tg' in df.columns else 'tg'

    print(f"\nLoaded Tg: {len(df)} samples", flush=True)

    graphs, ecfps, targets = [], [], []
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
    scaler = StandardScaler()
    targets_scaled = scaler.fit_transform(targets.reshape(-1, 1)).flatten()

    # Load pretrain
    cache_file = 'pretrain_cache_100000.pkl'
    with open(cache_file, 'rb') as f:
        pretrain_graphs = pickle.load(f)
    print(f"Pretrain graphs: {len(pretrain_graphs)}", flush=True)

    # Train multiple models with different seeds for ensemble
    print("\n" + "="*70, flush=True)
    print("TRAINING ENSEMBLE MODELS", flush=True)
    print("="*70, flush=True)

    seeds = [42, 123, 456, 789, 1001]
    all_fold_predictions = {i: [] for i in range(5)}
    all_fold_true = {i: [] for i in range(5)}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*70}", flush=True)
        print(f"SEED {seed_idx+1}/{len(seeds)}: {seed}", flush=True)
        print("="*70, flush=True)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Pretrain
        pretrain_model = PretrainModel(node_feat_dim=node_feat_dim, hidden_dim=256)
        pretrain_model = train_pretrain(pretrain_model, pretrain_graphs, epochs=80, temperature=0.05)

        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(graphs)):
            print(f"\nFold {fold+1}/5", flush=True)

            train_graphs = [graphs[i] for i in train_idx]
            val_graphs = [graphs[i] for i in val_idx]
            train_ecfps = ecfps[train_idx]
            val_ecfps = ecfps[val_idx]
            train_targets = targets_scaled[train_idx]
            val_targets = targets_scaled[val_idx]

            model = PSCPModel(node_feat_dim=node_feat_dim, hidden_dim=256, ecfp_dim=2048)
            model.encoder.load_state_dict(pretrain_model.encoder.state_dict())
            model = train_finetune(model, train_graphs, train_ecfps, train_targets,
                                   val_graphs, val_ecfps, val_targets,
                                   epochs=400, patience=50)

            model.eval()
            model = model.to(device)

            with torch.no_grad():
                val_data = Batch.from_data_list(val_graphs).to(device)
                val_ecfps_t = torch.tensor(val_ecfps, dtype=torch.float32).to(device)
                pred_scaled = model(val_data, val_ecfps_t).cpu().numpy()

            pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            true = targets[val_idx]
            rmse = np.sqrt(mean_squared_error(true, pred))
            print(f"  Fold {fold+1}: RMSE={rmse:.2f}", flush=True)

            # Store predictions
            all_fold_predictions[fold].append(pred)
            all_fold_true[fold] = true  # Same true values for each seed

    # Ensemble predictions
    print("\n" + "="*70, flush=True)
    print("ENSEMBLE RESULTS", flush=True)
    print("="*70, flush=True)

    ensemble_rmses = []
    individual_rmses = []

    for fold in range(5):
        # Ensemble = average of all seeds
        ensemble_pred = np.mean(all_fold_predictions[fold], axis=0)
        true = all_fold_true[fold]
        ensemble_rmse = np.sqrt(mean_squared_error(true, ensemble_pred))
        ensemble_rmses.append(ensemble_rmse)

        # Individual seed results
        for seed_idx in range(len(seeds)):
            ind_pred = all_fold_predictions[fold][seed_idx]
            ind_rmse = np.sqrt(mean_squared_error(true, ind_pred))
            individual_rmses.append(ind_rmse)

        print(f"Fold {fold+1}: Ensemble RMSE = {ensemble_rmse:.2f}", flush=True)

    mean_ensemble = np.mean(ensemble_rmses)
    std_ensemble = np.std(ensemble_rmses)
    mean_individual = np.mean(individual_rmses)
    std_individual = np.std(individual_rmses)

    print(f"\n{'='*70}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print("="*70, flush=True)
    print(f"\nIndividual models (25 runs): {mean_individual:.2f} +/- {std_individual:.2f}", flush=True)
    print(f"Ensemble (5 seeds avg): {mean_ensemble:.2f} +/- {std_ensemble:.2f}", flush=True)
    print(f"\nTarget: RMSE < 35.0", flush=True)
    print(f"Status: {'ACHIEVED!' if mean_ensemble < 35.0 else 'NOT ACHIEVED'}", flush=True)

    # Save
    results = {
        'ensemble_mean': mean_ensemble,
        'ensemble_std': std_ensemble,
        'individual_mean': mean_individual,
        'individual_std': std_individual,
        'ensemble_folds': ensemble_rmses,
        'target_achieved': mean_ensemble < 35.0,
        'num_seeds': len(seeds),
        'timestamp': datetime.now().isoformat()
    }

    with open('final_ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to final_ensemble_results.json", flush=True)
    print(f"Completed: {datetime.now()}", flush=True)

    return results


if __name__ == '__main__':
    results = main()

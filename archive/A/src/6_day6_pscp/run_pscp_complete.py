#!/usr/bin/env python3
"""
Complete PSCP (Polymer-Specific Contrastive Pretraining) Experiment
Executes pretraining and fine-tuning with ablation study
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
import warnings
import json
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch


# ============================================================================
# UTILITIES
# ============================================================================

def smiles_to_graph(smiles):
    """Convert SMILES to PyG Data object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None

        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum() / 100.0,
                atom.GetDegree() / 10.0,
                1.0 if atom.GetIsAromatic() else 0.0,
                atom.GetFormalCharge() / 10.0,
                atom.GetTotalNumHs() / 10.0,
            ]
            atom_features.append(features)

        x = torch.tensor(atom_features, dtype=torch.float)

        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)
    except:
        return None


def compute_ecfp(smiles, radius=2, n_bits=2048):
    """Compute ECFP fingerprint"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except:
        return np.zeros(n_bits, dtype=np.float32)


# ============================================================================
# MODELS
# ============================================================================

class SimpleGIN(nn.Module):
    """Simple GIN encoder"""

    def __init__(self, input_dim=5, hidden_dim=256, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        h = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            h = bn(conv(h, edge_index))
        return global_mean_pool(h, batch)


class TgPredictor(nn.Module):
    """Tg prediction model"""

    def __init__(self, gin_encoder, hidden_dim=256, ecfp_dim=2048):
        super().__init__()
        self.encoder = gin_encoder
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + ecfp_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, batch, ecfp):
        graph_emb = self.encoder(x.float(), edge_index, batch)
        combined = torch.cat([graph_emb, ecfp.float()], dim=-1)
        return self.predictor(combined).squeeze(-1)


# ============================================================================
# DATASETS
# ============================================================================

class TgDataset(Dataset):
    """Tg dataset with precomputed features"""

    def __init__(self, csv_path):
        print(f"Loading Tg dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples")

        print("Pre-computing graphs and fingerprints...")
        self.graphs = []
        self.ecfps = []
        self.tgs = []

        for idx in tqdm(range(len(self.df)), desc="Processing"):
            row = self.df.iloc[idx]
            smiles = row['Smiles'].replace('*', '')
            if not smiles:
                smiles = 'C'

            graph = smiles_to_graph(smiles)
            if graph is None:
                graph = Data(x=torch.zeros(1, 5), edge_index=torch.zeros(2, 0, dtype=torch.long))
            self.graphs.append(graph)
            self.ecfps.append(compute_ecfp(smiles))
            self.tgs.append(float(row['Tg']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'graph': self.graphs[idx],
            'ecfp': torch.from_numpy(self.ecfps[idx]),
            'tg': torch.tensor(self.tgs[idx], dtype=torch.float32)
        }


def collate_fn(batch):
    graphs = [item['graph'] for item in batch]
    return {
        'graph': Batch.from_data_list(graphs),
        'ecfp': torch.stack([item['ecfp'] for item in batch]),
        'tg': torch.stack([item['tg'] for item in batch])
    }


class PretrainDataset(Dataset):
    """Dataset for pretraining"""

    def __init__(self, csv_path, max_samples=100000):
        print(f"Loading PI1M from {csv_path}...")
        df = pd.read_csv(csv_path)
        self.smiles_list = []
        for smi in df['SMILES'].tolist():
            if len(self.smiles_list) >= max_samples:
                break
            clean = smi.replace('*', '')
            if clean and len(clean) > 1:
                self.smiles_list.append(clean)
        print(f"Loaded {len(self.smiles_list)} SMILES")

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        graph = smiles_to_graph(self.smiles_list[idx])
        if graph is None:
            graph = Data(x=torch.zeros(1, 5), edge_index=torch.zeros(2, 0, dtype=torch.long))
        return {'graph': graph, 'idx': idx}


def collate_pretrain(batch):
    graphs = [item['graph'] for item in batch]
    return {'graph': Batch.from_data_list(graphs)}


# ============================================================================
# PRETRAINING
# ============================================================================

def pretrain_contrastive(model, dataloader, optimizer, device, epochs=5):
    """Simple contrastive pretraining"""
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            graphs = batch['graph'].to(device)

            # Get embeddings with dropout augmentation
            emb1 = model(graphs.x, graphs.edge_index, graphs.batch)
            emb2 = model(graphs.x, graphs.edge_index, graphs.batch)

            # Normalize
            emb1 = F.normalize(emb1, dim=-1)
            emb2 = F.normalize(emb2, dim=-1)

            # InfoNCE loss
            batch_size = graphs.num_graphs
            sim = torch.mm(emb1, emb2.t()) / 0.1
            labels = torch.arange(batch_size, device=device)
            loss = F.cross_entropy(sim, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"  Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")


# ============================================================================
# FINE-TUNING
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        graphs = batch['graph'].to(device)
        ecfp = batch['ecfp'].to(device)
        targets = batch['tg'].to(device)

        pred = model(graphs.x, graphs.edge_index, graphs.batch, ecfp)
        loss = F.mse_loss(pred, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            graphs = batch['graph'].to(device)
            ecfp = batch['ecfp'].to(device)
            target = batch['tg']

            pred = model(graphs.x, graphs.edge_index, graphs.batch, ecfp)
            preds.extend(pred.cpu().numpy())
            targets.extend(target.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    return {
        'rmse': np.sqrt(mean_squared_error(targets, preds)),
        'mae': mean_absolute_error(targets, preds),
        'r2': r2_score(targets, preds)
    }


def cross_validate(encoder, dataset, config, device, n_folds=5, epochs=150, patience=20):
    """5-fold cross-validation"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold+1}/{n_folds}")

        # Create model
        model = TgPredictor(encoder, config['hidden_dim'], config['ecfp_bits']).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # DataLoaders
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=32, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=32, shuffle=False, collate_fn=collate_fn
        )

        # Training
        best_rmse = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step()

            if val_metrics['rmse'] < best_rmse:
                best_rmse = val_metrics['rmse']
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 25 == 0:
                print(f"  Epoch {epoch+1}: Loss={train_loss:.2f}, Val RMSE={val_metrics['rmse']:.2f}, R2={val_metrics['r2']:.4f}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Final evaluation
        model.load_state_dict(best_state)
        final = evaluate(model, val_loader, device)
        print(f"  Fold {fold+1} Final: RMSE={final['rmse']:.2f}, MAE={final['mae']:.2f}, R2={final['r2']:.4f}")

        fold_results.append({
            'fold': fold + 1, 'rmse': final['rmse'], 'mae': final['mae'], 'r2': final['r2']
        })

    # Summary
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

def main():
    print("=" * 70)
    print("PSCP: Polymer-Specific Contrastive Pretraining")
    print("COMPLETE EXPERIMENT")
    print("=" * 70)
    print(f"Time: {datetime.now()}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = {
        'input_dim': 5,
        'hidden_dim': 256,
        'num_layers': 3,
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

    # ABLATION STUDY
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    conditions = [
        ('no_pretrain', 0),
        ('pretrain_10k', 10000),
        ('pretrain_100k', 100000),
    ]

    for cond_name, n_samples in conditions:
        print(f"\n{'='*70}")
        print(f"Condition: {cond_name}")
        print(f"{'='*70}")

        # Create encoder
        encoder = SimpleGIN(config['input_dim'], config['hidden_dim'], config['num_layers'])

        # Pretrain if needed
        if n_samples > 0:
            print(f"\nPretraining on {n_samples} samples...")
            pretrain_data = PretrainDataset(pi1m_path, max_samples=n_samples)
            pretrain_loader = DataLoader(
                pretrain_data, batch_size=64, shuffle=True, collate_fn=collate_pretrain, num_workers=0
            )
            optimizer = optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=0.01)
            encoder.to(device)
            pretrain_contrastive(encoder, pretrain_loader, optimizer, device, epochs=5)

            # Save pretrained model
            torch.save(encoder.state_dict(), os.path.join(output_dir, f'pretrained_{cond_name}.pt'))
        else:
            print("Skipping pretraining (random initialization)")
            encoder.to(device)

        # Fine-tune
        print("\nFine-tuning on Tg dataset...")
        results = cross_validate(encoder, tg_dataset, config, device, n_folds=5, epochs=150, patience=20)

        # Save results
        for fr in results['fold_results']:
            all_results.append({
                'experiment': 'ablation',
                'condition': cond_name,
                'fold': fr['fold'],
                'rmse': fr['rmse'],
                'mae': fr['mae'],
                'r2': fr['r2']
            })

        all_results.append({
            'experiment': 'ablation',
            'condition': cond_name,
            'fold': 'mean',
            'rmse': results['mean_rmse'],
            'mae': np.mean([r['mae'] for r in results['fold_results']]),
            'r2': results['mean_r2']
        })

    # SAVE AND REPORT
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'pscp_results.csv'), index=False)

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    print("\n[RESULTS]")
    means = results_df[results_df['fold'] == 'mean']
    print("Condition       | RMSE (K)   | MAE (K)    | R2        | Std RMSE")
    print("-" * 65)
    for _, row in means.iterrows():
        std_rmse = results_df[(results_df['condition'] == row['condition']) &
                              (results_df['fold'] != 'mean')]['rmse'].std()
        print(f"{row['condition']:15} | {row['rmse']:9.2f} | {row['mae']:9.2f} | {row['r2']:9.4f} | {std_rmse:9.2f}")

    best_idx = means['rmse'].idxmin()
    best_row = means.loc[best_idx]
    best_rmse = best_row['rmse']
    best_cond = best_row['condition']

    print("\n[KEY FINDINGS]")
    print(f"- Best RMSE: {best_rmse:.2f} K ({best_cond})")
    print(f"- Results saved to: {os.path.join(output_dir, 'pscp_results.csv')}")
    print(f"- Completion time: {datetime.now()}")

    # Save summary
    summary = {
        'best_condition': best_cond,
        'best_rmse': float(best_rmse),
        'all_results': all_results,
        'timestamp': str(datetime.now())
    }
    with open(os.path.join(output_dir, 'pscp_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return results_df


if __name__ == '__main__':
    results = main()

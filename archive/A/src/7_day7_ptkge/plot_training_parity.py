"""
Single-run training to produce training curve + parity plot.

Protocol:
- Split: 72% train / 8% val / 20% test  (seed=42, single split)
- Loops over all 4 embedding strategies: onehot, random, physics, learnable
- Loads strategy-specific pretrained checkpoint (skip for onehot)
- 200 epochs, no early stopping; track best-by-val state
- Caches per-strategy history + predictions
- Plots ONLY the physics strategy (per request)
"""

import os
import sys
import random
import warnings
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PTKGEModel, compute_ecfp4_fingerprint

warnings.filterwarnings('ignore')


STRATEGIES = ['onehot', 'random', 'physics', 'learnable']
PLOT_STRATEGY = 'physics'
SPLIT_TAG = '72_8_20'
SEED = 42
EPOCHS = 200


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========== Data helpers (mirror train.py) ==========

def polymer_smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        processed = smiles.replace('*', '[H]')
        return Chem.MolFromSmiles(processed)
    except Exception:
        return None


def mol_to_graph(mol: Chem.Mol) -> Optional[Data]:
    if mol is None:
        return None
    try:
        atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
        if not atoms:
            return None
        x = torch.tensor(atoms, dtype=torch.long)
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx(); j = bond.GetEndAtomIdx()
            edges.append([i, j]); edges.append([j, i])
        if not edges:
            edges = [[0, 0]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
    except Exception:
        return None


class PolymerDataset(Dataset):
    def __init__(self, smiles_list, targets, fp_dim: int = 2048):
        self.smiles_list = smiles_list
        self.targets = targets
        self.fp_dim = fp_dim
        self.graphs: List[Data] = []
        self.fingerprints = []
        self.valid_indices = []
        print(f"Processing {len(smiles_list)} samples...")
        for i, smi in enumerate(smiles_list):
            mol = polymer_smiles_to_mol(smi)
            graph = mol_to_graph(mol) if mol else None
            fp = compute_ecfp4_fingerprint(smi, n_bits=fp_dim) if mol else None
            if graph is not None and fp is not None and fp.sum() > 0:
                self.graphs.append(graph)
                self.fingerprints.append(fp)
                self.valid_indices.append(i)
        print(f"Valid samples: {len(self.valid_indices)} / {len(smiles_list)}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.fingerprints[idx], float(self.targets[self.valid_indices[idx]])


def collate_fn(batch):
    graphs, fps, targets = zip(*batch)
    batched = Batch.from_data_list(graphs)
    fps = torch.tensor(np.array(fps), dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    return batched, fps, targets


# ========== Train / Eval ==========

def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    preds, tgts = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for graphs, fps, tgt in loader:
            graphs = graphs.to(device); fps = fps.to(device); tgt = tgt.to(device)
            if train:
                optimizer.zero_grad()
            pred = model(graphs.x, graphs.edge_index, graphs.batch, fps).squeeze(-1)
            loss = criterion(pred, tgt)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            preds.extend(pred.detach().cpu().numpy())
            tgts.extend(tgt.cpu().numpy())
    preds = np.array(preds); tgts = np.array(tgts)
    mse = total_loss / max(len(loader), 1)
    return mse, preds, tgts


# ========== Per-strategy run ==========

def run_strategy(strategy: str, splits, device, cache_path: str, ckpt_dir: str):
    if os.path.exists(cache_path):
        print(f"\n[{strategy}] Loading cached run from {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return {
            'train_losses': d['train_losses'], 'val_losses': d['val_losses'],
            'y_train': d['y_train'], 'p_train': d['p_train'],
            'y_val':   d['y_val'],   'p_val':   d['p_val'],
            'y_test':  d['y_test'],  'p_test':  d['p_test'],
        }

    train_smiles, val_smiles, test_smiles, train_targets, val_targets, test_targets = splits
    print(f"\n[{strategy}] Building datasets...")
    train_ds = PolymerDataset(train_smiles, train_targets)
    val_ds   = PolymerDataset(val_smiles,   val_targets)
    test_ds  = PolymerDataset(test_smiles,  test_targets)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = PTKGEModel(
        num_layers=5, hidden_dim=256,
        embedding_strategy=strategy, embedding_dim=8,
        fp_dim=2048, dropout=0.1,
    ).to(device)

    ckpt_path = os.path.join(ckpt_dir, f'pretrained_{strategy}.pt')
    if strategy != 'onehot' and os.path.exists(ckpt_path):
        print(f"[{strategy}] Loading pretrained weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print(f"[{strategy}] No pretrained checkpoint; training from scratch.")

    optimizer = AdamW([
        {'params': model.get_encoder_params(),   'lr': 1e-4},
        {'params': model.get_predictor_params(), 'lr': 1e-3},
    ], weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None
    for ep in range(1, EPOCHS + 1):
        tr_mse, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        vl_mse, _, _ = run_epoch(model, val_loader,   None,      criterion, device, train=False)
        train_losses.append(tr_mse); val_losses.append(vl_mse)
        if vl_mse < best_val:
            best_val = vl_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == EPOCHS:
            print(f"[{strategy}] Epoch {ep:>3}/{EPOCHS} - Train MSE: {tr_mse:.2f} - Val MSE: {vl_mse:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    _, p_train, y_train = run_epoch(model, train_loader, None, criterion, device, train=False)
    _, p_val,   y_val   = run_epoch(model, val_loader,   None, criterion, device, train=False)
    _, p_test,  y_test  = run_epoch(model, test_loader,  None, criterion, device, train=False)

    np.savez_compressed(
        cache_path,
        train_losses=np.array(train_losses), val_losses=np.array(val_losses),
        y_train=y_train, p_train=p_train,
        y_val=y_val, p_val=p_val,
        y_test=y_test, p_test=p_test,
    )
    print(f"[{strategy}] Cached to {cache_path}")

    rmse_test = float(np.sqrt(mean_squared_error(y_test, p_test)))
    r2_test   = float(r2_score(y_test, p_test))
    print(f"[{strategy}] TEST: RMSE={rmse_test:.4f}, R²={r2_test:.4f}")

    return {
        'train_losses': np.array(train_losses), 'val_losses': np.array(val_losses),
        'y_train': y_train, 'p_train': p_train,
        'y_val':   y_val,   'p_val':   p_val,
        'y_test':  y_test,  'p_test':  p_test,
    }


# ========== Plotting ==========

def make_figure(run, save_path: str):
    train_losses = run['train_losses']; val_losses = run['val_losses']
    y_train = run['y_train']; p_train = run['p_train']
    y_val   = run['y_val'];   p_val   = run['p_val']
    y_test  = run['y_test'];  p_test  = run['p_test']

    rmse_train = float(np.sqrt(mean_squared_error(y_train, p_train)))
    rmse_val   = float(np.sqrt(mean_squared_error(y_val,   p_val)))
    rmse_test  = float(np.sqrt(mean_squared_error(y_test,  p_test)))
    r2_train = float(r2_score(y_train, p_train))
    r2_val   = float(r2_score(y_val,   p_val))
    r2_test  = float(r2_score(y_test,  p_test))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: training curve
    ax = axes[0]
    n_epochs = len(train_losses)
    epochs = np.arange(n_epochs)
    ax.plot(epochs, train_losses, color='tab:blue', linestyle='-',  linewidth=2.4, label='Training Loss')
    ax.plot(epochs, val_losses,   color='tab:red',  linestyle='--', linewidth=2.4, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('MSE Loss', fontsize=13)
    ax.set_xlim(0, n_epochs)
    ax.set_ylim(bottom=0)
    ax.grid(True, color='#B0B0B0', linestyle='-', linewidth=0.6, alpha=0.6)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # Right: parity plot
    ax = axes[1]
    line_lo, line_hi = -200.0, 500.0
    ax.plot([line_lo, line_hi], [line_lo, line_hi],
            color='black', linestyle='--', linewidth=1.2)
    ax.scatter(y_train, p_train, c='tab:blue',  marker='o', s=28, alpha=0.5, edgecolors='none',
               label=f'Train (RMSE={rmse_train:.3f}, R²={r2_train:.3f})')
    ax.scatter(y_val,   p_val,   c='tab:red',   marker='s', s=30, alpha=0.7, edgecolors='none',
               label=f'Validation (RMSE={rmse_val:.3f}, R²={r2_val:.3f})')
    ax.scatter(y_test,  p_test,  c='tab:green', marker='^', s=30, alpha=0.65, edgecolors='none',
               label=f'Test (RMSE={rmse_test:.3f}, R²={r2_test:.3f})')
    ax.set_xlabel('Experimental Values', fontsize=13)
    ax.set_ylabel('Predicted Values', fontsize=13)
    ax.set_xlim(line_lo, line_hi); ax.set_ylim(line_lo, line_hi)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, color='#B0B0B0', linestyle='-', linewidth=0.6, alpha=0.6)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {save_path}")


# ========== Main ==========

def main():
    set_seed(SEED)

    base_dir = '/home/home_public/wuzl/AIscientist/archive/ablation2archive/src/7_day7_ptkge'
    data_dir = '/home/home_public/wuzl/AIscientist/archive/ablation2archive/src/alldataset'
    tg_path = os.path.join(data_dir, 'Tg.csv')
    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    cache_dir = os.path.join(base_dir, f'cache_{SPLIT_TAG}')
    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Tg
    print("\nLoading Tg dataset...")
    tg_df = pd.read_csv(tg_path)
    smiles_all = tg_df['Smiles'].tolist()
    targets_all = tg_df['Tg'].values.astype(np.float32)
    print(f"Tg dataset: {len(smiles_all)} samples; target range "
          f"[{targets_all.min():.1f}, {targets_all.max():.1f}]")

    # 72/8/20 split: first cut 20% test, then 10% of remaining 80% = 8% val
    idx = np.arange(len(smiles_all))
    train_val_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=SEED)
    train_idx, val_idx      = train_test_split(train_val_idx, test_size=0.10, random_state=SEED)
    train_smiles = [smiles_all[i] for i in train_idx]
    val_smiles   = [smiles_all[i] for i in val_idx]
    test_smiles  = [smiles_all[i] for i in test_idx]
    train_targets = targets_all[train_idx]
    val_targets   = targets_all[val_idx]
    test_targets  = targets_all[test_idx]
    splits = (train_smiles, val_smiles, test_smiles, train_targets, val_targets, test_targets)
    print(f"Split ({SPLIT_TAG}, seed={SEED}): "
          f"train={len(train_smiles)}, val={len(val_smiles)}, test={len(test_smiles)} "
          f"(sum={len(train_smiles)+len(val_smiles)+len(test_smiles)})")

    # Run all 4 strategies
    runs = {}
    for strategy in STRATEGIES:
        cache_path = os.path.join(cache_dir, f'run_{strategy}.npz')
        runs[strategy] = run_strategy(strategy, splits, device, cache_path, ckpt_dir)

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Strategy':<12}{'Train RMSE':>14}{'Val RMSE':>12}{'Test RMSE':>12}{'Test R²':>10}")
    print("-" * 70)
    for strategy, r in runs.items():
        rmse_tr = np.sqrt(mean_squared_error(r['y_train'], r['p_train']))
        rmse_va = np.sqrt(mean_squared_error(r['y_val'],   r['p_val']))
        rmse_te = np.sqrt(mean_squared_error(r['y_test'],  r['p_test']))
        r2_te   = r2_score(r['y_test'], r['p_test'])
        print(f"{strategy:<12}{rmse_tr:>14.4f}{rmse_va:>12.4f}{rmse_te:>12.4f}{r2_te:>10.4f}")
    print("=" * 70)

    # Plot every strategy (cache already exists, so this is cheap)
    for strategy in STRATEGIES:
        make_figure(runs[strategy],
                    save_path=os.path.join(figures_dir, f'training_and_parity_{strategy}_{SPLIT_TAG}.png'))


if __name__ == '__main__':
    main()

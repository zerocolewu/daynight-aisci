"""
Phase 3: TCP Pre-training with thermodynamic consistency loss.

Trains GIN encoder with multi-property prediction and consistency constraints.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
sys.path.append('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT')
from dataset import mol_to_graph, get_feature_dimensions
from tcp_dataset import PolymerMultiDataset, collate_multi_target
import pandas as pd

def torch_load_safe(path):
    """Safe torch.load wrapper for loading data with numpy arrays."""
    import torch
    return torch.load(path, weights_only=False)


def torch_load_state(path):
    """Safe torch.load wrapper for loading model state_dict."""
    import torch
    return torch.load(path, weights_only=False)



class TCPModel(nn.Module):
    """
    TCP Model: GIN encoder with multi-property prediction head.
    Predicts Tg, Tm, Egc simultaneously.
    """

    def __init__(self, num_features, hidden_dim=256, num_layers=5):
        super(TCPModel, self).__init__()

        # Encoder
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)

        # Input projection
        self.convs.append(GINConv(nn.Linear(num_features, hidden_dim)))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Linear(hidden_dim, hidden_dim)))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Multi-property prediction heads
        self.tg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.tm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.egc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, batch, return_features=False):
        """
        Forward pass.

        Args:
            batch: PyG batched data
            return_features: If True, return encoded features

        Returns:
            If return_features: encoded features
            Else: (tg_pred, tm_pred, egc_pred)
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # GIN encoder
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i > 0:
                x = self.dropout(x)

        # Global pooling
        x = global_add_pool(x, batch_idx)

        if return_features:
            return x

        # Multi-property predictions
        tg_pred = self.tg_head(x).squeeze()
        tm_pred = self.tm_head(x).squeeze()
        egc_pred = self.egc_head(x).squeeze()

        return tg_pred, tm_pred, egc_pred


class TCPLoss:
    """Thermodynamic consistency loss functions."""

    @staticmethod
    def prediction_loss(tg_pred, tm_pred, egc_pred, tg_true, tm_true, egc_true):
        """
        Standard MSE loss for each property.
        """
        loss_tg = F.mse_loss(tg_pred, tg_true)
        loss_tm = F.mse_loss(tm_pred, tm_true)
        loss_egc = F.mse_loss(egc_pred, egc_true)

        return loss_tg, loss_tm, loss_egc

    @staticmethod
    def consistency_loss(tg_pred, tm_pred, egc_pred):
        """
        Thermodynamic consistency penalty.

        Constraints:
        1. Tg < Tm (glass transition below melting)
        2. Tg/Tm ratio should be in reasonable range (~0.5-0.8)
        3. Egc should correlate with Tg
        """
        # Penalty 1: Tg >= Tm (should be Tg < Tm)
        penalty_tg_ge_tm = F.relu(tg_pred - tm_pred).mean() * 10.0

        # Penalty 2: Tg/Tm ratio deviations
        ratio = tg_pred / (tm_pred + 1e-8)
        # Ideal ratio around 0.6-0.7 for most polymers
        target_ratio = 0.65
        penalty_ratio = F.relu(torch.abs(ratio - target_ratio) - 0.2).mean() * 5.0

        # Penalty 3: Egc should correlate with Tg (roughly Egc ~ Tg/10 + constant)
        expected_egc = tg_pred / 10.0 + 30.0
        penalty_egc = F.relu(torch.abs(egc_pred - expected_egc) - 20.0).mean() * 0.5

        total_penalty = penalty_tg_ge_tm + penalty_ratio + penalty_egc

        return total_penalty, {
            'tg_ge_tm': penalty_tg_ge_tm.item(),
            'ratio': penalty_ratio.item(),
            'egc': penalty_egc.item(),
        }


def train_epoch_tcp(model, loader, optimizer, device, consistency_weight=0.1):
    """Train for one epoch with TCP loss."""
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_cons_loss = 0
    consistency_breakdown = {'tg_ge_tm': 0, 'ratio': 0, 'egc': 0}

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        tg_pred, tm_pred, egc_pred = model(batch)

        # Get targets
        tg_true = batch.multi_target[:, 0]
        tm_true = batch.multi_target[:, 1]
        egc_true = batch.multi_target[:, 2]

        # Prediction loss
        loss_tg, loss_tm, loss_egc = TCPLoss.prediction_loss(
            tg_pred, tm_pred, egc_pred, tg_true, tm_true, egc_true
        )
        pred_loss = loss_tg + loss_tm + loss_egc

        # Consistency loss
        cons_loss, breakdown = TCPLoss.consistency_loss(tg_pred, tm_pred, egc_pred)

        # Total loss
        loss = pred_loss + consistency_weight * cons_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_pred_loss += pred_loss.item() * batch.num_graphs
        total_cons_loss += cons_loss.item() * batch.num_graphs

        for key in consistency_breakdown:
            consistency_breakdown[key] += breakdown[key] * batch.num_graphs

    n = len(loader.dataset)
    return {
        'total': total_loss / n,
        'pred': total_pred_loss / n,
        'cons': total_cons_loss / n,
        **{k: v / n for k, v in consistency_breakdown.items()}
    }


def evaluate_tcp(model, loader, device):
    """Evaluate TCP model."""
    model.eval()
    total_loss = 0

    tg_preds, tm_preds, egc_preds = [], [], []
    tg_true, tm_true, egc_true = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            tg_pred, tm_pred, egc_pred = model(batch)

            tg_targets = batch.multi_target[:, 0]
            tm_targets = batch.multi_target[:, 1]
            egc_targets = batch.multi_target[:, 2]

            loss_tg = F.mse_loss(tg_pred, tg_targets)
            loss_tm = F.mse_loss(tm_pred, tm_targets)
            loss_egc = F.mse_loss(egc_pred, egc_targets)

            total_loss += (loss_tg + loss_tm + loss_egc).item() * batch.num_graphs

            tg_preds.extend(tg_pred.cpu().numpy())
            tm_preds.extend(tm_pred.cpu().numpy())
            egc_preds.extend(egc_pred.cpu().numpy())

            tg_true.extend(tg_targets.cpu().numpy())
            tm_true.extend(tm_targets.cpu().numpy())
            egc_true.extend(egc_targets.cpu().numpy())

    # Compute RMSE for each property
    tg_preds = np.array(tg_preds)
    tm_preds = np.array(tm_preds)
    egc_preds = np.array(egc_preds)
    tg_true = np.array(tg_true)
    tm_true = np.array(tm_true)
    egc_true = np.array(egc_true)

    rmse_tg = np.sqrt(np.mean((tg_preds - tg_true)**2))
    rmse_tm = np.sqrt(np.mean((tm_preds - tm_true)**2))
    rmse_egc = np.sqrt(np.mean((egc_preds - egc_true)**2))

    # Thermodynamic consistency check
    consistent = (tm_preds > tg_preds).sum()
    consistency_pct = 100 * consistent / len(tg_preds)

    return {
        'loss': total_loss / len(loader.dataset),
        'rmse_tg': rmse_tg,
        'rmse_tm': rmse_tm,
        'rmse_egc': rmse_egc,
        'consistency_pct': consistency_pct
    }


def tcp_pretraining():
    """
    Complete Phase 3: TCP Pre-training.
    """
    print("=" * 80)
    print("PHASE 3: TCP PRE-TRAINING")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load pseudo-labels
    print("\nLoading pseudo-labels...")
    pseudo_labels = torch_load_safe('/home/home_public/aisci/AIscientist/src/7_day8_tcp/PI1M_pseudo_labels.pt')
    print(f"Loaded {len(pseudo_labels['smiles'])} polymers")
    print(f"Valid samples: {pseudo_labels['valid'].sum()}")

    # Load selected indices
    print("\nLoading selected indices...")
    selected_indices = torch_load_safe('/home/home_public/aisci/AIscientist/src/7_day8_tcp/selected_indices.pt')
    print(f"Selected {len(selected_indices)} samples for pre-training")

    # Create data list for selected samples
    print("\nPreparing selected PI1M samples...")
    pi1m_df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv')

    selected_data = []
    selected_targets = []

    valid_indices = np.where(pseudo_labels['valid'])[0]

    for idx in tqdm(selected_indices, desc="Processing selected samples"):
        real_idx = valid_indices[idx]
        # Handle both 'SMILES' and 'Smiles' column names
        smiles = pi1m_df.iloc[real_idx].get('SMILES', pi1m_df.iloc[real_idx].get('Smiles', None))
        if smiles is None:
            continue

        data = mol_to_graph(smiles, target=None)
        if data is not None:
            selected_data.append(data)
            selected_targets.append(pseudo_labels['targets'][real_idx])

    print(f"Created {len(selected_data)} graph samples")

    # Create dataset
    selected_targets = np.array(selected_targets, dtype=np.float32)
    dataset = PolymerMultiDataset(selected_data, selected_targets)

    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.1,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    atom_dim, bond_dim = get_feature_dimensions()
    print(f"\nFeature dimensions: atom={atom_dim}, bond={bond_dim}")

    model = TCPModel(num_features=atom_dim, hidden_dim=256, num_layers=5).to(device)

    # Data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0,
                             collate_fn=collate_multi_target)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0,
                           collate_fn=collate_multi_target)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print("\nStarting TCP pre-training...")
    print(f"Epochs: 50")
    print(f"Batch size: 128")
    print(f"Consistency weight: 0.1")

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    for epoch in range(50):
        # Train
        train_metrics = train_epoch_tcp(model, train_loader, optimizer, device,
                                       consistency_weight=0.1)

        # Validate
        val_metrics = evaluate_tcp(model, val_loader, device)

        scheduler.step(val_metrics['loss'])

        print(f"\nEpoch {epoch+1:2d}/{50}:")
        print(f"  Train - Loss: {train_metrics['total']:.4f}, "
              f"Pred: {train_metrics['pred']:.4f}, Cons: {train_metrics['cons']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"RMSE(Tg): {val_metrics['rmse_tg']:.2f}, "
              f"RMSE(Tm): {val_metrics['rmse_tm']:.2f}, "
              f"RMSE(Egc): {val_metrics['rmse_egc']:.2f}")
        print(f"  Consistency: {val_metrics['consistency_pct']:.1f}% satisfy Tg < Tm")

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, '/home/home_public/aisci/AIscientist/src/7_day8_tcp/TCP_pretrained_model.pt')
            print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*80}")
    print("TCP PRE-TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: /home/home_public/aisci/AIscientist/src/7_day8_tcp/TCP_pretrained_model.pt")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Check for selected indices file
    selected_indices_path = '/home/home_public/aisci/AIscientist/src/7_day8_tcp/selected_indices.pt'
    if not os.path.exists(selected_indices_path):
        print("\nERROR: selected_indices.pt not found!")
        print("Please run Phase 2 (distribution_aware_sampling) first.")
        sys.exit(1)

    # Load selected indices
    print("\nLoading selected indices...")
    selected_indices = torch_load_safe(selected_indices_path)
    print(f"Selected {len(selected_indices)} samples")

    # Load pseudo-labels
    print("\nLoading pseudo-labels...")
    pseudo_labels = torch_load_safe('/home/home_public/aisci/AIscientist/src/7_day8_tcp/PI1M_pseudo_labels.pt')
    print(f"Loaded {len(pseudo_labels['smiles'])} polymers")

    # Get feature dimensions
    atom_dim, bond_dim = get_feature_dimensions()
    print(f"\nFeature dimensions: atom={atom_dim}, bond={bond_dim}")

    # Create model
    model = TCPModel(num_features=atom_dim, hidden_dim=256, num_layers=5).to(device)

    # Load PI1M data for selected samples only
    print(f"\nLoading PI1M data for {len(selected_indices)} selected samples...")
    pi1m_df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv')

    # Process only selected samples
    from dataset import mol_to_graph
    from tqdm import tqdm

    pi1m_data_list = []
    for idx in tqdm(selected_indices, desc="Processing PI1M samples"):
        row = pi1m_df.iloc[idx]
        smiles = row.get('SMILES', row.get('smiles', None))
        if smiles is None:
            continue
        data = mol_to_graph(smiles, target=None)
        if data is not None:
            pi1m_data_list.append(data)

    print(f"Loaded {len(pi1m_data_list)} valid samples")

    # Create dataset
    selected_pseudo_labels = pseudo_labels['pseudo_labels'].iloc[selected_indices[:len(pi1m_data_list)]].tolist()
    pretrain_dataset = PolymerMultiDataset(pi1m_data_list, selected_pseudo_labels)

    # Data loaders with custom collate function
    train_loader = DataLoader(pretrain_dataset, batch_size=128, shuffle=True, num_workers=0,
                             collate_fn=collate_multi_target)
    val_loader = DataLoader(pretrain_dataset, batch_size=256, shuffle=False, num_workers=0,
                           collate_fn=collate_multi_target)
"""
Phase 2: Distribution-aware Sampling using epistemic uncertainty.

Trains a simple GIN on Tg dataset, computes uncertainty on PI1M,
and selects top 100K samples with lowest uncertainty.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT')
from dataset import load_and_split_data, mol_to_graph
from tcp_dataset import load_tg_data
import os

def torch_load_safe(path):
    """Safe torch.load wrapper for loading data with numpy arrays."""
    import torch
    return torch.load(path, weights_only=False)


def torch_load_state(path):
    """Safe torch.load wrapper for loading model state_dict."""
    import torch
    return torch.load(path, weights_only=False)



class SimpleGIN(nn.Module):
    """Simple GIN for uncertainty estimation."""

    def __init__(self, num_features, hidden_dim=128, num_layers=3):
        super(SimpleGIN, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GINConv(nn.Linear(num_features, hidden_dim)))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Linear(hidden_dim, hidden_dim)))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        x = global_add_pool(x, batch)
        x = self.fc(x)
        return x.squeeze()


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        loss = F.mse_loss(pred, batch.y.squeeze())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y.squeeze())

            total_loss += loss.item() * batch.num_graphs
            predictions.extend(pred.cpu().numpy())
            targets.extend(batch.y.squeeze().cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)
    rmse = np.sqrt(np.mean((predictions - targets)**2))

    return total_loss / len(loader.dataset), rmse


def compute_epistemic_uncertainty(model, data_list, device, n_ensemble=5):
    """
    Compute epistemic uncertainty using ensemble variance.

    Uses MC dropout at test time to estimate uncertainty.
    """
    model.eval()
    uncertainties = []

    # Enable dropout during inference
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    loader = DataLoader(data_list, batch_size=256, shuffle=False)

    all_predictions = []

    with torch.no_grad():
        for _ in range(n_ensemble):
            predictions = []
            for batch in tqdm(loader, desc=f"Ensemble {_+1}/{n_ensemble}"):
                batch = batch.to(device)
                pred = model(batch)
                predictions.extend(pred.cpu().numpy())

            all_predictions.append(np.array(predictions))

    # Compute variance across ensemble
    all_predictions = np.array(all_predictions)  # [n_ensemble, N]
    uncertainties = np.var(all_predictions, axis=0)  # [N]

    return uncertainties


def select_low_uncertainty_samples(data_list, uncertainties, n_samples=100000):
    """Select samples with lowest uncertainty."""
    # Get indices of sorted uncertainties
    sorted_indices = np.argsort(uncertainties)

    # Select top n_samples
    selected_indices = sorted_indices[:n_samples]

    print(f"\nUncertainty statistics:")
    print(f"  Mean: {uncertainties.mean():.6f}")
    print(f"  Std: {uncertainties.std():.6f}")
    print(f"  Min: {uncertainties.min():.6f}")
    print(f"  Max: {uncertainties.max():.6f}")
    print(f"\nSelected {n_samples} samples with lowest uncertainty")
    print(f"  Uncertainty range: [{uncertainties[selected_indices].min():.6f}, {uncertainties[selected_indices].max():.6f}]")

    return selected_indices


def distribution_aware_sampling():
    """
    Complete Phase 2: Distribution-aware sampling.
    """
    print("=" * 80)
    print("PHASE 2: DISTRIBUTION-AWARE SAMPLING")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load Tg data
    print("\nLoading Tg dataset...")
    tg_data_list, tg_targets = load_tg_data(
        '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    )

    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(tg_data_list)),
        test_size=0.1,
        random_state=42
    )

    train_data = [tg_data_list[i] for i in train_idx]
    val_data = [tg_data_list[i] for i in val_idx]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Get feature dimensions
    from tcp_dataset import load_pi1m_data
    sys.path.append('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT')
    from dataset import get_feature_dimensions
    atom_dim, bond_dim = get_feature_dimensions()

    # Create model
    print("\nCreating SimpleGIN model...")
    model = SimpleGIN(num_features=atom_dim, hidden_dim=128, num_layers=3).to(device)

    # Training setup
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Train model
    print("\nTraining uncertainty estimation model...")
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    for epoch in range(100):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_rmse = evaluate(model, val_loader, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val RMSE={val_rmse:.2f} K")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(),
                      '/home/home_public/aisci/AIscientist/src/7_day8_tcp/uncertainty_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    # Load best model
    model.load_state_dict(torch.load(
        '/home/home_public/aisci/AIscientist/src/7_day8_tcp/uncertainty_model.pt'
    ))

    # Load PI1M data
    print("\n" + "=" * 80)
    print("Loading PI1M data for uncertainty estimation...")
    print("=" * 80)

    # Load pseudo-labels to get valid samples
    pseudo_labels = torch_load_safe('/home/home_public/aisci/AIscientist/src/7_day8_tcp/PI1M_pseudo_labels.pt')
    valid_mask = pseudo_labels['valid']

    # Load and process PI1M
    print("\nProcessing PI1M polymers...")
    pi1m_df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv')
    pi1m_data_list = []

    # Only process valid samples
    for i in tqdm(range(len(pi1m_df)), desc="Loading PI1M"):
        if valid_mask[i]:
            # Handle both 'SMILES' and 'Smiles' column names
            smiles = pi1m_df.iloc[i].get('SMILES', pi1m_df.iloc[i].get('Smiles', None))
            if smiles is None:
                continue
            data = mol_to_graph(smiles, target=None)
            if data is not None:
                pi1m_data_list.append(data)

    print(f"\nLoaded {len(pi1m_data_list)} valid PI1M samples")

    # Compute uncertainty
    print("\nComputing epistemic uncertainty...")
    uncertainties = compute_epistemic_uncertainty(model, pi1m_data_list, device, n_ensemble=5)

    # Select samples
    print("\nSelecting low-uncertainty samples...")
    selected_indices = select_low_uncertainty_samples(pi1m_data_list, uncertainties, n_samples=100000)

    # Save selected indices
    output_path = '/home/home_public/aisci/AIscientist/src/7_day8_tcp/selected_indices.pt'
    torch.save(selected_indices, output_path)

    print(f"\n{'='*80}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*80}")
    print(f"Selected {len(selected_indices)} low-uncertainty samples")
    print(f"Saved to: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import pandas as pd

    def distribution_aware_sampling():
        """
        Complete Phase 2: Distribution-aware sampling.

        Returns selected_indices tensor for use in Phase 3.
        """
        print("=" * 80)
        print("PHASE 2: DISTRIBUTION-AWARE SAMPLING")
        print("=" * 80)

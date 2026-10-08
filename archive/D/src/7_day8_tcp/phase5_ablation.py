"""
Phase 5: Ablation Studies.

Tests:
1. No pseudo-labels: Pre-train with L_consistency ONLY
2. No distribution sampling: Random 100K PI1M samples
3. No pre-training: Baseline (already done in Phase 4)
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
from dataset import mol_to_graph, get_feature_dimensions
from tcp_dataset import load_tg_data, create_k_fold_splits
from phase4_finetuning import finetune_fold
from phase3_pretraining import TCPModel, train_epoch_tcp, evaluate_tcp
from tcp_dataset import PolymerMultiDataset
import pandas as pd

def torch_load_state(path):
    """Safe torch.load wrapper that handles PyTorch 2.6 weights_only changes."""
    import torch
    return torch.load(path, weights_only=False)



class TCPModelConsistencyOnly(nn.Module):
    """TCP model trained with consistency loss only (no pseudo-labels)."""

    def __init__(self, num_features, hidden_dim=256, num_layers=5):
        super(TCPModelConsistencyOnly, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)

        self.convs.append(GINConv(nn.Linear(num_features, hidden_dim)))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Linear(hidden_dim, hidden_dim)))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Multi-property heads (still needed for consistency loss)
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

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i > 0:
                x = self.dropout(x)

        x = global_add_pool(x, batch_idx)

        tg_pred = self.tg_head(x).squeeze()
        tm_pred = self.tm_head(x).squeeze()
        egc_pred = self.egc_head(x).squeeze()

        return tg_pred, tm_pred, egc_pred


def train_epoch_consistency_only(model, loader, optimizer, device):
    """Train with consistency loss only (no pseudo-labels)."""
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        tg_pred, tm_pred, egc_pred = model(batch)

        # Only consistency loss (no prediction loss since we have no labels)
        penalty_tg_ge_tm = F.relu(tg_pred - tm_pred).mean() * 10.0
        ratio = tg_pred / (tm_pred + 1e-8)
        penalty_ratio = F.relu(torch.abs(ratio - 0.65) - 0.2).mean() * 5.0
        expected_egc = tg_pred / 10.0 + 30.0
        penalty_egc = F.relu(torch.abs(egc_pred - expected_egc) - 20.0).mean() * 0.5

        loss = penalty_tg_ge_tm + penalty_ratio + penalty_egc

        # Add regularization to prevent collapse
        loss += 0.01 * (tg_pred.mean() - 300)**2  # Encourage reasonable Tg values

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def ablation_1_no_pseudo_labels(device, selected_data, selected_targets):
    """
    Ablation 1: Pre-train with consistency loss only (no pseudo-labels).
    """
    print("\n" + "=" * 80)
    print("ABLATION 1: NO PSEUDO-LABELS (Consistency Loss Only)")
    print("=" * 80)

    # Create dataset (without targets)
    dataset = PolymerMultiDataset(selected_data, targets=None)

    # Split
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.1, random_state=42
    )

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Model
    atom_dim, _ = get_feature_dimensions()
    model = TCPModelConsistencyOnly(atom_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Train
    print("\nTraining with consistency loss only...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(30):
        train_loss = train_epoch_consistency_only(model, train_loader, optimizer, device)

        # Simple validation: check if predictions are reasonable
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                tg_pred, tm_pred, egc_pred = model(batch)
                val_loss = F.relu(tg_pred - tm_pred).mean().item()
                break

        scheduler.step(val_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                      '/home/home_public/aisci/AIscientist/src/7_day8_tcp/ablation1_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    # Load best model
    model.load_state_dict(torch_load_state('/home/home_public/aisci/AIscientist/src/7_day8_tcp/ablation1_model.pt'))

    return model


def ablation_2_no_distribution_sampling(device, pseudo_labels, pi1m_df):
    """
    Ablation 2: Use random 100K samples instead of distribution-aware sampling.
    """
    print("\n" + "=" * 80)
    print("ABLATION 2: NO DISTRIBUTION SAMPLING (Random 100K Samples)")
    print("=" * 80)

    # Select random 100K valid samples
    valid_indices = np.where(pseudo_labels['valid'])[0]
    np.random.seed(42)
    random_indices = np.random.choice(valid_indices, size=min(100000, len(valid_indices)), replace=False)

    print(f"Selected {len(random_indices)} random samples")

    # Create data
    selected_data = []
    selected_targets = []

    for idx in tqdm(random_indices, desc="Processing random samples"):
        # Handle both 'SMILES' and 'Smiles' column names
        smiles = pi1m_df.iloc[idx].get('SMILES', pi1m_df.iloc[idx].get('Smiles', None))
        if smiles is None:
            continue
        data = mol_to_graph(smiles, target=None)
        if data is not None:
            selected_data.append(data)
            selected_targets.append(pseudo_labels['targets'][idx])

    selected_targets = np.array(selected_targets, dtype=np.float32)

    return selected_data, selected_targets


def evaluate_ablation(ablation_name, ablation_model, tg_data_list, tg_targets, device):
    """
    Evaluate ablation model with 5-fold CV.
    """
    print("\n" + "-" * 80)
    print(f"EVALUATING: {ablation_name}")
    print("-" * 80)

    # Create encoder from ablation model
    class AblationEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.convs = model.convs
            self.batch_norms = model.batch_norms
            self.dropout = model.dropout

        def forward(self, batch):
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
            for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                if i > 0:
                    x = self.dropout(x)
            return global_add_pool(x, batch_idx)

    encoder = AblationEncoder(ablation_model)

    # 5-fold CV
    splits = create_k_fold_splits(tg_data_list, tg_targets, n_splits=5, seed=42)

    rmses = []
    for fold_idx, (train_data, val_data, train_targets, val_targets) in enumerate(splits):
        print(f"\nFold {fold_idx + 1}/5...")

        rmse = finetune_fold(
            fold_idx, train_data, val_data, train_targets, val_targets,
            pretrained_encoder=encoder, device=device
        )
        rmses.append(rmse)
        print(f"  Val RMSE: {rmse:.2f} K")

    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)

    print(f"\n{ablation_name} Results:")
    print(f"  Mean RMSE: {mean_rmse:.2f} ± {std_rmse:.2f} K")
    print(f"  Fold RMSEs: {[f'{r:.2f}' for r in rmses]}")

    return mean_rmse, std_rmse, rmses


def phase5_ablation_studies():
    """
    Complete Phase 5: All ablation studies.
    """
    print("=" * 80)
    print("PHASE 5: ABLATION STUDIES")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    print("\nLoading data...")
    tg_data_list, tg_targets = load_tg_data(
        '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    )

    pseudo_labels = torch_load_state('/home/home_public/aisci/AIscientist/src/7_day8_tcp/PI1M_pseudo_labels.pt')
    selected_indices = torch_load_state('/home/home_public/aisci/AIscientist/src/7_day8_tcp/selected_indices.pt')
    pi1m_df = pd.read_csv('/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv')

    # Prepare selected data for ablations
    valid_indices = np.where(pseudo_labels['valid'])[0]

    selected_data = []
    selected_targets = []

    for idx in tqdm(selected_indices, desc="Loading distribution-selected samples"):
        real_idx = valid_indices[idx]
        # Handle both 'SMILES' and 'Smiles' column names
        smiles = pi1m_df.iloc[real_idx].get('SMILES', pi1m_df.iloc[real_idx].get('Smiles', None))
        if smiles is None:
            continue
        data = mol_to_graph(smiles, target=None)
        if data is not None:
            selected_data.append(data)
            selected_targets.append(pseudo_labels['targets'][real_idx])

    selected_targets = np.array(selected_targets, dtype=np.float32)

    results = {}

    # Ablation 1: No pseudo-labels
    model1 = ablation_1_no_pseudo_labels(device, selected_data, selected_targets)
    rmse1, std1, folds1 = evaluate_ablation(
        "Ablation 1: No Pseudo-Labels",
        model1, tg_data_list, tg_targets, device
    )
    results['no_pseudo_labels'] = {'mean': rmse1, 'std': std1, 'folds': folds1}

    # Ablation 2: No distribution sampling
    data2, targets2 = ablation_2_no_distribution_sampling(device, pseudo_labels, pi1m_df)

    # Train model with random sampling (using full TCP training)
    from phase3_pretraining import TCPModel, train_epoch_tcp, evaluate_tcp

    dataset = PolymerMultiDataset(data2, targets2)
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.1, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    atom_dim, _ = get_feature_dimensions()
    model2 = TCPModel(atom_dim).to(device)

    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    print("\nTraining model with random sampling...")
    best_val_loss = float('inf')
    for epoch in range(30):
        train_metrics = train_epoch_tcp(model2, train_loader, optimizer, device, consistency_weight=0.1)
        val_metrics = evaluate_tcp(model2, val_loader, device)
        scheduler.step(val_metrics['loss'])

        if epoch % 5 == 0:
            print(f"Epoch {epoch+1:2d}: Val Loss={val_metrics['loss']:.4f}, RMSE(Tg)={val_metrics['rmse_tg']:.2f}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model2.state_dict(),
                      '/home/home_public/aisci/AIscientist/src/7_day8_tcp/ablation2_model.pt')

    model2.load_state_dict(torch_load_state('/home/home_public/aisci/AIscientist/src/7_day8_tcp/ablation2_model.pt'))

    rmse2, std2, folds2 = evaluate_ablation(
        "Ablation 2: No Distribution Sampling",
        model2, tg_data_list, tg_targets, device
    )
    results['no_distribution_sampling'] = {'mean': rmse2, 'std': std2, 'folds': folds2}

    # Ablation 3: No pre-training (baseline from Phase 4)
    # Already computed in Phase 4, but let's reference it
    print("\n" + "-" * 80)
    print("ABLATION 3: No Pre-training (Baseline)")
    print("-" * 80)
    print("(Results from Phase 4)")
    # Will be filled when reading Phase 4 results

    # Save all results
    output_path = '/home/home_public/aisci/AIscientist/src/7_day8_tcp/ablation_results.txt'
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 5: ABLATION STUDIES RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Ablation 1: No Pseudo-Labels (Consistency Loss Only)\n")
        f.write(f"  Mean RMSE: {rmse1:.2f} ± {std1:.2f} K\n")
        f.write(f"  Fold RMSEs: {[f'{r:.2f}' for r in folds1]}\n\n")

        f.write("Ablation 2: No Distribution Sampling (Random 100K)\n")
        f.write(f"  Mean RMSE: {rmse2:.2f} ± {std2:.2f} K\n")
        f.write(f"  Fold RMSEs: {[f'{r:.2f}' for r in folds2]}\n\n")

        f.write("Ablation 3: No Pre-training (Baseline)\n")
        f.write("  (See Phase 4 results)\n\n")

        f.write("=" * 80 + "\n")

    print(f"\n{'='*80}")
    print("ABLATION STUDIES COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Ablation 1 (No pseudo-labels):       {rmse1:.2f} ± {std1:.2f} K")
    print(f"  Ablation 2 (No distribution sampling): {rmse2:.2f} ± {std2:.2f} K")
    print(f"  Ablation 3 (No pre-training):         See Phase 4")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    phase5_ablation_studies()

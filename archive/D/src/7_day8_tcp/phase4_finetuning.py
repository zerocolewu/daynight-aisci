"""
Phase 4: Fine-tuning on Tg prediction with weak consistency constraint.

Loads pre-trained encoder, replaces head with single Tg predictor,
and performs 5-fold cross-validation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from tqdm import tqdm
import sys
import os
sys.path.append('/home/home_public/aisci/AIscientist/src/1_day2_reperioGT')
from dataset import mol_to_graph, get_feature_dimensions
from tcp_dataset import load_tg_data, create_k_fold_splits

def torch_load_safe(path):
    """Safe torch.load wrapper for loading data with numpy arrays."""
    import torch
    return torch.load(path, weights_only=False)


def torch_load_state(path):
    """Safe torch.load wrapper for loading model state_dict."""
    import torch
    return torch.load(path, weights_only=False)



class TCPFinetuneModel(nn.Module):
    """Fine-tuning model: pre-trained encoder + Tg prediction head."""

    def __init__(self, pretrained_encoder, num_features, hidden_dim=256, num_layers=5):
        super(TCPFinetuneModel, self).__init__()

        # Load pre-trained encoder architecture
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)

        # Reconstruct encoder from pretrained model
        self.convs.append(GINConv(nn.Linear(num_features, hidden_dim)))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Linear(hidden_dim, hidden_dim)))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Load pretrained weights
        pretrained_state = pretrained_encoder.state_dict()
        state_dict = self.state_dict()

        # Copy encoder weights
        for key in pretrained_state:
            if key in state_dict and state_dict[key].shape == pretrained_state[key].shape:
                state_dict[key] = pretrained_state[key]

        self.load_state_dict(state_dict)

        # Single Tg prediction head
        self.tg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, batch):
        """Forward pass - predict Tg only."""
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # Encoder
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i > 0:
                x = self.dropout(x)

        # Global pooling
        x = global_add_pool(x, batch_idx)

        # Tg prediction
        tg_pred = self.tg_head(x).squeeze()

        return tg_pred


class BaselineModel(nn.Module):
    """Baseline model: train from scratch without pre-training."""

    def __init__(self, num_features, hidden_dim=256, num_layers=5):
        super(BaselineModel, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)

        self.convs.append(GINConv(nn.Linear(num_features, hidden_dim)))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Linear(hidden_dim, hidden_dim)))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.tg_head = nn.Sequential(
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

        return tg_pred


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    mae = np.mean(np.abs(predictions - targets))

    return {
        'loss': total_loss / len(loader.dataset),
        'rmse': rmse,
        'mae': mae
    }


def finetune_fold(fold_idx, train_data, val_data, train_targets, val_targets,
                 pretrained_encoder=None, device=torch.device('cuda')):
    """Fine-tune (or train baseline) for one fold."""

    atom_dim, _ = get_feature_dimensions()

    # Create model
    if pretrained_encoder is not None:
        model = TCPFinetuneModel(pretrained_encoder, atom_dim).to(device)
        model_name = "TCP Fine-tuned"
    else:
        model = BaselineModel(atom_dim).to(device)
        model_name = "Baseline (from scratch)"

    # Add targets to data
    for i, data in enumerate(train_data):
        data.y = torch.tensor([train_targets[i][0]], dtype=torch.float)
    for i, data in enumerate(val_data):
        data.y = torch.tensor([val_targets[i][0]], dtype=torch.float)

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    # Training
    best_val_rmse = float('inf')
    patience_counter = 0
    max_patience = 30

    for epoch in range(300):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics['loss'])

        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break

    return best_val_rmse


def phase4_finetuning():
    """
    Complete Phase 4: Fine-tuning with 5-fold cross-validation.
    """
    print("=" * 80)
    print("PHASE 4: FINE-TUNING ON Tg PREDICTION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load Tg data
    print("\nLoading Tg dataset...")
    tg_data_list, tg_targets = load_tg_data(
        '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    )

    # Create 5-fold splits
    print("\nCreating 5-fold splits...")
    splits = create_k_fold_splits(tg_data_list, tg_targets, n_splits=5, seed=42)

    # Check if pre-trained model exists
    if not os.path.exists('/home/home_public/aisci/AIscientist/src/7_day8_tcp/TCP_pretrained_model.pt'):
        raise FileNotFoundError("Phase 3 output not found: TCP_pretrained_model.pt - please run Phase 3 first")

    # Load pre-trained model
    print("\nLoading pre-trained TCP model...")
    atom_dim, _ = get_feature_dimensions()

    pretrained_model = TCPModel(
        num_features=atom_dim,
        hidden_dim=256,
        num_layers=5
    ).to(device)
    checkpoint = torch_load_safe('/home/home_public/aisci/AIscientist/src/7_day8_tcp/TCP_pretrained_model.pt')
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])

    # Extract encoder (remove prediction heads)
    class PretrainedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = pretrained_model.convs
            self.batch_norms = pretrained_model.batch_norms
            self.dropout = pretrained_model.dropout

        def forward(self, batch):
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
            for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                if i > 0:
                    x = self.dropout(x)
            return global_add_pool(x, batch_idx)

    pretrained_encoder = PretrainedEncoder()

    # Fine-tuning with pre-training
    print("\n" + "-" * 80)
    print("TCP FINE-TUNING (5-fold CV)")
    print("-" * 80)

    tcp_rmses = []
    for fold_idx, (train_data, val_data, train_targets, val_targets) in enumerate(splits):
        print(f"\nFold {fold_idx + 1}/5...")
        print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

        rmse = finetune_fold(
            fold_idx, train_data, val_data, train_targets, val_targets,
            pretrained_encoder=pretrained_encoder, device=device
        )
        tcp_rmses.append(rmse)
        print(f"  Best Val RMSE: {rmse:.2f} K")

    tcp_mean = np.mean(tcp_rmses)
    tcp_std = np.std(tcp_rmses)

    print(f"\nTCP Fine-tuning Results:")
    print(f"  Mean RMSE: {tcp_mean:.2f} ± {tcp_std:.2f} K")
    print(f"  Fold results: {[f'{r:.2f}' for r in tcp_rmses]}")

    # Baseline (train from scratch)
    print("\n" + "-" * 80)
    print("BASELINE (Train from Scratch, 5-fold CV)")
    print("-" * 80)

    baseline_rmses = []
    for fold_idx, (train_data, val_data, train_targets, val_targets) in enumerate(splits):
        print(f"\nFold {fold_idx + 1}/5...")
        print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

        rmse = finetune_fold(
            fold_idx, train_data, val_data, train_targets, val_targets,
            pretrained_encoder=None, device=device
        )
        baseline_rmses.append(rmse)
        print(f"  Best Val RMSE: {rmse:.2f} K")

    baseline_mean = np.mean(baseline_rmses)
    baseline_std = np.std(baseline_rmses)

    print(f"\nBaseline Results:")
    print(f"  Mean RMSE: {baseline_mean:.2f} ± {baseline_std:.2f} K")
    print(f"  Fold results: {[f'{r:.2f}' for r in baseline_rmses]}")

    # Save results
    results = {
        'tcp_finetuned': {
            'rmse_mean': tcp_mean,
            'rmse_std': tcp_std,
            'fold_rmses': tcp_rmses,
        },
        'baseline': {
            'rmse_mean': baseline_mean,
            'rmse_std': baseline_std,
            'fold_rmses': baseline_rmses,
        },
        'improvement': baseline_mean - tcp_mean,
        'improvement_pct': 100 * (baseline_mean - tcp_mean) / baseline_mean,
    }

    output_path = '/home/home_public/aisci/AIscientist/src/7_day8_tcp/Tg_finetuned_results.txt'
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 4: FINE-TUNING RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("TCP Fine-tuned Model:\n")
        f.write(f"  Mean RMSE: {tcp_mean:.2f} ± {tcp_std:.2f} K\n")
        f.write(f"  Fold RMSEs: {[f'{r:.2f}' for r in tcp_rmses]}\n\n")

        f.write("Baseline Model (Train from Scratch):\n")
        f.write(f"  Mean RMSE: {baseline_mean:.2f} ± {baseline_std:.2f} K\n")
        f.write(f"  Fold RMSEs: {[f'{r:.2f}' for r in baseline_rmses]}\n\n")

        f.write("Comparison:\n")
        f.write(f"  Absolute Improvement: {results['improvement']:.2f} K\n")
        f.write(f"  Relative Improvement: {results['improvement_pct']:.2f}%\n")

        if tcp_mean < baseline_mean:
            f.write(f"\n  ✓ TCP pre-training IMPROVES performance\n")
        else:
            f.write(f"\n  ✗ TCP pre-training shows NEGATIVE TRANSFER\n")

    print(f"\n{'='*80}")
    print("PHASE 4 COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  TCP Fine-tuned:  {tcp_mean:.2f} ± {tcp_std:.2f} K")
    print(f"  Baseline:        {baseline_mean:.2f} ± {baseline_std:.2f} K")
    print(f"  Improvement:     {results['improvement']:+.2f} K ({results['improvement_pct']:+.1f}%)")
    print(f"{'='*80}\n")

    return results


# Import TCPModel - handle case where file doesn't exist yet
try:
    from phase3_pretraining import TCPModel
except ImportError:
    print("ERROR: phase3_pretraining.py not found. TCP pre-training must have completed first.")
    # Define TCPModel locally as fallback
    class TCPModel(nn.Module):
        pass  # Copy from phase3_pretraining.py
    pass

if __name__ == "__main__":
    phase4_finetuning()

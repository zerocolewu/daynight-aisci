"""
Training script for CPA-PSE experiment.
Handles data loading, PSE feature concatenation, training loop, and evaluation.
"""
import os
import sys
import json
import time
import pickle
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = "/home/home_public/aisci/AIscientist/src/4_day4_selfgnn_polymer/data/tg_cache/tg_graphs_3ru_42dim.pkl"
PSE_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/tg_pse_features.pt"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    """Load precomputed graphs and scaler."""
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['train_graphs'], data['val_graphs'], data['test_graphs'], data['scaler']


def load_pse_features():
    """Load precomputed PSE features."""
    return torch.load(PSE_PATH, weights_only=False)


def assemble_features(graph, pse_tensor, condition, base_dim=42):
    """
    Assemble node features according to condition.

    Base 42-dim layout:
      indices 0-7:  8 rich atom features
      index 8:      PSPE-dce (distance-to-chain-end)
      index 9:      PSPE-mi (monomer-index)
      indices 10-41: 32 RWPE (k=1 to k=32)

    PSE features:
      index 0: modular_position
      index 1: periodic_distance
      index 2: is_backbone

    Conditions:
      'A' (CPA baseline): 42-dim base only (no PSE)
      'B' (CPA-PSE full): 42 + 3 PSE = 45 dim
      'C' (modular_position only): 42 + 1 = 43 dim
      'D' (periodic_distance only): 42 + 1 = 43 dim
      'E' (is_backbone only): 42 + 1 = 43 dim
      'F' (PSE without PSPE): remove dce+mi (dims 8-9), add 3 PSE = 43 dim
      'H' (Non-CPA + full PSE): 42 + 3 PSE = 45 dim
    """
    x = graph.x  # [N, 42]

    if condition == 'A':
        return x

    elif condition == 'B' or condition == 'H':
        # Full PSE: add all 3 features
        return torch.cat([x, pse_tensor], dim=-1)  # [N, 45]

    elif condition == 'C':
        # modular_position only
        return torch.cat([x, pse_tensor[:, 0:1]], dim=-1)  # [N, 43]

    elif condition == 'D':
        # periodic_distance only
        return torch.cat([x, pse_tensor[:, 1:2]], dim=-1)  # [N, 43]

    elif condition == 'E':
        # is_backbone only
        return torch.cat([x, pse_tensor[:, 2:3]], dim=-1)  # [N, 43]

    elif condition == 'F':
        # Remove PSPE (dims 8-9), add PSE
        # Keep dims 0-7 (rich) and 10-41 (RWPE), then add PSE
        x_no_pspe = torch.cat([x[:, :8], x[:, 10:]], dim=-1)  # [N, 40]
        return torch.cat([x_no_pspe, pse_tensor], dim=-1)  # [N, 43]

    else:
        raise ValueError(f"Unknown condition: {condition}")


def prepare_graphs(graphs, pse_list, condition):
    """Create new graph copies with assembled features."""
    new_graphs = []
    for i, g in enumerate(graphs):
        pse = pse_list[i]
        new_x = assemble_features(g, pse, condition)
        new_g = Data(
            x=new_x,
            edge_index=g.edge_index.clone(),
            y=g.y.clone(),
        )
        new_graphs.append(new_g)
    return new_graphs


def train_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
            all_preds.append(pred.cpu())
            all_targets.append(batch.y.view(-1).cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    if scaler is not None:
        all_preds = all_preds * scaler.scale_[0] + scaler.mean_[0]
        all_targets = all_targets * scaler.scale_[0] + scaler.mean_[0]

    mse = ((all_preds - all_targets) ** 2).mean().item()
    rmse = np.sqrt(mse)
    mae = (all_preds - all_targets).abs().mean().item()

    # R² calculation
    ss_res = ((all_targets - all_preds) ** 2).sum().item()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return avg_loss, rmse, mae, r2, all_preds, all_targets


def compute_feature_correlations(train_graphs, pse_list):
    """Compute Pearson correlations between PSE features and existing features."""
    all_x = []
    all_pse = []
    for g, p in zip(train_graphs, pse_list):
        all_x.append(g.x)
        all_pse.append(p)

    all_x = torch.cat(all_x, dim=0)  # [N_nodes, 42]
    all_pse = torch.cat(all_pse, dim=0)  # [N_nodes, 3]

    correlations = {}
    pse_names = ['modular_position', 'periodic_distance', 'is_backbone']
    feature_names = [f'rich_{i}' for i in range(8)] + ['PSPE_dce', 'PSPE_mi'] + [f'RWPE_{i}' for i in range(1, 33)]

    for pi, pname in enumerate(pse_names):
        pse_col = all_pse[:, pi]
        corrs = {}
        for fi, fname in enumerate(feature_names):
            if pse_col.std() > 1e-8 and all_x[:, fi].std() > 1e-8:
                corr = np.corrcoef(pse_col.numpy(), all_x[:, fi].numpy())[0, 1]
                corrs[fname] = float(corr)
            else:
                corrs[fname] = 0.0
        correlations[pname] = corrs

    return correlations


def train_model(model_class, model_name, seed, condition, model_kwargs=None):
    """Train a model with a specific seed and condition, return results."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"Training: {model_name} | Condition: {condition} | Seed: {seed} | Device: {device}")
    print(f"{'='*60}")

    # Load data
    train_graphs, val_graphs, test_graphs, scaler = load_data()

    # Load PSE features
    pse_data = load_pse_features()
    pse_train = pse_data['train']
    pse_val = pse_data['val']
    pse_test = pse_data['test']

    # Prepare graphs with appropriate features
    train_graphs_prep = prepare_graphs(train_graphs, pse_train, condition)
    val_graphs_prep = prepare_graphs(val_graphs, pse_val, condition)
    test_graphs_prep = prepare_graphs(test_graphs, pse_test, condition)

    input_dim = train_graphs_prep[0].x.shape[1]
    print(f"Input dimension: {input_dim}")

    # Create dataloaders
    train_loader = DataLoader(train_graphs_prep, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_graphs_prep, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_graphs_prep, batch_size=64, shuffle=False)

    # Build model
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs['in_dim'] = input_dim
    model = model_class(**model_kwargs).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}, Params: {num_params:,}")

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    patience = 30
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(1, 301):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmse, val_mae, _, _, _ = evaluate(model, val_loader, criterion, device, scaler)
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'lr': current_lr,
        })

        scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.2f} K | Best: {best_val_rmse:.2f} K (ep {best_epoch}) | LR: {current_lr:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    train_time = time.time() - start_time

    # Load best model
    model.load_state_dict(best_state)
    model.to(device)

    # Test evaluation
    test_loss, test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, scaler
    )

    print(f"\nFinal Results for {model_name} Condition {condition} Seed {seed}:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val RMSE: {best_val_rmse:.4f} K")
    print(f"  Test RMSE: {test_rmse:.4f} K")
    print(f"  Test MAE: {test_mae:.4f} K")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Training Time: {train_time:.1f}s")

    # CPA gate statistics (only for CPA models)
    cpa_gate_stats = None
    if isinstance(model, type(torch.nn.Module)) and hasattr(model, 'get_cpa_gate_stats'):
        try:
            # Use a small batch from validation set
            sample_batch = next(iter(val_loader)).to(device)
            cpa_gate_stats = model.get_cpa_gate_stats(sample_batch)
        except Exception as e:
            print(f"  Warning: Could not compute CPA gate stats: {e}")

    # Feature correlations (computed once for condition, but included with first seed result)
    feature_corrs = None
    if seed == 71:  # Only compute once per condition
        feature_corrs = compute_feature_correlations(train_graphs, pse_train)

    # Save results
    result = {
        'model_name': model_name,
        'condition': condition,
        'seed': seed,
        'input_dim': input_dim,
        'best_epoch': best_epoch,
        'best_val_rmse': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'train_time': train_time,
        'num_params': num_params,
        'test_preds': test_preds.numpy().tolist(),
        'test_targets': test_targets.numpy().tolist(),
    }

    if cpa_gate_stats is not None:
        result['cpa_gate_stats'] = cpa_gate_stats
    if feature_corrs is not None:
        result['feature_correlations'] = feature_corrs

    cond_tag = f"{condition}_{seed}"
    result_file = os.path.join(RESULTS_DIR, f"{model_name}_{cond_tag}.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    history_file = os.path.join(RESULTS_DIR, f"{model_name}_{cond_tag}_history.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    model_file = os.path.join(RESULTS_DIR, f"{model_name}_{cond_tag}.pt")
    torch.save(best_state, model_file)

    return result


if __name__ == "__main__":
    from model import CPAGAT
    train_model(CPAGAT, "cpa_pse", 71, "B", {"in_dim": 45})

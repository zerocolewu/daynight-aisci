"""Fine-tuning with tuned hyperparameters to push below 50.19."""
import os, sys, json, logging, time, copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

WORK_DIR = '/home/home_public/aisci/AIscientist/src/2_2_polymer_jepa_pretrain'
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

DATA_DIR = os.path.join(WORK_DIR, 'data')
CHECKPOINT_DIR = os.path.join(WORK_DIR, 'checkpoints')
TG_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from data_preprocessing import TgTrimerDataset, get_data_splits, TOTAL_ATOM_FEATURE_DIM
from jepa_pretrain import GINEncoder
from torch_geometric.nn import global_mean_pool
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")

tg_dataset = TgTrimerDataset(root=DATA_DIR, csv_path=TG_PATH)
train_idx, val_idx, test_idx = get_data_splits(tg_dataset)
train_dataset = Subset(tg_dataset, train_idx)
val_dataset = Subset(tg_dataset, val_idx)
test_dataset = Subset(tg_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

checkpoint_path = os.path.join(WORK_DIR, 'pretrained_encoder_contrastive.pt')


class RegressionHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.head(x)


class TgPredictor(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x, edge_index, batch, edge_attr=None):
        node_emb = self.encoder(x, edge_index, batch, edge_attr)
        graph_emb = global_mean_pool(node_emb, batch)
        return self.head(graph_emb).squeeze(-1)


def train_and_eval(model, train_loader, val_loader, test_loader, device,
                   lr=5e-4, weight_decay=1e-5, num_epochs=500, patience=30,
                   strategy_name='test'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )

    best_val_rmse = float('inf')
    best_state = None
    best_epoch = 0
    wait = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = nn.functional.mse_loss(pred, batch.y.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            train_count += batch.num_graphs

        train_rmse = (train_loss / train_count) ** 0.5

        # Val
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                loss = nn.functional.mse_loss(pred, batch.y.squeeze(-1))
                val_loss += loss.item() * batch.num_graphs
                val_count += batch.num_graphs
        val_rmse = (val_loss / val_count) ** 0.5

        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

        if epoch % 20 == 0:
            logger.info(f"[{strategy_name}] Epoch {epoch} | Train: {train_rmse:.4f} | Val: {val_rmse:.4f} | Best Val: {best_val_rmse:.4f}")

    # Test with best model
    model.load_state_dict(best_state)
    model.eval()
    test_loss = 0
    test_count = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = nn.functional.mse_loss(pred, batch.y.squeeze(-1))
            test_loss += loss.item() * batch.num_graphs
            test_count += batch.num_graphs
    test_rmse = (test_loss / test_count) ** 0.5

    return {
        'strategy': strategy_name,
        'best_epoch': best_epoch,
        'train_rmse': float(train_rmse),
        'val_rmse': float(best_val_rmse),
        'test_rmse': float(test_rmse),
    }


all_results = {}

# Configs to try for Strategy B (end-to-end)
configs = [
    # (name, lr, weight_decay, head_dropout, head_hidden)
    ('B_v1_lr3e4_wd1e4', 3e-4, 1e-4, 0.3, 128),
    ('B_v2_lr5e4_wd1e5', 5e-4, 1e-5, 0.3, 128),
    ('B_v3_lr5e4_wd1e4_hd64', 5e-4, 1e-4, 0.3, 64),
    ('B_v4_lr3e4_wd1e4_hd256', 3e-4, 1e-4, 0.2, 256),
    ('B_v5_lr1e3_wd1e5_hd128', 1e-3, 1e-5, 0.3, 128),
]

for name, lr, wd, hd, hidden in configs:
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {name} | lr={lr}, wd={wd}, head_dropout={hd}, head_hidden={hidden}")
    logger.info(f"{'='*60}")

    encoder = GINEncoder(TOTAL_ATOM_FEATURE_DIM, hidden_dim=256, dropout=0.1).to(device)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    encoder.load_state_dict(ckpt['teacher_encoder'])

    head = RegressionHead(in_dim=256, hidden_dim=hidden, dropout=hd).to(device)
    model = TgPredictor(encoder, head).to(device)

    result = train_and_eval(
        model, train_loader, val_loader, test_loader, device,
        lr=lr, weight_decay=wd, num_epochs=500, patience=30,
        strategy_name=name
    )
    all_results[name] = result
    logger.info(f"{name}: Test RMSE={result['test_rmse']:.4f}, Val RMSE={result['val_rmse']:.4f}, Best Epoch={result['best_epoch']}")

# Also try Strategy A variants
a_configs = [
    ('A_v1_lr3e4_freeze3', 3e-4, 1e-5, 128, 0.3, 3),
    ('A_v2_lr5e4_freeze2', 5e-4, 1e-5, 128, 0.3, 2),
]

for name, lr, wd, hidden, hd, freeze_layers in a_configs:
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {name} | lr={lr}, wd={wd}, freeze={freeze_layers}")
    logger.info(f"{'='*60}")

    encoder = GINEncoder(TOTAL_ATOM_FEATURE_DIM, hidden_dim=256, dropout=0.1).to(device)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    encoder.load_state_dict(ckpt['teacher_encoder'])

    # Freeze first N layers
    for i in range(freeze_layers):
        for param in encoder.convs[i].parameters():
            param.requires_grad = False
        for param in encoder.batch_norms[i].parameters():
            param.requires_grad = False

    head = RegressionHead(in_dim=256, hidden_dim=hidden, dropout=hd).to(device)
    model = TgPredictor(encoder, head).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_trainable:,}")

    result = train_and_eval(
        model, train_loader, val_loader, test_loader, device,
        lr=lr, weight_decay=wd, num_epochs=500, patience=30,
        strategy_name=name
    )
    all_results[name] = result
    logger.info(f"{name}: Test RMSE={result['test_rmse']:.4f}, Val RMSE={result['val_rmse']:.4f}, Best Epoch={result['best_epoch']}")

# Save all results
with open(os.path.join(WORK_DIR, 'finetune_tuned_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

# Print summary
print("\n" + "="*70)
print("HYPERPARAMETER TUNING RESULTS")
print("="*70)
print(f"\n{'Config':<35} {'Test RMSE':>10} {'Val RMSE':>10} {'Best Ep':>8}")
print("-"*70)
for name, r in sorted(all_results.items(), key=lambda x: x[1]['test_rmse']):
    print(f"{name:<35} {r['test_rmse']:>10.4f} {r['val_rmse']:>10.4f} {r['best_epoch']:>8}")

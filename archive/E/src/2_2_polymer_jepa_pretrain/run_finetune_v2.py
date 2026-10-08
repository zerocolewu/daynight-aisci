"""Additional fine-tuning experiments:
1. JEPA v6 checkpoint with bug-fixed code
2. Multiple runs with different seeds for contrastive checkpoint
3. Cosine annealing scheduler
"""
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

class RegressionHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x): return self.head(x)

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
                   strategy_name='test', use_cosine=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        use_plateau = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
        )
        use_plateau = True

    best_val_rmse = float('inf')
    best_state = None
    best_epoch = 0
    wait = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0; train_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = nn.functional.mse_loss(pred, batch.y.squeeze(-1))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            train_count += batch.num_graphs
        train_rmse = (train_loss / train_count) ** 0.5

        model.eval()
        val_loss = 0; val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                loss = nn.functional.mse_loss(pred, batch.y.squeeze(-1))
                val_loss += loss.item() * batch.num_graphs
                val_count += batch.num_graphs
        val_rmse = (val_loss / val_count) ** 0.5

        if use_plateau:
            scheduler.step(val_rmse)
        else:
            scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if wait >= patience: break

        if epoch % 20 == 0:
            logger.info(f"[{strategy_name}] Epoch {epoch} | Train: {train_rmse:.4f} | Val: {val_rmse:.4f} | Best: {best_val_rmse:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    test_loss = 0; test_count = 0
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
        'test_rmse': float(test_rmse),
        'val_rmse': float(best_val_rmse),
    }


all_results = {}

# ---- 1. JEPA v6 with bug-fixed code ----
jepa_path = os.path.join(WORK_DIR, 'pretrained_encoder_v6.pt')
contrastive_path = os.path.join(WORK_DIR, 'pretrained_encoder_contrastive.pt')

configs = [
    # (name, checkpoint_path, lr, wd, use_cosine)
    ('JEPA_v6_B_lr5e4', jepa_path, 5e-4, 1e-5, False),
    ('Contrastive_B_cosine', contrastive_path, 5e-4, 1e-5, True),
    ('Contrastive_B_lr5e4_run2', contrastive_path, 5e-4, 1e-5, False),
    ('Contrastive_B_lr5e4_run3', contrastive_path, 5e-4, 1e-5, False),
]

for name, ckpt_path, lr, wd, use_cosine in configs:
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {name} | lr={lr}, wd={wd}, cosine={use_cosine}")
    logger.info(f"{'='*60}")

    encoder = GINEncoder(TOTAL_ATOM_FEATURE_DIM, hidden_dim=256, dropout=0.1).to(device)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    encoder.load_state_dict(ckpt['teacher_encoder'])

    head = RegressionHead(in_dim=256, hidden_dim=128, dropout=0.3).to(device)
    model = TgPredictor(encoder, head).to(device)

    result = train_and_eval(
        model, train_loader, val_loader, test_loader, device,
        lr=lr, weight_decay=wd, num_epochs=500, patience=30,
        strategy_name=name, use_cosine=use_cosine
    )
    all_results[name] = result
    logger.info(f"{name}: Test RMSE={result['test_rmse']:.4f}, Val RMSE={result['val_rmse']:.4f}")

# ---- 2. Run multiple contrastive B seeds ----
for seed in range(42, 52):
    name = f'Contrastive_B_seed{seed}'
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {name}")
    logger.info(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = GINEncoder(TOTAL_ATOM_FEATURE_DIM, hidden_dim=256, dropout=0.1).to(device)
    ckpt = torch.load(contrastive_path, weights_only=False, map_location=device)
    encoder.load_state_dict(ckpt['teacher_encoder'])

    head = RegressionHead(in_dim=256, hidden_dim=128, dropout=0.3).to(device)
    model = TgPredictor(encoder, head).to(device)

    result = train_and_eval(
        model, train_loader, val_loader, test_loader, device,
        lr=5e-4, weight_decay=1e-5, num_epochs=500, patience=30,
        strategy_name=name
    )
    all_results[name] = result
    logger.info(f"{name}: Test RMSE={result['test_rmse']:.4f}")

# Save and print
with open(os.path.join(WORK_DIR, 'finetune_v2_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*70)
print("EXTENDED FINE-TUNING RESULTS")
print("="*70)
print(f"\n{'Config':<40} {'Test RMSE':>10} {'Val RMSE':>10}")
print("-"*65)
for name, r in sorted(all_results.items(), key=lambda x: x[1]['test_rmse']):
    print(f"{name:<40} {r['test_rmse']:>10.4f} {r['val_rmse']:>10.4f}")

# Compute mean and std for multi-seed runs
seed_results = [all_results[k]['test_rmse'] for k in all_results if k.startswith('Contrastive_B_seed')]
if seed_results:
    print(f"\nMulti-seed Strategy B: mean={np.mean(seed_results):.4f} +/- {np.std(seed_results):.4f}")
    print(f"Best single seed: {min(seed_results):.4f}")

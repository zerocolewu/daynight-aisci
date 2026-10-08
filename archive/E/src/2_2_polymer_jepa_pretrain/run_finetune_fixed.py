"""Re-run fine-tuning with bug-fixed code (dropout, pooling, LR)."""
import os, sys, json, logging, time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

WORK_DIR = '/home/home_public/aisci/AIscientist/src/2_2_polymer_jepa_pretrain'
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

DATA_DIR = os.path.join(WORK_DIR, 'data')
CHECKPOINT_DIR = os.path.join(WORK_DIR, 'checkpoints')
TG_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from data_preprocessing import TgTrimerDataset, get_data_splits, TOTAL_ATOM_FEATURE_DIM
from jepa_pretrain import GINEncoder
from finetune import (
    TgPredictor, RegressionHead,
    strategy_baseline, strategy_a_partial_freeze,
    strategy_b_end_to_end, strategy_c_full_freeze,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")

# Load Tg dataset
tg_dataset = TgTrimerDataset(root=DATA_DIR, csv_path=TG_PATH)
logger.info(f"Tg dataset size: {len(tg_dataset)}")

train_idx, val_idx, test_idx = get_data_splits(tg_dataset)
train_dataset = Subset(tg_dataset, train_idx)
val_dataset = Subset(tg_dataset, val_idx)
test_dataset = Subset(tg_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

def make_encoder():
    return GINEncoder(TOTAL_ATOM_FEATURE_DIM, hidden_dim=256, dropout=0.1)

def make_head():
    return RegressionHead(in_dim=256, hidden_dim=128, dropout=0.5)

all_results = {}

# Baseline
logger.info("\n" + "="*70)
logger.info("BASELINE (bug-fixed: dropout=0.1, mean_pool, lr=5e-4)")
logger.info("="*70)
result_baseline = strategy_baseline(
    make_encoder, make_head,
    train_loader, val_loader, test_loader, device, CHECKPOINT_DIR
)
all_results['baseline'] = result_baseline
logger.info(f"Baseline Test RMSE: {result_baseline['test_rmse']:.4f}")

# Pre-trained strategies with contrastive checkpoint
checkpoint_path = os.path.join(WORK_DIR, 'pretrained_encoder_contrastive.pt')
if os.path.exists(checkpoint_path):
    logger.info("\n" + "="*70)
    logger.info("STRATEGY A (contrastive, bug-fixed)")
    logger.info("="*70)
    result_a = strategy_a_partial_freeze(
        make_encoder, make_head, checkpoint_path,
        train_loader, val_loader, test_loader, device, CHECKPOINT_DIR
    )
    all_results['strategy_A'] = result_a
    logger.info(f"Strategy A Test RMSE: {result_a['test_rmse']:.4f}")

    logger.info("\n" + "="*70)
    logger.info("STRATEGY B (contrastive, bug-fixed)")
    logger.info("="*70)
    result_b = strategy_b_end_to_end(
        make_encoder, make_head, checkpoint_path,
        train_loader, val_loader, test_loader, device, CHECKPOINT_DIR
    )
    all_results['strategy_B'] = result_b
    logger.info(f"Strategy B Test RMSE: {result_b['test_rmse']:.4f}")

    logger.info("\n" + "="*70)
    logger.info("STRATEGY C (contrastive, bug-fixed)")
    logger.info("="*70)
    result_c = strategy_c_full_freeze(
        make_encoder, make_head, checkpoint_path,
        train_loader, val_loader, test_loader, device, CHECKPOINT_DIR
    )
    all_results['strategy_C'] = result_c
    logger.info(f"Strategy C Test RMSE: {result_c['test_rmse']:.4f}")

# Save results
with open(os.path.join(WORK_DIR, 'finetune_results_fixed.json'), 'w') as f:
    json.dump({k: v for k, v in all_results.items()}, f, indent=2, default=str)

# Print summary
print("\n" + "="*70)
print("BUG-FIXED FINE-TUNING RESULTS (Contrastive Pre-trained)")
print("Fixes: dropout 0.5->0.1, add_pool->mean_pool, Strategy B/C lr fixed")
print("="*70)
baseline_rmse = all_results['baseline']['test_rmse']
print(f"\n{'Strategy':<30} {'Test RMSE':>10} {'Val RMSE':>10} {'Train RMSE':>10}")
print("-"*65)
for name in ['baseline', 'strategy_A', 'strategy_B', 'strategy_C']:
    r = all_results[name]
    print(f"{name:<30} {r['test_rmse']:>10.4f} {r['val_rmse']:>10.4f} {r['train_rmse']:>10.4f}")
print()
for name in ['strategy_A', 'strategy_B', 'strategy_C']:
    delta = all_results[name]['test_rmse'] - baseline_rmse
    print(f"{name} delta from baseline: {delta:+.4f}")

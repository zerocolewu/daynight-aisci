"""
PPFN Evaluation and Interpretability Analysis.
"""

import sys
import os
sys.path.insert(0, '/home/home_public/aisci/AIscientist/src/1_day2_ppfn')

import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from dataset import PPFNDataset, collate_fn
from model import PPFN

SAVE_DIR = '/home/home_public/aisci/AIscientist/src/1_day2_ppfn'
CSV_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load test dataset
    print("Loading test dataset...")
    test_ds = PPFNDataset(CSV_PATH, split='test')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # Load best model
    model_path = os.path.join(SAVE_DIR, 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run train.py first.")
        return

    model = PPFN(hidden_dim=256).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print("Model loaded.")

    # Run predictions
    model.eval()
    all_preds, all_targets = [], []
    all_frag_attn = []
    all_frag_smiles = []
    all_p_smiles = []
    all_batch_frag = []

    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch(batch, device)
            pred = model(batch)
            all_preds.append(pred.cpu())
            all_targets.append(batch['tg'].cpu())

            # Get interpretability
            interp = model.interpret(batch)
            all_frag_attn.append(interp['fragment_attention'].cpu())
            all_batch_frag.append(batch['batch_frag'].cpu())
            all_frag_smiles.extend(batch.get('frag_smiles_lists', []))
            all_p_smiles.extend(batch.get('p_smiles_list', []))

    all_preds = torch.cat(all_preds).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()
    all_frag_attn = torch.cat(all_frag_attn).numpy()

    # Metrics
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))
    r2 = r2_score(all_targets, all_preds)

    print(f"\nTest Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    # === Plots ===
    # 1. Parity plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(all_targets, all_preds, alpha=0.5, s=30, edgecolors='k', linewidths=0.3)
    mn, mx = min(all_targets.min(), all_preds.min()), max(all_targets.max(), all_preds.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual Tg')
    ax.set_ylabel('Predicted Tg')
    ax.set_title(f'PPFN Parity Plot\nRMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'parity_plot.png'), dpi=200)
    plt.close()
    print("Saved parity_plot.png")

    # 2. Error distribution
    errors = all_preds - all_targets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='r', ls='--', lw=2)
    ax1.set_xlabel('Error (Pred - Actual)')
    ax1.set_ylabel('Count')
    ax1.set_title('Error Distribution')
    ax2.boxplot(errors, vert=True)
    ax2.axhline(0, color='r', ls='--', lw=2)
    ax2.set_ylabel('Error')
    ax2.set_title('Error Box Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'error_distribution.png'), dpi=200)
    plt.close()
    print("Saved error_distribution.png")

    # 3. Training curve
    log_path = os.path.join(SAVE_DIR, 'training_log.txt')
    if os.path.exists(log_path):
        # Parse all runs - find the last run (starts after a header line or epoch reset)
        all_epochs, all_t_rmse, all_v_rmse = [], [], []
        current_epochs, current_t_rmse, current_v_rmse = [], [], []
        last_ep = 0
        with open(log_path) as f:
            for line in f:
                if line.startswith('Epoch') and 'Train RMSE:' in line:
                    parts = line.split('|')
                    ep = int(parts[0].strip().split()[1].split('/')[0])
                    # Detect new run start (epoch goes back to 1 after being higher)
                    if ep <= last_ep and last_ep > 1:
                        current_epochs, current_t_rmse, current_v_rmse = [], [], []
                    last_ep = ep
                    current_epochs.append(ep)
                    for p in parts:
                        if p.strip().startswith('Train RMSE:'):
                            current_t_rmse.append(float(p.split('Train RMSE:')[1].strip().split()[0]))
                        if p.strip().startswith('Val RMSE:'):
                            current_v_rmse.append(float(p.split('Val RMSE:')[1].strip().split()[0]))
        # Use the last run's data
        epochs, t_rmse_list, v_rmse_list = current_epochs, current_t_rmse, current_v_rmse
        if epochs:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, t_rmse_list, 'b-', lw=2, label='Train RMSE')
            plt.plot(epochs, v_rmse_list, 'r-', lw=2, label='Val RMSE')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.title('Training Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, 'training_curve.png'), dpi=200)
            plt.close()
            print("Saved training_curve.png")

    # === Physical Trend Verification ===
    errors_abs = np.abs(errors)

    # Check aromatic vs non-aromatic via p-SMILES
    aromatic_pred, non_aromatic_pred = [], []
    aromatic_tg, non_aromatic_tg = [], []
    for i, smi in enumerate(all_p_smiles):
        if i >= len(all_targets):
            break
        # aromatic indicator: lowercase letters in SMILES
        has_aromatic = any(c.islower() for c in str(smi) if c.isalpha())
        if has_aromatic:
            aromatic_pred.append(all_preds[i])
            aromatic_tg.append(all_targets[i])
        else:
            non_aromatic_pred.append(all_preds[i])
            non_aromatic_tg.append(all_targets[i])

    physical_trends = {}
    if aromatic_tg and non_aromatic_tg:
        physical_trends['aromatic_mean_Tg'] = float(np.mean(aromatic_tg))
        physical_trends['non_aromatic_mean_Tg'] = float(np.mean(non_aromatic_tg))
        physical_trends['aromatic_count'] = len(aromatic_tg)
        physical_trends['non_aromatic_count'] = len(non_aromatic_tg)
        physical_trends['aromatic_hypothesis'] = (
            "Confirmed: aromatic polymers have higher Tg" if np.mean(aromatic_tg) > np.mean(non_aromatic_tg)
            else "Not confirmed: aromatic polymers do NOT have higher Tg in this dataset"
        )

    # Accuracy by Tg range
    tg_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 600)]
    range_analysis = []
    for lo, hi in tg_ranges:
        mask = (all_targets >= lo) & (all_targets < hi)
        if mask.sum() > 0:
            range_analysis.append({
                'range': f'{lo}-{hi}',
                'count': int(mask.sum()),
                'rmse': float(np.sqrt(np.mean((all_preds[mask] - all_targets[mask]) ** 2))),
                'mae': float(np.mean(np.abs(all_preds[mask] - all_targets[mask]))),
            })
    physical_trends['accuracy_by_Tg_range'] = range_analysis

    # Error-Tg correlation
    physical_trends['error_Tg_correlation'] = float(np.corrcoef(all_targets, errors_abs)[0, 1])

    # Error stats
    error_stats = {
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'median': float(np.median(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
        'q25': float(np.percentile(errors, 25)),
        'q75': float(np.percentile(errors, 75)),
    }

    # Fragment attention analysis
    attn_stats = {
        'mean': float(np.mean(all_frag_attn)),
        'std': float(np.std(all_frag_attn)),
        'max': float(np.max(all_frag_attn)),
        'min': float(np.min(all_frag_attn)),
    }

    # Save results
    results = {
        'test_metrics': {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)},
        'error_distribution': error_stats,
        'physical_trends': physical_trends,
        'fragment_attention_stats': attn_stats,
    }

    with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(SAVE_DIR, 'interpretability_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {SAVE_DIR}/results.json")
    print(f"Interpretability saved to {SAVE_DIR}/interpretability_analysis.json")

    # Print summary
    print(f"\n=== Key Findings ===")
    print(f"Test RMSE: {rmse:.2f} (Target: < 30.8)")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R2: {r2:.4f}")
    if 'aromatic_hypothesis' in physical_trends:
        print(f"Aromatic hypothesis: {physical_trends['aromatic_hypothesis']}")
        print(f"  Aromatic mean Tg: {physical_trends['aromatic_mean_Tg']:.1f} (n={physical_trends['aromatic_count']})")
        print(f"  Non-aromatic mean Tg: {physical_trends['non_aromatic_mean_Tg']:.1f} (n={physical_trends['non_aromatic_count']})")
    print(f"Error-Tg correlation: {physical_trends['error_Tg_correlation']:.4f}")


if __name__ == '__main__':
    main()

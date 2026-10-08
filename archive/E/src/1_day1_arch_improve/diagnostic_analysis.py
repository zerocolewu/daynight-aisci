"""
Diagnostic analysis for Day 1 Experiment 2.
- Attention weight analysis for GIN-GAT models
- Virtual node representation analysis for VN models
- Learning curve visualization
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

from data_preprocessing import TgDataset, get_data_splits
from models import GINGATModel, GINModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

CSV_PATH = "/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv"
PROCESSED_DIR = "./data"
MODELS_DIR = "./models"
SEED = 71
HIDDEN_DIM = 256
NUM_LAYERS = 3


def analyze_attention_weights(exp_name, dataset, device='cuda'):
    """
    Load trained GIN-GAT model and analyze attention weights.
    Reports mean/std per layer, and per-atom-type analysis.
    """
    logger.info(f"Analyzing attention weights for {exp_name}")

    train_idx, val_idx, test_idx = get_data_splits(len(dataset), seed=SEED)
    # Use test set for analysis
    test_loader = PyGDataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False)

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    model = GINGATModel(node_dim=node_dim, edge_dim=edge_dim,
                        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model_path = os.path.join(MODELS_DIR, f'{exp_name}_best.pt')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()

    layer_attn_stats = {i: {'means': [], 'stds': [], 'maxes': [], 'mins': []}
                        for i in range(NUM_LAYERS)}

    n_analyzed = 0
    max_analyze = min(50, len(test_idx))  # Analyze up to 50 molecules

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_analyze:
                break

            batch = batch.to(device)
            _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            for layer_idx in range(NUM_LAYERS):
                attn_w = model.convs[layer_idx].get_attention_weights()
                if attn_w is not None:
                    layer_attn_stats[layer_idx]['means'].append(attn_w.mean().item())
                    layer_attn_stats[layer_idx]['stds'].append(attn_w.std().item())
                    layer_attn_stats[layer_idx]['maxes'].append(attn_w.max().item())
                    layer_attn_stats[layer_idx]['mins'].append(attn_w.min().item())

            n_analyzed += 1

    results = {
        'exp_name': exp_name,
        'n_molecules_analyzed': n_analyzed,
        'layers': {}
    }

    for layer_idx in range(NUM_LAYERS):
        stats = layer_attn_stats[layer_idx]
        layer_result = {
            'mean_attention': np.mean(stats['means']),
            'std_attention': np.mean(stats['stds']),
            'max_attention': np.mean(stats['maxes']),
            'min_attention': np.mean(stats['mins']),
            'attention_std_across_molecules': np.std(stats['means']),
        }
        results['layers'][layer_idx] = layer_result
        logger.info(f"  Layer {layer_idx}: mean_attn={layer_result['mean_attention']:.6f}, "
                    f"std_attn={layer_result['std_attention']:.6f}, "
                    f"max={layer_result['max_attention']:.6f}, min={layer_result['min_attention']:.6f}")

    # Selectivity ratio: std/mean (higher = more selective)
    for layer_idx in range(NUM_LAYERS):
        l = results['layers'][layer_idx]
        l['selectivity_ratio'] = l['std_attention'] / (l['mean_attention'] + 1e-10)
        logger.info(f"  Layer {layer_idx}: selectivity_ratio={l['selectivity_ratio']:.4f}")

    return results


def analyze_virtual_node(exp_name, dataset, device='cuda'):
    """
    Load trained GIN+VN model and analyze virtual node representations.
    Compute correlation between VN features and target Tg values.
    """
    logger.info(f"Analyzing virtual node representations for {exp_name}")

    train_idx, val_idx, test_idx = get_data_splits(len(dataset), seed=SEED)
    test_loader = PyGDataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False)

    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

    model = GINModel(node_dim=node_dim, edge_dim=edge_dim,
                     hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                     has_virtual_node=True)
    model_path = os.path.join(MODELS_DIR, f'{exp_name}_best.pt')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()

    vn_representations = []
    targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            n_nodes = batch.x.size(0)
            # Virtual node is the last node
            vn_feat = batch.x[-1]  # Input features of virtual node (all zeros)

            # Run through embedding and GIN layers to get final VN representation
            h = model.node_embedding(batch.x)
            for i in range(model.num_layers):
                h = model.convs[i](h, batch.edge_index, batch.edge_attr)
                h = model.batch_norms[i](h)
                h = torch.relu(h)

            # Virtual node final representation
            vn_repr = h[-1]  # Last node = virtual node
            vn_representations.append(vn_repr.cpu().numpy())
            targets.append(batch.y.item())

    vn_representations = np.array(vn_representations)
    targets = np.array(targets)

    # Compute correlation between each VN feature dimension and target
    correlations = []
    for dim in range(vn_representations.shape[1]):
        corr = np.corrcoef(vn_representations[:, dim], targets)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    results = {
        'exp_name': exp_name,
        'n_molecules_analyzed': len(targets),
        'vn_repr_dim': vn_representations.shape[1],
        'mean_abs_correlation': np.mean(np.abs(correlations)),
        'max_abs_correlation': np.max(np.abs(correlations)),
        'top5_corr_dims': sorted(range(len(correlations)),
                                  key=lambda i: abs(correlations[i]), reverse=True)[:5],
        'top5_corr_values': [correlations[i] for i in sorted(range(len(correlations)),
                                  key=lambda i: abs(correlations[i]), reverse=True)[:5]],
        'vn_repr_norm': np.mean(np.linalg.norm(vn_representations, axis=1)),
    }

    logger.info(f"  VN repr dim: {results['vn_repr_dim']}")
    logger.info(f"  Mean |corr(VN_dim, Tg)|: {results['mean_abs_correlation']:.4f}")
    logger.info(f"  Max  |corr(VN_dim, Tg)|: {results['max_abs_correlation']:.4f}")
    logger.info(f"  Top-5 corr dims: {results['top5_corr_dims']}")
    logger.info(f"  Top-5 corr values: {[f'{v:.4f}' for v in results['top5_corr_values']]}")
    logger.info(f"  Mean VN repr norm: {results['vn_repr_norm']:.4f}")

    return results


def run_all_diagnostics(all_results, device='cuda'):
    """Run all diagnostic analyses."""
    datasets = {}
    for mode in ['trimer', 'cyclized', 'trimer_vn', 'cyclized_vn']:
        datasets[mode] = TgDataset(
            root=PROCESSED_DIR,
            csv_path=CSV_PATH,
            mode=mode,
        )

    diagnostics = {
        'attention': {},
        'virtual_node': {},
    }

    # Attention weight analysis for GIN-GAT models
    for exp_name in ['gin_gat_trimer', 'gin_gat_cyclized']:
        mode = 'trimer' if 'trimer' in exp_name else 'cyclized'
        result = analyze_attention_weights(exp_name, datasets[mode], device=device)
        diagnostics['attention'][exp_name] = result

    # Virtual node analysis for VN models
    for exp_name in ['gin_vn_trimer', 'gin_vn_cyclized']:
        mode = 'trimer_vn' if 'trimer' in exp_name else 'cyclized_vn'
        result = analyze_virtual_node(exp_name, datasets[mode], device=device)
        diagnostics['virtual_node'][exp_name] = result

    return diagnostics


def generate_report(all_results, diagnostics):
    """Generate the comprehensive summary report."""
    report_lines = []

    report_lines.append("=" * 100)
    report_lines.append("DAY 1 EXPERIMENT 2: ARCHITECTURE IMPROVEMENT — GIN-GAT HYBRID & VIRTUAL NODE")
    report_lines.append("=" * 100)
    report_lines.append("")

    # Comparison table
    report_lines.append("RESULTS COMPARISON TABLE")
    report_lines.append("-" * 100)
    header = f"{'Model':<20} {'Data':<20} {'Train RMSE':<12} {'Val RMSE':<12} {'Test RMSE':<12} " \
             f"{'Best Epoch':<12} {'Params':<12} {'Gap':<10}"
    report_lines.append(header)
    report_lines.append("-" * 100)

    # Reference baselines
    baselines = [
        ("GIN (ref)", "Monomer", 36.38, 41.37, 52.71, 30, "445,700", 16.33),
        ("GIN (ref)", "Trimer", 41.29, 39.65, 50.19, 15, "~445K", 8.90),
        ("GIN (ref)", "Cyclized", 26.53, 39.62, 53.28, 64, "~445K", 26.75),
    ]
    for b in baselines:
        line = f"{b[0]:<20} {b[1]:<20} {b[2]:<12.2f} {b[3]:<12.2f} {b[4]:<12.2f} " \
               f"{b[5]:<12} {b[6]:<12} {b[7]:<10.2f}"
        report_lines.append(line)

    report_lines.append("-" * 100)

    for r in all_results:
        data_label = r['exp_name'].split('_', 2)[-1].replace('_', ' + ')
        model_label = "GIN-GAT" if r['model_type'] == 'gingat' else "GIN+VN"
        line = f"{model_label:<20} {data_label:<20} {r['train_rmse']:<12.4f} {r['val_rmse']:<12.4f} " \
               f"{r['test_rmse']:<12.4f} {r['best_epoch']:<12} {r['n_params']:<12,} {r['overfitting_gap']:<10.4f}"
        report_lines.append(line)

    report_lines.append("")
    report_lines.append("")

    # Diagnostic Analysis - Attention Weights
    report_lines.append("=" * 80)
    report_lines.append("ATTENTION WEIGHT ANALYSIS (GIN-GAT MODELS)")
    report_lines.append("=" * 80)

    for exp_name, attn_data in diagnostics.get('attention', {}).items():
        report_lines.append(f"\n--- {exp_name} ---")
        report_lines.append(f"  Molecules analyzed: {attn_data['n_molecules_analyzed']}")
        for layer_idx, layer_data in attn_data['layers'].items():
            report_lines.append(f"  Layer {layer_idx}:")
            report_lines.append(f"    Mean attention weight: {layer_data['mean_attention']:.6f}")
            report_lines.append(f"    Std attention weight:  {layer_data['std_attention']:.6f}")
            report_lines.append(f"    Max attention weight:  {layer_data['max_attention']:.6f}")
            report_lines.append(f"    Min attention weight:  {layer_data['min_attention']:.6f}")
            report_lines.append(f"    Selectivity ratio (std/mean): {layer_data.get('selectivity_ratio', 'N/A')}")
            if 'selectivity_ratio' in layer_data:
                sr = layer_data['selectivity_ratio']
                if sr > 1.0:
                    report_lines.append(f"    -> HIGH selectivity: attention is non-uniform (attention is working)")
                elif sr > 0.5:
                    report_lines.append(f"    -> MODERATE selectivity: some neighbor preference")
                else:
                    report_lines.append(f"    -> LOW selectivity: attention is near-uniform (~sum aggregation)")

    # Diagnostic Analysis - Virtual Node
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("VIRTUAL NODE REPRESENTATION ANALYSIS")
    report_lines.append("=" * 80)

    for exp_name, vn_data in diagnostics.get('virtual_node', {}).items():
        report_lines.append(f"\n--- {exp_name} ---")
        report_lines.append(f"  Molecules analyzed: {vn_data['n_molecules_analyzed']}")
        report_lines.append(f"  VN representation dim: {vn_data['vn_repr_dim']}")
        report_lines.append(f"  Mean |corr(VN_dim, Tg)|: {vn_data['mean_abs_correlation']:.4f}")
        report_lines.append(f"  Max  |corr(VN_dim, Tg)|: {vn_data['max_abs_correlation']:.4f}")
        report_lines.append(f"  Top-5 correlated dims: {vn_data['top5_corr_dims']}")
        report_lines.append(f"  Top-5 corr values: {[f'{v:.4f}' for v in vn_data['top5_corr_values']]}")
        report_lines.append(f"  Mean VN repr L2 norm: {vn_data['vn_repr_norm']:.4f}")

        corr = vn_data['mean_abs_correlation']
        if corr > 0.3:
            report_lines.append(f"  -> VN has moderate-strong correlation with Tg (VN is informative)")
        elif corr > 0.1:
            report_lines.append(f"  -> VN has weak correlation with Tg (VN partially informative)")
        else:
            report_lines.append(f"  -> VN has very weak correlation with Tg (VN not capturing Tg well)")

    # Key Findings
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("KEY FINDINGS & HYPOTHESIS EVALUATION")
    report_lines.append("=" * 80)

    # Find best results
    if all_results:
        best_result = min(all_results, key=lambda r: r['test_rmse'])
        report_lines.append(f"\nBest performing model: {best_result['exp_name']} "
                            f"(Test RMSE={best_result['test_rmse']:.4f})")

        # Compare with baselines
        trimer_results = [r for r in all_results if 'trimer' in r['exp_name'] and 'cyclized' not in r['exp_name']]
        cyclized_results = [r for r in all_results if 'cyclized' in r['exp_name']]

        report_lines.append("\nHypothesis 1 (GIN-GAT improves on trimer):")
        gingat_trimer = [r for r in trimer_results if r['model_type'] == 'gingat']
        if gingat_trimer:
            r = gingat_trimer[0]
            delta = 50.19 - r['test_rmse']
            report_lines.append(f"  GIN-GAT Trimer: {r['test_rmse']:.4f} (vs GIN Trimer 50.19, delta={delta:+.4f})")
            report_lines.append(f"  {'SUPPORTED' if delta > 0 else 'NOT SUPPORTED'}")

        report_lines.append("\nHypothesis 2 (Virtual node improves on trimer):")
        vn_trimer = [r for r in trimer_results if r['model_type'] == 'gin']
        if vn_trimer:
            r = vn_trimer[0]
            delta = 50.19 - r['test_rmse']
            report_lines.append(f"  GIN+VN Trimer: {r['test_rmse']:.4f} (vs GIN Trimer 50.19, delta={delta:+.4f})")
            report_lines.append(f"  {'SUPPORTED' if delta > 0 else 'NOT SUPPORTED'}")

        report_lines.append("\nHypothesis 3 (Attention rescues cyclized from overfitting):")
        gingat_cyclized = [r for r in cyclized_results if r['model_type'] == 'gingat']
        if gingat_cyclized:
            r = gingat_cyclized[0]
            delta = 53.28 - r['test_rmse']
            gap_delta = 26.75 - r['overfitting_gap']
            report_lines.append(f"  GIN-GAT Cyclized: test={r['test_rmse']:.4f} (vs GIN Cyclized 53.28, delta={delta:+.4f})")
            report_lines.append(f"  Overfitting gap: {r['overfitting_gap']:.4f} (vs 26.75, delta={gap_delta:+.4f})")
            report_lines.append(f"  {'SUPPORTED' if delta > 0 else 'NOT SUPPORTED'}")

        report_lines.append("\nHypothesis 4 (Virtual node rescues cyclized from overfitting):")
        vn_cyclized = [r for r in cyclized_results if r['model_type'] == 'gin']
        if vn_cyclized:
            r = vn_cyclized[0]
            delta = 53.28 - r['test_rmse']
            gap_delta = 26.75 - r['overfitting_gap']
            report_lines.append(f"  GIN+VN Cyclized: test={r['test_rmse']:.4f} (vs GIN Cyclized 53.28, delta={delta:+.4f})")
            report_lines.append(f"  Overfitting gap: {r['overfitting_gap']:.4f} (vs 26.75, delta={gap_delta:+.4f})")
            report_lines.append(f"  {'SUPPORTED' if delta > 0 else 'NOT SUPPORTED'}")

    report_text = "\n".join(report_lines)
    return report_text


if __name__ == "__main__":
    from train import main as train_main
    # First run training if not already done
    import glob
    model_files = glob.glob(os.path.join(MODELS_DIR, "*_best.pt"))
    if len(model_files) < 4:
        logger.info("Running training first...")
        all_results = train_main()
    else:
        logger.info("Found existing model checkpoints, loading results...")
        # Reconstruct results from CSV
        import csv as csv_mod
        all_results = []
        with open('./results.csv', 'r') as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                all_results.append({
                    'exp_name': row['Experiment'],
                    'model_type': row['Model'],
                    'train_rmse': float(row['Train RMSE']),
                    'val_rmse': float(row['Val RMSE']),
                    'test_rmse': float(row['Test RMSE']),
                    'best_epoch': int(row['Best Epoch']),
                    'total_epochs': int(row['Total Epochs']),
                    'n_params': int(row['Params'].replace(',', '')),
                    'overfitting_gap': float(row['Overfitting Gap']),
                })

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diagnostics = run_all_diagnostics(all_results, device=device)

    report_text = generate_report(all_results, diagnostics)
    with open('./summary_report.txt', 'w') as f:
        f.write(report_text)
    logger.info(f"\nReport saved to ./summary_report.txt")
    print(report_text)

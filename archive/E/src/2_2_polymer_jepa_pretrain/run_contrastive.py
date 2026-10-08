"""
Run contrastive pre-training pipeline (SimCLR-style) for polymer Tg prediction.

This replaces the JEPA approach with a contrastive learning approach that:
1. Creates positive pairs via graph augmentations
2. Uses NT-Xent loss for discriminative representations
3. VICReg on encoder output to prevent dimensional collapse
"""

import os
import sys
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

WORK_DIR = '/home/home_public/aisci/AIscientist/src/2_2_polymer_jepa_pretrain'
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

DATA_DIR = os.path.join(WORK_DIR, 'data')
FIGURE_DIR = os.path.join(WORK_DIR, 'figures')
CHECKPOINT_DIR = os.path.join(WORK_DIR, 'checkpoints')
PI1M_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv'
TG_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'

MAX_PI1M_SAMPLES = 300000
NUM_PRETRAIN_EPOCHS = 30


def phase1_data_preprocessing():
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("="*70)

    from data_preprocessing import PI1MTrimerDataset, TgTrimerDataset, get_data_splits

    pi1m_dataset = PI1MTrimerDataset(root=DATA_DIR, csv_path=PI1M_PATH, max_samples=MAX_PI1M_SAMPLES)
    logger.info(f"PI1M dataset size: {len(pi1m_dataset)}")

    tg_dataset = TgTrimerDataset(root=DATA_DIR, csv_path=TG_PATH)
    logger.info(f"Tg dataset size: {len(tg_dataset)}")

    train_idx, val_idx, test_idx = get_data_splits(tg_dataset)
    logger.info(f"Tg splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return pi1m_dataset, tg_dataset


def phase2_contrastive_pretrain(pi1m_dataset):
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: CONTRASTIVE PRE-TRAINING ON PI1M")
    logger.info("="*70)

    import torch
    from torch_geometric.loader import DataLoader
    from data_preprocessing import TOTAL_ATOM_FEATURE_DIM
    from contrastive_pretrain import GINEncoder, ProjectionHead, train_contrastive

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_loader = DataLoader(
        pi1m_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    in_dim = TOTAL_ATOM_FEATURE_DIM
    hidden_dim = 256

    encoder = GINEncoder(in_dim, hidden_dim, dropout=0.1).to(device)
    proj_head = ProjectionHead(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=128).to(device)

    loss_history, loss_components, collapse_history = train_contrastive(
        student_encoder=encoder,
        projection_head=proj_head,
        train_loader=train_loader,
        device=device,
        num_epochs=NUM_PRETRAIN_EPOCHS,
        lr=1e-3,
        weight_decay=1e-5,
        grad_clip=5.0,
        save_dir=WORK_DIR,
        temperature=0.1,
        lambda_var=25.0,
        lambda_cov=1.0,
        variance_threshold=1.0,
        drop_rate=0.2,
        edge_drop_rate=0.2,
    )

    # Now save the encoder with 'teacher_encoder' key for compatibility with finetune.py
    checkpoint_path = os.path.join(WORK_DIR, 'pretrained_encoder_contrastive.pt')
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    ckpt['teacher_encoder'] = ckpt.pop('student_encoder')
    torch.save(ckpt, checkpoint_path)
    logger.info(f"Re-saved encoder with 'teacher_encoder' key for fine-tuning compatibility")

    return loss_history, loss_components, collapse_history


def phase3_finetune(tg_dataset):
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: FINE-TUNING ON Tg")
    logger.info("="*70)

    import torch
    from torch_geometric.loader import DataLoader
    from torch.utils.data import Subset
    from data_preprocessing import get_data_splits, TOTAL_ATOM_FEATURE_DIM
    from jepa_pretrain import GINEncoder
    from finetune import (
        TgPredictor, RegressionHead,
        strategy_baseline, strategy_a_partial_freeze,
        strategy_b_end_to_end, strategy_c_full_freeze,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    checkpoint_path = os.path.join(WORK_DIR, 'pretrained_encoder_contrastive.pt')

    all_results = {}

    # Baseline
    result_baseline = strategy_baseline(
        make_encoder, make_head,
        train_loader, val_loader, test_loader, device, CHECKPOINT_DIR
    )
    all_results['baseline'] = result_baseline

    # Pre-trained strategies
    if os.path.exists(checkpoint_path):
        result_a = strategy_a_partial_freeze(
            make_encoder, make_head, checkpoint_path,
            train_loader, val_loader, test_loader, device, CHECKPOINT_DIR
        )
        all_results['strategy_A'] = result_a

        result_b = strategy_b_end_to_end(
            make_encoder, make_head, checkpoint_path,
            train_loader, val_loader, test_loader, device, CHECKPOINT_DIR
        )
        all_results['strategy_B'] = result_b

        result_c = strategy_c_full_freeze(
            make_encoder, make_head, checkpoint_path,
            train_loader, val_loader, test_loader, device, CHECKPOINT_DIR
        )
        all_results['strategy_C'] = result_c
    else:
        logger.warning(f"Pre-trained encoder not found at {checkpoint_path}")

    with open(os.path.join(WORK_DIR, 'finetune_results_contrastive.json'), 'w') as f:
        json.dump({k: v for k, v in all_results.items()}, f, indent=2, default=str)

    return all_results


def phase4_analysis(all_results, tg_dataset):
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: DIAGNOSTIC ANALYSIS")
    logger.info("="*70)

    import torch
    import numpy as np
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool
    from data_preprocessing import TOTAL_ATOM_FEATURE_DIM, get_data_splits
    from jepa_pretrain import GINEncoder
    from analysis import compute_monomer_trimer_similarity

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained encoder and check representations
    checkpoint_path = os.path.join(WORK_DIR, 'pretrained_encoder_contrastive.pt')
    if os.path.exists(checkpoint_path):
        encoder = GINEncoder(TOTAL_ATOM_FEATURE_DIM, hidden_dim=256, dropout=0.1).to(device)
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
        encoder.load_state_dict(ckpt['teacher_encoder'])
        encoder.eval()

        # Compute monomer-trimer similarity
        mono_trimer_sim, _ = compute_monomer_trimer_similarity(encoder, tg_dataset, device)
        logger.info(f"Monomer-Trimer Cosine Similarity: {mono_trimer_sim:.4f}")

        # Check embedding quality on Tg test set
        _, _, test_idx = get_data_splits(tg_dataset)
        from torch.utils.data import Subset
        test_loader = DataLoader(Subset(tg_dataset, test_idx), batch_size=128, shuffle=False)

        embeddings = []
        targets = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                node_emb = encoder(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                graph_emb = global_mean_pool(node_emb, batch.batch)
                embeddings.append(graph_emb.cpu().numpy())
                targets.append(batch.y.squeeze(-1).cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Dimensional analysis
        dim_vars = embeddings.var(axis=0)
        logger.info(f"Embedding dim analysis: active={((dim_vars > 1e-4).sum())}/256, "
                     f"mean_var={dim_vars.mean():.4f}")

        # Pairwise cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        idx = np.random.choice(len(embeddings), min(500, len(embeddings)), replace=False)
        cos_sims = cosine_similarity(embeddings[idx])
        triu_idx = np.triu_indices(len(idx), k=1)
        pairwise = cos_sims[triu_idx]
        logger.info(f"Pairwise cosine sim: mean={pairwise.mean():.4f}, "
                     f">0.99: {(pairwise > 0.99).mean()*100:.1f}%")

        # Correlation with Tg
        from scipy.stats import pearsonr
        active_mask = dim_vars > 1e-4
        corrs = []
        for dim in range(embeddings.shape[1]):
            if dim_vars[dim] < 1e-8:
                continue
            r, _ = pearsonr(embeddings[:, dim], targets)
            corrs.append(abs(r))
        if corrs:
            logger.info(f"Active dim correlation with Tg: max={max(corrs):.4f}, mean={np.mean(corrs):.4f}")
    else:
        mono_trimer_sim = -1.0

    # Generate results summary
    baseline_rmse = all_results['baseline']['test_rmse']
    summary_lines = [
        "=" * 70,
        "POLYMER CONTRASTIVE PRE-TRAINING EXPERIMENT RESULTS",
        "=" * 70,
        "",
        "PHASE 1: Contrastive Pre-training on PI1M",
        "-" * 40,
        f"Number of PI1M samples used: {MAX_PI1M_SAMPLES}",
        f"Number of epochs: {NUM_PRETRAIN_EPOCHS}",
        "",
        "PHASE 2: Fine-tuning on Tg",
        "-" * 40,
        f"{'Strategy':<30} {'Test RMSE':>10} {'Val RMSE':>10} {'Train RMSE':>10}",
        "-" * 70,
    ]
    for strategy_name in ['baseline', 'strategy_A', 'strategy_B', 'strategy_C']:
        if strategy_name in all_results:
            r = all_results[strategy_name]
            summary_lines.append(
                f"{strategy_name:<30} {r['test_rmse']:>10.4f} {r['val_rmse']:>10.4f} {r['train_rmse']:>10.4f}"
            )

    summary_lines.extend([
        "",
        "KEY FINDINGS",
        "-" * 40,
        f"Baseline (no pre-train) Test RMSE: {baseline_rmse:.4f}",
    ])

    for strategy_name in ['strategy_A', 'strategy_B', 'strategy_C']:
        if strategy_name in all_results:
            delta = all_results[strategy_name]['test_rmse'] - baseline_rmse
            summary_lines.append(
                f"{strategy_name} Test RMSE: {all_results[strategy_name]['test_rmse']:.4f} (delta: {delta:+.4f})"
            )

    any_beats = any(
        all_results[k]['test_rmse'] < 50.19
        for k in ['strategy_A', 'strategy_B', 'strategy_C']
        if k in all_results
    )
    summary_lines.extend([
        "",
        f"Any pre-training strategy beats baseline: {'True' if any_beats else 'False'}",
        "",
        f"Monomer-Trimer Cosine Similarity: {mono_trimer_sim:.4f}",
        "",
        "=" * 70,
        "SUCCESS CRITERION: ANY strategy with Test RMSE < 50.19 validates hypothesis",
        "=" * 70,
    ])

    summary_text = "\n".join(summary_lines)
    with open(os.path.join(WORK_DIR, 'results_summary.txt'), 'w') as f:
        f.write(summary_text)

    print(summary_text)
    return summary_text


if __name__ == '__main__':
    pipeline_start = time.time()

    # Phase 1: Data preprocessing (reuses existing data)
    pi1m_dataset, tg_dataset = phase1_data_preprocessing()
    phase1_time = time.time() - pipeline_start
    logger.info(f"Phase 1 completed in {phase1_time:.1f}s")

    # Phase 2: Contrastive pre-training
    phase2_start = time.time()
    loss_history, loss_components, collapse_history = phase2_contrastive_pretrain(pi1m_dataset)
    phase2_time = time.time() - phase2_start
    logger.info(f"Phase 2 completed in {phase2_time:.1f}s")

    # Phase 3: Fine-tuning
    phase3_start = time.time()
    all_results = phase3_finetune(tg_dataset)
    phase3_time = time.time() - phase3_start
    logger.info(f"Phase 3 completed in {phase3_time:.1f}s")

    # Phase 4: Analysis
    phase4_start = time.time()
    phase4_analysis(all_results, tg_dataset)
    phase4_time = time.time() - phase4_start
    logger.info(f"Phase 4 completed in {phase4_time:.1f}s")

    total_time = time.time() - pipeline_start
    logger.info(f"Total pipeline time: {total_time/3600:.2f} hours")

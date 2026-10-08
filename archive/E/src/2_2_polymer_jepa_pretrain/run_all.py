"""
Master script to run the full Polymer JEPA pre-training pipeline (v6):
1. Data preprocessing (PI1M + Tg) - reuse existing processed data
2. JEPA pre-training on PI1M with VICReg + node diversity
3. Fine-tuning on Tg (Strategies A/B/C + Baseline)
4. Diagnostic analysis with enhanced monitoring
"""

import os
import sys
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Working directory
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

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def phase1_data_preprocessing():
    """Phase 1: Convert PI1M and Tg to Trimer PyG datasets."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("="*70)

    from data_preprocessing import PI1MTrimerDataset, TgTrimerDataset, get_data_splits

    pi1m_dataset = PI1MTrimerDataset(
        root=DATA_DIR, csv_path=PI1M_PATH, max_samples=MAX_PI1M_SAMPLES,
    )
    logger.info(f"PI1M dataset size: {len(pi1m_dataset)}")

    tg_dataset = TgTrimerDataset(root=DATA_DIR, csv_path=TG_PATH)
    logger.info(f"Tg dataset size: {len(tg_dataset)}")

    train_idx, val_idx, test_idx = get_data_splits(tg_dataset)
    logger.info(f"Tg splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return pi1m_dataset, tg_dataset


def phase2_jepa_pretrain(pi1m_dataset):
    """Phase 2: JEPA pre-training on PI1M with VICReg + node diversity."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: JEPA PRE-TRAINING ON PI1M (v6: very strong VICReg var+cov)")
    logger.info("="*70)

    import torch
    from torch_geometric.loader import DataLoader
    from data_preprocessing import TOTAL_ATOM_FEATURE_DIM
    from jepa_pretrain import GINEncoder, JEPAPredictor, ProjectionHead, train_jepa

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_loader = DataLoader(
        pi1m_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    in_dim = TOTAL_ATOM_FEATURE_DIM
    hidden_dim = 256

    # Use dropout=0.1 instead of 0.5
    student_encoder = GINEncoder(in_dim, hidden_dim, dropout=0.1).to(device)
    teacher_encoder = GINEncoder(in_dim, hidden_dim, dropout=0.1).to(device)
    teacher_encoder.load_state_dict(student_encoder.state_dict())

    predictor = JEPAPredictor(context_dim=hidden_dim, pos_dim=3, hidden_dim=hidden_dim).to(device)

    # Projection heads for VICReg
    student_proj = ProjectionHead(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim).to(device)
    teacher_proj = ProjectionHead(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim).to(device)
    teacher_proj.load_state_dict(student_proj.state_dict())

    loss_history, loss_components, collapse_history = train_jepa(
        student_encoder=student_encoder,
        teacher_encoder=teacher_encoder,
        predictor=predictor,
        student_proj=student_proj,
        teacher_proj=teacher_proj,
        train_loader=train_loader,
        device=device,
        num_epochs=NUM_PRETRAIN_EPOCHS,
        lr=1e-3,
        weight_decay=1e-5,
        predictor_weight_decay=1e-4,
        ema_momentum_start=0.996,
        ema_momentum_end=0.999,
        ema_ramp_epochs=5,
        grad_clip=5.0,
        save_dir=WORK_DIR,
        context_ratio=0.6,
        target_ratio=0.1,
        lambda_inv=1.0,
        lambda_var=50.0,
        lambda_cov=50.0,
        lambda_node_div=5.0,
        variance_threshold=1.0,
    )

    return loss_history, loss_components, collapse_history


def phase3_finetune(tg_dataset):
    """Phase 3: Fine-tuning on Tg with all strategies."""
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

    checkpoint_path = os.path.join(WORK_DIR, 'pretrained_encoder_v6.pt')

    all_results = {}

    # Baseline first
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

    results_for_save = {}
    for k, v in all_results.items():
        results_for_save[k] = {kk: vv for kk, vv in v.items()}

    with open(os.path.join(WORK_DIR, 'finetune_results_v6.json'), 'w') as f:
        json.dump(results_for_save, f, indent=2, default=str)

    return all_results


def phase4_analysis(all_results, tg_dataset, loss_components=None, collapse_history=None):
    """Phase 4: Diagnostic analysis with enhanced monitoring."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: DIAGNOSTIC ANALYSIS")
    logger.info("="*70)

    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from data_preprocessing import TOTAL_ATOM_FEATURE_DIM
    from jepa_pretrain import GINEncoder
    from analysis import (
        plot_finetune_comparison,
        plot_training_curves, compute_tsne,
        compute_monomer_trimer_similarity, generate_results_summary
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pre-training loss curves
    jepa_losses = []
    loss_path = os.path.join(WORK_DIR, 'jepa_loss_history_v6.npz')
    if os.path.exists(loss_path):
        data = np.load(loss_path)
        jepa_losses = data['total'].tolist()

        # Plot all loss components
        n_components = min(3, len(data.files) - 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(jepa_losses) + 1)

        axes[0, 0].plot(epochs, data['total'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=13)
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, data['jepa'], 'r-', linewidth=2)
        axes[0, 1].set_title('JEPA (Invariance) Loss', fontsize=13)
        axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, data['variance'], 'g-', linewidth=2)
        axes[1, 0].set_title('VICReg Variance Loss', fontsize=13)
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, data['node_div'], 'c-', linewidth=2)
        axes[1, 1].set_title('Node Diversity Loss', fontsize=13)
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('JEPA Pre-training Loss Components (v6)', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(FIGURE_DIR, 'jepa_pretrain_loss_v6.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved loss components plot to {save_path}")

    # Collapse monitoring plot
    if collapse_history:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        valid = [c for c in collapse_history if c['mean_embedding_var'] >= 0]
        if valid:
            epochs_c = [c['epoch'] for c in valid]
            axes[0].plot(epochs_c, [c['mean_embedding_var'] for c in valid], 'b-', linewidth=2)
            axes[0].axhline(y=1.0, color='r', linestyle='--', label='VICReg threshold')
            axes[0].set_title('Graph Embedding Variance'); axes[0].set_xlabel('Epoch')
            axes[0].legend(); axes[0].grid(True, alpha=0.3)

            axes[1].plot(epochs_c, [c['mean_cosine_sim'] for c in valid], 'g-', linewidth=2)
            axes[1].axhline(y=0.99, color='r', linestyle='--', label='Collapse threshold')
            axes[1].set_title('Graph Cosine Similarity'); axes[1].set_xlabel('Epoch')
            axes[1].legend(); axes[1].grid(True, alpha=0.3)

            node_cos = [c['node_within_cos_sim'] for c in valid if c['node_within_cos_sim'] >= 0]
            node_epochs = [c['epoch'] for c in valid if c['node_within_cos_sim'] >= 0]
            if node_cos:
                axes[2].plot(node_epochs, node_cos, 'm-', linewidth=2)
                axes[2].axhline(y=0.99, color='r', linestyle='--', label='Node collapse')
                axes[2].set_title('Node Within-Graph Cos Sim'); axes[2].set_xlabel('Epoch')
                axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(FIGURE_DIR, 'collapse_monitoring_v6.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    plot_finetune_comparison(all_results, FIGURE_DIR)
    plot_training_curves(all_results, FIGURE_DIR)

    # t-SNE and similarity
    mean_sim = 0.0
    checkpoint_path = os.path.join(WORK_DIR, 'pretrained_encoder_v6.pt')
    if os.path.exists(checkpoint_path):
        encoder = GINEncoder(TOTAL_ATOM_FEATURE_DIM, hidden_dim=256, dropout=0.1).to(device)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        encoder.load_state_dict(checkpoint['teacher_encoder'])

        from torch.utils.data import Subset
        from data_preprocessing import get_data_splits
        _, _, test_idx = get_data_splits(tg_dataset)
        test_dataset = Subset(tg_dataset, test_idx)

        compute_tsne(encoder, test_dataset, device, FIGURE_DIR)
        mean_sim, _ = compute_monomer_trimer_similarity(encoder, tg_dataset, device)

    generate_results_summary(
        all_results, jepa_losses, mean_sim,
        MAX_PI1M_SAMPLES,
        os.path.join(WORK_DIR, 'results_summary.txt')
    )


def main():
    total_start = time.time()

    t0 = time.time()
    pi1m_dataset, tg_dataset = phase1_data_preprocessing()
    logger.info(f"Phase 1 completed in {time.time()-t0:.1f}s")

    t0 = time.time()
    loss_history, loss_components, collapse_history = phase2_jepa_pretrain(pi1m_dataset)
    logger.info(f"Phase 2 completed in {time.time()-t0:.1f}s")

    del pi1m_dataset
    import gc; gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.time()
    all_results = phase3_finetune(tg_dataset)
    logger.info(f"Phase 3 completed in {time.time()-t0:.1f}s")

    t0 = time.time()
    phase4_analysis(all_results, tg_dataset, loss_components, collapse_history)
    logger.info(f"Phase 4 completed in {time.time()-t0:.1f}s")

    total_time = time.time() - total_start
    logger.info(f"\nTotal pipeline time: {total_time/3600:.2f} hours")

    summary_path = os.path.join(WORK_DIR, 'results_summary.txt')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            print(f.read())


if __name__ == '__main__':
    main()

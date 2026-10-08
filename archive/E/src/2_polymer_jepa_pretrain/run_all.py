"""
Master script to run the full Polymer JEPA pre-training pipeline:
1. Data preprocessing (PI1M + Tg)
2. JEPA pre-training on PI1M
3. Fine-tuning on Tg (Strategies A/B/C + Baseline)
4. Diagnostic analysis
"""

import os
import sys
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Working directory
WORK_DIR = '/home/home_public/aisci/AIscientist/src/2_polymer_jepa_pretrain'
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)

DATA_DIR = os.path.join(WORK_DIR, 'data')
FIGURE_DIR = os.path.join(WORK_DIR, 'figures')
CHECKPOINT_DIR = os.path.join(WORK_DIR, 'checkpoints')
PI1M_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv'
TG_PATH = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'

MAX_PI1M_SAMPLES = 300000  # Use 300K subset for first experiment

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def phase1_data_preprocessing():
    """Phase 1: Convert PI1M and Tg to Trimer PyG datasets."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("="*70)

    from data_preprocessing import PI1MTrimerDataset, TgTrimerDataset, get_data_splits

    # Process PI1M
    logger.info("Processing PI1M dataset...")
    pi1m_dataset = PI1MTrimerDataset(
        root=DATA_DIR,
        csv_path=PI1M_PATH,
        max_samples=MAX_PI1M_SAMPLES,
    )
    logger.info(f"PI1M dataset size: {len(pi1m_dataset)}")

    # Process Tg
    logger.info("Processing Tg dataset...")
    tg_dataset = TgTrimerDataset(
        root=DATA_DIR,
        csv_path=TG_PATH,
    )
    logger.info(f"Tg dataset size: {len(tg_dataset)}")

    # Verify splits
    train_idx, val_idx, test_idx = get_data_splits(tg_dataset)
    logger.info(f"Tg splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return pi1m_dataset, tg_dataset


def phase2_jepa_pretrain(pi1m_dataset):
    """Phase 2: JEPA pre-training on PI1M."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: JEPA PRE-TRAINING ON PI1M")
    logger.info("="*70)

    import torch
    from torch_geometric.loader import DataLoader
    from data_preprocessing import TOTAL_ATOM_FEATURE_DIM
    from jepa_pretrain import GINEncoder, JEPAPredictor, train_jepa

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_loader = DataLoader(
        pi1m_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    in_dim = TOTAL_ATOM_FEATURE_DIM
    hidden_dim = 256

    student_encoder = GINEncoder(in_dim, hidden_dim, dropout=0.5).to(device)
    teacher_encoder = GINEncoder(in_dim, hidden_dim, dropout=0.5).to(device)
    teacher_encoder.load_state_dict(student_encoder.state_dict())

    predictor = JEPAPredictor(context_dim=hidden_dim, pos_dim=3, hidden_dim=hidden_dim).to(device)

    loss_history = train_jepa(
        student_encoder=student_encoder,
        teacher_encoder=teacher_encoder,
        predictor=predictor,
        train_loader=train_loader,
        device=device,
        num_epochs=15,
        lr=1e-3,
        weight_decay=1e-5,
        ema_momentum_start=0.996,
        ema_momentum_end=0.999,
        grad_clip=5.0,
        save_dir=WORK_DIR,
        context_ratio=0.6,
        target_ratio=0.1,
    )

    return loss_history


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
        return GINEncoder(TOTAL_ATOM_FEATURE_DIM, hidden_dim=256, dropout=0.5)

    def make_head():
        return RegressionHead(in_dim=256, hidden_dim=128, dropout=0.5)

    checkpoint_path = os.path.join(WORK_DIR, 'pretrained_encoder.pt')

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

    # Save results
    results_for_save = {}
    for k, v in all_results.items():
        results_for_save[k] = {kk: vv for kk, vv in v.items()}

    with open(os.path.join(WORK_DIR, 'finetune_results.json'), 'w') as f:
        json.dump(results_for_save, f, indent=2, default=str)

    return all_results


def phase4_analysis(all_results, tg_dataset):
    """Phase 4: Diagnostic analysis."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: DIAGNOSTIC ANALYSIS")
    logger.info("="*70)

    import torch
    from data_preprocessing import TOTAL_ATOM_FEATURE_DIM
    from jepa_pretrain import GINEncoder
    from analysis import (
        plot_pretraining_loss, plot_finetune_comparison,
        plot_training_curves, compute_tsne,
        compute_monomer_trimer_similarity, generate_results_summary
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pre-training loss
    loss_path = os.path.join(WORK_DIR, 'jepa_loss_history.npy')
    jepa_losses = []
    if os.path.exists(loss_path):
        jepa_losses = plot_pretraining_loss(loss_path, FIGURE_DIR)

    # Comparison plot
    plot_finetune_comparison(all_results, FIGURE_DIR)
    plot_training_curves(all_results, FIGURE_DIR)

    # t-SNE and similarity
    mean_sim = 0.0
    checkpoint_path = os.path.join(WORK_DIR, 'pretrained_encoder.pt')
    if os.path.exists(checkpoint_path):
        encoder = GINEncoder(TOTAL_ATOM_FEATURE_DIM, hidden_dim=256, dropout=0.5).to(device)
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        encoder.load_state_dict(checkpoint['teacher_encoder'])

        from torch.utils.data import Subset
        from data_preprocessing import get_data_splits
        _, _, test_idx = get_data_splits(tg_dataset)
        test_dataset = Subset(tg_dataset, test_idx)

        compute_tsne(encoder, test_dataset, device, FIGURE_DIR)
        mean_sim, _ = compute_monomer_trimer_similarity(encoder, tg_dataset, device)

    # Generate summary
    generate_results_summary(
        all_results, jepa_losses, mean_sim,
        MAX_PI1M_SAMPLES,
        os.path.join(WORK_DIR, 'results_summary.txt')
    )


def main():
    total_start = time.time()

    # Phase 1
    t0 = time.time()
    pi1m_dataset, tg_dataset = phase1_data_preprocessing()
    logger.info(f"Phase 1 completed in {time.time()-t0:.1f}s")

    # Phase 2
    t0 = time.time()
    loss_history = phase2_jepa_pretrain(pi1m_dataset)
    logger.info(f"Phase 2 completed in {time.time()-t0:.1f}s")

    # Free PI1M memory
    del pi1m_dataset
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase 3
    t0 = time.time()
    all_results = phase3_finetune(tg_dataset)
    logger.info(f"Phase 3 completed in {time.time()-t0:.1f}s")

    # Phase 4
    t0 = time.time()
    phase4_analysis(all_results, tg_dataset)
    logger.info(f"Phase 4 completed in {time.time()-t0:.1f}s")

    total_time = time.time() - total_start
    logger.info(f"\nTotal pipeline time: {total_time/3600:.2f} hours")

    # Print final summary
    summary_path = os.path.join(WORK_DIR, 'results_summary.txt')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            print(f.read())


if __name__ == '__main__':
    main()

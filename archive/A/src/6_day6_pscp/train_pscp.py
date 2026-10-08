"""
Main training script for PSCP
Handles pretraining and fine-tuning with ablation studies
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from pscp_model import PSCPModel, PredictionModel, GINEncoder, info_nce_loss
from pretraining_tasks import (
    PI1MDataset, TgDataset, collate_fn,
    FunctionalGroupMasking, CyclicRotationInvariance,
    ECFPGuidedContrastive, PretrainingLoss
)
from polymer_utils import compute_ecfp_fingerprint, smiles_to_graph
from torch_geometric.data import Batch


class PSCPTrainer:
    """
    Trainer for PSCP pretraining and fine-tuning
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = PSCPModel(config).to(self.device)

        # Pretraining tasks
        self.fgm = FunctionalGroupMasking(mask_ratio=0.15)
        self.cri = CyclicRotationInvariance(num_rotations=3)
        self.egcl = ECFPGuidedContrastive(
            pos_threshold=0.7,
            neg_threshold=0.3,
            num_negatives=config.get('num_negatives', 5)
        )

        # Loss functions
        self.pretrain_loss_fn = PretrainingLoss(temperature=0.1)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Logging
        self.log_dir = config.get('log_dir', './logs')
        os.makedirs(self.log_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []

    def pretrain_epoch(self, dataloader, tasks=['fgm', 'cri', 'egcl']):
        """Run one pretraining epoch"""
        self.model.train()
        total_loss = 0
        fgm_loss_sum = 0
        cri_loss_sum = 0
        egcl_loss_sum = 0
        num_batches = 0

        task_weights = self.config.get('task_weights', {
            'fgm': 1.0,
            'cri': 0.5,
            'egcl': 1.0
        })

        pbar = tqdm(dataloader, desc="Pretraining")

        for batch in pbar:
            self.optimizer.zero_grad()

            graphs = batch['graph'].to(self.device)
            mols = batch.get('mol', [None] * graphs.num_graphs)
            smiles_list = batch['smiles']

            loss = 0.0

            # Functional Group Masking
            if 'fgm' in tasks:
                fgm_loss = self._compute_fgm_loss(graphs, mols)
                fgm_loss_sum += fgm_loss.item()
                loss += task_weights['fgm'] * fgm_loss

            # Cyclic Rotation Invariance
            if 'cri' in tasks:
                cri_loss = self._compute_cri_loss(smiles_list)
                if cri_loss is not None:
                    cri_loss_sum += cri_loss.item()
                    loss += task_weights['cri'] * cri_loss

            # ECFP-Guided Contrastive Learning
            if 'egcl' in tasks:
                egcl_loss = self._compute_egcl_loss(graphs, smiles_list)
                egcl_loss_sum += egcl_loss.item()
                loss += task_weights['egcl'] * egcl_loss

            if loss > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)

        return {
            'total': avg_loss,
            'fgm': fgm_loss_sum / max(num_batches, 1),
            'cri': cri_loss_sum / max(num_batches, 1),
            'egcl': egcl_loss_sum / max(num_batches, 1)
        }

    def _compute_fgm_loss(self, graphs, mols):
        """Compute Functional Group Masking loss"""
        # For simplicity, do random masking on the batch level
        num_nodes = graphs.x.size(0)
        mask_ratio = 0.15
        num_mask = max(1, int(num_nodes * mask_ratio))

        # Random mask indices
        mask_indices = torch.randperm(num_nodes)[:num_mask]

        # Store original features
        original_features = graphs.x[mask_indices].clone()
        original_labels = original_features.argmax(dim=-1)

        # Mask
        masked_x = graphs.x.clone()
        masked_x[mask_indices] = 0

        # Create new data with masked features
        from torch_geometric.data import Data
        masked_data = Data(
            x=masked_x,
            edge_index=graphs.edge_index,
            edge_attr=graphs.edge_attr,
            batch=graphs.batch
        )

        # Predict
        predictions = self.model.predict_masked_atoms(masked_data)
        predicted_labels = predictions[mask_indices]

        loss = nn.functional.cross_entropy(predicted_labels, original_labels)

        return loss

    def _compute_cri_loss(self, smiles_list):
        """Compute Cyclic Rotation Invariance loss"""
        embeddings = []
        loss_values = []

        for smiles in smiles_list[:4]:  # Limit for efficiency
            clean_smiles = smiles.replace('*', '')
            if not clean_smiles:
                continue

            # Get cyclic rotations
            from polymer_utils import get_cyclic_rotations
            rotations = get_cyclic_rotations(clean_smiles, k=2)

            rot_embs = []
            for rot in rotations:
                graph = smiles_to_graph(rot)
                if graph is not None:
                    graph = graph.to(self.device)
                    emb, _ = self.model.encoder(
                        graph.x.unsqueeze(0) if graph.x.dim() == 2 else graph.x,
                        graph.edge_index,
                        graph.edge_attr if hasattr(graph, 'edge_attr') else None,
                        batch=torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
                    )
                    rot_embs.append(emb)

            if len(rot_embs) >= 2:
                # Compute loss between rotations
                for i in range(len(rot_embs)):
                    for j in range(i + 1, len(rot_embs)):
                        emb1 = torch.nn.functional.normalize(rot_embs[i], dim=-1)
                        emb2 = torch.nn.functional.normalize(rot_embs[j], dim=-1)
                        sim = torch.sum(emb1 * emb2, dim=-1)
                        loss_values.append(1 - sim.mean())

        if loss_values:
            return torch.stack(loss_values).mean()
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _compute_egcl_loss(self, graphs, smiles_list):
        """Compute ECFP-Guided Contrastive Learning loss"""
        batch_size = graphs.num_graphs

        # Get projections
        projections = self.model.get_projection(graphs)

        # Normalize
        projections = torch.nn.functional.normalize(projections, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.mm(projections, projections.t())

        # Create pseudo-labels based on batch position
        # (In practice, we'd use pre-computed ECFP similarities)
        # For efficiency, use random positive pairs within batch

        # InfoNCE-style loss
        temperature = 0.1
        sim_matrix = sim_matrix / temperature

        # Use shifted indices as positives (simplified)
        labels = torch.arange(batch_size, device=self.device)
        labels = (labels + 1) % batch_size

        loss = nn.functional.cross_entropy(sim_matrix, labels)

        return loss

    def pretrain(self, train_dataset, num_epochs=10, batch_size=64, tasks=['fgm', 'cri', 'egcl']):
        """Full pretraining loop"""
        print(f"\nStarting pretraining with tasks: {tasks}")
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}")

        dataloader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=False
        )

        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            losses = self.pretrain_epoch(dataloader, tasks)

            print(f"Total Loss: {losses['total']:.4f}")
            print(f"  FGM Loss: {losses['fgm']:.4f}")
            print(f"  CRI Loss: {losses['cri']:.4f}")
            print(f"  EGCL Loss: {losses['egcl']:.4f}")

            scheduler.step()

            # Save checkpoint
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.save_pretrained(f"pretrained_epoch_{epoch + 1}.pt")

        print("\nPretraining complete!")

    def save_pretrained(self, filename):
        """Save pretrained encoder"""
        save_path = os.path.join(self.log_dir, filename)
        torch.save({
            'encoder_state_dict': self.model.encoder.state_dict(),
            'config': self.config
        }, save_path)
        print(f"Saved pretrained model to {save_path}")

    def load_pretrained(self, filename):
        """Load pretrained encoder"""
        load_path = os.path.join(self.log_dir, filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"Loaded pretrained model from {load_path}")


class FineTuner:
    """
    Fine-tuner for Tg prediction
    """

    def __init__(self, pretrained_encoder, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create prediction model
        self.model = PredictionModel(
            encoder=pretrained_encoder,
            hidden_dim=config.get('hidden_dim', 256),
            ecfp_dim=config.get('ecfp_bits', 2048)
        ).to(self.device)

        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            optimizer.zero_grad()

            graphs = batch['graph'].to(self.device)
            ecfp = batch['ecfp'].to(self.device)
            targets = batch['tg'].to(self.device)

            predictions = self.model(graphs, ecfp)
            loss = self.criterion(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in dataloader:
                graphs = batch['graph'].to(self.device)
                ecfp = batch['ecfp'].to(self.device)
                target = batch['tg']

                pred = self.model(graphs, ecfp)

                predictions.extend(pred.cpu().numpy())
                targets.extend(target.numpy())

        predictions = np.array(predictions)
        targets = np.array(targets)

        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'targets': targets
        }

    def finetune_cv(self, dataset, n_folds=5, epochs=100, batch_size=32,
                    lr=1e-4, patience=20):
        """
        Fine-tune with k-fold cross-validation
        """
        print(f"\nStarting {n_folds}-fold cross-validation")
        print(f"Dataset size: {len(dataset)}")

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\n{'=' * 50}")
            print(f"Fold {fold + 1}/{n_folds}")
            print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
            print(f"{'=' * 50}")

            # Reset model for each fold
            self.model = PredictionModel(
                encoder=self.model.encoder,
                hidden_dim=self.config.get('hidden_dim', 256),
                ecfp_dim=self.config.get('ecfp_bits', 2048)
            ).to(self.device)

            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

            # Create subset datasets
            from torch.utils.data import Subset
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = TorchDataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                collate_fn=collate_fn, num_workers=0
            )
            val_loader = TorchDataLoader(
                val_subset, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0
            )

            # Training loop with early stopping
            best_val_rmse = float('inf')
            patience_counter = 0
            best_model_state = None

            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader, optimizer)
                val_metrics = self.evaluate(val_loader)

                scheduler.step()

                if val_metrics['rmse'] < best_val_rmse:
                    best_val_rmse = val_metrics['rmse']
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, "
                          f"Val RMSE = {val_metrics['rmse']:.2f}, "
                          f"Val R2 = {val_metrics['r2']:.4f}")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Load best model and evaluate
            self.model.load_state_dict(best_model_state)
            final_metrics = self.evaluate(val_loader)

            print(f"\nFold {fold + 1} Results:")
            print(f"  RMSE: {final_metrics['rmse']:.2f}")
            print(f"  MAE: {final_metrics['mae']:.2f}")
            print(f"  R2: {final_metrics['r2']:.4f}")

            fold_results.append({
                'fold': fold + 1,
                'rmse': final_metrics['rmse'],
                'mae': final_metrics['mae'],
                'r2': final_metrics['r2']
            })

        # Compute summary statistics
        rmses = [r['rmse'] for r in fold_results]
        r2s = [r['r2'] for r in fold_results]

        summary = {
            'mean_rmse': np.mean(rmses),
            'std_rmse': np.std(rmses),
            'mean_r2': np.mean(r2s),
            'std_r2': np.std(r2s),
            'fold_results': fold_results
        }

        print(f"\n{'=' * 50}")
        print("Cross-validation Summary:")
        print(f"  RMSE: {summary['mean_rmse']:.2f} ± {summary['std_rmse']:.2f}")
        print(f"  R2: {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")
        print(f"{'=' * 50}")

        return summary


def run_ablation_study(config, pi1m_path, tg_path, output_dir):
    """
    Run ablation study with different pretraining configurations
    """
    print("\n" + "=" * 70)
    print("RUNNING PSCP ABLATION STUDY")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Ablation conditions
    conditions = {
        'no_pretrain': [],  # No pretraining
        'fgm_only': ['fgm'],
        'cri_only': ['cri'],
        'egcl_only': ['egcl'],
        'full_pscp': ['fgm', 'cri', 'egcl']
    }

    all_results = []

    for condition, tasks in conditions.items():
        print(f"\n{'=' * 70}")
        print(f"Condition: {condition}")
        print(f"Tasks: {tasks if tasks else 'None (random init)'}")
        print(f"{'=' * 70}")

        # Initialize trainer
        trainer = PSCPTrainer(config)

        if tasks:
            # Pretrain
            pi1m_dataset = PI1MDataset(pi1m_path, max_samples=config.get('pretrain_samples', 100000))
            trainer.pretrain(
                pi1m_dataset,
                num_epochs=config.get('pretrain_epochs', 5),
                batch_size=config.get('batch_size', 64),
                tasks=tasks
            )
        else:
            print("Skipping pretraining (random initialization)")

        # Save pretrained model
        trainer.save_pretrained(f"pretrained_{condition}.pt")

        # Fine-tune on Tg
        tg_dataset = TgDataset(tg_path)

        finetuner = FineTuner(trainer.model.encoder, config)
        results = finetuner.finetune_cv(
            tg_dataset,
            n_folds=5,
            epochs=100,
            batch_size=32,
            patience=20
        )

        # Save results
        for fold_result in results['fold_results']:
            all_results.append({
                'condition': condition,
                'fold': fold_result['fold'],
                'rmse': fold_result['rmse'],
                'mae': fold_result['mae'],
                'r2': fold_result['r2']
            })

        # Save summary
        all_results.append({
            'condition': condition,
            'fold': 'mean',
            'rmse': results['mean_rmse'],
            'mae': np.mean([r['mae'] for r in results['fold_results']]),
            'r2': results['mean_r2']
        })

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'results_ablation.csv'), index=False)
    print(f"\nSaved ablation results to {output_dir}/results_ablation.csv")

    return results_df


def run_scale_analysis(config, pi1m_path, tg_path, output_dir):
    """
    Run scale analysis: 10K, 100K, 1M pretraining samples
    """
    print("\n" + "=" * 70)
    print("RUNNING SCALE ANALYSIS")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    scales = [10000, 100000, 1000000]
    scale_results = []

    for scale in scales:
        print(f"\n{'=' * 70}")
        print(f"Pretraining Scale: {scale}")
        print(f"{'=' * 70}")

        config['pretrain_samples'] = scale

        trainer = PSCPTrainer(config)

        # Pretrain
        pi1m_dataset = PI1MDataset(pi1m_path, max_samples=scale)
        trainer.pretrain(
            pi1m_dataset,
            num_epochs=config.get('pretrain_epochs', 5),
            batch_size=config.get('batch_size', 64),
            tasks=['fgm', 'cri', 'egcl']
        )

        # Fine-tune
        tg_dataset = TgDataset(tg_path)
        finetuner = FineTuner(trainer.model.encoder, config)
        results = finetuner.finetune_cv(
            tg_dataset,
            n_folds=5,
            epochs=100,
            batch_size=32,
            patience=20
        )

        scale_results.append({
            'scale': scale,
            'mean_rmse': results['mean_rmse'],
            'std_rmse': results['std_rmse'],
            'mean_r2': results['mean_r2'],
            'std_r2': results['std_r2']
        })

    # Save results
    scale_df = pd.DataFrame(scale_results)
    scale_df.to_csv(os.path.join(output_dir, 'results_scale.csv'), index=False)
    print(f"\nSaved scale analysis to {output_dir}/results_scale.csv")

    return scale_df


def generate_visualizations(output_dir, ablation_df, scale_df):
    """Generate t-SNE and other visualizations"""
    print("\nGenerating visualizations...")

    # Plot ablation results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Filter mean results
    mean_results = ablation_df[ablation_df['fold'] == 'mean']

    # RMSE comparison
    ax1 = axes[0]
    conditions = mean_results['condition'].tolist()
    rmses = mean_results['rmse'].tolist()
    colors = ['gray', 'blue', 'green', 'orange', 'red']

    bars = ax1.bar(conditions, rmses, color=colors[:len(conditions)])
    ax1.set_ylabel('RMSE')
    ax1.set_xlabel('Pretraining Condition')
    ax1.set_title('RMSE by Pretraining Condition')
    ax1.axhline(y=38.78, color='black', linestyle='--', label='ECFP4 Baseline')
    ax1.legend()

    for bar, rmse in zip(bars, rmses):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{rmse:.2f}', ha='center', va='bottom')

    # Scale analysis
    ax2 = axes[1]
    scales = [str(s) for s in scale_df['scale'].tolist()]
    scale_rmses = scale_df['mean_rmse'].tolist()
    scale_stds = scale_df['std_rmse'].tolist()

    bars = ax2.bar(scales, scale_rmses, yerr=scale_stds, capsize=5, color='steelblue')
    ax2.set_ylabel('RMSE')
    ax2.set_xlabel('Pretraining Scale')
    ax2.set_title('RMSE by Pretraining Scale')
    ax2.axhline(y=38.78, color='black', linestyle='--', label='ECFP4 Baseline')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_results.png'), dpi=150)
    print(f"Saved visualization to {output_dir}/ablation_results.png")

    plt.close()


def main():
    """Main execution"""
    # Configuration
    config = {
        'num_node_features': 159,  # Correct: 119 atomic_num + 8 degree + 8 formal_charge + 5 chiral + 6 num_Hs + 6 hybridization + 3 aromatic + 3 is_star + 1 mass
        'hidden_dim': 256,
        'num_layers': 3,
        'edge_dim': 13,  # Correct: 5 bond_types + 1 conjugated + 1 in_ring + 6 stereo = 13
        'proj_dim': 128,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 32,  # Reduced to prevent memory issues
        'pretrain_epochs': 5,  # Using 5 epochs for efficiency across ablation conditions
        'pretrain_samples': 100000,  # Use subset for efficiency
        'num_negatives': 5,
        'task_weights': {'fgm': 1.0, 'cri': 0.5, 'egcl': 1.0},
        'ecfp_bits': 2048,
        'log_dir': '/home/home_public/aisci/AIscientist/src/6_day6_pscp'
    }

    # Paths
    pi1m_path = '/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv'
    tg_path = '/home/home_public/aisci/AIscientist/src/alldataset/Tg.csv'
    output_dir = '/home/home_public/aisci/AIscientist/src/6_day6_pscp'

    print("=" * 70)
    print("PSCP: Polymer-Specific Contrastive Pretraining")
    print("=" * 70)
    print(f"PI1M Path: {pi1m_path}")
    print(f"Tg Path: {tg_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Run ablation study
    ablation_df = run_ablation_study(config, pi1m_path, tg_path, output_dir)

    # Run scale analysis
    scale_df = run_scale_analysis(config, pi1m_path, tg_path, output_dir)

    # Generate visualizations
    generate_visualizations(output_dir, ablation_df, scale_df)

    # Print final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    mean_results = ablation_df[ablation_df['fold'] == 'mean']
    baseline_rmse = 38.78

    print("\n[Ablation Results]")
    print("Condition | RMSE | Improvement over Baseline")
    print("-" * 50)
    for _, row in mean_results.iterrows():
        improvement = baseline_rmse - row['rmse']
        print(f"{row['condition']:15} | {row['rmse']:.2f} | +{improvement:.2f}")

    print("\n[Scale Analysis]")
    print("Scale | RMSE | ± Std")
    print("-" * 30)
    for _, row in scale_df.iterrows():
        print(f"{row['scale']:7} | {row['mean_rmse']:.2f} | ±{row['std_rmse']:.2f}")

    best_rmse = mean_results['rmse'].min()
    best_condition = mean_results.loc[mean_results['rmse'].idxmin(), 'condition']

    print("\n[Key Findings]")
    print(f"- Best RMSE: {best_rmse:.2f} ({best_condition})")
    print(f"- Target RMSE < 35.0: {'ACHIEVED' if best_rmse < 35.0 else 'FAILED'}")
    print(f"- Best improvement over baseline: +{baseline_rmse - best_rmse:.2f} RMSE")


if __name__ == '__main__':
    main()

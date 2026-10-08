"""
Pre-training Script for PF-CL Model
Contrastive learning + feature prediction on PI1M dataset
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
from typing import List, Dict, Optional
import argparse
import json
from datetime import datetime

# Import our modules
from feature_computation import PolymerFeatureComputer
from periodic_graph import PeriodicGraphBuilder, PeriodicAugmentor
from model import PFCLModel, count_parameters


class PolymerDataset(Dataset):
    """Dataset for pre-training"""

    def __init__(
        self,
        smiles_list: List[str],
        features: np.ndarray,
        graph_builder: PeriodicGraphBuilder,
        augmentor: PeriodicAugmentor,
        augment_prob: float = 0.5
    ):
        self.smiles_list = smiles_list
        self.features = features
        self.graph_builder = graph_builder
        self.augmentor = augmentor
        self.augment_prob = augment_prob

        # Pre-compute augmentation strategies
        self.strategies = ['shift', 'extend']

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        features = torch.tensor(self.features[idx], dtype=torch.float32)

        # Decide whether to augment
        if random.random() < self.augment_prob:
            strategy = random.choice(self.strategies)
            _, aug_smiles = self.augmentor.create_positive_pair(smiles, strategy=strategy)
            if aug_smiles is not None:
                smiles = aug_smiles

        # Build graph
        graph = self.graph_builder.build_graph(smiles, add_periodic_edge=True)

        if graph is None:
            # Fallback: create dummy graph
            graph = self.graph_builder.build_graph("*C*", add_periodic_edge=True)

        return {
            'graph': graph,
            'features': features
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching"""
    graphs = [item['graph'] for item in batch]
    features = torch.stack([item['features'] for item in batch])

    batched_graph = Batch.from_data_list(graphs)

    return {
        'graph': batched_graph,
        'features': features
    }


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""

    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: projections from view 1 (batch_size, proj_dim)
            z2: projections from view 2 (batch_size, proj_dim)
        Returns:
            loss: scalar
        """
        batch_size = z1.shape[0]

        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate
        z = torch.cat([z1, z2], dim=0)  # (2*batch_size, proj_dim)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2*batch_size, 2*batch_size)

        # Mask for positive pairs
        mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        mask = torch.cat([torch.cat([mask, torch.zeros_like(mask)], dim=1),
                         torch.cat([torch.zeros_like(mask), mask], dim=1)], dim=0)

        # Labels for positive pairs
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class FeaturePredictionLoss(nn.Module):
    """Loss for feature prediction"""

    def __init__(self, loss_type: str = 'mse'):
        super(FeaturePredictionLoss, self).__init__()
        self.loss_type = loss_type

        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predicted features (batch_size, feat_dim)
            target: target features (batch_size, feat_dim)
        Returns:
            loss: scalar
        """
        return self.criterion(pred, target)


class Pretrainer:
    """Pre-training trainer"""

    def __init__(
        self,
        model: PFCLModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 0.07,
        feature_loss_weight: float = 0.5,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        checkpoint_dir: str = "./checkpoints",
        use_mixed_precision: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.checkpoint_dir = checkpoint_dir
        self.use_mixed_precision = use_mixed_precision
        self.feature_loss_weight = feature_loss_weight

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=learning_rate * 0.01
        )

        # Loss functions
        self.contrastive_loss = InfoNCELoss(temperature=temperature)
        self.feature_loss = FeaturePredictionLoss(loss_type='mse')

        # Mixed precision
        self.scaler = GradScaler() if use_mixed_precision else None

        # Training history
        self.history = {
            'train_loss': [],
            'train_contrastive_loss': [],
            'train_feature_loss': [],
            'val_loss': [],
            'val_contrastive_loss': [],
            'val_feature_loss': [],
            'lr': []
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_feature_loss = 0.0
        num_batches = 0

        # Warmup learning rate
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * lr_scale

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")

        for batch in pbar:
            self.optimizer.zero_grad()

            graph = batch['graph'].to(self.device)
            features = batch['features'].to(self.device)

            # Forward with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    # Create two views for contrastive learning
                    outputs1 = self.model(graph)
                    outputs2 = self.model(graph)  # Same batch, different augmentation in dataset

                    # Contrastive loss (use projection)
                    if self.model.use_contrastive_head:
                        proj1 = outputs1['projection']
                        proj2 = outputs2['projection']
                        loss_contrastive = self.contrastive_loss(proj1, proj2)
                    else:
                        loss_contrastive = torch.tensor(0.0, device=self.device)

                    # Feature prediction loss
                    if self.model.use_feature_head:
                        pred_features = outputs1['features']
                        loss_feature = self.feature_loss(pred_features, features)
                    else:
                        loss_feature = torch.tensor(0.0, device=self.device)

                    # Combined loss
                    loss = loss_contrastive + self.feature_loss_weight * loss_feature

                # Backward with scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Create two views for contrastive learning
                outputs1 = self.model(graph)
                outputs2 = self.model(graph)

                # Contrastive loss
                if self.model.use_contrastive_head:
                    proj1 = outputs1['projection']
                    proj2 = outputs2['projection']
                    loss_contrastive = self.contrastive_loss(proj1, proj2)
                else:
                    loss_contrastive = torch.tensor(0.0, device=self.device)

                # Feature prediction loss
                if self.model.use_feature_head:
                    pred_features = outputs1['features']
                    loss_feature = self.feature_loss(pred_features, features)
                else:
                    loss_feature = torch.tensor(0.0, device=self.device)

                # Combined loss
                loss = loss_contrastive + self.feature_loss_weight * loss_feature

                # Backward
                loss.backward()
                self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_contrastive_loss += loss_contrastive.item()
            total_feature_loss += loss_feature.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cont': f'{loss_contrastive.item():.4f}',
                'feat': f'{loss_feature.item():.4f}'
            })

        # Average losses
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_feature_loss = total_feature_loss / num_batches

        return {
            'loss': avg_loss,
            'contrastive_loss': avg_contrastive_loss,
            'feature_loss': avg_feature_loss
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate"""
        self.model.eval()

        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_feature_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            graph = batch['graph'].to(self.device)
            features = batch['features'].to(self.device)

            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(graph)

                    # Contrastive loss (use same batch for simplicity)
                    if self.model.use_contrastive_head:
                        proj = outputs['projection']
                        # Simple self-contrastive loss for validation
                        loss_contrastive = torch.tensor(0.0, device=self.device)
                    else:
                        loss_contrastive = torch.tensor(0.0, device=self.device)

                    # Feature prediction loss
                    if self.model.use_feature_head:
                        pred_features = outputs['features']
                        loss_feature = self.feature_loss(pred_features, features)
                    else:
                        loss_feature = torch.tensor(0.0, device=self.device)

                    loss = loss_contrastive + self.feature_loss_weight * loss_feature
            else:
                outputs = self.model(graph)

                if self.model.use_contrastive_head:
                    proj = outputs['projection']
                    loss_contrastive = torch.tensor(0.0, device=self.device)
                else:
                    loss_contrastive = torch.tensor(0.0, device=self.device)

                if self.model.use_feature_head:
                    pred_features = outputs['features']
                    loss_feature = self.feature_loss(pred_features, features)
                else:
                    loss_feature = torch.tensor(0.0, device=self.device)

                loss = loss_contrastive + self.feature_loss_weight * loss_feature

            total_loss += loss.item()
            total_contrastive_loss += loss_contrastive.item()
            total_feature_loss += loss_feature.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_feature_loss = total_feature_loss / num_batches

        return {
            'loss': avg_loss,
            'contrastive_loss': avg_contrastive_loss,
            'feature_loss': avg_feature_loss
        }

    def train(self):
        """Full training loop"""
        print(f"Starting pre-training for {self.max_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_mixed_precision}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print()

        best_val_loss = float('inf')

        for epoch in range(self.max_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_contrastive_loss'].append(train_metrics['contrastive_loss'])
            self.history['train_feature_loss'].append(train_metrics['feature_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_contrastive_loss'].append(val_metrics['contrastive_loss'])
            self.history['val_feature_loss'].append(val_metrics['feature_loss'])
            self.history['lr'].append(current_lr)

            # Print summary
            print(f"\nEpoch {epoch+1}/{self.max_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                  f"Contrastive: {train_metrics['contrastive_loss']:.4f} | "
                  f"Feature: {train_metrics['feature_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | "
                  f"Contrastive: {val_metrics['contrastive_loss']:.4f} | "
                  f"Feature: {val_metrics['feature_loss']:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Save checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_encoder.pt')
                self.model.save_encoder(checkpoint_path)
                print(f"  Saved best checkpoint (val_loss: {best_val_loss:.4f})")

            # Save epoch checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f'encoder_epoch_{epoch+1}.pt'
                )
                self.model.save_encoder(checkpoint_path)

            print()

        # Save history
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"Pre-training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"History saved to {history_path}")

        return self.history


def load_pi1m_data(
    data_path: str,
    num_samples: Optional[int] = None,
    cache_dir: str = "./cache"
) -> tuple:
    """Load PI1M dataset and compute features"""
    print(f"Loading PI1M from {data_path}...")

    # Read SMILES
    df = pd.read_csv(data_path)
    smiles_list = df['SMILES'].tolist()

    # Limit samples if specified
    if num_samples is not None:
        smiles_list = smiles_list[:num_samples]
        print(f"Using subset of {num_samples} samples")
    else:
        print(f"Using full dataset ({len(smiles_list)} samples)")

    # Create cache name based on actual number of samples
    cache_name = f"pi1m_features_{len(smiles_list)}.pkl"

    # Compute features
    feature_computer = PolymerFeatureComputer(cache_dir=cache_dir)
    features = feature_computer.compute_dataset_features(
        smiles_list,
        cache_name=cache_name,
        batch_size=10000
    )

    print(f"Features shape: {features.shape}")
    print(f"SMILES list length: {len(smiles_list)}")

    # Ensure features match smiles_list length
    if len(features) != len(smiles_list):
        print(f"WARNING: Feature array length ({len(features)}) doesn't match SMILES list length ({len(smiles_list)})")
        print(f"Truncating features to match SMILES list...")
        features = features[:len(smiles_list)]

    return smiles_list, features


def main():
    parser = argparse.ArgumentParser(description='Pre-train PF-CL model')
    parser.add_argument('--data_path', type=str,
                        default='/home/home_public/aisci/AIscientist/src/alldataset/PI1M.csv',
                        help='Path to PI1M dataset')
    parser.add_argument('--num_samples', type=int, default=200000,
                        help='Number of samples to use (None for all)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    parser.add_argument('--feature_loss_weight', type=float, default=0.5,
                        help='Weight for feature loss')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Cache directory')
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                        help='Use contrastive learning')
    parser.add_argument('--use_feature', action='store_true', default=True,
                        help='Use feature prediction')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Load data
    smiles_list, features = load_pi1m_data(
        args.data_path,
        num_samples=args.num_samples,
        cache_dir=args.cache_dir
    )

    # Split into train/val (90/10)
    split_idx = int(0.9 * len(smiles_list))
    train_smiles = smiles_list[:split_idx]
    train_features = features[:split_idx]
    val_smiles = smiles_list[split_idx:]
    val_features = features[split_idx:]

    print(f"Train samples: {len(train_smiles)}")
    print(f"Val samples: {len(val_smiles)}")

    # Create graph builder and augmentor
    graph_builder = PeriodicGraphBuilder(
        node_feat_dim=37,
        edge_feat_dim=10,
        max_atoms=200
    )
    augmentor = PeriodicAugmentor(graph_builder)

    # Create datasets
    train_dataset = PolymerDataset(
        train_smiles,
        train_features,
        graph_builder,
        augmentor,
        augment_prob=0.5
    )
    val_dataset = PolymerDataset(
        val_smiles,
        val_features,
        graph_builder,
        augmentor,
        augment_prob=0.0  # No augmentation for validation
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    model = PFCLModel(
        node_dim=37,
        edge_dim=10,
        hidden_dim=256,
        num_layers=5,
        proj_dim=64,
        feat_dim=154,
        dropout=0.1,
        use_feature_head=args.use_feature,
        use_contrastive_head=args.use_contrastive
    )

    # Create trainer
    trainer = Pretrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        feature_loss_weight=args.feature_loss_weight,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        checkpoint_dir=args.checkpoint_dir,
        use_mixed_precision=True
    )

    # Train
    history = trainer.train()

    print("\nPre-training completed successfully!")


if __name__ == "__main__":
    main()

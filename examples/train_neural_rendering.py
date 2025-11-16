#!/usr/bin/env python3
"""Example script demonstrating neural rendering dataset usage.

This script shows how to:
1. Load the hierarchical voxel ray dataset
2. Apply data augmentation
3. Create data loaders with custom batching
4. Train a simple neural rendering model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from voxel_dataset_generator.datasets import (
    HierarchicalVoxelRayDataset,
    RayBatchSampler,
    collate_ray_batch,
    transforms,
)


class SimpleVoxelEncoder(nn.Module):
    """Simple 3D CNN encoder for voxel grids.

    This is a minimal example encoder that processes voxel grids
    into a latent representation.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            # Input: (batch, 1, D, H, W)
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # -> (batch, 32, D/2, H/2, W/2)

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # -> (batch, 64, D/4, H/4, W/4)

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),  # -> (batch, 128, 1, 1, 1)
        )

        self.fc = nn.Linear(128, latent_dim)

    def forward(self, voxels):
        """Encode voxel grid to latent vector.

        Args:
            voxels: (batch, 1, D, H, W) voxel grids

        Returns:
            (batch, latent_dim) latent vectors
        """
        features = self.encoder(voxels)
        features = features.view(features.size(0), -1)
        return self.fc(features)


class SimpleRayDecoder(nn.Module):
    """Simple MLP decoder for ray distance prediction.

    This decoder takes ray origin, direction, and voxel encoding
    and predicts the ray hit distance.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()

        # Process ray origin and direction
        self.ray_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 3 for origin + 3 for direction
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Combine with voxel latent
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),  # Predict distance
        )

    def forward(self, origins, directions, voxel_latents, ray_to_voxel):
        """Predict ray distances.

        Args:
            origins: (num_rays, 3) ray origins
            directions: (num_rays, 3) ray directions
            voxel_latents: (batch_size, latent_dim) voxel encodings
            ray_to_voxel: (num_rays,) mapping from rays to voxel indices

        Returns:
            (num_rays,) predicted distances
        """
        # Encode rays
        ray_input = torch.cat([origins, directions], dim=-1)
        ray_features = self.ray_encoder(ray_input)

        # Get corresponding voxel latents for each ray
        voxel_features = voxel_latents[ray_to_voxel]

        # Combine and decode
        combined = torch.cat([ray_features, voxel_features], dim=-1)
        distances = self.decoder(combined).squeeze(-1)

        return distances


class NeuralRenderingModel(nn.Module):
    """Complete neural rendering model combining encoder and decoder."""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.encoder = SimpleVoxelEncoder(latent_dim)
        self.decoder = SimpleRayDecoder(latent_dim, hidden_dim)

    def forward(self, batch):
        """Forward pass.

        Args:
            batch: Dictionary from collate_ray_batch containing:
                - voxels: (batch_size, 1, D, H, W)
                - origins: (num_rays, 3)
                - directions: (num_rays, 3)
                - ray_to_voxel: (num_rays,)

        Returns:
            (num_rays,) predicted distances
        """
        # Encode voxels
        voxel_latents = self.encoder(batch['voxels'])

        # Decode rays
        distances = self.decoder(
            batch['origins'],
            batch['directions'],
            voxel_latents,
            batch['ray_to_voxel']
        )

        return distances


def create_dataloaders(
    dataset_dir: Path,
    ray_dataset_dir: Path,
    batch_size: int = 8,
    rays_per_batch: int = 4096,
    num_workers: int = 4
):
    """Create train and validation dataloaders.

    Args:
        dataset_dir: Path to voxel dataset
        ray_dataset_dir: Path to ray dataset
        batch_size: Number of subvolumes per batch
        rays_per_batch: Total rays per batch
        num_workers: Number of worker processes

    Returns:
        train_loader, val_loader
    """
    # Define augmentation pipeline for training
    train_transform = transforms.Compose([
        transforms.RandomRotation90(p=0.5),
        transforms.RandomFlip(axes=[0, 1, 2], p=0.5),
        transforms.NormalizeRayOrigins(voxel_size=1.0),
        transforms.RandomRaySubsample(num_rays=rays_per_batch // batch_size),
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.NormalizeRayOrigins(voxel_size=1.0),
        transforms.RandomRaySubsample(num_rays=rays_per_batch // batch_size),
    ])

    # Create datasets
    train_dataset = HierarchicalVoxelRayDataset(
        dataset_dir=dataset_dir,
        ray_dataset_dir=ray_dataset_dir,
        split='train',
        levels=[3, 4, 5],  # Use mid-to-high resolution levels
        cache_size=100,
        transform=train_transform,
        seed=42
    )

    val_dataset = HierarchicalVoxelRayDataset(
        dataset_dir=dataset_dir,
        ray_dataset_dir=ray_dataset_dir,
        split='val',
        levels=[3, 4, 5],
        cache_size=50,
        transform=val_transform,
        seed=42
    )

    # Create dataloaders with custom batch sampler
    train_sampler = RayBatchSampler(
        train_dataset,
        rays_per_batch=rays_per_batch,
        subvolumes_per_batch=batch_size,
        shuffle=True,
    )

    val_sampler = RayBatchSampler(
        val_dataset,
        rays_per_batch=rays_per_batch,
        subvolumes_per_batch=batch_size,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_ray_batch,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_ray_batch,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch.

    Args:
        model: Neural rendering model
        train_loader: Training dataloader
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass
        pred_distances = model(batch)
        target_distances = batch['distances']

        # Compute loss (MSE for distance prediction)
        loss = nn.functional.mse_loss(pred_distances, target_distances)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate model.

    Args:
        model: Neural rendering model
        val_loader: Validation dataloader
        device: Device to validate on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Validation")
    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass
        pred_distances = model(batch)
        target_distances = batch['distances']

        # Compute loss
        loss = nn.functional.mse_loss(pred_distances, target_distances)

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


def main():
    """Main training loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Train neural rendering model")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"),
                       help="Path to voxel dataset")
    parser.add_argument("--ray-dataset-dir", type=Path, default=Path("ray_dataset_hierarchical"),
                       help="Path to ray dataset")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Number of subvolumes per batch")
    parser.add_argument("--rays-per-batch", type=int, default=4096,
                       help="Total rays per batch")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to train on")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of dataloader workers")

    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        dataset_dir=args.dataset_dir,
        ray_dataset_dir=args.ray_dataset_dir,
        batch_size=args.batch_size,
        rays_per_batch=args.rays_per_batch,
        num_workers=args.num_workers
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = NeuralRenderingModel(latent_dim=128, hidden_dim=256)
    model = model.to(args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        print(f"Train loss: {train_loss:.6f}")

        # Validate
        val_loss = validate(model, val_loader, args.device)
        print(f"Val loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
            print(f"Saved best model (val_loss: {val_loss:.6f})")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()

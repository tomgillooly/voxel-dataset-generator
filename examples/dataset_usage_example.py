#!/usr/bin/env python3
"""Simple example showing basic dataset usage.

This script demonstrates the basic functionality of the
HierarchicalVoxelRayDataset without requiring a full training setup.
"""

from pathlib import Path
from voxel_dataset_generator.datasets import (
    HierarchicalVoxelRayDataset,
    transforms,
)


def basic_usage_example():
    """Basic dataset loading and iteration."""
    print("="*60)
    print("Basic Dataset Usage Example")
    print("="*60)

    # Create dataset
    dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        cache_size=50,
    )

    print(f"\nDataset size: {len(dataset)} samples")
    print(f"Level distribution: {dataset.get_level_distribution()}")

    # Get a single sample
    sample = dataset[0]

    print(f"\nSample 0:")
    print(f"  Voxels shape: {sample['voxels'].shape}")
    print(f"  Origins shape: {sample['origins'].shape}")
    print(f"  Directions shape: {sample['directions'].shape}")
    print(f"  Distances shape: {sample['distances'].shape}")
    print(f"  Level: {sample['level']}")
    print(f"  Hash: {sample['hash'][:16]}...")

    # Check hit rate
    hit_rate = sample['hits'].float().mean().item()
    print(f"  Hit rate: {hit_rate:.2%}")

    # Check voxel occupancy
    occupancy = sample['voxels'].float().mean().item()
    print(f"  Voxel occupancy: {occupancy:.2%}")


def transform_example():
    """Example using data augmentation transforms."""
    print("\n" + "="*60)
    print("Transform Example")
    print("="*60)

    # Create transform pipeline
    transform = transforms.Compose([
        transforms.RandomRotation90(p=1.0),  # Always rotate for demo
        transforms.RandomFlip(axes=[0, 1, 2], p=0.5),
        transforms.NormalizeRayOrigins(voxel_size=1.0),
    ])

    # Create dataset with transforms
    dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        transform=transform,
        cache_size=10,
    )

    # Get transformed sample
    sample = dataset[0]

    print(f"\nTransformed sample:")
    print(f"  Origins range: [{sample['origins'].min():.2f}, {sample['origins'].max():.2f}]")
    print(f"  Directions normalized: {(sample['directions'].norm(dim=1).mean() - 1.0).abs() < 0.01}")


def level_filtering_example():
    """Example of filtering by hierarchy levels."""
    print("\n" + "="*60)
    print("Level Filtering Example")
    print("="*60)

    # Load only high-resolution levels
    high_res_dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        levels=[4, 5],  # Only finest levels
    )

    print(f"\nHigh-resolution dataset:")
    print(f"  Size: {len(high_res_dataset)} samples")
    print(f"  Levels: {high_res_dataset.get_level_distribution()}")

    # Load only low-resolution levels
    low_res_dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        levels=[1, 2, 3],  # Only coarse levels
    )

    print(f"\nLow-resolution dataset:")
    print(f"  Size: {len(low_res_dataset)} samples")
    print(f"  Levels: {low_res_dataset.get_level_distribution()}")


def split_comparison_example():
    """Compare train/val/test splits."""
    print("\n" + "="*60)
    print("Split Comparison Example")
    print("="*60)

    for split in ['train', 'val', 'test']:
        try:
            dataset = HierarchicalVoxelRayDataset(
                dataset_dir=Path("dataset"),
                ray_dataset_dir=Path("ray_dataset_hierarchical"),
                split=split,
            )
            print(f"\n{split.capitalize()} split:")
            print(f"  Size: {len(dataset)} samples")
            print(f"  Unique hashes: {len(set(s['hash'] for s in dataset.samples))}")
        except (FileNotFoundError, ValueError) as e:
            print(f"\n{split.capitalize()} split: Not available ({e})")


def memory_efficient_example():
    """Example of memory-efficient configuration."""
    print("\n" + "="*60)
    print("Memory-Efficient Configuration Example")
    print("="*60)

    # Configuration for limited memory
    dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        levels=[5],  # Only use finest level (smallest voxel grids)
        samples_per_subvolume=512,  # Limit rays per sample
        cache_size=10,  # Small cache
        include_empty=False,  # Skip empty subvolumes
    )

    print(f"\nMemory-efficient dataset:")
    print(f"  Size: {len(dataset)} samples")
    print(f"  Max rays per sample: 512")
    print(f"  Cache size: 10 voxel grids")

    # Check sample size
    sample = dataset[0]
    print(f"\nSample memory footprint (approximate):")
    voxel_mb = sample['voxels'].nbytes / 1024**2
    ray_mb = (sample['origins'].nbytes + sample['directions'].nbytes +
             sample['distances'].nbytes + sample['hits'].nbytes) / 1024**2
    print(f"  Voxels: {voxel_mb:.2f} MB")
    print(f"  Rays: {ray_mb:.2f} MB")
    print(f"  Total: {voxel_mb + ray_mb:.2f} MB")


def main():
    """Run all examples."""
    try:
        basic_usage_example()
    except Exception as e:
        print(f"Error in basic usage: {e}")

    try:
        transform_example()
    except Exception as e:
        print(f"Error in transform example: {e}")

    try:
        level_filtering_example()
    except Exception as e:
        print(f"Error in level filtering: {e}")

    try:
        split_comparison_example()
    except Exception as e:
        print(f"Error in split comparison: {e}")

    try:
        memory_efficient_example()
    except Exception as e:
        print(f"Error in memory-efficient example: {e}")

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()

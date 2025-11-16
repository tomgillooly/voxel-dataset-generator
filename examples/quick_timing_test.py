#!/usr/bin/env python3
"""Quick timing test for basic dataset operations.

This is a simpler, faster version of the benchmark script for quick checks.
"""

import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from voxel_dataset_generator.datasets import (
    HierarchicalVoxelRayDataset,
    RayBatchSampler,
    collate_ray_batch,
    transforms,
)


def main():
    print("Quick Dataset Timing Test")
    print("="*60)

    dataset_dir = Path("dataset")
    ray_dataset_dir = Path("ray_dataset_hierarchical")

    # 1. Dataset initialization
    print("\n1. Initializing dataset...")
    start = time.perf_counter()
    dataset = HierarchicalVoxelRayDataset(
        dataset_dir=dataset_dir,
        ray_dataset_dir=ray_dataset_dir,
        split='train',
        cache_size=50,
    )
    init_time = time.perf_counter() - start
    print(f"   Time: {init_time:.4f}s")
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Levels: {dataset.get_level_distribution()}")

    # 2. Single sample load
    print("\n2. Loading first sample...")
    start = time.perf_counter()
    sample = dataset[0]
    load_time = time.perf_counter() - start
    print(f"   Time: {load_time*1000:.2f}ms")
    print(f"   Voxels: {sample['voxels'].shape}")
    print(f"   Rays: {sample['origins'].shape[0]:,}")
    print(f"   Hit rate: {sample['hits'].float().mean():.2%}")

    # 3. Cached sample load
    print("\n3. Loading same sample (cached)...")
    start = time.perf_counter()
    sample = dataset[0]
    cached_load_time = time.perf_counter() - start
    print(f"   Time: {cached_load_time*1000:.2f}ms")
    print(f"   Speedup: {load_time / cached_load_time:.1f}x")

    # 4. Sequential iteration (10 samples)
    print("\n4. Sequential iteration (10 samples)...")
    start = time.perf_counter()
    for i in range(min(10, len(dataset))):
        _ = dataset[i]
    seq_time = time.perf_counter() - start
    print(f"   Total time: {seq_time:.4f}s")
    print(f"   Average per sample: {seq_time/10*1000:.2f}ms")

    # 5. DataLoader iteration (single epoch)
    print("\n5. DataLoader iteration (batch_size=4, workers=2)...")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    start = time.perf_counter()
    batch_count = 0
    total_samples = 0
    for batch in loader:
        batch_count += 1
        total_samples += len(batch['voxels'])
        if batch_count >= 20:  # Test first 20 batches
            break
    loader_time = time.perf_counter() - start

    print(f"   Processed {batch_count} batches ({total_samples} samples)")
    print(f"   Total time: {loader_time:.4f}s")
    print(f"   Throughput: {total_samples / loader_time:.1f} samples/sec")
    print(f"   Time per batch: {loader_time / batch_count * 1000:.2f}ms")

    # 6. Custom batch sampler
    print("\n6. Custom ray batch sampler (4096 rays, 8 subvolumes)...")
    sampler = RayBatchSampler(
        dataset,
        rays_per_batch=4096,
        subvolumes_per_batch=8,
        shuffle=True,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=2,
        collate_fn=collate_ray_batch,
        pin_memory=True,
    )

    start = time.perf_counter()
    batch_count = 0
    total_rays = 0
    for batch in loader:
        num_rays = len(batch['origins'])
        total_rays += num_rays
        batch_count += 1
        if batch_count >= 10:  # Test first 10 batches
            break
    custom_time = time.perf_counter() - start

    print(f"   Processed {batch_count} batches ({total_rays:,} rays)")
    print(f"   Total time: {custom_time:.4f}s")
    print(f"   Throughput: {total_rays / custom_time:,.0f} rays/sec")
    print(f"   Time per batch: {custom_time / batch_count * 1000:.2f}ms")

    # 7. With transforms
    print("\n7. Testing with transforms...")
    transform = transforms.Compose([
        transforms.RandomRotation90(p=0.5),
        transforms.RandomFlip(p=0.5),
        transforms.NormalizeRayOrigins(),
    ])

    dataset_transformed = HierarchicalVoxelRayDataset(
        dataset_dir=dataset_dir,
        ray_dataset_dir=ray_dataset_dir,
        split='train',
        transform=transform,
        cache_size=50,
    )

    start = time.perf_counter()
    for i in range(min(10, len(dataset_transformed))):
        _ = dataset_transformed[i]
    transform_time = time.perf_counter() - start

    print(f"   10 samples with transforms: {transform_time:.4f}s")
    print(f"   Average per sample: {transform_time/10*1000:.2f}ms")
    print(f"   Overhead vs no transform: {(transform_time - seq_time)/seq_time*100:.1f}%")

    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"  Dataset initialization:     {init_time:.4f}s")
    print(f"  Single sample load:         {load_time*1000:.2f}ms")
    print(f"  Cached sample load:         {cached_load_time*1000:.2f}ms")
    print(f"  DataLoader throughput:      {total_samples / loader_time:.1f} samples/sec")
    print(f"  Ray batch throughput:       {total_rays / custom_time:,.0f} rays/sec")
    print(f"  Transform overhead:         {(transform_time - seq_time)/seq_time*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()

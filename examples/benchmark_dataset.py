#!/usr/bin/env python3
"""Benchmark script for dataset loading and iteration performance.

This script measures:
- Single sample load time
- Batch iteration throughput
- Memory usage
- Cache effectiveness
- Impact of different configurations
"""

import time
import torch
import psutil
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, List
import argparse

from voxel_dataset_generator.datasets import (
    HierarchicalVoxelRayDataset,
    RayBatchSampler,
    collate_ray_batch,
    transforms,
)


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"{self.name}: {self.elapsed:.4f}s")


def benchmark_single_sample_load(dataset, num_samples: int = 100):
    """Benchmark loading individual samples.

    Args:
        dataset: Dataset instance
        num_samples: Number of samples to test
    """
    print("\n" + "="*60)
    print("Single Sample Load Benchmark")
    print("="*60)

    # Warm up
    _ = dataset[0]

    # Measure load times
    load_times = []
    indices = np.random.randint(0, len(dataset), size=num_samples)

    for idx in indices:
        start = time.perf_counter()
        sample = dataset[idx]
        elapsed = time.perf_counter() - start
        load_times.append(elapsed)

    load_times = np.array(load_times)

    print(f"\nLoaded {num_samples} samples:")
    print(f"  Mean time: {load_times.mean()*1000:.2f} ms")
    print(f"  Median time: {np.median(load_times)*1000:.2f} ms")
    print(f"  Std dev: {load_times.std()*1000:.2f} ms")
    print(f"  Min time: {load_times.min()*1000:.2f} ms")
    print(f"  Max time: {load_times.max()*1000:.2f} ms")
    print(f"  95th percentile: {np.percentile(load_times, 95)*1000:.2f} ms")


def benchmark_dataloader_iteration(
    dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    num_epochs: int = 3
):
    """Benchmark DataLoader iteration throughput.

    Args:
        dataset: Dataset instance
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        num_epochs: Number of epochs to benchmark
    """
    print("\n" + "="*60)
    print("DataLoader Iteration Benchmark")
    print("="*60)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_ray_batch,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batches per epoch: {len(loader)}")

    epoch_times = []
    total_samples = 0

    for epoch in range(num_epochs):
        start = time.perf_counter()
        batch_count = 0

        for batch in loader:
            batch_count += 1
            # Simulate minimal processing
            _ = batch['origins'].shape

        elapsed = time.perf_counter() - start
        epoch_times.append(elapsed)
        total_samples += len(dataset)

        samples_per_sec = len(dataset) / elapsed
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Time: {elapsed:.4f}s")
        print(f"  Samples/sec: {samples_per_sec:.1f}")
        print(f"  Batches processed: {batch_count}")

    epoch_times = np.array(epoch_times)
    print(f"\nOverall statistics ({num_epochs} epochs):")
    print(f"  Mean epoch time: {epoch_times.mean():.4f}s")
    print(f"  Std dev: {epoch_times.std():.4f}s")
    print(f"  Mean throughput: {len(dataset) / epoch_times.mean():.1f} samples/sec")


def benchmark_custom_batch_sampler(
    dataset,
    rays_per_batch: int = 4096,
    subvolumes_per_batch: int = 8,
    num_workers: int = 4,
    num_batches: int = 50
):
    """Benchmark custom ray batch sampler.

    Args:
        dataset: Dataset instance
        rays_per_batch: Total rays per batch
        subvolumes_per_batch: Number of subvolumes per batch
        num_workers: Number of worker processes
        num_batches: Number of batches to process
    """
    print("\n" + "="*60)
    print("Custom Ray Batch Sampler Benchmark")
    print("="*60)

    sampler = RayBatchSampler(
        dataset,
        rays_per_batch=rays_per_batch,
        subvolumes_per_batch=subvolumes_per_batch,
        shuffle=True,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_ray_batch,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print(f"\nConfiguration:")
    print(f"  Rays per batch: {rays_per_batch}")
    print(f"  Subvolumes per batch: {subvolumes_per_batch}")
    print(f"  Num workers: {num_workers}")

    batch_times = []
    ray_counts = []
    subvol_counts = []

    start = time.perf_counter()

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        batch_start = time.perf_counter()

        # Extract info
        num_rays = len(batch['origins'])
        num_subvols = len(batch['voxels'])

        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)
        ray_counts.append(num_rays)
        subvol_counts.append(num_subvols)

    total_time = time.perf_counter() - start
    batch_times = np.array(batch_times)
    ray_counts = np.array(ray_counts)

    print(f"\nProcessed {len(batch_times)} batches in {total_time:.4f}s:")
    print(f"  Mean batch time: {batch_times.mean()*1000:.2f} ms")
    print(f"  Throughput: {len(batch_times) / total_time:.1f} batches/sec")
    print(f"  Mean rays per batch: {ray_counts.mean():.0f}")
    print(f"  Mean subvolumes per batch: {np.mean(subvol_counts):.1f}")
    print(f"  Total rays processed: {ray_counts.sum():,}")
    print(f"  Ray throughput: {ray_counts.sum() / total_time:,.0f} rays/sec")


def benchmark_cache_effectiveness(dataset_dir, ray_dataset_dir, cache_sizes: List[int]):
    """Benchmark impact of different cache sizes.

    Args:
        dataset_dir: Path to voxel dataset
        ray_dataset_dir: Path to ray dataset
        cache_sizes: List of cache sizes to test
    """
    print("\n" + "="*60)
    print("Cache Effectiveness Benchmark")
    print("="*60)

    results = []

    for cache_size in cache_sizes:
        print(f"\nTesting cache_size={cache_size}...")

        # Create dataset
        dataset = HierarchicalVoxelRayDataset(
            dataset_dir=dataset_dir,
            ray_dataset_dir=ray_dataset_dir,
            split='train',
            cache_size=cache_size,
        )

        # Warm up
        _ = dataset[0]

        # Measure iteration time
        start = time.perf_counter()
        for i in range(min(100, len(dataset))):
            _ = dataset[i % len(dataset)]  # Access with wrapping
        elapsed = time.perf_counter() - start

        results.append({
            'cache_size': cache_size,
            'time': elapsed,
            'samples_per_sec': 100 / elapsed
        })

    print("\n" + "-"*60)
    print(f"{'Cache Size':<12} {'Time (s)':<12} {'Samples/sec':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['cache_size']:<12} {r['time']:<12.4f} {r['samples_per_sec']:<12.1f}")


def benchmark_transform_overhead(dataset_dir, ray_dataset_dir):
    """Benchmark overhead of different transforms.

    Args:
        dataset_dir: Path to voxel dataset
        ray_dataset_dir: Path to ray dataset
    """
    print("\n" + "="*60)
    print("Transform Overhead Benchmark")
    print("="*60)

    configs = [
        ("No transform", None),
        ("Normalize only", transforms.NormalizeRayOrigins()),
        ("Rotation", transforms.Compose([
            transforms.RandomRotation90(p=1.0),
            transforms.NormalizeRayOrigins(),
        ])),
        ("Rotation + Flip", transforms.Compose([
            transforms.RandomRotation90(p=1.0),
            transforms.RandomFlip(p=0.5),
            transforms.NormalizeRayOrigins(),
        ])),
        ("Full augmentation", transforms.Compose([
            transforms.RandomRotation90(p=0.5),
            transforms.RandomFlip(p=0.5),
            transforms.NormalizeRayOrigins(),
            transforms.RandomRaySubsample(num_rays=512),
        ])),
    ]

    num_samples = 50
    results = []

    for name, transform in configs:
        print(f"\nTesting: {name}")

        dataset = HierarchicalVoxelRayDataset(
            dataset_dir=dataset_dir,
            ray_dataset_dir=ray_dataset_dir,
            split='train',
            transform=transform,
            cache_size=50,
        )

        # Warm up
        _ = dataset[0]

        # Time iteration
        start = time.perf_counter()
        for i in range(num_samples):
            _ = dataset[i % len(dataset)]
        elapsed = time.perf_counter() - start

        results.append({
            'name': name,
            'time': elapsed,
            'samples_per_sec': num_samples / elapsed,
            'time_per_sample': elapsed / num_samples * 1000
        })

    print("\n" + "-"*60)
    print(f"{'Transform':<25} {'Time/sample (ms)':<20} {'Samples/sec':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<25} {r['time_per_sample']:<20.2f} {r['samples_per_sec']:<12.1f}")


def benchmark_memory_usage(dataset_dir, ray_dataset_dir):
    """Benchmark memory usage during iteration.

    Args:
        dataset_dir: Path to voxel dataset
        ray_dataset_dir: Path to ray dataset
    """
    print("\n" + "="*60)
    print("Memory Usage Benchmark")
    print("="*60)

    # Measure baseline
    baseline_mem = get_memory_usage()
    print(f"\nBaseline memory: {baseline_mem:.1f} MB")

    # Create dataset
    print("\nCreating dataset...")
    dataset = HierarchicalVoxelRayDataset(
        dataset_dir=dataset_dir,
        ray_dataset_dir=ray_dataset_dir,
        split='train',
        cache_size=50,
    )

    after_init_mem = get_memory_usage()
    print(f"After initialization: {after_init_mem:.1f} MB (+{after_init_mem - baseline_mem:.1f} MB)")

    # Load some samples
    print("\nLoading 10 samples...")
    for i in range(10):
        _ = dataset[i]

    after_load_mem = get_memory_usage()
    print(f"After 10 samples: {after_load_mem:.1f} MB (+{after_load_mem - after_init_mem:.1f} MB)")

    # Iterate through dataset
    print(f"\nIterating through {min(100, len(dataset))} samples...")
    for i in range(min(100, len(dataset))):
        _ = dataset[i]

    after_iter_mem = get_memory_usage()
    print(f"After iteration: {after_iter_mem:.1f} MB (+{after_iter_mem - after_load_mem:.1f} MB)")

    # Test with DataLoader
    print("\nTesting with DataLoader (4 workers)...")
    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=4,
        collate_fn=collate_ray_batch,
        pin_memory=True
    )

    for i, batch in enumerate(loader):
        if i >= 10:
            break
        _ = batch['origins'].shape

    after_loader_mem = get_memory_usage()
    print(f"After DataLoader: {after_loader_mem:.1f} MB (+{after_loader_mem - after_iter_mem:.1f} MB)")

    # Summary
    print("\n" + "-"*60)
    print("Memory breakdown:")
    print(f"  Dataset initialization: {after_init_mem - baseline_mem:.1f} MB")
    print(f"  Cache (first 10 samples): {after_load_mem - after_init_mem:.1f} MB")
    print(f"  Iteration overhead: {after_iter_mem - after_load_mem:.1f} MB")
    print(f"  DataLoader overhead: {after_loader_mem - after_iter_mem:.1f} MB")
    print(f"  Total: {after_loader_mem - baseline_mem:.1f} MB")


def benchmark_worker_scaling(dataset, max_workers: int = 8):
    """Benchmark throughput with different numbers of workers.

    Args:
        dataset: Dataset instance
        max_workers: Maximum number of workers to test
    """
    print("\n" + "="*60)
    print("Worker Scaling Benchmark")
    print("="*60)

    results = []

    for num_workers in range(0, max_workers + 1):
        print(f"\nTesting with {num_workers} workers...")

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_ray_batch,
            pin_memory=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        start = time.perf_counter()
        batch_count = 0

        for batch in loader:
            batch_count += 1
            if batch_count >= 50:  # Test first 50 batches
                break

        elapsed = time.perf_counter() - start
        throughput = batch_count / elapsed

        results.append({
            'workers': num_workers,
            'time': elapsed,
            'throughput': throughput
        })

    print("\n" + "-"*60)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Batches/sec':<12} {'Speedup':<10}")
    print("-"*60)

    baseline = results[0]['throughput']
    for r in results:
        speedup = r['throughput'] / baseline
        print(f"{r['workers']:<10} {r['time']:<12.4f} {r['throughput']:<12.1f} {speedup:<10.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark dataset performance")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"),
                       help="Path to voxel dataset")
    parser.add_argument("--ray-dataset-dir", type=Path, default=Path("ray_dataset_hierarchical"),
                       help="Path to ray dataset")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for DataLoader tests")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of DataLoader workers")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                       default=["all"],
                       choices=["all", "single", "loader", "batch", "cache",
                               "transform", "memory", "workers"],
                       help="Which benchmarks to run")

    args = parser.parse_args()

    print("="*60)
    print("Dataset Performance Benchmark")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dataset dir: {args.dataset_dir}")
    print(f"  Ray dataset dir: {args.ray_dataset_dir}")
    print(f"  Split: {args.split}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    # Create base dataset
    print("\nInitializing dataset...")
    with Timer("Dataset initialization"):
        dataset = HierarchicalVoxelRayDataset(
            dataset_dir=args.dataset_dir,
            ray_dataset_dir=args.ray_dataset_dir,
            split=args.split,
            cache_size=100,
        )

    print(f"  Dataset size: {len(dataset)} samples")
    print(f"  Level distribution: {dataset.get_level_distribution()}")

    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["single", "loader", "batch", "cache", "transform", "memory", "workers"]

    # Run benchmarks
    if "single" in benchmarks:
        benchmark_single_sample_load(dataset, num_samples=100)

    if "loader" in benchmarks:
        benchmark_dataloader_iteration(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_epochs=3
        )

    if "batch" in benchmarks:
        benchmark_custom_batch_sampler(
            dataset,
            rays_per_batch=4096,
            subvolumes_per_batch=8,
            num_workers=args.num_workers,
            num_batches=50
        )

    if "cache" in benchmarks:
        benchmark_cache_effectiveness(
            args.dataset_dir,
            args.ray_dataset_dir,
            cache_sizes=[0, 10, 50, 100, 200]
        )

    if "transform" in benchmarks:
        benchmark_transform_overhead(args.dataset_dir, args.ray_dataset_dir)

    if "memory" in benchmarks:
        benchmark_memory_usage(args.dataset_dir, args.ray_dataset_dir)

    if "workers" in benchmarks:
        benchmark_worker_scaling(dataset, max_workers=8)

    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

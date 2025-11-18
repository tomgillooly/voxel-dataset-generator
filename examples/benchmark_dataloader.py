#!/usr/bin/env python3
"""Benchmark script to profile dataloader performance and identify bottlenecks.

This script measures timing for various stages of data loading to help
identify performance issues.
"""

import time
import torch
import numpy as np
from pathlib import Path
from voxel_dataset_generator.datasets import (
    HierarchicalVoxelRayDataset,
    collate_ray_batch,
)
from torch.utils.data import DataLoader
import tqdm


def benchmark_dataset_init():
    """Benchmark dataset initialization time."""
    print("="*70)
    print("Benchmark: Dataset Initialization")
    print("="*70)

    configs = [
        {"name": "Dense, no chunking", "sparse_voxels": False, "rays_per_chunk": None},
        {"name": "Dense, chunked (4096)", "sparse_voxels": False, "rays_per_chunk": 4096},
        {"name": "Sparse COO, chunked (4096)", "sparse_voxels": True, "sparse_mode": "coo", "rays_per_chunk": 4096},
        {"name": "Sparse Graph, chunked (4096)", "sparse_voxels": True, "sparse_mode": "graph", "rays_per_chunk": 4096},
    ]

    for config in configs:
        name = config.pop("name")
        print(f"\n{name}:")

        start = time.time()
        dataset = HierarchicalVoxelRayDataset(
            dataset_dir=Path("dataset"),
            ray_dataset_dir=Path("ray_dataset_hierarchical"),
            split='train',
            levels=[0],
            cache_size=100,
            **config
        )
        init_time = time.time() - start

        print(f"  Init time: {init_time:.2f}s")
        print(f"  Dataset length: {len(dataset)} chunks")


def benchmark_single_sample_access():
    """Benchmark single sample access time."""
    print("\n" + "="*70)
    print("Benchmark: Single Sample Access")
    print("="*70)

    # Test different configurations
    configs = [
        {"name": "Dense voxels", "sparse_voxels": False, "sparse_mode": "coo"},
        {"name": "Sparse COO", "sparse_voxels": True, "sparse_mode": "coo"},
        {"name": "Sparse Graph", "sparse_voxels": True, "sparse_mode": "graph"},
    ]

    for config in configs:
        name = config.pop("name")
        print(f"\n{name}:")

        dataset = HierarchicalVoxelRayDataset(
            dataset_dir=Path("dataset"),
            ray_dataset_dir=Path("ray_dataset_hierarchical"),
            split='train',
            levels=[0],
            rays_per_chunk=4096,
            cache_size=100,
            **config
        )

        # Warm up - first access loads into cache
        _ = dataset[0]

        # Time cached access
        num_samples = min(10, len(dataset))
        start = time.time()
        for i in range(num_samples):
            sample = dataset[i]
        cached_time = (time.time() - start) / num_samples

        # Time uncached access (clear cache between each)
        dataset._voxel_cache.clear()
        dataset._cache_order.clear()

        start = time.time()
        for i in range(num_samples):
            dataset._voxel_cache.clear()
            dataset._cache_order.clear()
            sample = dataset[i]
        uncached_time = (time.time() - start) / num_samples

        print(f"  Cached access: {cached_time*1000:.2f}ms per sample")
        print(f"  Uncached access: {uncached_time*1000:.2f}ms per sample")
        print(f"  Sample shapes:")
        print(f"    origins: {sample['origins'].shape}")
        print(f"    directions: {sample['directions'].shape}")
        if 'voxels' in sample:
            print(f"    voxels: {sample['voxels'].shape}")
        if 'voxel_pos' in sample:
            print(f"    voxel_pos: {sample['voxel_pos'].shape}")
            print(f"    voxel_features: {sample['voxel_features'].shape}")


def benchmark_dataloader():
    """Benchmark DataLoader with different configurations."""
    print("\n" + "="*70)
    print("Benchmark: DataLoader Throughput")
    print("="*70)

    configs = [
        {
            "name": "Dense, batch=4, workers=0",
            "sparse_voxels": False,
            "batch_size": 4,
            "num_workers": 0,
        },
        {
            "name": "Dense, batch=4, workers=4",
            "sparse_voxels": False,
            "batch_size": 4,
            "num_workers": 4,
        },
        {
            "name": "Dense, batch=8, workers=4",
            "sparse_voxels": False,
            "batch_size": 8,
            "num_workers": 4,
        },
        {
            "name": "Sparse COO, batch=8, workers=4",
            "sparse_voxels": True,
            "sparse_mode": "coo",
            "batch_size": 8,
            "num_workers": 4,
        },
        {
            "name": "Sparse Graph, batch=8, workers=4",
            "sparse_voxels": True,
            "sparse_mode": "graph",
            "batch_size": 8,
            "num_workers": 4,
        },
    ]

    for config in configs:
        name = config.pop("name")
        batch_size = config.pop("batch_size")
        num_workers = config.pop("num_workers")

        print(f"\n{name}:")

        dataset = HierarchicalVoxelRayDataset(
            dataset_dir=Path("dataset"),
            ray_dataset_dir=Path("ray_dataset_hierarchical"),
            split='train',
            levels=[0],
            rays_per_chunk=4096,
            cache_size=100,
            **config
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_ray_batch,
            persistent_workers=num_workers > 0,
        )

        # Warm up
        for i, batch in enumerate(loader):
            if i >= 2:
                break

        # Benchmark
        num_batches = min(20, len(loader))
        start = time.time()

        for i, batch in enumerate(loader):
            if i >= num_batches:
                break

        total_time = time.time() - start
        time_per_batch = total_time / num_batches

        print(f"  Time per batch: {time_per_batch*1000:.2f}ms")
        print(f"  Batches per second: {1/time_per_batch:.2f}")
        print(f"  Total rays per batch: {batch['origins'].shape[0]}")
        print(f"  Throughput: {batch['origins'].shape[0] / time_per_batch:.0f} rays/sec")


def benchmark_bottlenecks():
    """Detailed breakdown of where time is spent during data loading."""
    print("\n" + "="*70)
    print("Benchmark: Detailed Bottleneck Analysis")
    print("="*70)

    dataset = HierarchicalVoxelRayDataset(
        dataset_dir=Path("dataset"),
        ray_dataset_dir=Path("ray_dataset_hierarchical"),
        split='train',
        levels=[0],
        rays_per_chunk=4096,
        cache_size=10,  # Small cache to force disk reads
        sparse_voxels=False,
    )

    num_samples = 50

    times = {
        'load_ray_data': [],
        'load_voxel_data': [],
        'extract_rays': [],
        'create_tensors': [],
        'total': [],
    }

    print(f"\nProfiling {num_samples} sample accesses...")

    for i in tqdm.tqdm(range(num_samples)):
        idx = i % len(dataset)

        overall_start = time.time()

        chunk_info = dataset.chunks[idx]
        sample_info = dataset.samples[chunk_info['subvolume_idx']]

        # Load ray data
        t0 = time.time()
        ray_data = np.load(sample_info['ray_path'], mmap_mode='r')
        t1 = time.time()
        times['load_ray_data'].append(t1 - t0)

        # Load voxel data
        t0 = time.time()
        voxels = dataset._load_voxels(sample_info['voxel_path'])
        t1 = time.time()
        times['load_voxel_data'].append(t1 - t0)

        # Extract ray components
        t0 = time.time()
        start = chunk_info['chunk_start']
        end = chunk_info['chunk_end']
        origins = ray_data['origins'][start:end]
        directions = ray_data['directions'][start:end]
        distances = ray_data['distances'][start:end]
        hits = ray_data['hits'][start:end]
        t1 = time.time()
        times['extract_rays'].append(t1 - t0)

        # Create tensors
        t0 = time.time()
        cube_size = 2.0**(7-sample_info['level'])
        cube_diag = np.sqrt(3*cube_size**2)
        voxels_tensor = torch.from_numpy(voxels.astype(np.float32)).unsqueeze(0)
        sample = {
            'origins': torch.from_numpy(origins.astype(np.float32)),
            'directions': torch.from_numpy(directions.astype(np.float32)),
            'distances': torch.from_numpy(distances.astype(np.float32)) / cube_diag,
            'hits': torch.from_numpy(hits.astype(np.float32)),
            'voxels': voxels_tensor,
            'level': sample_info['level'],
            'hash': sample_info['hash'],
            'chunk_idx': start // (dataset.rays_per_chunk if dataset.rays_per_chunk else 1),
        }
        t1 = time.time()
        times['create_tensors'].append(t1 - t0)

        times['total'].append(time.time() - overall_start)

    # Print results
    print("\nTiming breakdown (average over all samples):")
    for key, values in times.items():
        avg_ms = np.mean(values) * 1000
        std_ms = np.std(values) * 1000
        min_ms = np.min(values) * 1000
        max_ms = np.max(values) * 1000
        pct = (np.mean(values) / np.mean(times['total'])) * 100 if key != 'total' else 100

        print(f"  {key:20s}: {avg_ms:7.2f}ms ± {std_ms:6.2f}ms  "
              f"[min: {min_ms:6.2f}ms, max: {max_ms:6.2f}ms]  ({pct:5.1f}%)")


def benchmark_collate_function():
    """Benchmark the collate function separately."""
    print("\n" + "="*70)
    print("Benchmark: Collate Function")
    print("="*70)

    configs = [
        {"name": "Dense voxels", "sparse_voxels": False},
        {"name": "Sparse COO", "sparse_voxels": True, "sparse_mode": "coo"},
        {"name": "Sparse Graph", "sparse_voxels": True, "sparse_mode": "graph"},
    ]

    for config in configs:
        name = config.pop("name")
        print(f"\n{name}:")

        dataset = HierarchicalVoxelRayDataset(
            dataset_dir=Path("dataset"),
            ray_dataset_dir=Path("ray_dataset_hierarchical"),
            split='train',
            levels=[0],
            rays_per_chunk=4096,
            cache_size=100,
            **config
        )

        # Create batch of samples
        batch_sizes = [4, 8, 16]
        for batch_size in batch_sizes:
            batch = [dataset[i] for i in range(min(batch_size, len(dataset)))]

            # Warm up
            _ = collate_ray_batch(batch)

            # Benchmark
            num_runs = 100
            start = time.time()
            for _ in range(num_runs):
                batched = collate_ray_batch(batch)
            avg_time = (time.time() - start) / num_runs

            print(f"  Batch size {batch_size}: {avg_time*1000:.2f}ms per collate")


def benchmark_chunk_building():
    """Benchmark chunk index building time."""
    print("\n" + "="*70)
    print("Benchmark: Chunk Index Building")
    print("="*70)

    chunk_sizes = [None, 1024, 2048, 4096, 8192]

    for rays_per_chunk in chunk_sizes:
        print(f"\nrays_per_chunk={rays_per_chunk}:")

        # Manually time just the chunk building
        from voxel_dataset_generator.datasets.neural_rendering_dataset import HierarchicalVoxelRayDataset

        dataset = HierarchicalVoxelRayDataset.__new__(HierarchicalVoxelRayDataset)
        dataset.dataset_dir = Path("dataset")
        dataset.ray_dataset_dir = Path("ray_dataset_hierarchical")
        dataset.split = 'train'
        dataset.rays_per_chunk = rays_per_chunk
        dataset.cache_size = 100
        dataset.include_empty = False
        dataset.transform = None
        dataset.rng = np.random.RandomState(42)
        dataset.sparse_voxels = False
        dataset.sparse_mode = 'coo'
        dataset.sparse_connectivity = 6

        dataset._load_metadata()
        dataset._load_splits()
        dataset._collect_samples([0])

        start = time.time()
        dataset._build_chunk_index()
        build_time = time.time() - start

        print(f"  Build time: {build_time:.2f}s")
        print(f"  Number of chunks: {len(dataset.chunks)}")
        print(f"  Number of subvolumes: {len(dataset.samples)}")


def main():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("DATALOADER PERFORMANCE BENCHMARK")
    print("="*70)

    try:
        benchmark_chunk_building()
    except Exception as e:
        print(f"Error in chunk building benchmark: {e}")

    try:
        benchmark_dataset_init()
    except Exception as e:
        print(f"Error in dataset init benchmark: {e}")

    try:
        benchmark_single_sample_access()
    except Exception as e:
        print(f"Error in single sample access benchmark: {e}")

    try:
        benchmark_bottlenecks()
    except Exception as e:
        print(f"Error in bottleneck analysis: {e}")

    try:
        benchmark_collate_function()
    except Exception as e:
        print(f"Error in collate function benchmark: {e}")

    try:
        benchmark_dataloader()
    except Exception as e:
        print(f"Error in dataloader benchmark: {e}")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print("\nKey findings:")
    print("  - Check 'Bottleneck Analysis' to see where most time is spent")
    print("  - Compare cached vs uncached access times")
    print("  - Compare num_workers=0 vs num_workers=4 for multiprocessing benefit")
    print("  - Check if chunk building is the main initialization bottleneck")


if __name__ == "__main__":
    main()

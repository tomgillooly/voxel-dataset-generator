# Neural Rendering Dataset

PyTorch dataset classes for loading hierarchical voxel ray-tracing data for neural rendering applications.

## Overview

This dataset infrastructure is designed for training neural rendering models (e.g., Neural Radiance Fields, voxel-based rendering networks) using hierarchical voxelized 3D objects with pre-computed ray-tracing data.

### Key Features

- **Hierarchical multi-resolution support**: Load data from different octree levels
- **Train/val/test splits**: Automatic split handling with proper data isolation
- **Memory-efficient loading**: Configurable caching and lazy loading
- **Data augmentation**: Geometric transformations (rotations, flips) that preserve ray-voxel consistency
- **Flexible batching**: Custom samplers for efficient ray batching across multiple subvolumes
- **PyTorch integration**: Full compatibility with PyTorch DataLoader

## Data Format

### Dataset Structure

```
dataset/
├── metadata.json              # Dataset-level metadata
├── splits.json                # Train/val/test split assignments
├── objects/                   # Per-object data
│   └── object_0000/
│       ├── metadata.json      # Object metadata
│       ├── subdivision_map.json  # Octree subdivision structure
│       └── level_0.npz        # Top-level voxel grid
└── subvolumes/                # Deduplicated subvolumes
    └── level_1/
        └── f3/                # Hash prefix for organization
            └── f30c11...npz   # Voxel data (hash-based filename)

ray_dataset_hierarchical/
├── processing_summary.json    # Ray generation metadata
└── level_1/                   # Ray data organized by level
    └── f3/
        └── f30c11..._rays.npz  # Ray data for subvolume
```

### Sample Data Format

Each sample from the dataset contains:

```python
{
    'voxels': torch.Tensor,      # (1, D, H, W) boolean voxel grid
    'origins': torch.Tensor,     # (N, 3) ray origin points
    'directions': torch.Tensor,  # (N, 3) ray directions (normalized)
    'distances': torch.Tensor,   # (N,) ray hit distances (0 = miss)
    'hits': torch.Tensor,        # (N,) binary hit flags
    'level': int,                # Octree hierarchy level
    'hash': str,                 # Subvolume hash identifier
    'view_ids': torch.Tensor,    # (N,) viewpoint indices (optional)
    'face_ids': torch.Tensor,    # (N,) bounding box face IDs (optional)
    'view_positions': torch.Tensor,  # (num_views, 3) camera positions (optional)
}
```

## Basic Usage

### Simple Dataset Loading

```python
from pathlib import Path
from voxel_dataset_generator.datasets import HierarchicalVoxelRayDataset

# Create dataset
dataset = HierarchicalVoxelRayDataset(
    dataset_dir=Path("dataset"),
    ray_dataset_dir=Path("ray_dataset_hierarchical"),
    split='train',
    cache_size=100,
)

# Get a sample
sample = dataset[0]
print(f"Voxels: {sample['voxels'].shape}")
print(f"Rays: {sample['origins'].shape}")
print(f"Hit rate: {sample['hits'].float().mean():.2%}")
```

### With Data Augmentation

```python
from voxel_dataset_generator.datasets import transforms

# Create augmentation pipeline
transform = transforms.Compose([
    transforms.RandomRotation90(p=0.5),
    transforms.RandomFlip(axes=[0, 1, 2], p=0.5),
    transforms.NormalizeRayOrigins(voxel_size=1.0),
    transforms.RandomRaySubsample(num_rays=1024),
])

# Create dataset with transforms
dataset = HierarchicalVoxelRayDataset(
    dataset_dir=Path("dataset"),
    ray_dataset_dir=Path("ray_dataset_hierarchical"),
    split='train',
    transform=transform,
)
```

### PyTorch DataLoader Integration

```python
from torch.utils.data import DataLoader
from voxel_dataset_generator.datasets import (
    HierarchicalVoxelRayDataset,
    RayBatchSampler,
    collate_ray_batch,
)

# Create dataset
dataset = HierarchicalVoxelRayDataset(
    dataset_dir=Path("dataset"),
    ray_dataset_dir=Path("ray_dataset_hierarchical"),
    split='train',
)

# Create custom batch sampler for efficient ray batching
sampler = RayBatchSampler(
    dataset,
    rays_per_batch=4096,        # Total rays per batch
    subvolumes_per_batch=8,     # Number of subvolumes per batch
    shuffle=True,
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=collate_ray_batch,
    num_workers=4,
    pin_memory=True,
)

# Iterate
for batch in loader:
    voxels = batch['voxels']           # (8, 1, D, H, W)
    origins = batch['origins']         # (4096, 3)
    ray_to_voxel = batch['ray_to_voxel']  # (4096,) - maps rays to voxel indices
    # ... train model ...
```

## Advanced Usage

### Level-Specific Loading

Load only specific octree levels:

```python
# Load only high-resolution levels
dataset = HierarchicalVoxelRayDataset(
    dataset_dir=Path("dataset"),
    ray_dataset_dir=Path("ray_dataset_hierarchical"),
    split='train',
    levels=[4, 5],  # Only levels 4 and 5
)
```

### Memory-Efficient Configuration

For limited memory scenarios:

```python
dataset = HierarchicalVoxelRayDataset(
    dataset_dir=Path("dataset"),
    ray_dataset_dir=Path("ray_dataset_hierarchical"),
    split='train',
    levels=[5],                      # Use only finest level (smaller grids)
    samples_per_subvolume=512,       # Limit rays per sample
    cache_size=10,                   # Small voxel cache
    include_empty=False,             # Skip empty subvolumes
)
```

### Custom Batching Strategy

The `collate_ray_batch` function concatenates all rays from multiple subvolumes:

```python
batch = collate_ray_batch([sample1, sample2, sample3])

# Result structure:
# - All rays concatenated: batch['origins'] is (N1+N2+N3, 3)
# - Voxels stacked: batch['voxels'] is (3, 1, D, H, W)
# - Mapping tensor: batch['ray_to_voxel'][i] tells which voxel ray i belongs to
```

This allows models to process rays from multiple scenes efficiently.

## Available Transforms

### Geometric Transforms

```python
transforms.RandomRotation90(p=0.5)           # Random 90° rotations
transforms.RandomFlip(axes=[0,1,2], p=0.5)   # Random axis flips
```

### Ray Transforms

```python
transforms.NormalizeRayOrigins(voxel_size=1.0)  # Normalize to [-1, 1]
transforms.RandomRaySubsample(num_rays=1024)    # Subsample rays
transforms.AddNoise(origin_std=0.01,            # Add regularization noise
                   direction_std=0.001)
```

### Voxel Transforms

```python
transforms.VoxelOccupancyJitter(flip_prob=0.01)  # Random voxel flipping
```

### Utility Transforms

```python
transforms.ToDevice('cuda')    # Move to GPU
transforms.Compose([...])      # Chain transforms
```

## Dataset Statistics

Get insights into your dataset:

```python
# Level distribution
distribution = dataset.get_level_distribution()
print(distribution)  # {1: 8, 2: 13, 3: 51, ...}

# Unique subvolumes
num_unique = len(set(s['hash'] for s in dataset.samples))

# Average occupancy
occupancies = [sample['voxels'].float().mean().item()
               for sample in dataset]
avg_occupancy = sum(occupancies) / len(occupancies)
```

## Example Training Loop

See [examples/train_neural_rendering.py](examples/train_neural_rendering.py) for a complete example including:

- Simple voxel encoder (3D CNN)
- Ray decoder (MLP)
- Training and validation loops
- Model checkpointing

Run with:

```bash
uv run python examples/train_neural_rendering.py \
    --dataset-dir dataset \
    --ray-dataset-dir ray_dataset_hierarchical \
    --batch-size 8 \
    --rays-per-batch 4096 \
    --epochs 10
```

## Performance Tips

### Memory Management

1. **Adjust cache size**: Larger cache = faster loading, more memory
   ```python
   dataset = HierarchicalVoxelRayDataset(..., cache_size=100)
   ```

2. **Subsample rays**: Reduce memory per sample
   ```python
   transform = transforms.RandomRaySubsample(num_rays=512)
   ```

3. **Filter by level**: Smaller grids at higher levels
   ```python
   dataset = HierarchicalVoxelRayDataset(..., levels=[5])
   ```

### Training Speed

1. **Use multiple workers**: Parallel data loading
   ```python
   loader = DataLoader(dataset, num_workers=4, pin_memory=True)
   ```

2. **Pin memory**: Faster CPU-to-GPU transfer
   ```python
   loader = DataLoader(..., pin_memory=True)
   ```

3. **Prefetch**: Ensure workers stay busy
   ```python
   loader = DataLoader(..., prefetch_factor=2)
   ```

### Batch Size Tuning

Balance between:
- **rays_per_batch**: Total computational load
- **subvolumes_per_batch**: Diversity of training signal

Example configurations:

```python
# High diversity, moderate batch size
sampler = RayBatchSampler(dataset,
    rays_per_batch=4096,
    subvolumes_per_batch=16)

# Lower diversity, larger batch per subvolume
sampler = RayBatchSampler(dataset,
    rays_per_batch=4096,
    subvolumes_per_batch=4)
```

## Troubleshooting

### "No samples found for split"

- Ensure ray dataset exists and matches voxel dataset
- Check that split assignments exist in `splits.json`
- Verify level filtering isn't too restrictive

### High memory usage

- Reduce `cache_size`
- Use `samples_per_subvolume` to limit rays
- Filter to higher levels (smaller voxel grids)
- Reduce `num_workers` in DataLoader

### Slow loading

- Increase `cache_size` if memory permits
- Increase `num_workers` in DataLoader
- Use SSD storage for dataset
- Consider prefetch_factor in DataLoader

## API Reference

### HierarchicalVoxelRayDataset

```python
HierarchicalVoxelRayDataset(
    dataset_dir: Path,
    ray_dataset_dir: Path,
    split: Literal['train', 'val', 'test'] = 'train',
    levels: Optional[List[int]] = None,
    samples_per_subvolume: Optional[int] = None,
    cache_size: int = 100,
    include_empty: bool = False,
    transform: Optional[callable] = None,
    seed: int = 42
)
```

### RayBatchSampler

```python
RayBatchSampler(
    dataset: HierarchicalVoxelRayDataset,
    rays_per_batch: int = 4096,
    subvolumes_per_batch: int = 8,
    shuffle: bool = True,
    drop_last: bool = False
)
```

### collate_ray_batch

```python
collate_ray_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]
```

Combines multiple samples into a single batch with:
- Concatenated ray tensors
- Stacked voxel grids
- Ray-to-voxel mapping tensor

## Performance Benchmarking

Two scripts are provided for performance testing:

### Quick Timing Test

Fast basic performance check:

```bash
uv run python examples/quick_timing_test.py
```

This runs:
- Dataset initialization timing
- Single sample load time
- Cache effectiveness
- DataLoader throughput
- Custom batch sampler throughput
- Transform overhead

### Comprehensive Benchmark

Full performance analysis:

```bash
# Run all benchmarks
uv run python examples/benchmark_dataset.py

# Run specific benchmarks
uv run python examples/benchmark_dataset.py \
    --benchmarks single loader cache

# Custom configuration
uv run python examples/benchmark_dataset.py \
    --batch-size 16 \
    --num-workers 8 \
    --benchmarks all
```

Available benchmarks:
- `single`: Single sample load times
- `loader`: DataLoader iteration throughput
- `batch`: Custom ray batch sampler performance
- `cache`: Cache size effectiveness
- `transform`: Transform overhead comparison
- `memory`: Memory usage profiling
- `workers`: Worker scaling analysis

## Examples

- **[quick_timing_test.py](examples/quick_timing_test.py)**: Fast performance check
- **[benchmark_dataset.py](examples/benchmark_dataset.py)**: Comprehensive benchmarking suite
- **[dataset_usage_example.py](examples/dataset_usage_example.py)**: Basic dataset operations
- **[train_neural_rendering.py](examples/train_neural_rendering.py)**: Complete training example

## License

See main project LICENSE file.

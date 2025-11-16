# Neural Rendering Dataset - Quick Reference

## Installation

```bash
uv pip install -e ".[neural-rendering]"
```

## Quick Test

```bash
uv run python examples/quick_timing_test.py
```

## Basic Usage

```python
from pathlib import Path
from voxel_dataset_generator.datasets import HierarchicalVoxelRayDataset

# Load dataset
dataset = HierarchicalVoxelRayDataset(
    dataset_dir=Path("dataset"),
    ray_dataset_dir=Path("ray_dataset_hierarchical"),
    split='train',
)

# Get sample
sample = dataset[0]
print(sample.keys())  # voxels, origins, directions, distances, hits, level, hash
```

## Common Patterns

### Standard DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

for batch in loader:
    voxels = batch['voxels']      # (8, 1, D, H, W)
    origins = batch['origins']    # (8, N, 3)
    distances = batch['distances'] # (8, N)
```

### Custom Ray Batching

```python
from voxel_dataset_generator.datasets import RayBatchSampler, collate_ray_batch

sampler = RayBatchSampler(dataset, rays_per_batch=4096, subvolumes_per_batch=8)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_ray_batch)

for batch in loader:
    origins = batch['origins']         # (4096, 3) all rays
    voxels = batch['voxels']           # (8, 1, D, H, W)
    ray_to_voxel = batch['ray_to_voxel']  # (4096,) mapping
```

### With Augmentation

```python
from voxel_dataset_generator.datasets import transforms

transform = transforms.Compose([
    transforms.RandomRotation90(p=0.5),
    transforms.RandomFlip(p=0.5),
    transforms.NormalizeRayOrigins(),
])

dataset = HierarchicalVoxelRayDataset(..., transform=transform)
```

### Level Filtering

```python
# Only high-resolution levels
dataset = HierarchicalVoxelRayDataset(..., levels=[4, 5])

# Only low-resolution
dataset = HierarchicalVoxelRayDataset(..., levels=[1, 2, 3])
```

### Memory-Efficient

```python
dataset = HierarchicalVoxelRayDataset(
    ...,
    levels=[5],                    # Smaller grids
    samples_per_subvolume=512,     # Fewer rays
    cache_size=10,                 # Small cache
    include_empty=False,           # Skip empty
)
```

## Available Transforms

```python
from voxel_dataset_generator.datasets import transforms

transforms.RandomRotation90(p=0.5)              # 90° rotations
transforms.RandomFlip(axes=[0,1,2], p=0.5)      # Random flips
transforms.NormalizeRayOrigins(voxel_size=1.0)  # Normalize coordinates
transforms.RandomRaySubsample(num_rays=1024)    # Subsample rays
transforms.AddNoise(origin_std=0.01)            # Add noise
transforms.VoxelOccupancyJitter(flip_prob=0.01) # Voxel noise
transforms.ToDevice('cuda')                      # Move to GPU
transforms.Compose([...])                        # Chain transforms
```

## Sample Structure

```python
sample = {
    'voxels': torch.Tensor,       # (1, D, H, W) boolean grid
    'origins': torch.Tensor,      # (N, 3) ray origins
    'directions': torch.Tensor,   # (N, 3) ray directions
    'distances': torch.Tensor,    # (N,) hit distances
    'hits': torch.Tensor,         # (N,) binary hit flags
    'level': int,                 # hierarchy level
    'hash': str,                  # subvolume hash
    'view_ids': torch.Tensor,     # (N,) optional
    'face_ids': torch.Tensor,     # (N,) optional
}
```

## Batch Structure (collate_ray_batch)

```python
batch = {
    'origins': torch.Tensor,      # (total_rays, 3)
    'directions': torch.Tensor,   # (total_rays, 3)
    'distances': torch.Tensor,    # (total_rays,)
    'hits': torch.Tensor,         # (total_rays,)
    'voxels': torch.Tensor,       # (batch_size, 1, D, H, W)
    'ray_to_voxel': torch.Tensor, # (total_rays,) mapping
    'levels': torch.Tensor,       # (batch_size,)
    'hashes': List[str],          # [batch_size]
}
```

## Performance Tips

```python
# Optimal workers (usually num CPU cores)
loader = DataLoader(..., num_workers=8, pin_memory=True)

# Larger cache (if RAM available)
dataset = HierarchicalVoxelRayDataset(..., cache_size=200)

# Prefetch for faster iteration
loader = DataLoader(..., prefetch_factor=2)
```

## Troubleshooting Commands

```bash
# Test performance
uv run python examples/quick_timing_test.py

# Full benchmarks
uv run python examples/benchmark_dataset.py

# Check data
ls -la dataset/
ls -la ray_dataset_hierarchical/
cat dataset/metadata.json

# Regenerate splits
python -c "from pathlib import Path; from voxel_dataset_generator.pipeline import reassign_splits; reassign_splits(Path('dataset'))"
```

## Example Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from voxel_dataset_generator.datasets import HierarchicalVoxelRayDataset

# Setup
dataset = HierarchicalVoxelRayDataset(...)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

# Train
for epoch in range(num_epochs):
    for batch in loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        pred = model(batch['voxels'], batch['origins'], batch['directions'])
        loss = nn.functional.mse_loss(pred, batch['distances'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Typical Configurations

**High-end (32 cores, 64GB RAM):**
```python
dataset = HierarchicalVoxelRayDataset(..., cache_size=200)
loader = DataLoader(dataset, batch_size=16, num_workers=16, pin_memory=True)
```

**Mid-range (8 cores, 16GB RAM):**
```python
dataset = HierarchicalVoxelRayDataset(..., cache_size=50)
loader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True)
```

**Low-memory (4 cores, 8GB RAM):**
```python
dataset = HierarchicalVoxelRayDataset(..., cache_size=10, levels=[5])
loader = DataLoader(dataset, batch_size=4, num_workers=2)
```

## Files

- `quick_timing_test.py` - Quick performance test
- `benchmark_dataset.py` - Comprehensive benchmarks
- `dataset_usage_example.py` - Usage examples
- `train_neural_rendering.py` - Training example
- `NEURAL_RENDERING_DATASET.md` - Full documentation
- `SETUP_NEURAL_RENDERING.md` - Setup guide

# Neural Rendering Dataset Setup Guide

This guide walks you through setting up and using the PyTorch dataset for neural rendering with your hierarchical voxel data.

## Installation

Install the neural rendering dependencies:

```bash
uv pip install -e ".[neural-rendering]"
```

This installs:
- `torch>=2.0.0` - PyTorch for neural networks
- `psutil>=5.9.0` - For memory profiling in benchmarks

## Prerequisites

Before using the neural rendering dataset, you need:

1. **Hierarchical voxel dataset** - Generated using the main pipeline
2. **Ray-traced data** - Generated using the OptiX ray tracer

### Generating the Required Data

#### Step 1: Generate Hierarchical Voxel Dataset

```bash
# Using the main pipeline
uv run python -m voxel_dataset_generator.pipeline \
    --num-objects 100 \
    --output-dir dataset \
    --resolution 128 \
    --min-resolution 4
```

Or for Thingi10k:

```bash
# See pipeline.py for generate_dataset_from_thingi10k
uv run python examples/generate_thingi10k_dataset.py
```

#### Step 2: Generate Ray-Traced Data

```bash
# Build OptiX ray tracer first
cd optix_raytracer
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../..

# Generate ray data for all subvolumes
uv run python optix_raytracer/examples/generate_hierarchical_dataset.py \
    --dataset-dir dataset \
    --output-dir ray_dataset_hierarchical \
    --sphere-divisions 4 \
    --samples-per-view 3000 \
    --adaptive-sampling
```

This creates ray data organized like:
```
ray_dataset_hierarchical/
├── processing_summary.json
└── level_1/
    └── f3/
        └── f30c11..._rays.npz
```

## Quick Start

### 1. Verify Setup

Run the quick timing test to ensure everything is working:

```bash
uv run python examples/quick_timing_test.py
```

Expected output:
```
Quick Dataset Timing Test
============================================================

1. Initializing dataset...
   Time: 0.1234s
   Dataset size: 311 samples
   ...
```

If you see errors, check:
- Dataset directory exists: `ls -la dataset/`
- Ray dataset exists: `ls -la ray_dataset_hierarchical/`
- Splits file exists: `ls -la dataset/splits.json`

### 2. Explore the Dataset

```bash
uv run python examples/dataset_usage_example.py
```

This shows various usage patterns and dataset features.

### 3. Run Benchmarks

```bash
# Quick benchmark
uv run python examples/benchmark_dataset.py --benchmarks single loader

# Full benchmark suite
uv run python examples/benchmark_dataset.py
```

### 4. Train Example Model

```bash
uv run python examples/train_neural_rendering.py \
    --dataset-dir dataset \
    --ray-dataset-dir ray_dataset_hierarchical \
    --epochs 10
```

## Dataset API Overview

### Basic Usage

```python
from pathlib import Path
from voxel_dataset_generator.datasets import HierarchicalVoxelRayDataset

dataset = HierarchicalVoxelRayDataset(
    dataset_dir=Path("dataset"),
    ray_dataset_dir=Path("ray_dataset_hierarchical"),
    split='train',
)

sample = dataset[0]
# sample contains: voxels, origins, directions, distances, hits, etc.
```

### With DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

for batch in loader:
    voxels = batch['voxels']      # (8, 1, D, H, W)
    origins = batch['origins']    # (8, N, 3)
    # ... train your model
```

### Custom Ray Batching

```python
from voxel_dataset_generator.datasets import RayBatchSampler, collate_ray_batch

sampler = RayBatchSampler(
    dataset,
    rays_per_batch=4096,
    subvolumes_per_batch=8,
)

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=collate_ray_batch,
    num_workers=4,
)

for batch in loader:
    origins = batch['origins']         # (4096, 3) - all rays concatenated
    voxels = batch['voxels']           # (8, 1, D, H, W) - 8 subvolumes
    ray_to_voxel = batch['ray_to_voxel']  # (4096,) - which voxel each ray belongs to
```

## Data Augmentation

```python
from voxel_dataset_generator.datasets import transforms

transform = transforms.Compose([
    transforms.RandomRotation90(p=0.5),
    transforms.RandomFlip(axes=[0, 1, 2], p=0.5),
    transforms.NormalizeRayOrigins(voxel_size=1.0),
    transforms.RandomRaySubsample(num_rays=1024),
])

dataset = HierarchicalVoxelRayDataset(
    dataset_dir=Path("dataset"),
    ray_dataset_dir=Path("ray_dataset_hierarchical"),
    split='train',
    transform=transform,
)
```

## File Structure

After setup, your project should look like:

```
voxel-dataset-generator/
├── dataset/                           # Hierarchical voxel dataset
│   ├── metadata.json
│   ├── splits.json
│   ├── registry.json
│   ├── objects/
│   │   └── object_0000/
│   │       ├── metadata.json
│   │       ├── subdivision_map.json
│   │       └── level_0.npz
│   └── subvolumes/
│       └── level_1/
│           └── f3/
│               └── f30c11...npz
│
├── ray_dataset_hierarchical/          # Ray-traced data
│   ├── processing_summary.json
│   └── level_1/
│       └── f3/
│           └── f30c11..._rays.npz
│
├── src/voxel_dataset_generator/
│   └── datasets/                      # NEW: PyTorch dataset classes
│       ├── __init__.py
│       ├── neural_rendering_dataset.py
│       └── transforms.py
│
└── examples/
    ├── dataset_usage_example.py       # NEW: Basic usage
    ├── train_neural_rendering.py      # NEW: Training example
    ├── quick_timing_test.py          # NEW: Quick benchmark
    └── benchmark_dataset.py          # NEW: Full benchmarks
```

## Common Issues

### "No samples found for split 'train'"

**Cause:** Ray dataset doesn't exist or doesn't match voxel dataset.

**Fix:**
```bash
# Check if ray dataset exists
ls -la ray_dataset_hierarchical/

# Regenerate ray data if needed
uv run python optix_raytracer/examples/generate_hierarchical_dataset.py \
    --dataset-dir dataset \
    --output-dir ray_dataset_hierarchical
```

### Out of Memory

**Cause:** Loading too much data at once.

**Fix:**
```python
# Reduce batch size
loader = DataLoader(dataset, batch_size=4)  # Instead of 8

# Use level filtering (smaller voxel grids)
dataset = HierarchicalVoxelRayDataset(..., levels=[5])

# Reduce cache size
dataset = HierarchicalVoxelRayDataset(..., cache_size=10)

# Subsample rays
transform = transforms.RandomRaySubsample(num_rays=512)
dataset = HierarchicalVoxelRayDataset(..., transform=transform)
```

### Slow Performance

**Cause:** Not enough DataLoader workers or disk I/O bottleneck.

**Fix:**
```python
# Increase workers (up to num CPU cores)
loader = DataLoader(dataset, num_workers=8, pin_memory=True)

# Increase cache if you have RAM
dataset = HierarchicalVoxelRayDataset(..., cache_size=200)

# Use SSD for dataset storage
# Move dataset to SSD and update paths
```

### "FileNotFoundError: Splits file not found"

**Cause:** Dataset was generated without splits enabled.

**Fix:**
```python
# Regenerate splits for existing dataset
from pathlib import Path
from voxel_dataset_generator.pipeline import reassign_splits

reassign_splits(
    output_dir=Path("dataset"),
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.0,
)
```

## Performance Tuning

### Find Optimal Configuration

Run the comprehensive benchmark to find best settings for your hardware:

```bash
uv run python examples/benchmark_dataset.py
```

Pay attention to:
1. **Worker Scaling**: Shows optimal `num_workers`
2. **Cache Effectiveness**: Shows optimal `cache_size`
3. **Memory Usage**: Shows memory constraints

### Typical Configurations

**High-end workstation (32 cores, 64GB RAM, SSD):**
```python
dataset = HierarchicalVoxelRayDataset(..., cache_size=200)
loader = DataLoader(dataset, batch_size=16, num_workers=16, pin_memory=True)
```

**Mid-range system (8 cores, 16GB RAM, SSD):**
```python
dataset = HierarchicalVoxelRayDataset(..., cache_size=50)
loader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True)
```

**Limited memory (4 cores, 8GB RAM):**
```python
dataset = HierarchicalVoxelRayDataset(..., cache_size=10, levels=[5])
transform = transforms.RandomRaySubsample(num_rays=256)
dataset.transform = transform
loader = DataLoader(dataset, batch_size=4, num_workers=2)
```

## Next Steps

1. **Modify the example model** in `train_neural_rendering.py` for your use case
2. **Experiment with augmentations** to improve generalization
3. **Implement custom metrics** for your specific task
4. **Scale to multi-GPU** using PyTorch DDP or Lightning

## Documentation

- **[NEURAL_RENDERING_DATASET.md](NEURAL_RENDERING_DATASET.md)** - Full API documentation
- **[examples/README.md](examples/README.md)** - Example scripts guide
- **[src/voxel_dataset_generator/datasets/](src/voxel_dataset_generator/datasets/)** - Source code with docstrings

## Support

If you encounter issues:

1. Check the troubleshooting sections in this document
2. Run `quick_timing_test.py` to diagnose the problem
3. Run `benchmark_dataset.py --benchmarks memory` to check memory usage
4. Review the example scripts for correct usage patterns

## Citation

If you use this dataset infrastructure in your research, please cite:

```bibtex
@software{hierarchical_voxel_dataset,
  title = {Hierarchical Voxel Dataset Generator for Neural Rendering},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/voxel-dataset-generator}
}
```

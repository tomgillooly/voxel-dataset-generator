# Hierarchical Voxel Ray Dataset Format Specification

## Overview

The `HierarchicalVoxelRayDataset` provides ray-traced voxel data with flexible chunking and sparse representation options. This document specifies the exact format of data returned by the dataset for training neural rendering models.

## Installation & Import

```python
from pathlib import Path
from voxel_dataset_generator.datasets import (
    HierarchicalVoxelRayDataset,
    RayBatchSampler,
    collate_ray_batch,
)
from torch.utils.data import DataLoader
```

## Dataset Configuration Options

### Basic Parameters

- **`dataset_dir`**: `Path` - Path to hierarchical voxel dataset directory
- **`ray_dataset_dir`**: `Path` - Path to ray tracing dataset directory
- **`split`**: `Literal['train', 'val', 'test']` - Data split to use (default: `'train'`)
- **`levels`**: `Optional[List[int]]` - Hierarchy levels to include, e.g., `[3, 4, 5]` (default: `None` = all levels)
- **`rays_per_chunk`**: `Optional[int]` - Number of rays per chunk. If specified, rays are split into sequential chunks of this size. If `None`, each subvolume returns all rays in one chunk (default: `None`)
- **`cache_size`**: `int` - Number of voxel grids to cache in memory (default: `100`)
- **`include_empty`**: `bool` - Include empty subvolumes (default: `False`)
- **`transform`**: `Optional[callable]` - Optional transform function (default: `None`)
- **`seed`**: `int` - Random seed for reproducibility (default: `42`)

### Sparse Voxel Options (NEW)

- **`sparse_voxels`**: `bool` - If `True`, return voxels in sparse format instead of dense (default: `False`)
- **`sparse_mode`**: `Literal['coo', 'graph']` - Type of sparse representation:
  - `'coo'`: Sparse coordinates + features only
  - `'graph'`: Graph representation with edge indices for GNN operations
- **`sparse_connectivity`**: `int` - For graph mode, neighbor connectivity: `6`, `18`, or `26` (default: `6`)
  - `6`: Face neighbors only
  - `18`: Face + edge neighbors
  - `26`: All neighbors (face + edge + corner)

## Data Format

### Dataset Properties

```python
dataset = HierarchicalVoxelRayDataset(...)

# Number of chunks (not subvolumes!)
len(dataset)  # Returns total number of chunks across all subvolumes

# Access individual chunks
sample = dataset[idx]  # Returns a dictionary (format below)

# Get level distribution
dataset.get_level_distribution()  # Returns Dict[int, int] mapping level -> count
```

### Single Sample Format (Dense Voxels)

When `sparse_voxels=False` (default), each sample is a dictionary:

```python
sample = {
    # Ray data (for this chunk)
    'origins': torch.Tensor,      # Shape: (N, 3), dtype: float32
                                   # Ray origin points on bounding box surface

    'directions': torch.Tensor,   # Shape: (N, 3), dtype: float32
                                   # Ray directions (normalized unit vectors)

    'distances': torch.Tensor,    # Shape: (N,), dtype: float32
                                   # Ray hit distances, normalized by cube diagonal
                                   # Value of 0.0 indicates no hit (ray missed geometry)

    'hits': torch.Tensor,         # Shape: (N,), dtype: float32
                                   # Binary hit flags: 1.0 = hit, 0.0 = miss

    # Voxel data (dense format)
    'voxels': torch.Tensor,       # Shape: (1, D, H, W), dtype: float32
                                   # Channel-first format for conv nets
                                   # D=H=W depends on level: 2^(7-level)
                                   # Values: 1.0 = occupied, 0.0 = empty

    # Metadata
    'level': int,                 # Hierarchy level (0-7)
                                   # Level 0: full object (128³)
                                   # Level 7: smallest subvolume (1³)

    'hash': str,                  # Subvolume hash identifier
                                   # Format: "object_XXXX" for level 0
                                   # Format: hex hash for levels 1+

    'chunk_idx': int,             # Index of this chunk within the subvolume
                                   # 0-indexed, sequential

    # Optional fields (if present in ray data)
    'view_ids': torch.Tensor,     # Shape: (N,), dtype: int64 (optional)
                                   # View/camera index for each ray

    'face_ids': torch.Tensor,     # Shape: (N,), dtype: int64 (optional)
                                   # Bounding box face index (0-5)

    'view_positions': torch.Tensor, # Shape: (N, 3), dtype: float32 (optional)
                                     # Camera/view positions
}
```

**Note on chunking**:
- `N` = number of rays in this chunk
- If `rays_per_chunk=1000`, a subvolume with 10,000 rays creates 10 dataset items with N=1000 each
- If `rays_per_chunk=None`, N = total rays for entire subvolume (all rays in one chunk)

### Single Sample Format (Sparse Voxels - COO Mode)

When `sparse_voxels=True` and `sparse_mode='coo'`:

```python
sample = {
    # Ray data (same as dense format)
    'origins': torch.Tensor,      # Shape: (N, 3), dtype: float32
    'directions': torch.Tensor,   # Shape: (N, 3), dtype: float32
    'distances': torch.Tensor,    # Shape: (N,), dtype: float32
    'hits': torch.Tensor,         # Shape: (N,), dtype: float32

    # Sparse voxel data (COO format)
    'voxel_pos': torch.Tensor,    # Shape: (M, 3), dtype: float32
                                   # Coordinates of occupied voxels [x, y, z]
                                   # M = number of occupied voxels

    'voxel_features': torch.Tensor, # Shape: (M, 1), dtype: float32
                                     # Feature value for each occupied voxel
                                     # Currently all 1.0, but extensible

    'voxel_shape': torch.Tensor,  # Shape: (3,), dtype: int64
                                   # Original grid dimensions [D, H, W]

    'num_voxels': int,            # M, number of occupied voxels

    # Metadata (same as dense format)
    'level': int,
    'hash': str,
    'chunk_idx': int,

    # Optional fields (same as dense format)
    'view_ids': torch.Tensor,     # (optional)
    'face_ids': torch.Tensor,     # (optional)
    'view_positions': torch.Tensor, # (optional)
}
```

### Single Sample Format (Sparse Voxels - Graph Mode)

When `sparse_voxels=True` and `sparse_mode='graph'`:

```python
sample = {
    # Ray data (same as dense format)
    'origins': torch.Tensor,      # Shape: (N, 3), dtype: float32
    'directions': torch.Tensor,   # Shape: (N, 3), dtype: float32
    'distances': torch.Tensor,    # Shape: (N,), dtype: float32
    'hits': torch.Tensor,         # Shape: (N,), dtype: float32

    # Sparse voxel data (Graph format)
    'voxel_pos': torch.Tensor,    # Shape: (M, 3), dtype: float32
                                   # Coordinates of occupied voxels [x, y, z]

    'voxel_features': torch.Tensor, # Shape: (M, 1), dtype: float32
                                     # Node features for GNN

    'voxel_edge_index': torch.Tensor, # Shape: (2, E), dtype: int64
                                       # Edge connectivity in COO format
                                       # E = number of edges between occupied voxels
                                       # Format: [[src_nodes...], [dst_nodes...]]
                                       # Compatible with PyTorch Geometric

    'voxel_shape': torch.Tensor,  # Shape: (3,), dtype: int64
                                   # Original grid dimensions [D, H, W]

    # Metadata (same as dense format)
    'level': int,
    'hash': str,
    'chunk_idx': int,

    # Optional fields (same as dense format)
    'view_ids': torch.Tensor,     # (optional)
    'face_ids': torch.Tensor,     # (optional)
    'view_positions': torch.Tensor, # (optional)
}
```

## Batched Data Format

When using `collate_ray_batch()` or DataLoader with the custom collate function:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=8,  # Number of chunks per batch
    shuffle=True,
    collate_fn=collate_ray_batch,  # IMPORTANT: Use custom collate
    num_workers=4,
)

# Batched data format
for batch in loader:
    # batch is a dictionary with:
    batch = {
        # All rays concatenated across batch
        'origins': torch.Tensor,      # Shape: (total_rays, 3)
        'directions': torch.Tensor,   # Shape: (total_rays, 3)
        'distances': torch.Tensor,    # Shape: (total_rays,)
        'hits': torch.Tensor,         # Shape: (total_rays,)

        # Voxels stacked (padded if different sizes)
        'voxels': torch.Tensor,       # Shape: (batch_size, 1, D_max, H_max, W_max)
                                       # Padded with zeros if grids have different sizes

        # Mapping from rays to voxel grids
        'ray_to_voxel': torch.Tensor, # Shape: (total_rays,), dtype: int64
                                       # Maps each ray to its voxel grid index
                                       # Values in range [0, batch_size-1]

        # Metadata
        'levels': torch.Tensor,       # Shape: (batch_size,), dtype: int64
        'hashes': List[str],          # Length: batch_size

        # Optional fields (if present)
        'view_ids': torch.Tensor,     # Shape: (total_rays,) (optional)
        'face_ids': torch.Tensor,     # Shape: (total_rays,) (optional)
    }
```

**Important**: `total_rays = sum of rays across all chunks in batch`. If `rays_per_chunk=1000` and `batch_size=8`, then `total_rays ≈ 8000` (may vary for last chunk).

## Voxel Grid Dimensions by Level

```python
# Grid resolution by hierarchy level
level_to_resolution = {
    0: 128,  # 128³ voxels (full object)
    1: 64,   # 64³ voxels
    2: 32,   # 32³ voxels
    3: 16,   # 16³ voxels
    4: 8,    # 8³ voxels
    5: 4,    # 4³ voxels
    6: 2,    # 2³ voxels
    7: 1,    # 1³ voxels
}

resolution = 2 ** (7 - level)
```

## Distance Normalization

Ray distances are normalized by the cube diagonal for the given level:

```python
cube_size = 2.0 ** (7 - level)  # Physical size of cube
cube_diag = np.sqrt(3 * cube_size ** 2)  # Diagonal length
normalized_distance = raw_distance / cube_diag
```

This ensures distances are roughly in range [0, 1] for hits within the cube.

## Usage Examples

### Example 1: Basic Training Loop (Dense Voxels)

```python
from pathlib import Path
from voxel_dataset_generator.datasets import HierarchicalVoxelRayDataset
from torch.utils.data import DataLoader

# Create dataset with chunking
dataset = HierarchicalVoxelRayDataset(
    dataset_dir=Path("dataset"),
    ray_dataset_dir=Path("ray_dataset_hierarchical"),
    split='train',
    levels=[3, 4, 5],        # Use medium-resolution grids
    rays_per_chunk=4096,     # 4K rays per chunk
    cache_size=100,
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=8,            # 8 chunks per batch = ~32K rays total
    shuffle=True,
    num_workers=4,
)

# Training loop
for batch in loader:
    origins = batch['origins']        # (N_total, 3)
    directions = batch['directions']  # (N_total, 3)
    distances = batch['distances']    # (N_total,)
    hits = batch['hits']              # (N_total,)
    voxels = batch['voxels']          # (batch_size, 1, D, H, W)
    ray_to_voxel = batch['ray_to_voxel']  # (N_total,)

    # Your model forward pass
    # predictions = model(origins, directions, voxels, ray_to_voxel)
    # loss = criterion(predictions, distances)
```

### Example 2: Sparse Voxels for GNN

```python
# Create dataset with sparse graph representation
dataset = HierarchicalVoxelRayDataset(
    dataset_dir=Path("dataset"),
    ray_dataset_dir=Path("ray_dataset_hierarchical"),
    split='train',
    levels=[3, 4, 5],
    rays_per_chunk=2048,
    sparse_voxels=True,           # Enable sparse format
    sparse_mode='graph',          # Graph with edges
    sparse_connectivity=6,        # 6-connected neighbors
    cache_size=50,
)

# Iterate over samples
for sample in dataset:
    # Ray data
    origins = sample['origins']              # (N, 3)
    directions = sample['directions']        # (N, 3)

    # Sparse voxel graph
    voxel_pos = sample['voxel_pos']          # (M, 3) occupied voxel coords
    voxel_features = sample['voxel_features'] # (M, 1) node features
    edge_index = sample['voxel_edge_index']  # (2, E) edge connectivity

    # Use with PyTorch Geometric GNN
    # from torch_geometric.nn import GCNConv
    # encoded = gnn_encoder(voxel_features, edge_index)
    # predictions = ray_predictor(origins, directions, encoded, voxel_pos)
```

### Example 3: Memory-Efficient Sparse COO

```python
# For maximum memory efficiency
dataset = HierarchicalVoxelRayDataset(
    dataset_dir=Path("dataset"),
    ray_dataset_dir=Path("ray_dataset_hierarchical"),
    split='train',
    rays_per_chunk=1024,
    sparse_voxels=True,
    sparse_mode='coo',            # Coordinates only (no edges)
    cache_size=200,               # Can cache more since voxels are sparse
)

for sample in dataset:
    voxel_pos = sample['voxel_pos']          # (M, 3) - Only occupied voxels
    voxel_features = sample['voxel_features'] # (M, 1)

    # Typical sparse voxel grids use ~1-5% of dense memory
    # E.g., 16³ dense = 4096 values, sparse = ~50-200 occupied voxels
```

### Example 4: Custom Batch Sampler

```python
from voxel_dataset_generator.datasets import RayBatchSampler

# Custom sampler for controlling batch composition
sampler = RayBatchSampler(
    dataset,
    batch_size=16,        # 16 chunks per batch
    shuffle=True,
    drop_last=True,
)

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=4,
)
```

## Key Implementation Notes

1. **Chunking vs. Subsampling**: The dataset now uses sequential chunking instead of random subsampling. This means:
   - All rays are used (no data is discarded)
   - Chunks contain consecutive rays
   - Multiple chunks from the same subvolume share the same voxel grid

2. **Memory Considerations**:
   - Dense voxels: 128³ = 2MB per grid, 64³ = 256KB, 32³ = 32KB
   - Sparse voxels: Typically 1-5% of dense size for typical occupancy
   - Cache size should be tuned based on available RAM

3. **Batch Composition**:
   - Each batch may contain chunks from different subvolumes and levels
   - Use `ray_to_voxel` mapping to associate rays with their voxel grids
   - Voxel grids of different sizes are padded to max size in batch

4. **PyTorch Geometric Integration**:
   - Graph mode is directly compatible with PyTorch Geometric layers
   - No additional conversion needed for GNN operations
   - Edge indices follow standard COO format: `[source_nodes, target_nodes]`

5. **Validation/Testing**:
   - Use `split='val'` or `split='test'` for evaluation
   - Same format as training data
   - Data splits are pre-defined in `splits.json`

## Coordinate System

- **Voxel coordinates**: Integer indices in [0, resolution-1] for each dimension
- **Sparse voxel_pos**: Converted to float32 for compatibility with GNNs
- **Ray origins/directions**: Continuous coordinates in world space
- Coordinate convention: `(x, y, z)` = `(width, height, depth)`

## Version Compatibility

This specification is for the updated dataset with:
- Chunking support (instead of subsampling)
- Sparse voxel representations
- PyTorch Geometric compatibility

**Breaking changes from previous version**:
- `samples_per_subvolume` parameter renamed to `rays_per_chunk`
- Behavior changed from random sampling to sequential chunking
- Dataset length now returns number of chunks, not subvolumes

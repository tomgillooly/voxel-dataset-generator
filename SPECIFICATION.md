# Voxel Dataset Generator Specification

## Project Overview

This project generates hierarchical voxel datasets from 3D mesh objects for neural rendering research, specifically focusing on composing neural representations of sub-volumes.

## Objectives

Generate a dataset where:
- Top-level objects are voxelized at a consistent grid resolution
- Each voxel structure is recursively subdivided into spatially coherent sub-volumes
- Duplicate sub-volumes are identified and deduplicated
- The dataset enables research into compositional neural representations

## Input Data

- **Source**: Thingi10k dataset
- **Format**: STL files (mesh representations)
- **Characteristics**: Variable object sizes and complexities

## Core Requirements

### 1. Mesh to Voxel Conversion

- Convert STL mesh files to voxel grids
- **Normalization**: Adjust voxel pitch per object to ensure consistent grid dimensions
- **Target Resolution**: 128×128×128 voxels (configurable)
- **Binary Occupancy**: Each voxel is either occupied (1) or empty (0)

### 2. Recursive Subdivision

Generate a hierarchical structure through spatial subdivision:

```
Level 0: 128×128×128 (1 volume)
Level 1: 64×64×64 (8 sub-volumes)
Level 2: 32×32×32 (64 sub-volumes)
Level 3: 16×16×16 (512 sub-volumes)
Level 4: 8×8×8 (4096 sub-volumes)
Level 5: 4×4×4 (32768 sub-volumes)
```

**Spatial Division**: Each volume at level N is divided into 8 equal octants (2×2×2) to create level N+1

### 3. Deduplication Strategy

- **Hash-based identification**: Compute hash (e.g., SHA-256) of each sub-volume's voxel data
- **Unique storage**: Store only one instance of each unique sub-volume
- **Reference tracking**: Maintain mappings between parent objects and their unique sub-volumes
- **Statistics collection**: Track frequency of each unique sub-volume across dataset

### 4. Data Structure

```
dataset/
├── metadata.json                 # Dataset-level metadata
├── objects/
│   ├── object_0000/
│   │   ├── metadata.json        # Object metadata (source file, bbox, etc.)
│   │   ├── level_0.npy          # 128×128×128 voxel grid
│   │   └── subdivision_map.json # Maps to unique sub-volumes
│   ├── object_0001/
│   └── ...
└── subvolumes/
    ├── level_1/
    │   ├── <hash_1>.npy         # Unique 64×64×64 sub-volume
    │   ├── <hash_2>.npy
    │   └── ...
    ├── level_2/
    ├── level_3/
    ├── level_4/
    └── level_5/
```

## Technical Specifications

### Voxelization Algorithm

1. **Bounding Box Computation**: Calculate axis-aligned bounding box (AABB) of mesh
2. **Pitch Calculation**: `pitch = max(bbox_dimensions) / target_resolution`
3. **Grid Generation**: Create regular grid covering AABB
4. **Occupancy Testing**: Use ray-casting or surface voxelization to determine occupied voxels
5. **Padding**: Ensure grid is exactly target_resolution³ (pad with empty voxels if needed)

### Subdivision Algorithm

```python
def subdivide(voxel_grid, level):
    """
    Recursively subdivide voxel grid into octants
    """
    if grid_size < minimum_size:
        return

    sub_volumes = []
    half = grid_size // 2

    # Extract 8 octants
    for i in [0, half]:
        for j in [0, half]:
            for k in [0, half]:
                sub_volume = voxel_grid[i:i+half, j:j+half, k:k+half]
                hash_value = compute_hash(sub_volume)
                sub_volumes.append({
                    'hash': hash_value,
                    'position': (i, j, k),
                    'data': sub_volume
                })

    return sub_volumes
```

### Deduplication Process

1. Compute hash of each sub-volume's voxel data
2. Check if hash exists in global sub-volume registry
3. If new: Save sub-volume, add to registry
4. If duplicate: Increment reference count
5. Store hash reference in parent's subdivision map

## Output Metadata

### Dataset Metadata (metadata.json)
```json
{
  "dataset_name": "Thingi10k_Hierarchical_Voxels",
  "version": "1.0",
  "base_resolution": 128,
  "num_levels": 6,
  "num_objects": 10000,
  "deduplication_stats": {
    "level_1": {"unique": 1234, "total": 80000},
    "level_2": {"unique": 5678, "total": 640000},
    ...
  }
}
```

### Object Metadata (objects/object_XXXX/metadata.json)
```json
{
  "object_id": "0000",
  "source_file": "thingi10k/12345.stl",
  "original_bbox": {"min": [x, y, z], "max": [x, y, z]},
  "voxel_pitch": 0.0123,
  "num_occupied_voxels": 45678
}
```

### Subdivision Map (objects/object_XXXX/subdivision_map.json)

This file contains a flat list of all sub-volume dependencies, optimized for dataframe ingestion:

```json
[
  {
    "object_id": "0000",
    "level": 1,
    "octant_index": 0,
    "position_x": 0,
    "position_y": 0,
    "position_z": 0,
    "hash": "abc123...",
    "is_empty": false
  },
  {
    "object_id": "0000",
    "level": 1,
    "octant_index": 1,
    "position_x": 0,
    "position_y": 0,
    "position_z": 64,
    "hash": "def456...",
    "is_empty": false
  }
]
```

**Note**: This flat structure allows easy aggregation across all objects using:
```python
import polars as pl
# Load all subdivision maps into a single dataframe
df = pl.read_json("dataset/objects/*/subdivision_map.json")
# Analyze sub-volume frequency
frequency = df.group_by("hash").count().sort("count", descending=True)
```

## Implementation Phases

### Phase 1: Core Voxelization
- [ ] STL file loading and parsing
- [ ] Mesh to voxel conversion with pitch normalization
- [ ] Basic voxel grid storage

### Phase 2: Subdivision System
- [ ] Recursive octree subdivision
- [ ] Sub-volume extraction
- [ ] Hierarchical data structure generation

### Phase 3: Deduplication
- [ ] Hash computation for voxel data
- [ ] Global sub-volume registry
- [ ] Duplicate detection and storage optimization

### Phase 4: Dataset Generation
- [ ] Batch processing of Thingi10k dataset
- [ ] Metadata generation
- [ ] Statistics and analysis tools

### Phase 5: Utilities
- [ ] Visualization tools
- [ ] Dataset validation
- [ ] Loading utilities for training

## Dependencies

- **numpy**: Array operations and voxel storage
- **trimesh**: STL loading and mesh operations
- **scipy**: Spatial operations (optional)
- **polars**: Fast dataframe operations for analyzing sub-volume dependencies
- **thingi10k**: Python library for downloading Thingi10k dataset
- **hashlib**: Hash computation (stdlib)
- **json**: Metadata serialization (stdlib)

## Dataset Analysis Workflow

The flat JSON structure enables efficient cross-dataset analysis using Polars or Pandas:

### Loading All Subdivision Data
```python
import polars as pl
from pathlib import Path

# Load all subdivision maps into a single dataframe
subdivision_files = Path("dataset/objects").glob("*/subdivision_map.json")
df = pl.concat([pl.read_json(f) for f in subdivision_files])

# Alternative: if subdivision_map.json contains records per line (NDJSON)
df = pl.read_ndjson("dataset/objects/*/subdivision_map.json")
```

### Example Analyses
```python
# 1. Sub-volume frequency analysis
hash_frequency = df.group_by(["level", "hash"]).agg(
    pl.count().alias("frequency")
).sort("frequency", descending=True)

# 2. Find most reused sub-volumes
top_reused = df.filter(pl.col("level") == 3).group_by("hash").count().top_k(100)

# 3. Empty vs occupied ratio per level
empty_stats = df.group_by("level").agg([
    pl.col("is_empty").sum().alias("empty_count"),
    pl.count().alias("total_count")
])

# 4. Object complexity (number of unique sub-volumes per object)
object_complexity = df.group_by("object_id").agg(
    pl.col("hash").n_unique().alias("unique_subvolumes")
)

# 5. Co-occurrence matrix (which sub-volumes appear together)
from itertools import combinations
cooccurrence = df.group_by("object_id").agg(
    pl.col("hash").unique()
).explode("hash")
```

## Validation Criteria

1. All objects voxelized to consistent grid size
2. Subdivision maintains spatial coherence
3. No data loss during subdivision (union of sub-volumes equals parent)
4. Deduplication correctly identifies identical sub-volumes
5. Metadata allows reconstruction of hierarchical relationships
6. JSON files can be loaded and analyzed as a single dataframe

## Performance Considerations

- **Memory**: Process objects incrementally to manage memory
- **Storage**: Use compressed format (.npz) for sparse voxel grids
- **Speed**: Parallelize voxelization across multiple objects
- **Hash collisions**: Use cryptographic hash to minimize collision probability

## Future Extensions

- Support for multiple base resolutions
- Adaptive subdivision (stop at occupied/empty homogeneous regions)
- Multi-scale feature encoding
- Integration with neural rendering frameworks

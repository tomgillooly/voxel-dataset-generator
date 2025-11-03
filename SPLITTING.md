# Train/Validation Split Functionality

This document describes the hierarchical train/validation/test splitting functionality for voxel datasets.

## Overview

The splitting system ensures that when a top-level object is assigned to a particular split (train/val/test), all of its sub-volumes are tracked accordingly. This allows you to:

1. **Assign splits at the object level**: Each top-level voxel grid is assigned to train, validation, or test
2. **Track sub-volume membership**: Monitor which sub-volumes appear in which splits
3. **Analyze leakage**: Understand what percentage of validation objects' sub-volumes also appear in training data
4. **Handle trivial sub-volumes intelligently**: Empty (all 0s) and full (all 1s) sub-volumes are allowed in multiple splits since they contain no useful information

## Key Concepts

### Hierarchical Assignment

- **Top-level assignment**: Splits are assigned to objects (level 0 voxel grids)
- **Sub-volume tracking**: All sub-volumes of an object are tracked as belonging to that object's split
- **Cross-split sharing**: Sub-volumes can be shared across objects in different splits (this is tracked, not prevented)

### Leakage Analysis

The system provides statistics on "purity" vs "leakage":

- **Pure sub-volumes**: Only appear in one split
- **Shared (same split)**: Appear in multiple objects within the same split
- **Shared (cross-split)**: Appear in objects across different splits (leakage)
- **Trivial sub-volumes**: All 0s or all 1s - excluded from leakage calculations

## Usage

### Basic Example

```python
from pathlib import Path
from voxel_dataset_generator.pipeline import DatasetGenerator
from voxel_dataset_generator.utils.config import Config

# Configure with splitting enabled
config = Config(
    base_resolution=128,
    min_resolution=4,
    output_dir=Path("dataset"),
    enable_splitting=True,
    train_ratio=0.8,
    val_ratio=0.2,
    test_ratio=0.0,
    split_seed=42
)

# Create generator and process meshes
generator = DatasetGenerator(config)
results = generator.process_batch(mesh_files)

# Finalize - automatically assigns splits if not done
generator.finalize()
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_splitting` | bool | False | Enable train/val/test splitting |
| `train_ratio` | float | 0.8 | Ratio of objects for training (0.0-1.0) |
| `val_ratio` | float | 0.2 | Ratio of objects for validation (0.0-1.0) |
| `test_ratio` | float | 0.0 | Ratio of objects for testing (0.0-1.0) |
| `split_seed` | int | 42 | Random seed for reproducible splits |

**Note**: The three ratios must sum to 1.0.

### Output Files

When splitting is enabled, the following files are generated:

#### 1. `splits.json`

Contains all split assignments and statistics:

```json
{
  "config": {
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "test_ratio": 0.0,
    "seed": 42
  },
  "object_splits": {
    "0000": "train",
    "0001": "val",
    ...
  },
  "hash_object_usage": {
    "abc123...": {
      "0000": "train",
      "0042": "train",
      "0099": "val"
    },
    ...
  },
  "trivial_hashes": ["def456...", "ghi789...", ...],
  "statistics": {
    "objects": {
      "train": 80,
      "val": 20,
      "test": 0,
      "total": 100
    },
    "unique_nontrivial_subvolumes": {
      "total": 12345,
      "distribution": {
        "only_train": 9876,
        "only_val": 2100,
        "train_val": 369,
        ...
      }
    },
    "trivial_subvolumes": {
      "count": 543
    }
  }
}
```

#### 2. Updated `metadata.json`

The dataset-level metadata includes split configuration:

```json
{
  "splitting": {
    "enabled": true,
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "test_ratio": 0.0,
    "seed": 42
  },
  ...
}
```

## Analysis

### Analyzing Object Purity

Check what percentage of a validation object's sub-volumes are "pure" (not in training):

```python
from voxel_dataset_generator.splitting import HierarchicalSplitter
import json

# Load split assignments
splitter = HierarchicalSplitter.load_split_assignments(Path("dataset/splits.json"))

# Load object's subdivision map
with open("dataset/objects/object_0001/subdivision_map.json") as f:
    subdivision_map = json.load(f)

subvolume_hashes = [record["hash"] for record in subdivision_map]

# Analyze purity
purity_stats = splitter.analyze_object_subvolumes("0001", subvolume_hashes)

print(f"Purity: {purity_stats['purity_percentage']:.2f}%")
print(f"Leakage: {purity_stats['leakage_percentage']:.2f}%")
```

Output:
```
{
  "object_id": "0001",
  "split": "val",
  "total_subvolumes": 1092,
  "trivial": 500,
  "nontrivial": 592,
  "pure": 450,
  "shared_same_split": 30,
  "shared_other_split": 112,
  "purity_percentage": 76.01,
  "leakage_percentage": 18.92
}
```

### Querying Splits

```python
from voxel_dataset_generator.splitting import Split

# Get all training objects
train_objects = splitter.get_objects_by_split(Split.TRAIN)
print(f"Training objects: {train_objects}")

# Get all unique non-trivial sub-volumes in validation
val_hashes = splitter.get_nontrivial_hashes_by_split(Split.VAL)
print(f"Unique validation sub-volumes: {len(val_hashes)}")

# Check which splits use a specific hash
hash_val = "abc123..."
splits_using = splitter.get_splits_using_hash(hash_val)
print(f"Hash appears in: {splits_using}")

# Get all objects using a hash
objects_using = splitter.get_objects_using_hash(hash_val)
print(f"Objects: {objects_using}")
# Output: {"0001": Split.TRAIN, "0042": Split.TRAIN, "0099": Split.VAL}
```

### Export Detailed Sub-volume Information

```python
# Export complete sub-volume split information
subvolume_info = splitter.export_subvolume_split_info()

# Example entry:
{
  "abc123...": {
    "splits": ["train", "val"],
    "is_trivial": false,
    "used_by_objects": {
      "0001": "train",
      "0042": "train",
      "0099": "val"
    }
  }
}
```

## Understanding Leakage

### What is Leakage?

Leakage occurs when a sub-volume that appears in a training object also appears in a validation or test object. This can happen because:

1. **Deduplication**: The same geometric pattern appears in multiple objects
2. **Common structures**: Simple shapes (corners, edges, flat surfaces) are reused

### Trivial Sub-volumes

The system treats completely empty (all 0s) and completely full (all 1s) sub-volumes specially:

- **Excluded from leakage statistics**: They contain no useful information
- **Allowed in all splits**: Preventing their sharing would be impractical
- **Tracked separately**: Still counted in statistics but marked as trivial

### Interpreting Statistics

From `splits.json`:

```json
"unique_nontrivial_subvolumes": {
  "distribution": {
    "only_train": 9876,      // Only in training objects
    "only_val": 2100,        // Only in validation objects
    "train_val": 369,        // Appears in both (leakage)
    "train_test": 0,
    "val_test": 0,
    "all_splits": 0
  }
}
```

**Leakage percentage** = `train_val / (only_train + only_val + train_val) * 100`

For this example: `369 / (9876 + 2100 + 369) * 100 = 2.99%`

## Advanced Usage

### Manual Split Assignment

You can assign splits before processing or between processing batches:

```python
generator = DatasetGenerator(config)

# Process first batch
batch1_results = generator.process_batch(batch1_files)

# Manually assign splits
object_ids = [r["object_id"] for r in batch1_results]
generator.assign_splits(object_ids)

# Process more batches
batch2_results = generator.process_batch(batch2_files)

# Finalize
generator.finalize()
```

### Custom Split Ratios

```python
config = Config(
    enable_splitting=True,
    train_ratio=0.7,   # 70%
    val_ratio=0.2,     # 20%
    test_ratio=0.1,    # 10%
    split_seed=123
)
```

### Loading Splits for Analysis

```python
# Load splits without regenerating dataset
splitter = HierarchicalSplitter.load_split_assignments(
    Path("dataset/splits.json")
)

# Use for analysis, querying, etc.
stats = splitter.get_split_statistics()
```

## API Reference

### HierarchicalSplitter

Main class for managing splits.

**Methods:**

- `assign_splits(object_ids)`: Assign splits to objects
- `register_subvolume(object_id, hash_val, voxel_data, is_trivial)`: Register a sub-volume
- `get_object_split(object_id)`: Get split for an object
- `get_splits_using_hash(hash_val)`: Get all splits containing a hash
- `get_objects_using_hash(hash_val)`: Get all objects using a hash
- `analyze_object_subvolumes(object_id, hashes)`: Analyze object purity
- `get_split_statistics()`: Get overall statistics
- `get_objects_by_split(split)`: Get objects in a split
- `get_nontrivial_hashes_by_split(split)`: Get unique hashes in a split
- `export_subvolume_split_info()`: Export detailed info
- `save_split_assignments(path)`: Save to JSON
- `load_split_assignments(path)`: Load from JSON (class method)

### Split Enum

```python
from voxel_dataset_generator.splitting import Split

Split.TRAIN  # "train"
Split.VAL    # "val"
Split.TEST   # "test"
```

## Examples

See `examples/example_with_splits.py` for complete working examples:

1. Basic split generation
2. Analyzing object purity
3. Querying split information
4. Exporting sub-volume details
5. Custom workflows

## Best Practices

1. **Set a seed**: Use `split_seed` for reproducible splits
2. **Assign splits early**: Call `assign_splits()` after processing all objects but before `finalize()`
3. **Monitor leakage**: Check split statistics to understand cross-split sharing
4. **Account for trivial sub-volumes**: Remember they're excluded from leakage calculations
5. **Save split info**: Keep `splits.json` for reproducibility and analysis

## Future Enhancements

Potential additions:

- Stratified splitting (by object properties)
- K-fold cross-validation support
- Per-object leakage reports
- Visualization tools for split distribution

# Voxel Dataset Generator

A hierarchical voxel dataset generator for neural rendering research, specifically designed for compositional neural representations of sub-volumes.

## Overview

This tool converts 3D meshes (STL files) into hierarchical voxel datasets by:

1. **Voxelizing** meshes to a consistent resolution (default: 128x128x128)
2. **Subdividing** voxel grids recursively into octree hierarchies
3. **Deduplicating** sub-volumes to identify reusable components
4. **Generating** comprehensive metadata for analysis

## Features

- **Consistent Resolution**: Normalizes mesh sizes via pitch adjustment
- **Hierarchical Structure**: Recursive octree subdivision (128^3 -> 64^3 -> 32^3 -> 16^3 -> 8^3 -> 4^3)
- **Efficient Deduplication**: Hash-based identification of unique sub-volumes
- **Analysis-Ready**: Flat JSON structure optimized for Polars/Pandas dataframes
- **Thingi10k Integration**: Built-in support for downloading and processing Thingi10k dataset
- **GPU Ray Tracing** ⚡: OptiX-accelerated ray tracing through voxels with transparency and distance accumulation (see [optix_raytracer/](optix_raytracer/))

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd voxel-dataset-generator

# Install with uv
uv sync
```

## Quick Start

### Command Line

After installation, you can use the `voxel-gen` command:

```bash
# Generate dataset from Thingi10k
uv run voxel-gen --num-objects 10 --resolution 128

# Using the Python script directly
uv run python voxel_gen.py --num-objects 10

# Or as a module
uv run python -m voxel_dataset_generator.pipeline --num-objects 10
```

### Process Thingi10k Dataset

```python
from pathlib import Path
from voxel_dataset_generator.pipeline import generate_dataset_from_thingi10k

# Generate dataset from first 100 objects
generate_dataset_from_thingi10k(
    num_objects=100,
    output_dir=Path("dataset"),
    base_resolution=128,
    min_resolution=4
)
```

### Process Custom Meshes

```python
from pathlib import Path
from voxel_dataset_generator import Config
from voxel_dataset_generator.pipeline import DatasetGenerator

# Create configuration
config = Config(
    base_resolution=128,
    min_resolution=4,
    output_dir=Path("output/my_dataset")
)

# Initialize generator
generator = DatasetGenerator(config)

# Process mesh files
mesh_files = list(Path("meshes").glob("*.stl"))
results = generator.process_batch(mesh_files, show_progress=True)

# Finalize to save metadata
generator.finalize()
```

## Dataset Structure

```
dataset/
   metadata.json                 # Dataset-level metadata
   registry.json                 # Global sub-volume registry
   objects/
      object_0000/
         metadata.json        # Object metadata
         level_0.npz          # 128x128x128 voxel grid
         subdivision_map.json # Sub-volume dependencies
      ...
   subvolumes/
       level_1/
          <hash_prefix>/
              <hash>.npz       # Unique 64x64x64 sub-volume
       level_2/
       ...
```

## Analyzing the Dataset

### Using Polars

```python
import polars as pl
from pathlib import Path

# Load all subdivision maps
objects_dir = Path("dataset/objects")
subdivision_files = list(objects_dir.glob("*/subdivision_map.json"))
df = pl.concat([pl.read_json(f) for f in subdivision_files])

# Find most reused sub-volumes
frequency = df.group_by(["level", "hash"]).agg(
    pl.count().alias("frequency")
).sort("frequency", descending=True)

print(frequency.head(10))
```

### Using Built-in Analyzer

```python
from voxel_dataset_generator.utils.metadata import MetadataAnalyzer

analyzer = MetadataAnalyzer(Path("dataset"))
stats = analyzer.summary_statistics()
print(stats)

# Get hash frequencies
hash_freq = analyzer.get_hash_frequency(level=3)

# Export for Polars
analyzer.export_to_polars_compatible(Path("dataset/analysis.ndjson"))
```

## Visualization

Interactive visualizations are available via Jupyter notebook. Install visualization dependencies:

```bash
# Install with visualization support
uv sync --extra viz

# Start Jupyter
uv run jupyter notebook
```

Then open `visualize_dataset.ipynb` for an **interactive hierarchical explorer** with:
- **Drill-Down Navigation** - Click through octants to explore the hierarchy level by level
- **Multiple View Modes** - Switch between cubes, isosurfaces, and auto-selected styles
- **Real-time Updates** - Instantly see sub-volumes as you navigate deeper
- **Global Position Tracking** - Clear breadcrumb navigation showing your path through the tree

See [VISUALIZATION.md](VISUALIZATION.md) for complete guide.

## OptiX Ray Tracing (Optional)

High-performance GPU ray tracing for voxel grids is available via the OptiX module. Ray trace through voxels treating them as transparent and accumulate distance traveled through the object.

```python
from optix_voxel_tracer import VoxelRayTracer
import numpy as np

# Load voxel grid
voxels = np.load("dataset/objects/object_0001/level_0.npz")['voxels']

# Create tracer
tracer = VoxelRayTracer(voxels)

# Trace rays and get accumulated distances
distances = tracer.trace_rays(origins, directions)
```

**Features**:
- GPU-accelerated with NVIDIA OptiX
- Transparent voxel traversal
- Distance accumulation through occupied voxels
- Multiple viewpoint rendering
- Batch processing support

**See**:
- [optix_raytracer/QUICKSTART.md](optix_raytracer/QUICKSTART.md) - Build and usage guide
- [OPTIX_INTEGRATION.md](OPTIX_INTEGRATION.md) - Integration with dataset pipeline
- [optix_raytracer/examples/](optix_raytracer/examples/) - Example scripts

**Requirements**: NVIDIA GPU (RTX series), CUDA 11.0+, OptiX SDK 7.0+

## Examples

See the [examples](examples/) directory for detailed usage examples:

- `basic_usage.py`: Basic dataset generation examples
- `analyze_dataset.py`: Comprehensive dataset analysis with Polars
- `process_from_arrays.py`: Process meshes from numpy arrays (Thingi10k npz format)

Run an example:
```bash
uv run python examples/basic_usage.py
```

## Configuration

Key configuration parameters:

- `base_resolution`: Top-level voxel grid size (default: 128)
- `min_resolution`: Minimum subdivision size (default: 4)
- `output_dir`: Dataset output directory
- `compression`: Enable .npz compression (default: True)
- `solid_voxelization`: Fill interior of meshes (default: True)
  - `True`: Creates solid voxelized objects (filled interior)
  - `False`: Only voxelizes the surface boundary

Example with custom configuration:

```python
from voxel_dataset_generator import Config
from voxel_dataset_generator.pipeline import DatasetGenerator

config = Config(
    base_resolution=128,
    solid_voxelization=True,  # Fill interiors
    compression=True
)

generator = DatasetGenerator(config)
```

## Architecture

### Core Modules

- **voxelization**: STL mesh to voxel grid conversion with normalization
- **subdivision**: Recursive octree subdivision into hierarchies
- **deduplication**: Hash-based sub-volume registry for deduplication
- **utils**: Configuration, metadata generation, and analysis tools
- **pipeline**: Main orchestration pipeline

### Key Classes

- `Voxelizer`: Converts meshes to voxel grids
- `Subdivider`: Performs recursive octree subdivision
- `SubvolumeRegistry`: Manages deduplication across dataset
- `DatasetGenerator`: Main pipeline orchestrator
- `MetadataAnalyzer`: Dataset analysis utilities

## Testing

Run the test suite:

```bash
uv run python tests/test_basic.py
```

## Performance Considerations

- **Memory**: Processes objects incrementally
- **Storage**: Uses compressed .npz format for sparse voxel grids
- **Deduplication**: Sub-volume files organized by hash prefix to avoid filesystem limits
- **Batch Processing**: Progress tracking with tqdm

## Research Use Case

This tool is designed for research into compositional neural representations, where:

1. Objects are represented as compositions of learned sub-volume features
2. Sub-volumes can be shared across different objects
3. Hierarchical structure enables multi-scale representations

The deduplication statistics reveal which sub-volumes are most reusable, informing neural network architecture design.

## Contributing

Contributions are welcome! Please see [SPECIFICATION.md](SPECIFICATION.md) for detailed technical specifications.

## License

MIT License

## Acknowledgments

- Built with [trimesh](https://trimsh.org/) for mesh processing
- Uses [Thingi10k](https://ten-thousand-models.appspot.com/) dataset
- Data analysis powered by [Polars](https://pola.rs/)

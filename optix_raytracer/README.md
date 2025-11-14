# OptiX Voxel Ray Tracer

High-performance GPU ray tracer for voxel occupancy grids using NVIDIA OptiX 7+.

## Features

- **Transparent Voxel Ray Tracing**: Cast rays through voxel grids and accumulate distance traveled through occupied voxels
- **GPU Accelerated**: Leverages NVIDIA OptiX for high-performance ray tracing
- **Distance Accumulation**: Records total path length through solid voxels
- **Python Integration**: Easy-to-use Python bindings via pybind11
- **Compatible with Pipeline**: Reads .npz voxel files from the main dataset generator

## Requirements

- NVIDIA GPU with OptiX support (RTX cards recommended)
- CUDA Toolkit 11.0+
- OptiX SDK 7.0+ (download from NVIDIA developer site)
- CMake 3.18+
- C++17 compiler
- Python 3.11+

## Building

### 1. Install OptiX SDK

Download and install the OptiX SDK from:
https://developer.nvidia.com/optix

Set the `OptiX_INSTALL_DIR` environment variable:
```bash
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.x.x
```

### 2. Build the Module

```bash
cd optix_raytracer
mkdir build && cd build
cmake ..
make -j8
```

This will build:
- `liboptix_voxel_tracer.so` - Core C++ library
- `optix_voxel_tracer.*.so` - Python module

### 3. Install Python Module

```bash
# From the build directory
pip install .
```

Or add to your Python path:
```bash
export PYTHONPATH=/path/to/optix_raytracer/build:$PYTHONPATH
```

## Usage

### Python API

```python
from optix_voxel_tracer import VoxelRayTracer
import numpy as np

# Load voxel grid from your dataset
voxel_grid = np.load("dataset/objects/object_0001/level_0.npz")['voxels']

# Create ray tracer
tracer = VoxelRayTracer(voxel_grid)

# Define camera rays (origins and directions)
# Example: orthographic rays from +Z direction
resolution = (512, 512)
origins = np.zeros((resolution[0], resolution[1], 3), dtype=np.float32)
directions = np.zeros((resolution[0], resolution[1], 3), dtype=np.float32)

for i in range(resolution[0]):
    for j in range(resolution[1]):
        # Map to [-1, 1] range
        x = (i / resolution[0]) * 2 - 1
        y = (j / resolution[1]) * 2 - 1
        origins[i, j] = [x, y, 2.0]  # Start above grid
        directions[i, j] = [0, 0, -1]  # Point down

# Cast rays and get distance accumulation
distances = tracer.trace_rays(origins, directions)

# distances[i, j] contains total distance traveled through occupied voxels
print(f"Max distance through object: {distances.max()}")
```

### Advanced Usage

```python
# Perspective camera rays
def generate_perspective_rays(resolution, fov, camera_pos, look_at):
    """Generate rays for a perspective camera."""
    # ... camera ray generation logic ...
    return origins, directions

origins, directions = generate_perspective_rays(
    resolution=(1024, 1024),
    fov=60.0,
    camera_pos=np.array([0, 0, 3]),
    look_at=np.array([0, 0, 0])
)

distances = tracer.trace_rays(origins, directions)

# Visualize as depth map
import matplotlib.pyplot as plt
plt.imshow(distances, cmap='viridis')
plt.colorbar(label='Distance through voxels')
plt.show()
```

### Batch Processing

```python
from pathlib import Path
from optix_voxel_tracer import VoxelRayTracer
import numpy as np

# Process multiple objects
dataset_dir = Path("dataset/objects")

for obj_dir in dataset_dir.iterdir():
    voxel_file = obj_dir / "level_0.npz"
    if not voxel_file.exists():
        continue

    # Load and trace
    voxels = np.load(voxel_file)['voxels']
    tracer = VoxelRayTracer(voxels)

    # Generate rays (e.g., from multiple viewpoints)
    distances = tracer.trace_rays(origins, directions)

    # Save results
    output_file = obj_dir / "ray_distances.npz"
    np.savez_compressed(output_file, distances=distances)
```

## API Reference

### VoxelRayTracer

#### Constructor
```python
VoxelRayTracer(voxel_grid: np.ndarray, voxel_size: float = 1.0)
```
- `voxel_grid`: 3D numpy array (boolean or uint8), shape (X, Y, Z)
- `voxel_size`: Physical size of each voxel (default: 1.0)

#### Methods

**`trace_rays(origins, directions) -> np.ndarray`**
- `origins`: Array of ray origins, shape (H, W, 3) or (N, 3)
- `directions`: Array of ray directions (normalized), shape (H, W, 3) or (N, 3)
- Returns: Array of accumulated distances, shape matches input (H, W) or (N,)

**`set_voxel_grid(voxel_grid: np.ndarray)`**
- Update the voxel grid without recreating the tracer

**`get_grid_info() -> dict`**
- Returns information about the current voxel grid (resolution, bounds, etc.)

## Implementation Details

### Ray Marching Algorithm

The ray tracer uses a DDA (Digital Differential Analyzer) algorithm for efficient voxel grid traversal:

1. Ray enters the voxel grid bounding box
2. DDA algorithm steps through voxels along the ray path
3. For each occupied voxel, accumulate the distance traveled
4. Continue until ray exits the grid or reaches max distance

### OptiX Pipeline

- **Ray Generation**: Launches rays for each pixel/sample
- **Intersection**: Custom intersection program for voxel grid
- **Closest Hit**: Accumulates distance through occupied voxels
- **Miss**: Returns 0 for rays that don't intersect the grid

### Memory Layout

Voxel grids are uploaded to GPU memory as:
- 3D CUDA arrays for texture sampling (optimized memory access)
- Metadata stored in constant memory (resolution, bounds)

## Performance Tips

1. **Batch Processing**: Trace many rays at once for better GPU utilization
2. **Resolution**: Higher voxel resolutions increase memory but improve accuracy
3. **Grid Caching**: Reuse VoxelRayTracer instances for the same grid

## Troubleshooting

**OptiX SDK not found**
```bash
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.x.x
```

**CUDA not found**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Python module not found**
```bash
export PYTHONPATH=/path/to/optix_raytracer/build:$PYTHONPATH
```

## Dataset Generation

The `examples/` directory includes scripts for generating ray tracing datasets:

### Single Object Dataset

Generate ray tracing data for a single object at the top level:

```bash
python examples/generate_dataset.py \
    --dataset-dir dataset \
    --object-id 0001 \
    --sampling-mode sphere \
    --sphere-divisions 4 \
    --samples-per-view 3000
```

**Options:**
- `--sampling-mode`: `sphere` (spherical viewpoints) or `all_faces` (uniform sampling on all bbox faces)
- `--sphere-divisions`: Number of azimuth/elevation divisions (default: 4 = 16 viewpoints)
- `--samples-per-view`: Rays per viewpoint distributed across visible faces
- `--level`: Voxel grid level to use (default: 0)

**Output:**
- `ray_dataset/object_XXXX_level_Y_rays.npz`: Ray tracing data
- Includes: origins, directions, distances, hits, face_ids, view_ids, view_positions
- Visualization PNG showing ray sampling pattern

### Hierarchical Dataset

Generate ray tracing data for all subvolumes in the hierarchy:

```bash
python examples/generate_hierarchical_dataset.py \
    --dataset-dir dataset \
    --output-dir ray_dataset_hierarchical \
    --sphere-divisions 4 \
    --samples-per-view 3000 \
    --skip-empty
```

**Options:**
- `--object-ids`: Specific objects to process (default: all objects)
- `--min-level`, `--max-level`: Filter by hierarchy level
- `--skip-empty`: Skip empty subvolumes (recommended)
- `--adaptive-sampling`: Halve sampling density with each level (level 0: N, level 1: N/2, level 2: N/4, etc.)
- `--min-samples`: Minimum samples per view when using adaptive sampling (default: 100)

**Adaptive Sampling Example:**
```bash
# Use adaptive sampling: level 0 gets 3000 samples, level 1 gets 1500, level 2 gets 750, etc.
python examples/generate_hierarchical_dataset.py \
    --dataset-dir dataset \
    --samples-per-view 3000 \
    --adaptive-sampling \
    --min-samples 100
```

This is useful for balancing quality and processing time - higher resolution levels get more samples, while deeper levels (smaller subvolumes) get proportionally fewer samples.

**Output Structure:**
```
ray_dataset_hierarchical/
├── level_0/
│   ├── ab/
│   │   ├── abcd1234...._rays.npz
│   │   └── abef5678...._rays.npz
│   └── cd/
│       └── cdef9012...._rays.npz
├── level_1/
│   └── ...
└── processing_summary.json
```

Each `.npz` file contains the same fields as single object datasets, organized by level and hash prefix for efficient storage.

## Examples

See the `examples/` directory:
- `basic_tracing.py`: Simple ray tracing example
- `generate_dataset.py`: Generate ray dataset for single object
- `generate_hierarchical_dataset.py`: Generate datasets for all subvolumes
- `render_dataset.py`: Render voxel objects from multiple viewpoints

## License

MIT License (same as parent project)

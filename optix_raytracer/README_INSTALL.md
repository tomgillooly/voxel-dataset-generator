# Quick Install into UV Environment

## One-Line Install (Recommended)

```bash
cd optix_raytracer && ./install.sh
```

This will:
1. Build the C++ extension with CMake
2. Install into your UV environment
3. Verify the installation

## Prerequisites

- OptiX SDK installed: `export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK`
- CUDA Toolkit installed
- UV package manager

## Manual Install

```bash
cd optix_raytracer

# Set OptiX path
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.x.x

# Build
./build.sh

# Install into UV environment
uv pip install -e .

# Verify
uv run python -c "import optix_voxel_tracer; print('Success!')"
```

## Usage After Installation

```python
# Use from anywhere in the project
import numpy as np
from optix_voxel_tracer import VoxelRayTracer

voxels = np.load("dataset/objects/object_0001/level_0.npz")['voxels']
tracer = VoxelRayTracer(voxels)
distances = tracer.trace_rays(origins, directions)
```

## Run Examples

```bash
# From project root
uv run python optix_raytracer/examples/basic_tracing.py
uv run python optix_raytracer/examples/render_dataset.py --dataset-dir dataset --object-id 0001
```

## Uninstall

```bash
uv pip uninstall optix-voxel-tracer
```

## Troubleshooting

See [INSTALL.md](INSTALL.md) for detailed installation guide and troubleshooting.

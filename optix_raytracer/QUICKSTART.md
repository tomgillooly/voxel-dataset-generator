# Quick Start Guide

## Prerequisites

1. **NVIDIA GPU** with OptiX support (RTX 20xx series or newer recommended)
2. **CUDA Toolkit** 11.0+ ([download](https://developer.nvidia.com/cuda-downloads))
3. **OptiX SDK** 7.0+ ([download](https://developer.nvidia.com/optix))
4. **CMake** 3.18+
5. **Python** 3.11+

## Installation (5 minutes)

### Step 1: Install OptiX SDK

```bash
# Download OptiX SDK from NVIDIA Developer site (requires free account)
# Extract to a location of your choice, then:
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.7.0

# Add to your shell profile (~/.bashrc or ~/.zshrc)
echo 'export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.7.0' >> ~/.bashrc
```

### Step 2: Build the Module

```bash
cd optix_raytracer
./build.sh
```

The build script will:
- Check for OptiX SDK and CUDA
- Compile CUDA programs to PTX
- Build C++ library
- Create Python bindings

### Step 3: Test Installation

```bash
# Add to Python path
export PYTHONPATH=$(pwd)/build:$PYTHONPATH

# Test import
python3 -c "import optix_voxel_tracer; print('Success!')"
```

## Basic Usage (2 minutes)

### Load and Trace

```python
import numpy as np
from optix_voxel_tracer import VoxelRayTracer

# Load voxel grid from your dataset
voxels = np.load("dataset/objects/object_0001/level_0.npz")['voxels']

# Create tracer
tracer = VoxelRayTracer(voxels, voxel_size=1.0)

# Define rays (orthographic view from above)
resolution = (512, 512)
origins = np.zeros((*resolution, 3), dtype=np.float32)
directions = np.zeros((*resolution, 3), dtype=np.float32)

for i in range(resolution[0]):
    for j in range(resolution[1]):
        x = (i / resolution[0]) * 2 - 1
        y = (j / resolution[1]) * 2 - 1
        origins[i, j] = [x, y, 2.0]      # Above the object
        directions[i, j] = [0, 0, -1]    # Pointing down

# Trace rays
distances = tracer.trace_rays(origins, directions)

# distances[i,j] = total distance ray traveled through occupied voxels
print(f"Max distance through object: {distances.max():.2f}")
```

### Visualize Results

```python
import matplotlib.pyplot as plt

plt.imshow(distances, cmap='viridis')
plt.colorbar(label='Distance through voxels')
plt.title('Ray Traced Distance Map')
plt.savefig('distance_map.png')
```

## Examples

The `examples/` directory contains three ready-to-run scripts:

### 1. Basic Tracing
```bash
cd examples
python3 basic_tracing.py
```
Creates a test sphere and renders it from orthographic views.

### 2. Render Dataset Objects
```bash
python3 render_dataset.py --dataset-dir ../dataset --object-id 0001
```
Renders turntable views of a specific object.

### 3. Batch Processing
```bash
python3 batch_process.py --dataset-dir ../dataset --max-objects 10
```
Processes multiple objects in batch.

## Common Tasks

### Change Viewpoint

```python
# Perspective camera
camera_pos = [2, 2, 2]
look_at = [0, 0, 0]
up = [0, 0, 1]

# ... use generate_camera_rays() from examples
```

### Process Multiple Objects

```python
from pathlib import Path

dataset_dir = Path("dataset/objects")
for obj_dir in dataset_dir.iterdir():
    voxels = np.load(obj_dir / "level_0.npz")['voxels']
    tracer = VoxelRayTracer(voxels)
    # ... trace rays ...
```

### Save Results

```python
# Save as numpy array
np.savez_compressed('output.npz', distances=distances)

# Or as image
from PIL import Image
img = (distances / distances.max() * 255).astype(np.uint8)
Image.fromarray(img).save('output.png')
```

## Performance Tips

1. **Batch rays together**: Process many rays at once for better GPU utilization
2. **Reuse tracer**: Create one tracer per voxel grid, use it for multiple ray batches
3. **Resolution tradeoff**: Higher voxel resolution = more accuracy but more memory
4. **Ray organization**: Coherent rays (similar directions) trace faster

## Troubleshooting

### "OptiX SDK not found"
```bash
export OptiX_INSTALL_DIR=/correct/path/to/optix
```

### "CUDA not found"
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### "Module not found"
```bash
export PYTHONPATH=/path/to/optix_raytracer/build:$PYTHONPATH
```

### GPU out of memory
- Reduce voxel resolution
- Process fewer rays at once
- Use smaller batches

## Next Steps

- See [README.md](README.md) for full API documentation
- Check [examples/](examples/) for more usage patterns
- Integrate with your existing voxel dataset pipeline

## Need Help?

- OptiX Documentation: https://raytracing-docs.nvidia.com/optix7/
- CUDA Documentation: https://docs.nvidia.com/cuda/
- Issues: File in parent project repository

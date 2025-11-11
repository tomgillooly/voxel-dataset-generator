# Installation Guide for OptiX Voxel Ray Tracer

## Prerequisites

1. **OptiX SDK** 7.0+ installed
2. **CUDA Toolkit** 11.0+ installed
3. **CMake** 3.18+ installed
4. **NVIDIA GPU** with OptiX support

## Installation Methods

### Method 1: Install into UV Environment (Recommended for Development)

This installs the module into your main project's UV environment.

```bash
# Set OptiX SDK path
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.x.x

# From project root
cd optix_raytracer

# Install in development mode (editable)
uv pip install -e .
```

**Advantages:**
- Module available in your main UV environment
- Changes to C++ code require rebuild but Python is editable
- Integrates with main project dependencies

**Usage after installation:**
```bash
# From anywhere in the project
uv run python -c "import optix_voxel_tracer; print('Success!')"

# Run examples
uv run python optix_raytracer/examples/basic_tracing.py
```

### Method 2: Manual Build and PYTHONPATH

Build manually and add to PYTHONPATH (no installation).

```bash
cd optix_raytracer
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.x.x
./build.sh

# Add to Python path
export PYTHONPATH=$(pwd)/build:$PYTHONPATH

# Test
python3 -c "import optix_voxel_tracer; print('Success!')"
```

**Advantages:**
- No installation needed
- Quick for testing
- Easy to clean up

**Disadvantages:**
- Must set PYTHONPATH every time
- Not integrated with UV environment

### Method 3: System-wide Installation

Install for all users (requires sudo).

```bash
cd optix_raytracer
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.x.x

# Build
./build.sh

# Install (requires sudo)
cd build
sudo make install
```

**Advantages:**
- Available system-wide
- No PYTHONPATH needed

**Disadvantages:**
- Requires sudo
- Harder to uninstall
- May conflict with other versions

## Detailed: Installing into UV Environment

### Step 1: Build with CMake

First, build the C++ extension:

```bash
cd optix_raytracer
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.x.x

# Create build directory
mkdir -p build
cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DOptiX_INSTALL_DIR="$OptiX_INSTALL_DIR"

# Build
make -j$(nproc)

cd ..
```

### Step 2: Install with UV

```bash
# From optix_raytracer directory
uv pip install -e .
```

This will:
- Install the package in editable mode
- Make it available to UV environment
- Add dependencies (numpy)

### Step 3: Verify Installation

```bash
# Test import
uv run python -c "import optix_voxel_tracer; print('Installed successfully!')"

# Get module info
uv run python -c "import optix_voxel_tracer; print(optix_voxel_tracer.__file__)"

# Run example
uv run python examples/basic_tracing.py
```

## Using with Main Project

After installation, you can use the ray tracer from anywhere in your project:

```python
# In your main project scripts
from optix_voxel_tracer import VoxelRayTracer
import numpy as np

# Load voxel grid from dataset
voxels = np.load("dataset/objects/object_0001/level_0.npz")['voxels']

# Create tracer
tracer = VoxelRayTracer(voxels)

# Trace rays
distances = tracer.trace_rays(origins, directions)
```

## Rebuilding After Changes

If you modify C++ code:

```bash
cd optix_raytracer/build
make -j$(nproc)

# Module automatically updated (editable install)
```

If you modify Python bindings:

```bash
cd optix_raytracer
uv pip install -e . --force-reinstall --no-deps
```

## Uninstalling

### From UV environment:
```bash
uv pip uninstall optix-voxel-tracer
```

### System-wide:
```bash
cd optix_raytracer/build
sudo make uninstall
```

## Troubleshooting

### "OptiX_INSTALL_DIR not set"

```bash
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.x.x
# Add to ~/.bashrc or ~/.zshrc for persistence
```

### "CMake not found"

```bash
# Ubuntu/Debian
sudo apt install cmake

# macOS
brew install cmake

# Or use conda/mamba
conda install cmake
```

### "Module not found after installation"

```bash
# Check installation
uv pip list | grep optix

# Reinstall
uv pip uninstall optix-voxel-tracer
uv pip install -e optix_raytracer/
```

### "Cannot find liboptix_voxel_tracer_core.so"

The shared library should be in `build/`. Make sure:
1. Build completed successfully
2. You're using editable install (`-e` flag)
3. LD_LIBRARY_PATH includes build directory (usually automatic)

```bash
# If needed, add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/optix_raytracer/build
```

## Quick Reference

| Task | Command |
|------|---------|
| Install (dev mode) | `uv pip install -e optix_raytracer/` |
| Uninstall | `uv pip uninstall optix-voxel-tracer` |
| Rebuild C++ | `cd optix_raytracer/build && make -j$(nproc)` |
| Test import | `uv run python -c "import optix_voxel_tracer"` |
| Run example | `uv run python optix_raytracer/examples/basic_tracing.py` |
| Clean build | `rm -rf optix_raytracer/build` |

## Environment Variables

Set these in your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# Required
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.x.x

# Optional (usually set by CUDA installer)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Integration with Main Project pyproject.toml (Optional)

If you want to make OptiX an optional dependency of the main project:

```toml
# In main pyproject.toml
[project.optional-dependencies]
optix = [
    "optix-voxel-tracer @ file:///${PROJECT_ROOT}/optix_raytracer"
]
```

Then users can install with:
```bash
uv sync --extra optix
```

## Docker/Container Usage

For containerized environments:

```dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install OptiX SDK
COPY NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh /tmp/
RUN sh /tmp/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh --skip-license --prefix=/opt/optix

ENV OptiX_INSTALL_DIR=/opt/optix

# Install project
COPY . /workspace
WORKDIR /workspace
RUN pip install -e optix_raytracer/
```

## Next Steps

After installation, see:
- [QUICKSTART.md](QUICKSTART.md) - Usage examples
- [examples/](examples/) - Example scripts
- [../OPTIX_INTEGRATION.md](../OPTIX_INTEGRATION.md) - Integration with main pipeline

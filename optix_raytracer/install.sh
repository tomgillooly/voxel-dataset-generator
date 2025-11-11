#!/bin/bash
# Installation script for OptiX Voxel Ray Tracer into UV environment

set -e

echo "==================================="
echo "OptiX Voxel Ray Tracer Installer"
echo "==================================="

# Check for OptiX SDK
if [ -z "$OptiX_INSTALL_DIR" ]; then
    echo "ERROR: OptiX_INSTALL_DIR environment variable not set!"
    echo "Please set it to your OptiX SDK installation directory:"
    echo "  export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK-7.x.x"
    exit 1
fi

if [ ! -f "$OptiX_INSTALL_DIR/include/optix.h" ]; then
    echo "ERROR: OptiX headers not found at $OptiX_INSTALL_DIR/include"
    exit 1
fi

echo "Found OptiX SDK at: $OptiX_INSTALL_DIR"

# Check for UV
if ! command -v uv &> /dev/null; then
    echo "ERROR: UV not found. Please install UV first:"
    echo "  pip install uv"
    exit 1
fi

echo "Found UV: $(uv --version)"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build first
echo ""
echo "Step 1: Building C++ extension..."
if [ ! -f "./build.sh" ]; then
    echo "ERROR: build.sh not found in $SCRIPT_DIR"
    exit 1
fi

./build.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo ""
echo "Step 2: Installing into UV environment..."

# Install in editable mode
uv pip install -e .

if [ $? -ne 0 ]; then
    echo "ERROR: Installation failed"
    exit 1
fi

echo ""
echo "Step 3: Verifying installation..."

# Test import
uv run python -c "import optix_voxel_tracer; print('Module loaded successfully!')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "✓ Installation successful!"
    echo "==================================="
    echo ""
    echo "The module is now available in your UV environment."
    echo ""
    echo "Quick test:"
    echo "  uv run python -c 'from optix_voxel_tracer import VoxelRayTracer'"
    echo ""
    echo "Run examples:"
    echo "  cd examples"
    echo "  uv run python basic_tracing.py"
    echo ""
    echo "See INSTALL.md for more usage information."
else
    echo ""
    echo "==================================="
    echo "✗ Installation verification failed"
    echo "==================================="
    echo ""
    echo "The build succeeded but the module cannot be imported."
    echo "This might be a path or dependency issue."
    echo ""
    echo "Try manually:"
    echo "  uv run python -c 'import optix_voxel_tracer'"
    echo ""
    echo "See INSTALL.md for troubleshooting."
    exit 1
fi

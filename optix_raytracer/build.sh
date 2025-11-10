#!/bin/bash
# Build script for OptiX Voxel Ray Tracer

set -e  # Exit on error

echo "==================================="
echo "OptiX Voxel Ray Tracer Build Script"
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
    echo "Please check your OptiX_INSTALL_DIR setting"
    exit 1
fi

echo "Found OptiX SDK at: $OptiX_INSTALL_DIR"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA compiler (nvcc) not found in PATH"
    echo "Please install CUDA Toolkit or add it to your PATH"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "Found CUDA: $CUDA_VERSION"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake not found"
    echo "Please install CMake 3.18 or later"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
echo "Found CMake: $CMAKE_VERSION"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo ""
    read -p "Build directory exists. Clean and rebuild? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    fi
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo ""
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DOptiX_INSTALL_DIR="$OptiX_INSTALL_DIR"

# Build
echo ""
echo "Building..."
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
make -j$NPROC

echo ""
echo "==================================="
echo "Build completed successfully!"
echo "==================================="
echo ""
echo "To use the Python module, add to your PYTHONPATH:"
echo "  export PYTHONPATH=$(pwd):\$PYTHONPATH"
echo ""
echo "Or install system-wide:"
echo "  sudo make install"
echo ""
echo "Test the installation:"
echo "  python3 -c 'import optix_voxel_tracer; print(\"Success!\")'"
echo ""
echo "Run examples:"
echo "  cd ../examples"
echo "  python3 basic_tracing.py"

#!/bin/bash
# Setup script for modern GPUs (CUDA capability >= 7.0)
# V100, A100, RTX series, etc.

set -e

echo "Setting up environment for modern GPUs..."

# Switch to Python 3.12
echo "3.12" > .python-version

# Update pyproject.toml to use modern PyTorch and NumPy 2.x
echo "Configuring PyTorch 2.4+ for modern GPUs..."
sed -i 's/torch>=2.0.0,<2.1.0/torch>=2.4.0/' pyproject.toml
sed -i 's/torchvision>=0.15.0,<0.16.0/torchvision>=0.19.0/' pyproject.toml
sed -i 's/numpy>=1.24.0,<2.0/numpy>=2.0.0/' pyproject.toml

# Remove lock file and resync
echo "Regenerating lock file with PyTorch 2.4+..."
rm -f uv.lock
uv sync --extra neural-rendering

echo ""
echo "✓ Modern GPU environment setup complete!"
echo ""
echo "IMPORTANT: Load these modules before running:"
echo "  module load CUDA/12.0.0  # Or newer CUDA version"
echo "  module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1  # Or compatible NCCL version"
echo ""
echo "Then run: uv run python examples/benchmark_dataset.py"

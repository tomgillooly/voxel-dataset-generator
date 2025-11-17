#!/bin/bash
# Setup script for P100 GPUs (CUDA capability 6.0)
# This configures PyTorch 2.0.x which is the last version supporting sm_60

set -e

echo "Setting up environment for P100 GPUs..."

# Switch to Python 3.11 (required for PyTorch 2.0.x)
echo "3.11" > .python-version

# Update pyproject.toml to pin PyTorch 2.0.x and NumPy 1.x
echo "Configuring PyTorch 2.0.x for P100..."
sed -i 's/torch>=2.4.0/torch>=2.0.0,<2.1.0/' pyproject.toml
sed -i 's/torchvision>=0.19.0/torchvision>=0.15.0,<0.16.0/' pyproject.toml
sed -i 's/numpy>=2.0.0/numpy>=1.24.0,<2.0/' pyproject.toml
sed -i 's/numpy>=1.24.0,<3.0/numpy>=1.24.0,<2.0/' pyproject.toml

# Remove lock file and resync
echo "Regenerating lock file with PyTorch 2.0.x..."
rm -f uv.lock
uv sync --extra neural-rendering

echo ""
echo "✓ P100 environment setup complete!"
echo ""
echo "IMPORTANT: Load these modules before running:"
echo "  module load CUDA/11.7.0"
echo "  module load NCCL/2.16.2-GCCcore-12.2.0-CUDA-11.7.0"
echo ""
echo "Then run: uv run python examples/benchmark_dataset.py"

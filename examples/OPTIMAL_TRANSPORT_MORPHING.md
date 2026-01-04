# Optimal Transport Morphing Between Voxelized Objects

This directory contains scripts for performing optimal transport-based morphing between voxelized 3D objects.

## Overview

The morphing pipeline uses optimal transport (OT) to compute smooth deformations between two voxelized objects:

1. **Load objects**: Select two random top-level (level 0) voxelized objects
2. **Convert to point clouds**: Extract occupied voxel coordinates and assign uniform weights
3. **Compute OT plan**: Find optimal transport coupling between source and target distributions
4. **Interpolate**: Generate intermediate point clouds using barycentric projection
5. **Reconstruct**: Convert interpolated point clouds back to voxel grids

## Installation

Install the Python Optimal Transport library:

```bash
pip install POT
```

Or with conda:

```bash
conda install -c conda-forge pot
```

## Usage

### Basic Usage

Run with default parameters:

```bash
python examples/optimal_transport_morphing.py
```

This will:
- Load 2 random level 0 objects from your dataset
- Compute optimal transport using Sinkhorn algorithm
- Generate 10 interpolation steps
- Save results to `morphing_results/`

### Advanced Options

```bash
python examples/optimal_transport_morphing.py \
  --dataset-dir dataset \
  --ray-dataset-dir ray_dataset_hierarchical \
  --output-dir my_morphing_results \
  --num-steps 20 \
  --method emd \
  --sigma 0.7 \
  --seed 42
```

**Parameters**:

- `--dataset-dir`: Path to voxel dataset (default: `dataset`)
- `--output-dir`: Output directory for results (default: `morphing_results`)
- `--num-steps`: Number of interpolation steps including endpoints (default: 10)
- `--method`: OT solver method
  - `sinkhorn`: Fast, regularized OT (recommended for >10K voxels)
  - `emd`: Exact Earth Mover's Distance (slower but exact)
- `--sigma`: Gaussian splatting radius for voxel reconstruction (default: 0.5)
  - Larger values create smoother interpolations
  - Smaller values preserve sharper features
- `--split-and-merge`: Enable splitting mode (flag, default: off)
  - Source points that map to multiple targets will split and travel separately
  - Creates more dynamic morphing with visible splitting/merging effects
- `--split-threshold`: Minimum transport weight to trigger splitting (default: 0.1)
  - Only used when `--split-and-merge` is enabled
  - Lower values = more splitting (more points created)
  - Higher values = less splitting (more conservative)
- `--seed`: Random seed for reproducibility (default: 42)

### Visualization

After generating a morphing sequence, visualize it:

```bash
python examples/visualize_morphing.py \
  --input-dir morphing_results \
  --source-id 0000 \
  --target-id 0042 \
  --output-dir morphing_visualizations
```

**Parameters**:

- `--input-dir`: Directory containing morphing results
- `--source-id`: Source object ID (from morphing output)
- `--target-id`: Target object ID (from morphing output)
- `--output-dir`: Where to save visualizations (optional, shows interactively if not set)
- `--num-cols`: Columns in grid visualization (default: 5)
- `--slice-axis`: Axis for 2D slices (0=Z, 1=Y, 2=X, default: 0)

**Output**:

- `morph_*_grid.png`: 3D visualization grid of all steps
- `morph_*_slices.png`: 2D slice comparison across steps
- `morph_*_occupancy.png`: Occupancy ratio plot
- `morph_*_volume.png`: Occupied voxel count histogram

## Output Format

### Morphing Sequence Files

The script saves each interpolation step as:

```
morphing_results/
├── morph_{source_id}_to_{target_id}_step_000.npz  # Source (t=0.0)
├── morph_{source_id}_to_{target_id}_step_001.npz  # t=0.1
├── ...
├── morph_{source_id}_to_{target_id}_step_009.npz  # Target (t=1.0)
└── morph_{source_id}_to_{target_id}_metadata.npz  # Metadata
```

Each `.npz` file contains:
- `voxels`: (D, H, W) boolean array representing the voxel grid

### Loading Results

```python
import numpy as np

# Load a single step
data = np.load('morphing_results/morph_0000_to_0042_step_005.npz')
voxels = data['voxels']  # Shape: (128, 128, 128)

# Load metadata
metadata = np.load('morphing_results/morph_0000_to_0042_metadata.npz')
source_id = str(metadata['source_id'])
target_id = str(metadata['target_id'])
num_steps = int(metadata['num_steps'])
grid_shape = list(metadata['grid_shape'])
```

## Interpolation Modes

### Standard Mode (Default)

Uses **barycentric projection** where each source point moves toward the weighted centroid of its target points:

```
For each source point sᵢ:
  centroid = Σⱼ (weight_ij) × target_j
  interpolated_point(t) = (1-t) × sᵢ + t × centroid
```

**Characteristics:**
- Fixed number of points (= source points)
- Each point follows a straight line
- Smooth, blob-like morphing
- Mass stays together

### Split-and-Merge Mode (`--split-and-merge`)

Creates **separate trajectories** for each significant source→target connection:

```
For each source point sᵢ:
  For each target tⱼ where weight_ij > threshold:
    Create copy of sᵢ traveling to tⱼ
    interpolated_point(t) = (1-t) × sᵢ + t × tⱼ
```

**Characteristics:**
- Variable number of points (can increase significantly)
- Points visibly split apart during morphing
- More dynamic, fluid transformations
- Better preserves topology changes
- More computationally expensive

**When to use:**
- Objects with very different topology (splitting/merging parts)
- Wanting to visualize mass flow
- Artistic/dramatic morphing effects
- Objects where parts should separate then recombine

**Parameters:**
- `--split-threshold 0.05`: Aggressive splitting (many points)
- `--split-threshold 0.1`: Balanced (default, moderate splitting)
- `--split-threshold 0.2`: Conservative (minimal splitting)

## Algorithm Details

### Optimal Transport

The script uses the POT (Python Optimal Transport) library to solve:

```
min_{π ∈ Π(μ, ν)} ⟨π, C⟩
```

where:
- μ = uniform distribution over source voxels
- ν = uniform distribution over target voxels
- C = pairwise Euclidean distance matrix
- π = optimal transport plan (coupling matrix)

**Methods**:

1. **EMD (Exact)**: Solves exact linear program, slow for large problems O(N³)
2. **Sinkhorn**: Entropic regularization, fast approximation O(N²) iterations

### Interpolation

Uses **barycentric projection** (displacement interpolation):

For each source point sᵢ at time t ∈ [0,1]:

```
p(t) = (1-t)·sᵢ + t·(Σⱼ πᵢⱼ/Σⱼπᵢⱼ · tⱼ)
```

where tⱼ are target points and π is the transport plan.

### Reconstruction

Converts interpolated point clouds back to voxel grids using **Gaussian splatting**:

```
V(x,y,z) = Σᵢ exp(-||p - pᵢ||² / 2σ²)
```

Then threshold at 0.5 to obtain binary voxels.

## Tips

### Performance

- Use `--method sinkhorn` for objects with >10,000 occupied voxels
- Use `--method emd` for small objects or when exact transport is needed
- Reduce `--num-steps` if computation is slow

### Quality

- Increase `--num-steps` for smoother trajectories (20-50 recommended)
- Adjust `--sigma`:
  - Small (0.3-0.5): Sharp, preserves fine details, may be fragmented
  - Medium (0.5-1.0): Balanced (recommended)
  - Large (1.0-2.0): Smooth, may blur features
- Different `--seed` values select different random object pairs

### Troubleshooting

**Out of memory**: Large objects (>50K voxels) with EMD can exhaust memory
- Solution: Use `--method sinkhorn`

**Fragmented interpolations**: Voxels appear disconnected in middle steps
- Solution: Increase `--sigma` (try 0.8-1.2)

**Blurry results**: Loss of sharp features
- Solution: Decrease `--sigma` (try 0.3-0.5)

## Examples

### Quick test with fast method
```bash
python examples/optimal_transport_morphing.py \
  --num-steps 10 \
  --method sinkhorn \
  --seed 123
```

### High-quality morphing with many steps
```bash
python examples/optimal_transport_morphing.py \
  --num-steps 50 \
  --method emd \
  --sigma 0.6 \
  --seed 42
```

### Smooth, detailed reconstruction
```bash
python examples/optimal_transport_morphing.py \
  --num-steps 30 \
  --method sinkhorn \
  --sigma 1.0 \
  --seed 999
```

### Split-and-merge morphing (dynamic)
```bash
python examples/optimal_transport_morphing.py \
  --num-steps 20 \
  --method sinkhorn \
  --split-and-merge \
  --split-threshold 0.05 \
  --sigma 0.7
```

### Conservative split-and-merge
```bash
python examples/optimal_transport_morphing.py \
  --split-and-merge \
  --split-threshold 0.2 \
  --num-steps 30
```

## References

- **POT Library**: https://pythonot.github.io/
- **Optimal Transport**: Peyré, G., & Cuturi, M. (2019). "Computational Optimal Transport"
- **Displacement Interpolation**: Solomon et al. (2015). "Convolutional Wasserstein Distances"

## Limitations

1. **Topology changes**: OT preserves mass but not topology, so objects can split/merge during interpolation
2. **Uniform weights**: Assumes all voxels have equal importance (could be extended to weighted transport)
3. **Euclidean metric**: Uses L2 distance in voxel space (could use geodesic distances for better results)
4. **Binary voxels**: Output is thresholded to binary (could support continuous densities)

## Future Extensions

Potential improvements:
- Add support for weighted distributions (importance sampling)
- Implement partial OT for unbalanced masses
- Add geodesic interpolation in Wasserstein space
- Support for color/feature transport (not just geometry)
- GPU acceleration for large point clouds
- Interactive visualization with animation

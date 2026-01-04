#!/usr/bin/env python3
"""Optimal transport morphing between voxelized objects (pure numpy).

This script:
1. Loads two random top-level voxelized objects directly from disk
2. Computes optimal transport plan between their occupied voxels
3. Generates a trajectory of intermediate voxelized objects
4. Saves the morphing sequence

Dependencies: numpy, POT (Python Optimal Transport)
Install: pip install POT

Note: This script does NOT require PyTorch - it uses pure numpy.
"""

import argparse
from pathlib import Path
import numpy as np
from typing import Tuple, List, Optional
import warnings

try:
    import ot  # Python Optimal Transport library
except ImportError:
    print("ERROR: POT library not installed. Install with: pip install POT")
    exit(1)


def load_random_level0_objects(
    dataset_dir: Path,
    num_objects: int = 2,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, str]]:
    """Load random top-level voxelized objects directly from disk.

    Args:
        dataset_dir: Path to voxel dataset
        num_objects: Number of objects to load
        seed: Random seed for reproducibility

    Returns:
        List of (voxel_grid, object_id) tuples
    """
    print(f"Loading {num_objects} random level 0 objects...")

    # Find all level 0 objects
    objects_dir = dataset_dir / "objects"
    if not objects_dir.exists():
        raise FileNotFoundError(f"Objects directory not found: {objects_dir}")

    object_dirs = sorted([d for d in objects_dir.iterdir() if d.is_dir()])

    if len(object_dirs) < num_objects:
        raise ValueError(f"Dataset has only {len(object_dirs)} objects, need {num_objects}")

    # Sample random objects
    if seed is not None:
        np.random.seed(seed)

    # Convert to list for indexing
    indices = np.random.choice(len(object_dirs), size=num_objects, replace=False)
    selected_dirs = [object_dirs[i] for i in indices]

    objects = []
    for obj_dir in selected_dirs:
        # Load voxel grid from level_0.npz
        voxel_file = obj_dir / "level_0.npz"
        if not voxel_file.exists():
            print(f"  Warning: {voxel_file} not found, skipping")
            continue

        data = np.load(voxel_file)
        voxels = data['voxels']  # Shape: (D, H, W)

        # Add batch dimension for consistency: (1, D, H, W)
        if voxels.ndim == 3:
            voxels = voxels[np.newaxis, :]

        object_id = obj_dir.name.replace("object_", "")

        print(f"  Loaded object {object_id}: shape={voxels.shape}, "
              f"occupancy={voxels.mean():.2%}")

        objects.append((voxels, object_id))

    return objects


def voxel_grid_to_point_cloud(voxels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert voxel grid to point cloud representation.

    Args:
        voxels: Boolean voxel grid, shape (1, D, H, W) or (D, H, W)

    Returns:
        points: (N, 3) array of occupied voxel coordinates
        weights: (N,) array of uniform weights (1/N for each point)
    """
    # Remove batch dimension if present
    if voxels.ndim == 4:
        voxels = voxels.squeeze(0)  # (D, H, W)

    # Get occupied voxel coordinates
    occupied_indices = np.where(voxels > 0.5)
    points = np.stack(occupied_indices, axis=1).astype(np.float32)  # (N, 3)

    # Uniform weights (probability distribution)
    num_points = points.shape[0]
    weights = np.ones(num_points) / num_points

    return points, weights


def compute_optimal_transport(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_weights: np.ndarray,
    target_weights: np.ndarray,
    method: str = 'emd',
) -> np.ndarray:
    """Compute optimal transport plan between two point clouds.

    Args:
        source_points: (N, 3) source point coordinates
        target_points: (M, 3) target point coordinates
        source_weights: (N,) source probability distribution
        target_weights: (M,) target probability distribution
        method: 'emd' (exact) or 'sinkhorn' (regularized, faster)

    Returns:
        transport_plan: (N, M) optimal transport coupling matrix
    """
    print(f"\nComputing optimal transport ({method})...")
    print(f"  Source: {len(source_points)} points")
    print(f"  Target: {len(target_points)} points")

    # Compute pairwise distance matrix
    print("  Computing distance matrix...")
    cost_matrix = ot.dist(source_points, target_points, metric='euclidean')

    # Normalize cost matrix to improve numerical stability
    max_cost = cost_matrix.max()
    # if max_cost > 0:
    #     cost_matrix = cost_matrix / max_cost
    #     print(f"  Normalized cost matrix (max was {max_cost:.2f})")

    # Compute optimal transport
    if method == 'emd':
        print("  Solving exact EMD (may be slow for large point clouds)...")
        transport_plan = ot.emd(source_weights, target_weights, cost_matrix)
    elif method == 'sinkhorn':
        print("  Solving Sinkhorn (regularized OT)...")
        # Adaptive regularization based on problem size
        # Larger reg = more stable but less accurate
        reg = 0.1  # Increased from 0.01 for better stability

        try:
            transport_plan = ot.sinkhorn(
                source_weights,
                target_weights,
                cost_matrix,
                reg=reg,
                numItermax=1000,
                stopThr=1e-6,
                warn=False  # Suppress warnings
            )
        except Exception:
            print(f"  Sinkhorn failed, trying with higher regularization...")
            reg = 0.5
            transport_plan = ot.sinkhorn(
                source_weights,
                target_weights,
                cost_matrix,
                reg=reg,
                numItermax=2000,
                stopThr=1e-5,
                warn=False
            )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Report transport cost
    transport_cost = np.sum(transport_plan * cost_matrix)
    print(f"  Transport cost: {transport_cost:.4f}")
    print(f"  Plan sparsity: {np.mean(transport_plan > 1e-6):.2%}")

    # Check for numerical issues
    if np.any(np.isnan(transport_plan)) or np.any(np.isinf(transport_plan)):
        print("  WARNING: NaN or Inf values detected in transport plan!")
        print("  Falling back to uniform transport...")
        # Create uniform transport as fallback
        transport_plan = np.outer(source_weights, target_weights)

    return transport_plan


def interpolate_via_barycentric_projection(
    source_points: np.ndarray,
    target_points: np.ndarray,
    transport_plan: np.ndarray,
    t: float,
    split_and_merge: bool = False,
    split_threshold: float = 0.1,
) -> np.ndarray:
    """Interpolate between source and target using barycentric projection.

    This is the displacement interpolation method from optimal transport.

    Args:
        source_points: (N, 3) source coordinates
        target_points: (M, 3) target coordinates
        transport_plan: (N, M) transport coupling
        t: Interpolation parameter in [0, 1], where 0=source, 1=target
        split_and_merge: If True, split source points that map to multiple targets
        split_threshold: Minimum transport weight to consider for splitting

    Returns:
        interpolated_points: (N, 3) or (N', 3) interpolated coordinates
                            where N' >= N if split_and_merge=True
    """
    if not split_and_merge:
        # Original barycentric interpolation
        # For each source point, compute its barycentric target
        # barycentric_target[i] = sum_j (transport_plan[i,j] / row_sum[i]) * target_points[j]
        row_sums = transport_plan.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero

        # Normalized transport plan (conditional distribution)
        normalized_plan = transport_plan / row_sums

        # Compute barycentric targets for each source point
        barycentric_targets = normalized_plan @ target_points  # (N, 3)

        # Linear interpolation between source and barycentric target
        interpolated_points = (1 - t) * source_points + t * barycentric_targets

        return interpolated_points

    else:
        # Split-and-merge interpolation
        # For each source point, create separate trajectories to each significant target
        interpolated_points_list = []

        for i, source_pt in enumerate(source_points):
            # Find all targets this source maps to with significant weight
            row = transport_plan[i, :]
            row_norm = row / (row.sum() + 1e-10)
            significant_targets = np.where(row_norm > split_threshold)[0]

            if len(significant_targets) == 0:
                # No significant targets, keep at source
                interpolated_points_list.append(source_pt)
            else:
                # Create a copy for each significant target
                for j in significant_targets:
                    target_pt = target_points[j]
                    # Interpolate from source to this specific target
                    interp_pt = (1 - t) * source_pt + t * target_pt
                    interpolated_points_list.append(interp_pt)

        interpolated_points = np.array(interpolated_points_list)

        return interpolated_points


def point_cloud_to_voxel_grid(
    points: np.ndarray,
    grid_shape: Tuple[int, ...],
    sigma: float = 0.5,
) -> np.ndarray:
    """Convert point cloud back to voxel grid using Gaussian splatting.

    Args:
        points: (N, 3) point coordinates
        grid_shape: Target voxel grid shape (D, H, W)
        sigma: Gaussian kernel standard deviation for splatting

    Returns:
        voxels: (1, D, H, W) voxel grid
    """
    # Round points to nearest voxel
    voxel_indices = np.round(points).astype(np.int32)

    # Clip to valid range
    for dim in range(3):
        voxel_indices[:, dim] = np.clip(voxel_indices[:, dim], 0, grid_shape[dim] - 1)

    # Create empty grid
    voxels = np.zeros(grid_shape, dtype=np.float32)

    # Splat points with Gaussian kernel
    if sigma > 0:
        # Use soft splatting with Gaussian weights
        for i in range(len(points)):
            center = points[i]
            voxel_center = voxel_indices[i]

            # Define kernel support region (3 sigma)
            radius = int(np.ceil(3 * sigma))
            for dz in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        # Neighbor voxel
                        nx = voxel_center[0] + dz
                        ny = voxel_center[1] + dy
                        nz = voxel_center[2] + dx

                        # Check bounds
                        if not (0 <= nx < grid_shape[0] and
                               0 <= ny < grid_shape[1] and
                               0 <= nz < grid_shape[2]):
                            continue

                        # Compute Gaussian weight
                        neighbor_pos = np.array([nx, ny, nz], dtype=np.float32)
                        dist_sq = np.sum((center - neighbor_pos) ** 2)
                        weight = np.exp(-dist_sq / (2 * sigma ** 2))

                        voxels[nx, ny, nz] += weight
    else:
        # Hard assignment (no splatting)
        for i in range(len(voxel_indices)):
            idx = tuple(voxel_indices[i])
            voxels[idx] = 1.0

    # Normalize and threshold
    if voxels.max() > 0:
        voxels = voxels / voxels.max()
    voxels = (voxels > 0.5).astype(np.float32)

    # Add batch dimension
    voxels = voxels[np.newaxis, :]  # (1, D, H, W)

    return voxels


def generate_morphing_sequence(
    source_voxels: np.ndarray,
    target_voxels: np.ndarray,
    num_steps: int = 10,
    method: str = 'emd',
    sigma: float = 0.5,
    split_and_merge: bool = False,
    split_threshold: float = 0.1,
) -> List[np.ndarray]:
    """Generate morphing sequence between two voxel grids.

    Args:
        source_voxels: (1, D, H, W) source voxel grid
        target_voxels: (1, D, H, W) target voxel grid
        num_steps: Number of intermediate steps (including endpoints)
        method: OT method ('emd' or 'sinkhorn')
        sigma: Gaussian splatting radius for reconstruction
        split_and_merge: If True, split source points traveling to multiple targets
        split_threshold: Minimum transport weight to trigger splitting (0-1)

    Returns:
        sequence: List of (1, D, H, W) voxel grids along trajectory
    """
    # Convert to point clouds
    print("Computing source point cloud")
    source_points, source_weights = voxel_grid_to_point_cloud(source_voxels)
    print("Computing target point cloud")
    target_points, target_weights = voxel_grid_to_point_cloud(target_voxels)

    print("Computing optimal transport")
    # Compute optimal transport
    transport_plan = compute_optimal_transport(
        source_points, target_points,
        source_weights, target_weights,
        method=method,
    )

    # Generate interpolated point clouds
    print(f"\nGenerating {num_steps} interpolation steps...")
    if split_and_merge:
        print(f"  Using split-and-merge mode (threshold={split_threshold})")
    grid_shape = tuple(source_voxels.squeeze(0).shape)  # (D, H, W)
    sequence = []

    t_values = np.linspace(0, 1, num_steps)
    for i, t in enumerate(t_values):
        print(f"  Step {i+1}/{num_steps} (t={t:.2f})...")

        # Interpolate points
        interp_points = interpolate_via_barycentric_projection(
            source_points, target_points, transport_plan, t,
            split_and_merge=split_and_merge,
            split_threshold=split_threshold
        )

        if split_and_merge:
            print(f"    Generated {len(interp_points)} points (from {len(source_points)} source)")

        # Convert back to voxel grid
        interp_voxels = point_cloud_to_voxel_grid(interp_points, grid_shape, sigma=sigma)
        sequence.append(interp_voxels)

    return sequence


def save_morphing_sequence(
    sequence: List[np.ndarray],
    output_dir: Path,
    source_id: str,
    target_id: str,
):
    """Save morphing sequence to disk.

    Args:
        sequence: List of voxel grids
        output_dir: Output directory
        source_id: Source object ID
        target_id: Target object ID
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving morphing sequence to {output_dir}...")

    for i, voxels in enumerate(sequence):
        # Save as NPZ
        filename = f"morph_{source_id}_to_{target_id}_step_{i:03d}.npz"
        filepath = output_dir / filename

        voxels_np = voxels.squeeze(0)  # (D, H, W)
        np.savez_compressed(filepath, voxels=voxels_np)

        occupancy = voxels_np.mean()
        print(f"  Saved {filename} (occupancy: {occupancy:.2%})")

    # Save metadata
    metadata = {
        'source_id': source_id,
        'target_id': target_id,
        'num_steps': len(sequence),
        'grid_shape': list(sequence[0].squeeze(0).shape),
    }

    metadata_file = output_dir / f"morph_{source_id}_to_{target_id}_metadata.npz"
    np.savez(metadata_file, **metadata)
    print(f"  Saved metadata to {metadata_file.name}")


def visualize_morphing_stats(sequence: List[np.ndarray]):
    """Print statistics about the morphing sequence.

    Args:
        sequence: List of voxel grids
    """
    print("\nMorphing sequence statistics:")
    print("  Step | Occupancy | Num voxels | Change from prev")
    print("  " + "-" * 55)

    prev_occupancy = None
    for i, voxels in enumerate(sequence):
        occupancy = float(voxels.mean())
        num_voxels = int(voxels.sum())

        change_str = ""
        if prev_occupancy is not None:
            change = occupancy - prev_occupancy
            change_str = f"{change:+.4f}"

        print(f"  {i:4d} | {occupancy:8.4%} | {int(num_voxels):10d} | {change_str}")
        prev_occupancy = occupancy


def main():
    parser = argparse.ArgumentParser(
        description="Optimal transport morphing between voxelized objects"
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('dataset'),
        help='Path to voxel dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('morphing_results'),
        help='Output directory for morphing sequence'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=10,
        help='Number of interpolation steps (including endpoints)'
    )
    parser.add_argument(
        '--method',
        choices=['emd', 'sinkhorn'],
        default='sinkhorn',
        help='Optimal transport method (emd=exact, sinkhorn=fast)'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.5,
        help='Gaussian splatting sigma for reconstruction'
    )
    parser.add_argument(
        '--split-and-merge',
        action='store_true',
        help='Enable splitting of source points that map to multiple targets'
    )
    parser.add_argument(
        '--split-threshold',
        type=float,
        default=0.1,
        help='Minimum transport weight to trigger splitting (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Optimal Transport Morphing between Voxelized Objects")
    print("=" * 70)

    # Load two random objects directly from disk
    objects = load_random_level0_objects(
        args.dataset_dir,
        num_objects=2,
        seed=args.seed,
    )

    source_voxels, source_id = objects[0]
    target_voxels, target_id = objects[1]

    # Generate morphing sequence
    print("\n" + "=" * 70)
    print("Computing optimal transport and generating morphing sequence")
    print("=" * 70)

    sequence = generate_morphing_sequence(
        source_voxels,
        target_voxels,
        num_steps=args.num_steps,
        method=args.method,
        sigma=args.sigma,
        split_and_merge=args.split_and_merge,
        split_threshold=args.split_threshold,
    )

    # Print statistics
    visualize_morphing_stats(sequence)

    # Save results
    save_morphing_sequence(
        sequence,
        args.output_dir,
        source_id,
        target_id,
    )

    print("\n" + "=" * 70)
    print("Morphing complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()

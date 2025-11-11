#!/usr/bin/env python3
"""
Generate ray tracing dataset from voxel objects.

Samples rays uniformly on the surface of the voxel grid's bounding box,
traces them through the grid, and saves the results with hit flags.
"""

import numpy as np
from pathlib import Path
import sys
import argparse

# Add build directory to path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent / "build"))

from optix_voxel_tracer import VoxelRayTracer


def sample_rays_on_bounding_box(grid_info, num_samples_per_face):
    """
    Generate rays starting from the surface of the voxel grid's bounding box.

    Samples are distributed uniformly across all 6 faces, with rays pointing
    inward toward the grid center.

    Args:
        grid_info: Dict with 'resolution' and 'voxel_size'
        num_samples_per_face: Number of ray samples per face

    Returns:
        origins: (N, 3) array of ray origins on bounding box surface
        directions: (N, 3) array of ray directions (pointing inward)
        face_ids: (N,) array indicating which face each ray came from (0-5)
    """
    res_z, res_y, res_x = grid_info['resolution']
    voxel_size = grid_info['voxel_size']

    # Compute bounding box (grid is centered at origin)
    half_x = res_x * voxel_size / 2.0
    half_y = res_y * voxel_size / 2.0
    half_z = res_z * voxel_size / 2.0

    bbox_min = np.array([-half_x, -half_y, -half_z], dtype=np.float32)
    bbox_max = np.array([half_x, half_y, half_z], dtype=np.float32)

    print(f"Bounding box: min={bbox_min}, max={bbox_max}")

    origins = []
    directions = []
    face_ids = []

    # Face 0: -X face (left)
    # Sample uniformly on the YZ plane at x = bbox_min[0]
    for _ in range(num_samples_per_face):
        y = np.random.uniform(bbox_min[1], bbox_max[1])
        z = np.random.uniform(bbox_min[2], bbox_max[2])
        origins.append([bbox_min[0], y, z])
        directions.append([1.0, 0.0, 0.0])  # Point inward (+X)
        face_ids.append(0)

    # Face 1: +X face (right)
    for _ in range(num_samples_per_face):
        y = np.random.uniform(bbox_min[1], bbox_max[1])
        z = np.random.uniform(bbox_min[2], bbox_max[2])
        origins.append([bbox_max[0], y, z])
        directions.append([-1.0, 0.0, 0.0])  # Point inward (-X)
        face_ids.append(1)

    # Face 2: -Y face (bottom)
    for _ in range(num_samples_per_face):
        x = np.random.uniform(bbox_min[0], bbox_max[0])
        z = np.random.uniform(bbox_min[2], bbox_max[2])
        origins.append([x, bbox_min[1], z])
        directions.append([0.0, 1.0, 0.0])  # Point inward (+Y)
        face_ids.append(2)

    # Face 3: +Y face (top)
    for _ in range(num_samples_per_face):
        x = np.random.uniform(bbox_min[0], bbox_max[0])
        z = np.random.uniform(bbox_min[2], bbox_max[2])
        origins.append([x, bbox_max[1], z])
        directions.append([0.0, -1.0, 0.0])  # Point inward (-Y)
        face_ids.append(3)

    # Face 4: -Z face (back)
    for _ in range(num_samples_per_face):
        x = np.random.uniform(bbox_min[0], bbox_max[0])
        y = np.random.uniform(bbox_min[1], bbox_max[1])
        origins.append([x, y, bbox_min[2]])
        directions.append([0.0, 0.0, 1.0])  # Point inward (+Z)
        face_ids.append(4)

    # Face 5: +Z face (front)
    for _ in range(num_samples_per_face):
        x = np.random.uniform(bbox_min[0], bbox_max[0])
        y = np.random.uniform(bbox_min[1], bbox_max[1])
        origins.append([x, y, bbox_max[2]])
        directions.append([0.0, 0.0, -1.0])  # Point inward (-Z)
        face_ids.append(5)

    return (np.array(origins, dtype=np.float32),
            np.array(directions, dtype=np.float32),
            np.array(face_ids, dtype=np.int32))


def main():
    parser = argparse.ArgumentParser(
        description="Generate ray tracing dataset from voxel objects"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--object-id",
        type=str,
        required=True,
        help="Object ID to process (e.g., '0001')"
    )
    parser.add_argument(
        "--samples-per-face",
        type=int,
        default=1000,
        help="Number of ray samples per bounding box face (6 faces total)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ray_dataset"),
        help="Output directory for ray tracing dataset"
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Voxel grid level to use (default: 0 = highest resolution)"
    )

    args = parser.parse_args()

    # Load voxel data
    object_dir = args.dataset_dir / "objects" / f"object_{args.object_id}"
    voxel_file = object_dir / f"level_{args.level}.npz"

    if not voxel_file.exists():
        print(f"ERROR: Voxel file not found: {voxel_file}")
        return 1

    print(f"Loading voxel grid from: {voxel_file}")
    data = np.load(voxel_file)
    voxels = data['voxels']

    print(f"Voxel grid shape: {voxels.shape}")
    print(f"Occupied voxels: {voxels.sum()} / {voxels.size} ({100*voxels.mean():.2f}%)")

    # Create ray tracer
    print("\nInitializing OptiX ray tracer...")
    tracer = VoxelRayTracer(voxels, voxel_size=1.0)

    if not tracer.is_ready():
        print("ERROR: Tracer not ready!")
        return 1

    grid_info = tracer.get_grid_info()
    print(f"Grid info: {grid_info}")

    # Generate rays on bounding box surface
    print(f"\nGenerating {args.samples_per_face * 6} rays on bounding box surface...")
    origins, directions, face_ids = sample_rays_on_bounding_box(
        grid_info, args.samples_per_face
    )

    print(f"Ray origins shape: {origins.shape}")
    print(f"Ray directions shape: {directions.shape}")

    # Trace rays
    print("\nTracing rays through voxel grid...")
    distances = tracer.trace_rays(origins, directions)

    # Create hit flags (distance > 0 means ray hit occupied voxels)
    hits = (distances > 0.0).astype(np.uint8)

    # Statistics
    num_hits = hits.sum()
    num_total = len(hits)
    hit_rate = 100.0 * num_hits / num_total

    print(f"\nRay tracing statistics:")
    print(f"  Total rays: {num_total}")
    print(f"  Hits: {num_hits} ({hit_rate:.1f}%)")
    print(f"  Misses: {num_total - num_hits} ({100-hit_rate:.1f}%)")
    print(f"  Min distance (hits only): {distances[hits > 0].min():.3f}")
    print(f"  Max distance (hits only): {distances[hits > 0].max():.3f}")
    print(f"  Mean distance (hits only): {distances[hits > 0].mean():.3f}")

    # Per-face statistics
    print("\nPer-face statistics:")
    face_names = ["-X", "+X", "-Y", "+Y", "-Z", "+Z"]
    for face_id in range(6):
        mask = face_ids == face_id
        face_hits = hits[mask].sum()
        face_total = mask.sum()
        face_hit_rate = 100.0 * face_hits / face_total
        print(f"  Face {face_id} ({face_names[face_id]}): "
              f"{face_hits}/{face_total} hits ({face_hit_rate:.1f}%)")

    # Save dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"object_{args.object_id}_level_{args.level}_rays.npz"

    print(f"\nSaving dataset to: {output_file}")
    np.savez_compressed(
        output_file,
        origins=origins,
        directions=directions,
        distances=distances,
        hits=hits,
        face_ids=face_ids,
        grid_shape=voxels.shape,
        voxel_size=grid_info['voxel_size'],
        object_id=args.object_id,
        level=args.level
    )

    print("\nDataset contents:")
    print(f"  origins: {origins.shape} - Ray starting points on bounding box")
    print(f"  directions: {directions.shape} - Ray directions (normalized)")
    print(f"  distances: {distances.shape} - Distance through occupied voxels")
    print(f"  hits: {hits.shape} - Binary hit flags (1=hit, 0=miss)")
    print(f"  face_ids: {face_ids.shape} - Which bbox face ray originated from (0-5)")
    print(f"  grid_shape: {voxels.shape} - Original voxel grid dimensions")
    print(f"  voxel_size: {grid_info['voxel_size']} - Size of each voxel")

    # Create visualization if matplotlib available
    try:
        import matplotlib.pyplot as plt

        print("\nCreating visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Visualize each face
        for face_id in range(6):
            row = face_id // 3
            col = face_id % 3
            ax = axes[row, col]

            mask = face_ids == face_id
            face_distances = distances[mask]

            # Create 2D histogram based on origin positions
            face_origins = origins[mask]

            # Determine which 2 coordinates to use based on face
            if face_id in [0, 1]:  # X faces - use Y, Z
                x_coord = face_origins[:, 1]  # Y
                y_coord = face_origins[:, 2]  # Z
                xlabel, ylabel = 'Y', 'Z'
            elif face_id in [2, 3]:  # Y faces - use X, Z
                x_coord = face_origins[:, 0]  # X
                y_coord = face_origins[:, 2]  # Z
                xlabel, ylabel = 'X', 'Z'
            else:  # Z faces - use X, Y
                x_coord = face_origins[:, 0]  # X
                y_coord = face_origins[:, 1]  # Y
                xlabel, ylabel = 'X', 'Y'

            # Scatter plot with distance as color
            scatter = ax.scatter(x_coord, y_coord, c=face_distances,
                               cmap='viridis', s=1, alpha=0.6)
            ax.set_title(f'Face {face_id} ({face_names[face_id]})')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax, label='Distance')

        plt.suptitle(f'Object {args.object_id} - Ray Tracing Dataset', fontsize=16)
        plt.tight_layout()

        viz_file = args.output_dir / f"object_{args.object_id}_level_{args.level}_rays.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {viz_file}")

    except ImportError:
        print("Matplotlib not available, skipping visualization")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

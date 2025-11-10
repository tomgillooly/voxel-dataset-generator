#!/usr/bin/env python3
"""
Render voxel objects from the dataset using OptiX ray tracing.
Generates distance-accumulation views from multiple angles.
"""

import numpy as np
from pathlib import Path
import sys
import argparse

# Add build directory to path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent / "build"))

from optix_voxel_tracer import VoxelRayTracer


def generate_camera_rays(resolution, camera_pos, look_at, up, fov=60.0):
    """
    Generate perspective camera rays.

    Args:
        resolution: (width, height) tuple
        camera_pos: Camera position [x, y, z]
        look_at: Point to look at [x, y, z]
        up: Up vector [x, y, z]
        fov: Field of view in degrees

    Returns:
        origins: (H, W, 3) array of ray origins
        directions: (H, W, 3) array of normalized ray directions
    """
    w, h = resolution

    # Camera coordinate system
    forward = np.array(look_at) - np.array(camera_pos)
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, np.array(up))
    right = right / np.linalg.norm(right)

    up_vec = np.cross(right, forward)
    up_vec = up_vec / np.linalg.norm(up_vec)

    # Field of view
    aspect = w / h
    fov_rad = np.radians(fov)

    half_height = np.tan(fov_rad / 2)
    half_width = aspect * half_height

    # Generate rays
    origins = np.zeros((h, w, 3), dtype=np.float32)
    directions = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            # Normalized device coordinates
            u = (j + 0.5) / w
            v = (i + 0.5) / h

            # Screen space coordinates
            screen_x = (2 * u - 1) * half_width
            screen_y = (1 - 2 * v) * half_height

            # Ray direction in world space
            direction = forward + screen_x * right + screen_y * up_vec
            direction = direction / np.linalg.norm(direction)

            origins[i, j] = camera_pos
            directions[i, j] = direction

    return origins, directions


def render_turntable(tracer, num_views=8, resolution=(512, 512), radius=2.0):
    """
    Render object from multiple viewpoints in a turntable fashion.

    Args:
        tracer: VoxelRayTracer instance
        num_views: Number of viewpoints around the object
        resolution: (width, height) for rendering
        radius: Distance from object center

    Returns:
        List of distance images
    """
    renders = []

    for i in range(num_views):
        angle = 2 * np.pi * i / num_views

        # Camera position on a circle
        camera_pos = [
            radius * np.cos(angle),
            radius * np.sin(angle),
            radius * 0.3  # Slight elevation
        ]

        look_at = [0, 0, 0]
        up = [0, 0, 1]

        print(f"Rendering view {i+1}/{num_views} (angle: {np.degrees(angle):.1f}°)...")

        origins, directions = generate_camera_rays(
            resolution, camera_pos, look_at, up
        )

        distances = tracer.trace_rays(origins, directions)
        renders.append(distances)

    return renders


def main():
    parser = argparse.ArgumentParser(
        description="Render voxel objects from dataset using OptiX"
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
        default="0000",
        help="Object ID to render (e.g., '0000')"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Render resolution (width height)"
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=8,
        help="Number of turntable views"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("renders"),
        help="Output directory for renders"
    )

    args = parser.parse_args()

    # Load voxel data
    object_dir = args.dataset_dir / "objects" / f"object_{args.object_id}"
    voxel_file = object_dir / "level_0.npz"

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

    print("Tracer ready!")

    # Render turntable
    print(f"\nRendering {args.num_views} views...")
    renders = render_turntable(
        tracer,
        num_views=args.num_views,
        resolution=tuple(args.resolution),
        radius=2.0
    )

    # Save renders
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving renders to: {args.output_dir}")
    for i, distances in enumerate(renders):
        output_file = args.output_dir / f"object_{args.object_id}_view_{i:02d}.npz"
        np.savez_compressed(output_file, distances=distances)
        print(f"  Saved view {i+1}: {output_file}")

    # Create visualization
    try:
        import matplotlib.pyplot as plt

        print("\nCreating visualization...")

        # Determine grid layout
        cols = min(4, args.num_views)
        rows = (args.num_views + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = np.atleast_2d(axes)

        for i, distances in enumerate(renders):
            row = i // cols
            col = i % cols

            ax = axes[row, col] if rows > 1 else axes[0, col]

            im = ax.imshow(distances, cmap='viridis')
            ax.set_title(f'View {i+1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for i in range(args.num_views, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[0, col]
            ax.axis('off')

        plt.suptitle(f'Object {args.object_id} - Distance Accumulation Renders',
                     fontsize=16)
        plt.tight_layout()

        viz_file = args.output_dir / f"object_{args.object_id}_turntable.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {viz_file}")

    except ImportError:
        print("Matplotlib not available, skipping visualization")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Basic example of ray tracing through voxel grids using OptiX.
"""

import numpy as np
from pathlib import Path
import sys

# Add build directory to path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent / "build"))

from optix_voxel_tracer import VoxelRayTracer


def generate_orthographic_rays(resolution, grid_size=64.0, distance=100.0):
    """
    Generate orthographic rays pointing down (-Z direction).

    Args:
        resolution: Tuple (width, height) for ray grid
        grid_size: Size of the ray grid in world space (should match voxel grid extent)
        distance: Distance from which to cast rays (should be outside grid bounds)

    Returns:
        origins: (H, W, 3) array of ray origins
        directions: (H, W, 3) array of ray directions
    """
    h, w = resolution

    # Create grid of ray origins spanning the voxel grid
    x = np.linspace(-grid_size/2, grid_size/2, w, dtype=np.float32)
    y = np.linspace(-grid_size/2, grid_size/2, h, dtype=np.float32)

    xx, yy = np.meshgrid(x, y)

    # Origins at fixed Z distance (above the grid)
    origins = np.zeros((h, w, 3), dtype=np.float32)
    origins[:, :, 0] = xx
    origins[:, :, 1] = yy
    origins[:, :, 2] = distance

    # All rays point down
    directions = np.zeros((h, w, 3), dtype=np.float32)
    directions[:, :, 2] = -1.0

    return origins, directions


def main():
    # Example 1: Simple sphere voxel grid
    print("Creating test voxel grid (sphere)...")
    resolution = 64
    voxels = np.zeros((resolution, resolution, resolution), dtype=np.uint8)

    # Create a sphere in the center
    center = resolution // 2
    radius = resolution // 4

    for z in range(resolution):
        for y in range(resolution):
            for x in range(resolution):
                dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
                if dist <= radius:
                    voxels[z, y, x] = 1

    print(f"Voxel grid shape: {voxels.shape}")
    print(f"Occupied voxels: {voxels.sum()} / {voxels.size}")

    # Create ray tracer
    print("\nInitializing OptiX ray tracer...")
    tracer = VoxelRayTracer(voxels, voxel_size=1.0)

    if not tracer.is_ready():
        print("ERROR: Tracer not ready!")
        return 1

    print("Tracer ready!")

    # Get grid info
    info = tracer.get_grid_info()
    print(f"Grid info: {info}")

    # Generate rays
    print("\nGenerating orthographic rays...")
    ray_resolution = (256, 256)
    origins, directions = generate_orthographic_rays(ray_resolution)

    print(f"Ray origins shape: {origins.shape}")
    print(f"Ray directions shape: {directions.shape}")

    # Trace rays
    print("\nTracing rays through voxel grid...")
    distances = tracer.trace_rays(origins, directions)

    print(f"Output distances shape: {distances.shape}")
    print(f"Min distance: {distances.min():.3f}")
    print(f"Max distance: {distances.max():.3f}")
    print(f"Mean distance: {distances.mean():.3f}")

    # Rays that hit the sphere
    hits = distances > 0
    print(f"Rays that hit object: {hits.sum()} / {hits.size} ({100*hits.mean():.1f}%)")

    # Visualize if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Distance map
        im1 = axes[0].imshow(distances, cmap='viridis')
        axes[0].set_title('Accumulated Distance Through Voxels')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0], label='Distance')

        # Binary hit map
        im2 = axes[1].imshow(hits, cmap='gray')
        axes[1].set_title('Ray Hit Map')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1], label='Hit')

        plt.tight_layout()
        plt.savefig('basic_tracing_result.png', dpi=150)
        print("\nSaved visualization to: basic_tracing_result.png")

    except ImportError:
        print("\nMatplotlib not available, skipping visualization")

    # Example 2: Trace from different viewpoint
    print("\n" + "="*60)
    print("Example 2: Side view")
    print("="*60)

    # Generate rays from the side (outside the grid which extends to +/-32)
    origins2 = np.zeros((256, 256, 3), dtype=np.float32)
    directions2 = np.zeros((256, 256, 3), dtype=np.float32)

    for i in range(256):
        for j in range(256):
            y = (i / 256.0) * 64 - 32  # -32 to +32
            z = (j / 256.0) * 64 - 32  # -32 to +32
            origins2[i, j] = [100.0, y, z]  # From +X side, outside grid
            directions2[i, j] = [-1.0, 0, 0]  # Point towards -X

    distances2 = tracer.trace_rays(origins2, directions2)

    print(f"Side view - Max distance: {distances2.max():.3f}")
    print(f"Side view - Rays hit: {(distances2 > 0).sum()} / {distances2.size}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

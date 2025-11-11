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


def sample_rays_sphere(grid_info, sphere_divisions, samples_per_view):
    """
    Generate rays from evenly distributed spherical viewpoints.

    Distributes viewpoints uniformly on a sphere using azimuth and elevation angles.
    For each viewpoint, samples rays on the visible bounding box faces.

    Args:
        grid_info: Dict with 'resolution' and 'voxel_size'
        sphere_divisions: Number of divisions for both azimuth and elevation
        samples_per_view: Total samples per viewpoint (distributed across visible faces)

    Returns:
        origins: (N, 3) array of ray origins on bounding box surface
        directions: (N, 3) array of ray directions (toward camera)
        face_ids: (N,) array indicating which face each ray came from (0-5)
        view_ids: (N,) array indicating which viewpoint each ray belongs to
        view_positions: (num_views, 3) array of camera positions in spherical sampling
    """
    res_z, res_y, res_x = grid_info['resolution']
    voxel_size = grid_info['voxel_size']

    # Compute bounding box (grid is centered at origin)
    half_x = res_x * voxel_size / 2.0
    half_y = res_y * voxel_size / 2.0
    half_z = res_z * voxel_size / 2.0

    bbox_min = np.array([-half_x, -half_y, -half_z], dtype=np.float32)
    bbox_max = np.array([half_x, half_y, half_z], dtype=np.float32)

    # Camera distance (outside the bounding box)
    max_extent = max(half_x, half_y, half_z)
    camera_radius = max_extent * 2.0

    print(f"Bounding box: min={bbox_min}, max={bbox_max}")
    print(f"Camera radius: {camera_radius:.1f}")
    print(f"Sphere sampling: {sphere_divisions} azimuth × {sphere_divisions} elevation = {sphere_divisions**2} viewpoints")

    origins = []
    directions = []
    face_ids = []
    view_ids = []
    view_positions = []

    samples_per_face = samples_per_view // 3  # Distribute across 3 visible faces

    # Calculate azimuth and elevation angles for even sphere coverage
    num_azimuth = sphere_divisions
    num_elevation = sphere_divisions

    # Elevation from 0 (north pole) to π (south pole), avoid exact poles
    elevations = np.linspace(0.1 * np.pi, 0.9 * np.pi, num_elevation)

    # Azimuth from 0 to 2π (full circle), don't include endpoint (0 = 2π)
    azimuths = np.linspace(0, 2 * np.pi, num_azimuth, endpoint=False)

    view_idx = 0
    for elev in elevations:
        for azim in azimuths:
            # Spherical to Cartesian conversion
            # x = r*sin(θ)*cos(φ), y = r*sin(θ)*sin(φ), z = r*cos(θ)
            # where θ is elevation (polar angle) and φ is azimuth
            camera_pos = np.array([
                camera_radius * np.sin(elev) * np.cos(azim),
                camera_radius * np.sin(elev) * np.sin(azim),
                camera_radius * np.cos(elev)
            ], dtype=np.float32)

            view_positions.append(camera_pos)

            # View direction (from camera toward origin)
            view_dir = -camera_pos / np.linalg.norm(camera_pos)

            # Determine which faces are visible from this viewpoint
            # Face is visible if its outward normal points toward the camera
            visible_faces = []

            # Face normals: -X=[−1,0,0], +X=[1,0,0], -Y=[0,−1,0], +Y=[0,1,0], -Z=[0,0,−1], +Z=[0,0,1]
            face_normals = [
                np.array([-1, 0, 0]),  # Face 0: -X
                np.array([1, 0, 0]),   # Face 1: +X
                np.array([0, -1, 0]),  # Face 2: -Y
                np.array([0, 1, 0]),   # Face 3: +Y
                np.array([0, 0, -1]),  # Face 4: -Z
                np.array([0, 0, 1])    # Face 5: +Z
            ]

            for face_id, normal in enumerate(face_normals):
                # Face is visible if normal points away from camera
                # (i.e., dot product with view direction is positive)
                if np.dot(normal, view_dir) > 0:
                    visible_faces.append(face_id)

            print(f"View {view_idx} (azim={np.degrees(azim):.1f}°, elev={np.degrees(elev):.1f}°): "
                  f"visible faces = {visible_faces}")

            # Sample rays on visible faces
            for face_id in visible_faces:
                for _ in range(samples_per_face):
                    # Sample point on this face
                    if face_id == 0:  # -X face
                        y = np.random.uniform(bbox_min[1], bbox_max[1])
                        z = np.random.uniform(bbox_min[2], bbox_max[2])
                        origin = np.array([bbox_min[0], y, z], dtype=np.float32)
                    elif face_id == 1:  # +X face
                        y = np.random.uniform(bbox_min[1], bbox_max[1])
                        z = np.random.uniform(bbox_min[2], bbox_max[2])
                        origin = np.array([bbox_max[0], y, z], dtype=np.float32)
                    elif face_id == 2:  # -Y face
                        x = np.random.uniform(bbox_min[0], bbox_max[0])
                        z = np.random.uniform(bbox_min[2], bbox_max[2])
                        origin = np.array([x, bbox_min[1], z], dtype=np.float32)
                    elif face_id == 3:  # +Y face
                        x = np.random.uniform(bbox_min[0], bbox_max[0])
                        z = np.random.uniform(bbox_min[2], bbox_max[2])
                        origin = np.array([x, bbox_max[1], z], dtype=np.float32)
                    elif face_id == 4:  # -Z face
                        x = np.random.uniform(bbox_min[0], bbox_max[0])
                        y = np.random.uniform(bbox_min[1], bbox_max[1])
                        origin = np.array([x, y, bbox_min[2]], dtype=np.float32)
                    else:  # face_id == 5, +Z face
                        x = np.random.uniform(bbox_min[0], bbox_max[0])
                        y = np.random.uniform(bbox_min[1], bbox_max[1])
                        origin = np.array([x, y, bbox_max[2]], dtype=np.float32)

                    # Ray direction: from surface point toward camera
                    direction = camera_pos - origin
                    direction = direction / np.linalg.norm(direction)

                    origins.append(origin)
                    directions.append(direction)
                    face_ids.append(face_id)
                    view_ids.append(view_idx)

            view_idx += 1

    return (np.array(origins, dtype=np.float32),
            np.array(directions, dtype=np.float32),
            np.array(face_ids, dtype=np.int32),
            np.array(view_ids, dtype=np.int32),
            np.array(view_positions, dtype=np.float32))


def sample_rays_on_bounding_box(grid_info, num_samples_per_face):
    """
    Generate rays starting from all faces of the voxel grid's bounding box.

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
        "--sampling-mode",
        type=str,
        choices=["sphere", "all_faces"],
        default="sphere",
        help="Ray sampling mode: sphere (visible faces from spherical viewpoints) or all_faces (uniform on all faces)"
    )
    parser.add_argument(
        "--sphere-divisions",
        type=int,
        default=4,
        help="Number of divisions for azimuth and elevation (sphere mode only). Total viewpoints = divisions²"
    )
    parser.add_argument(
        "--samples-per-view",
        type=int,
        default=3000,
        help="Number of ray samples per viewpoint (only for sphere mode, distributed across visible faces)"
    )
    parser.add_argument(
        "--samples-per-face",
        type=int,
        default=1000,
        help="Number of ray samples per bounding box face (only for all_faces mode, 6 faces total)"
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

    # Generate rays based on sampling mode
    if args.sampling_mode == "sphere":
        num_views = args.sphere_divisions ** 2
        print(f"\nGenerating sphere rays: {args.sphere_divisions}×{args.sphere_divisions} = {num_views} views x {args.samples_per_view} samples...")
        origins, directions, face_ids, view_ids, view_positions = sample_rays_sphere(
            grid_info, args.sphere_divisions, args.samples_per_view
        )
    else:  # all_faces
        print(f"\nGenerating {args.samples_per_face * 6} rays on all bounding box faces...")
        origins, directions, face_ids = sample_rays_on_bounding_box(
            grid_info, args.samples_per_face
        )
        # Create dummy view_ids and view_positions for consistency
        view_ids = np.zeros(len(origins), dtype=np.int32)
        view_positions = np.zeros((1, 3), dtype=np.float32)

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
    if num_hits > 0:
        print(f"  Min distance (hits only): {distances[hits > 0].min():.3f}")
        print(f"  Max distance (hits only): {distances[hits > 0].max():.3f}")
        print(f"  Mean distance (hits only): {distances[hits > 0].mean():.3f}")

    # Per-view statistics (for sphere mode)
    if args.sampling_mode == "sphere":
        num_views = args.sphere_divisions ** 2
        print("\nPer-view statistics:")
        for view_id in range(num_views):
            mask = view_ids == view_id
            view_hits = hits[mask].sum()
            view_total = mask.sum()
            view_hit_rate = 100.0 * view_hits / view_total if view_total > 0 else 0
            cam_pos = view_positions[view_id]
            print(f"  View {view_id} (pos=[{cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}]): "
                  f"{view_hits}/{view_total} hits ({view_hit_rate:.1f}%)")

    # Per-face statistics
    print("\nPer-face statistics:")
    face_names = ["-X", "+X", "-Y", "+Y", "-Z", "+Z"]
    for face_id in range(6):
        mask = face_ids == face_id
        if mask.sum() == 0:
            continue
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
        view_ids=view_ids,
        view_positions=view_positions,
        grid_shape=voxels.shape,
        voxel_size=grid_info['voxel_size'],
        object_id=args.object_id,
        level=args.level,
        sampling_mode=args.sampling_mode
    )

    print("\nDataset contents:")
    print(f"  origins: {origins.shape} - Ray starting points on bounding box")
    print(f"  directions: {directions.shape} - Ray directions (normalized)")
    print(f"  distances: {distances.shape} - Distance through occupied voxels")
    print(f"  hits: {hits.shape} - Binary hit flags (1=hit, 0=miss)")
    print(f"  face_ids: {face_ids.shape} - Which bbox face ray originated from (0-5)")
    print(f"  view_ids: {view_ids.shape} - Which view (0 to num_views-1)")
    print(f"  view_positions: {view_positions.shape} - Camera positions for each view")
    print(f"  grid_shape: {voxels.shape} - Original voxel grid dimensions")
    print(f"  voxel_size: {grid_info['voxel_size']} - Size of each voxel")
    print(f"  sampling_mode: {args.sampling_mode}")

    # Create visualization if matplotlib available
    try:
        import matplotlib.pyplot as plt

        print("\nCreating visualization...")

        if args.sampling_mode == "sphere":
            # Create ragged grid: rows = views, cols = visible faces per view
            num_views = args.sphere_divisions ** 2
            max_faces = 6  # Maximum possible faces per view

            fig = plt.figure(figsize=(20, 4 * num_views))
            gs = fig.add_gridspec(num_views, max_faces, hspace=0.4, wspace=0.3)

            face_names = ["-X", "+X", "-Y", "+Y", "-Z", "+Z"]

            for view_id in range(num_views):
                # Get all rays for this view
                view_mask = view_ids == view_id
                view_face_ids = face_ids[view_mask]
                view_distances = distances[view_mask]
                view_hits = hits[view_mask]
                view_origins = origins[view_mask]

                # Find unique faces for this view (in sorted order)
                unique_faces = sorted(np.unique(view_face_ids))

                # Create subplots for each visible face
                for col_idx, face_id in enumerate(unique_faces):
                    ax = fig.add_subplot(gs[view_id, col_idx])

                    # Get rays for this specific face in this view
                    face_mask = view_face_ids == face_id
                    face_distances = view_distances[face_mask]
                    face_hits = view_hits[face_mask]
                    face_origins = view_origins[face_mask]

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
                                       cmap='viridis', s=2, alpha=0.6, vmin=0,
                                       vmax=distances[hits > 0].max() if (hits > 0).any() else 1)

                    # Title with view position and face
                    cam_pos = view_positions[view_id]
                    if col_idx == 0:
                        ax.set_ylabel(f'View {view_id}\nPos: [{cam_pos[0]:.1f},{cam_pos[1]:.1f},{cam_pos[2]:.1f}]\n\n{ylabel}',
                                    fontsize=9, fontweight='bold')
                    else:
                        ax.set_ylabel(ylabel, fontsize=9)

                    ax.set_xlabel(xlabel, fontsize=9)
                    ax.set_title(f'Face {face_names[face_id]}', fontsize=10)
                    ax.set_aspect('equal', adjustable='box')
                    ax.tick_params(labelsize=8)

                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Distance', fontsize=8)
                    cbar.ax.tick_params(labelsize=7)

                    # Add hit rate annotation
                    hit_rate = 100.0 * face_hits.sum() / len(face_hits) if len(face_hits) > 0 else 0
                    ax.text(0.02, 0.98, f'Hits: {hit_rate:.0f}%',
                           transform=ax.transAxes, fontsize=8,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.suptitle(f'Object {args.object_id} - Spherical Ray Sampling\n'
                        f'({args.sphere_divisions}×{args.sphere_divisions} viewpoints, Rows: Views, Columns: Visible Faces)',
                        fontsize=14, fontweight='bold')

        else:  # all_faces mode
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Visualize each face
            for face_id in range(6):
                row = face_id // 3
                col = face_id % 3
                ax = axes[row, col]

                mask = face_ids == face_id
                if mask.sum() == 0:
                    ax.axis('off')
                    continue

                face_distances = distances[mask]
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

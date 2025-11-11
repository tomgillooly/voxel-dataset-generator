#!/usr/bin/env python3
"""
Generate ray tracing datasets for all subvolumes in the hierarchical voxel dataset.

This script recursively processes all subvolumes in an object's hierarchy, generating
ray tracing data for each unique subvolume at each level.
"""

import numpy as np
from pathlib import Path
import sys
import argparse
import json
from collections import defaultdict
from tqdm import tqdm

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
            camera_pos = np.array([
                camera_radius * np.sin(elev) * np.cos(azim),
                camera_radius * np.sin(elev) * np.sin(azim),
                camera_radius * np.cos(elev)
            ], dtype=np.float32)

            view_positions.append(camera_pos)

            # View direction (from camera toward origin)
            view_dir = -camera_pos / np.linalg.norm(camera_pos)

            # Determine which faces are visible from this viewpoint
            visible_faces = []

            face_normals = [
                np.array([-1, 0, 0]),  # Face 0: -X
                np.array([1, 0, 0]),   # Face 1: +X
                np.array([0, -1, 0]),  # Face 2: -Y
                np.array([0, 1, 0]),   # Face 3: +Y
                np.array([0, 0, -1]),  # Face 4: -Z
                np.array([0, 0, 1])    # Face 5: +Z
            ]

            for face_id, normal in enumerate(face_normals):
                if np.dot(normal, view_dir) > 0:
                    visible_faces.append(face_id)

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


def process_subvolume(voxel_path, hash_val, level, output_dir, sphere_divisions, samples_per_view, skip_empty=True):
    """
    Process a single subvolume through ray tracing.

    Args:
        voxel_path: Path to the voxel npz file
        hash_val: Hash identifier for the subvolume
        level: Hierarchy level
        output_dir: Base output directory for ray datasets
        sphere_divisions: Number of sphere sampling divisions
        samples_per_view: Number of samples per viewpoint
        skip_empty: Skip empty subvolumes

    Returns:
        Dict with processing results or None if skipped
    """
    # Load voxel data
    data = np.load(voxel_path)
    voxels = data['voxels']

    # Skip empty voxels if requested
    if skip_empty and voxels.sum() == 0:
        return None

    # Create output directory organized by level and hash prefix
    hash_prefix = hash_val[:2]
    level_output_dir = output_dir / f"level_{level}" / hash_prefix
    level_output_dir.mkdir(parents=True, exist_ok=True)

    output_file = level_output_dir / f"{hash_val}_rays.npz"

    # Skip if already processed
    if output_file.exists():
        return {"hash": hash_val, "level": level, "skipped": True}

    # Create ray tracer
    tracer = VoxelRayTracer(voxels, voxel_size=1.0)

    if not tracer.is_ready():
        return {"hash": hash_val, "level": level, "error": "Tracer not ready"}

    grid_info = tracer.get_grid_info()

    # Generate rays
    origins, directions, face_ids, view_ids, view_positions = sample_rays_sphere(
        grid_info, sphere_divisions, samples_per_view
    )

    # Trace rays
    distances = tracer.trace_rays(origins, directions)

    # Create hit flags
    hits = (distances > 0.0).astype(np.uint8)

    # Save dataset
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
        hash=hash_val,
        level=level,
        sampling_mode='sphere'
    )

    # Statistics
    num_hits = hits.sum()
    num_total = len(hits)
    hit_rate = 100.0 * num_hits / num_total if num_total > 0 else 0

    return {
        "hash": hash_val,
        "level": level,
        "num_rays": num_total,
        "num_hits": num_hits,
        "hit_rate": hit_rate,
        "occupancy": voxels.mean(),
        "skipped": False
    }


def collect_unique_subvolumes(dataset_dir, object_ids=None):
    """
    Collect all unique subvolumes across all objects and levels.

    Args:
        dataset_dir: Path to dataset directory
        object_ids: List of object IDs to process (None = all objects)

    Returns:
        Dict mapping (level, hash) -> voxel_path
    """
    objects_dir = dataset_dir / "objects"
    subvolumes_dir = dataset_dir / "subvolumes"

    if not objects_dir.exists():
        raise FileNotFoundError(f"Objects directory not found: {objects_dir}")

    # Collect all subdivision maps
    if object_ids is None:
        # Discover all objects
        object_ids = []
        for d in sorted(objects_dir.iterdir()):
            if d.is_dir() and d.name.startswith("object_"):
                obj_id = d.name.replace("object_", "")
                object_ids.append(obj_id)

    print(f"Collecting subvolumes from {len(object_ids)} objects...")

    # Track unique subvolumes: (level, hash) -> info
    unique_subvolumes = {}

    for obj_id in tqdm(object_ids, desc="Reading subdivision maps"):
        subdivision_map_path = objects_dir / f"object_{obj_id}" / "subdivision_map.json"

        if not subdivision_map_path.exists():
            print(f"Warning: No subdivision map for object {obj_id}")
            continue

        # Read subdivision map
        with open(subdivision_map_path, 'r') as f:
            subdivision_records = json.load(f)

        # Register each unique subvolume
        for record in subdivision_records:
            level = record['level']
            hash_val = record['hash']
            is_empty = record.get('is_empty', False)

            key = (level, hash_val)

            if key not in unique_subvolumes:
                # Find voxel file path
                hash_prefix = hash_val[:2]
                voxel_path = subvolumes_dir / f"level_{level}" / hash_prefix / f"{hash_val}.npz"

                if voxel_path.exists():
                    unique_subvolumes[key] = {
                        'level': level,
                        'hash': hash_val,
                        'path': voxel_path,
                        'is_empty': is_empty
                    }

    print(f"Found {len(unique_subvolumes)} unique subvolumes across {len(set(k[0] for k in unique_subvolumes.keys()))} levels")

    # Print per-level statistics
    level_counts = defaultdict(int)
    for (level, _), _ in unique_subvolumes.items():
        level_counts[level] += 1

    print("\nSubvolumes per level:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]:,} unique subvolumes")

    return unique_subvolumes


def main():
    parser = argparse.ArgumentParser(
        description="Generate ray tracing datasets for hierarchical voxel subvolumes"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--object-ids",
        type=str,
        nargs="+",
        help="Specific object IDs to process (e.g., '0001' '0002'). If not specified, processes all objects."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ray_dataset_hierarchical"),
        help="Output directory for ray tracing datasets"
    )
    parser.add_argument(
        "--sphere-divisions",
        type=int,
        default=4,
        help="Number of divisions for azimuth and elevation. Total viewpoints = divisions²"
    )
    parser.add_argument(
        "--samples-per-view",
        type=int,
        default=3000,
        help="Number of ray samples per viewpoint (distributed across visible faces)"
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        default=True,
        help="Skip empty subvolumes (default: True)"
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=0,
        help="Minimum level to process (default: 0)"
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        help="Maximum level to process (default: None = all levels)"
    )

    args = parser.parse_args()

    # Collect all unique subvolumes
    unique_subvolumes = collect_unique_subvolumes(args.dataset_dir, args.object_ids)

    # Filter by level if specified
    if args.min_level > 0 or args.max_level is not None:
        filtered = {}
        for (level, hash_val), info in unique_subvolumes.items():
            if level < args.min_level:
                continue
            if args.max_level is not None and level > args.max_level:
                continue
            filtered[(level, hash_val)] = info
        unique_subvolumes = filtered
        print(f"\nFiltered to {len(unique_subvolumes)} subvolumes (levels {args.min_level}-{args.max_level or 'max'})")

    if len(unique_subvolumes) == 0:
        print("No subvolumes to process!")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each unique subvolume
    print(f"\nProcessing {len(unique_subvolumes)} unique subvolumes...")
    print(f"Output directory: {args.output_dir}")
    print(f"Sphere divisions: {args.sphere_divisions} (= {args.sphere_divisions**2} viewpoints)")
    print(f"Samples per view: {args.samples_per_view}")

    results = []
    skipped_count = 0
    empty_count = 0
    error_count = 0

    for (level, hash_val), info in tqdm(list(unique_subvolumes.items()), desc="Processing subvolumes"):
        try:
            # Skip empty if requested
            if args.skip_empty and info['is_empty']:
                empty_count += 1
                continue

            result = process_subvolume(
                voxel_path=info['path'],
                hash_val=hash_val,
                level=level,
                output_dir=args.output_dir,
                sphere_divisions=args.sphere_divisions,
                samples_per_view=args.samples_per_view,
                skip_empty=args.skip_empty
            )

            if result is None:
                empty_count += 1
                continue

            results.append(result)

            if result.get("skipped", False):
                skipped_count += 1
            elif "error" in result:
                error_count += 1

        except Exception as e:
            print(f"\nError processing level {level}, hash {hash_val[:16]}...: {e}")
            error_count += 1
            continue

    # Summary statistics
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Total unique subvolumes: {len(unique_subvolumes)}")
    print(f"Processed: {len(results)}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Skipped (empty): {empty_count}")
    print(f"Errors: {error_count}")

    # Per-level statistics
    if results:
        level_stats = defaultdict(lambda: {"count": 0, "total_rays": 0, "total_hits": 0, "total_occupancy": 0.0})
        for result in results:
            if not result.get("skipped", False) and "error" not in result:
                level = result['level']
                level_stats[level]["count"] += 1
                level_stats[level]["total_rays"] += result["num_rays"]
                level_stats[level]["total_hits"] += result["num_hits"]
                level_stats[level]["total_occupancy"] += result["occupancy"]

        print("\nPer-level statistics:")
        for level in sorted(level_stats.keys()):
            stats = level_stats[level]
            avg_occupancy = stats["total_occupancy"] / stats["count"] if stats["count"] > 0 else 0
            avg_hit_rate = 100.0 * stats["total_hits"] / stats["total_rays"] if stats["total_rays"] > 0 else 0
            print(f"  Level {level}: {stats['count']} subvolumes, "
                  f"avg occupancy={avg_occupancy:.3%}, avg hit rate={avg_hit_rate:.1f}%")

    # Save processing summary
    summary_path = args.output_dir / "processing_summary.json"
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "sphere_divisions": args.sphere_divisions,
        "samples_per_view": args.samples_per_view,
        "total_subvolumes": len(unique_subvolumes),
        "processed": len(results),
        "skipped_already_exists": skipped_count,
        "skipped_empty": empty_count,
        "errors": error_count,
        "min_level": args.min_level,
        "max_level": args.max_level
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved processing summary to {summary_path}")
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

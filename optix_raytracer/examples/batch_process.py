#!/usr/bin/env python3
"""
Batch process multiple voxel objects from the dataset.
Renders each object from a single viewpoint and saves distance maps.
"""

import numpy as np
from pathlib import Path
import sys
import argparse
from tqdm import tqdm

# Add build directory to path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent / "build"))

from optix_voxel_tracer import VoxelRayTracer


def generate_orthographic_rays(resolution, grid_size=1.5, distance=2.0):
    """Generate orthographic rays pointing down."""
    h, w = resolution

    x = np.linspace(-grid_size/2, grid_size/2, w, dtype=np.float32)
    y = np.linspace(-grid_size/2, grid_size/2, h, dtype=np.float32)

    xx, yy = np.meshgrid(x, y)

    origins = np.zeros((h, w, 3), dtype=np.float32)
    origins[:, :, 0] = xx
    origins[:, :, 1] = yy
    origins[:, :, 2] = distance

    directions = np.zeros((h, w, 3), dtype=np.float32)
    directions[:, :, 2] = -1.0

    return origins, directions


def process_object(voxel_file, resolution=(512, 512)):
    """
    Process a single voxel object.

    Args:
        voxel_file: Path to .npz file containing voxels
        resolution: Output resolution

    Returns:
        distance_map: Accumulated distances through voxels
    """
    # Load voxels
    data = np.load(voxel_file)
    voxels = data['voxels']

    # Create tracer
    tracer = VoxelRayTracer(voxels, voxel_size=1.0)

    if not tracer.is_ready():
        raise RuntimeError("Tracer initialization failed")

    # Generate rays
    origins, directions = generate_orthographic_rays(resolution)

    # Trace
    distances = tracer.trace_rays(origins, directions)

    return distances


def main():
    parser = argparse.ArgumentParser(
        description="Batch process voxel objects with OptiX ray tracing"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ray_traces"),
        help="Output directory for ray traces"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Render resolution (width height)"
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Maximum number of objects to process"
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Starting object ID"
    )

    args = parser.parse_args()

    # Find all object directories
    objects_dir = args.dataset_dir / "objects"

    if not objects_dir.exists():
        print(f"ERROR: Objects directory not found: {objects_dir}")
        return 1

    # Get list of voxel files
    voxel_files = sorted(objects_dir.glob("object_*/level_0.npz"))

    if not voxel_files:
        print(f"ERROR: No voxel files found in {objects_dir}")
        return 1

    # Filter by start ID
    if args.start_id > 0:
        voxel_files = [f for f in voxel_files
                      if int(f.parent.name.split('_')[1]) >= args.start_id]

    # Limit number of objects
    if args.max_objects is not None:
        voxel_files = voxel_files[:args.max_objects]

    print(f"Found {len(voxel_files)} objects to process")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each object
    stats = {
        'total': len(voxel_files),
        'success': 0,
        'failed': 0,
        'total_distance': 0.0
    }

    for voxel_file in tqdm(voxel_files, desc="Processing objects"):
        object_id = voxel_file.parent.name.split('_')[1]
        output_file = args.output_dir / f"object_{object_id}_distances.npz"

        # Skip if already processed
        if output_file.exists():
            stats['success'] += 1
            continue

        try:
            # Process object
            distances = process_object(voxel_file, tuple(args.resolution))

            # Save results
            np.savez_compressed(
                output_file,
                distances=distances,
                object_id=object_id,
                resolution=args.resolution
            )

            stats['success'] += 1
            stats['total_distance'] += distances.sum()

        except Exception as e:
            print(f"\nERROR processing {voxel_file}: {e}")
            stats['failed'] += 1

    # Print summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Total objects: {stats['total']}")
    print(f"Successfully processed: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    if stats['success'] > 0:
        avg_distance = stats['total_distance'] / stats['success']
        print(f"Average accumulated distance: {avg_distance:.2f}")
    print(f"\nOutput directory: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

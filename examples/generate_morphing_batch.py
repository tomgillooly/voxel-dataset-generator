#!/usr/bin/env python3
"""
Generate morphing sequences for multiple object pairs.
Extract from optimal_transport_demo.ipynb for batch processing.
"""

import sys
sys.path.insert(0, '..')

from pathlib import Path
import numpy as np
import ot
from itertools import combinations

def generate_morphing(source_id, target_id, dataset_dir, output_dir, num_steps=10):
    """
    Generate a morphing sequence between two objects.

    Args:
        source_id: Source object ID
        target_id: Target object ID
        dataset_dir: Path to dataset directory
        output_dir: Path to output directory
        num_steps: Number of interpolation steps (includes source and target)
    """
    # Load objects
    source_path = dataset_dir / 'objects' / f'object_{source_id:04d}' / 'level_0.npz'
    target_path = dataset_dir / 'objects' / f'object_{target_id:04d}' / 'level_0.npz'

    source_voxels = np.load(source_path)['voxels']
    target_voxels = np.load(target_path)['voxels']

    print(f"\nProcessing {source_id} -> {target_id}")
    print(f"  Source shape: {source_voxels.shape}, occupancy: {source_voxels.mean():.2%}, voxels: {int(source_voxels.sum())}")
    print(f"  Target shape: {target_voxels.shape}, occupancy: {target_voxels.mean():.2%}, voxels: {int(target_voxels.sum())}")

    # Extract point clouds
    source_points = np.stack(np.where(source_voxels), axis=-1)
    target_points = np.stack(np.where(target_voxels), axis=-1)

    # Compute distance matrix
    print("  Computing distance matrix...")
    dist = ot.dist(source_points, target_points, metric='euclidean')

    # Solve optimal transport
    print("  Solving optimal transport...")
    plan, log = ot.emd(
        ot.utils.unif(source_voxels.sum()),
        ot.utils.unif(target_voxels.sum()),
        dist,
        log=True,
        # numThreads=16,
        numItermax=int(1e10)
    )

    transport_cost = np.sum(plan * dist)
    sparsity = np.mean(plan > 0)
    print(f"  Transport cost: {transport_cost:.4f}, plan sparsity: {sparsity:.2%}")

    # Build ray map
    print("  Building ray map...")
    ray_map = {}
    for source_idx, (source_point, plan_row) in enumerate(zip(source_points, plan)):
        rays = []
        end_indices = np.nonzero(plan_row)[0]
        for end_idx in end_indices:
            ray = target_points[end_idx] - source_point
            weight = plan_row[end_idx]
            rays.append((weight, ray))
        ray_map[source_idx] = rays

    # Generate morphing sequence
    print(f"  Generating {num_steps} interpolation steps...")
    sequence = np.zeros((num_steps-2,) + source_voxels.shape)
    t_steps = np.linspace(0, 1, num_steps, endpoint=True)[1:-1]

    for source_idx, rays in ray_map.items():
        for weight, ray in rays:
            for t_idx, t in enumerate(t_steps):
                point = source_points[source_idx] + ray*t
                sequence[t_idx, *point.astype(int)] += weight

    # Save morphing sequence
    print(f"  Saving to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, voxels in enumerate(sequence):
        filename = f"morph_{source_id:04d}_to_{target_id:04d}_step_{i:03d}.npz"
        filepath = output_dir / filename
        np.savez_compressed(filepath, voxels=voxels)

    # Save metadata
    metadata_file = output_dir / f"morph_{source_id:04d}_to_{target_id:04d}_metadata.npz"
    np.savez(
        metadata_file,
        source_id=source_id,
        target_id=target_id,
        num_steps=len(sequence),
        grid_shape=list(sequence[0].shape),
        transport_cost=transport_cost,
        plan_sparsity=sparsity,
    )

    print(f"  ✓ Saved {len(sequence)} steps and metadata")
    return transport_cost, sparsity


def morphing_exists(source_id, target_id, output_dir):
    """Check if morphing results already exist for this pair."""
    metadata_file = output_dir / f"morph_{source_id:04d}_to_{target_id:04d}_metadata.npz"
    return metadata_file.exists()


def main():
    # Configuration
    dataset_dir = Path('dataset_64')
    output_dir = Path('morphing_results')
    num_steps = 10

    # Define object IDs to use
    # You can modify this list to try different objects
    # object_ids = [0, 1, 5, 17, 19, 21, 23, 24]
    object_folders = (dataset_dir / 'objects').glob('object_????')
    object_ids = [int(p.name.split('_')[1]) for p in object_folders]

    # Generate all combinations (order doesn't matter)
    # all_pairs = list(combinations(object_ids, 2))

    all_pairs = [(0, target) for target in object_ids if target != 0]

    print(f"Total possible combinations: {len(all_pairs)}")

    # Filter out existing morphings
    object_pairs = []
    skipped = []
    for source_id, target_id in all_pairs:
        if morphing_exists(source_id, target_id, output_dir):
            skipped.append((source_id, target_id))
        else:
            object_pairs.append((source_id, target_id))

    if skipped:
        print(f"\nSkipping {len(skipped)} existing morphings:")
        for source_id, target_id in skipped:
            print(f"  {source_id:4d} -> {target_id:4d} (already exists)")

    print(f"\nProcessing {len(object_pairs)} new morphings...")

    # Generate morphing for each pair
    results = []
    for source_id, target_id in object_pairs:
        try:
            cost, sparsity = generate_morphing(
                source_id=source_id,
                target_id=target_id,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                num_steps=num_steps
            )
            results.append((source_id, target_id, cost, sparsity, 'success'))
        except Exception as e:
            print(f"  ✗ Error processing {source_id} -> {target_id}: {e}")
            results.append((source_id, target_id, None, None, 'failed'))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for source_id, target_id, cost, sparsity, status in results:
        if status == 'success':
            print(f"  {source_id:4d} -> {target_id:4d}: cost={cost:8.4f}, sparsity={sparsity:6.2%}")
        else:
            print(f"  {source_id:4d} -> {target_id:4d}: FAILED")

    successful = sum(1 for _, _, _, _, s in results if s == 'success')
    print(f"\nCompleted {successful}/{len(results)} morphing sequences")


if __name__ == '__main__':
    main()

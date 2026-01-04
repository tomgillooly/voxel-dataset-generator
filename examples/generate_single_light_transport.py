#!/usr/bin/env python3
"""Generate light transport for a single object and optionally visualize.

Usage:
    # Generate for object 0000
    python generate_single_light_transport.py --object 0000

    # Generate for multiple objects
    python generate_single_light_transport.py --object 0000 --object 0001

    # Compare two objects side by side
    python generate_single_light_transport.py --object 0000 --object 0001 --compare

    # Quick test with fewer samples
    python generate_single_light_transport.py --object 0000 --samples 1000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voxel_dataset_generator.light_transport import (
    MetalVoxelTracer,
    OpticalProperties,
    TransferMatrixBuilder,
    save_transfer_matrix,
    analyze_transfer_matrix,
    MLX_AVAILABLE,
)


def find_object_voxels(object_id: str, dataset_dir: Path = Path("dataset_64")) -> Path:
    """Find voxel file for an object ID."""
    # Try dataset_64 structure first
    level0 = dataset_dir / "objects" / f"object_{object_id}" / "level_0.npz"
    if level0.exists():
        return level0

    # Try morphing_results
    morph_dir = Path("morphing_results")
    pattern = f"morph_{object_id}_*.npz"
    matches = list(morph_dir.glob(pattern))
    # Filter out metadata files
    matches = [m for m in matches if "metadata" not in m.name]
    if matches:
        return sorted(matches)[0]

    raise FileNotFoundError(f"Could not find voxels for object {object_id}")


def load_voxels(path: Path, fill_interior: bool = True) -> np.ndarray:
    """Load voxels from .npz file.

    Args:
        path: Path to .npz file
        fill_interior: If True, fill hollow interiors (required for subsurface scattering)
    """
    from scipy import ndimage

    data = np.load(path)

    # Try common key names
    for key in ['voxels', 'occupancy', 'data', 'arr_0']:
        if key in data:
            voxels = data[key]
            break
    else:
        # Use first array
        voxels = data[list(data.keys())[0]]

    # Convert to binary
    binary = voxels > 0

    # Fill interior holes (objects from dataset_64 are hollow shells)
    if fill_interior:
        binary = ndimage.binary_fill_holes(binary)

    return binary.astype(np.uint8)


def generate_transfer_matrix(
    voxels: np.ndarray,
    optical_props: OpticalProperties,
    patches_per_face: int = 8,
    sh_order: int = 2,
    samples_per_condition: int = 10000,
    seed: int = 42,
    verbose: bool = True,
    track_sh_error: bool = False,
) -> tuple:
    """Generate transfer matrix for voxel structure.

    Returns:
        Tuple of (transfer_matrix, metadata, sh_error_stats)
        where sh_error_stats is None if track_sh_error=False
    """
    from tqdm import tqdm

    start_time = time.time()

    # Create tracer
    tracer = MetalVoxelTracer(voxels, optical_props=optical_props)

    # Create builder
    builder = TransferMatrixBuilder(
        tracer=tracer,
        patches_per_face=patches_per_face,
        sh_order=sh_order,
        samples_per_condition=samples_per_condition,
        seed=seed,
        track_sh_error=track_sh_error,
    )

    if verbose:
        n_conditions = builder.n_conditions
        pbar = tqdm(total=n_conditions, desc="Computing transfer matrix")

        def progress_callback(current, total):
            pbar.n = current
            pbar.refresh()

        transfer_matrix, sh_error_stats = builder.compute(progress_callback=progress_callback)
        pbar.close()
    else:
        transfer_matrix, sh_error_stats = builder.compute()

    elapsed = time.time() - start_time

    # Get metadata
    metadata = builder.get_metadata()
    metadata['elapsed_seconds'] = elapsed
    metadata['voxel_shape'] = list(voxels.shape)
    metadata['occupancy_ratio'] = float(voxels.sum() / voxels.size)
    metadata['analysis'] = analyze_transfer_matrix(transfer_matrix)

    return transfer_matrix, metadata, sh_error_stats


def compare_transfer_matrices(
    transfers: list[tuple[str, np.ndarray, dict]],
    output_path: Path = None,
):
    """Compare multiple transfer matrices visually."""
    import matplotlib.pyplot as plt

    n = len(transfers)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))

    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (name, transfer, metadata) in enumerate(transfers):
        # Transfer matrix heatmap
        ax1 = axes[0, i]
        im = ax1.imshow(
            np.log10(np.abs(transfer) + 1e-10),
            cmap='viridis',
            aspect='auto'
        )
        ax1.set_title(f'{name}\nTransfer Matrix (log scale)')
        ax1.set_xlabel('Input condition')
        ax1.set_ylabel('Output condition')
        plt.colorbar(im, ax=ax1, label='log10(value)')

        # Column sums (energy per input)
        ax2 = axes[1, i]
        col_sums = transfer.sum(axis=0)
        ax2.plot(col_sums, 'b-', linewidth=0.5)
        ax2.set_title(f'Column Sums (energy conservation)')
        ax2.set_xlabel('Input condition')
        ax2.set_ylabel('Total exitant energy')
        ax2.set_ylim(0, max(1.5, col_sums.max() * 1.1))
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect conservation')
        ax2.legend()

        # Add stats text
        analysis = metadata.get('analysis', {})
        stats_text = (
            f"Occupancy: {metadata.get('occupancy_ratio', 0):.1%}\n"
            f"Mean: {analysis.get('mean', 0):.2e}\n"
            f"Max: {analysis.get('max', 0):.2e}\n"
            f"NNZ: {analysis.get('nnz_ratio', 0):.1%}"
        )
        ax1.text(
            0.02, 0.98, stats_text,
            transform=ax1.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {output_path}")
        plt.close()
    else:
        plt.show()


def compare_3d_radiance(
    transfers: list[tuple[str, np.ndarray, dict, np.ndarray]],
    incident_face: int = 0,
    output_path: Path = None,
):
    """Compare exitant radiance patterns in 3D for multiple objects."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    n = len(transfers)
    fig = plt.figure(figsize=(6 * n, 10))

    face_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']

    for i, (name, transfer, metadata, voxels) in enumerate(transfers):
        patches_per_face = metadata.get('patches_per_face', 8)
        n_patches = 6 * patches_per_face ** 2
        n_sh = metadata.get('n_sh', 9)

        # Get grid bounds
        resolution = np.array(voxels.shape)
        voxel_size = metadata.get('voxel_size', 1.0)
        half_extents = resolution * voxel_size / 2.0
        grid_min = -half_extents
        grid_max = half_extents

        # Compute radiance for incident face
        # Sum over SH coefficients for the incident patch-face
        start_patch = incident_face * patches_per_face ** 2
        end_patch = (incident_face + 1) * patches_per_face ** 2

        # Get columns for all SH coefficients of patches on incident face
        radiance = np.zeros(n_patches)
        for patch_idx in range(start_patch, end_patch):
            for sh_idx in range(n_sh):
                col_idx = patch_idx * n_sh + sh_idx
                radiance += transfer[:n_patches * n_sh:n_sh, col_idx]

        # Average over incident patches
        radiance /= (patches_per_face ** 2)

        # Normalize
        if radiance.max() > 0:
            radiance_norm = radiance / radiance.max()
        else:
            radiance_norm = radiance

        # Get patch centers
        centers = []
        for face_idx in range(6):
            face_min = grid_min.copy()
            face_max = grid_max.copy()

            axis = face_idx // 2
            is_positive = (face_idx % 2 == 0)

            if is_positive:
                face_min[axis] = grid_max[axis]
            else:
                face_max[axis] = grid_min[axis]

            u_axis = (axis + 1) % 3
            v_axis = (axis + 2) % 3

            for v in range(patches_per_face):
                for u in range(patches_per_face):
                    center = np.zeros(3)
                    center[axis] = face_min[axis] if not is_positive else face_max[axis]

                    u_range = grid_max[u_axis] - grid_min[u_axis]
                    v_range = grid_max[v_axis] - grid_min[v_axis]

                    center[u_axis] = grid_min[u_axis] + (u + 0.5) / patches_per_face * u_range
                    center[v_axis] = grid_min[v_axis] + (v + 0.5) / patches_per_face * v_range

                    centers.append(center)

        centers = np.array(centers)

        # Plot voxels
        ax1 = fig.add_subplot(2, n, i + 1, projection='3d')
        occupied = np.argwhere(voxels > 0)
        if len(occupied) > 0:
            # Subsample if too many
            if len(occupied) > 2000:
                indices = np.random.choice(len(occupied), 2000, replace=False)
                occupied = occupied[indices]

            # Convert to world coords
            world_coords = (occupied - resolution / 2) * voxel_size
            ax1.scatter(
                world_coords[:, 2], world_coords[:, 1], world_coords[:, 0],
                c='gray', s=1, alpha=0.3
            )

        ax1.set_title(f'{name}\nVoxel Structure')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Plot radiance
        ax2 = fig.add_subplot(2, n, n + i + 1, projection='3d')

        colors = np.zeros((n_patches, 4))
        sizes = np.zeros(n_patches)
        for j in range(n_patches):
            face = j // (patches_per_face ** 2)
            if face == incident_face:
                colors[j] = [0.0, 0.0, 1.0, 1.0]
                sizes[j] = 50
            else:
                intensity = radiance_norm[j]
                colors[j] = plt.cm.YlOrRd(intensity)
                sizes[j] = 20 + 80 * intensity

        ax2.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2],
            c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidths=0.5
        )

        ax2.set_title(f'Exitant Radiance (from {face_names[incident_face]})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D comparison to {output_path}")
        plt.close()
    else:
        plt.show()


def print_comparison_stats(transfers: list[tuple[str, np.ndarray, dict]]):
    """Print comparison statistics between transfer matrices."""
    print("\n" + "=" * 60)
    print("Transfer Matrix Comparison")
    print("=" * 60)

    for name, transfer, metadata in transfers:
        analysis = metadata.get('analysis', {})
        print(f"\n{name}:")
        print(f"  Shape: {transfer.shape}")
        print(f"  Occupancy: {metadata.get('occupancy_ratio', 0):.1%}")
        print(f"  Mean value: {analysis.get('mean', 0):.4e}")
        print(f"  Max value: {analysis.get('max', 0):.4e}")
        print(f"  Non-zero ratio: {analysis.get('nnz_ratio', 0):.2%}")
        print(f"  Column sum (mean): {analysis.get('column_sum_mean', 0):.4f}")
        print(f"  Column sum (std): {analysis.get('column_sum_std', 0):.4f}")

    # Pairwise differences
    if len(transfers) > 1:
        print("\n" + "-" * 40)
        print("Pairwise Differences:")
        for i in range(len(transfers)):
            for j in range(i + 1, len(transfers)):
                name_i, transfer_i, _ = transfers[i]
                name_j, transfer_j, _ = transfers[j]

                diff = np.abs(transfer_i - transfer_j)
                rel_diff = diff / (np.abs(transfer_i) + np.abs(transfer_j) + 1e-10)

                print(f"\n  {name_i} vs {name_j}:")
                print(f"    Mean abs diff: {diff.mean():.4e}")
                print(f"    Max abs diff: {diff.max():.4e}")
                print(f"    Mean rel diff: {rel_diff.mean():.2%}")
                print(f"    Frobenius norm diff: {np.linalg.norm(diff):.4e}")

                # Correlation
                corr = np.corrcoef(transfer_i.flatten(), transfer_j.flatten())[0, 1]
                print(f"    Correlation: {corr:.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate light transport for specific objects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--object', '-o', action='append', dest='objects',
        help="Object ID to process (can specify multiple)"
    )
    parser.add_argument(
        '--dataset-dir', type=Path, default=Path('dataset_64'),
        help="Dataset directory"
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('light_transport_dataset'),
        help="Output directory"
    )

    # Simulation parameters
    parser.add_argument('--patches-per-face', type=int, default=8)
    parser.add_argument('--sh-order', type=int, default=2)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--sigma-s', type=float, default=1.0)
    parser.add_argument('--sigma-a', type=float, default=0.1)
    parser.add_argument('--g', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--track-sh-error', action='store_true',
        help="Track SH reconstruction error statistics (Phase 0 validation)"
    )

    # Output options
    parser.add_argument('--compare', action='store_true', help="Show comparison plot")
    parser.add_argument('--compare-3d', action='store_true', help="Show 3D radiance comparison")
    parser.add_argument('--save-plot', type=Path, help="Save comparison plot to file")
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    if not args.objects:
        parser.error("At least one --object is required")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup optical properties
    optical_props = OpticalProperties(
        sigma_s=args.sigma_s,
        sigma_a=args.sigma_a,
        g=args.g
    )

    if not args.quiet:
        print("=" * 60)
        print("Light Transport Generator")
        print("=" * 60)
        print(f"MLX available: {MLX_AVAILABLE}")
        print(f"Objects: {args.objects}")
        print(f"Samples per condition: {args.samples}")
        print(f"Optical properties: σs={args.sigma_s}, σa={args.sigma_a}, g={args.g}")
        print("=" * 60)

    # Process each object
    results = []

    for obj_id in args.objects:
        output_path = args.output_dir / f"object_{obj_id}.npz"

        if args.skip_existing and output_path.exists():
            if not args.quiet:
                print(f"\nSkipping {obj_id} (already exists)")
            # Load existing
            data = np.load(output_path)
            transfer = data['transfer_matrix']
            metadata = {
                'patches_per_face': args.patches_per_face,
                'n_sh': (args.sh_order + 1) ** 2,
                'analysis': analyze_transfer_matrix(transfer),
            }
            voxels = load_voxels(find_object_voxels(obj_id, args.dataset_dir))
            metadata['occupancy_ratio'] = float(voxels.sum() / voxels.size)
            results.append((f"object_{obj_id}", transfer, metadata, voxels))
            continue

        if not args.quiet:
            print(f"\nProcessing object {obj_id}...")

        # Find and load voxels
        try:
            voxel_path = find_object_voxels(obj_id, args.dataset_dir)
            if not args.quiet:
                print(f"  Loading voxels from {voxel_path}")
            voxels = load_voxels(voxel_path)
            if not args.quiet:
                print(f"  Voxel shape: {voxels.shape}, occupied: {voxels.sum()}")
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            continue

        # Generate transfer matrix
        transfer, metadata, sh_error_stats = generate_transfer_matrix(
            voxels=voxels,
            optical_props=optical_props,
            patches_per_face=args.patches_per_face,
            sh_order=args.sh_order,
            samples_per_condition=args.samples,
            seed=args.seed,
            verbose=not args.quiet,
            track_sh_error=args.track_sh_error,
        )

        # Save
        save_transfer_matrix(output_path, transfer, metadata, sh_error_stats)
        if not args.quiet:
            print(f"  Saved to {output_path}")
            print(f"  Time: {metadata['elapsed_seconds']:.1f}s")

        results.append((f"object_{obj_id}", transfer, metadata, voxels))

    # Print comparison stats
    if len(results) > 1:
        print_comparison_stats([(n, t, m) for n, t, m, v in results])

    # Show comparison plot
    if args.compare or args.save_plot:
        compare_transfer_matrices(
            [(n, t, m) for n, t, m, v in results],
            output_path=args.save_plot
        )

    if args.compare_3d:
        compare_3d_radiance(
            results,
            incident_face=0,
            output_path=args.save_plot.with_suffix('.3d.png') if args.save_plot else None
        )


if __name__ == '__main__':
    main()

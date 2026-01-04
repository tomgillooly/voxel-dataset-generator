#!/usr/bin/env python3
"""Generate light transport ground truth dataset from morphing results.

This script processes voxel structures from the morphing_results directory
and generates transfer matrices for each structure using Monte Carlo
subsurface scattering simulation.

Usage:
    python generate_light_transport_dataset.py --input-dir morphing_results --output-dir light_transport_dataset

    # Process only first 10 structures for testing
    python generate_light_transport_dataset.py --limit 10 --samples 1000

    # Custom optical properties
    python generate_light_transport_dataset.py --sigma-s 2.0 --sigma-a 0.2 --g 0.5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
from tqdm import tqdm

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


def find_voxel_files(input_dir: Path, pattern: str = "morph_*_step_*.npz") -> List[Path]:
    """Find all voxel files in input directory."""
    files = sorted(input_dir.glob(pattern))
    # Exclude metadata files
    files = [f for f in files if "metadata" not in f.name]
    return files


def process_structure(
    npz_path: Path,
    output_dir: Path,
    optical_props: OpticalProperties,
    patches_per_face: int,
    sh_order: int,
    samples_per_condition: int,
    seed: int,
    save_analysis: bool = True,
) -> dict:
    """Process a single voxel structure.

    Args:
        npz_path: Path to input .npz file
        output_dir: Output directory
        optical_props: Material optical properties
        patches_per_face: Patches per axis per face
        sh_order: Spherical harmonics order
        samples_per_condition: Monte Carlo samples per incident condition
        seed: Random seed
        save_analysis: Whether to save analysis results

    Returns:
        Dictionary with processing metadata
    """
    from scipy import ndimage

    start_time = time.time()

    # Load voxels
    data = np.load(npz_path)
    voxels = data['voxels']

    # Convert to binary if needed
    if voxels.dtype == np.float64 or voxels.dtype == np.float32:
        voxels = (voxels > 0.5).astype(np.uint8)
    else:
        voxels = (voxels > 0).astype(np.uint8)

    # Fill interior holes - objects may be hollow shells but subsurface
    # scattering requires solid volumes
    voxels = ndimage.binary_fill_holes(voxels).astype(np.uint8)

    # Create tracer
    tracer = MetalVoxelTracer(voxels, optical_props=optical_props)

    # Create builder
    builder = TransferMatrixBuilder(
        tracer=tracer,
        patches_per_face=patches_per_face,
        sh_order=sh_order,
        samples_per_condition=samples_per_condition,
        seed=seed,
    )

    # Compute transfer matrix with progress bar
    n_conditions = builder.n_conditions
    pbar = tqdm(total=n_conditions, desc=f"  {npz_path.stem}", leave=False)

    def progress_callback(current, total):
        pbar.n = current
        pbar.refresh()

    transfer_matrix = builder.compute(progress_callback=progress_callback)
    pbar.close()

    elapsed = time.time() - start_time

    # Get metadata
    metadata = builder.get_metadata()
    metadata['source_file'] = str(npz_path)
    metadata['elapsed_seconds'] = elapsed
    metadata['voxel_shape'] = list(voxels.shape)
    metadata['occupancy_ratio'] = float(voxels.sum() / voxels.size)

    # Analyze transfer matrix
    if save_analysis:
        analysis = analyze_transfer_matrix(transfer_matrix)
        metadata['analysis'] = analysis

    # Save transfer matrix
    output_path = output_dir / (npz_path.stem + "_transfer.npz")
    save_transfer_matrix(output_path, transfer_matrix, metadata)

    return {
        'source': str(npz_path),
        'output': str(output_path),
        'elapsed_seconds': elapsed,
        'voxel_shape': list(voxels.shape),
        'occupancy_ratio': metadata['occupancy_ratio'],
        'transfer_shape': list(transfer_matrix.shape),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate light transport ground truth dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output
    parser.add_argument(
        '--input-dir', type=Path,
        default=Path('morphing_results'),
        help="Directory containing voxel .npz files"
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('light_transport_dataset'),
        help="Output directory for transfer matrices"
    )

    # Discretization parameters
    parser.add_argument(
        '--patches-per-face', type=int, default=8,
        help="Patches per axis per face (8 -> 384 total patches)"
    )
    parser.add_argument(
        '--sh-order', type=int, default=2,
        help="Spherical harmonics order (2 -> 9 coefficients)"
    )
    parser.add_argument(
        '--samples', type=int, default=10000,
        help="Monte Carlo samples per incident condition"
    )

    # Optical properties
    parser.add_argument(
        '--sigma-s', type=float, default=1.0,
        help="Scattering coefficient"
    )
    parser.add_argument(
        '--sigma-a', type=float, default=0.1,
        help="Absorption coefficient"
    )
    parser.add_argument(
        '--g', type=float, default=0.8,
        help="Henyey-Greenstein asymmetry parameter (-1 to 1)"
    )

    # Processing options
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help="Process only first N structures"
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help="Skip structures that already have output files"
    )
    parser.add_argument(
        '--pattern', type=str, default="morph_*_step_*.npz",
        help="Glob pattern for input files"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create optical properties
    optical_props = OpticalProperties(
        sigma_s=args.sigma_s,
        sigma_a=args.sigma_a,
        g=args.g
    )

    # Print configuration
    n_patches = 6 * args.patches_per_face ** 2
    n_sh = (args.sh_order + 1) ** 2
    n_conditions = n_patches * n_sh

    print("=" * 60)
    print("Light Transport Dataset Generator")
    print("=" * 60)
    print(f"MLX available: {MLX_AVAILABLE}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Discretization:")
    print(f"  Patches per face: {args.patches_per_face} ({n_patches} total)")
    print(f"  SH order: {args.sh_order} ({n_sh} coefficients)")
    print(f"  Transfer matrix: {n_conditions} x {n_conditions}")
    print(f"  Samples per condition: {args.samples}")
    print()
    print("Optical properties:")
    print(f"  sigma_s: {optical_props.sigma_s}")
    print(f"  sigma_a: {optical_props.sigma_a}")
    print(f"  g: {optical_props.g}")
    print(f"  albedo: {optical_props.albedo:.4f}")
    print(f"  mean free path: {optical_props.mean_free_path:.4f}")
    print("=" * 60)

    # Find input files
    npz_files = find_voxel_files(args.input_dir, args.pattern)

    if args.limit:
        npz_files = npz_files[:args.limit]

    if args.skip_existing:
        existing = set(f.stem.replace("_transfer", "") for f in args.output_dir.glob("*_transfer.npz"))
        npz_files = [f for f in npz_files if f.stem not in existing]

    print(f"\nProcessing {len(npz_files)} structures...")

    # Process all structures
    results = []
    total_time = 0
    errors = []

    for i, npz_path in enumerate(tqdm(npz_files, desc="Structures")):
        try:
            result = process_structure(
                npz_path=npz_path,
                output_dir=args.output_dir,
                optical_props=optical_props,
                patches_per_face=args.patches_per_face,
                sh_order=args.sh_order,
                samples_per_condition=args.samples,
                seed=args.seed + i,  # Different seed per structure
            )
            results.append(result)
            total_time += result['elapsed_seconds']

        except Exception as e:
            error_info = {
                'file': str(npz_path),
                'error': str(e),
            }
            errors.append(error_info)
            tqdm.write(f"Error processing {npz_path.name}: {e}")

    # Save dataset metadata
    metadata = {
        'optical_properties': {
            'sigma_s': optical_props.sigma_s,
            'sigma_a': optical_props.sigma_a,
            'g': optical_props.g,
            'albedo': optical_props.albedo,
            'mean_free_path': optical_props.mean_free_path,
        },
        'discretization': {
            'patches_per_face': args.patches_per_face,
            'sh_order': args.sh_order,
            'n_patches': n_patches,
            'n_sh': n_sh,
            'n_conditions': n_conditions,
            'samples_per_condition': args.samples,
        },
        'processing': {
            'seed': args.seed,
            'mlx_available': MLX_AVAILABLE,
            'total_time_seconds': total_time,
            'n_processed': len(results),
            'n_errors': len(errors),
        },
        'structures': results,
        'errors': errors if errors else None,
    }

    metadata_path = args.output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Processed: {len(results)} structures")
    print(f"Errors: {len(errors)}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    if results:
        avg_time = total_time / len(results)
        print(f"Average time per structure: {avg_time:.1f}s")
    print(f"Output directory: {args.output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Render a view of an object using its light transport transfer matrix.

This script takes a transfer matrix and renders what the object would look like
when illuminated from a given direction and viewed from another direction.
The transfer matrix encodes how light scatters through the object, so we can
use it to create a 2D image showing the exitant radiance pattern.

Usage:
    # Render object illuminated from +X, viewed from -Z
    python render_light_transport.py light_transport_dataset/object_0000.npz

    # Specify illumination and view directions
    python render_light_transport.py object_0000.npz --illuminate +X --view -Z

    # Render all 6 view directions
    python render_light_transport.py object_0000.npz --all-views

    # Include the voxel structure for comparison
    python render_light_transport.py object_0000.npz --voxels dataset_64/objects/object_0000/level_0.npz
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


FACE_NAMES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
FACE_NORMALS = np.array([
    [1, 0, 0],   # +X
    [-1, 0, 0],  # -X
    [0, 1, 0],   # +Y
    [0, -1, 0],  # -Y
    [0, 0, 1],   # +Z
    [0, 0, -1],  # -Z
], dtype=np.float32)


def parse_face(face_str: str) -> int:
    """Parse face string to index."""
    face_str = face_str.upper().replace(' ', '')
    try:
        return FACE_NAMES.index(face_str)
    except ValueError:
        raise ValueError(f"Invalid face: {face_str}. Must be one of {FACE_NAMES}")


def load_transfer_matrix(path: Path) -> tuple[np.ndarray, dict]:
    """Load transfer matrix and metadata."""
    data = np.load(path)
    transfer = data['transfer_matrix']

    # Extract metadata
    metadata = {}
    for key in data.keys():
        if key != 'transfer_matrix':
            val = data[key]
            if val.ndim == 0:
                metadata[key] = val.item()
            else:
                metadata[key] = val

    # Set defaults
    metadata.setdefault('patches_per_face', 8)
    metadata.setdefault('sh_order', 2)
    metadata.setdefault('n_sh', (metadata.get('sh_order', 2) + 1) ** 2)

    return transfer, metadata


def get_patch_grid_for_face(
    face_idx: int,
    patches_per_face: int,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get 2D grid of patch centers for a face, suitable for imshow.

    Returns:
        centers: (patches_per_face, patches_per_face, 3) array of patch centers
        u_axis: Index of U axis in world coordinates
        v_axis: Index of V axis in world coordinates
    """
    axis = face_idx // 2
    is_positive = (face_idx % 2 == 0)

    # Determine which axes are U and V for this face
    # We want consistent orientation when viewed from outside
    if axis == 0:  # X face
        u_axis, v_axis = 1, 2  # Y, Z
    elif axis == 1:  # Y face
        u_axis, v_axis = 0, 2  # X, Z
    else:  # Z face
        u_axis, v_axis = 0, 1  # X, Y

    # Create grid
    centers = np.zeros((patches_per_face, patches_per_face, 3))

    # Fixed coordinate on the face
    if is_positive:
        centers[:, :, axis] = grid_max[axis]
    else:
        centers[:, :, axis] = grid_min[axis]

    # U and V coordinates
    u_range = grid_max[u_axis] - grid_min[u_axis]
    v_range = grid_max[v_axis] - grid_min[v_axis]

    for vi in range(patches_per_face):
        for ui in range(patches_per_face):
            centers[vi, ui, u_axis] = grid_min[u_axis] + (ui + 0.5) / patches_per_face * u_range
            centers[vi, ui, v_axis] = grid_min[v_axis] + (vi + 0.5) / patches_per_face * v_range

    return centers, u_axis, v_axis


def compute_view_radiance(
    transfer: np.ndarray,
    metadata: dict,
    illuminate_face: int,
    view_face: int,
) -> np.ndarray:
    """Compute radiance image for viewing from a specific face.

    Args:
        transfer: Transfer matrix (n_conditions x n_conditions)
        metadata: Transfer matrix metadata
        illuminate_face: Face index for illumination (0-5)
        view_face: Face index we're viewing from (0-5)

    Returns:
        radiance: (patches_per_face, patches_per_face) image of exitant radiance
    """
    patches_per_face = metadata.get('patches_per_face', 8)
    n_sh = metadata.get('n_sh', 9)
    n_patches_per_face = patches_per_face ** 2

    # The view face is the face we're looking at (opposite to view direction)
    # When viewing from -Z, we see the +Z face
    # So view_face here is actually the face we see

    # Get patch indices for the viewed face
    view_start = view_face * n_patches_per_face
    view_end = view_start + n_patches_per_face

    # Get patch indices for illumination face
    illum_start = illuminate_face * n_patches_per_face
    illum_end = illum_start + n_patches_per_face

    # Sum contributions from all illumination patches and SH coefficients
    # to all viewed patches (using DC component for radiance)
    radiance = np.zeros(n_patches_per_face)

    for view_patch in range(n_patches_per_face):
        view_patch_global = view_start + view_patch
        # DC coefficient (Y_0^0) for this output patch
        view_condition = view_patch_global * n_sh  # DC is first coefficient

        # Sum over all illumination conditions
        for illum_patch in range(n_patches_per_face):
            illum_patch_global = illum_start + illum_patch
            # Use DC illumination (uniform over hemisphere)
            illum_condition = illum_patch_global * n_sh

            radiance[view_patch] += transfer[view_condition, illum_condition]

    # Reshape to 2D grid
    radiance = radiance.reshape(patches_per_face, patches_per_face)

    return radiance


def compute_directional_radiance(
    transfer: np.ndarray,
    metadata: dict,
    illuminate_face: int,
) -> np.ndarray:
    """Compute radiance for all boundary patches given illumination from one face.

    This gives a full picture of how light exits the object.

    Returns:
        radiance: (6, patches_per_face, patches_per_face) for all 6 faces
    """
    patches_per_face = metadata.get('patches_per_face', 8)
    n_sh = metadata.get('n_sh', 9)
    n_patches_per_face = patches_per_face ** 2
    n_patches = 6 * n_patches_per_face

    # Sum over all SH coefficients for illumination patches on the illumination face
    illum_start = illuminate_face * n_patches_per_face
    illum_end = illum_start + n_patches_per_face

    # Compute total exitant radiance for each output patch
    radiance_flat = np.zeros(n_patches)

    for out_patch in range(n_patches):
        # DC coefficient for output
        out_condition = out_patch * n_sh

        # Sum over all illumination patches (DC only for uniform illumination)
        for illum_patch_local in range(n_patches_per_face):
            illum_patch = illum_start + illum_patch_local
            illum_condition = illum_patch * n_sh
            radiance_flat[out_patch] += transfer[out_condition, illum_condition]

    # Reshape to (6, patches_per_face, patches_per_face)
    radiance = radiance_flat.reshape(6, patches_per_face, patches_per_face)

    return radiance


def render_view(
    transfer: np.ndarray,
    metadata: dict,
    illuminate_face: int,
    view_face: int,
    voxels: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "",
):
    """Render a single view of the object."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    patches_per_face = metadata.get('patches_per_face', 8)

    # Compute radiance for the viewed face
    radiance = compute_view_radiance(transfer, metadata, illuminate_face, view_face)

    # Create figure
    if voxels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_vox, ax_rad = axes
    else:
        fig, ax_rad = plt.subplots(1, 1, figsize=(6, 5))
        ax_vox = None

    # Render voxel silhouette if available
    if ax_vox is not None and voxels is not None:
        # Project voxels along view direction
        axis = view_face // 2
        silhouette = voxels.max(axis=axis)

        # Flip to match expected orientation
        if view_face in [0, 2, 4]:  # Positive faces - viewing from positive side
            silhouette = np.flip(silhouette, axis=0)

        ax_vox.imshow(silhouette.T, cmap='gray_r', origin='lower')
        ax_vox.set_title(f'Voxel Silhouette (view from {FACE_NAMES[view_face]})')
        ax_vox.set_xlabel('U')
        ax_vox.set_ylabel('V')
        ax_vox.set_aspect('equal')

    # Render radiance
    # Normalize for display
    if radiance.max() > 0:
        norm = Normalize(vmin=0, vmax=radiance.max())
    else:
        norm = Normalize(vmin=0, vmax=1)

    im = ax_rad.imshow(
        radiance.T,  # Transpose for correct orientation
        cmap='hot',
        origin='lower',
        norm=norm,
        interpolation='nearest',
    )
    plt.colorbar(im, ax=ax_rad, label='Exitant Radiance')

    ax_rad.set_title(f'Light Transport Render\nIlluminate: {FACE_NAMES[illuminate_face]}, View: {FACE_NAMES[view_face]}')
    ax_rad.set_xlabel('U')
    ax_rad.set_ylabel('V')
    ax_rad.set_aspect('equal')

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    else:
        plt.show()


def render_all_views(
    transfer: np.ndarray,
    metadata: dict,
    illuminate_face: int,
    voxels: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "",
):
    """Render all 6 view directions in a grid."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    patches_per_face = metadata.get('patches_per_face', 8)

    # Compute radiance for all faces
    all_radiance = compute_directional_radiance(transfer, metadata, illuminate_face)

    # Find global max for consistent colormap
    vmax = all_radiance.max()
    if vmax == 0:
        vmax = 1
    norm = Normalize(vmin=0, vmax=vmax)

    # Create figure: 2 rows x 3 cols for 6 faces
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for face_idx, ax in enumerate(axes.flat):
        radiance = all_radiance[face_idx]

        # Skip illumination face (it receives light, doesn't emit much)
        if face_idx == illuminate_face:
            ax.text(0.5, 0.5, f'{FACE_NAMES[face_idx]}\n(Illuminated)',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='blue')
            ax.set_facecolor('lightblue')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        im = ax.imshow(
            radiance.T,
            cmap='hot',
            origin='lower',
            norm=norm,
            interpolation='nearest',
        )

        ax.set_title(f'View: {FACE_NAMES[face_idx]}')
        ax.set_xlabel('U')
        ax.set_ylabel('V')
        ax.set_aspect('equal')

    # Add colorbar
    fig.colorbar(im, ax=axes, label='Exitant Radiance', shrink=0.6)

    fig.suptitle(f'{title}\nIlluminated from {FACE_NAMES[illuminate_face]}', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    else:
        plt.show()


def render_cube_unwrap(
    transfer: np.ndarray,
    metadata: dict,
    illuminate_face: int,
    voxels: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "",
):
    """Render an unwrapped cube showing all faces in standard cube net layout.

    Layout (cross pattern):
           +Y
        -X +Z +X -Z
           -Y
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    patches_per_face = metadata.get('patches_per_face', 8)

    # Compute radiance for all faces
    all_radiance = compute_directional_radiance(transfer, metadata, illuminate_face)

    # Find global max for consistent colormap
    vmax = all_radiance.max()
    if vmax == 0:
        vmax = 1
    norm = Normalize(vmin=0, vmax=vmax)

    # Create figure with cube net layout
    # Grid is 4 wide x 3 tall
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Turn off all axes initially
    for ax in axes.flat:
        ax.axis('off')

    # Face positions in the cross layout: (row, col)
    # Mapping: face_idx -> (row, col)
    face_positions = {
        0: (1, 2),  # +X
        1: (1, 0),  # -X
        2: (0, 1),  # +Y
        3: (2, 1),  # -Y
        4: (1, 1),  # +Z (front)
        5: (1, 3),  # -Z (back)
    }

    for face_idx, (row, col) in face_positions.items():
        ax = axes[row, col]
        ax.axis('on')

        radiance = all_radiance[face_idx]

        if face_idx == illuminate_face:
            # Show illumination source
            ax.imshow(np.ones_like(radiance.T) * 0.3, cmap='Blues', vmin=0, vmax=1)
            ax.text(0.5, 0.5, f'{FACE_NAMES[face_idx]}\nLIGHT',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', color='blue')
        else:
            im = ax.imshow(
                radiance.T,
                cmap='hot',
                origin='lower',
                norm=norm,
                interpolation='nearest',
            )

        ax.set_title(FACE_NAMES[face_idx], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
    fig.colorbar(sm, cax=cbar_ax, label='Exitant Radiance')

    fig.suptitle(f'{title}\nCube Unwrap - Illuminated from {FACE_NAMES[illuminate_face]}', fontsize=14)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    else:
        plt.show()


def render_with_voxel_overlay(
    transfer: np.ndarray,
    metadata: dict,
    illuminate_face: int,
    voxels: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "",
):
    """Render radiance with voxel silhouette overlay for each face."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from scipy import ndimage

    patches_per_face = metadata.get('patches_per_face', 8)

    # Compute radiance for all faces
    all_radiance = compute_directional_radiance(transfer, metadata, illuminate_face)

    # Ensure voxels are filled
    voxels_filled = ndimage.binary_fill_holes(voxels > 0)

    # Find global max for consistent colormap
    vmax = all_radiance.max()
    if vmax == 0:
        vmax = 1
    norm = Normalize(vmin=0, vmax=vmax)

    # Create figure: 2 rows x 3 cols for 6 faces
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for face_idx, ax in enumerate(axes.flat):
        radiance = all_radiance[face_idx]

        # Compute voxel silhouette for this view
        axis = face_idx // 2
        silhouette = voxels_filled.max(axis=axis).astype(float)

        # Resize silhouette to match patch resolution
        from scipy.ndimage import zoom
        scale = patches_per_face / silhouette.shape[0]
        silhouette_resized = zoom(silhouette, scale, order=0)

        # Ensure correct size
        silhouette_resized = silhouette_resized[:patches_per_face, :patches_per_face]

        if face_idx == illuminate_face:
            # Show silhouette with blue tint for illumination face
            ax.imshow(silhouette_resized.T, cmap='Blues', origin='lower', alpha=0.7)
            ax.text(0.5, 0.9, 'LIGHT SOURCE',
                   ha='center', va='top', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', color='blue',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Show radiance with silhouette overlay
            im = ax.imshow(
                radiance.T,
                cmap='hot',
                origin='lower',
                norm=norm,
                interpolation='nearest',
            )
            # Overlay silhouette contour
            ax.contour(silhouette_resized.T, levels=[0.5], colors='cyan', linewidths=2, origin='lower')

        ax.set_title(f'{FACE_NAMES[face_idx]}')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar
    fig.colorbar(im, ax=axes, label='Exitant Radiance', shrink=0.6)

    fig.suptitle(f'{title}\nIlluminated from {FACE_NAMES[illuminate_face]} (cyan = object silhouette)', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Render views of an object using its light transport transfer matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('file', type=Path, help="Path to transfer matrix .npz file")
    parser.add_argument('--voxels', type=Path, help="Path to voxels .npz file for overlay")
    parser.add_argument('--illuminate', '-i', default='+X',
                       help="Face to illuminate from (+X, -X, +Y, -Y, +Z, -Z)")
    parser.add_argument('--view', '-v', default='-Z',
                       help="Face to view from (+X, -X, +Y, -Y, +Z, -Z)")
    parser.add_argument('--all-views', action='store_true',
                       help="Render all 6 view directions")
    parser.add_argument('--cube-unwrap', action='store_true',
                       help="Render as unwrapped cube net")
    parser.add_argument('--overlay', action='store_true',
                       help="Overlay voxel silhouette on radiance (requires --voxels)")
    parser.add_argument('-o', '--output', type=Path, help="Save to file instead of displaying")
    parser.add_argument('--title', default="", help="Plot title")

    args = parser.parse_args()

    # Load transfer matrix
    print(f"Loading {args.file}...")
    transfer, metadata = load_transfer_matrix(args.file)
    print(f"Transfer matrix shape: {transfer.shape}")

    # Parse faces
    illuminate_face = parse_face(args.illuminate)
    view_face = parse_face(args.view)

    # Load voxels if specified
    voxels = None
    if args.voxels:
        from scipy import ndimage
        data = np.load(args.voxels)
        voxels = data[list(data.keys())[0]]
        voxels = ndimage.binary_fill_holes(voxels > 0)
        print(f"Loaded voxels: {voxels.shape}")

    # Set title
    title = args.title or args.file.stem

    # Render
    if args.overlay and voxels is not None:
        render_with_voxel_overlay(
            transfer, metadata, illuminate_face, voxels,
            output_path=args.output, title=title
        )
    elif args.cube_unwrap:
        render_cube_unwrap(
            transfer, metadata, illuminate_face, voxels,
            output_path=args.output, title=title
        )
    elif args.all_views:
        render_all_views(
            transfer, metadata, illuminate_face, voxels,
            output_path=args.output, title=title
        )
    else:
        render_view(
            transfer, metadata, illuminate_face, view_face, voxels,
            output_path=args.output, title=title
        )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""3D visualization of light transport results.

Renders the voxel structure and visualizes how light propagates through it
based on the computed transfer matrix.

Usage:
    # View voxel structure with transfer matrix overlay
    python visualize_light_transport_3d.py path/to/transfer.npz

    # View just the voxel structure
    python visualize_light_transport_3d.py --voxels path/to/level_0.npz

    # Illuminate from a specific face and see exitant light
    python visualize_light_transport_3d.py path/to/transfer.npz --illuminate +X
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_transfer_data(path: Path) -> Tuple[np.ndarray, dict]:
    """Load transfer matrix and metadata."""
    data = np.load(path, allow_pickle=True)
    transfer = data['transfer_matrix']

    metadata = {}
    for key in data.files:
        if key == 'transfer_matrix':
            continue
        val = data[key]
        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                val = val.item()
            elif val.size == 1:
                val = val.item()
            else:
                val = val.tolist()
        metadata[key] = val

    return transfer, metadata


def load_voxels_from_metadata(metadata: dict) -> Optional[np.ndarray]:
    """Try to load voxels from source file in metadata."""
    source = metadata.get('source_file')
    if source:
        source_path = Path(source)
        if source_path.exists():
            data = np.load(source_path)
            if 'voxels' in data:
                return data['voxels']
    return None


def create_voxel_mesh(voxels: np.ndarray, threshold: float = 0.5):
    """Create a mesh from voxel data for 3D rendering."""
    # Get occupied voxel positions
    if voxels.dtype == bool:
        occupied = voxels
    else:
        occupied = voxels > threshold

    positions = np.argwhere(occupied)

    if len(positions) == 0:
        return None, None, None

    # Center the voxels
    center = np.array(voxels.shape) / 2
    positions = positions - center

    return positions, occupied, center


def compute_boundary_radiance(
    transfer: np.ndarray,
    metadata: dict,
    incident_face: int = 0,
    incident_sh: int = 0
) -> np.ndarray:
    """Compute exitant radiance on boundary given incident illumination.

    Args:
        transfer: Transfer matrix
        metadata: Transfer matrix metadata
        incident_face: Which face to illuminate (0-5)
        incident_sh: Which SH mode to use (0 = diffuse)

    Returns:
        Array of radiance values per boundary patch
    """
    n_sh = metadata.get('n_sh', 9)
    patches_per_face = metadata.get('patches_per_face', 8)
    patches_per_face_sq = patches_per_face ** 2
    n_patches = 6 * patches_per_face_sq

    # Create incident illumination vector (uniform on one face)
    incident = np.zeros(n_patches * n_sh)

    # Illuminate all patches on the incident face with the specified SH mode
    for patch in range(patches_per_face_sq):
        global_patch = incident_face * patches_per_face_sq + patch
        incident[global_patch * n_sh + incident_sh] = 1.0

    # Apply transfer matrix
    exitant = transfer @ incident

    # Sum over SH modes to get total radiance per patch
    radiance = np.zeros(n_patches)
    for patch in range(n_patches):
        # Sum absolute values of all SH coefficients (approximation to total radiance)
        radiance[patch] = np.sqrt(np.sum(exitant[patch * n_sh:(patch + 1) * n_sh] ** 2))

    return radiance


def get_patch_centers(
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    patches_per_face: int
) -> np.ndarray:
    """Get world-space centers of all boundary patches."""
    n_patches = 6 * patches_per_face ** 2
    centers = np.zeros((n_patches, 3))

    grid_size = grid_max - grid_min

    face_defs = [
        (0, +1, 1, 2),  # +X
        (0, -1, 1, 2),  # -X
        (1, +1, 0, 2),  # +Y
        (1, -1, 0, 2),  # -Y
        (2, +1, 0, 1),  # +Z
        (2, -1, 0, 1),  # -Z
    ]

    idx = 0
    for face_id, (axis, sign, u_axis, v_axis) in enumerate(face_defs):
        face_pos = grid_max[axis] if sign > 0 else grid_min[axis]

        u_min, u_max = grid_min[u_axis], grid_max[u_axis]
        v_min, v_max = grid_min[v_axis], grid_max[v_axis]

        u_step = (u_max - u_min) / patches_per_face
        v_step = (v_max - v_min) / patches_per_face

        for vi in range(patches_per_face):
            for ui in range(patches_per_face):
                u_center = u_min + (ui + 0.5) * u_step
                v_center = v_min + (vi + 0.5) * v_step

                centers[idx, axis] = face_pos
                centers[idx, u_axis] = u_center
                centers[idx, v_axis] = v_center
                idx += 1

    return centers


def visualize_plotly(
    voxels: Optional[np.ndarray],
    transfer: Optional[np.ndarray],
    metadata: dict,
    incident_face: int = 0,
    title: str = "Light Transport 3D View"
):
    """Interactive 3D visualization using Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = go.Figure()

    # Get grid bounds
    if voxels is not None:
        resolution = np.array(voxels.shape)
    else:
        resolution = np.array(metadata.get('resolution', [64, 64, 64]))

    voxel_size = metadata.get('voxel_size', 1.0)
    half_extents = resolution * voxel_size / 2.0
    grid_min = -half_extents
    grid_max = half_extents

    # Plot voxels as scatter points
    if voxels is not None:
        positions, occupied, center = create_voxel_mesh(voxels)
        if positions is not None and len(positions) > 0:
            # Subsample if too many voxels
            max_points = 10000
            if len(positions) > max_points:
                indices = np.random.choice(len(positions), max_points, replace=False)
                positions = positions[indices]

            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='rgba(100, 100, 100, 0.3)',
                    symbol='square'
                ),
                name='Voxels',
                hoverinfo='skip'
            ))

    # Plot boundary patches with radiance coloring
    if transfer is not None:
        patches_per_face = metadata.get('patches_per_face', 8)
        n_patches = 6 * patches_per_face ** 2

        # Compute radiance from incident face
        radiance = compute_boundary_radiance(transfer, metadata, incident_face)

        # Get patch centers
        centers = get_patch_centers(grid_min, grid_max, patches_per_face)

        # Normalize radiance for coloring
        if radiance.max() > 0:
            radiance_norm = radiance / radiance.max()
        else:
            radiance_norm = radiance

        # Color patches by radiance
        colors = np.zeros((n_patches, 4))
        for i in range(n_patches):
            face = i // (patches_per_face ** 2)
            if face == incident_face:
                # Incident face - show in blue
                colors[i] = [0, 0, 1, 0.8]
            else:
                # Exitant faces - show radiance in red/yellow
                intensity = radiance_norm[i]
                colors[i] = [1, intensity, 0, 0.3 + 0.7 * intensity]

        # Convert to plotly color format
        color_strs = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]})' for c in colors]

        # Plot patches
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=color_strs,
                symbol='square',
                line=dict(width=1, color='black')
            ),
            name='Boundary Patches',
            text=[f'Patch {i}<br>Face {i//(patches_per_face**2)}<br>Radiance: {radiance[i]:.4f}'
                  for i in range(n_patches)],
            hoverinfo='text'
        ))

    # Add bounding box wireframe
    corners = np.array([
        [grid_min[0], grid_min[1], grid_min[2]],
        [grid_max[0], grid_min[1], grid_min[2]],
        [grid_max[0], grid_max[1], grid_min[2]],
        [grid_min[0], grid_max[1], grid_min[2]],
        [grid_min[0], grid_min[1], grid_max[2]],
        [grid_max[0], grid_min[1], grid_max[2]],
        [grid_max[0], grid_max[1], grid_max[2]],
        [grid_min[0], grid_max[1], grid_max[2]],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7),  # Verticals
    ]

    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[i, 0], corners[j, 0]],
            y=[corners[i, 1], corners[j, 1]],
            z=[corners[i, 2], corners[j, 2]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add face labels
    face_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    face_positions = [
        [grid_max[0] + 2, 0, 0],
        [grid_min[0] - 2, 0, 0],
        [0, grid_max[1] + 2, 0],
        [0, grid_min[1] - 2, 0],
        [0, 0, grid_max[2] + 2],
        [0, 0, grid_min[2] - 2],
    ]

    for i, (name, pos) in enumerate(zip(face_names, face_positions)):
        color = 'blue' if i == incident_face else 'black'
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='text',
            text=[name],
            textfont=dict(size=14, color=color),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.show()


def visualize_matplotlib(
    voxels: Optional[np.ndarray],
    transfer: Optional[np.ndarray],
    metadata: dict,
    incident_face: int = 0,
    output_path: Optional[Path] = None,
    title: str = "Light Transport 3D View"
):
    """Static 3D visualization using Matplotlib."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 6))

    # Get grid bounds
    if voxels is not None:
        resolution = np.array(voxels.shape)
    else:
        resolution = np.array(metadata.get('resolution', [64, 64, 64]))

    voxel_size = metadata.get('voxel_size', 1.0)
    half_extents = resolution * voxel_size / 2.0
    grid_min = -half_extents
    grid_max = half_extents

    # Left subplot: voxel structure
    ax1 = fig.add_subplot(121, projection='3d')

    if voxels is not None:
        positions, occupied, center = create_voxel_mesh(voxels)
        if positions is not None and len(positions) > 0:
            # Subsample for speed
            max_points = 5000
            if len(positions) > max_points:
                indices = np.random.choice(len(positions), max_points, replace=False)
                positions = positions[indices]

            ax1.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c='gray', alpha=0.3, s=1, marker='s'
            )

    # Draw bounding box
    corners = np.array([
        [grid_min[0], grid_min[1], grid_min[2]],
        [grid_max[0], grid_min[1], grid_min[2]],
        [grid_max[0], grid_max[1], grid_min[2]],
        [grid_min[0], grid_max[1], grid_min[2]],
        [grid_min[0], grid_min[1], grid_max[2]],
        [grid_max[0], grid_min[1], grid_max[2]],
        [grid_max[0], grid_max[1], grid_max[2]],
        [grid_min[0], grid_max[1], grid_max[2]],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        ax1.plot3D(
            [corners[i, 0], corners[j, 0]],
            [corners[i, 1], corners[j, 1]],
            [corners[i, 2], corners[j, 2]],
            'k-', linewidth=0.5, alpha=0.5
        )

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Voxel Structure')

    # Set equal aspect ratio
    max_range = max(grid_max - grid_min) / 2
    mid = (grid_max + grid_min) / 2
    ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Right subplot: boundary radiance
    ax2 = fig.add_subplot(122, projection='3d')

    if transfer is not None:
        patches_per_face = metadata.get('patches_per_face', 8)
        n_patches = 6 * patches_per_face ** 2

        # Compute radiance
        radiance = compute_boundary_radiance(transfer, metadata, incident_face)
        centers = get_patch_centers(grid_min, grid_max, patches_per_face)

        # Normalize
        if radiance.max() > 0:
            radiance_norm = radiance / radiance.max()
        else:
            radiance_norm = radiance

        # Color by face and radiance - use RGBA arrays for consistency
        colors = np.zeros((n_patches, 4))
        sizes = np.zeros(n_patches)
        for i in range(n_patches):
            face = i // (patches_per_face ** 2)
            if face == incident_face:
                colors[i] = [0.0, 0.0, 1.0, 1.0]  # blue
                sizes[i] = 50
            else:
                intensity = radiance_norm[i]
                colors[i] = plt.cm.YlOrRd(intensity)
                sizes[i] = 20 + 80 * intensity

        ax2.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2],
            c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidths=0.5
        )

        # Draw bounding box
        for i, j in edges:
            ax2.plot3D(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                'k-', linewidth=0.5, alpha=0.5
            )

    face_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Exitant Radiance (incident from {face_names[incident_face]})')

    ax2.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax2.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax2.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_all_faces(
    voxels: Optional[np.ndarray],
    transfer: np.ndarray,
    metadata: dict,
    output_path: Optional[Path] = None,
    title: str = "Light Transport - All Incident Faces"
):
    """Show exitant radiance for all 6 incident face directions."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(18, 12))

    # Get grid bounds
    if voxels is not None:
        resolution = np.array(voxels.shape)
    else:
        resolution = np.array(metadata.get('resolution', [64, 64, 64]))

    voxel_size = metadata.get('voxel_size', 1.0)
    half_extents = resolution * voxel_size / 2.0
    grid_min = -half_extents
    grid_max = half_extents

    patches_per_face = metadata.get('patches_per_face', 8)
    n_patches = 6 * patches_per_face ** 2
    centers = get_patch_centers(grid_min, grid_max, patches_per_face)

    face_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']

    # Draw bounding box corners
    corners = np.array([
        [grid_min[0], grid_min[1], grid_min[2]],
        [grid_max[0], grid_min[1], grid_min[2]],
        [grid_max[0], grid_max[1], grid_min[2]],
        [grid_min[0], grid_max[1], grid_min[2]],
        [grid_min[0], grid_min[1], grid_max[2]],
        [grid_max[0], grid_min[1], grid_max[2]],
        [grid_max[0], grid_max[1], grid_max[2]],
        [grid_min[0], grid_max[1], grid_max[2]],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    max_range = max(grid_max - grid_min) / 2
    mid = (grid_max + grid_min) / 2

    for face_idx in range(6):
        ax = fig.add_subplot(2, 3, face_idx + 1, projection='3d')

        # Compute radiance for this incident face
        radiance = compute_boundary_radiance(transfer, metadata, face_idx)

        if radiance.max() > 0:
            radiance_norm = radiance / radiance.max()
        else:
            radiance_norm = radiance

        # Color patches - use RGBA arrays for consistency
        colors = np.zeros((n_patches, 4))
        sizes = np.zeros(n_patches)
        for i in range(n_patches):
            patch_face = i // (patches_per_face ** 2)
            if patch_face == face_idx:
                colors[i] = [0.0, 0.0, 1.0, 1.0]  # blue
                sizes[i] = 30
            else:
                intensity = radiance_norm[i]
                colors[i] = plt.cm.YlOrRd(intensity)
                sizes[i] = 10 + 40 * intensity

        ax.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2],
            c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidths=0.3
        )

        # Draw bounding box
        for i, j in edges:
            ax.plot3D(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                'k-', linewidth=0.3, alpha=0.3
            )

        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.set_title(f'Incident: {face_names[face_idx]}', fontsize=10)

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.tick_params(labelsize=6)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="3D visualization of light transport results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('file', nargs='?', help='Path to transfer matrix .npz file')
    parser.add_argument('--voxels', type=Path, help='Path to voxels .npz file (optional)')
    parser.add_argument('-o', '--output', type=Path, help='Save to file instead of displaying')
    parser.add_argument('--illuminate', choices=['+X', '-X', '+Y', '-Y', '+Z', '-Z'],
                       default='+X', help='Face to illuminate (default: +X)')
    parser.add_argument('--all-faces', action='store_true',
                       help='Show all 6 incident face directions')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive Plotly visualization')
    parser.add_argument('--title', type=str, default=None, help='Plot title')

    args = parser.parse_args()

    if args.file is None and args.voxels is None:
        parser.print_help()
        return 1

    # Load transfer matrix if provided
    transfer = None
    metadata = {}
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            return 1
        transfer, metadata = load_transfer_data(path)
        print(f"Loaded transfer matrix: {transfer.shape}")

    # Load voxels
    voxels = None
    if args.voxels:
        if not args.voxels.exists():
            print(f"Voxels file not found: {args.voxels}")
            return 1
        data = np.load(args.voxels)
        voxels = data['voxels']
        print(f"Loaded voxels: {voxels.shape}, {voxels.sum()} occupied")
    elif transfer is not None:
        # Try to load from metadata
        voxels = load_voxels_from_metadata(metadata)
        if voxels is not None:
            print(f"Loaded voxels from source: {voxels.shape}")

    # Convert face name to index
    face_map = {'+X': 0, '-X': 1, '+Y': 2, '-Y': 3, '+Z': 4, '-Z': 5}
    incident_face = face_map[args.illuminate]

    # Generate title
    title = args.title
    if title is None:
        if args.file:
            title = Path(args.file).stem
        else:
            title = "Voxel Structure"

    # Visualize
    if args.all_faces and transfer is not None:
        visualize_all_faces(voxels, transfer, metadata, args.output, title)
    elif args.interactive:
        try:
            visualize_plotly(voxels, transfer, metadata, incident_face, title)
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")
            print("Falling back to matplotlib...")
            visualize_matplotlib(voxels, transfer, metadata, incident_face, args.output, title)
    else:
        visualize_matplotlib(voxels, transfer, metadata, incident_face, args.output, title)

    return 0


if __name__ == '__main__':
    sys.exit(main())

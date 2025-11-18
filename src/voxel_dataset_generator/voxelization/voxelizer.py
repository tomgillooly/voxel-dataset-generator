"""Mesh to voxel conversion with normalization."""

import numpy as np
import trimesh
from pathlib import Path
from typing import Optional, Tuple
from scipy import ndimage


class Voxelizer:
    """Converts 3D meshes to voxel grids with consistent resolution.

    The voxelizer normalizes mesh sizes by adjusting the voxel pitch to ensure
    all objects are voxelized to the same grid dimensions regardless of their
    original size.
    """

    def __init__(self, target_resolution: int = 128, solid: bool = True):
        """Initialize the voxelizer.

        Args:
            target_resolution: Target grid size (e.g., 128 for 128^3 grid)
            solid: If True, fill interior of mesh (default). If False, only voxelize surface.
        """
        self.target_resolution = target_resolution
        self.solid = solid

    def load_mesh(self, file_path: Path | str) -> trimesh.Trimesh:
        """Load a mesh from an STL file.

        Args:
            file_path: Path to STL file

        Returns:
            Loaded trimesh object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be loaded as a valid mesh
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {file_path}")

        try:
            mesh = trimesh.load(file_path)
            # Handle case where trimesh.load returns a Scene instead of Trimesh
            if isinstance(mesh, trimesh.Scene):
                # Combine all geometries in the scene
                mesh = trimesh.util.concatenate(
                    [geom for geom in mesh.geometry.values()]
                )
            return mesh
        except Exception as e:
            raise ValueError(f"Failed to load mesh from {file_path}: {e}")

    def create_mesh_from_arrays(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> trimesh.Trimesh:
        """Create a trimesh object from vertex and face arrays.

        This is useful for loading meshes from npz files (e.g., Thingi10k npz format).

        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices

        Returns:
            Trimesh object

        Raises:
            ValueError: If arrays are invalid
        """
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh
        except Exception as e:
            raise ValueError(f"Failed to create mesh from arrays: {e}")

    def compute_voxel_pitch(self, mesh: trimesh.Trimesh) -> float:
        """Compute voxel pitch to normalize mesh to target resolution.

        The pitch is calculated as max(bbox_dimensions) / target_resolution
        to ensure the largest dimension fits exactly in the target grid.

        Args:
            mesh: Input mesh

        Returns:
            Voxel pitch (size of each voxel cube)
        """
        bbox_extents = mesh.bounding_box.extents
        max_extent = np.max(bbox_extents)
        pitch = max_extent / self.target_resolution
        return pitch

    def voxelize(
        self,
        mesh: trimesh.Trimesh,
        pitch: Optional[float] = None,
        solid: Optional[bool] = None
    ) -> Tuple[np.ndarray, dict]:
        """Convert mesh to binary voxel grid.

        Args:
            mesh: Input mesh
            pitch: Voxel pitch (computed automatically if not provided)
            solid: Override instance solid setting. If True, fill interior.
                  If False, only voxelize surface. If None, use instance setting.

        Returns:
            Tuple of (voxel_grid, metadata) where:
                - voxel_grid: Binary 3D array of shape (res, res, res)
                - metadata: Dict with voxelization info (pitch, bbox, etc.)
        """
        if pitch is None:
            pitch = self.compute_voxel_pitch(mesh)

        if solid is None:
            solid = self.solid

        # Voxelize using trimesh
        # Note: trimesh.voxel.creation.voxelize returns a VoxelGrid object
        voxel_grid = mesh.voxelized(pitch=pitch)
        
        # Fill interior if solid voxelization requested
        if solid:
            voxel_grid.fill()

        # Get the binary occupancy matrix
        # The matrix may not be exactly target_resolution^3 due to rounding
        matrix = voxel_grid.matrix

        # Pad or crop to ensure exact target resolution
        matrix = self._ensure_resolution(matrix, self.target_resolution)

        # Prepare metadata
        metadata = {
            "voxel_pitch": float(pitch),
            "original_bbox_min": mesh.bounds[0].tolist(),
            "original_bbox_max": mesh.bounds[1].tolist(),
            "original_bbox_extents": mesh.bounding_box.extents.tolist(),
            "num_occupied_voxels": int(np.sum(matrix)),
            "occupancy_ratio": float(np.sum(matrix) / matrix.size),
        }

        return matrix.astype(bool), metadata

    def _ensure_resolution(
        self,
        matrix: np.ndarray,
        target_res: int
    ) -> np.ndarray:
        """Ensure voxel matrix is exactly target_res^3.

        If the matrix is smaller, pad with zeros (empty voxels).
        If larger, crop from the center.

        Args:
            matrix: Input voxel matrix
            target_res: Target resolution

        Returns:
            Matrix of shape (target_res, target_res, target_res)
        """
        current_shape = matrix.shape

        # If already correct size, return as is
        if current_shape == (target_res, target_res, target_res):
            return matrix

        # Create output array (empty)
        output = np.zeros((target_res, target_res, target_res), dtype=matrix.dtype)

        # Compute how to center the data
        for axis in range(3):
            current_size = current_shape[axis]

            if current_size > target_res:
                # Crop from center
                start = (current_size - target_res) // 2
                if axis == 0:
                    matrix = matrix[start:start + target_res, :, :]
                elif axis == 1:
                    matrix = matrix[:, start:start + target_res, :]
                else:
                    matrix = matrix[:, :, start:start + target_res]

        # Now pad if needed
        current_shape = matrix.shape
        slices = []
        for axis in range(3):
            current_size = current_shape[axis]
            if current_size < target_res:
                start = (target_res - current_size) // 2
                slices.append(slice(start, start + current_size))
            else:
                slices.append(slice(None))

        output[slices[0], slices[1], slices[2]] = matrix

        return output

    def voxelize_file(
        self,
        file_path: Path | str
    ) -> Tuple[np.ndarray, dict]:
        """Load and voxelize a mesh file in one step.

        Args:
            file_path: Path to mesh file (STL)

        Returns:
            Tuple of (voxel_grid, metadata)
        """
        mesh = self.load_mesh(file_path)
        return self.voxelize(mesh)

    def voxelize_from_arrays(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """Create mesh from arrays and voxelize in one step.

        This is a convenience method for working with npz format meshes
        (e.g., Thingi10k npz downloads).

        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices

        Returns:
            Tuple of (voxel_grid, metadata)

        Example:
            >>> # Load using thingi10k library
            >>> from thingi10k import Thingi10k
            >>> thingi = Thingi10k()
            >>> mesh_data = thingi[0].load()  # Returns vertices and faces
            >>> voxels, metadata = voxelizer.voxelize_from_arrays(
            ...     mesh_data['vertices'], mesh_data['faces']
            ... )
        """
        mesh = self.create_mesh_from_arrays(vertices, faces)
        return self.voxelize(mesh)

    def save_voxels(
        self,
        voxel_grid: np.ndarray,
        output_path: Path | str,
        compressed: bool = True
    ):
        """Save voxel grid to disk.

        Args:
            voxel_grid: Binary voxel grid
            output_path: Output file path
            compressed: If True, save as .npz (compressed), else .npy
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if compressed:
            # Save as compressed sparse format if mostly empty
            np.savez_compressed(output_path, voxels=voxel_grid)
        else:
            np.save(output_path, voxel_grid)

"""Recursive octree subdivision of voxel grids."""

import hashlib
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Subvolume:
    """Represents a sub-volume in the hierarchical structure.

    Attributes:
        level: Subdivision level (0 = top level, 1+ = subdivisions)
        octant_index: Index of octant in parent (0-7)
        position: (x, y, z) position of corner in parent grid (local coordinates)
        global_position: (x, y, z) position relative to top-level grid (global coordinates)
        data: Binary voxel data
        hash: SHA-256 hash of voxel data
        is_empty: Whether the sub-volume contains only empty voxels
    """

    level: int
    octant_index: int
    position: Tuple[int, int, int]
    global_position: Tuple[int, int, int]
    data: np.ndarray
    hash: str
    is_empty: bool


class Subdivider:
    """Recursively subdivide voxel grids into octree hierarchy.

    Each volume is divided into 8 equal octants (2x2x2 subdivision).
    Subdivision continues until the minimum resolution is reached.
    """

    def __init__(self, min_resolution: int = 4):
        """Initialize the subdivider.

        Args:
            min_resolution: Minimum voxel grid size (must be power of 2)
        """
        if not self._is_power_of_two(min_resolution):
            raise ValueError(f"min_resolution must be a power of 2, got {min_resolution}")

        self.min_resolution = min_resolution

    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        """Check if n is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    def compute_hash(self, voxel_data: np.ndarray) -> str:
        """Compute SHA-256 hash of voxel data.

        Args:
            voxel_data: Binary voxel array

        Returns:
            Hexadecimal hash string
        """
        # Convert to bytes for hashing (use bool to ensure consistency)
        data_bytes = voxel_data.astype(bool).tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def subdivide(
        self,
        voxel_grid: np.ndarray,
        level: int = 0,
        global_offset: Tuple[int, int, int] = (0, 0, 0)
    ) -> List[Subvolume]:
        """Subdivide a voxel grid into 8 octants.

        Args:
            voxel_grid: Binary voxel grid of shape (n, n, n) where n is power of 2
            level: Current subdivision level
            global_offset: Offset of this grid in the top-level coordinate system

        Returns:
            List of 8 Subvolume objects (one per octant)

        Raises:
            ValueError: If grid is not cubic or size is not power of 2
        """
        shape = voxel_grid.shape

        # Validate input
        if len(shape) != 3 or shape[0] != shape[1] or shape[1] != shape[2]:
            raise ValueError(f"Voxel grid must be cubic, got shape {shape}")

        grid_size = shape[0]
        if not self._is_power_of_two(grid_size):
            raise ValueError(f"Grid size must be power of 2, got {grid_size}")

        # Calculate half size for octants
        half = grid_size // 2

        # Extract 8 octants in consistent order
        subvolumes = []
        octant_idx = 0

        for i in [0, half]:
            for j in [0, half]:
                for k in [0, half]:
                    # Extract sub-volume
                    sub_data = voxel_grid[
                        i:i + half,
                        j:j + half,
                        k:k + half
                    ].copy()

                    # Compute hash and check if empty
                    data_hash = self.compute_hash(sub_data)
                    is_empty = not np.any(sub_data)

                    # Compute global position
                    global_pos = (
                        global_offset[0] + i,
                        global_offset[1] + j,
                        global_offset[2] + k
                    )

                    subvolume = Subvolume(
                        level=level + 1,
                        octant_index=octant_idx,
                        position=(i, j, k),
                        global_position=global_pos,
                        data=sub_data,
                        hash=data_hash,
                        is_empty=is_empty
                    )

                    subvolumes.append(subvolume)
                    octant_idx += 1

        return subvolumes

    def subdivide_recursive(
        self,
        voxel_grid: np.ndarray,
        current_level: int = 0,
        max_depth: int = None,
        global_offset: Tuple[int, int, int] = (0, 0, 0)
    ) -> Dict[int, List[Subvolume]]:
        """Recursively subdivide voxel grid to all levels.

        Args:
            voxel_grid: Top-level voxel grid
            current_level: Current recursion level (internal)
            max_depth: Maximum subdivision depth (None = subdivide to min_resolution)
            global_offset: Offset in top-level coordinate system (internal)

        Returns:
            Dictionary mapping level number to list of sub-volumes at that level
        """
        result = {}
        grid_size = voxel_grid.shape[0]

        # Check if we should stop subdividing
        if grid_size < self.min_resolution * 2:
            return result

        if max_depth is not None and current_level >= max_depth:
            return result

        # Subdivide current grid
        subvolumes = self.subdivide(voxel_grid, level=current_level, global_offset=global_offset)
        result[current_level + 1] = subvolumes

        # Recursively subdivide each non-empty octant
        next_level_subvolumes = []
        for subvol in subvolumes:
            if subvol.data.shape[0] >= self.min_resolution * 2:
                # Recursively subdivide this sub-volume
                # Pass the global position of this subvolume as the offset for its children
                sub_result = self.subdivide_recursive(
                    subvol.data,
                    current_level=current_level + 1,
                    max_depth=max_depth,
                    global_offset=subvol.global_position
                )

                # Merge results
                for level, subvols in sub_result.items():
                    if level not in result:
                        result[level] = []
                    result[level].extend(subvols)

        return result

    def subdivide_all_levels(
        self,
        voxel_grid: np.ndarray
    ) -> Dict[int, List[Subvolume]]:
        """Subdivide voxel grid to all possible levels.

        This is a convenience method that subdivides from the top level
        all the way down to min_resolution.

        Args:
            voxel_grid: Top-level voxel grid

        Returns:
            Dictionary mapping level to list of sub-volumes

        Example:
            For a 128^3 grid with min_resolution=4:
            {
                1: [8 sub-volumes of size 64^3],
                2: [64 sub-volumes of size 32^3],
                3: [512 sub-volumes of size 16^3],
                4: [4096 sub-volumes of size 8^3],
                5: [32768 sub-volumes of size 4^3]
            }
        """
        return self.subdivide_recursive(voxel_grid, current_level=0, max_depth=None)

    def to_flat_list(
        self,
        subdivisions: Dict[int, List[Subvolume]],
        object_id: str
    ) -> List[Dict]:
        """Convert subdivision hierarchy to flat list for dataframe export.

        Args:
            subdivisions: Dictionary of subdivisions by level
            object_id: ID of the parent object

        Returns:
            List of dictionaries, one per sub-volume
        """
        records = []

        for level, subvolumes in subdivisions.items():
            for subvol in subvolumes:
                record = {
                    "object_id": object_id,
                    "level": level,
                    "octant_index": subvol.octant_index,
                    "position_x": subvol.position[0],
                    "position_y": subvol.position[1],
                    "position_z": subvol.position[2],
                    "hash": subvol.hash,
                    "is_empty": subvol.is_empty,
                }
                records.append(record)

        return records

    def verify_subdivision(
        self,
        parent: np.ndarray,
        subvolumes: List[Subvolume]
    ) -> bool:
        """Verify that subdivision is lossless (union equals parent).

        Args:
            parent: Parent voxel grid
            subvolumes: List of 8 sub-volumes

        Returns:
            True if subdivision is valid

        Raises:
            AssertionError: If subdivision doesn't match parent
        """
        if len(subvolumes) != 8:
            raise AssertionError(f"Expected 8 sub-volumes, got {len(subvolumes)}")

        # Reconstruct parent from sub-volumes
        half = parent.shape[0] // 2
        reconstructed = np.zeros_like(parent)

        for subvol in subvolumes:
            i, j, k = subvol.position
            reconstructed[i:i + half, j:j + half, k:k + half] = subvol.data

        # Check if reconstructed matches original
        if not np.array_equal(parent, reconstructed):
            raise AssertionError("Subdivision does not preserve parent data")

        return True

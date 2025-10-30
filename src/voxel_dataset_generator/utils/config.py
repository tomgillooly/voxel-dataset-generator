"""Configuration management for voxel dataset generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for voxel dataset generation.

    Attributes:
        base_resolution: Target resolution for top-level voxel grid (e.g., 128)
        min_resolution: Minimum resolution for subdivision (e.g., 4)
        num_levels: Number of subdivision levels
        output_dir: Base directory for dataset output
        use_sparse: Whether to use sparse voxel representation
        compression: Whether to compress voxel data (.npz vs .npy)
        solid_voxelization: If True, fill interior of meshes (default: True)
    """

    base_resolution: int = 128
    min_resolution: int = 4
    output_dir: Path = Path("dataset")
    use_sparse: bool = False
    compression: bool = True
    solid_voxelization: bool = True

    def __post_init__(self):
        """Validate configuration and compute derived values."""
        if not self._is_power_of_two(self.base_resolution):
            raise ValueError(f"base_resolution must be a power of 2, got {self.base_resolution}")

        if not self._is_power_of_two(self.min_resolution):
            raise ValueError(f"min_resolution must be a power of 2, got {self.min_resolution}")

        if self.base_resolution < self.min_resolution:
            raise ValueError(
                f"base_resolution ({self.base_resolution}) must be >= "
                f"min_resolution ({self.min_resolution})"
            )

        # Ensure output directory exists
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def num_levels(self) -> int:
        """Calculate number of subdivision levels based on resolutions."""
        import math
        return int(math.log2(self.base_resolution) - math.log2(self.min_resolution)) + 1

    @property
    def level_resolutions(self) -> list[int]:
        """Get resolution at each subdivision level."""
        resolutions = []
        current = self.base_resolution
        while current >= self.min_resolution:
            resolutions.append(current)
            current //= 2
        return resolutions

    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        """Check if n is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    def get_object_dir(self, object_id: str) -> Path:
        """Get directory path for a specific object."""
        obj_dir = self.output_dir / "objects" / f"object_{object_id:0>4}"
        obj_dir.mkdir(parents=True, exist_ok=True)
        return obj_dir

    def get_subvolume_dir(self, level: int) -> Path:
        """Get directory path for sub-volumes at a specific level."""
        subvol_dir = self.output_dir / "subvolumes" / f"level_{level}"
        subvol_dir.mkdir(parents=True, exist_ok=True)
        return subvol_dir

    def get_metadata_path(self) -> Path:
        """Get path to dataset-level metadata file."""
        return self.output_dir / "metadata.json"

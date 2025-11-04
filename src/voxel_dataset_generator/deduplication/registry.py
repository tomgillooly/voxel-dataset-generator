"""Global registry for sub-volume deduplication."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Set
from collections import defaultdict


class SubvolumeRegistry:
    """Manages deduplication of sub-volumes across the dataset.

    The registry tracks unique sub-volumes by their hash and stores only
    one copy of each unique sub-volume to disk. It maintains reference
    counts and allows querying of sub-volume statistics.
    """

    def __init__(self, base_dir: Path):
        """Initialize the registry.

        Args:
            base_dir: Base directory for storing sub-volumes
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Registry data structures
        # hash -> {"level": int, "count": int, "path": str, "is_empty": bool}
        self._registry: Dict[str, Dict] = {}

        # level -> set of hashes at that level
        self._level_hashes: Dict[int, Set[str]] = defaultdict(set)

        # Track statistics
        self._total_subvolumes = 0
        self._unique_subvolumes = 0

    def register(
        self,
        data_hash: str,
        voxel_data: np.ndarray,
        level: int,
        is_empty: bool = False,
        save_to_disk: bool = True
    ) -> bool:
        """Register a sub-volume in the registry.

        Args:
            data_hash: Hash of the voxel data
            voxel_data: Binary voxel array
            level: Subdivision level
            is_empty: Whether sub-volume is empty
            save_to_disk: Whether to save to disk if new

        Returns:
            True if this is a new unique sub-volume, False if duplicate
        """
        self._total_subvolumes += 1

        # Check if hash already exists
        if data_hash in self._registry:
            # Increment reference count
            self._registry[data_hash]["count"] += 1
            return False

        # New unique sub-volume
        self._unique_subvolumes += 1

        # Save to disk if requested
        file_path = None
        if save_to_disk:
            file_path = self._get_subvolume_path(level, data_hash)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(file_path, voxels=voxel_data)

        # Add to registry
        self._registry[data_hash] = {
            "level": level,
            "count": 1,
            "path": str(file_path) if file_path else None,
            "is_empty": is_empty,
            "resolution": voxel_data.shape[0],
        }

        # Add to level index
        self._level_hashes[level].add(data_hash)

        return True

    def get_count(self, data_hash: str) -> int:
        """Get reference count for a sub-volume hash.

        Args:
            data_hash: Hash of the sub-volume

        Returns:
            Number of times this sub-volume appears in the dataset
        """
        if data_hash not in self._registry:
            return 0
        return self._registry[data_hash]["count"]

    def get_path(self, data_hash: str) -> Optional[Path]:
        """Get file path for a sub-volume.

        Args:
            data_hash: Hash of the sub-volume

        Returns:
            Path to .npz file, or None if not found or empty
        """
        if data_hash not in self._registry:
            return None

        path_str = self._registry[data_hash]["path"]
        return Path(path_str) if path_str else None

    def load_subvolume(self, data_hash: str) -> Optional[np.ndarray]:
        """Load a sub-volume's voxel data from disk.

        Args:
            data_hash: Hash of the sub-volume

        Returns:
            Voxel data array, or None if not found
        """
        path = self.get_path(data_hash)
        if path is None or not path.exists():
            return None

        data = np.load(path)
        return data["voxels"]

    def get_level_stats(self, level: int) -> Dict:
        """Get statistics for a specific subdivision level.

        Args:
            level: Subdivision level

        Returns:
            Dictionary with statistics
        """
        hashes_at_level = self._level_hashes.get(level, set())

        total_count = sum(
            self._registry[h]["count"]
            for h in hashes_at_level
        )

        empty_count = sum(
            1 for h in hashes_at_level
            if self._registry[h]["is_empty"]
        )

        return {
            "level": level,
            "unique": len(hashes_at_level),
            "total": total_count,
            "empty": empty_count,
            "deduplication_ratio": (
                len(hashes_at_level) / total_count
                if total_count > 0 else 0
            ),
        }

    def get_overall_stats(self) -> Dict:
        """Get overall registry statistics.

        Returns:
            Dictionary with overall statistics
        """
        levels = sorted(self._level_hashes.keys())

        return {
            "total_subvolumes": self._total_subvolumes,
            "unique_subvolumes": self._unique_subvolumes,
            "overall_deduplication_ratio": (
                self._unique_subvolumes / self._total_subvolumes
                if self._total_subvolumes > 0 else 0
            ),
            "levels": levels,
            "level_stats": {
                level: self.get_level_stats(level)
                for level in levels
            },
        }

    def get_most_common(self, level: int, top_k: int = 10) -> list:
        """Get most frequently occurring sub-volumes at a level.

        Args:
            level: Subdivision level
            top_k: Number of top results to return

        Returns:
            List of (hash, count) tuples, sorted by count descending
        """
        hashes_at_level = self._level_hashes.get(level, set())

        # Get counts and sort
        hash_counts = [
            (h, self._registry[h]["count"])
            for h in hashes_at_level
        ]

        hash_counts.sort(key=lambda x: x[1], reverse=True)

        return hash_counts[:top_k]

    def save_registry(self, output_path: Optional[Path] = None):
        """Save registry metadata to JSON.

        Args:
            output_path: Path to save registry (default: base_dir/registry.json)
        """
        if output_path is None:
            output_path = self.base_dir / "registry.json"

        registry_data = {
            "stats": self.get_overall_stats(),
            "registry": self._registry,
        }

        with open(output_path, "w") as f:
            json.dump(registry_data, f, indent=2)

    def load_registry(self, input_path: Optional[Path] = None):
        """Load registry metadata from JSON.

        Args:
            input_path: Path to load registry from (default: base_dir/registry.json)
        """
        if input_path is None:
            input_path = self.base_dir / "registry.json"

        if not input_path.exists():
            return

        with open(input_path, "r") as f:
            registry_data = json.load(f)

        self._registry = registry_data["registry"]

        # Rebuild level index
        self._level_hashes = defaultdict(set)
        for hash_val, info in self._registry.items():
            level = info["level"]
            self._level_hashes[level].add(hash_val)

        # Update counters
        stats = registry_data["stats"]
        self._total_subvolumes = stats["total_subvolumes"]
        self._unique_subvolumes = stats["unique_subvolumes"]

    def _get_subvolume_path(self, level: int, data_hash: str) -> Path:
        """Get file path for storing a sub-volume.

        Args:
            level: Subdivision level
            data_hash: Hash of the sub-volume

        Returns:
            Path to .npz file
        """
        level_dir = self.base_dir / "subvolumes" / f"level_{level}"
        # Use first 2 chars of hash as subdirectory to avoid too many files per dir
        subdir = level_dir / data_hash[:2]
        return subdir / f"{data_hash}.npz"

"""
Train/Validation split functionality that respects hierarchical relationships.

This module assigns train/val/test splits at the top-level object and tracks
which sub-volumes appear in which splits. It provides statistics on sub-volume
sharing across splits.

For validation objects, it calculates what percentage of their sub-volumes are
"pure" (only in validation) vs "leaked" (also appear in training objects).

Note: Completely empty and completely full sub-volumes are tracked separately
as they contain no useful information.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict
import json
import random
import numpy as np
from enum import Enum


class Split(str, Enum):
    """Dataset split types."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class SplitConfig:
    """Configuration for train/val/test splitting."""
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    seed: Optional[int] = 42

    def __post_init__(self):
        """Validate that ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )
        if self.train_ratio < 0 or self.val_ratio < 0 or self.test_ratio < 0:
            raise ValueError("Split ratios must be non-negative")


class HierarchicalSplitter:
    """
    Handles train/val/test splitting and tracks sub-volume membership across splits.

    Key principles:
    1. Split assignment happens at the top-level object (level 0)
    2. All sub-volumes of an object are tracked as belonging to that object's split
    3. Sub-volumes can appear in multiple objects across different splits
    4. Statistics track what percentage of each object's sub-volumes are "pure"
       (only in that split) vs "shared" (also in other splits)
    """

    def __init__(self, config: SplitConfig):
        """
        Initialize the splitter.

        Args:
            config: Split configuration specifying ratios and seed
        """
        self.config = config

        # Object assignments
        self._object_splits: Dict[str, Split] = {}  # object_id -> split

        # Track which objects use each hash: hash -> {object_id -> split}
        self._hash_object_usage: Dict[str, Dict[str, Split]] = defaultdict(dict)

        # Track trivial hashes (all 0s or all 1s) separately
        self._trivial_hashes: Set[str] = set()

        if config.seed is not None:
            random.seed(config.seed)

    def assign_splits(self, object_ids: List[str]) -> Dict[str, Split]:
        """
        Assign splits to a list of top-level objects.

        Args:
            object_ids: List of object IDs to assign to splits

        Returns:
            Dictionary mapping object_id to Split
        """
        # Shuffle for randomization
        shuffled_ids = object_ids.copy()
        random.shuffle(shuffled_ids)

        # Calculate split boundaries
        n = len(shuffled_ids)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        # Assign splits
        for i, obj_id in enumerate(shuffled_ids):
            if i < train_end:
                self._object_splits[obj_id] = Split.TRAIN
            elif i < val_end:
                self._object_splits[obj_id] = Split.VAL
            else:
                self._object_splits[obj_id] = Split.TEST

        return self._object_splits.copy()

    def get_object_split(self, object_id: str) -> Optional[Split]:
        """Get the split assignment for a top-level object."""
        return self._object_splits.get(object_id)

    @staticmethod
    def is_trivial_subvolume(voxel_data: np.ndarray) -> bool:
        """
        Check if a sub-volume is trivial (all 0s or all 1s).

        Args:
            voxel_data: The voxel grid data

        Returns:
            True if all voxels are 0 or all voxels are 1
        """
        # Handle both boolean and numeric arrays
        return np.all(~voxel_data.astype(bool)) or np.all(voxel_data.astype(bool))

    def register_subvolume(
        self,
        object_id: str,
        hash_val: str,
        voxel_data: Optional[np.ndarray] = None,
        is_trivial: Optional[bool] = None
    ) -> None:
        """
        Register that an object uses a specific sub-volume hash.

        This should be called during dataset generation when a sub-volume is
        encountered for an object.

        Args:
            object_id: The top-level object this sub-volume belongs to
            hash_val: The hash of the sub-volume
            voxel_data: The voxel data (optional, for trivial check)
            is_trivial: Pre-computed trivial flag (if voxel_data not provided)
        """
        if object_id not in self._object_splits:
            raise ValueError(f"Object {object_id} has not been assigned a split")

        obj_split = self._object_splits[object_id]

        # Track that this object uses this hash
        self._hash_object_usage[hash_val][object_id] = obj_split

        # Determine and mark if trivial
        if is_trivial is None:
            if voxel_data is not None:
                is_trivial = self.is_trivial_subvolume(voxel_data)
            else:
                # If we can't determine, assume non-trivial
                is_trivial = False

        if is_trivial:
            self._trivial_hashes.add(hash_val)

    def is_trivial_hash(self, hash_val: str) -> bool:
        """Check if a hash is marked as trivial."""
        return hash_val in self._trivial_hashes

    def get_splits_using_hash(self, hash_val: str) -> Set[Split]:
        """
        Get all splits that contain this hash.

        Args:
            hash_val: The hash to query

        Returns:
            Set of splits that contain objects using this hash
        """
        usage = self._hash_object_usage.get(hash_val, {})
        return set(usage.values())

    def get_objects_using_hash(self, hash_val: str) -> Dict[str, Split]:
        """
        Get all objects that use a specific sub-volume hash.

        Args:
            hash_val: The hash to query

        Returns:
            Dictionary mapping object_id -> split for all objects using this hash
        """
        return self._hash_object_usage.get(hash_val, {}).copy()

    def analyze_object_subvolumes(
        self,
        object_id: str,
        subvolume_hashes: List[str]
    ) -> Dict:
        """
        Analyze what percentage of an object's sub-volumes are pure vs shared.

        Args:
            object_id: The object to analyze
            subvolume_hashes: List of all hash values for this object's sub-volumes

        Returns:
            Dictionary with statistics about purity and sharing
        """
        if object_id not in self._object_splits:
            raise ValueError(f"Object {object_id} has not been assigned a split")

        obj_split = self._object_splits[object_id]

        total = len(subvolume_hashes)
        pure = 0  # Only in this split
        shared_same_split = 0  # Shared with other objects in same split
        shared_other_split = 0  # Shared with objects in different splits (leakage)
        trivial = 0

        for hash_val in subvolume_hashes:
            if hash_val in self._trivial_hashes:
                trivial += 1
                continue

            splits_using = self.get_splits_using_hash(hash_val)

            if len(splits_using) == 1 and obj_split in splits_using:
                # Check if shared with other objects in same split
                objects_using = self.get_objects_using_hash(hash_val)
                if len(objects_using) == 1:
                    pure += 1
                else:
                    shared_same_split += 1
            else:
                # Appears in other splits
                shared_other_split += 1

        return {
            "object_id": object_id,
            "split": obj_split.value,
            "total_subvolumes": total,
            "trivial": trivial,
            "nontrivial": total - trivial,
            "pure": pure,
            "shared_same_split": shared_same_split,
            "shared_other_split": shared_other_split,
            "purity_percentage": (pure / (total - trivial) * 100) if (total - trivial) > 0 else 0,
            "leakage_percentage": (shared_other_split / (total - trivial) * 100) if (total - trivial) > 0 else 0
        }

    def get_split_statistics(self) -> Dict:
        """
        Compute overall statistics about the split distribution.

        Returns:
            Dictionary with object and sub-volume counts per split
        """
        object_counts = {
            Split.TRAIN: 0,
            Split.VAL: 0,
            Split.TEST: 0
        }

        for split in self._object_splits.values():
            object_counts[split] += 1

        # Analyze hash distribution
        hash_split_distribution = {
            "only_train": 0,
            "only_val": 0,
            "only_test": 0,
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
            "all_splits": 0,
            "trivial": len(self._trivial_hashes)
        }

        for hash_val, usage in self._hash_object_usage.items():
            if hash_val in self._trivial_hashes:
                continue

            splits = set(usage.values())

            if splits == {Split.TRAIN}:
                hash_split_distribution["only_train"] += 1
            elif splits == {Split.VAL}:
                hash_split_distribution["only_val"] += 1
            elif splits == {Split.TEST}:
                hash_split_distribution["only_test"] += 1
            elif splits == {Split.TRAIN, Split.VAL}:
                hash_split_distribution["train_val"] += 1
            elif splits == {Split.TRAIN, Split.TEST}:
                hash_split_distribution["train_test"] += 1
            elif splits == {Split.VAL, Split.TEST}:
                hash_split_distribution["val_test"] += 1
            elif len(splits) == 3:
                hash_split_distribution["all_splits"] += 1

        return {
            "objects": {
                "train": object_counts[Split.TRAIN],
                "val": object_counts[Split.VAL],
                "test": object_counts[Split.TEST],
                "total": len(self._object_splits)
            },
            "unique_nontrivial_hashes": {
                "total": len(self._hash_object_usage) - len(self._trivial_hashes),
                "distribution": hash_split_distribution
            },
            "trivial_hashes": {
                "count": len(self._trivial_hashes)
            }
        }

    def save_split_assignments(self, output_path: Path) -> None:
        """
        Save split assignments to a JSON file.

        Args:
            output_path: Path to save the split assignments
        """
        data = {
            "config": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
                "seed": self.config.seed
            },
            "object_splits": {
                obj_id: split.value
                for obj_id, split in self._object_splits.items()
            },
            "hash_object_usage": {
                hash_val: {obj_id: split.value for obj_id, split in usage.items()}
                for hash_val, usage in self._hash_object_usage.items()
            },
            "trivial_hashes": list(self._trivial_hashes),
            "statistics": self.get_split_statistics()
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_split_assignments(cls, input_path: Path) -> 'HierarchicalSplitter':
        """
        Load split assignments from a JSON file.

        Args:
            input_path: Path to the saved split assignments

        Returns:
            HierarchicalSplitter instance with loaded assignments
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        config_data = data["config"]
        config = SplitConfig(
            train_ratio=config_data["train_ratio"],
            val_ratio=config_data["val_ratio"],
            test_ratio=config_data["test_ratio"],
            seed=config_data.get("seed")
        )

        splitter = cls(config)

        # Restore assignments
        splitter._object_splits = {
            obj_id: Split(split_val)
            for obj_id, split_val in data["object_splits"].items()
        }

        # Restore hash object usage
        for hash_val, usage_dict in data.get("hash_object_usage", {}).items():
            splitter._hash_object_usage[hash_val] = {
                obj_id: Split(split_val)
                for obj_id, split_val in usage_dict.items()
            }

        splitter._trivial_hashes = set(data.get("trivial_hashes", []))

        return splitter

    def get_objects_by_split(self, split: Split) -> List[str]:
        """
        Get all object IDs assigned to a specific split.

        Args:
            split: The split to query

        Returns:
            List of object IDs in that split
        """
        return [
            obj_id for obj_id, obj_split in self._object_splits.items()
            if obj_split == split
        ]

    def export_subvolume_split_info(self) -> Dict[str, Dict]:
        """
        Export detailed information about which splits each sub-volume appears in,
        including which objects use it.

        Returns:
            Dictionary mapping hash -> {
                "splits": [list of splits this hash appears in],
                "is_trivial": bool,
                "used_by_objects": {object_id: split_str}
            }
        """
        result = {}

        for hash_val, usage in self._hash_object_usage.items():
            is_trivial = hash_val in self._trivial_hashes
            splits = list(set(split.value for split in usage.values()))

            result[hash_val] = {
                "splits": splits,
                "is_trivial": is_trivial,
                "used_by_objects": {
                    obj_id: split.value for obj_id, split in usage.items()
                }
            }

        return result

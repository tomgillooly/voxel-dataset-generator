"""Metadata generation and management utilities."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class MetadataWriter:
    """Handles creation and writing of metadata files."""

    @staticmethod
    def write_dataset_metadata(
        output_path: Path,
        config: Dict,
        stats: Dict,
        num_objects: int
    ):
        """Write dataset-level metadata.

        Args:
            output_path: Path to metadata.json
            config: Configuration dictionary
            stats: Statistics from registry
            num_objects: Number of objects processed
        """
        metadata = {
            "dataset_name": "Thingi10k_Hierarchical_Voxels",
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "base_resolution": config.get("base_resolution", 128),
            "min_resolution": config.get("min_resolution", 4),
            "num_levels": len(config.get("level_resolutions", [])),
            "level_resolutions": config.get("level_resolutions", []),
            "num_objects": num_objects,
            "compression_enabled": config.get("compression", True),
            "deduplication_stats": stats.get("level_stats", {}),
            "overall_stats": {
                "total_subvolumes": stats.get("total_subvolumes", 0),
                "unique_subvolumes": stats.get("unique_subvolumes", 0),
                "deduplication_ratio": stats.get("overall_deduplication_ratio", 0),
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def write_object_metadata(
        output_path: Path,
        object_id: str,
        source_file: str,
        voxel_metadata: Dict,
        source_hash: Optional[str] = None
    ):
        """Write object-level metadata.

        Args:
            output_path: Path to object's metadata.json
            object_id: Object ID
            source_file: Original STL file path
            voxel_metadata: Metadata from voxelization
            source_hash: Hash of source mesh for resume verification (optional)
        """
        metadata = {
            "object_id": object_id,
            "source_file": str(source_file),
            "voxel_pitch": voxel_metadata.get("voxel_pitch"),
            "original_bbox_min": voxel_metadata.get("original_bbox_min"),
            "original_bbox_max": voxel_metadata.get("original_bbox_max"),
            "original_bbox_extents": voxel_metadata.get("original_bbox_extents"),
            "num_occupied_voxels": voxel_metadata.get("num_occupied_voxels"),
            "occupancy_ratio": voxel_metadata.get("occupancy_ratio"),
        }

        # Add source hash if provided (for resume support)
        if source_hash is not None:
            metadata["source_hash"] = source_hash

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def write_subdivision_map(
        output_path: Path,
        subdivision_records: List[Dict]
    ):
        """Write subdivision map as flat JSON array.

        Args:
            output_path: Path to subdivision_map.json
            subdivision_records: List of subdivision records
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(subdivision_records, f, indent=2)

    @staticmethod
    def read_subdivision_map(input_path: Path) -> List[Dict]:
        """Read subdivision map from JSON.

        Args:
            input_path: Path to subdivision_map.json

        Returns:
            List of subdivision records
        """
        with open(input_path, "r") as f:
            return json.load(f)


class MetadataAnalyzer:
    """Analyze metadata across the dataset."""

    def __init__(self, dataset_dir: Path):
        """Initialize analyzer.

        Args:
            dataset_dir: Root directory of dataset
        """
        self.dataset_dir = Path(dataset_dir)

    def load_all_subdivision_maps(self) -> List[Dict]:
        """Load all subdivision maps into a flat list.

        Returns:
            List of all subdivision records across all objects
        """
        objects_dir = self.dataset_dir / "objects"
        all_records = []

        for subdiv_file in objects_dir.glob("*/subdivision_map.json"):
            records = MetadataWriter.read_subdivision_map(subdiv_file)
            all_records.extend(records)

        return all_records

    def get_hash_frequency(self, level: Optional[int] = None) -> Dict[str, int]:
        """Get frequency count for each unique hash.

        Args:
            level: If specified, only count hashes at this level

        Returns:
            Dictionary mapping hash to frequency count
        """
        records = self.load_all_subdivision_maps()

        if level is not None:
            records = [r for r in records if r["level"] == level]

        frequency = {}
        for record in records:
            hash_val = record["hash"]
            frequency[hash_val] = frequency.get(hash_val, 0) + 1

        return frequency

    def get_object_complexity(self) -> Dict[str, int]:
        """Get number of unique sub-volumes per object.

        Returns:
            Dictionary mapping object_id to count of unique hashes
        """
        records = self.load_all_subdivision_maps()

        object_hashes = {}
        for record in records:
            obj_id = record["object_id"]
            if obj_id not in object_hashes:
                object_hashes[obj_id] = set()
            object_hashes[obj_id].add(record["hash"])

        return {
            obj_id: len(hashes)
            for obj_id, hashes in object_hashes.items()
        }

    def export_to_polars_compatible(self, output_path: Path):
        """Export all subdivision data to Polars-friendly format.

        This creates a single NDJSON file that can be efficiently loaded
        by Polars for analysis.

        Args:
            output_path: Path to output NDJSON file
        """
        import json

        records = self.load_all_subdivision_maps()

        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics across the dataset.

        Returns:
            Dictionary with various statistics
        """
        records = self.load_all_subdivision_maps()

        # Group by level
        level_stats = {}
        for record in records:
            level = record["level"]
            if level not in level_stats:
                level_stats[level] = {
                    "count": 0,
                    "empty_count": 0,
                    "unique_hashes": set()
                }

            level_stats[level]["count"] += 1
            if record.get("is_empty", False):
                level_stats[level]["empty_count"] += 1
            level_stats[level]["unique_hashes"].add(record["hash"])

        # Convert sets to counts
        for level in level_stats:
            level_stats[level]["unique"] = len(level_stats[level]["unique_hashes"])
            del level_stats[level]["unique_hashes"]

        return {
            "total_records": len(records),
            "num_objects": len(set(r["object_id"] for r in records)),
            "levels": sorted(level_stats.keys()),
            "level_statistics": level_stats,
        }

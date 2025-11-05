"""Main pipeline for generating hierarchical voxel datasets."""

import gc
import numpy as np
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm, trange
import hashlib
import json

from collections import defaultdict

from .voxelization.voxelizer import Voxelizer
from .subdivision.subdivider import Subdivider
from .deduplication.registry import SubvolumeRegistry
from .utils.config import Config
from .utils.metadata import MetadataWriter
from .splitting.splitter import HierarchicalSplitter, SplitConfig, Split


class DatasetGenerator:
    """Main pipeline for generating hierarchical voxel datasets.

    This class orchestrates the entire process:
    1. Voxelization of STL meshes
    2. Recursive subdivision into octree hierarchy
    3. Deduplication of sub-volumes
    4. Metadata generation
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the dataset generator.

        Args:
            config: Configuration object (uses defaults if not provided)
        """
        self.config = config or Config()

        # Initialize components
        self.voxelizer = Voxelizer(
            target_resolution=self.config.base_resolution,
            solid=self.config.solid_voxelization
        )
        self.subdivider = Subdivider(min_resolution=self.config.min_resolution)
        self.registry = SubvolumeRegistry(base_dir=self.config.output_dir)

        # Store split config for later use (splitting happens post-processing)
        self.split_config = None
        if self.config.enable_splitting:
            self.split_config = SplitConfig(
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                seed=self.config.split_seed
            )

        # Statistics
        self.num_processed = 0
        self._processed_object_ids = []  # Track object IDs for split assignment

    def _compute_source_hash(self, mesh_path: Optional[Path] = None,
                            vertices: Optional[np.ndarray] = None,
                            faces: Optional[np.ndarray] = None,
                            source_name: Optional[str] = None) -> str:
        """Compute a hash of the source mesh to verify we're processing the same object.

        Args:
            mesh_path: Path to mesh file
            vertices: Vertex array
            faces: Face array
            source_name: Source name for array-based meshes

        Returns:
            SHA256 hash string
        """
        hasher = hashlib.sha256()

        if mesh_path is not None:
            # Hash the file path
            hasher.update(str(mesh_path).encode('utf-8'))
        elif vertices is not None and faces is not None:
            # Hash the array data
            hasher.update(vertices.tobytes())
            hasher.update(faces.tobytes())
            if source_name:
                hasher.update(source_name.encode('utf-8'))

        return hasher.hexdigest()

    def _verify_object_complete(self, object_id: str, source_hash: str) -> bool:
        """Check if an object has been fully processed.

        Verifies:
        1. Object directory exists
        2. Source hash matches (same mesh)
        3. All required files exist (metadata.json, subdivision_map.json, level_0.npz)
        4. All subvolumes referenced in subdivision_map exist

        Args:
            object_id: The object ID to check
            source_hash: Hash of the source mesh to verify identity

        Returns:
            True if object is complete and matches the source, False otherwise
        """
        obj_dir = self.config.get_object_dir(object_id)

        # Check if directory exists
        if not obj_dir.exists():
            return False

        # Check for required files
        metadata_path = obj_dir / "metadata.json"
        subdivision_map_path = obj_dir / "subdivision_map.json"
        level_0_path = obj_dir / "level_0.npz"

        if not metadata_path.exists() or not subdivision_map_path.exists():
            return False

        # Read metadata to check source hash
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Check if source hash exists and matches
            stored_hash = metadata.get('source_hash')
            if stored_hash is None or stored_hash != source_hash:
                # Hash mismatch or not present - different object
                return False

        except (json.JSONDecodeError, IOError):
            return False

        # Check level_0.npz if save_voxels is expected
        # Note: We'll check this exists, but it's optional depending on save_voxels flag

        # Read subdivision map and verify all subvolumes exist
        try:
            with open(subdivision_map_path, 'r') as f:
                subdivision_records = json.load(f)
        except (json.JSONDecodeError, IOError):
            return False

        # Group subvolumes by level and hash
        subvolumes_by_level = defaultdict(set)
        for record in subdivision_records:
            level = record['level']
            hash_val = record['hash']
            subvolumes_by_level[level].add(hash_val)

        # Verify each unique subvolume file exists
        for level, hashes in subvolumes_by_level.items():
            for hash_val in hashes:
                npz_path = self.registry._get_subvolume_path(level, hash_val)
                
                if not npz_path.exists():# and not npy_path.exists():
                    # Missing subvolume file
                    return False

        return True

    def assign_splits_from_disk(self, object_ids: Optional[List[str]] = None):
        """
        Assign train/val/test splits to objects by reading subdivision maps from disk.

        This is a memory-efficient approach that doesn't require holding subvolume
        data in memory during subdivision. It reads the saved subdivision maps to
        register which subvolumes belong to which objects.

        Args:
            object_ids: List of object IDs to assign splits to.
                       If None, uses processed object IDs from tracking.

        Returns:
            HierarchicalSplitter instance with splits assigned
        """
        if not self.config.enable_splitting or self.split_config is None:
            return None

        if object_ids is None:
            object_ids = self._processed_object_ids

        if not object_ids:
            print("No objects to assign splits to")
            return None

        print(f"\nAssigning splits to {len(object_ids)} objects...")

        # Create splitter and assign object-level splits
        splitter = HierarchicalSplitter(self.split_config)
        splitter.assign_splits(object_ids)

        print(f"  Train: {len(splitter.get_objects_by_split(Split.TRAIN))}")
        print(f"  Val: {len(splitter.get_objects_by_split(Split.VAL))}")
        if self.config.test_ratio > 0:
            print(f"  Test: {len(splitter.get_objects_by_split(Split.TEST))}")

        # Register subvolumes by reading subdivision maps from disk
        print(f"\nRegistering subvolumes from disk...")
        for obj_id in tqdm(object_ids, desc="Reading subdivision maps"):
            obj_dir = self.config.get_object_dir(obj_id)
            subdivision_map_path = obj_dir / "subdivision_map.json"

            if not subdivision_map_path.exists():
                print(f"Warning: Subdivision map not found for object {obj_id}")
                continue

            # Read subdivision map
            with open(subdivision_map_path, 'r') as f:
                subdivision_records = json.load(f)

            # Register each subvolume hash
            # Track unique hashes per object to avoid duplicate registrations
            registered_hashes = set()
            for record in subdivision_records:
                hash_val = record['hash']

                # Skip if already registered for this object
                if hash_val in registered_hashes:
                    continue
                registered_hashes.add(hash_val)

                # Determine if trivial based on is_empty flag
                # Note: We approximate triviality with is_empty for memory efficiency
                # A more accurate check would require loading the voxel data
                is_trivial = record.get('is_empty', False)

                splitter.register_subvolume(
                    object_id=obj_id,
                    hash_val=hash_val,
                    voxel_data=None,
                    is_trivial=is_trivial
                )

        print("Subvolume registration complete")
        return splitter

    def process_mesh(
        self,
        object_id: str,
        mesh_path: Optional[Path] = None,
        save_voxels: bool = True,
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        source_name: Optional[str] = None,
        skip_if_complete: bool = True
    ) -> dict:
        """Process a single mesh through the full pipeline.

        Args:
            object_id: Unique identifier for this object
            mesh_path: Path to STL file (used if vertices/faces not provided)
            save_voxels: Whether to save top-level voxel grid to disk
            vertices: Nx3 array of vertex coordinates (alternative to mesh_path)
            faces: Mx3 array of face indices (alternative to mesh_path)
            source_name: Source identifier for metadata (used with vertices/faces)
            skip_if_complete: If True, skip processing if object already complete

        Returns:
            Dictionary with processing results

        Raises:
            ValueError: If neither mesh_path nor vertices/faces provided

        Examples:
            >>> # From file
            >>> generator.process_mesh(object_id="0001", mesh_path=Path("mesh.stl"))

            >>> # From arrays
            >>> generator.process_mesh(
            ...     object_id="0001",
            ...     vertices=verts,
            ...     faces=faces,
            ...     source_name="thingi10k/12345"
            ... )
        """
        # Compute source hash for resume verification
        source_hash = self._compute_source_hash(
            mesh_path=mesh_path,
            vertices=vertices,
            faces=faces,
            source_name=source_name
        )

        # Check if object is already complete
        if skip_if_complete and self._verify_object_complete(object_id, source_hash):
            # Object already processed, load existing metadata for stats
            obj_dir = self.config.get_object_dir(object_id)
            subdivision_map_path = obj_dir / "subdivision_map.json"

            with open(subdivision_map_path, 'r') as f:
                subdivision_records = json.load(f)

            metadata_path = obj_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Track for splitting if needed
            if self.config.enable_splitting and object_id not in self._processed_object_ids:
                self._processed_object_ids.append(object_id)

            self.num_processed += 1

            return {
                "object_id": object_id,
                "num_subvolumes": len(subdivision_records),
                "new_unique_subvolumes": 0,  # Already existed
                "occupancy_ratio": metadata.get("occupancy_ratio", 0.0),
                "skipped": True
            }

        # Step 1: Voxelize the mesh
        if vertices is not None and faces is not None:
            # Use array-based voxelization
            voxel_grid, voxel_metadata = self.voxelizer.voxelize_from_arrays(
                vertices, faces
            )
            source_file = source_name or "npz_arrays"
        elif mesh_path is not None:
            # Use file-based voxelization
            voxel_grid, voxel_metadata = self.voxelizer.voxelize_file(mesh_path)
            source_file = str(mesh_path)
        else:
            raise ValueError(
                "Must provide either mesh_path or both vertices and faces arrays"
            )

        # Get object directory
        obj_dir = self.config.get_object_dir(object_id)

        # Save top-level voxel grid
        if save_voxels:
            voxel_path = obj_dir / "level_0.npz"
            self.voxelizer.save_voxels(
                voxel_grid,
                voxel_path,
                compressed=self.config.compression
            )

        # Track object for splitting
        if self.config.enable_splitting and object_id not in self._processed_object_ids:
            self._processed_object_ids.append(object_id)

        # Step 2: Subdivide into hierarchy
        # Note: Subdivisions use views (not copies) for memory efficiency
        subdivisions = self.subdivider.subdivide_all_levels(voxel_grid)

        # Step 3: Register sub-volumes and deduplicate
        subdivision_records = []
        new_unique_count = 0

        for level, subvolumes in subdivisions.items():
            for subvol in subvolumes:
                # Register in global registry
                is_new = self.registry.register(
                    data_hash=subvol.hash,
                    voxel_data=subvol.data,
                    level=level,
                    is_empty=subvol.is_empty,
                    save_to_disk=True
                )

                if is_new:
                    new_unique_count += 1

                # Create record for subdivision map
                record = {
                    "object_id": object_id,
                    "level": level,
                    "octant_index": subvol.octant_index,
                    "position_x": subvol.position[0],
                    "position_y": subvol.position[1],
                    "position_z": subvol.position[2],
                    "global_position_x": subvol.global_position[0],
                    "global_position_y": subvol.global_position[1],
                    "global_position_z": subvol.global_position[2],
                    "hash": subvol.hash,
                    "is_empty": subvol.is_empty,
                }
                
                subdivision_records.append(record)

        # Step 4: Write metadata (including source hash for resume support)
        MetadataWriter.write_object_metadata(
            output_path=obj_dir / "metadata.json",
            object_id=object_id,
            source_file=source_file,
            voxel_metadata=voxel_metadata,
            source_hash=source_hash
        )

        MetadataWriter.write_subdivision_map(
            output_path=obj_dir / "subdivision_map.json",
            subdivision_records=subdivision_records
        )

        self.num_processed += 1

        # Store result before freeing memory
        result = {
            "object_id": object_id,
            "num_subvolumes": len(subdivision_records),
            "new_unique_subvolumes": new_unique_count,
            "occupancy_ratio": voxel_metadata["occupancy_ratio"],
        }

        # Free memory from subdivisions and voxel grid
        # Safe to delete now that all subvolumes have been saved to disk
        del voxel_grid
        del subdivisions
        del subdivision_records

        # Force garbage collection to free memory immediately
        gc.collect()

        return result

    def process_batch(
        self,
        mesh_files: List[Path],
        start_id: int = 0,
        show_progress: bool = True,
        skip_if_complete: bool = True
    ) -> List[dict]:
        """Process a batch of mesh files.

        Args:
            mesh_files: List of paths to STL files
            start_id: Starting object ID number
            show_progress: Whether to show progress bar
            skip_if_complete: If True, skip objects that are already complete

        Returns:
            List of processing results
        """
        results = []
        skipped_count = 0

        iterator = enumerate(mesh_files, start=start_id)
        pbar = tqdm(iterator, total=len(mesh_files), desc="Processing meshes") if show_progress else iterator

        for idx, mesh_path in pbar:
            object_id = f"{idx:04d}"

            try:
                result = self.process_mesh(
                    object_id=object_id,
                    mesh_path=mesh_path,
                    skip_if_complete=skip_if_complete
                )
                results.append(result)

                if result.get("skipped", False):
                    skipped_count += 1
                    if show_progress:
                        pbar.set_postfix(skipped=skipped_count, refresh=False)  # type: ignore

            except Exception as e:
                print(f"Error processing {mesh_path}: {e}")
                results.append({
                    "object_id": object_id,
                    "error": str(e)
                })

        if show_progress and skipped_count > 0:
            print(f"Skipped {skipped_count} already-complete objects")

        return results

    def finalize(self):
        """Finalize dataset generation by writing global metadata."""
        # Assign splits by reading from disk (memory-efficient approach)
        splitter = None
        if self.config.enable_splitting and self._processed_object_ids:
            splitter = self.assign_splits_from_disk()

        # Get statistics from registry
        stats = self.registry.get_overall_stats()

        # Write dataset-level metadata
        config_dict = {
            "base_resolution": self.config.base_resolution,
            "min_resolution": self.config.min_resolution,
            "level_resolutions": self.config.level_resolutions,
            "compression": self.config.compression,
        }

        # Add split configuration if enabled
        if self.config.enable_splitting:
            config_dict["splitting"] = {
                "enabled": True,
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
                "seed": self.config.split_seed
            }

        MetadataWriter.write_dataset_metadata(
            output_path=self.config.get_metadata_path(),
            config=config_dict,
            stats=stats,
            num_objects=self.num_processed
        )

        # Save registry
        self.registry.save_registry()

        # Save split assignments if enabled
        if splitter is not None:
            split_path = self.config.output_dir / "splits.json"
            splitter.save_split_assignments(split_path)
            print(f"\nSaved split assignments to {split_path}")

        print(f"\nDataset generation complete!")
        print(f"Processed {self.num_processed} objects")
        print(f"Total sub-volumes: {stats['total_subvolumes']}")
        print(f"Unique sub-volumes: {stats['unique_subvolumes']}")
        print(f"Deduplication ratio: {stats['overall_deduplication_ratio']:.2%}")

        # Print per-level statistics
        print("\nPer-level statistics:")
        for level, level_stats in stats["level_stats"].items():
            print(f"  Level {level}: {level_stats['unique']:,} unique / "
                  f"{level_stats['total']:,} total "
                  f"({level_stats['deduplication_ratio']:.2%} dedup ratio)")

        # Print split statistics if enabled
        if splitter is not None:
            print("\nSplit statistics:")
            split_stats = splitter.get_split_statistics()
            print(f"  Objects: Train={split_stats['objects']['train']}, "
                  f"Val={split_stats['objects']['val']}, "
                  f"Test={split_stats['objects']['test']}")

            hash_dist = split_stats['unique_nontrivial_hashes']['distribution']
            print(f"  Unique non-trivial hashes:")
            print(f"    Only in Train: {hash_dist['only_train']}")
            print(f"    Only in Val: {hash_dist['only_val']}")
            if self.config.test_ratio > 0:
                print(f"    Only in Test: {hash_dist['only_test']}")
            print(f"    Shared across splits: Train-Val={hash_dist['train_val']}, "
                  f"Train-Test={hash_dist['train_test']}, "
                  f"Val-Test={hash_dist['val_test']}, "
                  f"All={hash_dist['all_splits']}")
            print(f"  Trivial hashes (shared across all splits): "
                  f"{split_stats['trivial_hashes']['count']}")


def generate_dataset_from_thingi10k(
    start_object: int = 0,
    num_objects: int = 100,
    output_dir: Path = Path("dataset"),
    base_resolution: int = 128,
    min_resolution: int = 4,
    download_dir: Optional[Path] = None,
    resume: bool = True
):
    """Generate dataset from Thingi10k.

    This is a convenience function that downloads objects from Thingi10k
    and processes them through the pipeline.

    Args:
        num_objects: Number of objects to process
        output_dir: Output directory for dataset
        base_resolution: Top-level voxel resolution
        min_resolution: Minimum subdivision resolution
        download_dir: Directory to download STL files (default: ./thingi10k_cache)
        resume: If True, skip already-processed objects (default: True)
    """
    import thingi10k

    # Setup
    download_dir = download_dir or Path("thingi10k_cache")
    download_dir.mkdir(exist_ok=True)

    # Initialize Thingi10k dataset
    thingi10k.init(variant='raw', cache_dir=download_dir)
    dataset = thingi10k.dataset()

    # Create configuration
    config = Config(
        base_resolution=base_resolution,
        min_resolution=min_resolution,
        output_dir=output_dir,
        solid_voxelization=True,
        enable_splitting=True
    )

    # Create generator
    generator = DatasetGenerator(config)

    print(f"Downloading and processing {num_objects} objects from Thingi10k...")
    if resume:
        print("Resume mode enabled - skipping already-complete objects")

    processed_things = defaultdict(bool)
    skipped_count = 0

    # Download and process objects
    processed = 0
    for data_idx in trange(start_object, min(num_objects+start_object, len(dataset)), desc="Overall progress"):
        try:
            # Download object
            object_info = dataset[data_idx]
                
            if processed_things[object_info['thing_id']]:
                continue
            processed_things[object_info['thing_id']] = True

            mesh_path = Path(object_info['file_path'])

            # Download if not cached
            if not mesh_path.exists():
                raise FileNotFoundError(f"File {mesh_path} does not exist locally.")

            # Process through pipeline
            object_id = f"{processed:04d}"
            result = generator.process_mesh(
                object_id=object_id,
                mesh_path=mesh_path,
                skip_if_complete=resume
            )

            if result.get("skipped", False):
                skipped_count += 1
                pbar.set_postfix(skipped=skipped_count, refresh=False)

        except Exception as e:
            print(f"Error with object {processed}: {e}")
            continue

    pbar.close()

    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} already-complete objects")

    # Finalize
    generator.finalize()

    print(f"\nSuccessfully processed {processed}/{num_objects} objects")


def main():
    """Example main function for running the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate hierarchical voxel dataset from Thingi10k"
    )
    parser.add_argument(
        "--start-object",
        type=int,
        default=0,
        help="Object index to start processing at"
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=100,
        help="Number of objects to process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Output directory"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Base resolution (must be power of 2)"
    )
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=4,
        help="Minimum subdivision resolution"
    )

    args = parser.parse_args()

    generate_dataset_from_thingi10k(
        start_object=args.start_object,
        num_objects=args.num_objects,
        output_dir=args.output_dir,
        base_resolution=args.resolution,
        min_resolution=args.min_resolution
    )


if __name__ == "__main__":
    main()

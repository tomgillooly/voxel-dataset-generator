"""Main pipeline for generating hierarchical voxel datasets."""

import numpy as np
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

from collections import defaultdict

from .voxelization.voxelizer import Voxelizer
from .subdivision.subdivider import Subdivider
from .deduplication.registry import SubvolumeRegistry
from .utils.config import Config
from .utils.metadata import MetadataWriter


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

        # Statistics
        self.num_processed = 0

    def process_mesh(
        self,
        object_id: str,
        mesh_path: Optional[Path] = None,
        save_voxels: bool = True,
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        source_name: Optional[str] = None
    ) -> dict:
        """Process a single mesh through the full pipeline.

        Args:
            object_id: Unique identifier for this object
            mesh_path: Path to STL file (used if vertices/faces not provided)
            save_voxels: Whether to save top-level voxel grid to disk
            vertices: Nx3 array of vertex coordinates (alternative to mesh_path)
            faces: Mx3 array of face indices (alternative to mesh_path)
            source_name: Source identifier for metadata (used with vertices/faces)

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

        # Step 2: Subdivide into hierarchy
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

        # Step 4: Write metadata
        MetadataWriter.write_object_metadata(
            output_path=obj_dir / "metadata.json",
            object_id=object_id,
            source_file=source_file,
            voxel_metadata=voxel_metadata
        )

        MetadataWriter.write_subdivision_map(
            output_path=obj_dir / "subdivision_map.json",
            subdivision_records=subdivision_records
        )

        self.num_processed += 1

        return {
            "object_id": object_id,
            "num_subvolumes": len(subdivision_records),
            "new_unique_subvolumes": new_unique_count,
            "occupancy_ratio": voxel_metadata["occupancy_ratio"],
        }

    def process_batch(
        self,
        mesh_files: List[Path],
        start_id: int = 0,
        show_progress: bool = True
    ) -> List[dict]:
        """Process a batch of mesh files.

        Args:
            mesh_files: List of paths to STL files
            start_id: Starting object ID number
            show_progress: Whether to show progress bar

        Returns:
            List of processing results
        """
        results = []

        iterator = enumerate(mesh_files, start=start_id)
        if show_progress:
            iterator = tqdm(iterator, total=len(mesh_files), desc="Processing meshes")

        for idx, mesh_path in iterator:
            object_id = f"{idx:04d}"

            try:
                result = self.process_mesh(object_id=object_id, mesh_path=mesh_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {mesh_path}: {e}")
                results.append({
                    "object_id": object_id,
                    "error": str(e)
                })

        return results

    def finalize(self):
        """Finalize dataset generation by writing global metadata."""
        # Get statistics from registry
        stats = self.registry.get_overall_stats()

        # Write dataset-level metadata
        config_dict = {
            "base_resolution": self.config.base_resolution,
            "min_resolution": self.config.min_resolution,
            "level_resolutions": self.config.level_resolutions,
            "compression": self.config.compression,
        }

        MetadataWriter.write_dataset_metadata(
            output_path=self.config.get_metadata_path(),
            config=config_dict,
            stats=stats,
            num_objects=self.num_processed
        )

        # Save registry
        self.registry.save_registry()

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


def generate_dataset_from_thingi10k(
    num_objects: int = 100,
    output_dir: Path = Path("dataset"),
    base_resolution: int = 128,
    min_resolution: int = 4,
    download_dir: Optional[Path] = None
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
    """
    import thingi10k

    # Setup
    download_dir = download_dir or Path("thingi10k_cache")
    download_dir.mkdir(exist_ok=True)

    # Initialize Thingi10k dataset
    thingi10k.init(variant='raw')
    dataset = thingi10k.dataset()
    data_iter = iter(dataset)

    # Create configuration
    config = Config(
        base_resolution=base_resolution,
        min_resolution=min_resolution,
        output_dir=output_dir,
        solid_voxelization=True
    )

    # Create generator
    generator = DatasetGenerator(config)

    print(f"Downloading and processing {num_objects} objects from Thingi10k...")

    pbar = tqdm(range(num_objects), desc="Overall progress")

    processed_things = defaultdict(bool)

    # Download and process objects
    processed = 0
    while processed < num_objects:
        try:
            # Download object
            object_info = next(data_iter)
            if processed_things[object_info['thing_id']]:
                continue
            processed_things[object_info['thing_id']] = True
            
            mesh_path = Path(object_info['file_path'])

            # Download if not cached
            if not mesh_path.exists():
                raise FileNotFoundError(f"File {mesh_path} does not exist locally.")

            # Process through pipeline
            object_id = f"{processed:04d}"
            generator.process_mesh(object_id=object_id, mesh_path=mesh_path)

            processed += 1
            pbar.update(1)

        except Exception as e:
            print(f"Error with object {processed}: {e}")
            continue

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
        default=128,
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
        num_objects=args.num_objects,
        output_dir=args.output_dir,
        base_resolution=args.resolution,
        min_resolution=args.min_resolution
    )


if __name__ == "__main__":
    main()

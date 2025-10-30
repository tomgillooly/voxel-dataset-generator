"""Example of processing meshes from numpy arrays (e.g., Thingi10k npz format)."""

from pathlib import Path
import numpy as np
from voxel_dataset_generator import Config
from voxel_dataset_generator.pipeline import DatasetGenerator


def example_from_thingi10k_npz():
    """Process Thingi10k objects using the npz format directly."""
    from thingi10k import Thingi10k

    # Initialize Thingi10k dataset
    thingi = Thingi10k()

    # Create configuration
    config = Config(
        base_resolution=128,
        min_resolution=4,
        output_dir=Path("output/thingi10k_from_arrays")
    )

    # Create generator
    generator = DatasetGenerator(config)

    # Process first 10 objects using array format
    num_objects = 10
    for idx in range(num_objects):
        try:
            # Get object from Thingi10k
            obj = thingi[idx]

            # Load mesh data as arrays (instead of downloading STL)
            # The thingi10k library provides vertices and faces directly
            mesh_data = obj.load()  # Returns dict with 'vertices' and 'faces'

            # Process using arrays directly
            object_id = f"{idx:04d}"
            result = generator.process_mesh(
                object_id=object_id,
                vertices=mesh_data['vertices'],
                faces=mesh_data['faces'],
                source_name=f"thingi10k/{obj.id}"
            )

            print(f"Processed object {object_id}: {result['num_subvolumes']} sub-volumes")

        except Exception as e:
            print(f"Error processing object {idx}: {e}")
            continue

    # Finalize dataset
    generator.finalize()


def example_from_npz_file():
    """Process a mesh from a local npz file."""

    # Assume you have an npz file with vertices and faces
    npz_path = Path("mesh_data.npz")

    if not npz_path.exists():
        print(f"NPZ file not found: {npz_path}")
        print("Creating a test npz file...")

        # Create a simple test mesh
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2],
            [1, 2, 4],
            [0, 1, 3],
            [1, 3, 5],
            [0, 2, 3],
            [2, 3, 6],
            [7, 6, 5],
            [7, 6, 4],
            [7, 5, 4],
            [7, 5, 1],
            [7, 4, 2],
            [7, 2, 6]
        ], dtype=np.int32)

        np.savez(npz_path, vertices=vertices, faces=faces)
        print(f"Created test npz file: {npz_path}")

    # Load the npz file
    data = np.load(npz_path)
    vertices = data['vertices']
    faces = data['faces']

    # Create configuration
    config = Config(
        base_resolution=64,
        min_resolution=4,
        output_dir=Path("output/from_npz")
    )

    # Create generator and process
    generator = DatasetGenerator(config)

    result = generator.process_mesh(
        object_id="0001",
        vertices=vertices,
        faces=faces,
        source_name=str(npz_path)
    )

    print(f"\nProcessed mesh from {npz_path}")
    print(f"  Sub-volumes: {result['num_subvolumes']}")
    print(f"  New unique: {result['new_unique_subvolumes']}")
    print(f"  Occupancy ratio: {result['occupancy_ratio']:.2%}")

    generator.finalize()


def example_mixed_sources():
    """Process meshes from both files and arrays in the same dataset."""

    config = Config(
        base_resolution=128,
        min_resolution=4,
        output_dir=Path("output/mixed_sources")
    )

    generator = DatasetGenerator(config)

    # Process from STL file
    stl_path = Path("path/to/mesh.stl")
    if stl_path.exists():
        generator.process_mesh(
            object_id="0001",
            mesh_path=stl_path
        )
        print("Processed STL file")

    # Process from arrays
    vertices = np.random.rand(100, 3)
    faces = np.random.randint(0, 100, size=(50, 3))

    generator.process_mesh(
        object_id="0002",
        vertices=vertices,
        faces=faces,
        source_name="procedural_mesh"
    )
    print("Processed array mesh")

    generator.finalize()


if __name__ == "__main__":
    print("Voxel Dataset Generator - Array Processing Examples")
    print("=" * 60)

    # Uncomment the example you want to run:

    # example_from_thingi10k_npz()
    example_from_npz_file()
    # example_mixed_sources()

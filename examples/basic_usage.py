"""Basic usage example for the voxel dataset generator."""

from pathlib import Path
from voxel_dataset_generator import Config
from voxel_dataset_generator.pipeline import DatasetGenerator


def example_single_mesh():
    """Process a single mesh file."""
    # Create configuration
    config = Config(
        base_resolution=128,
        min_resolution=4,
        output_dir=Path("output/single_mesh_test")
    )

    # Create generator
    generator = DatasetGenerator(config)

    # Process a mesh (replace with your STL file path)
    mesh_path = Path("path/to/your/mesh.stl")

    if mesh_path.exists():
        result = generator.process_mesh(
            mesh_path=mesh_path,
            object_id="0001",
            save_voxels=True
        )

        print(f"Processed mesh: {result}")

        # Finalize to generate metadata
        generator.finalize()
    else:
        print(f"Mesh file not found: {mesh_path}")


def example_batch_processing():
    """Process a batch of mesh files."""
    # Create configuration
    config = Config(
        base_resolution=128,
        min_resolution=4,
        output_dir=Path("output/batch_test")
    )

    # Create generator
    generator = DatasetGenerator(config)

    # Get list of mesh files (replace with your directory)
    mesh_dir = Path("path/to/meshes")
    mesh_files = list(mesh_dir.glob("*.stl"))

    if mesh_files:
        print(f"Found {len(mesh_files)} mesh files")

        # Process all meshes
        results = generator.process_batch(
            mesh_files=mesh_files,
            start_id=0,
            show_progress=True
        )

        # Finalize
        generator.finalize()

        # Print results
        print("\nProcessing Results:")
        for result in results:
            if "error" not in result:
                print(f"  Object {result['object_id']}: "
                      f"{result['num_subvolumes']} sub-volumes, "
                      f"{result['new_unique_subvolumes']} new unique")
    else:
        print(f"No mesh files found in: {mesh_dir}")


def example_from_thingi10k():
    """Generate dataset from Thingi10k."""
    from voxel_dataset_generator.pipeline import generate_dataset_from_thingi10k

    # Generate dataset from first 100 objects in Thingi10k
    generate_dataset_from_thingi10k(
        num_objects=100,
        output_dir=Path("output/thingi10k_dataset"),
        base_resolution=128,
        min_resolution=4,
        download_dir=Path("thingi10k_cache")
    )


if __name__ == "__main__":
    print("Voxel Dataset Generator Examples")
    print("=" * 50)

    # Uncomment the example you want to run:

    # example_single_mesh()
    # example_batch_processing()
    example_from_thingi10k()

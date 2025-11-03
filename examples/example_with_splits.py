"""
Example script demonstrating train/val split functionality.

This script shows how to:
1. Generate a voxel dataset with train/val splits enabled
2. Analyze split purity and leakage statistics
3. Load and inspect split assignments
"""

from pathlib import Path
from voxel_dataset_generator.pipeline import DatasetGenerator
from voxel_dataset_generator.utils.config import Config
from voxel_dataset_generator.splitting import HierarchicalSplitter, Split


def example_basic_split():
    """Basic example of generating a dataset with train/val splits."""
    print("=" * 80)
    print("Example 1: Basic Train/Val Split")
    print("=" * 80)

    # Configure with splitting enabled
    config = Config(
        base_resolution=128,
        min_resolution=4,
        output_dir=Path("dataset_with_splits"),
        enable_splitting=True,
        train_ratio=0.8,
        val_ratio=0.2,
        split_seed=42
    )

    # Create generator
    generator = DatasetGenerator(config)

    # Process meshes (example with dummy paths - replace with real data)
    # In practice, you would do:
    # mesh_files = list(Path("meshes").glob("*.stl"))
    # results = generator.process_batch(mesh_files)

    # For this example, let's simulate processing some objects
    print("\nProcessing objects...")
    print("(In real usage, you would process actual mesh files here)")

    # Finalize - this will automatically assign splits if not already done
    generator.finalize()

    print("\nDataset with splits generated successfully!")


def example_analyze_object_purity():
    """Example of analyzing split purity for individual objects."""
    print("\n" + "=" * 80)
    print("Example 2: Analyzing Object Split Purity")
    print("=" * 80)

    # Load split assignments from a completed dataset
    split_path = Path("dataset_with_splits/splits.json")

    if not split_path.exists():
        print(f"Split file not found at {split_path}")
        print("Run example_basic_split() first to generate a dataset with splits")
        return

    # Load the splitter
    splitter = HierarchicalSplitter.load_split_assignments(split_path)

    # Analyze a specific object
    # You would load the subdivision_map.json for the object to get all its hashes
    # For demonstration:
    object_id = "0000"
    print(f"\nAnalyzing object {object_id}...")

    # Get object's split
    obj_split = splitter.get_object_split(object_id)
    print(f"Object {object_id} is in: {obj_split.value if obj_split else 'unknown'}")

    # Load subdivision map (in real usage)
    # import json
    # with open(f"dataset_with_splits/objects/object_{object_id}/subdivision_map.json") as f:
    #     subdivision_map = json.load(f)
    #
    # subvolume_hashes = [record["hash"] for record in subdivision_map]
    #
    # # Analyze purity
    # purity_stats = splitter.analyze_object_subvolumes(object_id, subvolume_hashes)
    #
    # print(f"\nPurity Analysis:")
    # print(f"  Total sub-volumes: {purity_stats['total_subvolumes']}")
    # print(f"  Trivial (empty/full): {purity_stats['trivial']}")
    # print(f"  Non-trivial: {purity_stats['nontrivial']}")
    # print(f"  Pure (only in this split): {purity_stats['pure']}")
    # print(f"  Shared with same split: {purity_stats['shared_same_split']}")
    # print(f"  Shared with other splits (leakage): {purity_stats['shared_other_split']}")
    # print(f"  Purity percentage: {purity_stats['purity_percentage']:.2f}%")
    # print(f"  Leakage percentage: {purity_stats['leakage_percentage']:.2f}%")


def example_query_splits():
    """Example of querying split information."""
    print("\n" + "=" * 80)
    print("Example 3: Querying Split Information")
    print("=" * 80)

    split_path = Path("dataset_with_splits/splits.json")

    if not split_path.exists():
        print(f"Split file not found at {split_path}")
        print("Run example_basic_split() first to generate a dataset with splits")
        return

    # Load the splitter
    splitter = HierarchicalSplitter.load_split_assignments(split_path)

    # Get all training objects
    train_objects = splitter.get_objects_by_split(Split.TRAIN)
    val_objects = splitter.get_objects_by_split(Split.VAL)

    print(f"\nTraining objects ({len(train_objects)}):")
    print(f"  {train_objects[:10]}..." if len(train_objects) > 10 else f"  {train_objects}")

    print(f"\nValidation objects ({len(val_objects)}):")
    print(f"  {val_objects[:10]}..." if len(val_objects) > 10 else f"  {val_objects}")

    # Get non-trivial hashes for each split
    train_hashes = splitter.get_nontrivial_hashes_by_split(Split.TRAIN)
    val_hashes = splitter.get_nontrivial_hashes_by_split(Split.VAL)

    print(f"\nUnique non-trivial sub-volumes:")
    print(f"  Train: {len(train_hashes)}")
    print(f"  Val: {len(val_hashes)}")

    # Get overall statistics
    stats = splitter.get_split_statistics()
    print(f"\nOverall statistics:")
    print(f"  Objects: {stats}")


def example_export_subvolume_info():
    """Example of exporting detailed sub-volume split information."""
    print("\n" + "=" * 80)
    print("Example 4: Exporting Sub-volume Split Information")
    print("=" * 80)

    split_path = Path("dataset_with_splits/splits.json")

    if not split_path.exists():
        print(f"Split file not found at {split_path}")
        print("Run example_basic_split() first to generate a dataset with splits")
        return

    # Load the splitter
    splitter = HierarchicalSplitter.load_split_assignments(split_path)

    # Export sub-volume split info
    subvolume_info = splitter.export_subvolume_split_info()

    print(f"\nExported information for {len(subvolume_info)} unique sub-volumes")

    # Show example entries
    print("\nExample entries:")
    for i, (hash_val, info) in enumerate(list(subvolume_info.items())[:3]):
        print(f"\n  Hash: {hash_val[:16]}...")
        print(f"  Split(s): {info['splits']}")
        print(f"  Is trivial: {info['is_trivial']}")
        print(f"  Used by {len(info['used_by_objects'])} object(s)")
        if len(info['used_by_objects']) <= 3:
            print(f"  Objects: {info['used_by_objects']}")

    # Save to file
    import json
    output_path = Path("dataset_with_splits/subvolume_split_info.json")
    with open(output_path, 'w') as f:
        json.dump(subvolume_info, f, indent=2)

    print(f"\nSaved detailed sub-volume split information to {output_path}")


def example_custom_workflow():
    """Example of a custom workflow with manual split assignment."""
    print("\n" + "=" * 80)
    print("Example 5: Custom Workflow with Manual Split Assignment")
    print("=" * 80)

    config = Config(
        base_resolution=128,
        min_resolution=4,
        output_dir=Path("dataset_custom_splits"),
        enable_splitting=True,
        train_ratio=0.7,  # 70% train
        val_ratio=0.2,    # 20% val
        test_ratio=0.1,   # 10% test
        split_seed=123    # Different seed
    )

    generator = DatasetGenerator(config)

    print("\nProcessing objects...")
    # Process your meshes here
    # results = generator.process_batch(mesh_files)

    # Manually assign splits before finalize (optional)
    # This gives you control over when splits are assigned
    # object_ids = [f"{i:04d}" for i in range(100)]
    # generator.assign_splits(object_ids)

    # Finalize
    generator.finalize()

    print("\nCustom split workflow complete!")


if __name__ == "__main__":
    print("Train/Val Split Examples")
    print("=" * 80)

    # Run examples
    # Uncomment the examples you want to run:

    # example_basic_split()
    # example_analyze_object_purity()
    # example_query_splits()
    # example_export_subvolume_info()
    # example_custom_workflow()

    print("\n" + "=" * 80)
    print("To run these examples, uncomment the desired example calls in __main__")
    print("=" * 80)

"""Basic tests to verify core functionality."""

import numpy as np
import tempfile
from pathlib import Path

from voxel_dataset_generator.utils.config import Config
from voxel_dataset_generator.subdivision.subdivider import Subdivider
from voxel_dataset_generator.deduplication.registry import SubvolumeRegistry


def test_config():
    """Test configuration creation and validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            base_resolution=128,
            min_resolution=4,
            output_dir=Path(tmpdir)
        )

        assert config.base_resolution == 128
        assert config.min_resolution == 4
        assert config.num_levels == 6  # 128, 64, 32, 16, 8, 4
        assert config.level_resolutions == [128, 64, 32, 16, 8, 4]

        print(" Config test passed")


def test_subdivider():
    """Test subdivision functionality."""
    # Create a simple test voxel grid
    grid_size = 128
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    # Add some occupied voxels in one octant
    voxel_grid[0:64, 0:64, 0:64] = True

    # Create subdivider
    subdivider = Subdivider(min_resolution=4)

    # Test single subdivision
    subvolumes = subdivider.subdivide(voxel_grid, level=0)

    assert len(subvolumes) == 8, f"Expected 8 subvolumes, got {len(subvolumes)}"

    # Check that first octant is not empty
    assert not subvolumes[0].is_empty, "First octant should not be empty"

    # Check that other octants are empty
    for i in range(1, 8):
        assert subvolumes[i].is_empty, f"Octant {i} should be empty"

    # Verify subdivision is lossless
    subdivider.verify_subdivision(voxel_grid, subvolumes)

    print(" Subdivider test passed")


def test_recursive_subdivision():
    """Test recursive subdivision."""
    # Create a test grid
    grid_size = 64
    voxel_grid = np.random.rand(grid_size, grid_size, grid_size) > 0.5

    # Create subdivider
    subdivider = Subdivider(min_resolution=4)

    # Recursively subdivide
    subdivisions = subdivider.subdivide_all_levels(voxel_grid)

    # Check we have the expected levels
    # 64 -> 32 -> 16 -> 8 -> 4
    expected_levels = [1, 2, 3, 4]
    assert list(subdivisions.keys()) == expected_levels

    # Check level 1 has 8 subvolumes
    assert len(subdivisions[1]) == 8

    # Check level 2 has 64 subvolumes (8 * 8)
    assert len(subdivisions[2]) == 64

    print(" Recursive subdivision test passed")


def test_hash_computation():
    """Test hash computation and deduplication."""
    subdivider = Subdivider(min_resolution=4)

    # Create two identical grids
    grid1 = np.ones((16, 16, 16), dtype=bool)
    grid2 = np.ones((16, 16, 16), dtype=bool)

    hash1 = subdivider.compute_hash(grid1)
    hash2 = subdivider.compute_hash(grid2)

    assert hash1 == hash2, "Identical grids should have same hash"

    # Create different grid
    grid3 = np.zeros((16, 16, 16), dtype=bool)
    hash3 = subdivider.compute_hash(grid3)

    assert hash1 != hash3, "Different grids should have different hashes"

    print(" Hash computation test passed")


def test_registry():
    """Test sub-volume registry and deduplication."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = SubvolumeRegistry(base_dir=Path(tmpdir))

        # Create test data
        data1 = np.ones((16, 16, 16), dtype=bool)
        data2 = np.ones((16, 16, 16), dtype=bool)  # Same as data1
        data3 = np.zeros((16, 16, 16), dtype=bool)  # Different

        subdivider = Subdivider(min_resolution=4)
        hash1 = subdivider.compute_hash(data1)
        hash2 = subdivider.compute_hash(data2)
        hash3 = subdivider.compute_hash(data3)

        # Register first occurrence
        is_new1 = registry.register(hash1, data1, level=1, save_to_disk=False)
        assert is_new1, "First registration should be new"

        # Register duplicate
        is_new2 = registry.register(hash2, data2, level=1, save_to_disk=False)
        assert not is_new2, "Duplicate should not be new"

        # Check count
        assert registry.get_count(hash1) == 2, "Hash1 should have count 2"

        # Register different data
        is_new3 = registry.register(hash3, data3, level=1, save_to_disk=False)
        assert is_new3, "Different data should be new"

        # Check stats
        stats = registry.get_overall_stats()
        assert stats["total_subvolumes"] == 3
        assert stats["unique_subvolumes"] == 2

        print(" Registry test passed")


def test_flat_list_export():
    """Test conversion to flat list for dataframe export."""
    # Create test grid and subdivide
    voxel_grid = np.random.rand(64, 64, 64) > 0.5
    subdivider = Subdivider(min_resolution=4)
    subdivisions = subdivider.subdivide_all_levels(voxel_grid)

    # Convert to flat list
    records = subdivider.to_flat_list(subdivisions, object_id="0001")

    # Verify structure
    assert len(records) > 0
    assert all("object_id" in r for r in records)
    assert all("level" in r for r in records)
    assert all("hash" in r for r in records)
    assert all("position_x" in r for r in records)

    # Check object_id is correct
    assert all(r["object_id"] == "0001" for r in records)

    print(" Flat list export test passed")


if __name__ == "__main__":
    print("Running basic tests...\n")

    test_config()
    test_subdivider()
    test_recursive_subdivision()
    test_hash_computation()
    test_registry()
    test_flat_list_export()

    print("\n" + "=" * 50)
    print("All tests passed! ")
    print("=" * 50)

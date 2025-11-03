"""
Standalone test for the hierarchical splitter functionality.
Can run without pytest.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voxel_dataset_generator.splitting import HierarchicalSplitter, SplitConfig, Split
from voxel_dataset_generator.subdivision.subdivider import Subdivider


def test_one_bit_difference_scenario():
    """
    Test scenario with two 8x8x8 volumes where the second has one bit flipped.

    Setup:
    - Volume 1 (8x8x8): randomly generated, assigned to a split
    - Volume 2 (8x8x8): copy of Volume 1 with one bit flipped, assigned to a different split

    Expected behavior:
    - All sub-volumes shared between the two should be tracked in both splits
    - Only the one sub-volume with the bit flip should be unique to volume 2
    - The modified sub-volume should appear in volume 2's split but not volume 1's
    """
    print("=" * 80)
    print("Test: One-Bit Difference Scenario")
    print("=" * 80)

    # Setup
    config = SplitConfig(train_ratio=0.5, val_ratio=0.5, seed=42)
    splitter = HierarchicalSplitter(config)
    subdivider = Subdivider(min_resolution=4)

    # Assign splits to two objects
    object_ids = ["0000", "0001"]
    splitter.assign_splits(object_ids)

    # Get split assignment
    obj1_split = splitter.get_object_split("0000")
    obj2_split = splitter.get_object_split("0001")

    print(f"\nSplit assignments:")
    print(f"  Object 0000: {obj1_split.value}")
    print(f"  Object 0001: {obj2_split.value}")

    # They should be in different splits
    assert obj1_split != obj2_split, "Objects should be in different splits"
    assert {obj1_split, obj2_split} == {Split.TRAIN, Split.VAL}, "Should have one TRAIN and one VAL"

    # Create first volume (8x8x8) - randomly generated
    np.random.seed(123)
    volume1 = np.random.rand(8, 8, 8) > 0.5

    # Create second volume - copy of first with one bit flipped
    volume2 = volume1.copy()
    # Flip a bit in a specific location
    flip_loc = (6, 6, 6)
    original_value = volume2[flip_loc]
    volume2[flip_loc] = not volume2[flip_loc]

    print(f"\nVolume setup:")
    print(f"  Volume 1 shape: {volume1.shape}")
    print(f"  Volume 2 shape: {volume2.shape}")
    print(f"  Bit flipped at: {flip_loc}")
    print(f"  Original value: {original_value}, New value: {volume2[flip_loc]}")

    # Subdivide both volumes (8x8x8 -> 4x4x4 subdivisions)
    subdivisions1 = subdivider.subdivide_all_levels(volume1)
    subdivisions2 = subdivider.subdivide_all_levels(volume2)

    print(f"\nSubdivision results:")
    for level in subdivisions1.keys():
        print(f"  Level {level}: {len(subdivisions1[level])} sub-volumes")

    # Register all sub-volumes from volume 1
    for level, subvolumes in subdivisions1.items():
        for subvol in subvolumes:
            splitter.register_subvolume(
                object_id="0000",
                hash_val=subvol.hash,
                voxel_data=subvol.data,
                is_trivial=subvol.is_empty
            )

    # Register all sub-volumes from volume 2
    for level, subvolumes in subdivisions2.items():
        for subvol in subvolumes:
            splitter.register_subvolume(
                object_id="0001",
                hash_val=subvol.hash,
                voxel_data=subvol.data,
                is_trivial=subvol.is_empty
            )

    # Analyze the results
    # Get all non-trivial hashes from both volumes
    hashes1 = set()
    hashes2 = set()
    trivial_count1 = 0
    trivial_count2 = 0

    for level, subvolumes in subdivisions1.items():
        for subvol in subvolumes:
            if subvol.is_empty or splitter.is_trivial_subvolume(subvol.data):
                trivial_count1 += 1
            else:
                hashes1.add(subvol.hash)

    for level, subvolumes in subdivisions2.items():
        for subvol in subvolumes:
            if subvol.is_empty or splitter.is_trivial_subvolume(subvol.data):
                trivial_count2 += 1
            else:
                hashes2.add(subvol.hash)

    # Find shared and unique hashes
    shared_hashes = hashes1 & hashes2
    unique_to_vol1 = hashes1 - hashes2
    unique_to_vol2 = hashes2 - hashes1

    print(f"\nHash analysis:")
    print(f"  Volume 1 non-trivial hashes: {len(hashes1)}")
    print(f"  Volume 1 trivial hashes: {trivial_count1}")
    print(f"  Volume 2 non-trivial hashes: {len(hashes2)}")
    print(f"  Volume 2 trivial hashes: {trivial_count2}")
    print(f"  Shared hashes: {len(shared_hashes)}")
    print(f"  Unique to volume 1: {len(unique_to_vol1)}")
    print(f"  Unique to volume 2: {len(unique_to_vol2)}")

    # There should be exactly 1 hash unique to volume 2 (the modified sub-volume)
    # OR 1 unique to volume 1 and 1 unique to volume 2 (old and new versions of the modified sub-volume)
    total_unique = len(unique_to_vol1) + len(unique_to_vol2)
    assert total_unique >= 1, f"Expected at least 1 unique hash due to bit flip, got {total_unique}"

    if len(unique_to_vol2) > 0:
        # The unique hash should be in volume 2's split only
        unique_hash = list(unique_to_vol2)[0]
        splits_using_unique = splitter.get_splits_using_hash(unique_hash)
        assert len(splits_using_unique) == 1, f"Unique hash should be in one split, found in {splits_using_unique}"
        assert obj2_split in splits_using_unique, f"Unique hash should be in {obj2_split.value}"

        print(f"\n  ✓ Unique hash from volume 2 is only in {obj2_split.value} split")

        # Verify hash-object usage tracking
        objects_using_unique = splitter.get_objects_using_hash(unique_hash)
        assert len(objects_using_unique) == 1, "Unique hash should be used by one object"
        assert "0001" in objects_using_unique, "Unique hash should be used by object 0001"
        assert objects_using_unique["0001"] == obj2_split

        print(f"  ✓ Unique hash is correctly tracked to object 0001")

    # Shared hashes should appear in both splits
    if len(shared_hashes) > 0:
        sample_shared = list(shared_hashes)[0]
        splits_using_shared = splitter.get_splits_using_hash(sample_shared)
        assert len(splits_using_shared) == 2, f"Shared hash should be in both splits, found in {splits_using_shared}"
        assert obj1_split in splits_using_shared, f"Shared hash should include {obj1_split.value}"
        assert obj2_split in splits_using_shared, f"Shared hash should include {obj2_split.value}"

        print(f"  ✓ Shared hashes appear in both {obj1_split.value} and {obj2_split.value} splits")

        # Shared hashes should be used by both objects
        objects_using_shared = splitter.get_objects_using_hash(sample_shared)
        assert len(objects_using_shared) == 2, "Shared hash should be used by both objects"
        assert "0000" in objects_using_shared, "Shared hash should be used by object 0000"
        assert "0001" in objects_using_shared, "Shared hash should be used by object 0001"

        print(f"  ✓ Shared hashes are correctly tracked to both objects")

    # Get statistics
    stats = splitter.get_split_statistics()
    print(f"\nSplit statistics:")
    print(f"  Objects: {stats['objects']}")
    print(f"  Unique non-trivial hashes: {stats['unique_nontrivial_hashes']}")
    print(f"  Trivial hashes: {stats['trivial_hashes']}")

    print("\n" + "=" * 80)
    print("✓ TEST PASSED: One-bit difference scenario")
    print("=" * 80)


def test_trivial_subvolume_detection():
    """Test detection of trivial (all 0s or all 1s) sub-volumes."""
    print("\n" + "=" * 80)
    print("Test: Trivial Sub-volume Detection")
    print("=" * 80)

    # All zeros
    zeros = np.zeros((4, 4, 4), dtype=bool)
    result = HierarchicalSplitter.is_trivial_subvolume(zeros)
    assert result, "All zeros should be trivial"
    print("  ✓ All zeros correctly identified as trivial")

    # All ones
    ones = np.ones((4, 4, 4), dtype=bool)
    result = HierarchicalSplitter.is_trivial_subvolume(ones)
    assert result, "All ones should be trivial"
    print("  ✓ All ones correctly identified as trivial")

    # Mixed
    mixed = np.zeros((4, 4, 4), dtype=bool)
    mixed[0, 0, 0] = True
    result = HierarchicalSplitter.is_trivial_subvolume(mixed)
    assert not result, "Mixed should not be trivial"
    print("  ✓ Mixed values correctly identified as non-trivial")

    print("✓ TEST PASSED: Trivial sub-volume detection")
    print("=" * 80)


def test_split_config_validation():
    """Test that split config validates ratios correctly."""
    print("\n" + "=" * 80)
    print("Test: Split Config Validation")
    print("=" * 80)

    # Valid config
    config = SplitConfig(train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
    assert config.train_ratio == 0.8
    print("  ✓ Valid config accepted")

    # Invalid - doesn't sum to 1.0
    try:
        SplitConfig(train_ratio=0.8, val_ratio=0.3, test_ratio=0.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must sum to 1.0" in str(e)
        print("  ✓ Invalid sum rejected")

    # Invalid - negative ratio
    try:
        SplitConfig(train_ratio=0.8, val_ratio=-0.2, test_ratio=0.4)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be non-negative" in str(e)
        print("  ✓ Negative ratio rejected")

    print("✓ TEST PASSED: Split config validation")
    print("=" * 80)


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("RUNNING HIERARCHICAL SPLITTER TESTS")
    print("=" * 80)

    try:
        test_split_config_validation()
        test_trivial_subvolume_detection()
        test_one_bit_difference_scenario()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print()

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

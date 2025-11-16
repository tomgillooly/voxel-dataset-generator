"""PyTorch dataset for neural rendering with hierarchical voxel data.

This module provides dataset classes for loading ray-traced voxel data
and paired voxel configurations for training neural rendering models.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Literal
from collections import defaultdict
from torch.utils.data import Dataset


class HierarchicalVoxelRayDataset(Dataset):
    """Dataset for hierarchical voxel ray tracing data.

    This dataset loads pre-generated ray tracing data paired with voxel grids
    from a hierarchical voxel dataset. It supports:
    - Train/val/test splits
    - Level-specific sampling
    - Efficient caching of frequently accessed subvolumes
    - Optional data augmentation (rotations, flips)

    Data format:
        Each sample consists of:
        - origins: (N, 3) ray origin points on bounding box surface
        - directions: (N, 3) ray directions (normalized)
        - distances: (N,) ray hit distances (0 = no hit)
        - hits: (N,) binary hit flags
        - voxels: (D, H, W) boolean voxel grid
        - Additional metadata (view_ids, face_ids, etc.)

    Args:
        dataset_dir: Path to hierarchical voxel dataset
        ray_dataset_dir: Path to ray tracing dataset
        split: Data split to load ('train', 'val', 'test')
        levels: List of hierarchy levels to include (None = all levels)
        samples_per_subvolume: Number of ray samples per subvolume (None = all rays)
        cache_size: Number of voxel grids to keep in memory (0 = no caching)
        include_empty: Include empty subvolumes
        transform: Optional transform to apply to samples
        seed: Random seed for reproducibility

    Example:
        >>> dataset = HierarchicalVoxelRayDataset(
        ...     dataset_dir=Path("dataset"),
        ...     ray_dataset_dir=Path("ray_dataset_hierarchical"),
        ...     split="train",
        ...     levels=[3, 4, 5],
        ...     samples_per_subvolume=1000
        ... )
        >>> sample = dataset[0]
        >>> print(sample['origins'].shape, sample['voxels'].shape)
    """

    def __init__(
        self,
        dataset_dir: Path,
        ray_dataset_dir: Path,
        split: Literal['train', 'val', 'test'] = 'train',
        levels: Optional[List[int]] = None,
        samples_per_subvolume: Optional[int] = None,
        cache_size: int = 100,
        include_empty: bool = False,
        transform: Optional[callable] = None,
        seed: int = 42
    ):
        self.dataset_dir = Path(dataset_dir)
        self.ray_dataset_dir = Path(ray_dataset_dir)
        self.split = split
        self.samples_per_subvolume = samples_per_subvolume
        self.cache_size = cache_size
        self.include_empty = include_empty
        self.transform = transform
        self.rng = np.random.RandomState(seed)

        # Load metadata
        self._load_metadata()

        # Load splits
        self._load_splits()

        # Collect available samples
        self._collect_samples(levels)

        # Initialize cache
        self._voxel_cache = {}
        self._cache_order = []

        print(f"Loaded {len(self.samples)} samples for split '{split}'")
        if levels:
            print(f"  Levels: {sorted(set(s['level'] for s in self.samples))}")
        print(f"  Unique subvolumes: {len(set(s['hash'] for s in self.samples))}")

    def _load_metadata(self):
        """Load dataset metadata."""
        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def _load_splits(self):
        """Load train/val/test split assignments."""
        splits_path = self.dataset_dir / "splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(f"Splits file not found: {splits_path}")

        with open(splits_path, 'r') as f:
            splits_data = json.load(f)

        # Get object IDs for this split
        self.object_ids = [
            obj_id for obj_id, obj_split in splits_data['object_splits'].items()
            if obj_split == self.split
        ]

        # Build hash -> objects mapping for filtering
        self.hash_to_objects = splits_data.get('hash_object_usage', {})

        # Get trivial hashes (shared across all splits)
        self.trivial_hashes = set(splits_data.get('trivial_hashes', []))

        print(f"Split '{self.split}': {len(self.object_ids)} objects")

    def _collect_samples(self, levels: Optional[List[int]] = None):
        """Collect all available ray samples matching criteria.

        Args:
            levels: List of hierarchy levels to include (None = all)
        """
        self.samples = []
        seen_hashes = set()

        # Iterate through ray dataset to find available samples
        if not self.ray_dataset_dir.exists():
            raise FileNotFoundError(f"Ray dataset not found: {self.ray_dataset_dir}")

        # Find all ray files
        for level_dir in sorted(self.ray_dataset_dir.glob("level_*")):
            level = int(level_dir.name.split("_")[1])

            # Skip if level not requested
            if levels is not None and level not in levels:
                continue

            # Find all ray files in this level
            for ray_file in level_dir.glob("**/*_rays.npz"):
                # Extract hash from filename
                hash_val = ray_file.stem.replace("_rays", "")

                # Skip if we've already seen this hash
                if hash_val in seen_hashes:
                    continue

                # Check if this hash belongs to objects in our split
                if not self._hash_in_split(hash_val):
                    continue

                # Skip empty subvolumes if requested
                if not self.include_empty and hash_val in self.trivial_hashes:
                    continue

                # Find corresponding voxel file
                hash_prefix = hash_val[:2]
                voxel_path = (self.dataset_dir / "subvolumes" /
                             f"level_{level}" / hash_prefix / f"{hash_val}.npz")

                if not voxel_path.exists():
                    print(f"Warning: Voxel file not found for {hash_val}")
                    continue

                # Add to samples
                self.samples.append({
                    'hash': hash_val,
                    'level': level,
                    'ray_path': ray_file,
                    'voxel_path': voxel_path,
                    'is_trivial': hash_val in self.trivial_hashes
                })
                seen_hashes.add(hash_val)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{self.split}' with given criteria")

    def _hash_in_split(self, hash_val: str) -> bool:
        """Check if a hash belongs to objects in this split."""
        # Trivial hashes are shared across all splits
        if hash_val in self.trivial_hashes:
            return True

        # Check if any object using this hash is in our split
        if hash_val in self.hash_to_objects:
            hash_objects = self.hash_to_objects[hash_val]
            for obj_id in hash_objects.keys():
                if obj_id in self.object_ids:
                    return True

        return False

    def _load_voxels(self, voxel_path: Path) -> np.ndarray:
        """Load voxel grid with caching.

        Args:
            voxel_path: Path to voxel .npz file

        Returns:
            Boolean voxel grid (D, H, W)
        """
        voxel_key = str(voxel_path)

        # Check cache
        if voxel_key in self._voxel_cache:
            return self._voxel_cache[voxel_key]

        # Load from disk
        data = np.load(voxel_path)
        voxels = data['voxels']

        # Add to cache if caching enabled
        if self.cache_size > 0:
            self._voxel_cache[voxel_key] = voxels
            self._cache_order.append(voxel_key)

            # Evict oldest if cache full
            if len(self._cache_order) > self.cache_size:
                oldest = self._cache_order.pop(0)
                del self._voxel_cache[oldest]

        return voxels

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - origins: (N, 3) ray origins
                - directions: (N, 3) ray directions
                - distances: (N,) ray distances
                - hits: (N,) binary hit flags
                - voxels: (1, D, H, W) voxel grid (channel-first for conv nets)
                - level: hierarchy level
                - hash: subvolume hash
                - view_ids: (N,) view indices (optional)
                - face_ids: (N,) face indices (optional)
        """
        sample_info = self.samples[idx]

        # Load ray data
        ray_data = np.load(sample_info['ray_path'])

        # Load voxel data
        voxels = self._load_voxels(sample_info['voxel_path'])

        # Extract ray components
        origins = ray_data['origins']
        directions = ray_data['directions']
        distances = ray_data['distances']
        hits = ray_data['hits']

        # Subsample rays if requested
        if self.samples_per_subvolume is not None:
            num_rays = len(origins)
            if num_rays > self.samples_per_subvolume:
                indices = self.rng.choice(
                    num_rays,
                    self.samples_per_subvolume,
                    replace=False
                )
                origins = origins[indices]
                directions = directions[indices]
                distances = distances[indices]
                hits = hits[indices]

        # Build sample dict
        sample = {
            'origins': torch.from_numpy(origins.astype(np.float32)),
            'directions': torch.from_numpy(directions.astype(np.float32)),
            'distances': torch.from_numpy(distances.astype(np.float32)),
            'hits': torch.from_numpy(hits.astype(np.float32)),
            'voxels': torch.from_numpy(voxels.astype(np.float32)).unsqueeze(0),  # Add channel dim
            'level': sample_info['level'],
            'hash': sample_info['hash'],
        }

        # Add optional fields if present
        if 'view_ids' in ray_data:
            sample['view_ids'] = torch.from_numpy(ray_data['view_ids'])
        if 'face_ids' in ray_data:
            sample['face_ids'] = torch.from_numpy(ray_data['face_ids'])
        if 'view_positions' in ray_data:
            sample['view_positions'] = torch.from_numpy(
                ray_data['view_positions'].astype(np.float32)
            )

        # Apply transform if provided
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_level_distribution(self) -> Dict[int, int]:
        """Get distribution of samples across hierarchy levels.

        Returns:
            Dictionary mapping level -> count
        """
        distribution = defaultdict(int)
        for sample in self.samples:
            distribution[sample['level']] += 1
        return dict(distribution)


class RayBatchSampler:
    """Custom sampler that yields batches of rays from multiple subvolumes.

    This sampler is useful for training neural rendering models where you want
    to sample rays from multiple different voxel grids in each batch.

    Args:
        dataset: HierarchicalVoxelRayDataset instance
        rays_per_batch: Total number of rays per batch
        subvolumes_per_batch: Number of different subvolumes per batch
        shuffle: Whether to shuffle subvolumes
        drop_last: Whether to drop incomplete batches

    Example:
        >>> dataset = HierarchicalVoxelRayDataset(...)
        >>> sampler = RayBatchSampler(dataset, rays_per_batch=4096, subvolumes_per_batch=8)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        dataset: HierarchicalVoxelRayDataset,
        rays_per_batch: int = 4096,
        subvolumes_per_batch: int = 8,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.rays_per_batch = rays_per_batch
        self.subvolumes_per_batch = subvolumes_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.rays_per_subvolume = rays_per_batch // subvolumes_per_batch

    def __iter__(self):
        """Yield batches of subvolume indices."""
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            np.random.shuffle(indices)

        # Yield batches of subvolume indices
        for i in range(0, len(indices), self.subvolumes_per_batch):
            batch = indices[i:i + self.subvolumes_per_batch]

            if len(batch) < self.subvolumes_per_batch and self.drop_last:
                continue

            yield batch

    def __len__(self):
        """Return number of batches."""
        num_batches = len(self.dataset) // self.subvolumes_per_batch
        if not self.drop_last and len(self.dataset) % self.subvolumes_per_batch != 0:
            num_batches += 1
        return num_batches


def collate_ray_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for ray batches.

    This function handles batching of ray data where each sample may have
    a different number of rays. It concatenates all rays into single tensors
    and tracks which rays belong to which subvolume.

    Args:
        batch: List of samples from HierarchicalVoxelRayDataset

    Returns:
        Dictionary with batched tensors:
            - origins: (total_rays, 3) all ray origins
            - directions: (total_rays, 3) all ray directions
            - distances: (total_rays,) all distances
            - hits: (total_rays,) all hit flags
            - voxels: (batch_size, 1, D, H, W) stacked voxel grids
            - ray_to_voxel: (total_rays,) mapping from ray index to voxel index
            - levels: (batch_size,) hierarchy levels
            - hashes: list of hashes
    """
    # Separate ray data from voxel data
    all_origins = []
    all_directions = []
    all_distances = []
    all_hits = []
    all_voxels = []
    ray_to_voxel = []
    levels = []
    hashes = []

    for batch_idx, sample in enumerate(batch):
        num_rays = len(sample['origins'])

        all_origins.append(sample['origins'])
        all_directions.append(sample['directions'])
        all_distances.append(sample['distances'])
        all_hits.append(sample['hits'])
        all_voxels.append(sample['voxels'])

        # Track which voxel each ray belongs to
        ray_to_voxel.extend([batch_idx] * num_rays)

        levels.append(sample['level'])
        hashes.append(sample['hash'])

    # Concatenate all rays
    batched = {
        'origins': torch.cat(all_origins, dim=0),
        'directions': torch.cat(all_directions, dim=0),
        'distances': torch.cat(all_distances, dim=0),
        'hits': torch.cat(all_hits, dim=0),
        'voxels': torch.stack(all_voxels, dim=0),
        'ray_to_voxel': torch.tensor(ray_to_voxel, dtype=torch.long),
        'levels': torch.tensor(levels, dtype=torch.long),
        'hashes': hashes,
    }

    # Add optional fields if present
    if 'view_ids' in batch[0]:
        batched['view_ids'] = torch.cat([s['view_ids'] for s in batch], dim=0)
    if 'face_ids' in batch[0]:
        batched['face_ids'] = torch.cat([s['face_ids'] for s in batch], dim=0)

    return batched

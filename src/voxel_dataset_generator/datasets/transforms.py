"""Data augmentation transforms for voxel and ray data.

These transforms apply geometric transformations to both voxel grids
and ray data simultaneously to maintain consistency.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


class Compose:
    """Compose multiple transforms together.

    Args:
        transforms: List of transform callables

    Example:
        >>> transform = Compose([
        ...     RandomRotation90(),
        ...     RandomFlip()
        ... ])
    """

    def __init__(self, transforms: List[callable]):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomRotation90:
    """Randomly rotate voxels and rays by 90/180/270 degrees around Z axis.

    This transform rotates both the voxel grid and transforms the ray
    origins and directions accordingly.

    Args:
        p: Probability of applying rotation (default: 0.5)

    Example:
        >>> transform = RandomRotation90(p=0.75)
        >>> sample = transform(sample)
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() > self.p:
            return sample

        # Choose random rotation (1, 2, or 3 for 90, 180, 270 degrees)
        k = np.random.randint(1, 4)

        # Rotate voxels (around Z axis = axis 0 in ZYX format)
        # Rotate in YX plane (axes 1,2)
        voxels = sample['voxels']
        voxels = torch.rot90(voxels, k=k, dims=(2, 3))  # Rotate in XY plane
        sample['voxels'] = voxels

        # Create rotation matrix for rays (2D rotation in XY plane)
        angle = k * np.pi / 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # Rotate ray origins and directions
        sample['origins'] = sample['origins'] @ rot_matrix.T
        sample['directions'] = sample['directions'] @ rot_matrix.T

        # Rotate view positions if present
        if 'view_positions' in sample:
            sample['view_positions'] = sample['view_positions'] @ rot_matrix.T

        return sample


class RandomFlip:
    """Randomly flip voxels and rays along X, Y, or Z axes.

    Args:
        axes: Which axes to potentially flip (0=Z, 1=Y, 2=X)
        p: Probability of flipping each axis

    Example:
        >>> transform = RandomFlip(axes=[1, 2], p=0.5)  # Flip X and Y only
    """

    def __init__(self, axes: List[int] = [0, 1, 2], p: float = 0.5):
        self.axes = axes
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        voxels = sample['voxels']
        origins = sample['origins']
        directions = sample['directions']

        for axis in self.axes:
            if np.random.rand() < self.p:
                # Flip voxels (axis + 1 because of channel dimension)
                voxels = torch.flip(voxels, dims=[axis + 1])

                # Flip ray origins and directions in corresponding dimension
                # axis 0 (Z) -> dim 2, axis 1 (Y) -> dim 1, axis 2 (X) -> dim 0
                ray_dim = 2 - axis
                origins[:, ray_dim] *= -1
                directions[:, ray_dim] *= -1

                # Flip view positions if present
                if 'view_positions' in sample:
                    sample['view_positions'][:, ray_dim] *= -1

        sample['voxels'] = voxels
        sample['origins'] = origins
        sample['directions'] = directions

        return sample


class NormalizeRayOrigins:
    """Normalize ray origins to [-1, 1] range based on voxel grid size.

    This is useful for neural networks that expect normalized coordinates.

    Args:
        voxel_size: Physical size of each voxel (default: 1.0)

    Example:
        >>> transform = NormalizeRayOrigins(voxel_size=1.0)
    """

    def __init__(self, voxel_size: float = 1.0):
        self.voxel_size = voxel_size

    def __call__(self, sample: Dict) -> Dict:
        voxels = sample['voxels']
        # Get grid dimensions (C, Z, Y, X)
        grid_size = torch.tensor([voxels.shape[3], voxels.shape[2], voxels.shape[1]],
                                 dtype=torch.float32)

        # Compute extent
        extent = grid_size * self.voxel_size / 2.0

        # Normalize origins to [-1, 1]
        sample['origins'] = sample['origins'] / extent

        # Normalize view positions if present
        if 'view_positions' in sample:
            sample['view_positions'] = sample['view_positions'] / extent

        return sample


class RandomRaySubsample:
    """Randomly subsample a fixed number of rays from the sample.

    This is useful for controlling memory usage during training.

    Args:
        num_rays: Number of rays to sample

    Example:
        >>> transform = RandomRaySubsample(num_rays=1024)
    """

    def __init__(self, num_rays: int):
        self.num_rays = num_rays

    def __call__(self, sample: Dict) -> Dict:
        num_available = len(sample['origins'])

        if num_available <= self.num_rays:
            return sample

        # Random sample without replacement
        indices = torch.randperm(num_available)[:self.num_rays]

        # Subsample all ray-related tensors
        sample['origins'] = sample['origins'][indices]
        sample['directions'] = sample['directions'][indices]
        sample['distances'] = sample['distances'][indices]
        sample['hits'] = sample['hits'][indices]

        if 'view_ids' in sample:
            sample['view_ids'] = sample['view_ids'][indices]
        if 'face_ids' in sample:
            sample['face_ids'] = sample['face_ids'][indices]

        return sample


class AddNoise:
    """Add small random noise to ray origins and directions.

    This can help with regularization during training.

    Args:
        origin_std: Standard deviation of noise for origins
        direction_std: Standard deviation of noise for directions (directions renormalized after)

    Example:
        >>> transform = AddNoise(origin_std=0.01, direction_std=0.001)
    """

    def __init__(self, origin_std: float = 0.01, direction_std: float = 0.001):
        self.origin_std = origin_std
        self.direction_std = direction_std

    def __call__(self, sample: Dict) -> Dict:
        if self.origin_std > 0:
            noise = torch.randn_like(sample['origins']) * self.origin_std
            sample['origins'] = sample['origins'] + noise

        if self.direction_std > 0:
            noise = torch.randn_like(sample['directions']) * self.direction_std
            sample['directions'] = sample['directions'] + noise
            # Renormalize directions
            sample['directions'] = torch.nn.functional.normalize(
                sample['directions'], dim=-1
            )

        return sample


class ToDevice:
    """Move all tensors to specified device.

    Args:
        device: Target device ('cuda', 'cpu', etc.)

    Example:
        >>> transform = ToDevice('cuda')
    """

    def __init__(self, device: str):
        self.device = torch.device(device)

    def __call__(self, sample: Dict) -> Dict:
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value.to(self.device)
        return sample


class VoxelOccupancyJitter:
    """Randomly flip voxel occupancy with small probability.

    This adds noise to the voxel grid which can help with robustness.

    Args:
        flip_prob: Probability of flipping each voxel

    Example:
        >>> transform = VoxelOccupancyJitter(flip_prob=0.01)
    """

    def __init__(self, flip_prob: float = 0.01):
        self.flip_prob = flip_prob

    def __call__(self, sample: Dict) -> Dict:
        if self.flip_prob <= 0:
            return sample

        voxels = sample['voxels']
        mask = torch.rand_like(voxels) < self.flip_prob
        voxels = torch.logical_xor(voxels.bool(), mask).float()
        sample['voxels'] = voxels

        return sample

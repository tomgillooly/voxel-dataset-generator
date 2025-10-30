"""Hierarchical voxel dataset generator for neural rendering research."""

from .voxelization.voxelizer import Voxelizer
from .subdivision.subdivider import Subdivider
from .deduplication.registry import SubvolumeRegistry
from .utils.config import Config

__version__ = "0.1.0"
__all__ = ["Voxelizer", "Subdivider", "SubvolumeRegistry", "Config"]

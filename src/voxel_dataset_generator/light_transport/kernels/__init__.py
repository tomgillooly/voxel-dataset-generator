"""Metal kernel source strings for voxel ray tracing.

This module provides Metal shader source code as Python strings for use with
MLX's mx.fast.metal_kernel() JIT compilation.

The kernels are organized as:
- common: Shared data structures and utility functions
- dda_traversal: DDA voxel grid traversal algorithm
- monte_carlo: Monte Carlo subsurface scattering simulation
- spherical_harmonics: SH basis evaluation and projection
"""

from .sources import (
    COMMON_HEADER,
    DDA_KERNEL_SOURCE,
    DDA_HEADER,
    MC_KERNEL_SOURCE,
    MC_HEADER,
    SH_KERNEL_SOURCE,
    SH_HEADER,
)

__all__ = [
    "COMMON_HEADER",
    "DDA_KERNEL_SOURCE",
    "DDA_HEADER",
    "MC_KERNEL_SOURCE",
    "MC_HEADER",
    "SH_KERNEL_SOURCE",
    "SH_HEADER",
]

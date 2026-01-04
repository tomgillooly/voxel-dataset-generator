"""Light transport simulation for voxel structures.

This module provides tools for generating subsurface scattering ground truth
using Monte Carlo ray tracing on Apple Silicon (Metal) or CPU fallback.

Main Components:
    MetalVoxelTracer: GPU-accelerated ray tracer using MLX/Metal
    OpticalProperties: Material optical parameters (scattering, absorption)
    TransferMatrixBuilder: Computes full transfer matrices
    BoundaryPatch: Discretization of voxel grid boundary

Example:
    >>> import numpy as np
    >>> from voxel_dataset_generator.light_transport import (
    ...     MetalVoxelTracer, OpticalProperties
    ... )
    >>>
    >>> # Load voxel data
    >>> voxels = np.load("morph_0000_to_0001_step_000.npz")['voxels']
    >>>
    >>> # Create tracer with optical properties
    >>> optical_props = OpticalProperties(sigma_s=1.0, sigma_a=0.1, g=0.8)
    >>> tracer = MetalVoxelTracer(voxels, optical_props=optical_props)
    >>>
    >>> # Compute transfer matrix
    >>> transfer_matrix = tracer.compute_transfer_matrix(
    ...     patches_per_face=8,
    ...     sh_order=2,
    ...     samples_per_condition=10000
    ... )
    >>> print(f"Transfer matrix shape: {transfer_matrix.shape}")
    Transfer matrix shape: (3456, 3456)
"""

from .optical_properties import OpticalProperties
from .metal_tracer import MetalVoxelTracer, MLX_AVAILABLE
from .boundary_patches import (
    BoundaryPatch,
    generate_boundary_patches,
    patches_to_arrays,
    get_patch_info,
    find_exit_patch,
    FACE_NORMALS,
)
from .spherical_harmonics import (
    eval_sh_basis,
    project_to_sh,
    reconstruct_from_sh,
    sample_cosine_hemisphere,
    sample_uniform_sphere,
    get_n_sh_coeffs,
    get_sh_order,
)
from .transfer_matrix import (
    TransferMatrixBuilder,
    save_transfer_matrix,
    load_transfer_matrix,
    analyze_transfer_matrix,
    visualize_transfer_matrix,
)

__all__ = [
    # Core classes
    "MetalVoxelTracer",
    "OpticalProperties",
    "TransferMatrixBuilder",
    "BoundaryPatch",

    # Boundary patches
    "generate_boundary_patches",
    "patches_to_arrays",
    "get_patch_info",
    "find_exit_patch",
    "FACE_NORMALS",

    # Spherical harmonics
    "eval_sh_basis",
    "project_to_sh",
    "reconstruct_from_sh",
    "sample_cosine_hemisphere",
    "sample_uniform_sphere",
    "get_n_sh_coeffs",
    "get_sh_order",

    # Transfer matrix I/O
    "save_transfer_matrix",
    "load_transfer_matrix",
    "analyze_transfer_matrix",
    "visualize_transfer_matrix",

    # Flags
    "MLX_AVAILABLE",
]

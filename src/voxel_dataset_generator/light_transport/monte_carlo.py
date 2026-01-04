"""Monte Carlo subsurface scattering simulation (CPU fallback).

This module provides a pure Python/NumPy implementation of the Monte Carlo
scattering simulation. It serves as a reference implementation and fallback
when Metal kernels aren't available or for validation.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from numba import njit, prange
import warnings

from .optical_properties import OpticalProperties
from .spherical_harmonics import eval_sh_basis, get_n_sh_coeffs

# Try to use Numba for acceleration
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Provide fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


# SH constants (matching spherical_harmonics.py and Metal kernels)
# Order 0-2
SH_C0 = 0.282094791773878      # 1 / (2 * sqrt(pi))
SH_C1 = 0.488602511902920      # sqrt(3 / (4 * pi))
SH_C2_0 = 1.092548430592079    # sqrt(15 / (4 * pi))
SH_C2_1 = 0.315391565252520    # sqrt(5 / (16 * pi))
SH_C2_2 = 0.546274215296040    # sqrt(15 / (16 * pi))

# Order 3
SH_C3_0 = 0.590043589926644    # sqrt(7 / (16 * pi))
SH_C3_1 = 2.890611442640554    # sqrt(21 / (4 * pi)) * sqrt(2)
SH_C3_2 = 0.457045799464466    # sqrt(21 / (16 * pi)) / sqrt(2)
SH_C3_3 = 0.373176332590115    # sqrt(7 / (16 * pi)) / sqrt(2)

# Order 4
SH_C4_0 = 0.105785546915204    # 3 / (16 * sqrt(pi))
SH_C4_1 = 0.473087347878780    # 3 / (8 * sqrt(pi))
SH_C4_2 = 1.445305721320277    # 15 / (8 * sqrt(pi)) * sqrt(2)
SH_C4_3 = 0.590043589926644    # sqrt(7 / (16 * pi)) -- same as C3_0
SH_C4_4 = 0.625835735449176    # 3 * sqrt(35) / (16 * sqrt(pi))

# Maximum supported SH order
MAX_SH_ORDER = 4
MAX_SH_COEFFS = 25  # (4+1)^2


@njit(cache=True)
def _eval_sh_basis_numba(x: float, y: float, z: float) -> np.ndarray:
    """Evaluate up to order-4 SH basis for a single direction.

    Returns 25 coefficients for full order-4 support.
    """
    coeffs = np.zeros(MAX_SH_COEFFS, dtype=np.float64)

    # Precompute powers
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z

    # l=0 (1 coefficient)
    coeffs[0] = SH_C0

    # l=1 (3 coefficients)
    coeffs[1] = SH_C1 * y
    coeffs[2] = SH_C1 * z
    coeffs[3] = SH_C1 * x

    # l=2 (5 coefficients)
    coeffs[4] = SH_C2_0 * xy           # Y_2^-2
    coeffs[5] = SH_C2_0 * yz           # Y_2^-1
    coeffs[6] = SH_C2_1 * (3.0 * z2 - 1.0)  # Y_2^0
    coeffs[7] = SH_C2_0 * xz           # Y_2^1
    coeffs[8] = SH_C2_2 * (x2 - y2)    # Y_2^2

    # l=3 (7 coefficients)
    coeffs[9] = SH_C3_1 * y * (3.0 * x2 - y2) / 4.0           # Y_3^-3
    coeffs[10] = SH_C3_0 * xy * z * 2.0                        # Y_3^-2
    coeffs[11] = SH_C3_2 * y * (5.0 * z2 - 1.0)               # Y_3^-1
    coeffs[12] = SH_C3_3 * z * (5.0 * z2 - 3.0)               # Y_3^0
    coeffs[13] = SH_C3_2 * x * (5.0 * z2 - 1.0)               # Y_3^1
    coeffs[14] = SH_C3_0 * (x2 - y2) * z                       # Y_3^2
    coeffs[15] = SH_C3_1 * x * (x2 - 3.0 * y2) / 4.0          # Y_3^3

    # l=4 (9 coefficients)
    coeffs[16] = SH_C4_4 * xy * (x2 - y2)                      # Y_4^-4
    coeffs[17] = SH_C4_3 * yz * (3.0 * x2 - y2)               # Y_4^-3
    coeffs[18] = SH_C4_2 * xy * (7.0 * z2 - 1.0) / 2.0        # Y_4^-2
    coeffs[19] = SH_C4_1 * yz * (7.0 * z2 - 3.0)              # Y_4^-1
    coeffs[20] = SH_C4_0 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0) # Y_4^0
    coeffs[21] = SH_C4_1 * xz * (7.0 * z2 - 3.0)              # Y_4^1
    coeffs[22] = SH_C4_2 * (x2 - y2) * (7.0 * z2 - 1.0) / 2.0 # Y_4^2
    coeffs[23] = SH_C4_3 * xz * (x2 - 3.0 * y2)               # Y_4^3
    coeffs[24] = SH_C4_4 * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2)) / 4.0  # Y_4^4

    return coeffs


@njit(cache=True)
def _sample_henyey_greenstein(g: float, u1: float, u2: float) -> Tuple[float, float, float]:
    """Sample direction from Henyey-Greenstein phase function."""
    if abs(g) < 1e-4:
        cos_theta = 1.0 - 2.0 * u1
    else:
        sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1)
        cos_theta = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g)

    cos_theta = max(-1.0, min(1.0, cos_theta))
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * np.pi * u2

    return sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta


@njit(cache=True)
def _sample_cosine_hemisphere(u1: float, u2: float) -> Tuple[float, float, float]:
    """Sample direction from cosine-weighted hemisphere."""
    phi = 2.0 * np.pi * u1
    cos_theta = np.sqrt(u2)
    sin_theta = np.sqrt(1.0 - u2)

    return sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta


@njit(cache=True)
def _build_orthonormal_basis(nx: float, ny: float, nz: float):
    """Build orthonormal basis from a normal vector."""
    # Choose up vector that's not parallel to normal
    if abs(ny) < 0.999:
        ux, uy, uz = 0.0, 1.0, 0.0
    else:
        ux, uy, uz = 1.0, 0.0, 0.0

    # tangent = up x normal
    tx = uy * nz - uz * ny
    ty = uz * nx - ux * nz
    tz = ux * ny - uy * nx

    # Normalize tangent
    t_len = np.sqrt(tx*tx + ty*ty + tz*tz)
    if t_len > 1e-8:
        tx, ty, tz = tx / t_len, ty / t_len, tz / t_len

    # bitangent = normal x tangent
    bx = ny * tz - nz * ty
    by = nz * tx - nx * tz
    bz = nx * ty - ny * tx

    return tx, ty, tz, bx, by, bz


@njit(cache=True)
def _local_to_world(lx: float, ly: float, lz: float,
                    tx: float, ty: float, tz: float,
                    bx: float, by: float, bz: float,
                    nx: float, ny: float, nz: float) -> Tuple[float, float, float]:
    """Transform local direction to world space."""
    wx = lx * tx + ly * bx + lz * nx
    wy = lx * ty + ly * by + lz * ny
    wz = lx * tz + ly * bz + lz * nz
    return wx, wy, wz


@njit(cache=True)
def _trace_dda(
    ox: float, oy: float, oz: float,
    dx: float, dy: float, dz: float,
    voxels: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    resolution: np.ndarray,
    voxel_size: float
) -> Tuple[float, float, float, float, int, bool]:
    """DDA voxel traversal, returns (distance, exit_x, exit_y, exit_z, exit_face, hit)."""
    # Normalize direction
    d_len = np.sqrt(dx*dx + dy*dy + dz*dz)
    if d_len < 1e-8:
        return 0.0, ox, oy, oz, -1, False
    dx, dy, dz = dx / d_len, dy / d_len, dz / d_len

    # Inverse direction
    inv_dx = 1.0 / dx if abs(dx) > 1e-8 else 1e20
    inv_dy = 1.0 / dy if abs(dy) > 1e-8 else 1e20
    inv_dz = 1.0 / dz if abs(dz) > 1e-8 else 1e20

    # Ray-box intersection
    t0x = (grid_min[0] - ox) * inv_dx
    t0y = (grid_min[1] - oy) * inv_dy
    t0z = (grid_min[2] - oz) * inv_dz

    t1x = (grid_max[0] - ox) * inv_dx
    t1y = (grid_max[1] - oy) * inv_dy
    t1z = (grid_max[2] - oz) * inv_dz

    tmin_x = min(t0x, t1x)
    tmin_y = min(t0y, t1y)
    tmin_z = min(t0z, t1z)

    tmax_x = max(t0x, t1x)
    tmax_y = max(t0y, t1y)
    tmax_z = max(t0z, t1z)

    t_enter = max(tmin_x, max(tmin_y, tmin_z))
    t_exit = min(tmax_x, min(tmax_y, tmax_z))

    if t_enter > t_exit or t_exit < 0:
        return 0.0, ox, oy, oz, -1, False

    t_enter = max(t_enter, 0.0)

    # Determine exit face
    exit_face = 0
    if tmax_x <= tmax_y and tmax_x <= tmax_z:
        exit_face = 0 if dx > 0 else 1
    elif tmax_y <= tmax_z:
        exit_face = 2 if dy > 0 else 3
    else:
        exit_face = 4 if dz > 0 else 5

    # Exit point
    exit_x = ox + dx * t_exit
    exit_y = oy + dy * t_exit
    exit_z = oz + dz * t_exit

    # Entry point
    entry_x = ox + dx * (t_enter + 1e-5)
    entry_y = oy + dy * (t_enter + 1e-5)
    entry_z = oz + dz * (t_enter + 1e-5)

    # Voxel coordinates
    vx = int((entry_x - grid_min[0]) / voxel_size)
    vy = int((entry_y - grid_min[1]) / voxel_size)
    vz = int((entry_z - grid_min[2]) / voxel_size)

    vx = max(0, min(vx, resolution[0] - 1))
    vy = max(0, min(vy, resolution[1] - 1))
    vz = max(0, min(vz, resolution[2] - 1))

    # Step direction
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    step_z = 1 if dz > 0 else (-1 if dz < 0 else 0)

    # tDelta
    tDelta_x = abs(voxel_size * inv_dx) if step_x != 0 else 1e20
    tDelta_y = abs(voxel_size * inv_dy) if step_y != 0 else 1e20
    tDelta_z = abs(voxel_size * inv_dz) if step_z != 0 else 1e20

    # tMax
    voxel_corner_x = grid_min[0] + vx * voxel_size
    voxel_corner_y = grid_min[1] + vy * voxel_size
    voxel_corner_z = grid_min[2] + vz * voxel_size

    if step_x != 0:
        if step_x > 0:
            tMax_x = (voxel_corner_x + voxel_size - ox) * inv_dx
        else:
            tMax_x = (voxel_corner_x - ox) * inv_dx
    else:
        tMax_x = 1e20

    if step_y != 0:
        if step_y > 0:
            tMax_y = (voxel_corner_y + voxel_size - oy) * inv_dy
        else:
            tMax_y = (voxel_corner_y - oy) * inv_dy
    else:
        tMax_y = 1e20

    if step_z != 0:
        if step_z > 0:
            tMax_z = (voxel_corner_z + voxel_size - oz) * inv_dz
        else:
            tMax_z = (voxel_corner_z - oz) * inv_dz
    else:
        tMax_z = 1e20

    # DDA loop
    accumulated = 0.0
    current_t = t_enter
    max_steps = resolution[0] + resolution[1] + resolution[2]

    for _ in range(max_steps):
        if vx < 0 or vx >= resolution[0] or \
           vy < 0 or vy >= resolution[1] or \
           vz < 0 or vz >= resolution[2]:
            break

        next_t = min(tMax_x, min(tMax_y, tMax_z))
        next_t = min(next_t, t_exit)

        # Check occupancy (Z-Y-X layout: voxels[z, y, x])
        idx = vz * (resolution[1] * resolution[0]) + vy * resolution[0] + vx
        if voxels.flat[idx]:
            accumulated += next_t - current_t

        # Step
        if tMax_x < tMax_y and tMax_x < tMax_z:
            tMax_x += tDelta_x
            vx += step_x
        elif tMax_y < tMax_z:
            tMax_y += tDelta_y
            vy += step_y
        else:
            tMax_z += tDelta_z
            vz += step_z

        current_t = next_t
        if next_t >= t_exit:
            break

    return accumulated, exit_x, exit_y, exit_z, exit_face, True


@njit(cache=True)
def _compute_exit_patch(
    px: float, py: float, pz: float,
    exit_face: int,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    patches_per_face: int
) -> int:
    """Compute which patch an exit point belongs to."""
    grid_size = grid_max - grid_min
    nx = (px - grid_min[0]) / grid_size[0]
    ny = (py - grid_min[1]) / grid_size[1]
    nz = (pz - grid_min[2]) / grid_size[2]

    nx = max(0.0, min(0.999999, nx))
    ny = max(0.0, min(0.999999, ny))
    nz = max(0.0, min(0.999999, nz))

    if exit_face == 0 or exit_face == 1:  # +X or -X
        patch_u = int(ny * patches_per_face)
        patch_v = int(nz * patches_per_face)
    elif exit_face == 2 or exit_face == 3:  # +Y or -Y
        patch_u = int(nx * patches_per_face)
        patch_v = int(nz * patches_per_face)
    else:  # +Z or -Z
        patch_u = int(nx * patches_per_face)
        patch_v = int(ny * patches_per_face)

    patch_u = min(patch_u, patches_per_face - 1)
    patch_v = min(patch_v, patches_per_face - 1)

    patches_per_face_sq = patches_per_face * patches_per_face
    return exit_face * patches_per_face_sq + patch_v * patches_per_face + patch_u


@njit(cache=True)
def _get_patch_geometry(
    patch_idx: int,
    patches_per_face: int,
    grid_min: np.ndarray,
    grid_max: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    """Get patch center and normal."""
    patches_per_face_sq = patches_per_face * patches_per_face
    face = patch_idx // patches_per_face_sq
    local_idx = patch_idx % patches_per_face_sq
    patch_v = local_idx // patches_per_face
    patch_u = local_idx % patches_per_face

    grid_size = grid_max - grid_min
    grid_center = (grid_min + grid_max) * 0.5

    cx, cy, cz = grid_center[0], grid_center[1], grid_center[2]
    nx, ny, nz = 0.0, 0.0, 0.0

    if face == 0:  # +X
        patch_size_u = grid_size[1] / patches_per_face
        patch_size_v = grid_size[2] / patches_per_face
        cx = grid_max[0]
        cy = grid_min[1] + (patch_u + 0.5) * patch_size_u
        cz = grid_min[2] + (patch_v + 0.5) * patch_size_v
        nx = 1.0
    elif face == 1:  # -X
        patch_size_u = grid_size[1] / patches_per_face
        patch_size_v = grid_size[2] / patches_per_face
        cx = grid_min[0]
        cy = grid_min[1] + (patch_u + 0.5) * patch_size_u
        cz = grid_min[2] + (patch_v + 0.5) * patch_size_v
        nx = -1.0
    elif face == 2:  # +Y
        patch_size_u = grid_size[0] / patches_per_face
        patch_size_v = grid_size[2] / patches_per_face
        cy = grid_max[1]
        cx = grid_min[0] + (patch_u + 0.5) * patch_size_u
        cz = grid_min[2] + (patch_v + 0.5) * patch_size_v
        ny = 1.0
    elif face == 3:  # -Y
        patch_size_u = grid_size[0] / patches_per_face
        patch_size_v = grid_size[2] / patches_per_face
        cy = grid_min[1]
        cx = grid_min[0] + (patch_u + 0.5) * patch_size_u
        cz = grid_min[2] + (patch_v + 0.5) * patch_size_v
        ny = -1.0
    elif face == 4:  # +Z
        patch_size_u = grid_size[0] / patches_per_face
        patch_size_v = grid_size[1] / patches_per_face
        cz = grid_max[2]
        cx = grid_min[0] + (patch_u + 0.5) * patch_size_u
        cy = grid_min[1] + (patch_v + 0.5) * patch_size_v
        nz = 1.0
    else:  # -Z
        patch_size_u = grid_size[0] / patches_per_face
        patch_size_v = grid_size[1] / patches_per_face
        cz = grid_min[2]
        cx = grid_min[0] + (patch_u + 0.5) * patch_size_u
        cy = grid_min[1] + (patch_v + 0.5) * patch_size_v
        nz = -1.0

    return cx, cy, cz, nx, ny, nz


@njit(parallel=True, cache=True)
def _simulate_mc_samples(
    voxels: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    resolution: np.ndarray,
    voxel_size: float,
    sigma_s: float,
    sigma_a: float,
    g: float,
    incident_patch: int,
    incident_sh_idx: int,
    patches_per_face: int,
    n_sh: int,
    n_patches: int,
    n_samples: int,
    seed: int,
    track_sh_error: bool = True
) -> np.ndarray:
    """Run Monte Carlo simulation for a single incident condition.

    Returns:
        If track_sh_error is False: output_sh array of shape (n_patches, n_sh)
        If track_sh_error is True: stacked array of shape (n_patches, n_sh + 4)
            where the last 4 columns are:
            - weight_sum: sum of actual weights per patch
            - weight_sq_sum: sum of squared weights per patch
            - recon_error_sum: sum of squared reconstruction errors
            - sample_count: number of samples per patch
    """
    # Output: SH coefficients + optional error tracking columns
    n_cols = n_sh + 4 if track_sh_error else n_sh
    output = np.zeros((n_patches, n_cols), dtype=np.float64)

    sigma_t = sigma_s + sigma_a
    albedo = sigma_s / sigma_t
    mfp = 1.0 / sigma_t

    # Get incident patch geometry
    cx, cy, cz, nx, ny, nz = _get_patch_geometry(
        incident_patch, patches_per_face, grid_min, grid_max
    )

    # Inward normal
    inx, iny, inz = -nx, -ny, -nz

    for sample_idx in prange(n_samples):
        # Initialize RNG
        rng_state = np.uint64((sample_idx ^ (seed * 0x9E3779B9)) * 0x85EBCA6B)
        rng_state = (rng_state * 0xC2B2AE35) ^ (rng_state >> 16)

        def rand():
            nonlocal rng_state
            rng_state = rng_state * 6364136223846793005 + 1442695040888963407
            return float((rng_state >> 17) & 0x7FFFFFFF) / float(0x7FFFFFFF)

        # Sample incident direction (cosine-weighted hemisphere)
        u1, u2 = rand(), rand()
        local_x, local_y, local_z = _sample_cosine_hemisphere(u1, u2)

        # Build basis for inward normal
        tx, ty, tz, bx, by, bz = _build_orthonormal_basis(inx, iny, inz)

        # Transform to world space
        dx, dy, dz = _local_to_world(local_x, local_y, local_z,
                                      tx, ty, tz, bx, by, bz, inx, iny, inz)

        # SH weight for incident direction
        sh_coeffs = _eval_sh_basis_numba(dx, dy, dz)
        sh_weight = sh_coeffs[incident_sh_idx]

        # Start ray slightly inside volume
        ox = cx + dx * 1e-4
        oy = cy + dy * 1e-4
        oz = cz + dz * 1e-4

        weight = sh_weight
        MAX_BOUNCES = 256
        RR_THRESHOLD = 0.01

        for bounce in range(MAX_BOUNCES):
            # Sample free-flight distance
            u = rand()
            u = max(u, 1e-10)
            t_scatter = -np.log(u) * mfp

            # Trace through voxel grid
            distance, exit_x, exit_y, exit_z, exit_face, hit = _trace_dda(
                ox, oy, oz, dx, dy, dz,
                voxels, grid_min, grid_max, resolution, voxel_size
            )

            if not hit:
                break

            if t_scatter < distance:
                # Scattering event
                ox = ox + dx * t_scatter
                oy = oy + dy * t_scatter
                oz = oz + dz * t_scatter

                weight *= albedo

                # Sample new direction
                u1, u2 = rand(), rand()
                local_x, local_y, local_z = _sample_henyey_greenstein(g, u1, u2)

                tx, ty, tz, bx, by, bz = _build_orthonormal_basis(dx, dy, dz)
                dx, dy, dz = _local_to_world(local_x, local_y, local_z,
                                              tx, ty, tz, bx, by, bz, dx, dy, dz)

            else:
                # Exit volume
                exit_patch = _compute_exit_patch(
                    exit_x, exit_y, exit_z, exit_face,
                    grid_min, grid_max, patches_per_face
                )

                # Get exit patch normal
                _, _, _, enx, eny, enz = _get_patch_geometry(
                    exit_patch, patches_per_face, grid_min, grid_max
                )

                cos_theta = abs(dx * enx + dy * eny + dz * enz)

                # Project to SH
                exit_sh = _eval_sh_basis_numba(dx, dy, dz)
                actual_weight = weight * cos_theta

                for i in range(n_sh):
                    output[exit_patch, i] += actual_weight * exit_sh[i]

                # Track statistics for SH reconstruction error analysis
                if track_sh_error:
                    # Accumulate statistics for post-hoc error analysis:
                    # - weight_sum: for mean weight computation
                    # - weight_sq_sum: ||f||^2 = sum(w_i^2), the "original energy"
                    # - sh_weight_product: sum(w_i * sum_j(Y_j(d_i)^2))
                    #   This is needed because reconstruction at d is sum(coeff_j * Y_j(d))
                    #   = sum_j(sum_i(w_i * Y_j(d_i)) * Y_j(d)) / n
                    #   For self-reconstruction: sum_i(w_i * sum_j(Y_j(d_i)^2)) / n
                    # - sample_count: number of samples
                    #
                    # Post-hoc, we can compute:
                    # - SH energy: ||f_SH||^2 = sum_j(coeff_j^2) by Parseval
                    # - Relative error: 1 - SH_energy / original_energy

                    # Compute sum_j(Y_j(d)^2) for this direction
                    sh_basis_sq_sum = 0.0
                    for j in range(n_sh):
                        sh_basis_sq_sum += exit_sh[j] * exit_sh[j]

                    output[exit_patch, n_sh] += actual_weight           # weight_sum
                    output[exit_patch, n_sh + 1] += actual_weight ** 2  # weight_sq_sum
                    output[exit_patch, n_sh + 2] += actual_weight * sh_basis_sq_sum  # for reconstruction
                    output[exit_patch, n_sh + 3] += 1.0                 # sample_count

                break

            # Russian roulette
            if weight < RR_THRESHOLD and bounce > 3:
                if rand() > weight / RR_THRESHOLD:
                    break
                weight = RR_THRESHOLD

    return output


def simulate_incident_condition_cpu(
    voxels: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    resolution: np.ndarray,
    voxel_size: float,
    optical_props: OpticalProperties,
    incident_patch: int,
    incident_sh: int,
    patches: Dict[str, np.ndarray],
    n_patches: int,
    n_sh: int,
    n_samples: int,
    seed: int,
    track_sh_error: bool = False
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """Simulate one incident condition using CPU (with optional Numba JIT).

    Args:
        voxels: Binary voxel grid
        grid_min: Grid minimum corner
        grid_max: Grid maximum corner
        resolution: Grid resolution
        voxel_size: Voxel size
        optical_props: Material optical properties
        incident_patch: Incident patch index
        incident_sh: Incident SH basis index
        patches: Patch geometry arrays
        n_patches: Total number of patches
        n_sh: Number of SH coefficients
        n_samples: Number of MC samples
        seed: Random seed
        track_sh_error: If True, track statistics for SH reconstruction error

    Returns:
        Tuple of:
        - Exitant SH coefficients, shape (n_patches, n_sh)
        - Error stats dict (if track_sh_error=True), else None. Contains:
            - 'weight_sum': sum of weights per exit patch, shape (n_patches,)
            - 'weight_sq_sum': sum of squared weights per exit patch
            - 'sample_count': number of samples per exit patch
    """
    patches_per_face = int(np.sqrt(n_patches / 6))

    # Ensure arrays are contiguous
    voxels = np.ascontiguousarray(voxels)
    grid_min = np.ascontiguousarray(grid_min, dtype=np.float64)
    grid_max = np.ascontiguousarray(grid_max, dtype=np.float64)
    resolution = np.ascontiguousarray(resolution, dtype=np.int64)

    if NUMBA_AVAILABLE:
        output = _simulate_mc_samples(
            voxels, grid_min, grid_max, resolution, float(voxel_size),
            float(optical_props.sigma_s), float(optical_props.sigma_a),
            float(optical_props.g),
            int(incident_patch), int(incident_sh),
            int(patches_per_face), int(n_sh), int(n_patches),
            int(n_samples), int(seed),
            track_sh_error
        )
    else:
        warnings.warn(
            "Numba not available, using slow Python fallback. "
            "Install numba for 100x speedup: pip install numba"
        )
        output = _simulate_mc_samples_python(
            voxels, grid_min, grid_max, resolution, voxel_size,
            optical_props.sigma_s, optical_props.sigma_a, optical_props.g,
            incident_patch, incident_sh, patches_per_face, n_sh, n_patches,
            n_samples, seed
        )
        # Python fallback doesn't support error tracking yet
        if track_sh_error:
            # Pad with zeros for compatibility
            output = np.concatenate([output, np.zeros((n_patches, 4))], axis=1)

    # Extract SH coefficients and error stats
    output_sh = output[:, :n_sh]

    error_stats = None
    if track_sh_error:
        error_stats = {
            'weight_sum': output[:, n_sh],
            'weight_sq_sum': output[:, n_sh + 1],
            # Column n_sh+2 stores sum(w * ||Y||^2), not currently used
            'sample_count': output[:, n_sh + 3],
        }

    # Normalize SH by sample count
    output_sh_normalized = (output_sh / n_samples).astype(np.float32)

    return output_sh_normalized, error_stats


def _simulate_mc_samples_python(
    voxels, grid_min, grid_max, resolution, voxel_size,
    sigma_s, sigma_a, g, incident_patch, incident_sh_idx,
    patches_per_face, n_sh, n_patches, n_samples, seed
):
    """Pure Python fallback (very slow)."""
    rng = np.random.RandomState(seed)
    output_sh = np.zeros((n_patches, n_sh), dtype=np.float64)

    sigma_t = sigma_s + sigma_a
    albedo = sigma_s / sigma_t
    mfp = 1.0 / sigma_t

    # Get incident patch geometry
    patches_per_face_sq = patches_per_face * patches_per_face
    face = incident_patch // patches_per_face_sq
    local_idx = incident_patch % patches_per_face_sq
    patch_v = local_idx // patches_per_face
    patch_u = local_idx % patches_per_face

    grid_size = grid_max - grid_min

    # Compute patch center and normal
    cx, cy, cz, nx, ny, nz = _get_patch_geometry(
        incident_patch, patches_per_face, grid_min, grid_max
    )

    inward_normal = np.array([-nx, -ny, -nz])

    for _ in range(n_samples):
        # Sample direction
        u = rng.random(2)
        local_dir = np.array(_sample_cosine_hemisphere(u[0], u[1]))

        # Build basis
        up = np.array([0, 1, 0]) if abs(inward_normal[1]) < 0.999 else np.array([1, 0, 0])
        tangent = np.cross(up, inward_normal)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(inward_normal, tangent)

        direction = local_dir[0] * tangent + local_dir[1] * bitangent + local_dir[2] * inward_normal

        # SH weight
        sh_coeffs = eval_sh_basis(direction.reshape(1, 3))[0]
        sh_weight = sh_coeffs[incident_sh_idx]

        origin = np.array([cx, cy, cz]) + direction * 1e-4
        weight = sh_weight

        for bounce in range(256):
            t_scatter = -np.log(max(rng.random(), 1e-10)) * mfp

            # Simple trace (just check if we hit boundary)
            distance, exit_x, exit_y, exit_z, exit_face, hit = _trace_dda(
                origin[0], origin[1], origin[2],
                direction[0], direction[1], direction[2],
                voxels, grid_min, grid_max, resolution, voxel_size
            )

            if not hit:
                break

            if t_scatter < distance:
                origin = origin + direction * t_scatter
                weight *= albedo

                # Sample HG
                u = rng.random(2)
                local_scatter = np.array(_sample_henyey_greenstein(g, u[0], u[1]))

                up = np.array([0, 1, 0]) if abs(direction[1]) < 0.999 else np.array([1, 0, 0])
                tangent = np.cross(up, direction)
                if np.linalg.norm(tangent) > 1e-8:
                    tangent /= np.linalg.norm(tangent)
                bitangent = np.cross(direction, tangent)

                direction = local_scatter[0] * tangent + local_scatter[1] * bitangent + local_scatter[2] * direction

            else:
                exit_point = np.array([exit_x, exit_y, exit_z])
                exit_patch = _compute_exit_patch(
                    exit_x, exit_y, exit_z, exit_face,
                    grid_min, grid_max, patches_per_face
                )

                _, _, _, enx, eny, enz = _get_patch_geometry(
                    exit_patch, patches_per_face, grid_min, grid_max
                )
                exit_normal = np.array([enx, eny, enz])

                cos_theta = abs(np.dot(direction, exit_normal))
                exit_sh = eval_sh_basis(direction.reshape(1, 3))[0]

                output_sh[exit_patch] += weight * cos_theta * exit_sh

                break

            if weight < 0.01 and bounce > 3:
                if rng.random() > weight / 0.01:
                    break
                weight = 0.01

    return output_sh

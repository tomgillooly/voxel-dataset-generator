"""Spherical harmonics utilities for light transport.

Provides Python implementations of SH evaluation and projection for use
outside of Metal kernels (e.g., validation, visualization).
"""

import numpy as np
from typing import Union, Tuple

# SH normalization constants (matching Metal kernel)
SH_C0 = 0.282094791773878   # 1 / (2 * sqrt(pi))
SH_C1 = 0.488602511902920   # sqrt(3 / (4 * pi))
SH_C2_0 = 1.092548430592079  # sqrt(15 / (4 * pi))
SH_C2_1 = 0.315391565252520  # sqrt(5 / (16 * pi))
SH_C2_2 = 0.546274215296040  # sqrt(15 / (16 * pi))


def eval_sh_basis(directions: np.ndarray) -> np.ndarray:
    """Evaluate order-2 spherical harmonics basis functions.

    Args:
        directions: Unit vectors, shape (..., 3)

    Returns:
        SH coefficients, shape (..., 9)
        Order: Y_0^0, Y_1^-1, Y_1^0, Y_1^1, Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2
    """
    directions = np.asarray(directions)
    orig_shape = directions.shape[:-1]

    # Flatten for processing
    dirs = directions.reshape(-1, 3)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    # Allocate output
    coeffs = np.zeros((dirs.shape[0], 9), dtype=np.float64)

    # l=0
    coeffs[:, 0] = SH_C0

    # l=1
    coeffs[:, 1] = SH_C1 * y
    coeffs[:, 2] = SH_C1 * z
    coeffs[:, 3] = SH_C1 * x

    # l=2
    coeffs[:, 4] = SH_C2_0 * x * y           # Y_2^-2
    coeffs[:, 5] = SH_C2_0 * y * z           # Y_2^-1
    coeffs[:, 6] = SH_C2_1 * (3.0 * z * z - 1.0)  # Y_2^0
    coeffs[:, 7] = SH_C2_0 * x * z           # Y_2^1
    coeffs[:, 8] = SH_C2_2 * (x * x - y * y)  # Y_2^2

    return coeffs.reshape(*orig_shape, 9)


def project_to_sh(
    directions: np.ndarray,
    values: np.ndarray,
    n_sh: int = 9
) -> np.ndarray:
    """Project values onto SH basis using Monte Carlo integration.

    Args:
        directions: Sample directions, shape (N, 3)
        values: Sample values, shape (N,) or (N, C) for C channels
        n_sh: Number of SH coefficients (default 9 for order 2)

    Returns:
        SH coefficients, shape (n_sh,) or (n_sh, C)
    """
    directions = np.asarray(directions)
    values = np.asarray(values)

    # Evaluate SH basis at all directions
    sh_basis = eval_sh_basis(directions)  # (N, 9)

    # Monte Carlo integration: (4*pi/N) * sum(value * Y)
    n_samples = directions.shape[0]
    weight = 4.0 * np.pi / n_samples

    if values.ndim == 1:
        # (N,) values -> (n_sh,) coefficients
        coeffs = weight * np.sum(values[:, None] * sh_basis[:, :n_sh], axis=0)
    else:
        # (N, C) values -> (n_sh, C) coefficients
        coeffs = weight * np.einsum('nc,ns->sc', values, sh_basis[:, :n_sh])

    return coeffs


def reconstruct_from_sh(
    directions: np.ndarray,
    coeffs: np.ndarray
) -> np.ndarray:
    """Reconstruct values from SH coefficients.

    Args:
        directions: Query directions, shape (..., 3)
        coeffs: SH coefficients, shape (n_sh,) or (n_sh, C)

    Returns:
        Reconstructed values, shape (...) or (..., C)
    """
    directions = np.asarray(directions)
    coeffs = np.asarray(coeffs)

    # Evaluate SH basis
    n_sh = coeffs.shape[0] if coeffs.ndim == 1 else coeffs.shape[0]
    sh_basis = eval_sh_basis(directions)  # (..., 9)
    sh_basis = sh_basis[..., :n_sh]  # (..., n_sh)

    if coeffs.ndim == 1:
        # (n_sh,) coefficients -> (...) values
        return np.sum(sh_basis * coeffs, axis=-1)
    else:
        # (n_sh, C) coefficients -> (..., C) values
        return np.einsum('...s,sc->...c', sh_basis, coeffs)


def sh_triple_product(
    l1: int, m1: int,
    l2: int, m2: int,
    l3: int, m3: int
) -> float:
    """Compute integral of product of three SH basis functions.

    Integral over sphere: int Y_l1^m1 * Y_l2^m2 * Y_l3^m3 dw

    This is used for rotating transfer functions and for analytical
    integration in some rendering equations.

    Args:
        l1, m1: First SH index (l, m)
        l2, m2: Second SH index
        l3, m3: Third SH index

    Returns:
        Triple product value (Gaunt coefficient)
    """
    # This is the Gaunt coefficient, computed via Wigner 3j symbols
    # For simplicity, we use numerical integration for order-2
    # A full implementation would use sympy or precomputed tables

    n_samples = 10000
    phi = np.random.uniform(0, 2 * np.pi, n_samples)
    cos_theta = np.random.uniform(-1, 1, n_samples)
    sin_theta = np.sqrt(1 - cos_theta ** 2)

    directions = np.stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ], axis=-1)

    sh_all = eval_sh_basis(directions)  # (N, 9)

    def lm_to_idx(l: int, m: int) -> int:
        return l * l + l + m

    idx1 = lm_to_idx(l1, m1)
    idx2 = lm_to_idx(l2, m2)
    idx3 = lm_to_idx(l3, m3)

    if max(idx1, idx2, idx3) >= 9:
        return 0.0  # Out of range for order-2

    # Monte Carlo integration
    return (4 * np.pi / n_samples) * np.sum(
        sh_all[:, idx1] * sh_all[:, idx2] * sh_all[:, idx3]
    )


def sample_cosine_hemisphere(n_samples: int, seed: int = None) -> np.ndarray:
    """Sample directions from cosine-weighted hemisphere (z-up).

    Args:
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Directions, shape (n_samples, 3)
    """
    rng = np.random.default_rng(seed)
    u = rng.random((n_samples, 2))

    phi = 2 * np.pi * u[:, 0]
    cos_theta = np.sqrt(u[:, 1])
    sin_theta = np.sqrt(1 - u[:, 1])

    return np.stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ], axis=-1)


def sample_uniform_sphere(n_samples: int, seed: int = None) -> np.ndarray:
    """Sample directions uniformly on unit sphere.

    Args:
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Directions, shape (n_samples, 3)
    """
    rng = np.random.default_rng(seed)
    u = rng.random((n_samples, 2))

    z = 1 - 2 * u[:, 0]
    r = np.sqrt(np.maximum(0, 1 - z * z))
    phi = 2 * np.pi * u[:, 1]

    return np.stack([
        r * np.cos(phi),
        r * np.sin(phi),
        z
    ], axis=-1)


def verify_sh_orthonormality(n_samples: int = 100000, seed: int = 42) -> np.ndarray:
    """Verify SH basis orthonormality via Monte Carlo integration.

    Args:
        n_samples: Number of samples for integration
        seed: Random seed

    Returns:
        Inner product matrix, shape (9, 9). Should be close to identity.
    """
    directions = sample_uniform_sphere(n_samples, seed)
    sh_values = eval_sh_basis(directions)  # (N, 9)

    # Compute inner products: int Y_i * Y_j dw = (4pi/N) sum Y_i * Y_j
    weight = 4 * np.pi / n_samples
    inner_products = weight * (sh_values.T @ sh_values)

    return inner_products


def get_sh_order(n_coeffs: int) -> int:
    """Get SH order from number of coefficients.

    Args:
        n_coeffs: Number of SH coefficients

    Returns:
        SH order (l_max)

    Raises:
        ValueError: If n_coeffs is not a valid SH coefficient count
    """
    # n_coeffs = (order + 1)^2
    order = int(np.sqrt(n_coeffs)) - 1
    if (order + 1) ** 2 != n_coeffs:
        raise ValueError(f"Invalid SH coefficient count: {n_coeffs}")
    return order


def get_n_sh_coeffs(order: int) -> int:
    """Get number of SH coefficients for given order.

    Args:
        order: SH order (l_max)

    Returns:
        Number of coefficients
    """
    return (order + 1) ** 2

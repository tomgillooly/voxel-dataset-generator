"""Optical properties for homogeneous translucent materials."""

from dataclasses import dataclass


@dataclass
class OpticalProperties:
    """Material optical properties for homogeneous subsurface scattering.

    All occupied voxels in a structure share these identical optical properties.

    Attributes:
        sigma_s: Scattering coefficient (probability of scattering per unit length)
        sigma_a: Absorption coefficient (probability of absorption per unit length)
        g: Henyey-Greenstein asymmetry parameter (-1 to 1)
            - g > 0: Forward scattering (light tends to continue in same direction)
            - g = 0: Isotropic scattering (equal probability in all directions)
            - g < 0: Back scattering (light tends to reverse direction)

    Example:
        >>> props = OpticalProperties(sigma_s=1.0, sigma_a=0.1, g=0.8)
        >>> props.albedo
        0.909090909...
        >>> props.mean_free_path
        0.909090909...
    """
    sigma_s: float = 1.0  # Scattering coefficient
    sigma_a: float = 0.1  # Absorption coefficient
    g: float = 0.8        # HG asymmetry parameter

    def __post_init__(self):
        """Validate optical properties."""
        if self.sigma_s < 0:
            raise ValueError(f"sigma_s must be non-negative, got {self.sigma_s}")
        if self.sigma_a < 0:
            raise ValueError(f"sigma_a must be non-negative, got {self.sigma_a}")
        if not -1 <= self.g <= 1:
            raise ValueError(f"g must be in [-1, 1], got {self.g}")
        if self.sigma_s == 0 and self.sigma_a == 0:
            raise ValueError("At least one of sigma_s or sigma_a must be positive")

    @property
    def sigma_t(self) -> float:
        """Extinction coefficient (total attenuation per unit length)."""
        return self.sigma_s + self.sigma_a

    @property
    def albedo(self) -> float:
        """Single-scattering albedo (probability of scattering vs absorption).

        Range [0, 1] where:
        - 0: Pure absorption (all light absorbed)
        - 1: Pure scattering (no absorption)
        """
        return self.sigma_s / self.sigma_t

    @property
    def mean_free_path(self) -> float:
        """Mean free path (average distance between interactions)."""
        return 1.0 / self.sigma_t

    def to_array(self) -> tuple:
        """Convert to tuple for kernel parameter passing."""
        return (self.sigma_s, self.sigma_a, self.sigma_t, self.albedo, self.g, self.mean_free_path)

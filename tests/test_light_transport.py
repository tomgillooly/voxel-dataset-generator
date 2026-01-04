"""Tests for light transport module."""

import numpy as np
import pytest
from pathlib import Path

from voxel_dataset_generator.light_transport import (
    OpticalProperties,
    MetalVoxelTracer,
    BoundaryPatch,
    generate_boundary_patches,
    patches_to_arrays,
    get_patch_info,
    eval_sh_basis,
    project_to_sh,
    reconstruct_from_sh,
    sample_uniform_sphere,
    get_n_sh_coeffs,
    get_sh_order,
    TransferMatrixBuilder,
    analyze_transfer_matrix,
)


class TestOpticalProperties:
    """Tests for OpticalProperties class."""

    def test_default_values(self):
        """Test default optical properties."""
        props = OpticalProperties()
        assert props.sigma_s == 1.0
        assert props.sigma_a == 0.1
        assert props.g == 0.8

    def test_derived_properties(self):
        """Test derived property calculations."""
        props = OpticalProperties(sigma_s=2.0, sigma_a=0.5, g=0.0)
        assert props.sigma_t == 2.5
        assert props.albedo == 0.8
        assert props.mean_free_path == 0.4

    def test_pure_scattering(self):
        """Test pure scattering (no absorption)."""
        props = OpticalProperties(sigma_s=1.0, sigma_a=0.0, g=0.0)
        assert props.albedo == 1.0

    def test_validation_negative_sigma(self):
        """Test validation rejects negative coefficients."""
        with pytest.raises(ValueError):
            OpticalProperties(sigma_s=-1.0)
        with pytest.raises(ValueError):
            OpticalProperties(sigma_a=-1.0)

    def test_validation_g_range(self):
        """Test validation of g parameter range."""
        with pytest.raises(ValueError):
            OpticalProperties(g=1.5)
        with pytest.raises(ValueError):
            OpticalProperties(g=-1.5)

    def test_validation_zero_coefficients(self):
        """Test validation rejects both coefficients being zero."""
        with pytest.raises(ValueError):
            OpticalProperties(sigma_s=0.0, sigma_a=0.0)


class TestBoundaryPatches:
    """Tests for boundary patch generation."""

    def test_patch_count(self):
        """Test correct number of patches generated."""
        grid_min = np.array([-32, -32, -32])
        grid_max = np.array([32, 32, 32])

        patches = generate_boundary_patches(grid_min, grid_max, patches_per_face=8)
        assert len(patches) == 6 * 64  # 384 patches

        patches = generate_boundary_patches(grid_min, grid_max, patches_per_face=4)
        assert len(patches) == 6 * 16  # 96 patches

    def test_patch_normals_unit_length(self):
        """Test that all patch normals are unit vectors."""
        grid_min = np.array([-1, -1, -1])
        grid_max = np.array([1, 1, 1])

        patches = generate_boundary_patches(grid_min, grid_max, patches_per_face=4)

        for patch in patches:
            norm = np.linalg.norm(patch.normal)
            assert np.isclose(norm, 1.0), f"Patch {patch.patch_id} normal not unit: {norm}"

    def test_patch_normals_outward(self):
        """Test that patch normals point outward from grid center."""
        grid_min = np.array([-1, -1, -1])
        grid_max = np.array([1, 1, 1])

        patches = generate_boundary_patches(grid_min, grid_max, patches_per_face=2)

        for patch in patches:
            # Vector from center to patch should have positive dot product with normal
            to_patch = patch.center  # Grid is centered at origin
            dot = np.dot(to_patch, patch.normal)
            assert dot > 0, f"Patch {patch.patch_id} normal not outward: dot={dot}"

    def test_patch_coverage(self):
        """Test that patches cover the entire boundary surface."""
        grid_min = np.array([-1, -1, -1])
        grid_max = np.array([1, 1, 1])

        patches = generate_boundary_patches(grid_min, grid_max, patches_per_face=4)

        # Total area should equal surface area of bounding box
        total_area = sum(p.area for p in patches)
        expected_area = 6 * 4.0  # 6 faces, each 2x2
        assert np.isclose(total_area, expected_area, rtol=1e-5)

    def test_patches_to_arrays(self):
        """Test conversion to numpy arrays."""
        grid_min = np.array([-1, -1, -1])
        grid_max = np.array([1, 1, 1])

        patches = generate_boundary_patches(grid_min, grid_max, patches_per_face=2)
        arrays = patches_to_arrays(patches)

        assert arrays['centers'].shape == (24, 3)
        assert arrays['normals'].shape == (24, 3)
        assert arrays['areas'].shape == (24,)
        assert arrays['face_ids'].shape == (24,)

    def test_get_patch_info(self):
        """Test patch info extraction."""
        face_id, patch_u, patch_v = get_patch_info(65, patches_per_face=8)
        assert face_id == 1  # Second face (-X)
        assert patch_u == 1
        assert patch_v == 0


class TestSphericalHarmonics:
    """Tests for spherical harmonics functions."""

    def test_sh_coefficients_count(self):
        """Test correct number of SH coefficients."""
        assert get_n_sh_coeffs(0) == 1
        assert get_n_sh_coeffs(1) == 4
        assert get_n_sh_coeffs(2) == 9
        assert get_n_sh_coeffs(3) == 16

    def test_sh_order_from_coeffs(self):
        """Test extracting order from coefficient count."""
        assert get_sh_order(1) == 0
        assert get_sh_order(4) == 1
        assert get_sh_order(9) == 2
        assert get_sh_order(16) == 3

    def test_sh_order_invalid(self):
        """Test invalid coefficient counts."""
        with pytest.raises(ValueError):
            get_sh_order(5)  # Not a valid count

    def test_eval_sh_single_direction(self):
        """Test SH evaluation for a single direction."""
        direction = np.array([0, 0, 1])  # +Z direction
        coeffs = eval_sh_basis(direction)

        assert coeffs.shape == (9,)
        # Y_0^0 should be constant
        assert np.isclose(coeffs[0], 0.282095, rtol=1e-4)

    def test_eval_sh_batch(self):
        """Test SH evaluation for multiple directions."""
        directions = np.random.randn(100, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        coeffs = eval_sh_basis(directions)
        assert coeffs.shape == (100, 9)

    def test_sh_orthonormality(self):
        """Test that SH basis functions are orthonormal."""
        # Monte Carlo integration
        n_samples = 50000
        directions = sample_uniform_sphere(n_samples, seed=42)
        sh_values = eval_sh_basis(directions)

        # Compute inner products: int Y_i * Y_j dw = (4*pi/N) * sum(Y_i * Y_j)
        weight = 4 * np.pi / n_samples
        inner_products = weight * (sh_values.T @ sh_values)

        # Should be close to identity matrix
        expected = np.eye(9)
        np.testing.assert_allclose(inner_products, expected, atol=0.05)

    def test_sh_projection_reconstruction(self):
        """Test that projection and reconstruction are inverses."""
        # Create a simple function (constant + linear in z)
        n_samples = 10000
        directions = sample_uniform_sphere(n_samples, seed=42)
        values = 1.0 + 0.5 * directions[:, 2]  # 1 + 0.5*z

        # Project to SH
        coeffs = project_to_sh(directions, values)

        # Reconstruct at test points
        test_dirs = sample_uniform_sphere(1000, seed=123)
        reconstructed = reconstruct_from_sh(test_dirs, coeffs)

        # True values at test points
        true_values = 1.0 + 0.5 * test_dirs[:, 2]

        # Should be close (order-2 SH can represent linear functions exactly)
        np.testing.assert_allclose(reconstructed, true_values, rtol=0.1)


class TestMetalVoxelTracer:
    """Tests for MetalVoxelTracer class."""

    @pytest.fixture
    def empty_grid(self):
        """Create an empty 64x64x64 voxel grid."""
        return np.zeros((64, 64, 64), dtype=np.uint8)

    @pytest.fixture
    def full_grid(self):
        """Create a fully occupied 64x64x64 voxel grid."""
        return np.ones((64, 64, 64), dtype=np.uint8)

    @pytest.fixture
    def sphere_grid(self):
        """Create a voxel grid with a sphere."""
        grid = np.zeros((64, 64, 64), dtype=np.uint8)
        center = np.array([32, 32, 32])
        radius = 20

        for z in range(64):
            for y in range(64):
                for x in range(64):
                    dist = np.linalg.norm([x - center[0], y - center[1], z - center[2]])
                    if dist <= radius:
                        grid[z, y, x] = 1

        return grid

    def test_empty_grid_zero_distance(self, empty_grid):
        """Test that empty grid returns zero distance."""
        tracer = MetalVoxelTracer(empty_grid)

        origins = np.array([[0, 0, -50]], dtype=np.float32)
        directions = np.array([[0, 0, 1]], dtype=np.float32)

        distances = tracer.trace_rays(origins, directions)
        assert np.allclose(distances, 0.0)

    def test_full_grid_diagonal(self, full_grid):
        """Test ray through full grid."""
        tracer = MetalVoxelTracer(full_grid, voxel_size=1.0)

        # Ray through center along Z axis
        origins = np.array([[0, 0, -50]], dtype=np.float32)
        directions = np.array([[0, 0, 1]], dtype=np.float32)

        distances = tracer.trace_rays(origins, directions)

        # Should be approximately 64 (grid size)
        assert np.isclose(distances[0], 64.0, rtol=0.05)

    def test_sphere_chord(self, sphere_grid):
        """Test that ray through sphere gives chord length."""
        tracer = MetalVoxelTracer(sphere_grid, voxel_size=1.0)

        # Ray through center should give diameter
        origins = np.array([[0, 0, -50]], dtype=np.float32)
        directions = np.array([[0, 0, 1]], dtype=np.float32)

        distances = tracer.trace_rays(origins, directions)

        # Expected: diameter = 2 * radius = 40
        assert np.isclose(distances[0], 40.0, rtol=0.15)

    def test_batch_rays(self, full_grid):
        """Test tracing multiple rays."""
        tracer = MetalVoxelTracer(full_grid, voxel_size=1.0)

        # 10 rays along Z axis
        origins = np.zeros((10, 3), dtype=np.float32)
        origins[:, 2] = -50

        directions = np.zeros((10, 3), dtype=np.float32)
        directions[:, 2] = 1

        distances = tracer.trace_rays(origins, directions)

        assert distances.shape == (10,)
        assert np.all(distances > 60)  # All should traverse most of grid

    def test_2d_ray_grid(self, full_grid):
        """Test tracing 2D grid of rays."""
        tracer = MetalVoxelTracer(full_grid, voxel_size=1.0)

        # 5x5 grid of rays
        origins = np.zeros((5, 5, 3), dtype=np.float32)
        origins[:, :, 2] = -50

        directions = np.zeros((5, 5, 3), dtype=np.float32)
        directions[:, :, 2] = 1

        distances = tracer.trace_rays(origins, directions)

        assert distances.shape == (5, 5)

    def test_update_voxels(self, empty_grid, full_grid):
        """Test updating voxel grid."""
        tracer = MetalVoxelTracer(empty_grid)

        origins = np.array([[0, 0, -50]], dtype=np.float32)
        directions = np.array([[0, 0, 1]], dtype=np.float32)

        # Initially empty
        distances = tracer.trace_rays(origins, directions)
        assert distances[0] == 0.0

        # Update to full
        tracer.update_voxels(full_grid)
        distances = tracer.trace_rays(origins, directions)
        assert distances[0] > 60

    def test_optical_properties(self, full_grid):
        """Test that optical properties are stored correctly."""
        props = OpticalProperties(sigma_s=2.0, sigma_a=0.5, g=-0.5)
        tracer = MetalVoxelTracer(full_grid, optical_props=props)

        assert tracer.optical_props.sigma_s == 2.0
        assert tracer.optical_props.sigma_a == 0.5
        assert tracer.optical_props.g == -0.5

    def test_get_info(self, full_grid):
        """Test getting tracer info."""
        tracer = MetalVoxelTracer(full_grid, voxel_size=0.5)
        info = tracer.get_info()

        assert info['resolution'] == (64, 64, 64)
        assert info['voxel_size'] == 0.5
        assert 'optical_properties' in info


class TestTransferMatrixBuilder:
    """Tests for transfer matrix computation."""

    @pytest.fixture
    def small_grid(self):
        """Create a small 16x16x16 grid for faster testing."""
        grid = np.zeros((16, 16, 16), dtype=np.uint8)
        # Fill center region
        grid[4:12, 4:12, 4:12] = 1
        return grid

    def test_builder_initialization(self, small_grid):
        """Test builder initialization."""
        tracer = MetalVoxelTracer(small_grid)
        builder = TransferMatrixBuilder(
            tracer,
            patches_per_face=2,
            sh_order=2,
            samples_per_condition=100
        )

        assert builder.n_patches == 24  # 6 faces * 4 patches
        assert builder.n_sh == 9
        assert builder.n_conditions == 24 * 9

    def test_metadata(self, small_grid):
        """Test builder metadata."""
        tracer = MetalVoxelTracer(small_grid)
        builder = TransferMatrixBuilder(
            tracer,
            patches_per_face=2,
            sh_order=2,
            samples_per_condition=100
        )

        metadata = builder.get_metadata()

        assert metadata['patches_per_face'] == 2
        assert metadata['sh_order'] == 2
        assert 'optical_properties' in metadata

    @pytest.mark.slow
    def test_compute_small(self, small_grid):
        """Test computing a small transfer matrix."""
        tracer = MetalVoxelTracer(small_grid, optical_props=OpticalProperties(
            sigma_s=1.0, sigma_a=0.1, g=0.0
        ))
        builder = TransferMatrixBuilder(
            tracer,
            patches_per_face=2,
            sh_order=1,  # Only 4 coefficients
            samples_per_condition=100
        )

        transfer = builder.compute()

        # Shape check
        expected_dim = 24 * 4  # 24 patches * 4 SH coeffs
        assert transfer.shape == (expected_dim, expected_dim)

        # Should have some non-zero entries
        assert np.any(transfer != 0)


class TestTransferMatrixAnalysis:
    """Tests for transfer matrix analysis functions."""

    def test_analyze_simple_matrix(self):
        """Test analysis of a simple matrix."""
        matrix = np.random.rand(100, 100).astype(np.float32)
        analysis = analyze_transfer_matrix(matrix)

        assert 'shape' in analysis
        assert 'min' in analysis
        assert 'max' in analysis
        assert 'mean' in analysis
        assert 'column_sum_mean' in analysis

    def test_analyze_sparse_matrix(self):
        """Test analysis of sparse matrix."""
        matrix = np.zeros((100, 100), dtype=np.float32)
        matrix[0, 0] = 1.0
        matrix[50, 50] = 0.5

        analysis = analyze_transfer_matrix(matrix)

        assert analysis['nnz_ratio'] < 0.01  # Very sparse


# Mark slow tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

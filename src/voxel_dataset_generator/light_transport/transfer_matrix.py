"""Transfer matrix computation for light transport.

Builds the full transfer matrix T where T[j, i] represents the transport
from incident condition i to exitant condition j.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import time

from .boundary_patches import generate_boundary_patches, patches_to_arrays
from .spherical_harmonics import get_n_sh_coeffs


class TransferMatrixBuilder:
    """Builds transfer matrices via Monte Carlo simulation.

    For each incident condition (patch × SH basis function), traces rays
    through the volume and records exitant radiance at all patches as
    SH coefficients.

    Transfer matrix T has shape (n_out, n_in) where:
    - n_in = n_out = n_patches × n_sh
    - T[j, i] = transport from incident condition i to exitant condition j
    - Incident condition i = (patch_i // n_sh, patch_i % n_sh)
    - Exitant condition j = (patch_j // n_sh, patch_j % n_sh)

    Example:
        >>> from voxel_dataset_generator.light_transport import (
        ...     MetalVoxelTracer, TransferMatrixBuilder
        ... )
        >>> voxels = np.load("voxels.npz")['voxels']
        >>> tracer = MetalVoxelTracer(voxels)
        >>> builder = TransferMatrixBuilder(
        ...     tracer, patches_per_face=8, sh_order=2,
        ...     samples_per_condition=10000
        ... )
        >>> transfer_matrix = builder.compute()
    """

    def __init__(
        self,
        tracer,  # MetalVoxelTracer
        patches_per_face: int = 8,
        sh_order: int = 2,
        samples_per_condition: int = 10000,
        seed: int = 42,
        batch_size: int = 1024,
        track_sh_error: bool = False
    ):
        """Initialize transfer matrix builder.

        Args:
            tracer: MetalVoxelTracer instance
            patches_per_face: Patches per axis per face (8 -> 384 total patches)
            sh_order: Spherical harmonics order (2 -> 9 coefficients, max 4 -> 25)
            samples_per_condition: Monte Carlo samples per incident condition
            seed: Random seed for reproducibility
            batch_size: Rays to process in parallel
            track_sh_error: If True, track per-patch SH reconstruction error statistics
        """
        # Validate SH order
        max_sh_order = 4
        if sh_order > max_sh_order:
            raise ValueError(
                f"SH order {sh_order} not supported. Maximum is {max_sh_order} "
                f"({(max_sh_order + 1) ** 2} coefficients)."
            )

        self.tracer = tracer
        self.patches_per_face = patches_per_face
        self.sh_order = sh_order
        self.samples_per_condition = samples_per_condition
        self.seed = seed
        self.batch_size = batch_size
        self.track_sh_error = track_sh_error

        # Computed values
        self.n_sh = get_n_sh_coeffs(sh_order)
        self.n_patches = 6 * patches_per_face * patches_per_face
        self.n_conditions = self.n_patches * self.n_sh

        # Generate patches
        self.patches = generate_boundary_patches(
            tracer.grid_min,
            tracer.grid_max,
            patches_per_face
        )
        self.patch_arrays = patches_to_arrays(self.patches)

    def compute(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        checkpoint_callback: Optional[Callable[[np.ndarray, int], None]] = None,
        checkpoint_interval: int = 100,
        resume_from: int = 0
    ) -> tuple:
        """Compute full transfer matrix.

        Args:
            progress_callback: Called with (current, total) for progress
            checkpoint_callback: Called with (partial_matrix, last_idx) for checkpointing
            checkpoint_interval: How often to call checkpoint callback
            resume_from: Start from this incident condition index

        Returns:
            If track_sh_error=False:
                Transfer matrix, shape (n_conditions, n_conditions)
            If track_sh_error=True:
                Tuple of (transfer_matrix, sh_error_stats) where sh_error_stats contains:
                - 'weight_sum': (n_conditions, n_patches) total weight per exit patch
                - 'weight_sq_sum': (n_conditions, n_patches) squared weight sum
                - 'sample_count': (n_conditions, n_patches) sample counts
                - 'sh_energy': (n_conditions, n_patches) energy captured by SH coeffs
        """
        dim = self.n_conditions
        transfer_matrix = np.zeros((dim, dim), dtype=np.float32)

        # Error tracking accumulators (only allocated if needed)
        if self.track_sh_error:
            weight_sum_all = np.zeros((self.n_conditions, self.n_patches), dtype=np.float64)
            weight_sq_sum_all = np.zeros((self.n_conditions, self.n_patches), dtype=np.float64)
            sample_count_all = np.zeros((self.n_conditions, self.n_patches), dtype=np.float64)

        start_time = time.time()

        for incident_idx in range(resume_from, self.n_conditions):
            incident_patch = incident_idx // self.n_sh
            incident_sh = incident_idx % self.n_sh

            # Simulate this incident condition
            exitant_sh, error_stats = self._simulate_condition(
                incident_patch, incident_sh, incident_idx
            )

            # Store in transfer matrix column
            transfer_matrix[:, incident_idx] = exitant_sh.flatten()

            # Aggregate error stats
            if self.track_sh_error and error_stats is not None:
                weight_sum_all[incident_idx, :] = error_stats['weight_sum']
                weight_sq_sum_all[incident_idx, :] = error_stats['weight_sq_sum']
                sample_count_all[incident_idx, :] = error_stats['sample_count']

            if progress_callback:
                progress_callback(incident_idx + 1, self.n_conditions)

            if checkpoint_callback and (incident_idx + 1) % checkpoint_interval == 0:
                checkpoint_callback(transfer_matrix, incident_idx)

        elapsed = time.time() - start_time
        print(f"Transfer matrix computed in {elapsed:.1f}s "
              f"({elapsed / self.n_conditions:.3f}s per condition)")

        if self.track_sh_error:
            # Compute SH energy from transfer matrix coefficients
            # For each incident condition, sum(sh_coeff^2) over each exit patch
            sh_energy_all = np.zeros((self.n_conditions, self.n_patches), dtype=np.float64)
            for incident_idx in range(self.n_conditions):
                col = transfer_matrix[:, incident_idx]
                for exit_patch in range(self.n_patches):
                    start = exit_patch * self.n_sh
                    end = start + self.n_sh
                    sh_energy_all[incident_idx, exit_patch] = np.sum(col[start:end] ** 2)

            sh_error_stats = {
                'weight_sum': weight_sum_all,
                'weight_sq_sum': weight_sq_sum_all,
                'sample_count': sample_count_all,
                'sh_energy': sh_energy_all,
            }

            # Print summary
            self._print_error_summary(sh_error_stats)

            return transfer_matrix, sh_error_stats

        return transfer_matrix, None

    def _simulate_condition(
        self,
        incident_patch: int,
        incident_sh: int,
        condition_idx: int
    ) -> tuple:
        """Simulate one incident condition.

        Returns:
            Tuple of (exitant_sh, error_stats) where error_stats is None
            if track_sh_error is False.
        """
        from .monte_carlo import simulate_incident_condition_cpu

        return simulate_incident_condition_cpu(
            voxels=self.tracer._voxels_np,
            grid_min=self.tracer._grid_min,
            grid_max=self.tracer._grid_max,
            resolution=self.tracer._resolution,
            voxel_size=self.tracer.voxel_size,
            optical_props=self.tracer.optical_props,
            incident_patch=incident_patch,
            incident_sh=incident_sh,
            patches=self.patch_arrays,
            n_patches=self.n_patches,
            n_sh=self.n_sh,
            n_samples=self.samples_per_condition,
            seed=self.seed + condition_idx,
            track_sh_error=self.track_sh_error
        )

    def _print_error_summary(self, sh_error_stats: Dict[str, np.ndarray]) -> None:
        """Print summary of SH reconstruction error statistics.

        Uses Parseval's theorem: for SH projection,
        ||f||^2 = sum(coeff_j^2) when f is in the SH basis span.

        For a truncated SH basis, we measure energy fraction captured:
        - Original energy proxy: sum(w_i^2) / n (mean squared weight)
        - SH energy: sum(coeff_j^2) where coeff_j = sum(w_i * Y_j(d_i)) / n
        - Energy fraction = SH_energy / Original_energy
        - Error = 1 - Energy_fraction
        """
        weight_sq_sum = sh_error_stats['weight_sq_sum']
        sample_count = sh_error_stats['sample_count']
        sh_energy = sh_error_stats['sh_energy']

        # Minimum samples needed for statistically meaningful error estimate
        MIN_SAMPLES = 10

        # Patch pairs with any samples
        any_samples_mask = sample_count > 0
        # Patch pairs with enough samples for reliable statistics
        enough_samples_mask = sample_count >= MIN_SAMPLES

        if not any_samples_mask.any():
            print("\nSH Error Analysis: No samples collected")
            return

        print(f"\n=== SH Reconstruction Error Analysis (order {self.sh_order}) ===")
        print(f"Patch pairs with any samples: {any_samples_mask.sum()} / {any_samples_mask.size}")
        print(f"Patch pairs with >= {MIN_SAMPLES} samples: {enough_samples_mask.sum()}")

        # Sample count distribution
        counts_with_samples = sample_count[any_samples_mask]
        print(f"\nSample count distribution (for pairs with samples):")
        print(f"  Min:    {counts_with_samples.min():.0f}")
        print(f"  Median: {np.median(counts_with_samples):.0f}")
        print(f"  Mean:   {counts_with_samples.mean():.1f}")
        print(f"  Max:    {counts_with_samples.max():.0f}")

        if not enough_samples_mask.any():
            print(f"\nNo patch pairs have >= {MIN_SAMPLES} samples. "
                  "Increase --samples for meaningful error analysis.")
            return

        # Use only well-sampled patch pairs for error statistics
        mask = enough_samples_mask
        counts = sample_count[mask]

        # Original energy: mean squared weight = sum(w_i^2) / n
        # This represents E[f^2] estimated from samples
        original_energy = weight_sq_sum[mask] / counts

        # SH energy: sum(coeff_j^2) where coeff is in transfer matrix
        # Transfer matrix stores: sum(w_i * Y_j(d_i)) / n_total_samples
        # So sh_energy = sum(coeff^2) which is already in the right units
        # But we need to scale: coeff was normalized by n_total, not n_exit
        # Actually: coeff = sum_exits(w * Y) / n_total
        # And original_energy = sum_exits(w^2) / n_exit
        # To compare: SH_energy should be scaled to per-exit-sample basis
        # sh_energy * (n_total / n_exit)^2 gives energy per exit sample?
        #
        # Simpler: both are "per total sample" normalized:
        # - original per exit: sum(w^2)/n_exit
        # - SH energy: sum(coeff^2) where coeff = sum(w*Y)/n_total
        #
        # Energy fraction = SH_energy * n_total^2 / (sum(w^2) * n_total / n_exit)
        #                 = SH_energy * n_total * n_exit / sum(w^2)
        n_total = self.samples_per_condition
        sh_energy_per_sample = sh_energy[mask] * (n_total ** 2) / counts

        # Energy fraction captured by SH
        with np.errstate(divide='ignore', invalid='ignore'):
            energy_fraction = np.where(
                original_energy > 1e-10,
                sh_energy_per_sample / original_energy,
                0.0
            )

        # Clamp to [0, 1] - values > 1 indicate numerical issues
        energy_fraction = np.clip(energy_fraction, 0.0, 1.0)

        # Error = energy NOT captured
        relative_error = 1.0 - energy_fraction

        # Filter to patch pairs with meaningful signal
        valid_mask = original_energy > 1e-10
        valid_error = relative_error[valid_mask]

        if len(valid_error) == 0:
            print("\nNo patch pairs with significant signal energy.")
            return

        # Statistics
        mean_err = np.mean(valid_error)
        median_err = np.median(valid_error)
        max_err = np.max(valid_error)
        p90_err = np.percentile(valid_error, 90)
        p95_err = np.percentile(valid_error, 95)
        p99_err = np.percentile(valid_error, 99)

        print(f"\nSH Truncation Error (for {len(valid_error)} well-sampled pairs with signal):")
        print(f"  Mean:   {mean_err:.1%}")
        print(f"  Median: {median_err:.1%}")
        print(f"  90th %%: {p90_err:.1%}")
        print(f"  95th %%: {p95_err:.1%}")
        print(f"  99th %%: {p99_err:.1%}")
        print(f"  Max:    {max_err:.1%}")
        print(f"  (0% = all energy captured, 100% = no energy captured)")

        # Also show energy fraction for positive interpretation
        valid_frac = energy_fraction[valid_mask]
        print(f"\nEnergy Captured by SH order {self.sh_order}:")
        print(f"  Mean:   {np.mean(valid_frac):.1%}")
        print(f"  Median: {np.median(valid_frac):.1%}")
        print(f"  Min:    {np.min(valid_frac):.1%}")

        # Identify worst cases among well-sampled pairs
        full_err = np.full(sample_count.shape, -1.0)
        full_err[mask] = relative_error

        # Only consider pairs with valid signal
        signal_energy = weight_sq_sum / np.maximum(sample_count, 1)
        valid_signal_mask = mask & (signal_energy > 1e-10)
        full_err[~valid_signal_mask] = -1.0

        # Find worst cases
        flat_err = full_err.flatten()
        flat_counts = sample_count.flatten()
        worst_indices = np.argsort(-flat_err)[:5]

        print(f"\nWorst patch pairs (highest error, >= {MIN_SAMPLES} samples):")
        flat_weight_sq = weight_sq_sum.flatten()
        flat_sh_energy = sh_energy.flatten()
        for i, global_idx in enumerate(worst_indices):
            if flat_err[global_idx] < 0:
                continue
            incident_cond = global_idx // self.n_patches
            exit_patch = global_idx % self.n_patches
            incident_patch = incident_cond // self.n_sh
            incident_sh_idx = incident_cond % self.n_sh
            error_val = flat_err[global_idx]
            n_samples = int(flat_counts[global_idx])
            orig_e = flat_weight_sq[global_idx] / n_samples
            sh_e = flat_sh_energy[global_idx] * (n_total ** 2) / n_samples
            print(f"  {i+1}. Patch {incident_patch}(SH{incident_sh_idx}) -> {exit_patch}: "
                  f"{error_val:.1%} error, {n_samples} samples, "
                  f"orig={orig_e:.2e}, sh={sh_e:.2e}")

    def get_metadata(self) -> Dict[str, Any]:
        """Get builder metadata for saving with results."""
        return {
            'patches_per_face': self.patches_per_face,
            'sh_order': self.sh_order,
            'n_sh': self.n_sh,
            'n_patches': self.n_patches,
            'track_sh_error': self.track_sh_error,
            'n_conditions': self.n_conditions,
            'samples_per_condition': self.samples_per_condition,
            'seed': self.seed,
            'grid_min': self.tracer.grid_min.tolist(),
            'grid_max': self.tracer.grid_max.tolist(),
            'resolution': list(self.tracer.resolution),
            'voxel_size': self.tracer.voxel_size,
            'optical_properties': {
                'sigma_s': self.tracer.optical_props.sigma_s,
                'sigma_a': self.tracer.optical_props.sigma_a,
                'g': self.tracer.optical_props.g,
            }
        }


def save_transfer_matrix(
    path: Path,
    transfer_matrix: np.ndarray,
    metadata: Dict[str, Any],
    sh_error_stats: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """Save transfer matrix to compressed npz file.

    Args:
        path: Output file path (.npz)
        transfer_matrix: Transfer matrix array
        metadata: Metadata dictionary
        sh_error_stats: Optional SH error statistics from Phase 0 validation
    """
    path = Path(path)

    # Build save dict
    save_dict = {'transfer_matrix': transfer_matrix}

    # Add metadata (flatten nested dicts)
    for k, v in metadata.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                save_dict[f"{k}_{kk}"] = vv
        elif isinstance(v, list):
            save_dict[k] = np.array(v)
        else:
            save_dict[k] = v

    # Add error stats if provided
    if sh_error_stats is not None:
        for k, v in sh_error_stats.items():
            save_dict[f"sh_error_{k}"] = v

    np.savez_compressed(path, **save_dict)


def load_transfer_matrix(path: Path) -> tuple:
    """Load transfer matrix from npz file.

    Args:
        path: Input file path (.npz)

    Returns:
        Tuple of (transfer_matrix, metadata_dict)
    """
    data = np.load(path)
    transfer_matrix = data['transfer_matrix']

    # Reconstruct metadata
    metadata = {}
    for key in data.files:
        if key == 'transfer_matrix':
            continue
        value = data[key]
        # Convert single-element arrays back to scalars
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        elif isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()

        # Handle nested keys like "optical_properties_sigma_s"
        if '_' in key and any(key.startswith(p + '_') for p in
                              ['optical_properties', 'grid', 'resolution']):
            parts = key.split('_', 1)
            if parts[0] not in metadata:
                metadata[parts[0]] = {}
            if isinstance(metadata[parts[0]], dict):
                metadata[parts[0]][parts[1]] = value
            else:
                metadata[key] = value
        else:
            metadata[key] = value

    return transfer_matrix, metadata


def analyze_transfer_matrix(transfer_matrix: np.ndarray) -> Dict[str, Any]:
    """Analyze transfer matrix properties.

    Args:
        transfer_matrix: Transfer matrix to analyze

    Returns:
        Dictionary with analysis results
    """
    # Basic statistics
    stats = {
        'shape': transfer_matrix.shape,
        'dtype': str(transfer_matrix.dtype),
        'min': float(transfer_matrix.min()),
        'max': float(transfer_matrix.max()),
        'mean': float(transfer_matrix.mean()),
        'std': float(transfer_matrix.std()),
        'nnz_ratio': float((transfer_matrix != 0).sum() / transfer_matrix.size),
    }

    # Energy conservation check (column sums)
    column_sums = np.abs(transfer_matrix).sum(axis=0)
    stats['column_sum_min'] = float(column_sums.min())
    stats['column_sum_max'] = float(column_sums.max())
    stats['column_sum_mean'] = float(column_sums.mean())

    # Symmetry check
    if transfer_matrix.shape[0] == transfer_matrix.shape[1]:
        symmetry_error = np.abs(transfer_matrix - transfer_matrix.T).max()
        stats['symmetry_error'] = float(symmetry_error)

    # Spectral properties (for small matrices)
    if transfer_matrix.shape[0] <= 500:
        try:
            eigenvalues = np.linalg.eigvals(transfer_matrix)
            stats['spectral_radius'] = float(np.abs(eigenvalues).max())
            stats['eigenvalue_real_range'] = (
                float(eigenvalues.real.min()),
                float(eigenvalues.real.max())
            )
        except np.linalg.LinAlgError:
            pass

    return stats


def visualize_transfer_matrix(
    transfer_matrix: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Transfer Matrix",
    log_scale: bool = True
) -> None:
    """Visualize transfer matrix as an image.

    Args:
        transfer_matrix: Transfer matrix to visualize
        output_path: Path to save image (if None, displays interactively)
        title: Plot title
        log_scale: Use log scale for better visibility
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    data = np.abs(transfer_matrix)
    if log_scale:
        data = np.log10(data + 1e-10)
        label = "log10(|T|)"
    else:
        label = "|T|"

    im = ax.imshow(data, cmap='viridis', aspect='equal')
    plt.colorbar(im, ax=ax, label=label)

    ax.set_xlabel("Incident condition (patch × SH)")
    ax.set_ylabel("Exitant condition (patch × SH)")
    ax.set_title(title)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

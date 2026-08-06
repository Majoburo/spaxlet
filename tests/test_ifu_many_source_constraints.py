"""Known-truth gates for soft constraints on the recovery cube."""

from functools import lru_cache

import numpy as np

from benchmarks.run_synthetic_many_source_recovery import fit_recovery_cube


@lru_cache(maxsize=None)
def _arm(spectral=0, spatial=0):
    return fit_recovery_cube(
        start="A",
        max_iter=300,
        optimizer="adaprox",
        spectral_smoothness_strength=spectral,
        spatial_smoothness_strength=spatial,
    )


def test_spectral_smoothness_300_improves_recovery_without_erasing_lines():
    baseline = _arm()
    selected = _arm(spectral=300)
    assert selected.chi2_per_valid_voxel < baseline.chi2_per_valid_voxel + 0.012
    assert np.max(selected.spectrum_relative_l2) < 0.045
    assert np.max(selected.morphology_relative_l2) < 0.15
    assert np.max(selected.integrated_flux_relative_error) < 0.035
    assert np.max(selected.line_flux_relative_error) < 0.075
    assert np.max(selected.line_peak_relative_error) < 0.06
    assert np.mean(selected.spectrum_relative_l2) < np.mean(
        baseline.spectrum_relative_l2
    )
    assert np.mean(selected.morphology_relative_l2) < np.mean(
        baseline.morphology_relative_l2
    )


def test_strong_spatial_smoothness_is_rejected_by_source_truth_gates():
    baseline = _arm()
    spatial = _arm(spatial=100)
    assert np.max(spatial.integrated_flux_relative_error) > 0.10
    assert np.max(spatial.morphology_relative_l2) > 0.22
    assert np.min(spatial.morphology_identity_margin) < 0.80
    assert np.max(spatial.integrated_flux_relative_error) > np.max(
        baseline.integrated_flux_relative_error
    )

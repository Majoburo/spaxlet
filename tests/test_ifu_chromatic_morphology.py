"""Recovery and fail-closed tests for wavelength-dependent morphology."""

from functools import lru_cache

import numpy as np

from benchmarks.many_source_recovery_contract import (
    RECOVERY_SOURCE_SPECS,
    chromatic_line_morphologies,
    chromatic_source_cubes,
    recovery_morphologies,
)
from benchmarks.run_synthetic_chromatic_morphology_recovery import (
    fit_chromatic_cube,
)


@lru_cache(maxsize=None)
def _fit(model_kind, start="A", seed=8675309, spectral_mode="oracle", max_iter=300):
    return fit_chromatic_cube(
        model_kind=model_kind,
        spectral_mode=spectral_mode,
        optimizer="adaprox" if spectral_mode == "oracle" else "variable_projection",
        start=start,
        seed=seed,
        max_iter=max_iter,
    )


def test_chromatic_fixture_has_real_source_level_morphology_changes():
    continuum = recovery_morphologies()
    lines = chromatic_line_morphologies()
    assert chromatic_source_cubes().shape == (10, 96, 33, 33)
    assert len(RECOVERY_SOURCE_SPECS) == 10
    difference = np.linalg.norm(lines - continuum, axis=(1, 2)) / np.linalg.norm(
        continuum, axis=(1, 2)
    )
    assert np.min(difference) > 0.35
    np.testing.assert_allclose(lines.sum(axis=(1, 2)), 1.0)


def test_rank_one_model_fails_predictive_and_attribution_gates():
    rank1 = _fit("rank1")
    assert rank1.heldout_chi2_per_voxel > 2.0
    assert np.max(rank1.line_flux_relative_error) > 0.20
    assert np.max(rank1.source_cube_relative_l2) > 0.60
    assert np.max(rank1.line_centroid_error_px) > 1.0


def test_truth_independent_start_selection_recovers_two_morphologies():
    results = [_fit("continuum_line", start) for start in "ABC"]
    selected = min(results, key=lambda result: result.heldout_chi2_per_voxel)
    assert selected.start == "A"
    assert selected.heldout_chi2_per_voxel < 1.05
    assert selected.relative_projected_gradient < 3e-5
    assert np.max(selected.line_flux_relative_error) < 0.06
    assert np.max(selected.line_peak_relative_error) < 0.06
    assert np.max(selected.source_cube_relative_l2) < 0.23
    assert np.max(selected.continuum_morphology_relative_l2) < 0.25
    assert np.max(selected.line_morphology_relative_l2) < 0.30
    assert np.max(selected.line_centroid_error_px) < 0.17


def test_two_morphology_recovery_survives_an_independent_noise_draw():
    result = _fit("continuum_line", seed=104729)
    assert result.heldout_chi2_per_voxel < 1.05
    assert np.max(result.line_flux_relative_error) < 0.05
    assert np.max(result.source_cube_relative_l2) < 0.30
    assert np.max(result.line_morphology_relative_l2) < 0.31
    assert np.max(result.line_centroid_error_px) < 0.13


def test_free_two_component_spectra_improve_scene_but_fail_attribution():
    result = _fit(
        "continuum_line",
        spectral_mode="free",
        max_iter=160,
    )
    assert result.heldout_chi2_per_voxel < 2.0
    assert np.max(result.source_cube_relative_l2) > 0.60
    assert np.max(result.integrated_flux_relative_error) > 0.30

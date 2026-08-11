"""Tests for truth-gated real-residual injection calibration."""

import json

import numpy as np

from benchmarks.calibrate_spt0311_residual_injections import (
    SOURCE_DIAGNOSTICS,
    build_calibration,
)


def _write_product(directory, spectra, morphologies, models):
    directory.mkdir()
    np.savez_compressed(
        directory / "spt0311_joint_deblend.npz",
        names=np.asarray(["lens", "W"]),
        latent_wavelength_um=np.asarray([3.92, 3.95, 3.98]),
        latent_spectra_jy=spectra,
        morphologies=morphologies,
        prism_model=models,
        g395h_model=models,
    )


def _campaign(tmp_path, bad_source=False):
    spectra = np.asarray([[1.0, 2.0, 1.0], [1.5, 0.5, 1.0]])
    morphologies = np.zeros((2, 3, 3))
    morphologies[0, :2, :2] = 0.25
    morphologies[1, 1:, 1:] = 0.25
    model = np.einsum("sc,syx->cyx", spectra, morphologies)
    baseline = tmp_path / "baseline"
    _write_product(baseline, spectra, morphologies, model)
    (baseline / "spt0311_joint_report.json").write_text(
        json.dumps(
            {"fit": {"science_windows_um": {"oiii": [3.91, 3.99]}}}
        )
    )
    baseline_product = str((baseline / "spt0311_joint_deblend.npz").resolve())

    runs = {}
    for label, scale in (("A", 1.001), ("B", 0.999)):
        directory = tmp_path / label
        fitted_spectra = spectra * scale
        if bad_source:
            fitted_spectra = fitted_spectra.copy()
            fitted_spectra[1] = fitted_spectra[1, ::-1] * 1.4
        fitted_model = np.einsum("sc,syx->cyx", fitted_spectra, morphologies)
        _write_product(directory, fitted_spectra, morphologies, fitted_model)
        (directory / "spt0311_joint_report.json").write_text(
            json.dumps(
                {
                    "data_override": {
                        "metadata": {
                            "kind": "real_residual_injection",
                            "baseline_product": baseline_product,
                        }
                    }
                }
            )
        )
        runs[label] = {"directory": str(directory)}
    per_source = {
        name: {metric: 0.01 for metric in SOURCE_DIAGNOSTICS}
        for name in ("lens", "W")
    }
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "numerical_gate_accepted": True,
                "chi_square_relative_spread": 0.005,
                "maximum_model_relative_l2": 0.01,
                "sources": per_source,
                "runs": runs,
            }
        )
    )
    return baseline, validation


def test_calibration_records_only_sources_that_recover(tmp_path):
    baseline, validation = _campaign(tmp_path)
    result = build_calibration(baseline, [validation])
    assert result["kind"] == "real_residual_injection"
    assert result["scene"]["recoverable"]
    assert result["sources"]["lens"]["reportable"]
    assert result["sources"]["W"]["reportable"]
    assert result["sources"]["W"]["maximum_morphology_relative_l2"] == 0.0125


def test_calibration_marks_failed_truth_recovery_nonreportable(tmp_path):
    baseline, validation = _campaign(tmp_path, bad_source=True)
    result = build_calibration(baseline, [validation])
    assert result["sources"]["lens"]["reportable"]
    assert not result["sources"]["W"]["reportable"]
    assert (
        result["sources"]["W"]["injection_truth_recovery"]
        ["integrated_flux_relative_error"]
        > 0.15
    )

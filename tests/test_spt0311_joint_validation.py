"""Fail-closed validation tests for realistic SPT0311 campaigns."""

import json

import numpy as np

from benchmarks.validate_spt0311_joint_starts import validate_runs


def _run(label, converged=True, collision=False):
    spectra = np.asarray([[1.0, 0.2, 0.1], [0.1, 0.5, 1.0]])
    morphologies = np.zeros((2, 5, 5))
    morphologies[0, 1:3, 1:3] = 1
    morphologies[1, 2:4, 2:4] = 1
    if collision:
        spectra[1] = spectra[0]
        morphologies[1] = morphologies[0]
    model = np.sum(spectra[:, :, None, None] * morphologies[:, None], axis=0)
    first_observables = {
        name: {
            "integrated_spectrum_l1": value,
            "effective_morphology_pixels": 4.0,
            "centroid_yx": [1.5 + index, 1.5 + index],
            "window_flux_density_integrals": {"oiii": value},
        }
        for index, (name, value) in enumerate((("lens", 1.30), ("W", 1.60)))
    }
    final_observables = {
        name: {
            **value,
            "integrated_spectrum_l1": value["integrated_spectrum_l1"] * 1.001,
            "window_flux_density_integrals": {
                "oiii": value["window_flux_density_integrals"]["oiii"] * 1.001
            },
        }
        for name, value in first_observables.items()
    }
    return {
        "label": label,
        "directory": "/tmp/" + label,
        "report": {
            "start": label,
            "fit": {
                "iterations": 40,
                "relative_projected_gradient": 1e-5 if converged else 0.1,
                "optimality_converged": converged,
                "chi_square_per_valid_voxel": 1.2,
                "optimality_checks": [
                    {"iterations": 20, "source_observables": first_observables}
                ],
                "final_source_observables": final_observables,
            },
        },
        "signature": {"same": True},
        "names": ("lens", "W"),
        "spectra": spectra / np.linalg.norm(spectra, axis=1)[:, None],
        "morphologies": morphologies
        / np.linalg.norm(morphologies, axis=(1, 2))[:, None, None],
        "models": {"prism": model, "g395h": model},
        "whitened": {"prism": np.ones((5, 5)), "g395h": np.ones((5, 5))},
    }


CALIBRATION = {
    "kind": "real_residual_injection",
    "scene": {
        "maximum_chi_square_relative_spread": 0.01,
        "maximum_model_relative_l2": 0.02,
    },
    "sources": {
        "default": {
            "maximum_spectrum_shape_relative_l2": 0.1,
            "maximum_morphology_relative_l2": 0.1,
            "maximum_joint_collision_score": 0.8,
            "maximum_checkpoint_spectrum_l1_relative_change": 0.01,
            "maximum_checkpoint_effective_area_relative_change": 0.01,
            "maximum_checkpoint_centroid_shift_px": 0.01,
            "maximum_checkpoint_window_flux_relative_change": 0.01,
        }
    },
}


def test_numerically_stable_campaign_waits_for_real_residual_calibration():
    result = validate_runs([_run("A"), _run("B")])
    assert result["numerical_gate_accepted"]
    assert not result["science_ready"]
    assert "real-residual" in result["scene_rejection_reasons"][0]


def test_empirically_calibrated_stable_campaign_is_accepted():
    result = validate_runs([_run("A"), _run("B")], calibration=CALIBRATION)
    assert result["scene_fit_accepted"]
    assert result["science_ready"]


def test_unconverged_campaign_fails_the_hard_gate():
    result = validate_runs([_run("A", False), _run("B", False)], calibration=CALIBRATION)
    assert not result["numerical_gate_accepted"]
    assert not result["science_ready"]


def test_collision_is_judged_against_empirical_calibration():
    result = validate_runs(
        [_run("A", collision=True), _run("B", collision=True)],
        calibration=CALIBRATION,
    )
    assert result["scene_fit_accepted"]
    assert not result["component_attribution_accepted"]


def test_failed_injection_recovery_cannot_be_overridden_by_loose_thresholds():
    calibration = json.loads(json.dumps(CALIBRATION))
    calibration["sources"]["W"] = {
        **calibration["sources"]["default"],
        "reportable": False,
    }
    result = validate_runs([_run("A"), _run("B")], calibration=calibration)
    assert result["scene_fit_accepted"]
    assert not result["sources"]["W"]["attribution_accepted"]
    assert "failed recovery" in result["sources"]["W"]["reasons"][0]

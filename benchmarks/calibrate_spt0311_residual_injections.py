"""Build fail-closed validation thresholds from real-residual injection fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SOURCE_DIAGNOSTICS = (
    "maximum_spectrum_shape_relative_l2",
    "maximum_morphology_relative_l2",
    "maximum_joint_collision_score",
    "maximum_checkpoint_spectrum_l1_relative_change",
    "maximum_checkpoint_effective_area_relative_change",
    "maximum_checkpoint_centroid_shift_px",
    "maximum_checkpoint_window_flux_relative_change",
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--validation", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin", type=float, default=1.25)
    parser.add_argument("--maximum-scene-model-relative-l2", type=float, default=0.03)
    parser.add_argument("--maximum-spectrum-shape-relative-l2", type=float, default=0.15)
    parser.add_argument("--maximum-morphology-relative-l2", type=float, default=0.25)
    parser.add_argument("--maximum-integrated-flux-relative-error", type=float, default=0.15)
    parser.add_argument("--maximum-window-flux-relative-error", type=float, default=0.20)
    return parser


def _relative_l2(first, second):
    return float(np.linalg.norm(first - second)) / max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )


def _normalized(values, axes):
    norm = np.linalg.norm(values, axis=axes, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm > 0)


def _product(directory):
    path = Path(directory) / "spt0311_joint_deblend.npz"
    with np.load(path) as product:
        result = {
            "path": str(path.resolve()),
            "names": tuple(str(value) for value in product["names"]),
            "spectra": np.asarray(product["latent_spectra_jy"], dtype=float),
            "wavelength_um": np.asarray(product["latent_wavelength_um"], dtype=float),
            "morphologies": np.asarray(product["morphologies"], dtype=float),
            "models": {
                arm: np.asarray(product[arm + "_model"], dtype=float)
                for arm in ("prism", "g395h")
            },
        }
    return result


def _window_integrals(product, windows):
    result = {}
    wavelength = product["wavelength_um"]
    for label, bounds in windows.items():
        selected = (wavelength >= bounds[0]) & (wavelength <= bounds[1])
        if np.count_nonzero(selected) < 2:
            continue
        result[label] = np.trapz(
            product["spectra"][:, selected], wavelength[selected], axis=1
        )
    return result


def _truth_recovery(candidate, truth, windows):
    if candidate["names"] != truth["names"]:
        raise ValueError("injection fit source order differs from the baseline")
    truth_spectra = _normalized(truth["spectra"], (1,))
    candidate_spectra = _normalized(candidate["spectra"], (1,))
    truth_morphologies = _normalized(truth["morphologies"], (1, 2))
    candidate_morphologies = _normalized(candidate["morphologies"], (1, 2))
    spectral = np.linalg.norm(candidate_spectra - truth_spectra, axis=1)
    morphology = np.linalg.norm(
        candidate_morphologies - truth_morphologies, axis=(1, 2)
    )
    truth_flux = np.sum(truth["spectra"], axis=1)
    candidate_flux = np.sum(candidate["spectra"], axis=1)
    flux = np.abs(candidate_flux - truth_flux) / np.maximum(
        np.abs(truth_flux), np.finfo(float).tiny
    )
    truth_windows = _window_integrals(truth, windows)
    candidate_windows = _window_integrals(candidate, windows)
    window_errors = []
    for label in truth_windows:
        window_errors.append(
            np.abs(candidate_windows[label] - truth_windows[label])
            / np.maximum(np.abs(truth_windows[label]), np.finfo(float).tiny)
        )
    window_flux = (
        np.max(np.asarray(window_errors), axis=0)
        if window_errors
        else np.zeros(len(truth["names"]))
    )
    models = {
        arm: _relative_l2(candidate["models"][arm], truth["models"][arm])
        for arm in truth["models"]
    }
    return spectral, morphology, flux, window_flux, models


def build_calibration(
    baseline_dir,
    validation_paths,
    *,
    margin=1.25,
    maximum_scene_model_relative_l2=0.03,
    maximum_spectrum_shape_relative_l2=0.15,
    maximum_morphology_relative_l2=0.25,
    maximum_integrated_flux_relative_error=0.15,
    maximum_window_flux_relative_error=0.20,
):
    """Convert validated injection campaigns into empirical acceptance limits."""

    if not np.isfinite(margin) or margin < 1:
        raise ValueError("calibration margin must be finite and at least one")
    truth = _product(baseline_dir)
    baseline_product = Path(truth["path"])
    baseline_report = json.loads(
        (Path(baseline_dir) / "spt0311_joint_report.json").read_text()
    )
    windows = baseline_report.get("fit", {}).get("science_windows_um")
    if not windows:
        raise ValueError("baseline report does not declare science windows")
    validations = [json.loads(Path(path).read_text()) for path in validation_paths]
    if not validations:
        raise ValueError("at least one injection validation is required")

    truth_metrics = {name: [] for name in truth["names"]}
    model_errors = []
    for validation in validations:
        if not validation.get("numerical_gate_accepted", False):
            raise ValueError("an injection campaign failed its numerical gate")
        for run in validation["runs"].values():
            directory = Path(run["directory"])
            report = json.loads((directory / "spt0311_joint_report.json").read_text())
            metadata = (report.get("data_override") or {}).get("metadata") or {}
            if metadata.get("kind") != "real_residual_injection":
                raise ValueError("a campaign run is not a real-residual injection")
            declared_baseline = Path(metadata.get("baseline_product", "")).resolve()
            if declared_baseline != baseline_product:
                raise ValueError("an injection uses a different baseline product")
            spectral, morphology, flux, window_flux, models = _truth_recovery(
                _product(directory), truth, windows
            )
            model_errors.extend(models.values())
            for index, name in enumerate(truth["names"]):
                truth_metrics[name].append(
                    {
                        "spectrum_shape_relative_l2": float(spectral[index]),
                        "morphology_relative_l2": float(morphology[index]),
                        "integrated_flux_relative_error": float(flux[index]),
                        "window_flux_relative_error": float(window_flux[index]),
                    }
                )

    scene_recoverable = max(model_errors) <= maximum_scene_model_relative_l2
    scene = {
        "recoverable": scene_recoverable,
        "maximum_truth_model_relative_l2": max(model_errors),
        "maximum_chi_square_relative_spread": margin
        * max(value["chi_square_relative_spread"] for value in validations),
        "maximum_model_relative_l2": margin
        * max(value["maximum_model_relative_l2"] for value in validations),
    }
    sources = {}
    for name in truth["names"]:
        recovery = {
            metric: max(values[metric] for values in truth_metrics[name])
            for metric in (
                "spectrum_shape_relative_l2",
                "morphology_relative_l2",
                "integrated_flux_relative_error",
                "window_flux_relative_error",
            )
        }
        reportable = (
            recovery["spectrum_shape_relative_l2"]
            <= maximum_spectrum_shape_relative_l2
            and recovery["morphology_relative_l2"]
            <= maximum_morphology_relative_l2
            and recovery["integrated_flux_relative_error"]
            <= maximum_integrated_flux_relative_error
            and recovery["window_flux_relative_error"]
            <= maximum_window_flux_relative_error
        )
        thresholds = {
            metric: margin
            * max(validation["sources"][name][metric] for validation in validations)
            for metric in SOURCE_DIAGNOSTICS
        }
        sources[name] = {
            "reportable": reportable,
            "injection_truth_recovery": recovery,
            **thresholds,
        }
    return {
        "kind": "real_residual_injection",
        "baseline_product": str(baseline_product),
        "campaigns": [str(Path(path).resolve()) for path in validation_paths],
        "margin": float(margin),
        "truth_recovery_limits": {
            "maximum_scene_model_relative_l2": maximum_scene_model_relative_l2,
            "maximum_spectrum_shape_relative_l2": maximum_spectrum_shape_relative_l2,
            "maximum_morphology_relative_l2": maximum_morphology_relative_l2,
            "maximum_integrated_flux_relative_error": maximum_integrated_flux_relative_error,
            "maximum_window_flux_relative_error": maximum_window_flux_relative_error,
        },
        "scene": scene,
        "sources": sources,
    }


def main():
    args = _parser().parse_args()
    values = vars(args)
    output = values.pop("output")
    validation = values.pop("validation")
    result = build_calibration(validation_paths=validation, **values)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(str(output.resolve()), flush=True)


if __name__ == "__main__":
    main()

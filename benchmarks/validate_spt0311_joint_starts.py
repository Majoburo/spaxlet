"""Validate SPT0311 starts without importing idealized-mock thresholds."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-converged-starts", type=int, default=2)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="thresholds measured by injections into real-cube residuals",
    )
    parser.add_argument("--fail-on-rejection", action="store_true")
    return parser


def _relative_l2(first, second):
    scale = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(first - second) / scale)


def _cosine(first, second):
    scale = float(np.linalg.norm(first) * np.linalg.norm(second))
    return (
        float(np.vdot(first, second) / scale)
        if scale > np.finfo(float).tiny
        else float("nan")
    )


def _normalize(values, axes):
    norm = np.sqrt(np.sum(values**2, axis=axes, keepdims=True))
    return np.divide(
        values,
        norm,
        out=np.zeros_like(values, dtype=float),
        where=norm > 0,
    )


def _relative_scalar(first, second):
    return abs(float(first) - float(second)) / max(
        abs(float(first)), abs(float(second)), np.finfo(float).tiny
    )


def _observable_checkpoint_drift(report, names):
    """Measure late-iteration changes in directly reportable quantities."""

    fit = report["fit"]
    history = []
    for checkpoint in fit.get("optimality_checks", []):
        if "source_observables" in checkpoint:
            history.append(
                (int(checkpoint["iterations"]), checkpoint["source_observables"])
            )
    final = fit.get("final_source_observables")
    if final is not None:
        history.append((int(fit["iterations"]), final))
    # A final report at the same iteration replaces the periodic snapshot.
    deduplicated = {iteration: values for iteration, values in history}
    history = sorted(deduplicated.items())
    if len(history) < 2:
        return {name: None for name in names}
    previous, current = history[-2][1], history[-1][1]
    result = {}
    for name in names:
        if name not in previous or name not in current:
            result[name] = None
            continue
        first, second = previous[name], current[name]
        window_changes = []
        for window, first_value in first["window_flux_density_integrals"].items():
            second_value = second["window_flux_density_integrals"].get(window)
            if first_value is not None and second_value is not None:
                window_changes.append(_relative_scalar(first_value, second_value))
        result[name] = {
            "checkpoint_spectrum_l1_relative_change": _relative_scalar(
                first["integrated_spectrum_l1"], second["integrated_spectrum_l1"]
            ),
            "checkpoint_effective_area_relative_change": _relative_scalar(
                first["effective_morphology_pixels"],
                second["effective_morphology_pixels"],
            ),
            "checkpoint_centroid_shift_px": float(
                np.linalg.norm(
                    np.asarray(first["centroid_yx"], dtype=float)
                    - np.asarray(second["centroid_yx"], dtype=float)
                )
            ),
            "checkpoint_window_flux_relative_change": max(
                window_changes, default=0.0
            ),
        }
    return result


def _lag1(image, axis):
    first = np.take(image, np.arange(image.shape[axis] - 1), axis=axis).ravel()
    second = np.take(image, np.arange(1, image.shape[axis]), axis=axis).ravel()
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _signature(report):
    return {
        key: report[key]
        for key in (
            "model",
            "spectral_model",
            "prism_line_response",
            "source_names",
            "source_groups",
            "source_morphology_constraints",
            "noise_model",
            "high_redshift_spectral_support",
            "data_override",
        )
    } | {
        "observations": {
            arm: {
                key: value[key]
                for key in ("cube", "psf_cube", "shape", "wavelength_um")
            }
            for arm, value in report["observations"].items()
        }
    }


def load_run(label, directory):
    report = json.loads((directory / "spt0311_joint_report.json").read_text())
    with np.load(directory / "spt0311_joint_deblend.npz") as product:
        names = tuple(str(name) for name in product["names"])
        spectra = np.asarray(product["latent_spectra_jy"], dtype=float)
        morphologies = np.asarray(product["morphologies"], dtype=float)
        models = {
            arm: np.asarray(product[arm + "_model"], dtype=float)
            for arm in ("prism", "g395h")
        }
        whitened = {
            arm: np.asarray(
                product[arm + "_collapsed_whitened_residual"], dtype=float
            )
            for arm in ("prism", "g395h")
        }
    if names != tuple(report["source_names"]):
        raise ValueError("{} source order differs between report and product".format(label))
    if any(
        not np.all(np.isfinite(array))
        for array in (spectra, morphologies, *models.values(), *whitened.values())
    ):
        raise ValueError("{} contains non-finite output arrays".format(label))
    return {
        "label": label,
        "directory": str(directory.resolve()),
        "report": report,
        "signature": _signature(report),
        "names": names,
        "spectra": _normalize(spectra, (1,)),
        "morphologies": _normalize(morphologies, (1, 2)),
        "models": models,
        "whitened": whitened,
    }


def _thresholds_for_source(calibration, name):
    sources = calibration["sources"]
    return sources.get(name, sources.get("default"))


def validate_runs(runs, minimum_converged_starts=2, calibration=None):
    if minimum_converged_starts < 2:
        raise ValueError("at least two converged starts are required")
    if len(runs) < minimum_converged_starts:
        raise ValueError("fewer runs were supplied than the required minimum")
    if len({run["label"] for run in runs}) != len(runs):
        raise ValueError("run labels must be unique")

    hard_reasons = []
    if any(run["signature"] != runs[0]["signature"] for run in runs[1:]):
        hard_reasons.append("input/model provenance differs across starts")
    converged = [
        run for run in runs if run["report"]["fit"].get("optimality_converged", False)
    ]
    if len(converged) < minimum_converged_starts:
        hard_reasons.append("too few starts reached the projected-gradient gate")
    compared = converged if len(converged) >= 2 else runs
    names = runs[0]["names"]
    checkpoint_drift = {
        run["label"]: _observable_checkpoint_drift(run["report"], names)
        for run in runs
    }

    chi_square = {
        run["label"]: float(run["report"]["fit"]["chi_square_per_valid_voxel"])
        for run in runs
    }
    compared_chi = [chi_square[run["label"]] for run in compared]
    chi_spread = (
        (max(compared_chi) - min(compared_chi))
        / max(min(compared_chi), np.finfo(float).tiny)
        if compared_chi
        else float("inf")
    )
    model_pairwise = {}
    source_pairwise = {}
    for first, second in combinations(compared, 2):
        pair = first["label"] + "/" + second["label"]
        model_pairwise[pair] = {
            arm: _relative_l2(first["models"][arm], second["models"][arm])
            for arm in first["models"]
        }
        source_pairwise[pair] = {
            name: {
                "spectrum_shape_relative_l2": _relative_l2(
                    first["spectra"][index], second["spectra"][index]
                ),
                "morphology_relative_l2": _relative_l2(
                    first["morphologies"][index], second["morphologies"][index]
                ),
            }
            for index, name in enumerate(names)
        }
    maximum_model_difference = max(
        (
            value
            for comparison in model_pairwise.values()
            for value in comparison.values()
        ),
        default=float("inf"),
    )

    residual_structure = {
        run["label"]: {
            arm: {
                "rms": float(np.sqrt(np.mean(image**2))),
                "lag1_y": _lag1(image, 0),
                "lag1_x": _lag1(image, 1),
            }
            for arm, image in run["whitened"].items()
        }
        for run in runs
    }
    sources = {}
    for index, name in enumerate(names):
        spectral = [
            values[name]["spectrum_shape_relative_l2"]
            for values in source_pairwise.values()
        ]
        spatial = [
            values[name]["morphology_relative_l2"]
            for values in source_pairwise.values()
        ]
        collision = []
        for run in compared:
            for other_index, other in enumerate(names):
                if index == other_index:
                    continue
                collision.append(
                    {
                        "run": run["label"],
                        "other": other,
                        "spectrum_cosine": _cosine(
                            run["spectra"][index], run["spectra"][other_index]
                        ),
                        "morphology_cosine": _cosine(
                            run["morphologies"][index],
                            run["morphologies"][other_index],
                        ),
                    }
                )
        collision.sort(
            key=lambda item: item["spectrum_cosine"] * item["morphology_cosine"],
            reverse=True,
        )
        metrics = {
            "maximum_spectrum_shape_relative_l2": max(spectral, default=float("inf")),
            "maximum_morphology_relative_l2": max(spatial, default=float("inf")),
            "maximum_joint_collision_score": max(
                (
                    item["spectrum_cosine"] * item["morphology_cosine"]
                    for item in collision
                    if np.isfinite(item["spectrum_cosine"])
                    and np.isfinite(item["morphology_cosine"])
                ),
                default=float("nan"),
            ),
        }
        drift_values = [
            checkpoint_drift[run["label"]][name]
            for run in compared
            if checkpoint_drift[run["label"]][name] is not None
        ]
        if len(drift_values) != len(compared):
            drift_reason = "fewer than two observable checkpoints are available"
        else:
            drift_reason = None
            for metric in (
                "checkpoint_spectrum_l1_relative_change",
                "checkpoint_effective_area_relative_change",
                "checkpoint_centroid_shift_px",
                "checkpoint_window_flux_relative_change",
            ):
                metrics["maximum_" + metric] = max(
                    value[metric] for value in drift_values
                )
        reasons = []
        if drift_reason is not None:
            reasons.append(drift_reason)
        thresholds = _thresholds_for_source(calibration, name) if calibration else None
        if thresholds is None:
            reasons.append("requires calibration from real-residual injections")
        else:
            if thresholds.get("reportable") is False:
                reasons.append("source failed recovery in real-residual injections")
            for metric, value in metrics.items():
                threshold = thresholds.get(metric)
                if threshold is None:
                    reasons.append("calibration omits {}".format(metric))
                elif value > threshold:
                    reasons.append("{} exceeds empirical threshold".format(metric))
        sources[name] = {
            **metrics,
            "largest_pairwise_collisions": collision[:5],
            "attribution_accepted": not hard_reasons and not reasons,
            "reasons": reasons,
        }

    scene_reasons = list(hard_reasons)
    if calibration is None:
        scene_reasons.append("real-residual injection calibration was not supplied")
    else:
        if calibration.get("kind") != "real_residual_injection":
            scene_reasons.append("calibration is not a real-residual injection campaign")
        scene = calibration.get("scene", {})
        if scene.get("recoverable") is False:
            scene_reasons.append("scene failed recovery in real-residual injections")
        if chi_spread > scene.get("maximum_chi_square_relative_spread", -1):
            scene_reasons.append("chi-square start spread exceeds empirical threshold")
        if maximum_model_difference > scene.get("maximum_model_relative_l2", -1):
            scene_reasons.append("rendered-model spread exceeds empirical threshold")
    scene_accepted = not scene_reasons
    attribution_accepted = scene_accepted and all(
        source["attribution_accepted"] for source in sources.values()
    )
    return {
        "science_ready": attribution_accepted,
        "numerical_gate_accepted": not hard_reasons,
        "scene_fit_accepted": scene_accepted,
        "component_attribution_accepted": attribution_accepted,
        "hard_rejection_reasons": hard_reasons,
        "scene_rejection_reasons": scene_reasons,
        "calibration_kind": calibration.get("kind") if calibration else None,
        "converged_starts": [run["label"] for run in converged],
        "chi_square_relative_spread": chi_spread,
        "maximum_model_relative_l2": maximum_model_difference,
        "model_pairwise_relative_l2": model_pairwise,
        "source_pairwise_differences": source_pairwise,
        "residual_structure": residual_structure,
        "observable_checkpoint_drift": checkpoint_drift,
        "sources": sources,
        "runs": {
            run["label"]: {
                "directory": run["directory"],
                "start": run["report"].get("start"),
                "iterations": run["report"]["fit"]["iterations"],
                "relative_projected_gradient": run["report"]["fit"][
                    "relative_projected_gradient"
                ],
                "optimality_converged": run["report"]["fit"].get(
                    "optimality_converged", False
                ),
                "chi_square_per_valid_voxel": chi_square[run["label"]],
            }
            for run in runs
        },
    }


def main():
    args = _parser().parse_args()
    parsed = []
    for value in args.run:
        if value.count("=") != 1:
            raise ValueError("each run must be LABEL=DIRECTORY")
        label, directory = value.split("=", 1)
        parsed.append(load_run(label.strip(), Path(directory.strip())))
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    result = validate_runs(parsed, args.minimum_converged_starts, calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.fail_on_rejection and not result["science_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

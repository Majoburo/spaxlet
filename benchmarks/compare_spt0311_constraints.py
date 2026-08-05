"""Compare truth-independent diagnostics across SPT0311 constraint arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="label and benchmark output directory; first run is the spectral reference",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _parse_run(value):
    if "=" not in value:
        raise ValueError("each --run must be LABEL=DIR")
    label, directory = value.split("=", 1)
    return label, Path(directory)


def _normalized_spectra(spectra):
    norm = np.linalg.norm(spectra, axis=1)
    return np.divide(
        spectra,
        norm[:, None],
        out=np.zeros_like(spectra, dtype=float),
        where=norm[:, None] > 0,
    )


def _rotational_asymmetry(morphology, centroid):
    geometric_center = (np.asarray(morphology.shape) - 1) / 2
    centered = shift(
        morphology,
        geometric_center - centroid,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    norm = np.linalg.norm(centered)
    return float(
        np.linalg.norm(centered - np.rot90(centered, 2))
        / max(norm, np.finfo(float).tiny)
    )


def main():
    args = _parser().parse_args()
    loaded = []
    for value in args.run:
        label, directory = _parse_run(value)
        report = json.loads((directory / "spt0311_deblend_report.json").read_text())
        with np.load(directory / "spt0311_deblend.npz") as product:
            loaded.append(
                {
                    "label": label,
                    "report": report,
                    "names": tuple(str(name) for name in product["names"]),
                    "spectra": np.asarray(product["spectra_jy"], dtype=float),
                    "morphologies": np.asarray(product["morphologies"], dtype=float),
                    "centers": np.asarray(
                        product["latent_constraint_centers_yx"], dtype=float
                    ),
                    "fitted_centers": np.asarray(
                        product["fitted_centroids_yx"], dtype=float
                    ),
                    "model": np.asarray(product["model"], dtype=float),
                    "residual": np.asarray(product["residual"], dtype=float),
                    "valid": np.asarray(product["valid_mask"], dtype=bool),
                }
            )
    reference = _normalized_spectra(loaded[0]["spectra"])
    records = []
    for run in loaded:
        data = run["model"] + run["residual"]
        valid = run["valid"]
        data_rms = float(np.sqrt(np.mean(data[valid] ** 2)))
        residual_rms = float(np.sqrt(np.mean(run["residual"][valid] ** 2)))
        drift = np.linalg.norm(run["fitted_centers"] - run["centers"], axis=1)
        spectra = _normalized_spectra(run["spectra"])
        spectral_change = np.linalg.norm(spectra - reference, axis=1)
        effective_area = 1 / np.sum(run["morphologies"] ** 2, axis=(1, 2))
        asymmetry = np.asarray(
            [
                _rotational_asymmetry(morphology, center)
                for morphology, center in zip(
                    run["morphologies"], run["fitted_centers"]
                )
            ]
        )
        record = {
            "label": run["label"],
            "chi_square_per_valid_voxel": run["report"]["fit"][
                "chi_square_per_valid_voxel"
            ],
            "relative_projected_gradient": run["report"]["fit"][
                "relative_projected_gradient"
            ],
            "residual_rms_mjy_sr": residual_rms,
            "residual_to_data_rms": residual_rms / data_rms,
            "median_catalog_drift_px": float(np.median(drift)),
            "maximum_catalog_drift_px": float(np.max(drift)),
            "sources_over_one_pixel": int(np.count_nonzero(drift > 1)),
            "median_effective_morphology_pixels": float(np.median(effective_area)),
            "median_normalized_spectrum_change_from_first": float(
                np.median(spectral_change)
            ),
            "per_source_catalog_drift_px": dict(zip(run["names"], drift.tolist())),
            "per_source_spectrum_change_from_first": dict(
                zip(run["names"], spectral_change.tolist())
            ),
            "per_source_180_degree_asymmetry": dict(
                zip(run["names"], asymmetry.tolist())
            ),
        }
        records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_drift = records[0]["per_source_catalog_drift_px"]
    reference_asymmetry = records[0]["per_source_180_degree_asymmetry"]
    symmetry_candidates = [
        name
        for name in loaded[0]["names"]
        if reference_asymmetry[name] < 0.5 and reference_drift[name] < 1
    ]
    comparison = {
        "spectral_reference": records[0]["label"],
        "symmetry_candidate_rule": "180-degree asymmetry <0.5 and catalog drift <1 pixel",
        "symmetry_candidates_from_reference": symmetry_candidates,
        "runs": records,
    }
    (args.output_dir / "constraint_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    labels = [record["label"] for record in records]
    metrics = (
        ("chi_square_per_valid_voxel", "$\\chi^2/N_{valid}$"),
        ("residual_to_data_rms", "residual/data RMS"),
        ("median_catalog_drift_px", "median catalog drift [px]"),
        ("sources_over_one_pixel", "sources drifting >1 px"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, metrics):
        values = [record[key] for record in records]
        bars = axis.bar(labels, values, color="tab:blue", alpha=0.8)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
        axis.bar_label(bars, fmt="%.3g", padding=2, fontsize=8)
    figure.savefig(args.output_dir / "constraint_comparison.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()

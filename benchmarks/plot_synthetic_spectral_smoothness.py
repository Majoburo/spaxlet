"""Compare spectral smoothness strengths against known many-source IFU truth."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.ifu_parity_contracts import MANY_SOURCE_SPECS
from benchmarks.run_synthetic_many_source_ifu import fit_joint_model


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strengths", default="0,50,300")
    parser.add_argument("--slices", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--spatial-smoothness-strength", type=float, default=0)
    parser.add_argument("--start", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--constraint",
        choices=("positivity", "centroid", "selective_symmetry", "global_symmetry"),
        default="selective_symmetry",
    )
    parser.add_argument(
        "--support", choices=("truncated", "correct", "oversized"), default="correct"
    )
    return parser


def _strengths(text):
    values = tuple(float(value) for value in text.split(","))
    if not values or any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("strengths must be comma-separated non-negative numbers")
    return values


def _label(result):
    spectral = result.spectral_smoothness_strength
    spatial = result.spatial_smoothness_strength
    if spectral == 0 and spatial == 0:
        return "no smoothness"
    parts = []
    if spectral:
        parts.append(f"spectral {spectral:g}")
    if spatial:
        parts.append(f"spatial {spatial:g}")
    return " + ".join(parts)


def _plot_spectra(results, output):
    wavelength = results[0].channel_indices
    figure, axes = plt.subplots(
        len(MANY_SOURCE_SPECS),
        1,
        figsize=(15, 2.35 * len(MANY_SOURCE_SPECS)),
        sharex=True,
        constrained_layout=True,
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    for source, (axis, spec) in enumerate(zip(axes, MANY_SOURCE_SPECS)):
        truth = results[0].truth_spectra[source]
        axis.plot(wavelength, truth, color="black", lw=1.7, label="truth", zorder=5)
        for color, result in zip(colors, results):
            axis.plot(
                wavelength,
                result.fitted_spectra[source],
                color=color,
                lw=0.75,
                alpha=0.9,
                label=(
                    f"{_label(result)}; "
                    f"cos={result.spectrum_cosine[source]:.3f}"
                ),
            )
        axis.set_ylabel(spec.name)
        axis.text(
            0.995,
            0.94,
            (
                f"median peak S/N={results[0].median_peak_snr_per_slice[source]:.2f}; "
                f"integrated={results[0].integrated_peak_snr[source]:.1f}"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        axis.grid(alpha=0.15)
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    axes[-1].set_xlabel("native channel")
    figure.suptitle("Known truth versus native recovered spectra", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_morphologies(results, output):
    columns = 1 + len(results)
    figure, axes = plt.subplots(
        len(MANY_SOURCE_SPECS),
        columns,
        figsize=(2.5 * columns, 2.35 * len(MANY_SOURCE_SPECS)),
        constrained_layout=True,
        squeeze=False,
    )
    headings = ("truth",) + tuple(
        _label(result) for result in results
    )
    for source, spec in enumerate(MANY_SOURCE_SPECS):
        images = (results[0].truth_morphologies[source],) + tuple(
            result.fitted_morphologies[source] for result in results
        )
        common = max(float(np.max(image)) for image in images)
        for column, (axis, image) in enumerate(zip(axes[source], images)):
            axis.imshow(image, origin="lower", cmap="magma", vmin=0, vmax=common)
            axis.set_xticks([])
            axis.set_yticks([])
            if source == 0:
                axis.set_title(headings[column])
            if column == 0:
                axis.set_ylabel(spec.name)
            elif column > 0:
                result = results[column - 1]
                axis.text(
                    0.98,
                    0.03,
                    f"rel L2={result.morphology_relative_l2[source]:.2f}",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    color="white",
                    fontsize=7,
                )
    figure.suptitle("One shared morphology per source", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _median_image(cube, valid):
    return np.nanmedian(np.where(valid, cube, np.nan), axis=0)


def _plot_residuals(results, output):
    data_image = _median_image(results[0].data, results[0].valid)
    panels = [("data", data_image)]
    for result in results:
        panels.extend(
            (
                (f"model\n{_label(result)}", _median_image(result.model, result.valid)),
                (
                    (
                        f"residual\n{_label(result)}\n"
                        f"RMS={result.whitened_residual_rms:.3f}, "
                        f"chi2/N={result.chi2_per_voxel:.3f}"
                    ),
                    _median_image(result.residual, result.valid),
                ),
            )
        )
    common = float(
        np.nanpercentile(
            np.abs(np.concatenate([image.ravel() for _, image in panels])), 99.5
        )
    )
    figure, axes = plt.subplots(
        1, len(panels), figsize=(3.2 * len(panels), 3.7), constrained_layout=True
    )
    rendered = None
    for axis, (title, image) in zip(axes, panels):
        rendered = axis.imshow(
            image, origin="lower", cmap="coolwarm", vmin=-common, vmax=common
        )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("x [pixel]")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(rendered, ax=axes, shrink=0.7, label="median flux; common scale")
    figure.suptitle("Data, models, and residuals on exactly the same scale", fontsize=14)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _jsonable(result):
    omitted = {
        "channel_indices",
        "fitted_spectra",
        "fitted_morphologies",
        "truth_spectra",
        "truth_morphologies",
        "data",
        "model",
        "residual",
        "variance",
        "valid",
    }
    values = asdict(result)
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in values.items()
        if key not in omitted
    }


def main():
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = tuple(
        fit_joint_model(
            slices=args.slices,
            support=args.support,
            constraint=args.constraint,
            start=args.start,
            max_iter=args.max_iter,
            spectral_smoothness_strength=strength,
            spatial_smoothness_strength=args.spatial_smoothness_strength,
        )
        for strength in _strengths(args.strengths)
    )
    _plot_spectra(results, args.output_dir / "spectra_truth_comparison.png")
    _plot_morphologies(results, args.output_dir / "morphology_comparison.png")
    _plot_residuals(results, args.output_dir / "residual_comparison.png")
    report = {
        "purpose": "truth-based spectral smoothness comparison",
        "selection": "no post-hoc smoothing; every curve is a fitted model parameter",
        "results": [_jsonable(result) for result in results],
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

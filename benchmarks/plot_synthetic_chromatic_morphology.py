"""Plot rank-one failure and continuum-plus-line recovery diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.many_source_recovery_contract import (
    RECOVERY_SEED,
    RECOVERY_SHAPE,
    RECOVERY_SOURCE_SPECS,
    chromatic_line_morphologies,
    recovery_morphologies,
)
from benchmarks.run_synthetic_chromatic_morphology_recovery import (
    fit_chromatic_cube,
)


def _levels(image):
    return np.asarray((0.2, 0.5, 0.8)) * np.max(image)


def _scene_row(axes, result, title, truth_continuum, truth_line):
    train_count = np.maximum(np.sum(result.train_valid, axis=0), 1)
    data = np.sum(np.where(result.train_valid, result.data, 0.0), axis=0) / train_count
    model = np.sum(np.where(result.train_valid, result.model, 0.0), axis=0) / train_count
    residual = np.sum(
        np.where(
            result.train_valid,
            result.residual / np.sqrt(result.variance),
            0.0,
        ),
        axis=0,
    ) / np.sqrt(train_count)
    surface_limit = np.percentile(np.abs(np.concatenate((data.ravel(), model.ravel()))), 99.5)
    residual_limit = np.percentile(np.abs(residual), 99.5)
    for index, (axis, image, label) in enumerate(
        zip(axes, (data, model, residual), ("data", "model", "whitened residual"))
    ):
        limit = surface_limit if index < 2 else residual_limit
        shown = axis.imshow(
            image,
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(-0.5, RECOVERY_SHAPE[2] - 0.5, -0.5, RECOVERY_SHAPE[1] - 0.5),
        )
        for source_index, spec in enumerate(RECOVERY_SOURCE_SPECS):
            axis.contour(
                truth_continuum[source_index],
                levels=[0.5 * np.max(truth_continuum[source_index])],
                colors="cyan",
                linewidths=0.65,
                origin="lower",
            )
            axis.contour(
                truth_line[source_index],
                levels=[0.5 * np.max(truth_line[source_index])],
                colors="lime",
                linewidths=0.65,
                origin="lower",
            )
            axis.contour(
                result.fitted_continuum_morphologies[source_index],
                levels=[
                    0.5
                    * np.max(result.fitted_continuum_morphologies[source_index])
                ],
                colors="magenta",
                linewidths=0.65,
                linestyles="--",
                origin="lower",
            )
            if result.model_kind == "continuum_line":
                axis.contour(
                    result.fitted_line_morphologies[source_index],
                    levels=[0.5 * np.max(result.fitted_line_morphologies[source_index])],
                    colors="tab:orange",
                    linewidths=0.65,
                    linestyles=":",
                    origin="lower",
                )
            if index == 0:
                axis.text(
                    spec.center[1] + 0.3,
                    spec.center[0] + 0.3,
                    spec.name,
                    fontsize=5.5,
                    color="black",
                )
        axis.set(title=label, xlabel="x pixel", ylabel="y pixel")
        plt.colorbar(shown, ax=axis, fraction=0.046)
    axes[0].set_ylabel(title + "\ny pixel")


def _source_gallery(result, output, truth_continuum, truth_line):
    figure, axes = plt.subplots(5, 4, figsize=(13, 16), constrained_layout=True)
    for index, spec in enumerate(RECOVERY_SOURCE_SPECS):
        row, pair = divmod(index, 2)
        for column_offset, truth, fitted, label, colors in (
            (
                0,
                truth_continuum[index],
                result.fitted_continuum_morphologies[index],
                "continuum",
                ("cyan", "magenta"),
            ),
            (
                1,
                truth_line[index],
                result.fitted_line_morphologies[index],
                "line",
                ("lime", "tab:orange"),
            ),
        ):
            axis = axes[row, 2 * pair + column_offset]
            half = spec.support // 2
            y0, x0 = spec.center[0] - half, spec.center[1] - half
            y1, x1 = y0 + spec.support, x0 + spec.support
            truth_cutout = truth[y0:y1, x0:x1]
            fit_cutout = fitted[y0:y1, x0:x1]
            extent = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)
            axis.imshow(truth_cutout, origin="lower", cmap="Greys", extent=extent)
            axis.contour(
                truth_cutout,
                levels=_levels(truth_cutout),
                colors=colors[0],
                linewidths=0.9,
                origin="lower",
                extent=extent,
            )
            axis.contour(
                fit_cutout,
                levels=_levels(fit_cutout),
                colors=colors[1],
                linewidths=0.9,
                linestyles="--",
                origin="lower",
                extent=extent,
            )
            error = (
                result.continuum_morphology_relative_l2[index]
                if label == "continuum"
                else result.line_morphology_relative_l2[index]
            )
            axis.set(
                title="{} {} (rel L2={:.2f})".format(spec.name, label, error),
                xlabel="x pixel",
                ylabel="y pixel",
            )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _summary(result):
    return {
        "model_kind": result.model_kind,
        "start": result.start,
        "seed": result.seed,
        "iterations": result.iterations,
        "train_chi2_per_voxel": result.train_chi2_per_voxel,
        "heldout_chi2_per_voxel": result.heldout_chi2_per_voxel,
        "relative_projected_gradient": result.relative_projected_gradient,
        "maximum_spectrum_relative_l2": float(np.max(result.spectrum_relative_l2)),
        "maximum_integrated_flux_relative_error": float(
            np.max(result.integrated_flux_relative_error)
        ),
        "maximum_line_flux_relative_error": float(
            np.max(result.line_flux_relative_error)
        ),
        "maximum_line_peak_relative_error": float(
            np.max(result.line_peak_relative_error)
        ),
        "maximum_source_cube_relative_l2": float(
            np.max(result.source_cube_relative_l2)
        ),
        "maximum_continuum_morphology_relative_l2": float(
            np.max(result.continuum_morphology_relative_l2)
        ),
        "maximum_line_morphology_relative_l2": float(
            np.max(result.line_morphology_relative_l2)
        ),
        "maximum_line_centroid_error_px": float(
            np.max(result.line_centroid_error_px)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=RECOVERY_SEED)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank1 = fit_chromatic_cube(
        model_kind="rank1", spectral_mode="oracle", optimizer="adaprox",
        start="A", seed=args.seed, max_iter=args.max_iter,
    )
    candidates = [
        fit_chromatic_cube(
            model_kind="continuum_line", spectral_mode="oracle", optimizer="adaprox",
            start=start, seed=args.seed, max_iter=args.max_iter,
        )
        for start in "ABC"
    ]
    selected = min(candidates, key=lambda result: result.heldout_chi2_per_voxel)
    truth_continuum = recovery_morphologies()
    truth_line = chromatic_line_morphologies()
    figure, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    _scene_row(axes[0], rank1, "rank one", truth_continuum, truth_line)
    _scene_row(
        axes[1], selected, "continuum + line ({})".format(selected.start),
        truth_continuum, truth_line,
    )
    figure.suptitle(
        "50% contours — cyan/lime: true continuum/line; "
        "magenta/orange: fitted continuum/line"
    )
    figure.savefig(args.output_dir / "chromatic_scene_comparison.png", dpi=180)
    plt.close(figure)
    _source_gallery(
        selected,
        args.output_dir / "chromatic_source_contours.png",
        truth_continuum,
        truth_line,
    )
    report = {
        "selection_rule": "minimum held-out chi-square across A/B/C",
        "rank1": _summary(rank1),
        "continuum_line_candidates": [_summary(result) for result in candidates],
        "selected": _summary(selected),
    }
    (args.output_dir / "chromatic_recovery.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

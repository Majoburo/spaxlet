"""Plot absolute-position and source-level diagnostics for the recovery cube."""

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
    recovery_morphologies,
    recovery_spectra,
)
from benchmarks.run_synthetic_many_source_recovery import (
    RECOVERY_PILOT_MAX_ITER,
    fit_recovery_cube,
)


CONTOUR_FRACTIONS = (0.2, 0.5, 0.8)


def _levels(image):
    peak = float(np.max(image))
    return [fraction * peak for fraction in CONTOUR_FRACTIONS]


def _catalog_overlay(axis, morphologies, *, color, linestyle="-", labels=False):
    for spec, morphology in zip(RECOVERY_SOURCE_SPECS, morphologies):
        axis.contour(
            morphology,
            levels=_levels(morphology),
            colors=color,
            linewidths=0.75,
            linestyles=linestyle,
            origin="lower",
        )
        if labels:
            axis.text(
                spec.center[1] + 0.35,
                spec.center[0] + 0.35,
                spec.name,
                color=color,
                fontsize=6,
            )


def plot_recovery(result, output_dir):
    """Write scene and per-source diagnostics for one fitted start."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_morphologies = recovery_morphologies()
    truth_spectra = recovery_spectra()
    valid_count = np.maximum(np.sum(result.valid, axis=0), 1)
    collapsed = (
        np.sum(np.where(result.valid, result.data, 0.0), axis=0) / valid_count,
        np.sum(np.where(result.valid, result.model, 0.0), axis=0) / valid_count,
        np.sum(
            np.where(
                result.valid,
                result.residual / np.sqrt(result.variance),
                0.0,
            ),
            axis=0,
        )
        / np.sqrt(valid_count),
    )
    surface_limit = np.percentile(np.abs(np.concatenate([value.ravel() for value in collapsed[:2]])), 99.5)
    residual_limit = np.percentile(np.abs(collapsed[2]), 99.5)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    titles = ("masked data mean", "fitted model mean", "collapsed whitened residual")
    for index, (axis, image, title) in enumerate(zip(axes, collapsed, titles)):
        limit = surface_limit if index < 2 else residual_limit
        shown = axis.imshow(
            image,
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(-0.5, RECOVERY_SHAPE[2] - 0.5, -0.5, RECOVERY_SHAPE[1] - 0.5),
        )
        _catalog_overlay(axis, truth_morphologies, color="cyan", labels=True)
        _catalog_overlay(axis, result.fitted_morphologies, color="magenta", linestyle="--")
        axis.set(title=title, xlabel="x pixel", ylabel="y pixel")
        figure.colorbar(shown, ax=axis, fraction=0.047)
    axes[0].plot([], [], color="cyan", label="truth contours")
    axes[0].plot([], [], color="magenta", linestyle="--", label="fit contours")
    axes[0].legend(loc="upper right", fontsize=7)
    figure.suptitle(
        "start {}: chi2/N={:.4f}, projected gradient={:.2e}".format(
            result.start,
            result.chi2_per_valid_voxel,
            result.relative_projected_gradient,
        )
    )
    figure.savefig(output_dir / "scene_recovery.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(5, 4, figsize=(15, 17), constrained_layout=True)
    channel = np.arange(RECOVERY_SHAPE[0])
    for source_index, spec in enumerate(RECOVERY_SOURCE_SPECS):
        row = source_index // 2
        pair = source_index % 2
        morphology_axis = axes[row, 2 * pair]
        spectrum_axis = axes[row, 2 * pair + 1]
        half = spec.support // 2
        y0, x0 = spec.center[0] - half, spec.center[1] - half
        y1, x1 = y0 + spec.support, x0 + spec.support
        truth_image = truth_morphologies[source_index, y0:y1, x0:x1]
        fit_image = result.fitted_morphologies[source_index, y0:y1, x0:x1]
        morphology_axis.imshow(
            truth_image,
            origin="lower",
            cmap="Greys",
            extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5),
        )
        morphology_axis.contour(
            truth_image,
            levels=_levels(truth_image),
            colors="cyan",
            linewidths=1,
            origin="lower",
            extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5),
        )
        morphology_axis.contour(
            fit_image,
            levels=_levels(fit_image),
            colors="magenta",
            linewidths=1,
            linestyles="--",
            origin="lower",
            extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5),
        )
        morphology_axis.set(
            title="{} morphology (cos={:.4f})".format(
                spec.name, result.morphology_cosine[source_index]
            ),
            xlabel="x pixel",
            ylabel="y pixel",
        )
        spectrum_axis.plot(channel, truth_spectra[source_index], color="black", label="truth")
        spectrum_axis.plot(
            channel,
            result.fitted_spectra[source_index],
            color="tab:orange",
            linewidth=1,
            label="fit",
        )
        spectrum_axis.set(
            title="{} spectrum (rel L2={:.3f})".format(
                spec.name, result.spectrum_relative_l2[source_index]
            ),
            xlabel="channel",
            ylabel="integrated flux density",
        )
        if source_index == 0:
            spectrum_axis.legend(fontsize=7)
    figure.savefig(output_dir / "source_recovery.png", dpi=180)
    plt.close(figure)

    metrics = {
        "start": result.start,
        "seed": result.seed,
        "iterations": result.iterations,
        "chi2_per_valid_voxel": result.chi2_per_valid_voxel,
        "relative_projected_gradient": result.relative_projected_gradient,
        "flux_cancellation_ratio": result.flux_cancellation_ratio,
    }
    for name in (
        "spectrum_relative_l2",
        "spectrum_cosine",
        "morphology_relative_l2",
        "morphology_cosine",
        "centroid_error_px",
        "integrated_flux_relative_error",
        "signed_integrated_flux_relative_error",
        "signed_continuum_flux_relative_error",
        "signed_line_flux_relative_error",
        "morphology_identity_margin",
    ):
        metrics[name] = dict(
            zip(
                (spec.name for spec in RECOVERY_SOURCE_SPECS),
                np.asarray(getattr(result, name), dtype=float).tolist(),
            )
        )
    (output_dir / "recovery_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", choices=("A", "B", "C"), default="A")
    parser.add_argument("--max-iter", type=int, default=RECOVERY_PILOT_MAX_ITER)
    parser.add_argument("--seed", type=int, default=RECOVERY_SEED)
    args = parser.parse_args()
    plot_recovery(
        fit_recovery_cube(start=args.start, max_iter=args.max_iter, seed=args.seed),
        args.output_dir,
    )


if __name__ == "__main__":
    main()

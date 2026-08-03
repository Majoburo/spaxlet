"""Plot one Scarlet collaborator-fit product without lisasep dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from benchmarks.ifu_parity_metrics import morphology_metrics, translate_morphology


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _normalized(value):
    value = np.maximum(np.asarray(value, dtype=float), 0)
    total = float(np.sum(value))
    if total <= np.finfo(float).tiny:
        raise ValueError("morphology must contain positive flux")
    return value / total


def _load(args):
    metrics_path = args.metrics
    if metrics_path is None:
        metrics_path = args.product.with_name("spaxlet_matched_metrics.json")
    report = json.loads(metrics_path.read_text())
    truth_path = args.truth if args.truth is not None else Path(report["truth"])
    with fits.open(truth_path) as hdul:
        table = hdul["TRUTH_SPECTRA"].data
        truth_spectra = (
            np.asarray(table["galaxy_1_spectrum"], dtype=float),
            np.asarray(table["galaxy_2_spectrum"], dtype=float),
        )
        truth_morphologies = np.asarray(hdul["TRUTH_MORPHOLOGY"].data, dtype=float)
        truth_wavelength = np.asarray(table["wavelength_um"], dtype=float)

    with np.load(args.product, allow_pickle=False) as product:
        wavelength = np.asarray(product["wave"], dtype=float)
        spectra = (
            np.asarray(product["sed1"], dtype=float),
            np.asarray(product["sed2"], dtype=float),
        )
        morphologies = (
            np.asarray(product["morph1"], dtype=float),
            np.asarray(product["morph2"], dtype=float),
        )
        collapsed_residual = np.asarray(
            product["collapsed_whitened_residual"], dtype=float
        )
        spectral_envelopes = (
            (
                np.asarray(product["structural_sed1_lower"], dtype=float),
                np.asarray(product["structural_sed1_upper"], dtype=float),
            ),
            (
                np.asarray(product["structural_sed2_lower"], dtype=float),
                np.asarray(product["structural_sed2_upper"], dtype=float),
            ),
        )

    np.testing.assert_allclose(wavelength, truth_wavelength, rtol=0, atol=1e-10)
    if len(report["sources"]) != 2:
        raise ValueError("collaborator report must contain exactly two sources")
    morphology_reference_offset = np.asarray(
        report.get(
            "morphology_reference_offset_yx",
            report.get("removed_shift_median_px", (0.0, 0.0)),
        ),
        dtype=float,
    )
    truth_morphologies = np.asarray(
        [
            translate_morphology(morphology, morphology_reference_offset)
            for morphology in truth_morphologies
        ]
    )
    morphology_scores = tuple(
        morphology_metrics(value, reference)
        for value, reference in zip(morphologies, truth_morphologies)
    )
    return {
        "report": report,
        "wavelength": wavelength,
        "spectra": spectra,
        "morphologies": morphologies,
        "collapsed_residual": collapsed_residual,
        "spectral_envelopes": spectral_envelopes,
        "truth_spectra": truth_spectra,
        "truth_morphologies": truth_morphologies,
        "morphology_scores": morphology_scores,
    }


def _plot_spectra(values, output, dpi):
    figure, axes = plt.subplots(
        3, 2, figsize=(12, 9.5), sharex="col", constrained_layout=True
    )
    for source in range(2):
        score = values["report"]["sources"][source]["spectrum"]
        truth = values["truth_spectra"][source]
        fitted = values["spectra"][source]
        lower, upper = values["spectral_envelopes"][source]
        axes[0, source].fill_between(
            values["wavelength"],
            lower,
            upper,
            color="#56B4E9",
            alpha=0.35,
            label="exact structural envelope",
        )
        axes[0, source].plot(
            values["wavelength"],
            truth,
            color="black",
            lw=1.2,
            label="truth",
        )
        axes[0, source].plot(
            values["wavelength"],
            fitted,
            color="#0072B2",
            lw=0.9,
            label="Scarlet",
        )
        axes[0, source].set_title(
            "Galaxy {}: spectrum rel-L2 {:.2f}%".format(
                source + 1, 100 * score["relative_l2"]
            )
        )
        axes[0, source].set_ylabel("integrated source flux")
        axes[0, source].legend(frameon=False, fontsize=8)
        axes[1, source].axhline(0, color="black", lw=0.7)
        axes[1, source].plot(
            values["wavelength"],
            fitted - truth,
            color="#0072B2",
            lw=0.7,
        )
        axes[1, source].set_ylabel("Scarlet − truth\n(source flux)")
        axes[1, source].grid(alpha=0.2)
        axes[2, source].axhline(0, color="black", lw=0.7)
        axes[2, source].plot(
            score["binned_wavelength"],
            100 * np.asarray(score["binned_fractional_error"]),
            color="#D55E00",
            marker="o",
            ms=2.5,
        )
        axes[2, source].set_xlabel("wavelength (µm)")
        axes[2, source].set_ylabel("24-bin (Scarlet − truth) / truth (%)")
        axes[2, source].grid(alpha=0.2)
    figure.suptitle(
        "Scarlet collaborator recovery; envelope is structural, not ±1σ"
    )
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def _plot_morphologies(values, output, dpi):
    figure, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    for source in range(2):
        truth = _normalized(values["truth_morphologies"][source])
        fitted = _normalized(values["morphologies"][source])
        residual = 100 * (fitted - truth) / float(np.max(truth))
        limit = max(float(np.max(np.abs(residual))), np.finfo(float).eps)
        axes[source, 0].imshow(np.sqrt(truth), origin="lower", cmap="magma")
        axes[source, 1].imshow(np.sqrt(fitted), origin="lower", cmap="magma")
        image = axes[source, 2].imshow(
            residual,
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        score = values["morphology_scores"][source]
        axes[source, 0].set_ylabel("Galaxy {}".format(source + 1))
        axes[source, 1].set_title(
            "rel-L2 {:.2f}%, centroid {:.3f}px".format(
                100 * score["relative_l2"], score["centroid_error_px"]
            )
        )
        figure.colorbar(
            image,
            ax=axes[source, 2],
            shrink=0.8,
            label="(fit − truth) / truth peak (%)",
        )
        for axis in axes[source]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("truth in latent PSF frame")
    axes[0, 2].set_title("unit-flux residual")
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def _plot_residual_and_metrics(values, output, dpi):
    report = values["report"]
    residual = report["residual"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    limit = max(
        float(np.percentile(np.abs(values["collapsed_residual"]), 99.5)),
        np.finfo(float).eps,
    )
    image = axes[0].imshow(
        values["collapsed_residual"],
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axes[0].set_title("collapsed whitened data − model")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    figure.colorbar(image, ax=axes[0], shrink=0.8)

    rows = [
        ("start", report["start"]),
        ("iterations", report["iterations"]),
        ("runtime", "{:.2f} s".format(report["runtime_seconds"])),
        ("χ² / voxel", "{:.5f}".format(residual["chi_square_per_voxel"])),
        ("power entropy H", "{:.5f}".format(residual["power_spectral_entropy"])),
        ("|lag-1 ρ|", "{:.5f}".format(residual["lag1_autocorrelation"])),
        (
            "projected gradient",
            "{:.3g}".format(report["parameter_relative_projected_gradient"]),
        ),
        (
            "optimizer",
            report.get("optimizer", report["optimizer_scheme"]),
        ),
        ("dtype / chunk", "{} / {}".format(report["fit_dtype"], report["channel_chunk_size"])),
    ]
    axes[1].axis("off")
    table = axes[1].table(
        cellText=rows,
        colLabels=("diagnostic", "value"),
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    axes[1].set_title("Truth-independent fit diagnostics")
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main():
    args = _parser().parse_args()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values = _load(args)
    _plot_spectra(
        values, args.output_dir / "spaxlet_collaborator_spectra.png", args.dpi
    )
    _plot_morphologies(
        values, args.output_dir / "spaxlet_collaborator_morphologies.png", args.dpi
    )
    _plot_residual_and_metrics(
        values, args.output_dir / "spaxlet_collaborator_residual.png", args.dpi
    )
    print("wrote {}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()

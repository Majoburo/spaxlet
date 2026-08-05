"""Plot spatial residuals and extracted spectra from the SPT0311 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, type=Path)
    parser.add_argument("--spectra", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-model-output",
        type=Path,
        help="optional gallery of every latent morphology and extracted spectrum",
    )
    parser.add_argument(
        "--spectra-only-output",
        type=Path,
        help="optional unsmoothed, spectra-only gallery (SVG is recommended)",
    )
    parser.add_argument(
        "--sources",
        default="lens,E,W,C1,C2,C3",
        help="comma-separated spectra to plot when present",
    )
    return parser


def _limits(image):
    finite = np.asarray(image)[np.isfinite(image)]
    if not finite.size:
        return 0, 1
    lower, upper = np.percentile(finite, (2, 99.5))
    if lower == upper:
        upper = lower + 1
    return lower, upper


def _morphology_crop(morphology):
    threshold = np.nanmax(morphology) * 1e-8
    rows, columns = np.where(morphology > threshold)
    if not rows.size:
        return morphology, (0, 0)
    y0, y1 = max(int(rows.min()) - 1, 0), min(int(rows.max()) + 2, morphology.shape[0])
    x0, x1 = max(int(columns.min()) - 1, 0), min(
        int(columns.max()) + 2, morphology.shape[1]
    )
    return morphology[y0:y1, x0:x1], (y0, x0)


def plot_source_models(product_path, table, output_path):
    """Plot each fixed-gauge latent morphology next to its model spectrum."""
    with np.load(product_path) as product:
        names = tuple(str(name) for name in product["names"])
        morphologies = np.asarray(product["morphologies"], dtype=float)
        centroids = np.asarray(product["fitted_centroids_yx"], dtype=float)
    wavelength = np.asarray(table["wavelength_um"], dtype=float)
    columns = 4
    rows = int(np.ceil(len(names) / columns))
    figure = plt.figure(figsize=(18, 3.2 * rows), constrained_layout=True)
    outer = figure.add_gridspec(rows, columns)
    for index, (name, morphology, centroid) in enumerate(
        zip(names, morphologies, centroids)
    ):
        row, column = divmod(index, columns)
        inner = outer[row, column].subgridspec(1, 2, width_ratios=(0.85, 1.65))
        morphology_axis = figure.add_subplot(inner[0, 0])
        spectrum_axis = figure.add_subplot(inner[0, 1])
        cropped, origin = _morphology_crop(morphology)
        morphology_axis.imshow(cropped, origin="lower", cmap="magma")
        morphology_axis.plot(
            centroid[1] - origin[1],
            centroid[0] - origin[0],
            marker="+",
            ms=7,
            mew=1,
            color="cyan",
        )
        morphology_axis.set_title(name, fontsize=11, fontweight="bold")
        morphology_axis.set_xticks([])
        morphology_axis.set_yticks([])
        morphology_axis.set_xlabel("unit-flux morphology", fontsize=7)

        flux = np.asarray(table[name + "_flux_jy"], dtype=float) * 1e6
        smooth = gaussian_filter1d(flux, 1.5)
        spectrum_axis.plot(wavelength, flux, color="0.65", alpha=0.45, lw=0.5)
        spectrum_axis.plot(wavelength, smooth, color="tab:blue", lw=1.0)
        spectrum_axis.grid(alpha=0.18)
        spectrum_axis.tick_params(labelsize=7)
        spectrum_axis.set_xlabel("wavelength [µm]", fontsize=7)
        spectrum_axis.set_ylabel("flux [µJy]", fontsize=7)
    for index in range(len(names), rows * columns):
        row, column = divmod(index, columns)
        axis = figure.add_subplot(outer[row, column])
        axis.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_native_spectra(product_path, table, output_path):
    """Plot every extracted spectral channel without smoothing or binning."""
    with np.load(product_path) as product:
        names = tuple(str(name) for name in product["names"])
    wavelength = np.asarray(table["wavelength_um"], dtype=float)
    columns = 4
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(24, 4.2 * rows),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, name in zip(axes.flat, names):
        flux = np.asarray(table[name + "_flux_jy"], dtype=float) * 1e6
        axis.step(wavelength, flux, where="mid", color="tab:blue", lw=0.7)
        axis.set_title(name, fontsize=14, fontweight="bold")
        axis.set_ylabel("flux [µJy]")
        axis.set_ylim(bottom=min(0, float(np.nanmin(flux))))
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("observed wavelength [µm]")
    for axis in axes.flat[len(names) :]:
        axis.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main():
    args = _parser().parse_args()
    with np.load(args.product) as product:
        model = np.asarray(product["model"], dtype=float)
        residual = np.asarray(product["residual"], dtype=float)
        valid = np.asarray(product["valid_mask"], dtype=bool)
        collapsed_whitened = np.asarray(
            product["collapsed_whitened_residual"], dtype=float
        )
        centers = np.asarray(product["catalog_centers_yx"], dtype=float)
        names = tuple(str(name) for name in product["names"])
    data = model + residual
    data_rms = float(np.sqrt(np.mean(data[valid] ** 2)))
    model_rms = float(np.sqrt(np.mean(model[valid] ** 2)))
    residual_rms = float(np.sqrt(np.mean(residual[valid] ** 2)))
    residual_fraction = residual_rms / max(data_rms, np.finfo(float).tiny)
    report_path = args.product.with_name("spt0311_deblend_report.json")
    chi_square_per_voxel = np.nan
    if report_path.exists():
        report = json.loads(report_path.read_text())
        chi_square_per_voxel = report["fit"]["chi_square_per_valid_voxel"]
    images = []
    for cube in (data, model, residual):
        masked = np.where(valid, cube, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            images.append(np.nanmedian(masked, axis=0))
    images.append(collapsed_whitened)

    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(1, 0.85))
    titles = (
        "data\nRMS={:.3g} MJy/sr".format(data_rms),
        "Scarlet model\nRMS={:.3g} MJy/sr".format(model_rms),
        "residual\nRMS={:.3g} ({:.1%} of data)".format(
            residual_rms, residual_fraction
        ),
        "collapsed whitened residual\n$\\chi^2/N_{{valid}}$={:.3f}".format(
            chi_square_per_voxel
        ),
    )
    color_maps = ("coolwarm", "coolwarm", "coolwarm", "coolwarm")
    comparison_pixels = np.concatenate(
        [images[0].ravel(), images[1].ravel()]
    )
    common_limit = float(np.nanpercentile(np.abs(comparison_pixels), 99.5))
    for index, (image, title, color_map) in enumerate(zip(images, titles, color_maps)):
        axis = figure.add_subplot(grid[0, index])
        if index < 3:
            lower, upper = -common_limit, common_limit
        else:
            absolute = np.nanpercentile(np.abs(image), 99)
            lower, upper = -absolute, absolute
        rendered = axis.imshow(
            image, origin="lower", cmap=color_map, vmin=lower, vmax=upper
        )
        colorbar = figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.025)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label(
            "median MJy/sr" if index < 3 else "summed whitened residual",
            fontsize=7,
        )
        axis.set_title(title)
        axis.set_xlabel("x [spaxel]")
        if index == 0:
            axis.set_ylabel("y [spaxel]")
            for name, center in zip(names, centers):
                axis.text(center[1], center[0], name, color="cyan", fontsize=6)

    table = Table.read(args.spectra)
    wavelength = np.asarray(table["wavelength_um"], dtype=float)
    requested = tuple(name.strip() for name in args.sources.split(",") if name.strip())
    selected = [name for name in requested if name + "_flux_jy" in table.colnames]
    axis = figure.add_subplot(grid[1, :])
    for offset, name in enumerate(selected):
        flux = np.asarray(table[name + "_flux_jy"], dtype=float) * 1e6
        smoothed = gaussian_filter1d(flux, 1.5)
        scale = np.nanpercentile(np.abs(smoothed), 99)
        scale = max(float(scale), np.finfo(float).tiny)
        axis.plot(wavelength, smoothed / scale + offset, lw=1, label=name)
    axis.set_xlabel("observed wavelength [µm]")
    axis.set_ylabel("normalized flux + offset")
    axis.legend(ncol=max(len(selected), 1), loc="upper center", frameon=False)
    axis.grid(alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    if args.source_model_output is not None:
        plot_source_models(args.product, table, args.source_model_output)
    if args.spectra_only_output is not None:
        plot_native_spectra(args.product, table, args.spectra_only_output)


if __name__ == "__main__":
    main()

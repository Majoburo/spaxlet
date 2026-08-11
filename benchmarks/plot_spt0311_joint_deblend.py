"""Plot auditable PRISM+G395H source models and residual diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from astropy.table import Table
from matplotlib.ticker import MaxNLocator


CONTOUR_FRACTIONS = (0.2, 0.5, 0.8)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, type=Path)
    parser.add_argument("--spectra", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _collapsed(cube, valid):
    masked = np.where(valid, cube, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(masked, axis=0)


def _morphology_crop(morphology):
    selected = morphology > np.nanmax(morphology) * 1e-8
    rows, columns = np.where(selected)
    if not rows.size:
        return morphology, (0, 0)
    y0, y1 = max(int(rows.min()) - 1, 0), min(int(rows.max()) + 2, morphology.shape[0])
    x0, x1 = max(int(columns.min()) - 1, 0), min(
        int(columns.max()) + 2, morphology.shape[1]
    )
    return morphology[y0:y1, x0:x1], (y0, x0)


def _contour_levels(morphology):
    """Return positive morphology levels as fractions of the fitted peak."""

    peak = float(np.nanmax(morphology))
    if not np.isfinite(peak) or peak <= 0:
        return np.asarray([], dtype=float)
    return peak * np.asarray(CONTOUR_FRACTIONS)


def _source_colors(count):
    color_map = plt.get_cmap("tab20")
    return [color_map(index % color_map.N) for index in range(count)]


def _overlay_source_contours(axis, names, morphologies, centroids, colors, labels=False):
    """Overlay intrinsic, pre-PSF component shapes in latent-grid coordinates."""

    for name, morphology, centroid, color in zip(
        names, morphologies, centroids, colors
    ):
        levels = _contour_levels(morphology)
        if not levels.size:
            continue
        contours = axis.contour(
            morphology,
            levels=levels,
            colors=[color],
            linewidths=(0.55, 0.8, 1.05),
            alpha=0.9,
        )
        contours.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=1.75, foreground="black", alpha=0.55
                ),
                path_effects.Normal(),
            ]
        )
        if labels:
            annotation = axis.annotate(
                name,
                xy=(centroid[1], centroid[0]),
                xytext=(2, 2),
                textcoords="offset points",
                color=color,
                fontsize=6.5,
                fontweight="bold",
            )
            annotation.set_path_effects(
                [path_effects.Stroke(linewidth=1.8, foreground="black"), path_effects.Normal()]
            )


def _format_spatial_grid(axis, shape):
    axis.set_xlim(-0.5, shape[1] - 0.5)
    axis.set_ylim(-0.5, shape[0] - 0.5)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    axis.grid(color="white", linewidth=0.35, alpha=0.18)
    axis.tick_params(labelsize=7)


def plot_residual_arm(product, report, label, path):
    model = np.asarray(product[label + "_model"], dtype=float)
    residual = np.asarray(product[label + "_residual"], dtype=float)
    valid = np.asarray(product[label + "_valid_mask"], dtype=bool)
    whitened = np.asarray(
        product[label + "_collapsed_whitened_residual"], dtype=float
    )
    data = model + residual
    names = tuple(str(name) for name in product["names"])
    morphologies = np.asarray(product["morphologies"], dtype=float)
    centroids = np.asarray(product["fitted_centroids_yx"], dtype=float)
    colors = _source_colors(len(names))
    data_rms = float(np.sqrt(np.mean(data[valid] ** 2)))
    model_rms = float(np.sqrt(np.mean(model[valid] ** 2)))
    residual_rms = float(np.sqrt(np.mean(residual[valid] ** 2)))
    fraction = residual_rms / max(data_rms, np.finfo(float).tiny)
    images = (
        _collapsed(data, valid),
        _collapsed(model, valid),
        _collapsed(residual, valid),
        whitened,
    )
    common_limit = float(
        np.nanpercentile(np.abs(np.concatenate([image.ravel() for image in images[:2]])), 99.5)
    )
    titles = (
        "data\nRMS={:.4g} MJy/sr".format(data_rms),
        "joint model\nRMS={:.4g} MJy/sr".format(model_rms),
        "data − model\nRMS={:.4g}; {:.1%} of data".format(residual_rms, fraction),
        "whitened residual\nχ²/N={:.4f}".format(
            report["observations"][label]["chi_square_per_valid_voxel"]
        ),
    )
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.3), constrained_layout=True)
    for index, (axis, image, title) in enumerate(zip(axes, images, titles)):
        if index < 3:
            limit = common_limit
            units = "median MJy/sr; common scale"
        else:
            limit = float(np.nanpercentile(np.abs(image), 99))
            units = "summed whitened residual"
        limit = max(limit, np.finfo(float).tiny)
        rendered = axis.imshow(
            image, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit
        )
        colorbar = figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.025)
        colorbar.set_label(units, fontsize=8)
        _overlay_source_contours(
            axis,
            names,
            morphologies,
            centroids,
            colors,
            labels=index == 0,
        )
        _format_spatial_grid(axis, image.shape)
        axis.set_title(title)
        axis.set_xlabel("x [spaxel]")
        axis.set_ylabel("y [spaxel]")
    figure.suptitle(
        "{} residual audit — {}\n"
        "colored contours: intrinsic morphology at {}% of peak; "
        "fixed variance=(JWST ERR × empirical scale)²".format(
            label.upper(),
            report["model"],
            "/".join(str(int(100 * value)) for value in CONTOUR_FRACTIONS),
        )
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_sources(product, spectra_path, report, path):
    names = tuple(str(name) for name in product["names"])
    morphologies = np.asarray(product["morphologies"], dtype=float)
    centers = np.asarray(product["common_latent_centers_yx"], dtype=float)
    centroids = np.asarray(product["fitted_centroids_yx"], dtype=float)
    prism = Table.read(spectra_path, hdu="PRISM")
    g395h = Table.read(spectra_path, hdu="G395H")
    columns = 3
    rows = int(np.ceil(len(names) / columns))
    figure = plt.figure(figsize=(18, 3.1 * rows), constrained_layout=True)
    outer = figure.add_gridspec(rows, columns)
    for index, (name, morphology, center, centroid) in enumerate(
        zip(names, morphologies, centers, centroids)
    ):
        row, column = divmod(index, columns)
        inner = outer[row, column].subgridspec(1, 2, width_ratios=(0.8, 2.0))
        image_axis = figure.add_subplot(inner[0, 0])
        spectrum_axis = figure.add_subplot(inner[0, 1])
        cropped, origin = _morphology_crop(morphology)
        extent = (
            origin[1] - 0.5,
            origin[1] + cropped.shape[1] - 0.5,
            origin[0] - 0.5,
            origin[0] + cropped.shape[0] - 0.5,
        )
        image_axis.imshow(cropped, origin="lower", cmap="magma", extent=extent)
        levels = _contour_levels(cropped)
        if levels.size:
            y_grid = np.arange(cropped.shape[0]) + origin[0]
            x_grid = np.arange(cropped.shape[1]) + origin[1]
            image_axis.contour(
                x_grid,
                y_grid,
                cropped,
                levels=levels,
                colors="white",
                linewidths=(0.55, 0.8, 1.05),
            )
        image_axis.plot(
            center[1],
            center[0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="lime",
            ms=5,
            mew=0.9,
        )
        image_axis.plot(
            centroid[1],
            centroid[0],
            marker="+",
            color="cyan",
            ms=7,
        )
        image_axis.set_title(
            "{}  x={:.2f}, y={:.2f}".format(name, centroid[1], centroid[0]),
            fontweight="bold",
            fontsize=8,
        )
        image_axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        image_axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        image_axis.tick_params(labelsize=6)
        image_axis.grid(color="white", linewidth=0.35, alpha=0.2)
        image_axis.set_xlabel("x [latent spaxel]", fontsize=7)
        image_axis.set_ylabel("y [latent spaxel]", fontsize=7)
        for table, label, color, width in (
            (prism, "PRISM response", "tab:orange", 1.0),
            (g395h, "G395H / latent overlap", "tab:blue", 0.55),
        ):
            spectrum_axis.step(
                np.asarray(table["wavelength_um"]),
                np.asarray(table[name + "_flux_jy"]) * 1e6,
                where="mid",
                color=color,
                lw=width,
                label=label,
            )
        spectrum_axis.set_xlabel("observed wavelength [µm]", fontsize=7)
        spectrum_axis.set_ylabel("flux [µJy]", fontsize=7)
        spectrum_axis.tick_params(labelsize=7)
        spectrum_axis.grid(alpha=0.2)
        if index == 0:
            spectrum_axis.legend(fontsize=7, frameon=False)
    for index in range(len(names), rows * columns):
        row, column = divmod(index, columns)
        axis = figure.add_subplot(outer[row, column])
        axis.set_axis_off()
    figure.suptitle(
        report["model"]
        + "\nwhite contours: 20/50/80% of peak; green circle: registered center; cyan +: fitted centroid"
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    args = _parser().parse_args()
    report = json.loads(
        args.product.with_name("spt0311_joint_report.json").read_text()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.product) as product:
        for label in ("prism", "g395h"):
            plot_residual_arm(
                product,
                report,
                label,
                args.output_dir / "spt0311_joint_{}_residual.png".format(label),
            )
        plot_sources(
            product,
            args.spectra,
            report,
            args.output_dir / "spt0311_joint_source_models.png",
        )


if __name__ == "__main__":
    main()

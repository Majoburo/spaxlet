"""Figures for the Scarlet point-source reproduction on the planet cube.

The galaxy analogue, ``plot_collaborator_reproduction.py``, cannot be reused:
it reads a ``collapsed_whitened_residual`` and the structural mixing envelopes,
and neither exists here.  The cube is noiseless, so there is nothing to whiten
against, and with both morphologies pinned at the PSF the model is linear in the
spectra, so there is no free-mixing degeneracy to bound.  The figures kept are
the ones that still carry information: truth-versus-recovered spectra with a
fractional-error row, recovered morphologies against the PSF, and observable
data-minus-model maps.

Run ``run_planet_reproduction.py`` first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from benchmarks.run_planet_reproduction import corrected_kernels

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 11,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

COLORS = ["#D55E00", "#0072B2"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cube", default="majo_planet_cube.fits")
    parser.add_argument("--psf", default="nirspec_ifu_PRISM_CLEAR_allwave.cube.fits")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kernel-size", type=int, default=47)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _stamp(kernel: np.ndarray, center, shape) -> np.ndarray:
    ny, nx = shape
    size = kernel.shape[0]
    canvas = np.zeros((ny, nx))
    y0 = int(round(center[0])) - size // 2
    x0 = int(round(center[1])) - size // 2
    ys = slice(max(y0, 0), min(y0 + size, ny))
    xs = slice(max(x0, 0), min(x0 + size, nx))
    canvas[ys, xs] = kernel[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0]
    return canvas


def main() -> None:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.report or args.product.with_name("scarlet_planet_report.json")
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    with np.load(args.product, allow_pickle=False) as product:
        wave = np.asarray(product["wave"], dtype=float)
        spectra = [np.asarray(product["sed1"], float), np.asarray(product["sed2"], float)]
        truth = [np.asarray(product["t1"], float), np.asarray(product["t2"], float)]
        morphologies = [
            np.asarray(product["morph1"], float), np.asarray(product["morph2"], float)
        ]
        names = [str(name) for name in product["names"]]

    with fits.open(args.data_root / args.cube) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        sources_table = hdul["SOURCES"].data
        centers = [
            (float(y), float(x)) for y, x in zip(sources_table["y"], sources_table["x"])
        ]

    # --- spectra ---------------------------------------------------------------
    figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col", constrained_layout=True)
    for source, name in enumerate(names):
        true_spectrum, value = truth[source], spectra[source]
        good = true_spectrum > 0
        relative = np.linalg.norm(value[good] - true_spectrum[good]) / np.linalg.norm(
            true_spectrum[good]
        )
        flux = value[good].sum() / true_spectrum[good].sum() - 1.0
        top = axes[0, source]
        top.plot(wave, true_spectrum, "k--", lw=1.4, label="truth")
        top.plot(wave, value, color=COLORS[source], lw=0.9,
                 label=f"scarlet  L2={100*relative:.2f}%, flux={100*flux:+.3f}%")
        top.set(yscale="log", ylabel="flux (arbitrary)", title=name)
        top.legend(frameon=False, fontsize=9)

        # A fractional error means nothing where the source has no flux: below
        # ~0.7 um the brown dwarf truth is ~1e-4 of its peak.
        meaningful = true_spectrum > 1e-4 * true_spectrum.max()
        fractional = np.full_like(value, np.nan)
        fractional[meaningful] = (
            100.0 * (value[meaningful] - true_spectrum[meaningful])
            / true_spectrum[meaningful]
        )
        bottom = axes[1, source]
        bottom.axhline(0.0, color="0.6", lw=0.8)
        bottom.plot(wave, fractional, color=COLORS[source], lw=0.8)
        span = np.nanpercentile(np.abs(fractional), 99.5)
        bottom.set(xlabel="wavelength (um)", ylabel="fractional error (%)",
                   ylim=(-1.5 * span, 1.5 * span))
        excluded = int((~meaningful).sum())
        if excluded:
            bottom.set_title(f"{excluded} channels below 1e-4 of peak excluded",
                             fontsize=8, color="0.4")
    figure.savefig(args.output_dir / "scarlet_planet_spectra.png", dpi=args.dpi)
    plt.close(figure)

    # --- morphologies against the PSF -----------------------------------------
    kernels = corrected_kernels(args.data_root / args.psf, args.kernel_size)
    reference_index = len(wave) // 2
    figure, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4.6),
                                constrained_layout=True)
    axes = np.atleast_1d(axes)
    for source, name in enumerate(names):
        recovered = morphologies[source]
        total = recovered.sum()
        normalized = recovered / total if total > 0 else recovered
        handle = axes[source].imshow(normalized, origin="lower", cmap="inferno")
        figure.colorbar(handle, ax=axes[source], fraction=0.046)
        axes[source].set_title(f"{name} recovered morphology", fontsize=10)
    figure.savefig(args.output_dir / "scarlet_planet_morphologies.png", dpi=args.dpi)
    plt.close(figure)

    # --- observable data-minus-model ------------------------------------------
    probe = [len(wave) // 6, len(wave) // 2, 5 * len(wave) // 6]
    figure, axes = plt.subplots(len(probe), 3, figsize=(11, 3.4 * len(probe)),
                               constrained_layout=True)
    for row, index in enumerate(probe):
        design = np.stack(
            [_stamp(kernels[index], center, data.shape[1:]).ravel() for center in centers],
            axis=1,
        )
        model = (design @ np.array([spectra[0][index], spectra[1][index]])).reshape(
            data.shape[1:]
        )
        observed = data[index]
        floor = observed.max() * 1e-8
        panels = (
            (np.log10(np.maximum(observed, floor)), "data", {"cmap": "inferno"}),
            (np.log10(np.maximum(model, floor)), "scarlet model", {"cmap": "inferno"}),
            (100.0 * (observed - model) / observed.max(), "residual (% of peak)",
             {"cmap": "RdBu_r", "vmin": -0.5, "vmax": 0.5}),
        )
        for column, (image, title, kwargs) in enumerate(panels):
            handle = axes[row, column].imshow(image, origin="lower", **kwargs)
            figure.colorbar(handle, ax=axes[row, column], fraction=0.046)
            axes[row, column].set_title(f"{wave[index]:.2f} um  {title}", fontsize=9)
    figure.savefig(args.output_dir / "scarlet_planet_residual.png", dpi=args.dpi)
    plt.close(figure)

    if report:
        print(json.dumps(report, indent=2))
    print(f"wrote 3 figures to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Recover the spectra in the noiseless blended binary-star simulation.

The science wavelengths fall halfway between the wavelength planes in the
STPSF cube (except for the final science channel, which is beyond the final
PSF plane and therefore uses that endpoint).  The simulator also used the
even-sized oversampled PSF at a phase that corresponds to a two-cell roll in
the detector-y direction and no roll in detector-x.  After 4x4 binning, the
odd 47x47 crop is centered on detector index 24.

The effective source coordinates in the rendered cube are the catalogued
half-pixel coordinates.  In the latent delta-pixel frame these correspond to
integer pixels (23, 23) and (23, 28); the calibrated PSF phase supplies the
remaining subpixel offset.  With those spatial factors fixed, the spectral
problem is linear and separable by wavelength.  This driver uses Scarlet's
matched renderer to construct the two templates and solves each two-column
least-squares problem exactly instead of iterating a nonlinear optimizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import spaxlet
from astropy.io import fits


OVERSAMPLE = 4
PHASE_ROLL_CELLS = (2, 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", required=True, type=Path)
    parser.add_argument("--psf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kernel-size", type=int, default=47)
    return parser


def _psf_wavelengths(header, count: int) -> np.ndarray:
    """Read STPSF wavelength cards and return microns."""
    return np.asarray(
        [float(header[f"WVLN{index:04d}"]) * 1e6 for index in range(count)]
    )


def corrected_kernels(
    psf_path: Path, wavelengths_um: np.ndarray, kernel_size: int
) -> np.ndarray:
    """Interpolate, phase, detector-bin, crop, and normalize STPSF kernels."""
    with fits.open(psf_path, memmap=True) as handle:
        source = handle["OVERSAMP"]
        psf_wave = _psf_wavelengths(source.header, source.data.shape[0])

        upper = np.searchsorted(psf_wave, wavelengths_um, side="right")
        upper = np.clip(upper, 1, len(psf_wave) - 1)
        lower = upper - 1
        below = wavelengths_um <= psf_wave[0]
        above = wavelengths_um >= psf_wave[-1]
        lower[below] = upper[below] = 0
        lower[above] = upper[above] = len(psf_wave) - 1

        denominator = psf_wave[upper] - psf_wave[lower]
        fraction = np.divide(
            wavelengths_um - psf_wave[lower],
            denominator,
            out=np.zeros_like(wavelengths_um, dtype=float),
            where=denominator != 0,
        )
        low = np.asarray(source.data[lower], dtype=float)
        if np.array_equal(lower, upper):
            oversampled = low
        else:
            high = np.asarray(source.data[upper], dtype=float)
            oversampled = low + fraction[:, None, None] * (high - low)

    oversampled = np.roll(oversampled, PHASE_ROLL_CELLS, axis=(1, 2))
    channels, size, _ = oversampled.shape
    detector_size = size // OVERSAMPLE
    detector = oversampled.reshape(
        channels, detector_size, OVERSAMPLE, detector_size, OVERSAMPLE
    ).sum(axis=(2, 4))

    if kernel_size % 2 == 0 or kernel_size > detector_size - 1:
        raise ValueError("kernel size must be odd and no larger than 47")
    center = detector_size // 2
    half = kernel_size // 2
    kernels = detector[
        :, center - half : center + half + 1, center - half : center + half + 1
    ]
    kernels /= kernels.sum(axis=(1, 2), keepdims=True)
    return kernels


def _half_up_centers(catalog) -> list[tuple[int, int]]:
    """Map effective half-pixel coordinates to the phased latent pixel grid."""
    return [
        tuple(
            int(value)
            for value in np.floor(np.asarray(center, dtype=float) + 0.5)
        )
        for center in catalog
    ]


def _render_templates(images, kernels, centers):
    channels = [f"ch{index:04d}" for index in range(images.shape[0])]
    frame = spaxlet.Frame(
        images.shape,
        psf=spaxlet.DeltaPSF(images.shape[0], dtype=images.dtype),
        channels=channels,
    )
    observation = spaxlet.Observation(
        images,
        psf=spaxlet.ImagePSF(kernels),
        weights=np.ones_like(images),
        channels=channels,
    ).match(frame)

    templates = []
    morphologies = []
    for row, column in centers:
        morphology = np.zeros(images.shape[-2:], dtype=images.dtype)
        morphology[row, column] = 1
        latent = np.broadcast_to(morphology, images.shape).copy()
        templates.append(np.asarray(observation.render(latent), dtype=float))
        morphologies.append(morphology)
    return np.asarray(templates), np.asarray(morphologies)


def _solve_spectra(images, templates):
    data = images.reshape(images.shape[0], -1)
    first = templates[0].reshape(images.shape[0], -1)
    second = templates[1].reshape(images.shape[0], -1)
    g00 = np.sum(first * first, axis=1)
    g01 = np.sum(first * second, axis=1)
    g11 = np.sum(second * second, axis=1)
    rhs0 = np.sum(first * data, axis=1)
    rhs1 = np.sum(second * data, axis=1)
    determinant = g00 * g11 - g01**2
    spectra = np.asarray(
        [
            (rhs0 * g11 - rhs1 * g01) / determinant,
            (rhs1 * g00 - rhs0 * g01) / determinant,
        ]
    )
    correlation = g01 / np.sqrt(g00 * g11)
    condition = np.sqrt((1 + correlation) / (1 - correlation))
    return spectra, correlation, condition


def _relative_l2(value, reference) -> float:
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def main() -> None:
    args = _parser().parse_args()
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(args.cube, memmap=True) as handle:
        images = np.asarray(handle[0].data, dtype=float)
        sources = handle["SOURCES"].data
        truth_table = handle["TRUTH_SPECTRA"].data
        names = [str(name).strip() for name in sources["name"]]
        catalog = [
            (float(row), float(column))
            for row, column in zip(sources["y"], sources["x"])
        ]
        wavelengths = np.asarray(truth_table["wavelength_um"], dtype=float)
        truth = np.asarray(
            [
                np.asarray(truth_table[f"{name}_spectrum"], dtype=float)
                for name in names
            ]
        )

    if len(names) != 2:
        raise ValueError(f"expected two sources, found {len(names)}")
    if images.shape[0] != wavelengths.size:
        raise ValueError("science cube and truth table have different channel counts")

    centers = _half_up_centers(catalog)
    kernels = corrected_kernels(args.psf, wavelengths, args.kernel_size)
    templates, morphologies = _render_templates(images, kernels, centers)
    spectra, correlation, condition = _solve_spectra(images, templates)
    if np.any(spectra < 0):
        raise ArithmeticError("unconstrained point-source solution contains negative flux")

    model = np.sum(spectra[:, :, None, None] * templates, axis=0)
    truth_model = np.sum(truth[:, :, None, None] * templates, axis=0)
    residual = images - model

    report = {
        "cube": str(args.cube.resolve()),
        "psf": str(args.psf.resolve()),
        "shape": list(images.shape),
        "catalog_centers": [list(center) for center in catalog],
        "latent_centers": [list(center) for center in centers],
        "phase_roll_oversampled_cells_yx": list(PHASE_ROLL_CELLS),
        "psf_wavelength_interpolation": "linear with endpoint clamping",
        "negative_data_pixels": int(np.count_nonzero(images < 0)),
        "template_correlation_min": float(correlation.min()),
        "template_correlation_max": float(correlation.max()),
        "design_condition_min": float(condition.min()),
        "design_condition_max": float(condition.max()),
        "truth_forward_relative_l2": _relative_l2(truth_model, images),
        "fit_relative_l2": _relative_l2(model, images),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "residual_max_abs": float(np.max(np.abs(residual))),
        "runtime_seconds": float(time.perf_counter() - start),
        "sources": {},
    }
    for name, recovered, reference in zip(names, spectra, truth):
        meaningful = reference > 1e-4 * reference.max()
        report["sources"][name] = {
            "relative_l2": _relative_l2(recovered, reference),
            "integrated_flux_ratio": float(
                np.trapz(recovered, wavelengths)
                / np.trapz(reference, wavelengths)
            ),
            "median_abs_fractional_error": float(
                np.median(np.abs((recovered[meaningful] - reference[meaningful])
                                 / reference[meaningful]))
            ),
        }

    product_path = args.output_dir / "binary_star_recovery.npz"
    report_path = args.output_dir / "binary_star_report.json"
    np.savez_compressed(
        product_path,
        sed1=spectra[0],
        sed2=spectra[1],
        t1=truth[0],
        t2=truth[1],
        morph1=morphologies[0],
        morph2=morphologies[1],
        wave=wavelengths,
        names=np.asarray(names),
        model=model,
        residual=residual,
        template_correlation=correlation,
        design_condition=condition,
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {product_path}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()

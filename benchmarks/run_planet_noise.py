"""Noise sensitivity of the planet-cube separation, normalized to HD 19467 B.

``majo_planet_cube.fits`` is noiseless, so the 0.3 percent recovery it gives is a
forward-model correctness check, not a sensitivity result.  This driver injects
noise and measures what survives.

**Normalization.**  The cube emulates HD 19467 B, whose published NIRSpec
extraction reaches S/N ~ 10 *per resolution element* over 2.9-5.2 um.  A
resolution element spans ``lambda / (R * dlambda)`` channels of this cube, so the
per-channel S/N that target implies is lower than 10 by that factor's square
root.  The noise scale is solved to hit the target rather than asserted.

**Noise model.**  Variance proportional to signal, i.e. shot-noise-like, plus a
floor.  A constant sigma would understate the thing this cube exists to test: in
real high-contrast data it is the bright star's PSF wings landing on the
companion that limit the companion's S/N, and that term scales with the local
signal.

**Two estimators are run on identical noise realizations**, because with a
signal-dependent variance they are not the same thing:

``gls``
    Generalized least squares, weights ``1 / (alpha * m + floor)``.  Linear and
    exact, but ``sum (d-m)^2 / sigma^2(m)`` is *not* a log-likelihood when sigma
    depends on m -- it is missing the ``sum log sigma^2(m)`` term.

``likelihood``
    The full Gaussian negative log-likelihood including the log-determinant,
    minimized under non-negativity.  Dropping the log-det term rewards inflating
    the model, since a larger model buys a larger variance that "explains"
    residuals for free, so the gap between these two estimators *is* that bias.

**Caveats on reading these against the paper.**  The cube is 5000:1, while
HD 19467 B is 1e-5 to 1e-4 -- 2 to 20 times harder.  The cube is PRISM/CLEAR
while the published spectrum is R ~ 2700.  And the variance here is built from
the true model, which is an oracle: on real data it must be iterated from the
fitted model, as ``fit_final.py`` does in the two-galaxy benchmark, and that is
strictly worse than what is measured here.

Run from the repository root::

    PYTHONPATH=. venv-scarlet/bin/python -m benchmarks.run_planet_noise \
        --data-root .../jwst/collab --output-dir .../benchmark_artifacts/planet/noise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import minimize

from benchmarks.run_planet_reproduction import corrected_kernels

BAND = (2.9, 5.2)  # um, the published extraction range


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cube", default="majo_planet_cube.fits")
    parser.add_argument("--psf", default="nirspec_ifu_PRISM_CLEAR_allwave.cube.fits")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kernel-size", type=int, default=47)
    parser.add_argument("--target-snr", type=float, default=10.0)
    parser.add_argument(
        "--resolving-power", type=float, default=100.0,
        help="nominal PRISM/CLEAR R used to size a resolution element; the cube "
             "ships no R(lambda), so this is an explicit approximation",
    )
    parser.add_argument("--realizations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--variance-floor", type=float, default=1e-3)
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


def _neg_log_likelihood(coefficients, design, observed, alpha, floor):
    """Gaussian NLL with variance alpha*model+floor, including the log-det term."""
    model = design @ coefficients
    variance = alpha * np.maximum(model, 0.0) + floor
    residual = observed - model
    value = float(np.sum(residual**2 / variance + np.log(variance)))
    # d/dc_k of both terms; the log-det is what the plain GLS objective omits.
    active = (model > 0.0).astype(float)
    weight = (
        -2.0 * residual / variance
        + alpha * active * (1.0 / variance - residual**2 / variance**2)
    )
    return value, design.T @ weight


def _fit_likelihood(design, observed, alpha, floor, start):
    result = minimize(
        _neg_log_likelihood, start, args=(design, observed, alpha, floor),
        jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * design.shape[1],
    )
    return result.x


def main() -> None:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(args.data_root / args.cube) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        sources_table = hdul["SOURCES"].data
        table = hdul["TRUTH_SPECTRA"].data
        names = [str(name).strip() for name in sources_table["name"]]
        centers = [(float(y), float(x)) for y, x in zip(sources_table["y"], sources_table["x"])]
        wave = np.asarray(table["wavelength_um"], dtype=float)

    kernels = corrected_kernels(args.data_root / args.psf, args.kernel_size)
    n_wave, ny, nx = data.shape
    in_band = (wave >= BAND[0]) & (wave <= BAND[1])

    # Channels per resolution element: lambda / (R * dlambda).
    channel_width = float(np.median(np.diff(wave)))
    channels_per_element = wave / (args.resolving_power * channel_width)
    print(f"cube {data.shape}; {int(in_band.sum())} channels in {BAND[0]}-{BAND[1]} um",
          flush=True)
    print(f"resolution element spans {channels_per_element[in_band].min():.1f}-"
          f"{channels_per_element[in_band].max():.1f} channels at R={args.resolving_power:g}",
          flush=True)

    designs = np.empty((n_wave, ny * nx, 2))
    for index in range(n_wave):
        designs[index] = np.stack(
            [_stamp(kernels[index], center, (ny, nx)).ravel() for center in centers], axis=1
        )

    latent = np.empty((n_wave, 2))
    for index in range(n_wave):
        design = designs[index]
        latent[index] = np.linalg.solve(design.T @ design, design.T @ data[index].ravel())
    model = np.einsum("lpk,lk->lp", designs, latent)

    # var = alpha*model + floor makes the companion covariance scale as alpha, so
    # solve alpha in closed form from a unit-alpha pass instead of searching.
    unit_variance = np.maximum(model, 0.0) + args.variance_floor
    companion_variance_unit = np.empty(n_wave)
    for index in range(n_wave):
        design = designs[index]
        companion_variance_unit[index] = np.linalg.inv(
            design.T @ (design / unit_variance[index][:, None])
        )[1, 1]
    per_channel_unit = latent[:, 1] / np.sqrt(companion_variance_unit)
    per_element_unit = per_channel_unit * np.sqrt(channels_per_element)
    alpha = (float(np.median(per_element_unit[in_band])) / args.target_snr) ** 2

    variance = alpha * np.maximum(model, 0.0) + args.variance_floor
    per_channel = per_channel_unit / np.sqrt(alpha)
    per_element = per_element_unit / np.sqrt(alpha)
    print(f"alpha {alpha:.6g}; in-band companion S/N median "
          f"{np.median(per_element[in_band]):.2f} per resolution element, "
          f"{np.median(per_channel[in_band]):.2f} per channel", flush=True)

    rng = np.random.default_rng(args.seed)
    estimates = {
        "gls": np.empty((args.realizations, n_wave, 2)),
        "likelihood": np.empty((args.realizations, n_wave, 2)),
    }
    for realization in range(args.realizations):
        noisy = model + rng.normal(0.0, np.sqrt(variance))
        for index in range(n_wave):
            design = designs[index]
            observed = noisy[index]
            weighted = design / variance[index][:, None]
            gls = np.linalg.solve(design.T @ weighted, weighted.T @ observed)
            estimates["gls"][realization, index] = gls
            estimates["likelihood"][realization, index] = _fit_likelihood(
                design, observed, alpha, args.variance_floor, np.maximum(gls, 0.0)
            )
        print(f"  realization {realization + 1}/{args.realizations}", flush=True)

    report = {
        "band_um": list(BAND),
        "target_snr_per_resolution_element": args.target_snr,
        "resolving_power": args.resolving_power,
        "alpha": alpha,
        "realizations": args.realizations,
        "median_in_band_snr_per_resolution_element": float(np.median(per_element[in_band])),
        "median_in_band_snr_per_channel": float(np.median(per_channel[in_band])),
        "estimators": {},
    }
    for estimator, values in estimates.items():
        report["estimators"][estimator] = {}
        print(f"\n{estimator}:", flush=True)
        for column, name in enumerate(names):
            true_spectrum = latent[:, column]
            sample = values[:, :, column]
            bias = np.mean(sample, axis=0) - true_spectrum
            scatter = np.std(sample, axis=0)
            band_true = true_spectrum[in_band]
            scores = {
                "in_band_relative_bias": float(
                    np.linalg.norm(bias[in_band]) / np.linalg.norm(band_true)
                ),
                "in_band_relative_scatter": float(
                    np.mean(scatter[in_band]) / np.mean(band_true)
                ),
                "in_band_flux_bias": float(bias[in_band].sum() / band_true.sum()),
            }
            report["estimators"][estimator][name] = scores
            print(f"  {name:12s} bias {scores['in_band_relative_bias']:.4e}"
                  f"   scatter {scores['in_band_relative_scatter']:.4e}"
                  f"   flux bias {scores['in_band_flux_bias']:+.4e}", flush=True)

    np.savez_compressed(
        args.output_dir / "planet_noise_recovery.npz",
        wave=wave, latent=latent, in_band=in_band,
        gls=estimates["gls"], likelihood=estimates["likelihood"],
        snr_per_element=per_element, snr_per_channel=per_channel,
        variance_alpha=alpha, names=np.array(names),
    )
    (args.output_dir / "planet_noise_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote products to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

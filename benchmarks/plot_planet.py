"""Figures for both planet-cube arms, noiseless reproduction and noise injection.

One script for both, selected from the product's own keys: a noise product
carries the per-realization ``gls``/``likelihood`` estimates, a reproduction
product carries a single ``sed1``/``sed2`` pair.

``plot_collaborator_reproduction.py`` is deliberately not reused.  Beyond the
arrays it reads ``report["start"]``, ``["runtime_seconds"]``,
``["parameter_relative_projected_gradient"]``, ``["optimizer_scheme"]`` and the
structural mixing envelopes for its provenance panel.  A fixed-morphology linear
fit has no optimizer, no declared start and no mixing degeneracy, so satisfying
that schema would mean inventing values for a provenance table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 11,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

COLORS = {"source": ["#D55E00", "#0072B2"], "gls": "#D55E00", "likelihood": "#0072B2"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _fractional(value: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Percent error, blanked where the source has no flux to speak of."""
    meaningful = truth > 1e-4 * truth.max()
    result = np.full_like(value, np.nan)
    result[meaningful] = 100.0 * (value[meaningful] - truth[meaningful]) / truth[meaningful]
    return result


def _plot_reproduction(product, report, output_dir, dpi):
    wave = np.asarray(product["wave"], float)
    names = [str(name) for name in product["names"]]
    spectra = [np.asarray(product["sed1"], float), np.asarray(product["sed2"], float)]
    truth = [np.asarray(product["t1"], float), np.asarray(product["t2"], float)]

    figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col", constrained_layout=True)
    for source, name in enumerate(names):
        good = truth[source] > 0
        relative = np.linalg.norm(spectra[source][good] - truth[source][good]) / np.linalg.norm(
            truth[source][good]
        )
        flux = spectra[source][good].sum() / truth[source][good].sum() - 1.0
        axes[0, source].plot(wave, truth[source], "k--", lw=1.4, label="truth")
        axes[0, source].plot(wave, spectra[source], color=COLORS["source"][source], lw=0.9,
                             label=f"recovered  L2={100*relative:.2f}%, flux={100*flux:+.3f}%")
        axes[0, source].set(yscale="log", ylabel="flux (arbitrary)", title=name)
        axes[0, source].legend(frameon=False, fontsize=9)

        fractional = _fractional(spectra[source], truth[source])
        axes[1, source].axhline(0.0, color="0.6", lw=0.8)
        axes[1, source].plot(wave, fractional, color=COLORS["source"][source], lw=0.8)
        span = np.nanpercentile(np.abs(fractional), 99.5)
        axes[1, source].set(xlabel="wavelength (um)", ylabel="fractional error (%)",
                            ylim=(-1.5 * span, 1.5 * span))
    figure.savefig(output_dir / "planet_spectra.png", dpi=dpi)
    plt.close(figure)
    return 1


def _plot_noise(product, report, output_dir, dpi):
    wave = np.asarray(product["wave"], float)
    latent = np.asarray(product["latent"], float)
    in_band = np.asarray(product["in_band"], bool)
    names = [str(name) for name in product["names"]]
    estimates = {name: np.asarray(product[name], float) for name in ("gls", "likelihood")}
    band, companion = wave[in_band], 1
    truth = latent[:, companion]

    figure, axes = plt.subplots(len(estimates) + 1, 1, figsize=(11, 3.6 * len(estimates) + 3.4),
                                sharex=True, constrained_layout=True)
    for axis, (estimator, values) in zip(axes, estimates.items()):
        sample = values[:, :, companion]
        mean, scatter = sample.mean(axis=0), sample.std(axis=0)
        axis.plot(band, truth[in_band], "k--", lw=1.4, label="truth")
        axis.fill_between(band, (mean - scatter)[in_band], (mean + scatter)[in_band],
                          color=COLORS[estimator], alpha=0.3, lw=0, label="+/- 1 sigma")
        axis.plot(band, mean[in_band], color=COLORS[estimator], lw=1.0, label=f"{estimator} mean")
        scores = report["estimators"][estimator][names[companion]]
        axis.set(ylabel="flux (arbitrary)",
                 title=f"{names[companion]} -- {estimator}: "
                       f"bias {scores['in_band_relative_bias']:.1e}, "
                       f"scatter {scores['in_band_relative_scatter']:.1e}")
        axis.legend(frameon=False, fontsize=9)

    for estimator, values in estimates.items():
        sample = values[:, :, companion]
        axes[-1].plot(band, (100.0 * sample.std(axis=0) / truth)[in_band],
                      color=COLORS[estimator], lw=0.8, label=estimator)
    axes[-1].plot(band, (100.0 / np.asarray(product["snr_per_channel"], float))[in_band],
                  color="0.4", ls=":", lw=1.0, label="100 / (S/N per channel)")
    axes[-1].set(xlabel="wavelength (um)", ylabel="fractional scatter (%)", yscale="log",
                 title=f"median in-band S/N "
                       f"{report['median_in_band_snr_per_resolution_element']:.2f} per "
                       f"resolution element = "
                       f"{report['median_in_band_snr_per_channel']:.2f} per channel")
    axes[-1].legend(frameon=False, fontsize=9)
    figure.savefig(output_dir / "planet_noise_companion.png", dpi=dpi)
    plt.close(figure)
    return 1


def main() -> None:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    default_report = (
        "planet_noise_report.json" if "noise" in args.product.name
        else "scarlet_planet_report.json"
    )
    report_path = args.report or args.product.with_name(default_report)
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    with np.load(args.product, allow_pickle=False) as product:
        plot = _plot_noise if "gls" in product else _plot_reproduction
        written = plot(product, report, args.output_dir, args.dpi)
    print(f"wrote {written} figure to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Run Scarlet against the corrected lisasep collaborator-fit contract.

Unlike the historical Scarlet comparison, the model frame uses a per-channel
delta PSF. The observation renderer therefore applies the declared corrected
instrument PSF directly instead of matching it from an arbitrary Gaussian
model PSF. Raw factorized components receive the same all-one spectra and
catalog-centered A/B/C morphology starts as lisasep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
import scarlet
from astropy.io import fits

from benchmarks.ifu_parity_metrics import morphology_metrics, residual_metrics, spectral_metrics
from lisasep import crop_psf_kernels, recenter_psf_kernels


READ_VARIANCE = 0.000880653
POISSON_COEFFICIENT = 0.00405419
CATALOG_CENTERS = ((21.296, 25.784), (23.250, 24.480))
START_SCALES = {"A": (3.0, 3.0), "B": (5.0, 2.0), "C": (2.0, 5.0)}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", required=True, choices=tuple(START_SCALES))
    parser.add_argument("--kernel-size", type=int, default=47)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--relative-tolerance", type=float, default=1e-11)
    return parser


def _centroid(value):
    value = np.maximum(np.asarray(value, dtype=float), 0.0)
    value /= value.sum()
    rows, columns = np.indices(value.shape, dtype=float)
    return np.asarray([np.sum(rows * value), np.sum(columns * value)])


def _catalog_order(morphologies):
    centers = [_centroid(value) for value in morphologies]
    direct = sum(
        np.linalg.norm(center - catalog)
        for center, catalog in zip(centers, CATALOG_CENTERS)
    )
    swapped = sum(
        np.linalg.norm(center - catalog)
        for center, catalog in zip(centers[::-1], CATALOG_CENTERS)
    )
    return (1, 0) if swapped < direct else (0, 1)


def _jsonable(metrics):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
    }


def main():
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth_path = args.data_root / "morphology_galaxy_cube_004.fits"
    psf_path = args.data_root / "nirspec_ifu_PRISM_CLEAR_allwave.cube.fits"
    with fits.open(truth_path) as hdul:
        data = np.asarray(hdul["SCI"].data, dtype=float)
        table = hdul["TRUTH_SPECTRA"].data
        wavelength = np.asarray(table["wavelength_um"], dtype=float)
        reference_spectra = (
            np.asarray(table["galaxy_1_spectrum"], dtype=float),
            np.asarray(table["galaxy_2_spectrum"], dtype=float),
        )
        reference_morphologies = np.asarray(
            hdul["TRUTH_MORPHOLOGY"].data, dtype=float
        )
    with fits.open(psf_path) as hdul:
        kernels = np.asarray(hdul["DET_SAMP"].data, dtype=float)
    if kernels.shape[0] != data.shape[0]:
        raise ValueError("PSF and science cube must share the spectral grid")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernels, retained_flux = crop_psf_kernels(kernels, args.kernel_size)
    kernels, removed_shift = recenter_psf_kernels(kernels)

    variance = READ_VARIANCE + POISSON_COEFFICIENT * np.maximum(data, 0.0)
    channels = ["ch{:04d}".format(index) for index in range(data.shape[0])]
    delta_psf = scarlet.DeltaPSF(data.shape[0])
    frame = scarlet.Frame(data.shape, psf=delta_psf, channels=channels)
    observation = scarlet.Observation(
        data,
        psf=scarlet.ImagePSF(kernels),
        weights=1.0 / variance,
        channels=channels,
    ).match(frame)

    rows, columns = np.indices(data.shape[1:], dtype=float)

    def blob(center, scale):
        value = np.exp(-np.hypot(rows - center[0], columns - center[1]) / scale)
        return value / value.sum()

    scales = START_SCALES[args.start]
    starting_morphologies = [
        blob(center, scale) for center, scale in zip(CATALOG_CENTERS, scales)
    ]
    sources = []
    for morphology_start in starting_morphologies:
        spectrum = scarlet.TabulatedSpectrum(frame, np.ones(data.shape[0]))
        morphology = scarlet.ImageMorphology(
            frame, morphology_start.copy(), resizing=False
        )
        sources.append(scarlet.FactorizedComponent(frame, spectrum, morphology))

    initial_model = np.asarray(
        observation.render(scarlet.Blend(sources, observation).get_model()),
        dtype=float,
    )
    initial_chi_square = float(np.mean((data - initial_model) ** 2 / variance))
    blend = scarlet.Blend(sources, observation)
    started = time.perf_counter()
    iterations, log_likelihood = blend.fit(
        args.max_iter,
        e_rel=args.relative_tolerance,
        project_initial=True,
    )
    runtime = time.perf_counter() - started

    spectra = []
    morphologies = []
    for source in sources:
        morphology = np.maximum(
            np.asarray(source.morphology.get_model(), dtype=float), 0.0
        )
        scale = float(morphology.sum())
        spectra.append(np.asarray(source.spectrum.get_model(), dtype=float) * scale)
        morphologies.append(morphology / scale)
    order = _catalog_order(morphologies)
    spectra = [spectra[index] for index in order]
    morphologies = [morphologies[index] for index in order]

    model = np.asarray(observation.render(blend.get_model()), dtype=float)
    residual = data - model
    residual_score = residual_metrics(residual, 1.0 / variance)
    source_scores = []
    for spectrum, reference_spectrum, morphology, reference_morphology in zip(
        spectra,
        reference_spectra,
        morphologies,
        reference_morphologies,
    ):
        source_scores.append(
            {
                "spectrum": _jsonable(
                    spectral_metrics(
                        spectrum,
                        reference_spectrum,
                        wavelength,
                        n_bin=24,
                    )
                ),
                "morphology": _jsonable(
                    morphology_metrics(morphology, reference_morphology)
                ),
            }
        )

    history = np.asarray(blend.log_likelihood, dtype=float)
    relative_change = (
        abs(float(history[-1] - history[-2])) / max(abs(float(history[-1])), 1.0)
        if history.size > 1
        else float("nan")
    )
    report = {
        "start": args.start,
        "start_scales": list(scales),
        "catalog_order": list(order),
        "runtime_seconds": runtime,
        "iterations": int(iterations),
        "max_iter": args.max_iter,
        "converged_before_cap": int(iterations) < args.max_iter,
        "log_likelihood": float(log_likelihood),
        "final_relative_objective_change": relative_change,
        "initial_projection_relative_l2": float(
            blend.initial_projection_relative_l2
        ),
        "initial_chi_square_per_voxel": initial_chi_square,
        "residual": residual_score,
        "sources": source_scores,
        "truth": str(truth_path.resolve()),
        "psf": str(psf_path.resolve()),
        "model_frame_psf": "per-channel 1x1 delta",
        "observation_psf": "crop_then_recenter",
        "kernel_size": args.kernel_size,
        "retained_flux_min": float(np.min(retained_flux)),
        "retained_flux_max": float(np.max(retained_flux)),
        "removed_shift_median_px": np.median(removed_shift, axis=0).tolist(),
        "variance": {
            "kind": "fixed_data_based_read_plus_poisson",
            "read_variance": READ_VARIANCE,
            "poisson_coefficient": POISSON_COEFFICIENT,
        },
    }
    output = args.output_dir / "scarlet_matched_start{}.npz".format(args.start)
    np.savez_compressed(
        output,
        sed1=spectra[0],
        sed2=spectra[1],
        morph1=morphologies[0],
        morph2=morphologies[1],
        wave=wavelength,
        iterations=int(iterations),
        runtime_seconds=runtime,
        collapsed_whitened_residual=np.sum(
            residual / np.sqrt(variance), axis=0
        )
        / np.sqrt(data.shape[0]),
        psf_centering=np.asarray("crop_then_recenter"),
        model_frame_psf=np.asarray("per-channel_1x1_delta"),
        kernel_size=args.kernel_size,
    )
    (args.output_dir / "scarlet_matched_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "start {}: {:.2f}s logL={:.8g} chi2/voxel={:.4f} H={:.4f} "
        "|rho1|={:.4f}".format(
            args.start,
            runtime,
            log_likelihood,
            residual_score["chi_square_per_voxel"],
            residual_score["power_spectral_entropy"],
            residual_score["lag1_autocorrelation"],
        ),
        flush=True,
    )
    for index, score in enumerate(source_scores, 1):
        print(
            "  galaxy {}: spectrum={:.2f}% morphology={:.2f}% centroid={:.3f}px".format(
                index,
                100 * score["spectrum"]["relative_l2"],
                100 * score["morphology"]["relative_l2"],
                score["morphology"]["centroid_error_px"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

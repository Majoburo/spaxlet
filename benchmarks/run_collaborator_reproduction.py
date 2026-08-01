"""Run Scarlet against the corrected lisasep collaborator-fit contract.

Unlike the historical Scarlet comparison, the model frame uses a per-channel
delta PSF. The observation renderer therefore applies the declared corrected
instrument PSF directly instead of matching it from an arbitrary Gaussian
model PSF. Raw factorized components receive the same all-one spectra and
catalog-centered A/B/C morphology starts as lisasep.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
import subprocess
import time
import warnings

import numpy as np
import scarlet
from astropy import units as u
from astropy.io import fits

from benchmarks.ifu_parity_metrics import morphology_metrics, residual_metrics, spectral_metrics


READ_VARIANCE = 0.000880653
POISSON_COEFFICIENT = 0.00405419
CATALOG_CENTERS = ((21.296, 25.784), (23.250, 24.480))
START_SCALES = {"A": (3.0, 3.0), "B": (5.0, 2.0), "C": (2.0, 5.0)}
OPTIMIZER_SCHEMES = ("adam", "nadam", "adamx", "amsgrad", "padam", "radam")
FEATURES = ("positivity", "centroid")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", required=True, choices=tuple(START_SCALES))
    parser.add_argument("--kernel-size", type=int, default=47)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--relative-tolerance", type=float, default=1e-11)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--channel-chunk-size", type=int)
    parser.add_argument(
        "--optimizer-scheme", choices=OPTIMIZER_SCHEMES, default="amsgrad"
    )
    parser.add_argument(
        "--feature",
        choices=FEATURES,
        default="positivity",
        help="opt-in morphology constraint; positivity preserves the baseline",
    )
    parser.add_argument("--optimality-tolerance", type=float)
    parser.add_argument("--optimality-check-interval", type=int, default=100)
    parser.add_argument("--profile-memory", action="store_true")
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


def _memory_checkpoint(label, enabled):
    if not enabled:
        return
    gc.collect()
    current_kib = None
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            current_kib = int(line.split()[1])
            break
    peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(
        "MEMORY {} current={:.1f}MiB peak={:.1f}MiB".format(
            label, current_kib / 1024.0, peak_kib / 1024.0
        ),
        flush=True,
    )


def _repository_provenance():
    root = Path(__file__).resolve().parents[1]

    def git(*arguments):
        return subprocess.check_output(
            ("git", "-C", str(root), *arguments),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = git("rev-parse", "HEAD")
        branch = git("branch", "--show-current")
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit = None
        branch = None
        dirty = None
    return {
        "path": str(root),
        "version": scarlet.__version__,
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def _morphology_parameter(value, feature, center):
    image = np.asarray(value).copy()
    if feature == "positivity":
        return image
    if feature == "centroid":
        constraint = scarlet.DykstraConstraintChain(
            scarlet.CentroidConstraint(center),
            scarlet.PositivityConstraint(),
            max_iter=20000,
            rtol=1e-12,
            atol=1e-13,
        )
        return scarlet.Parameter(
            image,
            name="image",
            step=scarlet.parameter.relative_step,
            constraint=constraint,
        )
    raise ValueError("unknown morphology feature {!r}".format(feature))


def main():
    args = _parser().parse_args()
    if args.optimality_tolerance is not None and args.optimality_tolerance < 0:
        raise ValueError("optimality_tolerance must be non-negative")
    if args.optimality_check_interval <= 0:
        raise ValueError("optimality_check_interval must be positive")
    fit_dtype = np.dtype(args.dtype)
    _memory_checkpoint("imports", args.profile_memory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth_path = args.data_root / "morphology_galaxy_cube_004.fits"
    psf_path = args.data_root / "nirspec_ifu_PRISM_CLEAR_allwave.cube.fits"
    with fits.open(truth_path) as hdul:
        data = np.asarray(hdul["SCI"].data, dtype=fit_dtype)
        table = hdul["TRUTH_SPECTRA"].data
        wavelength = np.asarray(table["wavelength_um"], dtype=float)
        reference_spectra = (
            np.asarray(table["galaxy_1_spectrum"], dtype=float),
            np.asarray(table["galaxy_2_spectrum"], dtype=float),
        )
        reference_morphologies = np.asarray(
            hdul["TRUTH_MORPHOLOGY"].data, dtype=float
        )
    _memory_checkpoint("science", args.profile_memory)
    with fits.open(psf_path) as hdul:
        kernels = np.asarray(hdul["DET_SAMP"].data, dtype=fit_dtype)
    _memory_checkpoint("raw_psf", args.profile_memory)
    if kernels.shape[0] != data.shape[0]:
        raise ValueError("PSF and science cube must share the spectral grid")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernels, retained_flux = scarlet.crop_psf_kernels(kernels, args.kernel_size)
    kernels, removed_shift = scarlet.recenter_psf_kernels(kernels)
    kernels = np.asarray(kernels, dtype=fit_dtype)
    _memory_checkpoint("corrected_psf", args.profile_memory)

    variance = np.asarray(
        READ_VARIANCE + POISSON_COEFFICIENT * np.maximum(data, 0.0),
        dtype=fit_dtype,
    )
    physical_wavelengths = wavelength * u.um
    channels = ["ch{:04d}".format(index) for index in range(data.shape[0])]
    delta_psf = scarlet.DeltaPSF(data.shape[0], dtype=fit_dtype)
    frame = scarlet.Frame(
        data.shape,
        psf=delta_psf,
        channels=channels,
        wavelengths=physical_wavelengths,
    )
    observation = scarlet.Observation.from_ifu_arrays(
        data,
        physical_wavelengths,
        variance,
        psf=scarlet.ImagePSF(kernels),
        channels=channels,
        dtype=fit_dtype,
    ).match(frame)
    _memory_checkpoint("matched_observation", args.profile_memory)

    rows, columns = np.indices(data.shape[1:], dtype=fit_dtype)

    def blob(center, scale):
        value = np.exp(-np.hypot(rows - center[0], columns - center[1]) / scale)
        return np.asarray(value / value.sum(), dtype=fit_dtype)

    scales = START_SCALES[args.start]
    starting_morphologies = [
        blob(center, scale) for center, scale in zip(CATALOG_CENTERS, scales)
    ]
    sources = []
    for morphology_start, center in zip(starting_morphologies, CATALOG_CENTERS):
        spectrum = scarlet.TabulatedSpectrum(
            frame, np.ones(data.shape[0], dtype=fit_dtype)
        )
        morphology = scarlet.ImageMorphology(
            frame,
            _morphology_parameter(morphology_start, args.feature, center),
            resizing=False,
        )
        sources.append(scarlet.FactorizedComponent(frame, spectrum, morphology))
    _memory_checkpoint("sources", args.profile_memory)

    initial_model = np.asarray(
        observation.render(scarlet.Blend(sources, observation).get_model()),
        dtype=float,
    )
    initial_chi_square = float(np.mean((data - initial_model) ** 2 / variance))
    del initial_model
    _memory_checkpoint("initial_model", args.profile_memory)
    blend = scarlet.Blend(sources, observation)
    optimality_checks = []
    periodic_optimality_runtime = 0.0

    def check_optimality(*parameters, it=None):
        nonlocal periodic_optimality_runtime
        if (
            args.optimality_tolerance is None
            or it == 0
            or it % args.optimality_check_interval != 0
        ):
            return
        check_started = time.perf_counter()
        diagnostic = blend.parameter_optimization_diagnostics()
        periodic_optimality_runtime += time.perf_counter() - check_started
        optimality_checks.append(
            {
                "iterations": len(blend.log_likelihood),
                "parameter_relative_projected_gradient": (
                    diagnostic.relative_projected_gradient
                ),
                "spectral_relative_projected_gradient": (
                    diagnostic.spectral_relative_projected_gradient
                ),
                "morphology_relative_projected_gradient": (
                    diagnostic.morphology_relative_projected_gradient
                ),
            }
        )
        if (
            diagnostic.relative_projected_gradient
            <= args.optimality_tolerance
        ):
            raise StopIteration("proximal optimality tolerance reached")

    started = time.perf_counter()
    iterations, log_likelihood = blend.fit(
        args.max_iter,
        e_rel=(
            0.0
            if args.optimality_tolerance is not None
            else args.relative_tolerance
        ),
        project_initial=True,
        normalize_initial_factors=True,
        channel_chunk_size=args.channel_chunk_size,
        scheme=args.optimizer_scheme,
        callback=check_optimality,
    )
    runtime = time.perf_counter() - started
    _memory_checkpoint("fit", args.profile_memory)
    optimality_started = time.perf_counter()
    optimality = blend.parameter_optimization_diagnostics()
    optimality_runtime = (
        periodic_optimality_runtime + time.perf_counter() - optimality_started
    )
    optimality_converged = (
        args.optimality_tolerance is not None
        and optimality.relative_projected_gradient
        <= args.optimality_tolerance
    )
    _memory_checkpoint("optimality", args.profile_memory)

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
    ordered_sources = [sources[index] for index in order]
    mixing_intervals = scarlet.bilinear_mixing_intervals(ordered_sources)
    mixing_envelopes = scarlet.pairwise_mixing_envelopes(ordered_sources)

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
    _memory_checkpoint("scores", args.profile_memory)

    history = np.asarray(blend.log_likelihood, dtype=float)
    relative_change = (
        abs(float(history[-1] - history[-2])) / max(abs(float(history[-1])), 1.0)
        if history.size > 1
        else float("nan")
    )
    report = {
        "scarlet": _repository_provenance(),
        "feature": args.feature,
        "constraint_centers_yx": (
            [list(center) for center in CATALOG_CENTERS]
            if args.feature == "centroid"
            else None
        ),
        "start": args.start,
        "start_scales": list(scales),
        "catalog_order": list(order),
        "runtime_seconds": runtime,
        "iterations": int(iterations),
        "max_iter": args.max_iter,
        "converged_before_cap": int(iterations) < args.max_iter,
        "optimality_converged": optimality_converged,
        "optimality_tolerance": args.optimality_tolerance,
        "optimality_check_interval": args.optimality_check_interval,
        "optimality_checks": optimality_checks,
        "log_likelihood": float(log_likelihood),
        "final_relative_objective_change": relative_change,
        "initial_projection_relative_l2": float(
            blend.initial_projection_relative_l2
        ),
        "initial_normalization_relative_l2": float(
            blend.initial_normalization_relative_l2
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
        "ifu_ingestion": {
            "kind": "measured_variance_and_mask_safe_arrays",
            "wavelength_unit": "um",
            "wavelength_channels": int(wavelength.size),
            "wavelength_min": float(wavelength[0]),
            "wavelength_max": float(wavelength[-1]),
            "mask_summary": observation.ifu_mask_summary,
        },
        "fit_dtype": args.dtype,
        "channel_chunk_size": args.channel_chunk_size,
        "optimizer_scheme": args.optimizer_scheme,
        "optimality_runtime_seconds": optimality_runtime,
        "parameter_relative_projected_gradient": (
            optimality.relative_projected_gradient
        ),
        "spectral_relative_projected_gradient": (
            optimality.spectral_relative_projected_gradient
        ),
        "morphology_relative_projected_gradient": (
            optimality.morphology_relative_projected_gradient
        ),
        "structural_mixing": {
            "interpretation": (
                "exact pairwise sensitivity floor; not posterior or +/-1 sigma"
            ),
            "intervals": [
                {
                    "donor": interval.donor,
                    "receiver": interval.receiver,
                    "delta_min": interval.delta_min,
                    "delta_max": interval.delta_max,
                }
                for interval in mixing_intervals
            ],
            "components": [
                {
                    "component": envelope.component,
                    "total_flux_min": envelope.total_flux_min,
                    "total_flux_max": envelope.total_flux_max,
                    "total_flux_min_fraction": (
                        envelope.total_flux_min
                        / max(float(np.sum(spectra[index])), np.finfo(float).tiny)
                    ),
                    "total_flux_max_fraction": (
                        envelope.total_flux_max
                        / max(float(np.sum(spectra[index])), np.finfo(float).tiny)
                    ),
                }
                for index, envelope in enumerate(mixing_envelopes)
            ],
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
        feature=np.asarray(args.feature),
        kernel_size=args.kernel_size,
        structural_sed1_lower=mixing_envelopes[0].spectrum_lower,
        structural_sed1_upper=mixing_envelopes[0].spectrum_upper,
        structural_sed2_lower=mixing_envelopes[1].spectrum_lower,
        structural_sed2_upper=mixing_envelopes[1].spectrum_upper,
        structural_morph1_lower=mixing_envelopes[0].morphology_lower,
        structural_morph1_upper=mixing_envelopes[0].morphology_upper,
        structural_morph2_lower=mixing_envelopes[1].morphology_lower,
        structural_morph2_upper=mixing_envelopes[1].morphology_upper,
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

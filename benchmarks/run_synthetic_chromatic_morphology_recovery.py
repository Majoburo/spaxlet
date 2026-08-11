"""Compare rank-one and continuum-plus-line models on chromatic source truth."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

import spaxlet
from benchmarks.many_source_recovery_contract import (
    RECOVERY_SEED,
    RECOVERY_SHAPE,
    RECOVERY_SOURCE_SPECS,
    chromatic_line_morphologies,
    chromatic_noisy_cube,
    chromatic_source_cubes,
    recovery_morphologies,
    recovery_psfs,
    recovery_spectral_components,
    recovery_spectra,
)
from benchmarks.run_synthetic_many_source_recovery import (
    START_WIDTH_SCALE,
    _centroid,
    _initial_image,
    _line_observables,
)


MODELS = ("rank1", "continuum_line")
SPECTRAL_MODES = ("free", "oracle")


@dataclass(frozen=True)
class ChromaticRecoveryMetrics:
    model_kind: str
    spectral_mode: str
    optimizer: str
    start: str
    seed: int
    iterations: int
    train_chi2_per_voxel: float
    heldout_chi2_per_voxel: float
    relative_projected_gradient: float
    spectrum_relative_l2: np.ndarray
    integrated_flux_relative_error: np.ndarray
    line_flux_relative_error: np.ndarray
    line_peak_relative_error: np.ndarray
    source_cube_relative_l2: np.ndarray
    continuum_morphology_relative_l2: np.ndarray
    line_morphology_relative_l2: np.ndarray
    line_centroid_error_px: np.ndarray
    fitted_spectra: np.ndarray
    fitted_continuum_morphologies: np.ndarray
    fitted_line_morphologies: np.ndarray
    data: np.ndarray
    model: np.ndarray
    residual: np.ndarray
    variance: np.ndarray
    train_valid: np.ndarray
    heldout_valid: np.ndarray


def _local_truth(image, spec):
    half = spec.support // 2
    origin = (spec.center[0] - half, spec.center[1] - half)
    return (
        image[
            origin[0] : origin[0] + spec.support,
            origin[1] : origin[1] + spec.support,
        ],
        origin,
    )


def _centroid_constraint(image, center):
    return spaxlet.DykstraConstraintChain(
        spaxlet.CentroidConstraint(center),
        spaxlet.PositivityConstraint(),
        max_iter=20000,
        rtol=1e-12,
        atol=1e-13,
    )


def _component(
    frame,
    spectrum,
    image,
    box,
    constraint,
    spectral_constraint,
    *,
    fixed_spectrum=False,
):
    morphology = spaxlet.ImageMorphology(
        frame,
        spaxlet.Parameter(
            image,
            name="image",
            step=spaxlet.parameter.relative_step,
            constraint=constraint,
        ),
        bbox=box,
        resizing=False,
    )
    if fixed_spectrum:
        spectral = spaxlet.TabulatedSpectrum(
            frame,
            spaxlet.Parameter(
                np.asarray(spectrum, dtype=float),
                name="spectrum",
                fixed=True,
            ),
        )
    else:
        spectral = spaxlet.TabulatedSpectrum(
            frame, spectrum, constraint=spectral_constraint
        )
    return spaxlet.FactorizedComponent(frame, spectral, morphology)


def _sources(frame, data, model_kind, spectral_mode, start, spectral_strength):
    rows, columns = np.indices(RECOVERY_SHAPE[1:], dtype=float)
    truth_continuum = recovery_morphologies()
    truth_continuum_spectra, truth_line_spectra = recovery_spectral_components()
    sources = []
    source_groups = []
    for spec, truth_image, true_continuum_spectrum, true_line_spectrum in zip(
        RECOVERY_SOURCE_SPECS,
        truth_continuum,
        truth_continuum_spectra,
        truth_line_spectra,
    ):
        local_truth, origin = _local_truth(truth_image, spec)
        local_center = _centroid(local_truth)
        box = spaxlet.Box((spec.support, spec.support), origin=origin)
        aperture = np.hypot(
            rows - spec.center[0], columns - spec.center[1]
        ) <= max(2.0, 0.35 * spec.support)
        aperture_spectrum = np.maximum(np.sum(data[:, aperture], axis=1), 1e-10)
        smooth = (
            spaxlet.SpectralSmoothnessConstraint(
                np.arange(RECOVERY_SHAPE[0], dtype=float),
                spectral_strength,
                reference_scale=max(float(np.mean(aperture_spectrum)), 1e-20),
            )
            if spectral_strength > 0
            else None
        )
        continuum_image = _initial_image(spec, START_WIDTH_SCALE[start])
        continuum_spectrum_start = (
            true_continuum_spectrum + true_line_spectrum
            if spectral_mode == "oracle" and model_kind == "rank1"
            else (
                true_continuum_spectrum
                if spectral_mode == "oracle"
                else aperture_spectrum
            )
        )
        continuum = _component(
            frame,
            continuum_spectrum_start,
            continuum_image,
            box,
            _centroid_constraint(continuum_image, local_center),
            smooth,
            fixed_spectrum=spectral_mode == "oracle",
        )
        if model_kind == "rank1":
            source_groups.append((len(sources),))
            sources.append(continuum)
            continue

        continuum_start = np.maximum(
            gaussian_filter1d(aperture_spectrum, sigma=4.0, mode="nearest"),
            1e-10,
        )
        if spectral_mode == "free":
            continuum.spectrum.parameters[0][...] = continuum_start
        line_support = np.zeros(RECOVERY_SHAPE[0], dtype=bool)
        channel = np.arange(RECOVERY_SHAPE[0])
        for center in spec.line_channels:
            line_support |= np.abs(channel - center) <= 6
        line_start = (
            true_line_spectrum
            if spectral_mode == "oracle"
            else np.where(
                line_support,
                np.maximum(aperture_spectrum - continuum_start, 1e-10),
                0.0,
            )
        )
        line_image = _initial_image(spec, 0.72 * START_WIDTH_SCALE[start])
        line = _component(
            frame,
            line_start,
            line_image,
            box,
            spaxlet.PositivityConstraint(),
            spaxlet.SpectralSupportConstraint(line_support),
            fixed_spectrum=spectral_mode == "oracle",
        )
        source_groups.append((len(sources), len(sources) + 1))
        sources.extend((continuum, line))
    return sources, source_groups


def _full_morphology(source):
    factor = spaxlet.measure.factorization(source)
    image = np.zeros(RECOVERY_SHAPE[1:], dtype=float)
    y0, x0 = source.morphology.bbox.origin
    height, width = source.morphology.bbox.shape
    image[y0 : y0 + height, x0 : x0 + width] = factor.morphology
    return factor.spectrum, image


def fit_chromatic_cube(
    *,
    model_kind="rank1",
    spectral_mode="oracle",
    start="A",
    seed=RECOVERY_SEED,
    max_iter=300,
    optimizer="adaprox",
    spectral_smoothness_strength=0,
):
    """Fit one chromatic truth realization with a one- or two-morphology model."""

    if model_kind not in MODELS:
        raise ValueError("model_kind must be rank1 or continuum_line")
    if spectral_mode not in SPECTRAL_MODES:
        raise ValueError("spectral_mode must be free or oracle")
    if start not in START_WIDTH_SCALE:
        raise ValueError("start must be A, B, or C")
    if optimizer not in ("variable_projection", "adaprox"):
        raise ValueError("optimizer must be variable_projection or adaprox")
    if spectral_smoothness_strength < 0:
        raise ValueError("spectral smoothness strength must be non-negative")
    if optimizer == "variable_projection" and spectral_smoothness_strength > 0:
        raise ValueError("spectral smoothness requires the matched adaprox arm")
    if spectral_mode == "oracle" and optimizer != "adaprox":
        raise ValueError("fixed oracle spectra require adaprox")
    data, variance, valid, _, _ = chromatic_noisy_cube(seed)
    channel, rows, columns = np.indices(RECOVERY_SHAPE)
    heldout = valid & ((channel + 2 * rows + 3 * columns) % 11 == 0)
    train = valid & ~heldout
    weights = np.zeros_like(variance)
    weights[train] = 1.0 / variance[train]
    channels = tuple("ch{:03d}".format(index) for index in range(RECOVERY_SHAPE[0]))
    frame = spaxlet.Frame(
        RECOVERY_SHAPE,
        channels=channels,
        psf=spaxlet.DeltaPSF(RECOVERY_SHAPE[0]),
    )
    observation = spaxlet.Observation(
        data,
        channels=channels,
        psf=spaxlet.ImagePSF(recovery_psfs()),
        weights=weights,
    ).match(frame)
    sources, groups = _sources(
        frame,
        data,
        model_kind,
        spectral_mode,
        start,
        spectral_smoothness_strength,
    )
    blend = spaxlet.Blend(sources, observation)
    optimizer_arguments = (
        {"projected_max_backtracks": 20}
        if optimizer == "variable_projection"
        else {"scheme": "amsgrad", "channel_chunk_size": 32}
    )
    iterations, _ = blend.fit(
        max_iter,
        optimizer=optimizer,
        e_rel=0,
        project_initial=True,
        normalize_initial_factors=True,
        **optimizer_arguments,
    )

    component_factors = [_full_morphology(source) for source in sources]
    source_spectra = []
    continuum_morphologies = []
    line_morphologies = []
    source_cubes = []
    for group in groups:
        factors = [component_factors[index] for index in group]
        source_spectra.append(sum(spectrum for spectrum, _ in factors))
        continuum_morphologies.append(factors[0][1])
        line_morphologies.append(factors[-1][1])
        source_cubes.append(
            sum(
                spectrum[:, None, None] * morphology
                for spectrum, morphology in factors
            )
        )
    source_spectra = np.asarray(source_spectra)
    continuum_morphologies = np.asarray(continuum_morphologies)
    line_morphologies = np.asarray(line_morphologies)
    source_cubes = np.asarray(source_cubes)

    truth_spectra = recovery_spectra()
    truth_continuum = recovery_morphologies()
    truth_line = chromatic_line_morphologies()
    truth_source_cubes = chromatic_source_cubes()
    spectrum_error = np.linalg.norm(
        source_spectra - truth_spectra, axis=1
    ) / np.linalg.norm(truth_spectra, axis=1)
    flux_error = np.abs(
        np.sum(source_spectra, axis=1) - np.sum(truth_spectra, axis=1)
    ) / np.sum(truth_spectra, axis=1)
    fitted_line_flux, fitted_line_peak = _line_observables(source_spectra)
    truth_line_flux, truth_line_peak = _line_observables(truth_spectra)
    line_flux_error = np.abs(fitted_line_flux - truth_line_flux) / np.abs(
        truth_line_flux
    )
    line_peak_error = np.abs(fitted_line_peak - truth_line_peak) / np.abs(
        truth_line_peak
    )
    source_cube_error = np.sqrt(
        np.sum((source_cubes - truth_source_cubes) ** 2, axis=(1, 2, 3))
    ) / np.sqrt(np.sum(truth_source_cubes**2, axis=(1, 2, 3)))

    def morphology_error(fitted, truth):
        return np.linalg.norm(fitted - truth, axis=(1, 2)) / np.linalg.norm(
            truth, axis=(1, 2)
        )

    fitted_line_centroids = np.asarray([_centroid(image) for image in line_morphologies])
    truth_line_centroids = np.asarray([_centroid(image) for image in truth_line])
    model = np.asarray(observation.render(blend.get_model()), dtype=float)
    residual = data - model
    optimality = blend.parameter_optimization_diagnostics()
    return ChromaticRecoveryMetrics(
        model_kind=model_kind,
        spectral_mode=spectral_mode,
        optimizer=optimizer,
        start=start,
        seed=int(seed),
        iterations=int(iterations),
        train_chi2_per_voxel=float(np.mean(residual[train] ** 2 / variance[train])),
        heldout_chi2_per_voxel=float(
            np.mean(residual[heldout] ** 2 / variance[heldout])
        ),
        relative_projected_gradient=optimality.relative_projected_gradient,
        spectrum_relative_l2=spectrum_error,
        integrated_flux_relative_error=flux_error,
        line_flux_relative_error=line_flux_error,
        line_peak_relative_error=line_peak_error,
        source_cube_relative_l2=source_cube_error,
        continuum_morphology_relative_l2=morphology_error(
            continuum_morphologies, truth_continuum
        ),
        line_morphology_relative_l2=morphology_error(line_morphologies, truth_line),
        line_centroid_error_px=np.linalg.norm(
            fitted_line_centroids - truth_line_centroids, axis=1
        ),
        fitted_spectra=source_spectra,
        fitted_continuum_morphologies=continuum_morphologies,
        fitted_line_morphologies=line_morphologies,
        data=data,
        model=model,
        residual=residual,
        variance=variance,
        train_valid=train,
        heldout_valid=heldout,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=MODELS, default="rank1")
    parser.add_argument("--spectral-mode", choices=SPECTRAL_MODES, default="oracle")
    parser.add_argument("--start", choices=tuple(START_WIDTH_SCALE), default="A")
    parser.add_argument("--seed", type=int, default=RECOVERY_SEED)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--optimizer",
        choices=("variable_projection", "adaprox"),
        default="adaprox",
    )
    parser.add_argument("--spectral-smoothness-strength", type=float, default=0)
    args = parser.parse_args()
    result = fit_chromatic_cube(**vars(args))
    bulky = {
        "fitted_spectra",
        "fitted_continuum_morphologies",
        "fitted_line_morphologies",
        "data",
        "model",
        "residual",
        "variance",
        "train_valid",
        "heldout_valid",
    }
    for field in result.__dataclass_fields__:
        if field in bulky:
            continue
        value = getattr(result, field)
        if isinstance(value, np.ndarray):
            value = np.array2string(value, precision=4)
        print("{}: {}".format(field, value))


if __name__ == "__main__":
    main()

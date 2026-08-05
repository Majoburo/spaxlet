"""Fit the intrinsic 1,024-color many-source IFU regression case."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import spaxlet

from benchmarks.ifu_parity_contracts import (
    MANY_SOURCE_SHAPE,
    MANY_SOURCE_SPECS,
    many_source_morphologies,
    many_source_noisy_cube,
    many_source_spectra,
)
from benchmarks.run_spt0311_deblend import morphology_parameter


@dataclass(frozen=True)
class JointFitMetrics:
    """Truth and residual diagnostics for one predeclared joint fit."""

    slices: int
    support: str
    constraint: str
    spectral_smoothness_strength: float
    spatial_smoothness_strength: float
    start: int
    iterations: int
    objective: float
    initial_chi2_per_voxel: float
    chi2_per_voxel: float
    whitened_residual_rms: float
    relative_projected_gradient: float
    spectrum_relative_l2: np.ndarray
    morphology_relative_l2: np.ndarray
    centroid_error_px: np.ndarray
    spectrum_cosine: np.ndarray
    spectrum_curvature_rms: np.ndarray
    morphology_identity_margin: np.ndarray
    median_peak_snr_per_slice: np.ndarray
    integrated_peak_snr: np.ndarray
    channel_indices: np.ndarray
    fitted_spectra: np.ndarray
    fitted_morphologies: np.ndarray
    truth_spectra: np.ndarray
    truth_morphologies: np.ndarray
    data: np.ndarray
    model: np.ndarray
    residual: np.ndarray
    variance: np.ndarray
    valid: np.ndarray


def channel_indices(slices):
    """Return an evenly spaced subset that always includes both endpoints."""

    if not isinstance(slices, (int, np.integer)) or not 2 <= slices <= 1024:
        raise ValueError("slices must be an integer between 2 and 1024")
    return np.rint(np.linspace(0, 1023, slices)).astype(int)


def support_size(spec, mode):
    """Return the local morphology width for one declared support arm."""

    if mode == "truncated":
        return max(5, spec.support - 4)
    if mode == "correct":
        return spec.support
    if mode == "oversized":
        requested = spec.support + 4
        maximum = 2 * min(
            int(spec.center[0]),
            int(spec.center[1]),
            MANY_SOURCE_SHAPE[1] - 1 - int(spec.center[0]),
            MANY_SOURCE_SHAPE[2] - 1 - int(spec.center[1]),
        ) + 1
        return min(requested, maximum)
    raise ValueError("unknown support arm: {}".format(mode))


def source_constraint(spec, arm):
    """Map a joint constraint arm to one morphology constraint."""

    if arm in ("positivity", "centroid"):
        return arm
    if arm == "selective_symmetry":
        return "symmetry" if spec.knot is None else "centroid"
    if arm == "global_symmetry":
        return "symmetry"
    raise ValueError("unknown constraint arm: {}".format(arm))


def _initial_morphology(spec, size, start):
    half = size // 2
    axis = np.arange(size, dtype=float) - half
    rows, columns = np.meshgrid(axis, axis, indexing="ij")
    width_scale = (1.25, 0.80)[start]
    value = np.exp(-0.5 * (rows**2 + columns**2) / (width_scale * spec.sigma) ** 2)
    return value / value.sum()


def _sources(
    frame, data, support, constraint, start, spectral_wavelengths,
    spectral_smoothness_strength, spatial_smoothness_strength,
):
    rows, columns = np.indices(MANY_SOURCE_SHAPE[1:], dtype=float)
    sources = []
    for spec in MANY_SOURCE_SPECS:
        size = support_size(spec, support)
        half = size // 2
        origin = (int(spec.center[0]) - half, int(spec.center[1]) - half)
        box = spaxlet.Box((size, size), origin=origin)
        morphology_start = _initial_morphology(spec, size, start)
        local_center = (half, half)
        morphology = spaxlet.ImageMorphology(
            frame,
            morphology_parameter(
                morphology_start,
                local_center,
                source_constraint(spec, constraint),
                spatial_smoothness_strength,
            ),
            bbox=box,
            resizing=False,
        )
        aperture_radius = (2.0, 3.0)[start]
        aperture = np.hypot(
            rows - spec.center[0], columns - spec.center[1]
        ) <= aperture_radius
        spectrum_start = np.sum(data[:, aperture], axis=1)
        floor = max(float(np.percentile(np.abs(spectrum_start), 10)) * 1e-6, 1e-20)
        spectrum_start = np.maximum(spectrum_start, floor)
        spectral_constraint = None
        if spectral_smoothness_strength > 0:
            spectral_scale = max(
                float(np.mean(spectrum_start)), np.finfo(float).tiny
            )
            spectral_constraint = spaxlet.SpectralSmoothnessConstraint(
                spectral_wavelengths,
                spectral_smoothness_strength,
                reference_scale=spectral_scale,
                zero=1e-20,
            )
        spectrum = spaxlet.TabulatedSpectrum(
            frame, spectrum_start, constraint=spectral_constraint
        )
        sources.append(spaxlet.FactorizedComponent(frame, spectrum, morphology))
    return sources


def _full_morphologies(sources):
    result = []
    for source in sources:
        factor = spaxlet.measure.factorization(source)
        image = np.zeros(MANY_SOURCE_SHAPE[1:], dtype=float)
        y0, x0 = source.morphology.bbox.origin
        height, width = source.morphology.bbox.shape
        image[y0 : y0 + height, x0 : x0 + width] = factor.morphology
        result.append(image)
    return np.asarray(result)


def _centroids(morphologies):
    rows, columns = np.indices(MANY_SOURCE_SHAPE[1:], dtype=float)
    return np.asarray(
        [
            (np.sum(rows * image), np.sum(columns * image))
            for image in morphologies
        ]
    )


def fit_joint_model(
    *,
    slices=1024,
    support="correct",
    constraint="centroid",
    start=0,
    max_iter=60,
    spectral_smoothness_strength=0,
    spatial_smoothness_strength=0,
):
    """Fit one shared constrained morphology per source across all colors.

    This isolates joint factorization with a delta PSF. The independent
    channel-dependent PSF forward contract is tested on the same fixture.
    """

    if start not in (0, 1):
        raise ValueError("start must be 0 or 1")
    if (
        not np.isfinite(spectral_smoothness_strength)
        or spectral_smoothness_strength < 0
    ):
        raise ValueError("spectral smoothness strength must be non-negative")
    if (
        not np.isfinite(spatial_smoothness_strength)
        or spatial_smoothness_strength < 0
    ):
        raise ValueError("spatial smoothness strength must be non-negative")
    indices = channel_indices(slices)
    data, variance, valid, _ = many_source_noisy_cube(convolved=False)
    data = data[indices]
    variance = variance[indices]
    valid = valid[indices]
    weights = np.zeros(data.shape, dtype=float)
    weights[valid] = 1 / variance[valid]

    channels = tuple("ch{:04d}".format(index) for index in indices)
    frame = spaxlet.Frame(
        data.shape,
        channels=channels,
        psf=spaxlet.DeltaPSF(slices),
    )
    observation = spaxlet.Observation(
        data,
        channels=channels,
        psf=spaxlet.DeltaPSF(slices),
        weights=weights,
    ).match(frame)
    sources = _sources(
        frame,
        data,
        support,
        constraint,
        start,
        indices.astype(float),
        spectral_smoothness_strength,
        spatial_smoothness_strength,
    )
    blend = spaxlet.Blend(sources, observation)

    initial_model = np.asarray(blend.get_model())
    initial_chi2 = float(np.sum(weights * (data - initial_model) ** 2))
    iterations, objective = blend.fit(
        max_iter,
        e_rel=0,
        project_initial=True,
        normalize_initial_factors=True,
        optimizer="adaprox",
        scheme="amsgrad",
        channel_chunk_size=64,
    )

    model = np.asarray(blend.get_model())
    residual = data - model
    valid_count = int(np.count_nonzero(valid))
    chi2 = float(np.sum(weights * residual**2))
    relative_projected_gradient = (
        blend.parameter_optimization_diagnostics().relative_projected_gradient
    )
    fitted_factors = [spaxlet.measure.factorization(source) for source in sources]
    fitted_spectra = np.asarray([factor.spectrum for factor in fitted_factors])
    fitted_morphologies = _full_morphologies(sources)
    truth_spectra = many_source_spectra()[:, indices]
    truth_morphologies = many_source_morphologies()

    spectrum_error = np.linalg.norm(fitted_spectra - truth_spectra, axis=1)
    spectrum_error /= np.linalg.norm(truth_spectra, axis=1)
    morphology_error = np.linalg.norm(
        fitted_morphologies - truth_morphologies, axis=(1, 2)
    )
    morphology_error /= np.linalg.norm(truth_morphologies, axis=(1, 2))
    fitted_centroids = _centroids(fitted_morphologies)
    truth_centroids = np.asarray([spec.center for spec in MANY_SOURCE_SPECS])
    centroid_error = np.linalg.norm(fitted_centroids - truth_centroids, axis=1)
    spectrum_cosine = np.sum(fitted_spectra * truth_spectra, axis=1)
    spectrum_cosine /= np.linalg.norm(fitted_spectra, axis=1) * np.linalg.norm(
        truth_spectra, axis=1
    )
    if slices > 2:
        normalized_spacing = np.diff(indices) / np.median(np.diff(indices))
        left, right = normalized_spacing[:-1], normalized_spacing[1:]
        curvature = 2 / (left + right)[None] * (
            (fitted_spectra[:, 2:] - fitted_spectra[:, 1:-1]) / right[None]
            - (fitted_spectra[:, 1:-1] - fitted_spectra[:, :-2]) / left[None]
        )
        spectrum_curvature_rms = np.sqrt(np.mean(curvature**2, axis=1))
    else:
        spectrum_curvature_rms = np.zeros(len(sources), dtype=float)
    fitted_unit = fitted_morphologies.reshape(len(sources), -1).copy()
    truth_unit = truth_morphologies.reshape(len(sources), -1).copy()
    fitted_unit /= np.linalg.norm(fitted_unit, axis=1)[:, None]
    truth_unit /= np.linalg.norm(truth_unit, axis=1)[:, None]
    morphology_cosine = fitted_unit @ truth_unit.T
    identity_margin = np.diag(morphology_cosine) - np.max(
        morphology_cosine - 2 * np.eye(len(sources)), axis=1
    )
    source_peaks = np.max(
        truth_spectra[:, :, None, None] * truth_morphologies[:, None],
        axis=(2, 3),
    )
    peak_snr = source_peaks / np.sqrt(variance[:, 0, 0])[None]

    return JointFitMetrics(
        slices=slices,
        support=support,
        constraint=constraint,
        spectral_smoothness_strength=float(spectral_smoothness_strength),
        spatial_smoothness_strength=float(spatial_smoothness_strength),
        start=start,
        iterations=int(iterations),
        objective=float(objective),
        initial_chi2_per_voxel=initial_chi2 / valid_count,
        chi2_per_voxel=chi2 / valid_count,
        whitened_residual_rms=np.sqrt(chi2 / valid_count),
        relative_projected_gradient=relative_projected_gradient,
        spectrum_relative_l2=spectrum_error,
        morphology_relative_l2=morphology_error,
        centroid_error_px=centroid_error,
        spectrum_cosine=spectrum_cosine,
        spectrum_curvature_rms=spectrum_curvature_rms,
        morphology_identity_margin=identity_margin,
        median_peak_snr_per_slice=np.median(peak_snr, axis=1),
        integrated_peak_snr=np.sqrt(np.sum(peak_snr**2, axis=1)),
        channel_indices=indices,
        fitted_spectra=fitted_spectra,
        fitted_morphologies=fitted_morphologies,
        truth_spectra=truth_spectra,
        truth_morphologies=truth_morphologies,
        data=data,
        model=model,
        residual=residual,
        variance=variance,
        valid=valid,
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slices", type=int, default=1024)
    parser.add_argument(
        "--support", choices=("truncated", "correct", "oversized"), default="correct"
    )
    parser.add_argument(
        "--constraint",
        choices=("positivity", "centroid", "selective_symmetry", "global_symmetry"),
        default="centroid",
    )
    parser.add_argument("--start", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--spectral-smoothness-strength", type=float, default=0)
    parser.add_argument("--spatial-smoothness-strength", type=float, default=0)
    return parser


def main():
    args = _parser().parse_args()
    metrics = fit_joint_model(**vars(args))
    bulky = {
        "channel_indices",
        "fitted_spectra",
        "fitted_morphologies",
        "truth_spectra",
        "truth_morphologies",
        "data",
        "model",
        "residual",
        "variance",
        "valid",
    }
    for field in metrics.__dataclass_fields__:
        if field in bulky:
            continue
        value = getattr(metrics, field)
        if isinstance(value, np.ndarray):
            value = np.array2string(value, precision=4)
        print("{}: {}".format(field, value))


if __name__ == "__main__":
    main()

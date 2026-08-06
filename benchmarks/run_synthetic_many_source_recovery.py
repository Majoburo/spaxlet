"""Fit and score the identifiable many-source IFU recovery contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import spaxlet

from benchmarks.many_source_recovery_contract import (
    RECOVERY_SEED,
    RECOVERY_SHAPE,
    RECOVERY_SOURCE_SPECS,
    recovery_morphologies,
    recovery_noisy_cube,
    recovery_psfs,
    recovery_spectra,
)
from benchmarks.signed_recovery_metrics import (
    cancellation_ratio,
    signed_source_metrics,
)


# The 180-iteration pilot budget is deliberately not converged.  On the
# noiseless cube the maximum signed source error falls 5.96% -> 1.86% ->
# 0.31% -> 0.024% at 180, 600, 2000 and 6000 iterations, so a pilot number
# still carries an optimization transient.  On the noisy cube the same sweep
# gives +7.64%, +6.06%, +6.20% and +6.26%: about 1.4 points is transient and
# the remaining 6.3% is a converged estimation bias.  Quote converged-budget
# results for anything that reports a per-source flux.
RECOVERY_PILOT_MAX_ITER = 180
RECOVERY_CONVERGED_MAX_ITER = 2000


@dataclass(frozen=True)
class RecoveryMetrics:
    start: str
    seed: int
    optimizer: str
    spectral_smoothness_strength: float
    spatial_smoothness_strength: float
    iterations: int
    chi2_per_valid_voxel: float
    relative_projected_gradient: float
    spectrum_relative_l2: np.ndarray
    spectrum_cosine: np.ndarray
    morphology_relative_l2: np.ndarray
    morphology_cosine: np.ndarray
    centroid_error_px: np.ndarray
    integrated_flux_relative_error: np.ndarray
    signed_integrated_flux_relative_error: np.ndarray
    signed_continuum_flux_relative_error: np.ndarray
    signed_line_flux_relative_error: np.ndarray
    binned_signed_flux_relative_error: np.ndarray
    flux_cancellation_ratio: float
    line_flux_relative_error: np.ndarray
    line_peak_relative_error: np.ndarray
    morphology_identity_margin: np.ndarray
    fitted_spectra: np.ndarray
    fitted_morphologies: np.ndarray
    data: np.ndarray
    model: np.ndarray
    residual: np.ndarray
    variance: np.ndarray
    valid: np.ndarray


START_WIDTH_SCALE = {"A": 1.25, "B": 0.72, "C": 1.75}


def _centroid(image):
    rows, columns = np.indices(image.shape, dtype=float)
    total = float(np.sum(image))
    return np.asarray([np.sum(rows * image), np.sum(columns * image)]) / total


@dataclass(frozen=True)
class CatalogEntry:
    """One catalog row: everything the fit is told about a source.

    Separating this from ``RecoverySourceSpec`` makes the oracle content
    explicit.  ``center``, ``support``, ``start_width`` and ``centroid`` are
    all quantities a real pipeline must estimate from the data.
    """

    name: str
    center: tuple[int, int]
    support: int
    start_width: float
    centroid: tuple[float, float]


def oracle_catalog():
    """Return the catalog built from truth, as the declared contract uses."""

    entries = []
    for spec, truth in zip(RECOVERY_SOURCE_SPECS, recovery_morphologies()):
        half = spec.support // 2
        origin = (spec.center[0] - half, spec.center[1] - half)
        local_truth = truth[
            origin[0] : origin[0] + spec.support,
            origin[1] : origin[1] + spec.support,
        ]
        entries.append(
            CatalogEntry(
                name=spec.name,
                center=spec.center,
                support=spec.support,
                start_width=float(np.sqrt(spec.sigma_y * spec.sigma_x)),
                centroid=tuple(_centroid(local_truth)),
            )
        )
    return tuple(entries)


def _initial_image(entry, width_scale):
    half = entry.support // 2
    axis = np.arange(entry.support, dtype=float) - half
    rows, columns = np.meshgrid(axis, axis, indexing="ij")
    width = width_scale * entry.start_width
    value = np.exp(-0.5 * (rows**2 + columns**2) / width**2)
    return value / value.sum()


def _sources(
    frame,
    start,
    data,
    spectral_smoothness_strength,
    spatial_smoothness_strength,
    catalog=None,
    positivity=True,
):
    if catalog is None:
        catalog = oracle_catalog()
    rows, columns = np.indices(RECOVERY_SHAPE[1:], dtype=float)
    sources = []
    for spec in catalog:
        half = spec.support // 2
        origin = (spec.center[0] - half, spec.center[1] - half)
        box = spaxlet.Box((spec.support, spec.support), origin=origin)
        local_center = np.asarray(spec.centroid, dtype=float)
        # Dropping positivity is a diagnostic arm only.  It isolates how much
        # of the converged flux bias is a clipped-estimator effect: the fitted
        # morphologies pin 25-49% of their in-support pixels at exactly zero
        # while truth is strictly positive at every one of them.  Spectra stay
        # nonnegative regardless, because the variable-projection amplitude
        # step solves a nonnegative Gram system.
        identity_constraints = (
            (
                spaxlet.CentroidConstraint(local_center),
                spaxlet.PositivityConstraint(),
            )
            if positivity
            else (spaxlet.CentroidConstraint(local_center),)
        )
        if spatial_smoothness_strength > 0:
            image_start = _initial_image(spec, START_WIDTH_SCALE[start])
            constraint = spaxlet.ProximalDykstraConstraintChain(
                spaxlet.SpatialSmoothnessConstraint(
                    spatial_smoothness_strength,
                    reference_scale=max(float(np.mean(image_start)), 1e-20),
                ),
                *identity_constraints,
                max_iter=2000,
                rtol=1e-9,
                atol=1e-11,
            )
        else:
            image_start = _initial_image(spec, START_WIDTH_SCALE[start])
            constraint = spaxlet.DykstraConstraintChain(
                *identity_constraints,
                max_iter=20000,
                rtol=1e-12,
                atol=1e-13,
            )
        image = spaxlet.Parameter(
            image_start,
            name="image",
            step=spaxlet.parameter.relative_step,
            constraint=constraint,
        )
        morphology = spaxlet.ImageMorphology(
            frame, image, bbox=box, resizing=False
        )
        aperture_radius = max(2.0, 0.35 * spec.support)
        aperture = np.hypot(
            rows - spec.center[0], columns - spec.center[1]
        ) <= aperture_radius
        spectrum_start = np.maximum(np.sum(data[:, aperture], axis=1), 1e-10)
        spectral_constraint = (
            spaxlet.SpectralSmoothnessConstraint(
                np.arange(RECOVERY_SHAPE[0], dtype=float),
                spectral_smoothness_strength,
                reference_scale=max(float(np.mean(spectrum_start)), 1e-20),
            )
            if spectral_smoothness_strength > 0
            else None
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
        image = np.zeros(RECOVERY_SHAPE[1:], dtype=float)
        y0, x0 = source.morphology.bbox.origin
        height, width = source.morphology.bbox.shape
        image[y0 : y0 + height, x0 : x0 + width] = factor.morphology
        result.append(image)
    return np.asarray(result)


def _line_observables(spectra):
    """Measure local continuum-subtracted flux and peak for declared lines."""

    channel = np.arange(RECOVERY_SHAPE[0], dtype=float)
    fluxes = []
    peaks = []
    for spectrum, spec in zip(spectra, RECOVERY_SOURCE_SPECS):
        source_fluxes = []
        source_peaks = []
        for center in spec.line_channels:
            distance = np.abs(channel - center)
            sideband = (distance >= 5) & (distance <= 9)
            coefficients = np.polyfit(channel[sideband], spectrum[sideband], 1)
            continuum = np.polyval(coefficients, channel)
            line = spectrum - continuum
            source_fluxes.append(float(np.sum(line[distance <= 4])))
            source_peaks.append(float(np.max(line[distance <= 2])))
        fluxes.append(source_fluxes)
        peaks.append(source_peaks)
    return np.asarray(fluxes), np.asarray(peaks)


def fit_catalog(
    *,
    catalog=None,
    start="A",
    max_iter=RECOVERY_PILOT_MAX_ITER,
    seed=RECOVERY_SEED,
    optimizer="variable_projection",
    spectral_smoothness_strength=0,
    spatial_smoothness_strength=0,
    positivity=True,
):
    """Fit any catalog and return fitted factors with no truth comparison.

    Split out of ``fit_recovery_cube`` so a catalog with a different source
    count, such as one produced by detection, can be fitted with exactly the
    same forward model, optimizer and constraints.  Truth-referenced scoring
    is the caller's responsibility because it needs a source matching.
    """

    if start not in START_WIDTH_SCALE:
        raise ValueError("start must be A, B, or C")
    if optimizer not in ("variable_projection", "adaprox"):
        raise ValueError("optimizer must be variable_projection or adaprox")
    for label, strength in (
        ("spectral", spectral_smoothness_strength),
        ("spatial", spatial_smoothness_strength),
    ):
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("{} smoothness strength must be non-negative".format(label))
    if optimizer == "variable_projection" and (
        spectral_smoothness_strength > 0 or spatial_smoothness_strength > 0
    ):
        raise ValueError("smoothness arms require the matched adaprox optimizer")
    data, variance, valid, _, _ = recovery_noisy_cube(seed=seed)
    weights = np.zeros_like(variance)
    weights[valid] = 1.0 / variance[valid]
    channels = tuple("ch{:03d}".format(i) for i in range(RECOVERY_SHAPE[0]))
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
    sources = _sources(
        frame,
        start,
        data,
        spectral_smoothness_strength,
        spatial_smoothness_strength,
        catalog=catalog,
        positivity=positivity,
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
        # Every declared start is already in the unit-L1 morphology gauge.
        # Penalty proximal maps are intentionally not treated as scale-invariant
        # feasibility projections by Blend's generic normalization guard.
        # The unit-L1 gauge is only well posed for a non-negative morphology,
        # and Blend refuses it otherwise, so the diagnostic arm forgoes it.
        normalize_initial_factors=spatial_smoothness_strength == 0 and positivity,
        **optimizer_arguments,
    )

    fitted_spectra = []
    for source in sources:
        factor = spaxlet.measure.factorization(source)
        fitted_spectra.append(factor.spectrum)
    fitted_spectra = np.asarray(fitted_spectra)
    fitted_morphologies = _full_morphologies(sources)

    model = np.asarray(observation.render(blend.get_model()), dtype=float)
    residual = data - model
    chi2 = float(np.sum(weights * residual**2)) / int(np.count_nonzero(valid))
    return {
        "fitted_spectra": fitted_spectra,
        "fitted_morphologies": fitted_morphologies,
        "iterations": int(iterations),
        "chi2_per_valid_voxel": chi2,
        "relative_projected_gradient": (
            blend.parameter_optimization_diagnostics().relative_projected_gradient
        ),
        "data": data,
        "model": model,
        "residual": residual,
        "variance": variance,
        "valid": valid,
    }


def fit_recovery_cube(
    *,
    start="A",
    max_iter=RECOVERY_PILOT_MAX_ITER,
    seed=RECOVERY_SEED,
    optimizer="variable_projection",
    spectral_smoothness_strength=0,
    spatial_smoothness_strength=0,
    catalog=None,
    positivity=True,
):
    """Fit one deterministic start and return source-resolved truth metrics."""

    fit = fit_catalog(
        catalog=catalog,
        start=start,
        max_iter=max_iter,
        seed=seed,
        optimizer=optimizer,
        spectral_smoothness_strength=spectral_smoothness_strength,
        spatial_smoothness_strength=spatial_smoothness_strength,
        positivity=positivity,
    )
    fitted_spectra = fit["fitted_spectra"]
    fitted_morphologies = fit["fitted_morphologies"]
    data, variance, valid = fit["data"], fit["variance"], fit["valid"]
    model, residual = fit["model"], fit["residual"]
    iterations = fit["iterations"]
    truth_spectra = recovery_spectra()
    truth_morphologies = recovery_morphologies()

    spectrum_relative_l2 = np.linalg.norm(
        fitted_spectra - truth_spectra, axis=1
    ) / np.linalg.norm(truth_spectra, axis=1)
    spectrum_cosine = np.sum(fitted_spectra * truth_spectra, axis=1) / (
        np.linalg.norm(fitted_spectra, axis=1)
        * np.linalg.norm(truth_spectra, axis=1)
    )
    morphology_relative_l2 = np.linalg.norm(
        fitted_morphologies - truth_morphologies, axis=(1, 2)
    ) / np.linalg.norm(truth_morphologies, axis=(1, 2))
    morphology_cosine = np.sum(
        fitted_morphologies * truth_morphologies, axis=(1, 2)
    ) / (
        np.linalg.norm(fitted_morphologies, axis=(1, 2))
        * np.linalg.norm(truth_morphologies, axis=(1, 2))
    )
    fitted_centroids = np.asarray([_centroid(image) for image in fitted_morphologies])
    truth_centroids = np.asarray([_centroid(image) for image in truth_morphologies])
    centroid_error = np.linalg.norm(fitted_centroids - truth_centroids, axis=1)
    signed = signed_source_metrics(fitted_spectra, truth_spectra)
    integrated_flux_relative_error = np.abs(signed["signed_total_relative"])
    fitted_line_flux, fitted_line_peak = _line_observables(fitted_spectra)
    truth_line_flux, truth_line_peak = _line_observables(truth_spectra)
    line_flux_relative_error = np.abs(fitted_line_flux - truth_line_flux) / np.maximum(
        np.abs(truth_line_flux), np.finfo(float).tiny
    )
    line_peak_relative_error = np.abs(fitted_line_peak - truth_line_peak) / np.maximum(
        np.abs(truth_line_peak), np.finfo(float).tiny
    )

    count = len(fitted_morphologies)
    fitted_unit = fitted_morphologies.reshape(count, -1).copy()
    truth_unit = truth_morphologies.reshape(count, -1).copy()
    fitted_unit /= np.linalg.norm(fitted_unit, axis=1)[:, None]
    truth_unit /= np.linalg.norm(truth_unit, axis=1)[:, None]
    cross_cosine = fitted_unit @ truth_unit.T
    identity_margin = np.diag(cross_cosine) - np.max(
        cross_cosine - 2 * np.eye(count), axis=1
    )

    return RecoveryMetrics(
        start=start,
        seed=int(seed),
        optimizer=optimizer,
        spectral_smoothness_strength=float(spectral_smoothness_strength),
        spatial_smoothness_strength=float(spatial_smoothness_strength),
        iterations=iterations,
        chi2_per_valid_voxel=fit["chi2_per_valid_voxel"],
        relative_projected_gradient=fit["relative_projected_gradient"],
        spectrum_relative_l2=spectrum_relative_l2,
        spectrum_cosine=spectrum_cosine,
        morphology_relative_l2=morphology_relative_l2,
        morphology_cosine=morphology_cosine,
        centroid_error_px=centroid_error,
        integrated_flux_relative_error=integrated_flux_relative_error,
        signed_integrated_flux_relative_error=signed["signed_total_relative"],
        signed_continuum_flux_relative_error=signed["signed_continuum_relative"],
        signed_line_flux_relative_error=signed["signed_line_relative"],
        binned_signed_flux_relative_error=signed["binned_signed_relative"],
        flux_cancellation_ratio=cancellation_ratio(signed["signed_total_absolute"]),
        line_flux_relative_error=line_flux_relative_error,
        line_peak_relative_error=line_peak_relative_error,
        morphology_identity_margin=identity_margin,
        fitted_spectra=fitted_spectra,
        fitted_morphologies=fitted_morphologies,
        data=data,
        model=model,
        residual=residual,
        variance=variance,
        valid=valid,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", choices=tuple(START_WIDTH_SCALE), default="A")
    parser.add_argument("--max-iter", type=int, default=RECOVERY_PILOT_MAX_ITER)
    parser.add_argument("--seed", type=int, default=RECOVERY_SEED)
    parser.add_argument(
        "--optimizer",
        choices=("variable_projection", "adaprox"),
        default="variable_projection",
    )
    parser.add_argument("--spectral-smoothness-strength", type=float, default=0)
    parser.add_argument("--spatial-smoothness-strength", type=float, default=0)
    args = parser.parse_args()
    result = fit_recovery_cube(**vars(args))
    for field in result.__dataclass_fields__:
        if field in {"fitted_spectra", "fitted_morphologies", "data", "model", "residual", "variance", "valid"}:
            continue
        value = getattr(result, field)
        if isinstance(value, np.ndarray):
            value = np.array2string(value, precision=4)
        print("{}: {}".format(field, value))


if __name__ == "__main__":
    main()

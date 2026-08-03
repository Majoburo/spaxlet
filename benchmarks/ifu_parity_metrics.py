"""Strict shared metrics for the lisasep/Scarlet IFU comparison.

All functions consume plain NumPy arrays so both codes are scored after fitting
by the same implementation. Truth-referenced metrics are diagnostic only;
observable residual metrics and held-out prediction are the model-selection
quantities available on real data.
"""

from __future__ import annotations

import numpy as np


def _matching_finite(value, reference, name):
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if value.shape != reference.shape:
        raise ValueError("{} arrays must have matching shapes".format(name))
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("{} value must be non-empty and finite".format(name))
    if not np.all(np.isfinite(reference)):
        raise ValueError("{} reference must be finite".format(name))
    return value, reference


def relative_l2(value, reference):
    value, reference = _matching_finite(value, reference, "relative-L2")
    denominator = float(np.linalg.norm(reference))
    if denominator <= np.finfo(float).tiny:
        raise ValueError("relative-L2 reference must have non-zero norm")
    return float(np.linalg.norm(value - reference) / denominator)


def cosine_similarity(value, reference):
    value, reference = _matching_finite(value, reference, "cosine")
    denominator = float(np.linalg.norm(value) * np.linalg.norm(reference))
    if denominator <= np.finfo(float).tiny:
        raise ValueError("cosine arrays must have non-zero norm")
    return float(np.vdot(value, reference).real / denominator)


def normalized_morphology(value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("morphology must be a finite two-dimensional array")
    if np.any(value < 0):
        raise ValueError("morphology must be non-negative")
    total = float(np.sum(value))
    if total <= np.finfo(float).tiny:
        raise ValueError("morphology must have positive total flux")
    return value / total


def centroid(value):
    value = normalized_morphology(value)
    rows, columns = np.indices(value.shape, dtype=float)
    return np.asarray([np.sum(rows * value), np.sum(columns * value)])


def translate_morphology(value, offset_yx):
    """Translate a morphology with bilinear sampling and zero boundaries.

    This expresses injected morphologies in the latent coordinate frame after
    an off-center instrumental PSF has been recentered. It is a coordinate
    conversion for truth scoring, not registration fitted from the recovery.
    """

    value = np.asarray(value, dtype=float)
    offset = np.asarray(offset_yx, dtype=float)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("morphology must be a finite two-dimensional array")
    if offset.shape != (2,) or not np.all(np.isfinite(offset)):
        raise ValueError("morphology offset must contain two finite coordinates")

    rows, columns = np.indices(value.shape, dtype=float)
    source_rows = rows - offset[0]
    source_columns = columns - offset[1]
    row0 = np.floor(source_rows).astype(int)
    column0 = np.floor(source_columns).astype(int)
    row_fraction = source_rows - row0
    column_fraction = source_columns - column0
    translated = np.zeros_like(value)
    for delta_row, row_weight in (
        (0, 1.0 - row_fraction),
        (1, row_fraction),
    ):
        for delta_column, column_weight in (
            (0, 1.0 - column_fraction),
            (1, column_fraction),
        ):
            source_row = row0 + delta_row
            source_column = column0 + delta_column
            valid = (
                (source_row >= 0)
                & (source_row < value.shape[0])
                & (source_column >= 0)
                & (source_column < value.shape[1])
            )
            translated[valid] += (
                row_weight[valid]
                * column_weight[valid]
                * value[source_row[valid], source_column[valid]]
            )
    return translated


def _gaussian_smooth(value, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("smoothing scales must be positive and finite")
    radius = int(np.ceil(4.0 * sigma))
    coordinates = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
    kernel /= kernel.sum()
    result = np.asarray(value, dtype=float)
    for axis in range(result.ndim):
        result = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), axis, result
        )
    return result


def structured_relative_l2(value, reference, scales=(1.0, 2.0)):
    value, reference = _matching_finite(value, reference, "structured morphology")
    if value.ndim != 2:
        raise ValueError("structured morphology arrays must be two-dimensional")
    scales = tuple(float(scale) for scale in scales)
    if not scales:
        raise ValueError("at least one smoothing scale is required")
    by_scale = tuple(
        relative_l2(_gaussian_smooth(value, scale), _gaussian_smooth(reference, scale))
        for scale in scales
    )
    return float(np.sqrt(np.mean(np.square(by_scale)))), by_scale


def spectral_metrics(value, reference, wavelength, n_bin=6):
    """Return scale-sensitive integrated and binned spectrum scores."""

    value, reference = _matching_finite(value, reference, "spectrum")
    wavelength = np.asarray(wavelength, dtype=float)
    if value.ndim != 1 or wavelength.shape != value.shape:
        raise ValueError("spectra and wavelength must be matching one-dimensional arrays")
    if not np.all(np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0):
        raise ValueError("wavelength must be finite and strictly increasing")
    if not isinstance(n_bin, (int, np.integer)) or n_bin <= 0 or n_bin > value.size:
        raise ValueError("n_bin must be between one and the spectrum length")
    reference_total = float(np.sum(reference))
    if abs(reference_total) <= np.finfo(float).tiny:
        raise ValueError("reference spectrum must have non-zero integrated flux")

    centers = []
    errors = []
    for index in np.array_split(np.arange(value.size), n_bin):
        bin_total = float(np.sum(reference[index]))
        if abs(bin_total) <= np.finfo(float).tiny:
            raise ValueError("every reference spectral bin must have non-zero flux")
        centers.append(float(np.mean(wavelength[index])))
        errors.append(float(np.sum(value[index] - reference[index]) / bin_total))
    errors = np.asarray(errors)
    return {
        "relative_l2": relative_l2(value, reference),
        "cosine": cosine_similarity(value, reference),
        "integrated_flux_error": float(np.sum(value) / reference_total - 1.0),
        "binned_wavelength": np.asarray(centers),
        "binned_fractional_error": errors,
        "binned_fractional_error_rms": float(np.sqrt(np.mean(errors**2))),
        "binned_fractional_error_max_abs": float(np.max(np.abs(errors))),
    }


def morphology_metrics(value, reference, scales=(1.0, 2.0)):
    """Score intrinsic morphologies after explicit unit-flux normalization."""

    value = normalized_morphology(value)
    reference = normalized_morphology(reference)
    if value.shape != reference.shape:
        raise ValueError("morphologies must have matching shapes")
    structured, by_scale = structured_relative_l2(value, reference, scales=scales)
    return {
        "relative_l2": relative_l2(value, reference),
        "cosine": cosine_similarity(value, reference),
        "centroid_error_px": float(np.linalg.norm(centroid(value) - centroid(reference))),
        "structured_relative_l2": structured,
        "structured_relative_l2_by_scale": np.asarray(by_scale),
        "structured_scales_px": np.asarray(scales, dtype=float),
    }


def residual_metrics(residual, inverse_variance):
    """Return truth-independent fit and spatial-whiteness diagnostics."""

    residual = np.asarray(residual, dtype=float)
    weights = np.asarray(inverse_variance, dtype=float)
    if residual.ndim < 2 or residual.shape != weights.shape:
        raise ValueError("residual and inverse variance must match and be channel-first")
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(weights)):
        raise ValueError("residual and inverse variance must be finite")
    if np.any(weights < 0):
        raise ValueError("inverse variance must be non-negative")
    valid = weights > 0
    valid_voxels = int(np.count_nonzero(valid))
    if valid_voxels == 0:
        raise ValueError("at least one voxel must have positive inverse variance")
    whitened = np.where(valid, residual * np.sqrt(weights), 0.0)
    whitened = np.where(valid, whitened - np.mean(whitened[valid]), 0.0)

    entropies = []
    transform_axes = tuple(range(residual.ndim - 1))
    for channel in range(residual.shape[0]):
        if np.count_nonzero(valid[channel]) < 3:
            continue
        transformed = np.fft.rfftn(whitened[channel], axes=transform_axes)
        power = np.abs(transformed).reshape(-1) ** 2
        if power.size > 1:
            power = power[1:]
        total = float(np.sum(power))
        if total <= np.finfo(float).tiny or power.size <= 1:
            continue
        probability = power / total
        positive = probability > 0
        entropy = -float(np.sum(probability[positive] * np.log(probability[positive])))
        entropies.append(entropy / np.log(power.size))

    correlations = []
    for spatial_axis in range(1, residual.ndim):
        lower = [slice(None)] * residual.ndim
        upper = [slice(None)] * residual.ndim
        lower[spatial_axis] = slice(None, -1)
        upper[spatial_axis] = slice(1, None)
        lower = tuple(lower)
        upper = tuple(upper)
        pair_valid = valid[lower] & valid[upper]
        left = whitened[lower][pair_valid]
        right = whitened[upper][pair_valid]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > np.finfo(float).tiny:
            correlations.append(abs(float(np.vdot(left, right).real)) / denominator)

    return {
        "chi_square_per_voxel": float(np.sum(weights * residual**2) / valid_voxels),
        "valid_voxels": valid_voxels,
        "power_spectral_entropy": (
            float(np.mean(entropies)) if entropies else float("nan")
        ),
        "lag1_autocorrelation": (
            float(np.mean(correlations)) if correlations else float("nan")
        ),
    }


def start_sensitivity(values):
    """Return symmetric pairwise spread across all predeclared starts."""

    values = [np.asarray(value, dtype=float) for value in values]
    if len(values) < 2:
        raise ValueError("start sensitivity requires at least two runs")
    shape = values[0].shape
    if any(value.shape != shape or not np.all(np.isfinite(value)) for value in values):
        raise ValueError("start-sensitivity arrays must be finite with matching shapes")
    distances = []
    for left_index, left in enumerate(values[:-1]):
        for right in values[left_index + 1 :]:
            denominator = 0.5 * (float(np.linalg.norm(left)) + float(np.linalg.norm(right)))
            if denominator <= np.finfo(float).tiny:
                distance = 0.0
            else:
                distance = float(np.linalg.norm(left - right) / denominator)
            distances.append(distance)
    return {
        "pairwise_relative_l2": np.asarray(distances),
        "median_pairwise_relative_l2": float(np.median(distances)),
        "max_pairwise_relative_l2": float(np.max(distances)),
    }

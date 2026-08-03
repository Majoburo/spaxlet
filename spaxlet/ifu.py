"""Small preprocessing utilities for wavelength-resolved IFU PSFs."""

import warnings

import numpy as np
from scipy.ndimage import shift


def _validate_psf_kernels(kernels):
    values = np.asarray(kernels, dtype=float)
    if values.ndim == 2:
        values = values[None, ...]
    if (
        values.ndim != 3
        or values.shape[1] <= 0
        or values.shape[2] <= 0
        or np.any(~np.isfinite(values))
        or np.any(values < 0)
    ):
        raise ValueError(
            "PSF kernels must be finite non-negative two-dimensional planes"
        )
    mass = np.sum(values, axis=(1, 2))
    if np.any(mass <= np.finfo(float).tiny):
        raise ValueError("every PSF kernel must contain positive flux")
    return values, mass


def psf_centroids(kernels):
    """Return channel centroids in kernel-centered ``(dy, dx)`` coordinates."""
    values, mass = _validate_psf_kernels(kernels)
    yy, xx = np.indices(values.shape[1:], dtype=float)
    center = 0.5 * (np.asarray(values.shape[1:]) - 1)
    y = np.sum(values * yy[None], axis=(1, 2)) / mass - center[0]
    x = np.sum(values * xx[None], axis=(1, 2)) / mass - center[1]
    return np.column_stack((y, x))


def crop_psf_kernels(kernels, size, *, normalize=True):
    """Centrally crop PSF planes and return retained-flux fractions."""
    values, full_sum = _validate_psf_kernels(kernels)
    size = int(size)
    if size <= 0 or size % 2 == 0:
        raise ValueError("PSF crop size must be a positive odd integer")
    if values.shape[1] != values.shape[2] or size > values.shape[1]:
        raise ValueError("PSF crop must fit inside square input kernels")

    start = (values.shape[1] - size) // 2
    cropped = values[:, start : start + size, start : start + size].copy()
    cropped_sum = np.sum(cropped, axis=(1, 2))
    retained = cropped_sum / full_sum
    if np.any(cropped_sum <= np.finfo(float).tiny):
        raise ValueError("every cropped PSF kernel must contain positive flux")
    if normalize:
        cropped /= cropped_sum[:, None, None]

    offset = float(np.max(np.linalg.norm(psf_centroids(cropped), axis=1)))
    if offset > 0.05:
        warnings.warn(
            "cropped PSF centroid lies {:.3f} px from the array centre; "
            "recenter after cropping to avoid an artificial astrometric "
            "shift".format(offset),
            RuntimeWarning,
            stacklevel=2,
        )
    return cropped, retained


def recenter_psf_kernels(kernels):
    """Shift every PSF to the kernel center and return removed centroids."""
    values, _ = _validate_psf_kernels(kernels)
    centroids = psf_centroids(values)
    centered = np.zeros_like(values)
    for channel, centroid in enumerate(centroids):
        centered[channel] = shift(
            values[channel],
            shift=-centroid,
            order=3,
            mode="constant",
            cval=0,
            prefilter=True,
        )
    centered = np.maximum(centered, 0)
    mass = np.sum(centered, axis=(1, 2))
    if np.any(mass <= np.finfo(float).tiny):
        raise ValueError("PSF recentering produced an empty channel")
    centered /= mass[:, None, None]
    return centered, centroids


def spatial_interpolation_weights(anchors_yx, image_shape):
    """Bilinear field-anchor weights on an image pixel grid.

    Coordinates outside the anchor rectangle use the nearest edge response.
    The returned ``(n_anchor, y, x)`` array sums to one at every pixel.
    """
    try:
        raw_y, raw_x = anchors_yx
    except (TypeError, ValueError) as error:
        raise ValueError(
            "anchors_yx must contain row and column coordinates"
        ) from error
    anchors_y = np.asarray(raw_y, dtype=float).reshape(-1)
    anchors_x = np.asarray(raw_x, dtype=float).reshape(-1)
    if anchors_y.size == 0 or anchors_x.size == 0:
        raise ValueError("at least one anchor is required on each axis")
    if np.any(~np.isfinite(anchors_y)) or np.any(~np.isfinite(anchors_x)):
        raise ValueError("anchor coordinates must be finite")
    if np.any(np.diff(anchors_y) <= 0) or np.any(np.diff(anchors_x) <= 0):
        raise ValueError("anchor coordinates must be strictly increasing")
    try:
        resolved_shape = tuple(int(size) for size in image_shape)
    except (TypeError, ValueError) as error:
        raise ValueError("image_shape must contain two positive dimensions") from error
    if len(resolved_shape) != 2:
        raise ValueError("a varying response requires a two-dimensional image shape")
    if any(size <= 0 for size in resolved_shape):
        raise ValueError("image_shape must contain positive dimensions")

    rows, columns = np.indices(resolved_shape, dtype=float)

    def axis_weights(anchors, coordinate):
        weights = np.zeros((anchors.size, *resolved_shape))
        if anchors.size == 1:
            weights[0] = 1
            return weights
        clamped = np.clip(coordinate, anchors[0], anchors[-1])
        index = np.clip(np.searchsorted(anchors, clamped) - 1, 0, anchors.size - 2)
        fraction = (clamped - anchors[index]) / (
            anchors[index + 1] - anchors[index]
        )
        flat_index = index.reshape(-1)
        positions = np.arange(flat_index.size)
        flat_weights = weights.reshape(anchors.size, -1)
        flat_weights[flat_index, positions] = (1 - fraction).reshape(-1)
        np.add.at(
            flat_weights,
            (flat_index + 1, positions),
            fraction.reshape(-1),
        )
        return weights

    weight_y = axis_weights(anchors_y, rows)
    weight_x = axis_weights(anchors_x, columns)
    return (weight_y[:, None] * weight_x[None, :]).reshape(
        anchors_y.size * anchors_x.size, *resolved_shape
    )

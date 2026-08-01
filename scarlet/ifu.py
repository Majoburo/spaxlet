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

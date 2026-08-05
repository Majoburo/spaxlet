"""Preprocessing utilities for wavelength-resolved IFU observations."""

from collections import namedtuple
import warnings

import numpy as np
from scipy.ndimage import shift


IFUBackgroundEstimate = namedtuple(
    "IFUBackgroundEstimate",
    ("background", "noise_scale", "background_voxels"),
)


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


def empirical_psf_kernels(
    cube,
    *,
    channel_indices=None,
    kernel_size=21,
    spectral_half_width=0,
    peak_yx=None,
):
    """Extract normalized wavelength-dependent kernels from a stellar IFU cube.

    Each output channel is formed by summing a local spectral window, removing
    the median of the two-pixel crop border, clipping negative values, and
    normalizing. The kernels are then subpixel-recentered with
    :func:`recenter_psf_kernels`.

    Parameters
    ----------
    cube: array
        Stellar calibration cube with shape ``(channel, y, x)``. Non-finite
        samples are ignored.
    channel_indices: one-dimensional integer array or None
        Channels for which kernels are required. Defaults to every channel.
    kernel_size: positive odd int
        Spatial crop size.
    spectral_half_width: non-negative int
        Number of neighboring channels included on either side.
    peak_yx: pair of ints or None
        Stellar peak in the full cube. If omitted, it is measured from the
        collapsed selected channels.

    Returns
    -------
    kernels, removed_centroids, peak_yx
        Unit-sum recentered kernels, the removed ``(dy, dx)`` shifts, and the
        integer crop center in the input cube.
    """
    values = np.asarray(cube, dtype=float)
    if values.ndim != 3 or any(size <= 0 for size in values.shape):
        raise ValueError("an empirical PSF cube must have shape (channel, y, x)")
    if not isinstance(kernel_size, (int, np.integer)):
        raise TypeError("kernel_size must be an integer")
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if not isinstance(spectral_half_width, (int, np.integer)):
        raise TypeError("spectral_half_width must be an integer")
    spectral_half_width = int(spectral_half_width)
    if spectral_half_width < 0:
        raise ValueError("spectral_half_width must be non-negative")
    if channel_indices is None:
        indices = np.arange(values.shape[0], dtype=int)
    else:
        indices = np.asarray(channel_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("channel_indices must be a one-dimensional integer array")
        indices = indices.astype(int)
        if indices.size == 0:
            raise ValueError("at least one empirical PSF channel is required")
        if np.any(indices < 0) or np.any(indices >= values.shape[0]):
            raise ValueError("an empirical PSF channel index is out of bounds")
        if np.unique(indices).size != indices.size:
            raise ValueError("empirical PSF channel indices must be unique")

    if peak_yx is None:
        if not np.any(np.isfinite(values[indices])):
            raise ValueError("the empirical PSF cube has no finite selected samples")
        collapsed = np.nansum(values[indices], axis=0)
        peak = tuple(int(index) for index in np.unravel_index(
            np.nanargmax(collapsed), collapsed.shape
        ))
    else:
        peak_array = np.asarray(peak_yx)
        if (
            peak_array.shape != (2,)
            or not np.issubdtype(peak_array.dtype, np.integer)
        ):
            raise TypeError("peak_yx must contain two integer coordinates")
        peak = tuple(int(index) for index in peak_array)

    half = kernel_size // 2
    if (
        peak[0] - half < 0
        or peak[0] + half >= values.shape[1]
        or peak[1] - half < 0
        or peak[1] + half >= values.shape[2]
    ):
        raise ValueError("empirical PSF crop extends outside the calibration cube")
    crop = values[
        :,
        peak[0] - half : peak[0] + half + 1,
        peak[1] - half : peak[1] + half + 1,
    ]
    border_width = min(2, half + 1)
    border = np.zeros((kernel_size, kernel_size), dtype=bool)
    border[:border_width] = border[-border_width:] = True
    border[:, :border_width] = border[:, -border_width:] = True
    kernels = np.zeros((indices.size, kernel_size, kernel_size), dtype=float)
    for output, channel in enumerate(indices):
        lower = max(channel - spectral_half_width, 0)
        upper = min(channel + spectral_half_width + 1, crop.shape[0])
        plane = np.nansum(crop[lower:upper], axis=0)
        plane -= np.nanmedian(plane[border])
        plane = np.maximum(np.nan_to_num(plane, nan=0.0), 0.0)
        mass = float(plane.sum())
        if mass <= np.finfo(float).tiny:
            raise ValueError(
                "empirical PSF channel {} has no positive flux".format(channel)
            )
        kernels[output] = plane / mass
    kernels, removed_centroids = recenter_psf_kernels(kernels)
    return kernels, removed_centroids, peak


def estimate_ifu_background(
    data,
    variance,
    *,
    valid_mask=None,
    source_mask=None,
    minimum_noise_scale=1.0,
):
    """Estimate per-channel blank-sky levels and a robust variance scale.

    The background is the median of valid, non-source spaxels in every
    channel. The reported noise scale is the pooled median absolute deviation
    of the background-subtracted, variance-whitened samples. It is bounded
    below by ``minimum_noise_scale`` so measured variances are not silently
    made smaller.

    This function estimates preprocessing quantities but does not modify or
    copy the input cube.
    """
    science = np.asarray(data)
    measured_variance = np.asarray(variance)
    if science.ndim != 3 or measured_variance.shape != science.shape:
        raise ValueError("data and variance must share shape (channel, y, x)")
    valid = (
        np.isfinite(science)
        & np.isfinite(measured_variance)
        & (measured_variance > 0)
    )
    if valid_mask is not None:
        declared_valid = np.asarray(valid_mask)
        if declared_valid.shape != science.shape or declared_valid.dtype != bool:
            raise ValueError("valid_mask must be a boolean array matching data")
        valid &= declared_valid
    if source_mask is not None:
        sources = np.asarray(source_mask)
        if sources.shape != science.shape[1:] or sources.dtype != bool:
            raise ValueError("source_mask must match the spatial image shape")
        valid &= ~sources[None]
    minimum_noise_scale = float(minimum_noise_scale)
    if not np.isfinite(minimum_noise_scale) or minimum_noise_scale <= 0:
        raise ValueError("minimum_noise_scale must be finite and positive")

    background = np.zeros(science.shape[0], dtype=float)
    for channel in range(science.shape[0]):
        selected = science[channel][valid[channel]]
        background[channel] = np.median(selected) if selected.size else 0.0
    residual = science - background[:, None, None]
    whitened = np.divide(
        residual,
        np.sqrt(measured_variance),
        out=np.full(science.shape, np.nan, dtype=float),
        where=valid,
    )
    selected = whitened[np.isfinite(whitened)]
    if selected.size:
        center = np.median(selected)
        scale = 1.4826 * np.median(np.abs(selected - center))
    else:
        scale = minimum_noise_scale
    if not np.isfinite(scale) or scale <= 0:
        scale = minimum_noise_scale
    return IFUBackgroundEstimate(
        background=background,
        noise_scale=max(float(scale), minimum_noise_scale),
        background_voxels=int(np.count_nonzero(valid)),
    )


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

"""Preprocessing utilities for wavelength-resolved IFU observations."""

from collections import namedtuple
import warnings

import numpy as np
from scipy.ndimage import label, shift
from scipy.special import ndtr
from scipy.sparse import csr_matrix


IFUBackgroundEstimate = namedtuple(
    "IFUBackgroundEstimate",
    ("background", "noise_scale", "background_voxels"),
)


def isolated_spatial_outlier_mask(
    data,
    variance,
    *,
    valid_mask=None,
    sigma=12.0,
    grow_sigma=3.0,
    max_spatial_pixels=12,
):
    """Flag compact high-significance islands in individual IFU slices.

    This is an opt-in reduction helper for resampled cubes whose detector-level
    outlier rejection left small positive or negative footprints behind.  A
    candidate island must contain a voxel above ``sigma``, remain connected at
    ``grow_sigma``, and cover no more than ``max_spatial_pixels`` in that
    wavelength slice.  Larger PSF-supported sources are therefore retained.

    Parameters
    ----------
    data, variance: array
        Matching ``(channel, y, x)`` background-subtracted science and variance.
    valid_mask: boolean array or None
        Samples eligible for testing.  Invalid samples are never newly flagged.
    sigma, grow_sigma: float
        Seed and connected-footprint significance thresholds.  ``sigma`` must
        be strictly larger than ``grow_sigma``.
    max_spatial_pixels: int
        Largest 8-connected footprint classified as an outlier.
    """

    values = np.asarray(data)
    measured_variance = np.asarray(variance)
    if values.ndim != 3 or measured_variance.shape != values.shape:
        raise ValueError("IFU data and variance must have matching 3D shapes")
    sigma = float(sigma)
    grow_sigma = float(grow_sigma)
    if (
        not np.isfinite(sigma)
        or not np.isfinite(grow_sigma)
        or grow_sigma <= 0
        or sigma <= grow_sigma
    ):
        raise ValueError("outlier sigma must exceed a positive grow sigma")
    if not isinstance(max_spatial_pixels, (int, np.integer)):
        raise TypeError("max_spatial_pixels must be an integer")
    max_spatial_pixels = int(max_spatial_pixels)
    if max_spatial_pixels <= 0:
        raise ValueError("max_spatial_pixels must be positive")

    finite = (
        np.isfinite(values)
        & np.isfinite(measured_variance)
        & (measured_variance > 0)
    )
    if valid_mask is None:
        valid = finite
    else:
        valid = np.asarray(valid_mask)
        if valid.shape != values.shape or valid.dtype != bool:
            raise ValueError("valid_mask must be boolean and match the IFU cube")
        valid = valid & finite

    significance = np.zeros(values.shape, dtype=float)
    significance[valid] = values[valid] / np.sqrt(measured_variance[valid])
    result = np.zeros(values.shape, dtype=bool)
    connectivity = np.ones((3, 3), dtype=int)
    for channel in range(values.shape[0]):
        channel_valid = valid[channel]
        for sign in (-1.0, 1.0):
            footprint = channel_valid & (
                sign * significance[channel] >= grow_sigma
            )
            labels, count = label(footprint, structure=connectivity)
            if count == 0:
                continue
            sizes = np.bincount(labels.ravel())
            seeds = np.unique(
                labels[
                    channel_valid
                    & (sign * significance[channel] >= sigma)
                ]
            )
            seeds = seeds[seeds != 0]
            compact = seeds[sizes[seeds] <= max_spatial_pixels]
            if compact.size:
                result[channel] |= np.isin(labels, compact)
    return result


class SpectralResponse:
    """Sparse, fixed mapping from a latent to an observed wavelength grid.

    ``indices[row]`` identifies the latent spectral samples contributing to
    one observed channel and ``weights[row]`` gives their flux-density
    weights.  Rows are padded with zero-weight entries so the response stays
    a small, human-auditable pair of two-dimensional arrays rather than a
    usually enormous dense matrix.
    """

    def __init__(self, indices, weights, model_channel_count):
        indices = np.asarray(indices)
        weights = np.asarray(weights)
        if indices.ndim != 2 or weights.shape != indices.shape:
            raise ValueError("spectral response indices and weights must match in 2D")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("spectral response indices must be integers")
        if not isinstance(model_channel_count, (int, np.integer)):
            raise TypeError("model_channel_count must be an integer")
        model_channel_count = int(model_channel_count)
        if model_channel_count <= 0:
            raise ValueError("model_channel_count must be positive")
        if indices.shape[0] == 0 or indices.shape[1] == 0:
            raise ValueError("a spectral response must contain at least one entry")
        if np.any(indices < 0) or np.any(indices >= model_channel_count):
            raise ValueError("spectral response index is outside the latent grid")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("spectral response weights must be finite and non-negative")
        row_sum = np.sum(weights, axis=1)
        if np.any(row_sum <= np.finfo(float).tiny):
            raise ValueError("every spectral response row must contain positive weight")
        if not np.allclose(row_sum, 1.0, rtol=1e-7, atol=1e-7):
            raise ValueError("spectral response weights must sum to one in every row")
        self.indices = indices.astype(int, copy=False)
        self.weights = weights
        self.model_channel_count = model_channel_count
        rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
        self.matrix = csr_matrix(
            (weights.ravel(), (rows, self.indices.ravel())),
            shape=(indices.shape[0], model_channel_count),
        )
        self.matrix.eliminate_zeros()

    @property
    def observation_channel_count(self):
        return self.indices.shape[0]


def _wavelength_values(wavelengths, unit=None):
    if hasattr(wavelengths, "unit"):
        resolved_unit = wavelengths.unit if unit is None else unit
        values = np.asarray(wavelengths.to_value(resolved_unit), dtype=float)
        return values, resolved_unit
    if unit is not None:
        raise TypeError("both wavelength grids must carry compatible units")
    return np.asarray(wavelengths, dtype=float), None


def _wavelength_bin_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("a wavelength grid must contain at least two samples")
    if np.any(~np.isfinite(centers)) or np.any(np.diff(centers) <= 0):
        raise ValueError("wavelength samples must be finite and strictly increasing")
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def binned_spectral_response(
    model_wavelengths,
    observed_wavelengths,
    dtype=None,
    *,
    extrapolate_edges=False,
):
    """Integrate latent flux density into observed wavelength bins.

    Both grids contain bin centers and must be strictly increasing.  The
    returned sparse response uses exact top-hat bin overlaps and preserves a
    constant flux density.  The latent grid must cover every observed bin;
    instrumental line-spread broadening, when known, belongs in a subsequent
    response rather than being silently guessed here.  ``extrapolate_edges``
    extends the nearest latent flux density only across a partially cropped
    first or last observed bin.
    """

    model, unit = _wavelength_values(model_wavelengths)
    observed, _ = _wavelength_values(observed_wavelengths, unit)
    model_edges = _wavelength_bin_edges(model)
    observed_edges = _wavelength_bin_edges(observed)
    scale = max(abs(observed_edges[0]), abs(observed_edges[-1]), 1.0)
    tolerance = 32 * np.finfo(float).eps * scale
    outside = (
        observed_edges[0] < model_edges[0] - tolerance
        or observed_edges[-1] > model_edges[-1] + tolerance
    )
    if outside and not extrapolate_edges:
        raise ValueError("latent wavelength bins do not cover the observation")

    rows = []
    row_weights = []
    maximum_support = 0
    for lower, upper in zip(observed_edges[:-1], observed_edges[1:]):
        clipped_lower = max(lower, model_edges[0])
        clipped_upper = min(upper, model_edges[-1])
        first = max(
            int(np.searchsorted(model_edges, clipped_lower, side="right")) - 1,
            0,
        )
        last = min(
            int(np.searchsorted(model_edges, clipped_upper, side="left")),
            model.size,
        )
        candidates = np.arange(first, last, dtype=int)
        overlap = np.maximum(
            0.0,
            np.minimum(clipped_upper, model_edges[candidates + 1])
            - np.maximum(clipped_lower, model_edges[candidates]),
        )
        selected = overlap > tolerance
        candidates = candidates[selected]
        overlap = overlap[selected]
        if extrapolate_edges and lower < model_edges[0]:
            missing = model_edges[0] - lower
            if candidates.size and candidates[0] == 0:
                overlap[0] += missing
            else:
                candidates = np.insert(candidates, 0, 0)
                overlap = np.insert(overlap, 0, missing)
        if extrapolate_edges and upper > model_edges[-1]:
            missing = upper - model_edges[-1]
            if candidates.size and candidates[-1] == model.size - 1:
                overlap[-1] += missing
            else:
                candidates = np.append(candidates, model.size - 1)
                overlap = np.append(overlap, missing)
        covered = float(np.sum(overlap))
        width = upper - lower
        if not np.isclose(covered, width, rtol=1e-9, atol=tolerance):
            raise ValueError("latent wavelength bins leave an observed bin uncovered")
        rows.append(candidates)
        row_weights.append(overlap / covered)
        maximum_support = max(maximum_support, candidates.size)

    resolved_dtype = np.dtype(float if dtype is None else dtype)
    indices = np.zeros((observed.size, maximum_support), dtype=int)
    weights = np.zeros((observed.size, maximum_support), dtype=resolved_dtype)
    for row, (selected_indices, selected_weights) in enumerate(zip(rows, row_weights)):
        count = selected_indices.size
        indices[row, :count] = selected_indices
        weights[row, :count] = selected_weights
    return SpectralResponse(indices, weights, model.size)


def selected_spectral_response(model_channel_count, indices, dtype=None):
    """Return an exact channel-selection response on a latent spectral grid."""

    selected = np.asarray(indices)
    if selected.ndim != 1:
        raise ValueError("selected spectral indices must be one-dimensional")
    weights = np.ones((selected.size, 1), dtype=float if dtype is None else dtype)
    return SpectralResponse(selected[:, None], weights, model_channel_count)


def gaussian_spectral_response(
    model_wavelengths,
    observed_wavelengths,
    fwhm,
    dtype=None,
    *,
    truncate=4.0,
):
    """Integrate latent flux-density bins through a Gaussian line response.

    ``fwhm`` supplies one line full width at half maximum per observed
    channel, in the same units as numeric wavelength inputs or as a compatible
    quantity. Each sparse row contains exact Gaussian-CDF integrals over the
    latent bin edges and is normalized after finite-tail and grid-edge
    truncation.
    """

    model, unit = _wavelength_values(model_wavelengths)
    observed, _ = _wavelength_values(observed_wavelengths, unit)
    if hasattr(fwhm, "unit"):
        widths = np.asarray(fwhm.to_value(unit), dtype=float)
    else:
        widths = np.asarray(fwhm, dtype=float)
    if widths.ndim == 0:
        widths = np.full(observed.shape, float(widths))
    if widths.shape != observed.shape:
        raise ValueError("fwhm must be scalar or match observed wavelengths")
    truncate = float(truncate)
    if (
        np.any(~np.isfinite(widths))
        or np.any(widths <= 0)
        or not np.isfinite(truncate)
        or truncate <= 0
    ):
        raise ValueError("Gaussian spectral widths and truncation must be positive")
    model_edges = _wavelength_bin_edges(model)
    sigma = widths / np.sqrt(8.0 * np.log(2.0))
    rows = []
    row_weights = []
    maximum_support = 0
    for center, scale in zip(observed, sigma):
        lower = center - truncate * scale
        upper = center + truncate * scale
        first = max(int(np.searchsorted(model_edges, lower, side="right")) - 1, 0)
        last = min(int(np.searchsorted(model_edges, upper, side="left")), model.size)
        indices = np.arange(first, last, dtype=int)
        weights = ndtr((model_edges[indices + 1] - center) / scale) - ndtr(
            (model_edges[indices] - center) / scale
        )
        selected = weights > np.finfo(float).eps
        indices = indices[selected]
        weights = weights[selected]
        mass = float(np.sum(weights))
        if mass <= np.finfo(float).tiny:
            raise ValueError("Gaussian spectral response misses the latent grid")
        rows.append(indices)
        row_weights.append(weights / mass)
        maximum_support = max(maximum_support, indices.size)

    resolved_dtype = np.dtype(float if dtype is None else dtype)
    indices = np.zeros((observed.size, maximum_support), dtype=int)
    weights = np.zeros((observed.size, maximum_support), dtype=resolved_dtype)
    for row, (selected_indices, selected_weights) in enumerate(zip(rows, row_weights)):
        count = selected_indices.size
        indices[row, :count] = selected_indices
        weights[row, :count] = selected_weights
    return SpectralResponse(indices, weights, model.size)


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

"""Deterministic, identifiable many-source IFU recovery contract.

This fixture is intentionally separate from ``ifu_parity_contracts``.  The
historical many-source cube is a stress test containing sources that are not
expected to be recovered accurately.  Every source in this cube is instead
declared recoverable, so tests may impose source-by-source truth gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift as image_shift
from scipy.signal import fftconvolve

from benchmarks.ifu_parity_contracts import gaussian_kernel


RECOVERY_SHAPE = (96, 33, 33)
RECOVERY_SEED = 8675309


@dataclass(frozen=True)
class RecoverySourceSpec:
    """One source in the many-source recovery scene."""

    name: str
    center: tuple[int, int]
    sigma_y: float
    sigma_x: float
    angle_deg: float
    support: int
    flux: float
    slope: float
    line_channels: tuple[float, float]
    clumpy: bool = False


RECOVERY_SOURCE_SPECS = (
    RecoverySourceSpec("lens", (16, 16), 2.8, 2.3, 22, 15, 12.0, -0.18, (19, 69)),
    RecoverySourceSpec("northwest", (9, 9), 1.25, 0.85, -28, 9, 5.2, 0.24, (11, 55)),
    RecoverySourceSpec("north", (9, 16), 1.05, 1.45, 8, 9, 4.6, -0.32, (27, 78)),
    RecoverySourceSpec("northeast", (9, 23), 0.85, 1.30, 35, 9, 4.1, 0.12, (37, 84), True),
    RecoverySourceSpec("west", (16, 9), 1.45, 0.95, 51, 9, 5.6, -0.08, (45, 8)),
    RecoverySourceSpec("east", (16, 23), 1.10, 1.55, -12, 9, 5.0, 0.31, (58, 31), True),
    RecoverySourceSpec("southwest", (23, 9), 1.30, 1.05, 17, 9, 4.8, -0.25, (66, 22)),
    RecoverySourceSpec("south", (23, 16), 0.95, 1.35, -41, 9, 4.3, 0.19, (74, 42), True),
    RecoverySourceSpec("southeast", (23, 23), 1.25, 0.90, 63, 9, 4.5, -0.02, (86, 49)),
    RecoverySourceSpec("inner", (13, 20), 0.80, 1.05, 26, 7, 3.8, 0.38, (33, 63)),
)


def _elliptical_image(spec):
    rows, columns = np.indices(RECOVERY_SHAPE[1:], dtype=float)
    dy = rows - spec.center[0]
    dx = columns - spec.center[1]
    angle = np.deg2rad(spec.angle_deg)
    major = np.cos(angle) * dy + np.sin(angle) * dx
    minor = -np.sin(angle) * dy + np.cos(angle) * dx
    image = np.exp(-0.5 * ((major / spec.sigma_y) ** 2 + (minor / spec.sigma_x) ** 2))

    if spec.clumpy:
        # The integer offsets sum to zero, preserving the declared centroid,
        # but are not a reflection-symmetric set.
        for offset_y, offset_x in ((-2, -1), (1, 2), (2, -1), (-1, 0)):
            knot = np.exp(
                -0.5
                * (
                    (rows - spec.center[0] - offset_y) ** 2
                    + (columns - spec.center[1] - offset_x) ** 2
                )
                / 0.42**2
            )
            image += 0.075 * image.sum() * knot / knot.sum()

    half = spec.support // 2
    image *= (
        (np.abs(rows - spec.center[0]) <= half)
        & (np.abs(columns - spec.center[1]) <= half)
    )
    return image / image.sum()


def recovery_morphologies():
    """Return source-ordered, unit-flux latent morphologies."""

    return np.asarray([_elliptical_image(spec) for spec in RECOVERY_SOURCE_SPECS])


def recovery_spectral_components():
    """Return source-ordered continuum and emission-line spectra."""

    channel = np.arange(RECOVERY_SHAPE[0], dtype=float)
    coordinate = channel / (RECOVERY_SHAPE[0] - 1) - 0.5
    continua = []
    lines = []
    for index, spec in enumerate(RECOVERY_SOURCE_SPECS):
        continuum = spec.flux * (
            1 + spec.slope * coordinate
            + 0.035 * np.sin(2 * np.pi * coordinate * (1 + index % 3) + 0.4 * index)
        )
        first = 1.45 * spec.flux * np.exp(
            -0.5 * ((channel - spec.line_channels[0]) / 1.55) ** 2
        )
        second = (0.65 + 0.06 * index) * spec.flux * np.exp(
            -0.5 * ((channel - spec.line_channels[1]) / 2.15) ** 2
        )
        continua.append(continuum)
        lines.append(first + second)
    return np.asarray(continua), np.asarray(lines)


def recovery_spectra():
    """Return distinct positive continua plus line complexes."""

    continuum, lines = recovery_spectral_components()
    return continuum + lines


def chromatic_line_morphologies():
    """Return offset/clumpy line maps that violate the rank-one scene model."""

    offsets = (
        (+1.0, -1.0),
        (+0.6, +0.8),
        (-0.8, +0.5),
        (+1.1, +0.4),
        (-0.5, -0.9),
        (+0.7, -1.1),
        (-1.0, +0.6),
        (+0.8, +0.8),
        (-0.6, -0.8),
        (+0.9, -0.5),
    )
    rows, columns = np.indices(RECOVERY_SHAPE[1:], dtype=float)
    result = []
    for spec, continuum, offset in zip(
        RECOVERY_SOURCE_SPECS, recovery_morphologies(), offsets
    ):
        line = image_shift(
            continuum,
            shift=offset,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        knot_center = np.asarray(spec.center, dtype=float) + 1.7 * np.asarray(offset)
        knot = np.exp(
            -0.5
            * (
                (rows - knot_center[0]) ** 2
                + (columns - knot_center[1]) ** 2
            )
            / 0.48**2
        )
        line += 0.22 * line.sum() * knot / knot.sum()
        half = spec.support // 2
        line *= (
            (np.abs(rows - spec.center[0]) <= half)
            & (np.abs(columns - spec.center[1]) <= half)
        )
        result.append(line / line.sum())
    return np.asarray(result)


def chromatic_source_cubes():
    """Return per-source cubes with different continuum and line morphologies."""

    continuum, lines = recovery_spectral_components()
    return (
        continuum[:, :, None, None] * recovery_morphologies()[:, None]
        + lines[:, :, None, None] * chromatic_line_morphologies()[:, None]
    )


def chromatic_latent_cube():
    return np.sum(chromatic_source_cubes(), axis=0)


def chromatic_noiseless_cube():
    return np.asarray(
        [
            fftconvolve(image, kernel, mode="same")
            for image, kernel in zip(chromatic_latent_cube(), recovery_psfs())
        ]
    )


def recovery_psfs():
    """Return a normalized wavelength-dependent instrument PSF."""

    sigmas = np.linspace(0.62, 1.12, RECOVERY_SHAPE[0])
    return np.asarray([gaussian_kernel(sigma=sigma, size=9) for sigma in sigmas])


def recovery_latent_cube():
    """Return the source-summed cube before PSF convolution."""

    return np.einsum("sc,syx->cyx", recovery_spectra(), recovery_morphologies())


def recovery_noiseless_cube():
    """Render the latent cube through the channel-dependent PSF."""

    return np.asarray(
        [
            fftconvolve(image, kernel, mode="same")
            for image, kernel in zip(recovery_latent_cube(), recovery_psfs())
        ]
    )


def _correlated_standard_noise(shape, generator):
    """Draw spatially and spectrally correlated, unit-RMS residual structure."""

    white = generator.normal(size=shape)
    spatial_kernel = gaussian_kernel(sigma=0.65, size=5)
    spatial = np.asarray(
        [fftconvolve(image, spatial_kernel, mode="same") for image in white]
    )
    correlated = np.empty_like(spatial)
    correlated[0] = spatial[0]
    rho = 0.42
    innovation = np.sqrt(1 - rho**2)
    for channel in range(1, shape[0]):
        correlated[channel] = rho * correlated[channel - 1] + innovation * spatial[channel]
    correlated -= np.mean(correlated)
    return correlated / np.std(correlated)


def _add_recovery_noise(truth, seed):
    """Apply the declared variance, covariance, and mask to one truth cube."""

    if not isinstance(seed, (int, np.integer)):
        raise ValueError("recovery noise seed must be an integer")
    peak = float(np.max(truth))
    channel = np.linspace(0, 1, RECOVERY_SHAPE[0])[:, None, None]
    rows, columns = np.indices(RECOVERY_SHAPE[1:], dtype=float)
    radius = np.hypot(rows - 16, columns - 16)[None] / np.hypot(16, 16)
    read_rms = 0.0085 * peak * (
        1 + 0.28 * np.sin(2 * np.pi * channel) ** 2 + 0.20 * radius
    )
    poisson_coefficient = (0.0045 * peak) ** 2 / peak
    variance = read_rms**2 + poisson_coefficient * np.maximum(truth, 0.0)

    generator = np.random.default_rng(seed)
    standardized = _correlated_standard_noise(truth.shape, generator)
    residual = standardized * np.sqrt(variance)
    data = truth + residual

    valid = np.ones(truth.shape, dtype=bool)
    valid[18:23, 2:7, 0:9] = False
    valid[57:61, 25:32, 24:32] = False
    valid[::19, 0:2, :] = False
    data[~valid] = 0.0
    return data, variance, valid, truth, residual


def recovery_noisy_cube(seed=RECOVERY_SEED):
    """Return data, diagonal variance, mask, truth, and injected residuals.

    The residual realization has spatial and spectral correlation even though
    the likelihood variance is diagonal.  This mirrors the modest covariance
    misspecification expected in reduced science cubes without making any
    source intrinsically unidentifiable.
    """

    return _add_recovery_noise(recovery_noiseless_cube(), seed)


def chromatic_noisy_cube(seed=RECOVERY_SEED):
    """Return the same realistic noise contract around chromatic source truth."""

    return _add_recovery_noise(chromatic_noiseless_cube(), seed)

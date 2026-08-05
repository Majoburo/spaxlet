"""Framework-neutral fixtures for the lisasep/Scarlet IFU parity gates.

The fixtures are generated rather than stored as opaque binary products.
Their dimensions, centers, random seeds, and source ordering are part of the
comparison contract and change only in a dedicated baseline update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


OPERATOR_SHAPE = (17, 19)
OPERATOR_CENTER = (5, 12)

DEBLEND_SHAPE = (17, 17)
N_CHANNELS = 6
WAVELENGTHS = np.linspace(1.0, 4.0, N_CHANNELS)
FEATURES = ("positivity", "monotonicity", "symmetry")
CASE_NAMES = (
    "compatible_null",
    "close_blend_helpful",
    "clumpy_misspecified",
)


MANY_SOURCE_SHAPE = (1024, 25, 25)


@dataclass(frozen=True)
class ManySourceSpec:
    """One readable row in the synthetic many-source catalog."""

    name: str
    center: tuple[float, float]
    sigma: float
    support: int
    flux: float
    slope: float
    line_fraction: float
    knot: tuple[float, float, float] | None = None


MANY_SOURCE_SPECS = (
    ManySourceSpec("wide", (12, 12), 3.4, 17, 8.0, -0.10, 0.46),
    ManySourceSpec("northwest", (7, 7), 1.2, 9, 3.5, +0.15, 0.18),
    ManySourceSpec("inner_north", (8, 10), 1.0, 9, 3.0, +0.15, 0.20),
    ManySourceSpec("northeast", (7, 17), 0.9, 7, 2.4, +0.05, 0.33),
    ManySourceSpec("east", (12, 17), 1.4, 9, 3.2, +0.20, 0.52, (-1.5, +1.5, 0.30)),
    ManySourceSpec("southeast", (17, 16), 1.1, 9, 1.8, -0.15, 0.65),
    ManySourceSpec("southwest", (17, 9), 1.5, 9, 2.0, +0.10, 0.78, (+1.5, -1.0, 0.25)),
    ManySourceSpec("west_faint", (12, 6), 0.9, 7, 0.05, -0.05, 0.86),
    ManySourceSpec("north_faint", (5, 13), 1.0, 7, 0.035, +0.25, 0.24),
)


def operator_morphologies():
    """Return the null, secondary-peak, and negative-control cases."""

    rows, columns = np.indices(OPERATOR_SHAPE, dtype=float)
    radius2 = (
        (rows - OPERATOR_CENTER[0]) ** 2
        + (columns - OPERATOR_CENTER[1]) ** 2
    )
    compatible = np.exp(-0.5 * radius2 / 2.3**2)

    secondary_peak = compatible.copy()
    secondary_peak[8, 8] += 2.5

    clumpy = compatible.copy()
    clumpy += 0.75 * np.exp(
        -0.5 * ((rows - 8.0) ** 2 + (columns - 14.5) ** 2) / 0.9**2
    )
    clumpy += 0.45 * np.exp(
        -0.5 * ((rows - 3.0) ** 2 + (columns - 7.0) ** 2) / 1.2**2
    )
    return {
        "compatible": compatible,
        "secondary_peak": secondary_peak,
        "clumpy_misspecified": clumpy,
    }


def gaussian_kernel(sigma=1.0, size=9):
    axis = np.arange(size, dtype=float) - (size - 1) / 2
    rows, columns = np.meshgrid(axis, axis, indexing="ij")
    value = np.exp(-0.5 * (rows**2 + columns**2) / sigma**2)
    return value / value.sum()


def gaussian_morphology(center, sigma=2.0):
    rows, columns = np.indices(DEBLEND_SHAPE, dtype=float)
    value = np.exp(
        -0.5
        * ((rows - center[0]) ** 2 + (columns - center[1]) ** 2)
        / sigma**2
    )
    return value / value.sum()


def spectra():
    """Return source-ordered spectra for all six synthetic channels."""

    return (
        2.0 * (WAVELENGTHS / WAVELENGTHS[0]) ** -0.45,
        1.7 * (WAVELENGTHS / WAVELENGTHS[0]) ** 0.55,
    )


def deblend_cases():
    """Return predeclared null, helpful, and misspecified fit cases."""

    cases = {}
    for name, separation, contamination, clumpy in (
        ("compatible_null", 8.0, 0.05, False),
        ("close_blend_helpful", 4.0, 0.35, False),
        ("clumpy_misspecified", 4.0, 0.35, True),
    ):
        centers = (
            (8.0, 8.0 - separation / 2),
            (8.0, 8.0 + separation / 2),
        )
        morphologies = [gaussian_morphology(center) for center in centers]
        if clumpy:
            rows, columns = np.indices(DEBLEND_SHAPE, dtype=float)
            knot = np.exp(
                -0.5
                * ((rows - 4.5) ** 2 + (columns - 5.5) ** 2)
                / 0.7**2
            )
            morphologies[0] = morphologies[0] + 0.35 * knot / knot.sum()
            morphologies[0] /= morphologies[0].sum()
        starts = tuple(
            (1.0 - contamination) * morphologies[index]
            + contamination * morphologies[1 - index]
            for index in range(2)
        )
        cases[name] = {
            "centers": centers,
            "morphologies": tuple(morphologies),
            "starts": starts,
        }
    return cases


def latent_cube(case):
    """Return the source-summed intrinsic cube before PSF convolution."""

    result = np.empty((N_CHANNELS,) + DEBLEND_SHAPE, dtype=float)
    for channel in range(N_CHANNELS):
        result[channel] = sum(
            source_spectrum[channel] * morphology
            for source_spectrum, morphology in zip(
                spectra(), case["morphologies"]
            )
        )
    return result


def noiseless_cube(case):
    """Render a case with a shared zero-padded shift-invariant PSF."""

    kernel = gaussian_kernel()
    return np.asarray(
        [fftconvolve(channel, kernel, mode="same") for channel in latent_cube(case)]
    )


def noisy_cube(case_name, noise_fraction=0.003):
    """Return deterministic data, scalar noise, and corresponding truth."""

    cases = deblend_cases()
    case_index = CASE_NAMES.index(case_name)
    truth = noiseless_cube(cases[case_name])
    noise = noise_fraction * float(np.max(truth))
    generator = np.random.default_rng(100 + case_index)
    data = truth + generator.normal(0.0, noise, truth.shape)
    return data, noise, truth


def _source_image(spec):
    """Return one finite-support morphology on the common spatial grid."""

    rows, columns = np.indices(MANY_SOURCE_SHAPE[1:], dtype=float)
    value = np.exp(
        -0.5
        * ((rows - spec.center[0]) ** 2 + (columns - spec.center[1]) ** 2)
        / spec.sigma**2
    )
    if spec.knot is not None:
        dy, dx, fraction = spec.knot
        knot = np.exp(
            -0.5
            * (
                (rows - spec.center[0] - dy) ** 2
                + (columns - spec.center[1] - dx) ** 2
            )
            / 0.55**2
        )
        value += fraction * value.sum() * knot / knot.sum()

    half = spec.support // 2
    inside = (
        (np.abs(rows - spec.center[0]) <= half)
        & (np.abs(columns - spec.center[1]) <= half)
    )
    value *= inside
    return value / value.sum()


def many_source_morphologies():
    """Return source-ordered unit-flux 2D morphologies."""

    return np.asarray([_source_image(spec) for spec in MANY_SOURCE_SPECS])


def many_source_spectra():
    """Return positive continua plus source-specific spectral lines."""

    channel = np.arange(MANY_SOURCE_SHAPE[0], dtype=float)
    centered_channel = channel / (MANY_SOURCE_SHAPE[0] - 1) - 0.5
    result = []
    for spec in MANY_SOURCE_SPECS:
        continuum = spec.flux * (1.0 + spec.slope * centered_channel)
        line_center = spec.line_fraction * (MANY_SOURCE_SHAPE[0] - 1)
        line_width = 0.018 * MANY_SOURCE_SHAPE[0]
        line = 0.8 * spec.flux * np.exp(
            -0.5 * ((channel - line_center) / line_width) ** 2
        )
        second_center = ((spec.line_fraction + 0.31) % 1.0) * (
            MANY_SOURCE_SHAPE[0] - 1
        )
        second_line = 0.25 * spec.flux * np.exp(
            -0.5 * ((channel - second_center) / (1.4 * line_width)) ** 2
        )
        result.append(continuum + line + second_line)
    return np.asarray(result)


def many_source_psfs():
    """Return a mildly broader normalized PSF at each later slice."""

    sigmas = np.linspace(0.65, 1.05, MANY_SOURCE_SHAPE[0])
    return np.asarray([gaussian_kernel(sigma=sigma, size=7) for sigma in sigmas])


def many_source_latent_cube():
    """Return the summed outer products before PSF convolution."""

    return np.einsum(
        "sc,syx->cyx",
        many_source_spectra(),
        many_source_morphologies(),
    )


def many_source_noiseless_cube():
    """Render the many-source fixture through its channel-dependent PSF."""

    return np.asarray(
        [
            fftconvolve(image, kernel, mode="same")
            for image, kernel in zip(
                many_source_latent_cube(), many_source_psfs()
            )
        ]
    )


def many_source_noisy_cube(
    noise_fraction=0.012,
    *,
    convolved=True,
    heterogeneous=False,
    masked=False,
):
    """Return seeded data, variance, validity mask, and noiseless truth."""

    truth = many_source_noiseless_cube() if convolved else many_source_latent_cube()
    base_rms = noise_fraction * float(np.max(truth))
    rms = np.full(truth.shape, base_rms)
    if heterogeneous:
        channel = np.linspace(0, 1, MANY_SOURCE_SHAPE[0])[:, None, None]
        rows, columns = np.indices(MANY_SOURCE_SHAPE[1:], dtype=float)
        radius = np.hypot(rows - 12, columns - 12)[None] / np.hypot(12, 12)
        rms *= (1 + 0.35 * np.sin(2 * np.pi * channel) ** 2) * (
            1 + 0.25 * radius
        )
    variance = np.broadcast_to(rms**2, truth.shape).copy()

    valid = np.ones(truth.shape, dtype=bool)
    if masked:
        valid[300:324] = False
        valid[::97, :3] = False
    generator = np.random.default_rng(231200899)
    data = truth + generator.normal(size=truth.shape) * np.sqrt(variance)
    data[~valid] = 0
    return data, variance, valid, truth

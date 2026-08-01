"""Framework-neutral fixtures for the lisasep/Scarlet IFU parity gates.

The fixtures are generated rather than stored as opaque binary products.
Their dimensions, centers, random seeds, and source ordering are part of the
comparison contract and change only in a dedicated baseline update.
"""

from __future__ import annotations

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


def noiseless_cube(case):
    """Render a case with a shared zero-padded shift-invariant PSF."""

    kernel = gaussian_kernel()
    result = np.empty((N_CHANNELS,) + DEBLEND_SHAPE, dtype=float)
    for channel in range(N_CHANNELS):
        latent = sum(
            source_spectrum[channel] * morphology
            for source_spectrum, morphology in zip(
                spectra(), case["morphologies"]
            )
        )
        result[channel] = fftconvolve(latent, kernel, mode="same")
    return result


def noisy_cube(case_name, noise_fraction=0.003):
    """Return deterministic data, scalar noise, and corresponding truth."""

    cases = deblend_cases()
    case_index = CASE_NAMES.index(case_name)
    truth = noiseless_cube(cases[case_name])
    noise = noise_fraction * float(np.max(truth))
    generator = np.random.default_rng(100 + case_index)
    data = truth + generator.normal(0.0, noise, truth.shape)
    return data, noise, truth

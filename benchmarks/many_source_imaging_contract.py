"""A sharper broadband imaging observation of the many-source recovery scene.

The detected-catalog arm showed that detection recovers every source position
exactly while the *support extents* are what the oracle catalog was really
supplying, and that getting them wrong costs a factor of ten in per-source
flux error.  Support extent is an angular-resolution question, so the natural
test is to observe the same latent scene again with a sharper PSF.

This fixture renders the identical truth morphologies and spectra into a few
broad bands through a narrow, wavelength-independent imaging PSF.  It shares
the pixel grid with the IFU deliberately: the information that constrains a
support comes from the PSF width, not from finer sampling, and reusing the
grid keeps the comparison to the IFU-only arms exact.  Nothing here is a
substitute for a genuinely finer detector, which would additionally need the
untested ``ResolutionRenderer`` path and real WCS objects on both frames.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from benchmarks.ifu_parity_contracts import gaussian_kernel
from benchmarks.many_source_recovery_contract import (
    RECOVERY_SEED,
    RECOVERY_SHAPE,
    recovery_morphologies,
    recovery_spectra,
)


# Three broad bands spanning the cube.  They are contiguous and cover every
# channel, so the imaging arm never sees a wavelength the IFU does not.
IMAGING_BAND_EDGES = ((0, 32), (32, 64), (64, 96))
IMAGING_CHANNELS = tuple("img{:d}".format(i) for i in range(len(IMAGING_BAND_EDGES)))

# The IFU PSF runs from sigma 0.62 to 1.12 pixels.  A sharper imaging PSF is
# the entire point of the arm; 0.32 is about half the narrowest IFU channel.
IMAGING_PSF_SIGMA = 0.32

# Imaging read noise as a fraction of the band peak.  Chosen so the arm tests
# resolution rather than depth: per-band surface brightness noise is close to
# the IFU's per-channel level rather than dramatically deeper.
IMAGING_READ_FRACTION = 0.0085


def imaging_psfs():
    """Return one narrow, wavelength-independent kernel per band."""

    return np.asarray(
        [
            gaussian_kernel(sigma=IMAGING_PSF_SIGMA, size=9)
            for _ in IMAGING_BAND_EDGES
        ]
    )


def imaging_band_spectra(spectra=None):
    """Return each source's mean spectral density inside every band."""

    if spectra is None:
        spectra = recovery_spectra()
    return np.asarray(
        [
            [np.mean(spectrum[start:stop]) for start, stop in IMAGING_BAND_EDGES]
            for spectrum in spectra
        ]
    )


def imaging_latent_bands():
    """Return the band images before the imaging PSF is applied."""

    return np.einsum("sb,syx->byx", imaging_band_spectra(), recovery_morphologies())


def imaging_noiseless_bands():
    """Render the latent bands through the narrow imaging PSF."""

    return np.asarray(
        [
            fftconvolve(image, kernel, mode="same")
            for image, kernel in zip(imaging_latent_bands(), imaging_psfs())
        ]
    )


def imaging_noisy_bands(seed=RECOVERY_SEED):
    """Return imaging data, diagonal variance, mask and truth.

    The imaging residual is drawn white.  The IFU fixture deliberately injects
    correlated residuals to mimic a reduced cube; keeping imaging white means
    this arm changes one thing at a time, namely the angular resolution.
    """

    truth = imaging_noiseless_bands()
    peak = float(np.max(truth))
    read_rms = IMAGING_READ_FRACTION * peak
    poisson_coefficient = (0.0045 * peak) ** 2 / peak
    variance = read_rms**2 + poisson_coefficient * np.maximum(truth, 0.0)

    generator = np.random.default_rng(seed + 1)
    data = truth + generator.normal(size=truth.shape) * np.sqrt(variance)
    valid = np.ones(truth.shape, dtype=bool)
    return data, variance, valid, truth


def imaging_shape():
    return (len(IMAGING_BAND_EDGES),) + RECOVERY_SHAPE[1:]

"""Signed per-source recovery metrics for the many-source contract.

Absolute per-source errors cannot distinguish ten independent errors from a
systematic redistribution of flux between overlapping factors.  The two-galaxy
benchmark established the redistribution case as the dangerous one, because the
scene residual is nearly blind to it.  These helpers keep the sign and add the
cancellation statistic that detects it.

They live in their own module so both the fitting driver and the standalone
diagnostic can use them without a circular import.
"""

from __future__ import annotations

import numpy as np

from benchmarks.many_source_recovery_contract import (
    RECOVERY_SHAPE,
    RECOVERY_SOURCE_SPECS,
    recovery_spectra,
)


SOURCE_NAMES = tuple(spec.name for spec in RECOVERY_SOURCE_SPECS)
BIN_COUNT = 8
LINE_HALF_WIDTH = 4


def line_windows(half_width=LINE_HALF_WIDTH):
    """Return a per-source boolean mask of the declared line channels."""

    channel = np.arange(RECOVERY_SHAPE[0], dtype=float)
    masks = []
    for spec in RECOVERY_SOURCE_SPECS:
        mask = np.zeros(RECOVERY_SHAPE[0], dtype=bool)
        for center in spec.line_channels:
            mask |= np.abs(channel - center) <= half_width
        masks.append(mask)
    return np.asarray(masks)


def signed_source_metrics(fitted_spectra, truth_spectra=None):
    """Return signed per-source spectral errors in absolute and relative form.

    Absolute differences are retained because only they are additive across
    sources; relative errors cannot be summed and so cannot test cancellation.
    The continuum/line split uses the contract's declared line windows rather
    than a continuum fitted to the spectrum being scored.
    """

    if truth_spectra is None:
        truth_spectra = recovery_spectra()
    difference = np.asarray(fitted_spectra) - truth_spectra
    mask = line_windows()

    total_difference = np.sum(difference, axis=1)
    total_truth = np.sum(truth_spectra, axis=1)
    line_difference = np.sum(np.where(mask, difference, 0.0), axis=1)
    continuum_difference = total_difference - line_difference
    line_truth = np.sum(np.where(mask, truth_spectra, 0.0), axis=1)
    continuum_truth = total_truth - line_truth

    edges = np.linspace(0, RECOVERY_SHAPE[0], BIN_COUNT + 1).astype(int)
    binned = np.asarray(
        [
            [
                np.sum(difference[source, edges[b] : edges[b + 1]])
                / np.sum(truth_spectra[source, edges[b] : edges[b + 1]])
                for b in range(BIN_COUNT)
            ]
            for source in range(len(SOURCE_NAMES))
        ]
    )

    return {
        "signed_total_absolute": total_difference,
        "signed_total_relative": total_difference / total_truth,
        "signed_continuum_relative": continuum_difference / continuum_truth,
        "signed_line_relative": line_difference / line_truth,
        "binned_signed_relative": binned,
        "truth_total": total_truth,
    }


def cancellation_ratio(signed_total_absolute):
    """Return ``|sum d| / sum |d|`` over the per-source flux differences.

    The data constrain the summed cube, so a redistribution between overlapping
    factors leaves the scene total nearly unchanged while individual sources
    move.  A ratio near zero therefore means the per-source errors are one
    redistribution; a ratio near one means they are independent.
    """

    signed_total_absolute = np.asarray(signed_total_absolute)
    gross = float(np.sum(np.abs(signed_total_absolute)))
    if gross == 0:
        return float("nan")
    return abs(float(np.sum(signed_total_absolute))) / gross

"""Exact pairwise ambiguity diagnostics for non-negative factorized sources.

For two factors, the shear

``m_receiver += delta * m_donor`` and
``s_donor -= delta * s_receiver``

leaves their summed latent cube unchanged. Non-negativity bounds both ends of
the feasible interval. The utilities below report those intervals and the
resulting per-source envelopes; they do not choose a preferred decomposition.
"""

from dataclasses import dataclass

import numpy as np

from .component import FactorizedComponent


@dataclass(frozen=True)
class MixingInterval:
    """Two-sided feasible interval for one exact pairwise mixing shear."""

    donor: int
    receiver: int
    delta_min: float
    delta_max: float


@dataclass(frozen=True)
class PairwiseMixingEnvelope:
    """Structural factor envelope over all finite one-shear endpoints.

    These ranges are exact sensitivity floors, not posterior intervals or
    ``+/-1 sigma`` errors. Simultaneous multi-source mixings can make the
    ambiguity wider when more than two sources participate.
    """

    component: int
    spectrum_lower: np.ndarray
    spectrum_upper: np.ndarray
    morphology_lower: np.ndarray
    morphology_upper: np.ndarray
    total_flux_min: float
    total_flux_max: float


def _factor_arrays(sources):
    sources = tuple(sources)
    if not sources:
        raise ValueError("mixing diagnostics require at least one source")
    if any(not isinstance(source, FactorizedComponent) for source in sources):
        raise TypeError("mixing diagnostics require FactorizedComponent sources")
    boxes = {
        (tuple(source.bbox.shape), tuple(source.bbox.origin)) for source in sources
    }
    if len(boxes) != 1:
        raise ValueError(
            "pairwise mixing currently requires sources on identical latent bounds"
        )

    spectra = []
    morphologies = []
    morphology_free = []
    for index, source in enumerate(sources):
        spectrum = np.asarray(source.spectrum.get_model(), dtype=float)
        morphology = np.asarray(source.morphology.get_model(), dtype=float)
        if spectrum.ndim != 1 or morphology.ndim != 2:
            raise ValueError("mixing diagnostics require 1-D by 2-D factors")
        if np.any(spectrum < -1e-12) or np.any(morphology < -1e-12):
            raise ValueError(
                "mixing diagnostics require non-negative factors; source {} "
                "contains a negative entry".format(index)
            )
        spectra.append(np.maximum(spectrum, 0.0))
        morphologies.append(np.maximum(morphology, 0.0))
        image_parameters = tuple(
            parameter
            for parameter in source.morphology.parameters
            if parameter.name in ("image", "coeffs")
        )
        morphology_free.append(
            bool(image_parameters) and any(not value.fixed for value in image_parameters)
        )
    return sources, spectra, morphologies, morphology_free


def bilinear_mixing_intervals(sources):
    """Return every exact two-sided pairwise interval.

    A fixed morphology may donate but cannot receive. Other morphology
    constraints are intentionally not interpreted: use this diagnostic for
    the free non-negative factor model, or treat its result as conditional on
    ignoring a stronger declared morphology family.
    """
    _, spectra, morphologies, morphology_free = _factor_arrays(sources)
    intervals = []
    for donor, (donor_spectrum, donor_morphology) in enumerate(
        zip(spectra, morphologies)
    ):
        for receiver, (receiver_spectrum, receiver_morphology) in enumerate(
            zip(spectra, morphologies)
        ):
            if donor == receiver or not morphology_free[receiver]:
                continue
            active_spectrum = receiver_spectrum > 0.0
            upper = (
                float(
                    np.min(
                        donor_spectrum[active_spectrum]
                        / receiver_spectrum[active_spectrum]
                    )
                )
                if np.any(active_spectrum)
                else np.inf
            )
            active_morphology = donor_morphology > 0.0
            lower = (
                -float(
                    np.min(
                        receiver_morphology[active_morphology]
                        / donor_morphology[active_morphology]
                    )
                )
                if np.any(active_morphology)
                else 0.0
            )
            intervals.append(
                MixingInterval(
                    donor=donor,
                    receiver=receiver,
                    delta_min=lower,
                    delta_max=upper,
                )
            )
    return tuple(intervals)


def pairwise_mixing_envelopes(sources):
    """Envelope integrated spectra and unit-flux morphologies.

    The supplied decomposition and every finite endpoint of every ordered-pair
    interval are included. Every candidate renders exactly the same summed
    latent cube up to floating-point roundoff.
    """
    _, spectra, morphologies, _ = _factor_arrays(sources)
    spectrum_candidates = [
        [spectrum * float(np.sum(morphology))]
        for spectrum, morphology in zip(spectra, morphologies)
    ]
    morphology_candidates = []
    for morphology in morphologies:
        total = float(np.sum(morphology))
        morphology_candidates.append(
            [
                morphology / total
                if total > np.finfo(float).tiny
                else np.zeros_like(morphology)
            ]
        )

    for interval in bilinear_mixing_intervals(sources):
        endpoints = [interval.delta_min]
        if np.isfinite(interval.delta_max):
            endpoints.append(interval.delta_max)
        for delta in endpoints:
            if abs(delta) <= np.finfo(float).eps:
                continue
            trial_spectra = [value.copy() for value in spectra]
            trial_morphologies = [value.copy() for value in morphologies]
            trial_spectra[interval.donor] -= (
                delta * trial_spectra[interval.receiver]
            )
            trial_morphologies[interval.receiver] += (
                delta * trial_morphologies[interval.donor]
            )
            trial_spectra[interval.donor] = np.maximum(
                trial_spectra[interval.donor], 0.0
            )
            trial_morphologies[interval.receiver] = np.maximum(
                trial_morphologies[interval.receiver], 0.0
            )
            for index, (spectrum, morphology) in enumerate(
                zip(trial_spectra, trial_morphologies)
            ):
                total = float(np.sum(morphology))
                spectrum_candidates[index].append(spectrum * total)
                morphology_candidates[index].append(
                    morphology / total
                    if total > np.finfo(float).tiny
                    else np.zeros_like(morphology)
                )

    envelopes = []
    for index, (spectral, spatial) in enumerate(
        zip(spectrum_candidates, morphology_candidates)
    ):
        spectral_stack = np.stack(spectral)
        spatial_stack = np.stack(spatial)
        totals = np.sum(spectral_stack, axis=1)
        envelopes.append(
            PairwiseMixingEnvelope(
                component=index,
                spectrum_lower=np.min(spectral_stack, axis=0),
                spectrum_upper=np.max(spectral_stack, axis=0),
                morphology_lower=np.min(spatial_stack, axis=0),
                morphology_upper=np.max(spatial_stack, axis=0),
                total_flux_min=float(np.min(totals)),
                total_flux_max=float(np.max(totals)),
            )
        )
    return tuple(envelopes)


__all__ = [
    "MixingInterval",
    "PairwiseMixingEnvelope",
    "bilinear_mixing_intervals",
    "pairwise_mixing_envelopes",
]

"""Exact structural ambiguity tests for factorized Scarlet sources."""

import numpy as np

import scarlet


def _sources(spectra, morphologies):
    shape = (len(spectra[0]),) + morphologies[0].shape
    frame = scarlet.Frame(
        shape,
        psf=scarlet.DeltaPSF(shape[0]),
        channels=np.arange(shape[0]),
    )
    return [
        scarlet.FactorizedComponent(
            frame,
            scarlet.TabulatedSpectrum(frame, np.asarray(spectrum, dtype=float)),
            scarlet.ImageMorphology(
                frame, np.asarray(morphology, dtype=float), resizing=False
            ),
        )
        for spectrum, morphology in zip(spectra, morphologies)
    ]


class TestPairwiseMixingDiagnostics(object):
    def test_intervals_reach_exact_nonnegative_boundaries(self):
        sources = _sources(
            ([2.0, 4.0], [1.0, 2.0]),
            (np.ones((5, 5)), 0.25 * np.ones((5, 5))),
        )
        interval = [
            value
            for value in scarlet.bilinear_mixing_intervals(sources)
            if value.donor == 0 and value.receiver == 1
        ][0]
        assert abs(interval.delta_min + 0.25) < 1e-12
        assert abs(interval.delta_max - 2.0) < 1e-12

    def test_every_endpoint_preserves_the_latent_model(self):
        morphology = np.arange(1, 26, dtype=float).reshape(5, 5)
        sources = _sources(
            ([2.0, 4.0, 3.0], [1.0, 2.0, 5.0]),
            (morphology, np.flip(morphology) + 2.0),
        )
        spectra = [source.spectrum.get_model().copy() for source in sources]
        morphologies = [
            source.morphology.get_model().copy() for source in sources
        ]
        original = sum(
            spectrum[:, None, None] * spatial
            for spectrum, spatial in zip(spectra, morphologies)
        )
        for interval in scarlet.bilinear_mixing_intervals(sources):
            for delta in (interval.delta_min, interval.delta_max):
                if not np.isfinite(delta):
                    continue
                changed_spectra = [value.copy() for value in spectra]
                changed_morphologies = [value.copy() for value in morphologies]
                changed_spectra[interval.donor] -= (
                    delta * changed_spectra[interval.receiver]
                )
                changed_morphologies[interval.receiver] += (
                    delta * changed_morphologies[interval.donor]
                )
                changed = sum(
                    spectrum[:, None, None] * spatial
                    for spectrum, spatial in zip(
                        changed_spectra, changed_morphologies
                    )
                )
                np.testing.assert_allclose(changed, original, atol=1e-12)

    def test_envelopes_contain_the_supplied_factors(self):
        morphology = np.arange(1, 26, dtype=float).reshape(5, 5)
        sources = _sources(
            ([2.0, 4.0, 3.0], [1.0, 2.0, 5.0]),
            (morphology, np.flip(morphology) + 2.0),
        )
        envelopes = scarlet.pairwise_mixing_envelopes(sources)
        for source, envelope in zip(sources, envelopes):
            integrated = (
                source.spectrum.get_model()
                * float(np.sum(source.morphology.get_model()))
            )
            assert np.all(envelope.spectrum_lower <= integrated + 1e-12)
            assert np.all(envelope.spectrum_upper >= integrated - 1e-12)
            assert envelope.total_flux_min <= float(np.sum(integrated))
            assert envelope.total_flux_max >= float(np.sum(integrated))

    def test_negative_factors_are_rejected(self):
        sources = _sources(
            ([1.0, 2.0], [1.0, 2.0]),
            (np.ones((3, 3)), np.ones((3, 3))),
        )
        sources[0].spectrum.parameters[0][0] = -1.0
        try:
            scarlet.bilinear_mixing_intervals(sources)
        except ValueError:
            pass
        else:
            raise AssertionError("negative spectra were accepted")

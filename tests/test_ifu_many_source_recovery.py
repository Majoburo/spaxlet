"""Truth gates for the identifiable many-source IFU recovery cube."""

import os
from functools import lru_cache
import unittest

import numpy as np

import spaxlet
from benchmarks.many_source_recovery_contract import (
    RECOVERY_SHAPE,
    RECOVERY_SOURCE_SPECS,
    recovery_latent_cube,
    recovery_morphologies,
    recovery_noiseless_cube,
    recovery_noisy_cube,
    recovery_psfs,
    recovery_spectra,
)
from benchmarks.run_synthetic_many_source_recovery import (
    RECOVERY_CONVERGED_MAX_ITER,
    RECOVERY_PILOT_MAX_ITER,
    fit_recovery_cube,
)


@lru_cache(maxsize=None)
def _fit(start):
    return fit_recovery_cube(start=start, max_iter=RECOVERY_PILOT_MAX_ITER)


@lru_cache(maxsize=None)
def _fit_seed(seed):
    return fit_recovery_cube(
        start="A", max_iter=RECOVERY_PILOT_MAX_ITER, seed=seed
    )


@lru_cache(maxsize=None)
def _converged_fit():
    return fit_recovery_cube(start="A", max_iter=RECOVERY_CONVERGED_MAX_ITER)


class ManySourceRecoveryTest(unittest.TestCase):
    def test_fixture_is_nontrivial_but_every_source_is_detectable(self):
        spectra = recovery_spectra()
        morphologies = recovery_morphologies()
        data, variance, valid, truth, injected_residual = recovery_noisy_cube()

        self.assertEqual(len(RECOVERY_SOURCE_SPECS), 10)
        self.assertEqual(spectra.shape, (10, RECOVERY_SHAPE[0]))
        self.assertEqual(morphologies.shape, (10,) + RECOVERY_SHAPE[1:])
        np.testing.assert_allclose(morphologies.sum(axis=(1, 2)), 1.0)
        self.assertTrue(np.all(spectra > 0))
        self.assertGreater(np.max(variance), 2.0 * np.min(variance))
        self.assertGreater(np.count_nonzero(~valid), 400)
        np.testing.assert_array_equal(data[~valid], 0.0)
        np.testing.assert_allclose(truth, recovery_noiseless_cube())

        source_peaks = np.max(
            spectra[:, :, None, None] * morphologies[:, None], axis=(2, 3)
        )
        representative_rms = np.median(np.sqrt(variance), axis=(1, 2))
        integrated_peak_snr = np.sqrt(
            np.sum((source_peaks / representative_rms[None]) ** 2, axis=1)
        )
        self.assertTrue(np.all(integrated_peak_snr > 100))

        standardized = injected_residual / np.sqrt(variance)
        spectral_lag = np.corrcoef(
            standardized[:-1].ravel(), standardized[1:].ravel()
        )[0, 1]
        spatial_lag = np.corrcoef(
            standardized[:, :, :-1].ravel(), standardized[:, :, 1:].ravel()
        )[0, 1]
        self.assertGreater(spectral_lag, 0.25)
        self.assertGreater(spatial_lag, 0.20)

    def test_channel_dependent_psf_forward_model_is_exact(self):
        channels = tuple(range(RECOVERY_SHAPE[0]))
        frame = spaxlet.Frame(
            RECOVERY_SHAPE,
            channels=channels,
            psf=spaxlet.DeltaPSF(RECOVERY_SHAPE[0]),
        )
        observation = spaxlet.Observation(
            np.zeros(RECOVERY_SHAPE),
            channels=channels,
            psf=spaxlet.ImagePSF(recovery_psfs()),
            weights=np.ones(RECOVERY_SHAPE),
        ).match(frame)
        sources = [
            spaxlet.FactorizedComponent(
                frame,
                spaxlet.TabulatedSpectrum(frame, spectrum),
                spaxlet.ImageMorphology(frame, morphology, resizing=False),
            )
            for spectrum, morphology in zip(recovery_spectra(), recovery_morphologies())
        ]
        rendered = np.asarray(
            observation.render(sum(source.get_model(frame=frame) for source in sources))
        )
        np.testing.assert_allclose(rendered, recovery_noiseless_cube(), atol=2e-7)
        np.testing.assert_allclose(
            recovery_latent_cube(),
            np.einsum("sc,syx->cyx", recovery_spectra(), recovery_morphologies()),
        )

    def test_all_declared_starts_recover_every_source(self):
        for result in (_fit("A"), _fit("B"), _fit("C")):
            self.assertGreater(result.chi2_per_valid_voxel, 0.85)
            self.assertLess(result.chi2_per_valid_voxel, 1.05)
            self.assertLess(result.relative_projected_gradient, 4e-4)
            self.assertLess(np.max(result.spectrum_relative_l2), 0.085)
            self.assertGreater(np.min(result.spectrum_cosine), 0.999)
            self.assertLess(np.max(result.morphology_relative_l2), 0.22)
            self.assertGreater(np.min(result.morphology_cosine), 0.975)
            self.assertLess(np.max(result.centroid_error_px), 1e-7)
            self.assertLess(np.max(result.integrated_flux_relative_error), 0.085)
            self.assertGreater(np.min(result.morphology_identity_margin), 0.80)

    def test_truth_independent_best_loss_selection_also_passes_truth_gates(self):
        results = [_fit(start) for start in "ABC"]
        selected = min(results, key=lambda result: result.chi2_per_valid_voxel)
        self.assertLess(np.max(selected.spectrum_relative_l2), 0.08)
        self.assertLess(np.max(selected.morphology_relative_l2), 0.21)
        self.assertLess(np.max(selected.integrated_flux_relative_error), 0.08)
        np.testing.assert_allclose(
            selected.fitted_morphologies.sum(axis=(1, 2)), 1.0, atol=1e-12
        )
        np.testing.assert_allclose(selected.data - selected.model, selected.residual)

    def test_per_source_flux_errors_are_a_redistribution_not_independent(self):
        """The signed per-source errors cancel, so they share one cause.

        Absolute errors alone would report ten sources inside the flux gate and
        say nothing about this.  The data constrain the summed cube, so a
        near-zero cancellation ratio means flux moved between overlapping
        factors rather than each source being independently mismeasured.
        """

        for result in (_fit("A"), _fit("B"), _fit("C")):
            self.assertLess(result.flux_cancellation_ratio, 0.15)
            np.testing.assert_allclose(
                np.abs(result.signed_integrated_flux_relative_error),
                result.integrated_flux_relative_error,
            )

    def test_broad_central_source_gains_flux_from_its_neighbours(self):
        """Pin the direction and achromaticity of the known transfer.

        ``lens`` is the broad central factor every other source overlaps.  It
        gains flux in every start and every noise draw, in all eight
        wavelength bins, while the majority of the remaining sources lose it.
        The bias is a flux transfer, not a spectral-shape error, so it is not
        reduced by adding channels.
        """

        lens = tuple(spec.name for spec in RECOVERY_SOURCE_SPECS).index("lens")
        for result in (_fit("A"), _fit("B"), _fit("C"), _fit_seed(104729)):
            signed = result.signed_integrated_flux_relative_error
            self.assertGreater(signed[lens], 0.03)
            self.assertGreater(np.count_nonzero(signed < 0), 5)
            self.assertTrue(np.all(result.binned_signed_flux_relative_error[lens] > 0))

    def test_recovery_is_not_specific_to_one_correlated_noise_draw(self):
        for result in (_fit_seed(104729), _fit_seed(130363)):
            self.assertLess(result.relative_projected_gradient, 4e-4)
            self.assertLess(np.max(result.spectrum_relative_l2), 0.10)
            self.assertLess(np.max(result.morphology_relative_l2), 0.22)
            self.assertLess(np.max(result.integrated_flux_relative_error), 0.10)
            self.assertGreater(np.min(result.morphology_identity_margin), 0.80)

    def test_pilot_budget_is_stationary_by_the_gate_but_not_converged(self):
        """Document that the pilot stationarity gate does not certify flux.

        Every pilot fit satisfies ``relative_projected_gradient < 4e-4`` while
        sitting two orders of magnitude above the converged value, and its
        per-source fluxes are still moving.  The gate therefore states that
        the optimizer is making little progress, not that the reported fluxes
        are the ones the model implies.
        """

        for result in (_fit("A"), _fit("B"), _fit("C")):
            self.assertLess(result.relative_projected_gradient, 4e-4)
            self.assertGreater(result.relative_projected_gradient, 5e-5)

    @unittest.skipUnless(
        os.environ.get("SPAXLET_CONVERGED_RECOVERY"),
        "converged budget costs about 11x the pilot fit",
    )
    def test_converged_budget_separates_transient_from_bias(self):
        """Separate the optimization transient from the estimation bias.

        On the noiseless cube the maximum signed error falls to 0.024% by
        6,000 iterations, so the model and the oracle catalog do identify
        every source.  On the noisy cube it plateaus near 6% instead, which
        is a genuine converged bias and not a budget artifact.
        """

        result = _converged_fit()
        self.assertLess(result.relative_projected_gradient, 2e-5)
        self.assertLess(result.flux_cancellation_ratio, 0.15)

        lens = tuple(spec.name for spec in RECOVERY_SOURCE_SPECS).index("lens")
        signed = result.signed_integrated_flux_relative_error
        self.assertGreater(signed[lens], 0.04)
        self.assertLess(signed[lens], 0.09)

        pilot = _fit("A").signed_integrated_flux_relative_error[lens]
        self.assertGreater(pilot, signed[lens])


if __name__ == "__main__":
    unittest.main()

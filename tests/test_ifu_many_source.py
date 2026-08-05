"""Small synthetic analogue of the many-source SPT0311 IFU fit."""

from functools import lru_cache
import unittest

import numpy as np

import spaxlet
from benchmarks.ifu_parity_contracts import (
    MANY_SOURCE_SHAPE,
    MANY_SOURCE_SPECS,
    many_source_latent_cube,
    many_source_morphologies,
    many_source_noiseless_cube,
    many_source_noisy_cube,
    many_source_psfs,
    many_source_spectra,
)
from benchmarks.run_synthetic_many_source_ifu import fit_joint_model


def _render(frame, observation, spectra, morphologies):
    sources = [
        spaxlet.FactorizedComponent(
            frame,
            spaxlet.TabulatedSpectrum(frame, spectrum),
            spaxlet.ImageMorphology(frame, morphology, resizing=False),
        )
        for spectrum, morphology in zip(spectra, morphologies)
    ]
    latent = sum(source.get_model(frame=frame) for source in sources)
    return np.asarray(observation.render(latent))


@lru_cache(maxsize=None)
def _fit(
    slices=1024,
    support="correct",
    constraint="centroid",
    start=0,
    spectral_smoothness_strength=0,
    spatial_smoothness_strength=0,
):
    return fit_joint_model(
        slices=slices,
        support=support,
        constraint=constraint,
        start=start,
        max_iter=40,
        spectral_smoothness_strength=spectral_smoothness_strength,
        spatial_smoothness_strength=spatial_smoothness_strength,
    )


class ManySourceIFUTest(unittest.TestCase):
    def setUp(self):
        channels = tuple(range(MANY_SOURCE_SHAPE[0]))
        self.frame = spaxlet.Frame(
            MANY_SOURCE_SHAPE,
            channels=channels,
            psf=spaxlet.DeltaPSF(MANY_SOURCE_SHAPE[0]),
        )
        self.observation = spaxlet.Observation(
            np.zeros(MANY_SOURCE_SHAPE),
            channels=channels,
            psf=spaxlet.ImagePSF(many_source_psfs()),
            weights=np.ones(MANY_SOURCE_SHAPE),
        ).match(self.frame)

    def test_fixture_has_many_rank_one_sources_and_spatial_variety(self):
        spectra = many_source_spectra()
        morphologies = many_source_morphologies()

        self.assertEqual(len(MANY_SOURCE_SPECS), 9)
        self.assertEqual(MANY_SOURCE_SHAPE[0], 1024)
        self.assertEqual(spectra.shape, (9, 1024))
        self.assertEqual(morphologies.shape, (9,) + MANY_SOURCE_SHAPE[1:])
        np.testing.assert_allclose(morphologies.sum(axis=(1, 2)), 1)
        self.assertTrue(np.all(spectra > 0))

        rows, columns = np.indices(MANY_SOURCE_SHAPE[1:], dtype=float)
        areas = []
        for spec, morphology in zip(MANY_SOURCE_SPECS, morphologies):
            areas.append(
                np.sum(
                    morphology
                    * ((rows - spec.center[0]) ** 2 + (columns - spec.center[1]) ** 2)
                )
            )
        self.assertGreater(areas[0], 4 * np.median(areas[1:]))

        def local_cutout(index):
            spec = MANY_SOURCE_SPECS[index]
            half = spec.support // 2
            y0 = int(spec.center[0]) - half
            x0 = int(spec.center[1]) - half
            return morphologies[
                index, y0 : y0 + spec.support, x0 : x0 + spec.support
            ]

        wide = local_cutout(0)
        asymmetric = local_cutout(4)
        self.assertLess(np.linalg.norm(wide - np.flip(wide)), 1e-14)
        self.assertGreater(np.linalg.norm(asymmetric - np.flip(asymmetric)), 0.05)

        wide_symmetry = spaxlet.SymmetryConstraint(center=(8, 8))(wide.copy(), 0)
        asymmetric_symmetry = spaxlet.SymmetryConstraint(center=(4, 4))(
            asymmetric.copy(), 0
        )
        self.assertLess(np.linalg.norm(wide_symmetry - wide), 1e-14)
        self.assertGreater(np.linalg.norm(asymmetric_symmetry - asymmetric), 0.02)

        _, variance, _, _ = many_source_noisy_cube(convolved=False)
        source_peaks = np.max(
            spectra[:, :, None, None] * morphologies[:, None], axis=(2, 3)
        )
        per_slice_peak_snr = np.median(
            source_peaks / np.sqrt(variance[:, 0, 0]), axis=1
        )
        integrated_peak_snr = np.sqrt(
            np.sum(source_peaks**2 / variance[:, 0, 0], axis=1)
        )
        self.assertTrue(np.all(per_slice_peak_snr[-2:] < 1))
        self.assertTrue(np.all(integrated_peak_snr[-2:] > 15))

    def test_spaxlet_forward_model_matches_the_declared_cube(self):
        rendered = _render(
            self.frame,
            self.observation,
            many_source_spectra(),
            many_source_morphologies(),
        )
        np.testing.assert_allclose(rendered, many_source_noiseless_cube(), atol=2e-7)
        np.testing.assert_allclose(
            many_source_latent_cube(),
            np.einsum(
                "sc,syx->cyx",
                many_source_spectra(),
                many_source_morphologies(),
            ),
        )

    def test_wide_support_and_nonuniform_noise_are_explicit_controls(self):
        _, variance, valid, _ = many_source_noisy_cube(
            heterogeneous=True, masked=True
        )
        self.assertGreater(np.max(variance), 2 * np.min(variance))
        self.assertGreater(np.count_nonzero(~valid), 20 * 25 * 25)

        wide = many_source_morphologies()[0]
        center = np.asarray(MANY_SOURCE_SPECS[0].center, dtype=int)
        rows, columns = np.indices(wide.shape)
        narrow_mask = (
            (np.abs(rows - center[0]) <= 4)
            & (np.abs(columns - center[1]) <= 4)
        )
        outside_fraction = 1 - np.sum(wide[narrow_mask])
        self.assertGreater(outside_fraction, 0.25)

    def test_1024_colors_recover_faint_shared_morphologies_better(self):
        shallow = _fit(slices=16, constraint="selective_symmetry")
        full = _fit(slices=1024, constraint="selective_symmetry")
        self.assertLess(
            np.mean(full.morphology_relative_l2[-2:]),
            0.92 * np.mean(shallow.morphology_relative_l2[-2:]),
        )
        self.assertLess(full.chi2_per_voxel, 1.06)

    def test_correct_support_beats_truncation_and_oversizing(self):
        truncated = _fit(support="truncated")
        correct = _fit(support="correct")
        oversized = _fit(support="oversized")
        self.assertLess(correct.chi2_per_voxel, 0.9 * truncated.chi2_per_voxel)
        self.assertLess(correct.chi2_per_voxel, oversized.chi2_per_voxel)

    def test_selective_symmetry_has_the_best_joint_residual(self):
        positivity = _fit(constraint="positivity")
        centroid = _fit(constraint="centroid")
        selective = _fit(constraint="selective_symmetry")
        global_symmetry = _fit(constraint="global_symmetry")
        self.assertLess(selective.chi2_per_voxel, positivity.chi2_per_voxel)
        self.assertLess(selective.chi2_per_voxel, centroid.chi2_per_voxel)
        self.assertLess(
            selective.chi2_per_voxel, 0.9 * global_symmetry.chi2_per_voxel
        )

    def test_spectral_smoothness_improves_shape_without_spending_fit(self):
        baseline = _fit()
        smoothed = _fit(spectral_smoothness_strength=50)

        self.assertGreater(
            np.mean(smoothed.spectrum_cosine),
            np.mean(baseline.spectrum_cosine) + 0.004,
        )
        self.assertLess(
            np.mean(smoothed.spectrum_curvature_rms),
            0.75 * np.mean(baseline.spectrum_curvature_rms),
        )
        self.assertLess(
            smoothed.chi2_per_voxel,
            baseline.chi2_per_voxel + 0.001,
        )

    def test_reported_factors_use_the_physical_unit_flux_gauge(self):
        result = _fit(constraint="selective_symmetry")

        np.testing.assert_allclose(
            result.fitted_morphologies.sum(axis=(1, 2)), 1, atol=1e-12
        )
        np.testing.assert_allclose(
            np.einsum(
                "sc,syx->cyx", result.fitted_spectra, result.fitted_morphologies
            ),
            result.model,
            rtol=2e-7,
            atol=2e-7,
        )
        np.testing.assert_allclose(result.data - result.model, result.residual)

    def test_joint_result_is_selected_over_two_deterministic_starts(self):
        start_zero = _fit(constraint="selective_symmetry", start=0)
        start_one = _fit(constraint="selective_symmetry", start=1)
        best = min((start_zero, start_one), key=lambda result: result.chi2_per_voxel)
        self.assertLess(best.chi2_per_voxel, 1.06)
        for result in (start_zero, start_one):
            self.assertLess(
                result.chi2_per_voxel, 0.25 * result.initial_chi2_per_voxel
            )


if __name__ == "__main__":
    unittest.main()

"""Tests for framework-neutral strict IFU comparison metrics."""

import unittest

import numpy as np

from benchmarks.collaborator_metrics import (
    centroid,
    morphology_metrics,
    normalized_morphology,
    residual_metrics,
    spectral_metrics,
    start_sensitivity,
    translate_morphology,
)


class IFUParityMetrics(unittest.TestCase):
    def test_spectral_metrics_are_scale_sensitive(self):
        wavelength = np.linspace(1.0, 4.0, 6)
        truth = np.linspace(1.0, 2.0, 6)
        score = spectral_metrics(1.1 * truth, truth, wavelength, n_bin=3)
        self.assertAlmostEqual(score["relative_l2"], 0.1)
        self.assertAlmostEqual(score["cosine"], 1.0)
        self.assertAlmostEqual(score["integrated_flux_error"], 0.1)
        np.testing.assert_allclose(score["binned_fractional_error"], 0.1)

    def test_morphology_metrics_use_unit_flux_and_structure(self):
        truth = np.zeros((17, 17))
        truth[8, 8] = 1.0
        self.assertAlmostEqual(
            morphology_metrics(7.0 * truth, truth)["relative_l2"], 0.0
        )
        shifted = np.zeros_like(truth)
        shifted[8, 10] = 3.0
        score = morphology_metrics(shifted, truth)
        self.assertAlmostEqual(score["centroid_error_px"], 2.0)
        self.assertGreater(score["structured_relative_l2"], 0.0)
        self.assertLess(score["cosine"], 1.0)

    def test_morphology_validation_rejects_negative_or_empty_arrays(self):
        with self.assertRaises(ValueError):
            normalized_morphology(np.zeros((3, 3)))
        invalid = np.ones((3, 3))
        invalid[0, 0] = -1.0
        with self.assertRaises(ValueError):
            normalized_morphology(invalid)

    def test_translate_morphology_uses_declared_subpixel_offset(self):
        truth = np.zeros((7, 9))
        truth[3, 4] = 1.0
        translated = translate_morphology(truth, (0.25, -0.5))
        self.assertAlmostEqual(float(np.sum(translated)), 1.0)
        np.testing.assert_allclose(centroid(translated), (3.25, 3.5))
        expected = np.zeros_like(truth)
        expected[3, 3] = 0.375
        expected[3, 4] = 0.375
        expected[4, 3] = 0.125
        expected[4, 4] = 0.125
        np.testing.assert_allclose(translated, expected)

    def test_translate_morphology_validates_offset(self):
        with self.assertRaises(ValueError):
            translate_morphology(np.ones((3, 3)), (1.0,))

    def test_residual_metrics_apply_weights_and_mask(self):
        residual = np.ones((2, 5, 7))
        weights = np.full_like(residual, 4.0)
        weights[:, 0, 0] = 0.0
        score = residual_metrics(residual, weights)
        self.assertEqual(score["valid_voxels"], residual.size - 2)
        self.assertAlmostEqual(score["chi_square_per_voxel"], 4.0)

    def test_residual_whiteness_rejects_coherent_structure(self):
        generator = np.random.default_rng(12)
        white = generator.normal(size=(32, 31, 33))
        rows = np.arange(31, dtype=float)[None, :, None]
        coherent = np.broadcast_to(np.sin(2 * np.pi * rows / 31), white.shape)
        weights = np.ones_like(white)
        white_score = residual_metrics(white, weights)
        coherent_score = residual_metrics(coherent, weights)
        self.assertGreater(
            white_score["power_spectral_entropy"],
            coherent_score["power_spectral_entropy"],
        )
        self.assertLess(
            white_score["lag1_autocorrelation"],
            coherent_score["lag1_autocorrelation"],
        )

    def test_start_sensitivity_uses_every_pair(self):
        baseline = np.arange(1.0, 7.0)
        identical = start_sensitivity([baseline, baseline.copy(), baseline.copy()])
        np.testing.assert_array_equal(identical["pairwise_relative_l2"], 0.0)
        changed = start_sensitivity([baseline, 1.1 * baseline, baseline[::-1]])
        self.assertEqual(changed["pairwise_relative_l2"].size, 3)
        self.assertGreater(changed["max_pairwise_relative_l2"], 0.0)

"""The full collaborator driver keeps morphology features explicit."""

import unittest

import numpy as np

import spaxlet
from benchmarks.run_collaborator_reproduction import (
    _catalog_order,
    _morphology_parameter,
)


class IFUCollaboratorFeatureTest(unittest.TestCase):
    def test_positivity_feature_preserves_plain_array_construction(self):
        value = np.arange(20, dtype=float).reshape(4, 5) + 1
        result = _morphology_parameter(value, "positivity", (1.2, 2.3))
        self.assertIs(type(result), np.ndarray)
        np.testing.assert_array_equal(result, value)
        self.assertIsNot(result, value)

    def test_centroid_feature_projects_to_declared_nonnegative_centroid(self):
        value = np.arange(20, dtype=float).reshape(4, 5) + 1
        center = (1.2, 2.3)
        parameter = _morphology_parameter(value, "centroid", center)
        self.assertIsInstance(parameter, spaxlet.Parameter)
        projected = parameter.constraint(parameter.copy(), 1)
        rows, columns = np.indices(projected.shape, dtype=float)
        self.assertGreater(float(np.sum(projected)), 0)
        self.assertGreaterEqual(float(np.min(projected)), 0)
        measured = (
            float(np.sum(projected * rows) / np.sum(projected)),
            float(np.sum(projected * columns) / np.sum(projected)),
        )
        np.testing.assert_allclose(measured, center, rtol=0, atol=2e-11)

    def test_psf_frame_centroid_uses_the_supplied_corrected_center(self):
        value = np.arange(20, dtype=float).reshape(4, 5) + 1
        center = (1.7, 2.6)
        parameter = _morphology_parameter(value, "centroid_psf", center)
        projected = parameter.constraint(parameter.copy(), 1)
        rows, columns = np.indices(projected.shape, dtype=float)
        measured = (
            float(np.sum(projected * rows) / np.sum(projected)),
            float(np.sum(projected * columns) / np.sum(projected)),
        )
        np.testing.assert_allclose(measured, center, rtol=0, atol=2e-11)

    def test_unknown_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown morphology feature"):
            _morphology_parameter(np.ones((3, 3)), "volume", (1, 1))

    def test_catalog_order_uses_centers_in_the_latent_psf_frame(self):
        first = np.zeros((9, 11))
        second = np.zeros_like(first)
        first[3, 8] = 1
        second[7, 2] = 1
        reference_centers = ((3.0, 8.0), (7.0, 2.0))
        self.assertEqual(
            _catalog_order((first, second), reference_centers),
            (0, 1),
        )
        self.assertEqual(
            _catalog_order((second, first), reference_centers),
            (1, 0),
        )


if __name__ == "__main__":
    unittest.main()

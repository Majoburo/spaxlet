"""The full collaborator driver keeps morphology features explicit."""

import unittest

import numpy as np

import scarlet
from benchmarks.run_collaborator_reproduction import _morphology_parameter


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
        self.assertIsInstance(parameter, scarlet.Parameter)
        projected = parameter.constraint(parameter.copy(), 1)
        rows, columns = np.indices(projected.shape, dtype=float)
        self.assertGreater(float(np.sum(projected)), 0)
        self.assertGreaterEqual(float(np.min(projected)), 0)
        measured = (
            float(np.sum(projected * rows) / np.sum(projected)),
            float(np.sum(projected * columns) / np.sum(projected)),
        )
        np.testing.assert_allclose(measured, center, rtol=0, atol=2e-11)

    def test_unknown_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown morphology feature"):
            _morphology_parameter(np.ones((3, 3)), "volume", (1, 1))


if __name__ == "__main__":
    unittest.main()

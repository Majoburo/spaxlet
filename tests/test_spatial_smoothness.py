"""Contracts for convex spatial coherence of factorized morphologies."""

import unittest

import numpy as np

import spaxlet


class SpatialSmoothnessConstraintTest(unittest.TestCase):
    def test_exact_quadratic_prox_spreads_an_impulse_and_conserves_mass(self):
        image = np.zeros((5, 5))
        image[2, 2] = 1
        constraint = spaxlet.SpatialSmoothnessConstraint(strength=2)

        result = constraint(image, step=0.5)

        self.assertLess(result[2, 2], 1)
        self.assertGreater(result[2, 1], 0)
        self.assertGreater(result[1, 2], 0)
        np.testing.assert_allclose(result.sum(), image.sum(), atol=1e-12)
        self.assertLess(
            np.sum(np.diff(result, axis=0) ** 2)
            + np.sum(np.diff(result, axis=1) ** 2),
            np.sum(np.diff(image, axis=0) ** 2)
            + np.sum(np.diff(image, axis=1) ** 2),
        )

    def test_response_is_invariant_to_factor_amplitude_gauge(self):
        image = np.arange(1, 17, dtype=float).reshape(4, 4)
        base = spaxlet.SpatialSmoothnessConstraint(3, reference_scale=2)
        scaled = spaxlet.SpatialSmoothnessConstraint(3, reference_scale=14)

        expected = base(image, step=0.4)
        actual = scaled(7 * image, step=7 * 0.4)

        np.testing.assert_allclose(actual, 7 * expected, rtol=1e-12, atol=1e-12)

    def test_proximal_dykstra_combines_smoothness_positivity_and_centroid(self):
        image = np.zeros((5, 5))
        image[0, 0] = 1
        image[4, 4] = 0.4
        image[2, 2] = -0.2
        chain = spaxlet.ProximalDykstraConstraintChain(
            spaxlet.SpatialSmoothnessConstraint(1),
            spaxlet.CentroidConstraint((2, 2)),
            spaxlet.PositivityConstraint(),
            max_iter=2000,
            rtol=1e-10,
            atol=1e-12,
        )

        result = chain(image, step=0.2)
        rows, columns = np.indices(result.shape, dtype=float)

        self.assertTrue(np.all(result >= -1e-12))
        np.testing.assert_allclose(
            [np.sum(rows * result), np.sum(columns * result)],
            2 * result.sum(),
            rtol=0,
            atol=2e-9,
        )

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            spaxlet.SpatialSmoothnessConstraint(-1)
        with self.assertRaises(ValueError):
            spaxlet.SpatialSmoothnessConstraint(1, reference_scale=0)
        constraint = spaxlet.SpatialSmoothnessConstraint(1)
        with self.assertRaises(ValueError):
            constraint(np.ones(5), step=1)
        with self.assertRaises(ValueError):
            constraint(np.ones((2, 2)), step=np.arange(4).reshape(2, 2))


if __name__ == "__main__":
    unittest.main()

"""Tests for reporting factorized Scarlet components in a fixed gauge."""

import unittest

import numpy as np

import spaxlet


class FactorizationMeasureTest(unittest.TestCase):
    def test_factorization_preserves_model_and_normalizes_morphology(self):
        frame = spaxlet.Frame((3, 4, 5), channels=("a", "b", "c"))
        spectrum = spaxlet.TabulatedSpectrum(frame, np.asarray([2.0, 3.0, 5.0]))
        raw_morphology = np.arange(1, 21, dtype=float).reshape(4, 5)
        morphology = spaxlet.ImageMorphology(
            frame, raw_morphology, resizing=False
        )
        component = spaxlet.FactorizedComponent(frame, spectrum, morphology)

        factors = spaxlet.measure.factorization(component)

        self.assertAlmostEqual(float(factors.morphology.sum()), 1.0)
        np.testing.assert_allclose(
            factors.spectrum[:, None, None] * factors.morphology,
            component.get_model(),
        )

    def test_non_factorized_component_fails(self):
        frame = spaxlet.Frame((2, 3, 3), channels=("a", "b"))
        component = spaxlet.CubeComponent(
            frame, spaxlet.Parameter(np.ones(frame.shape), name="cube")
        )
        with self.assertRaises(TypeError):
            spaxlet.measure.factorization(component)


if __name__ == "__main__":
    unittest.main()

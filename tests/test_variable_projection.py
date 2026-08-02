"""Tests for the opt-in constrained variable-projection optimizer."""

import unittest

import numpy as np

import scarlet
from scarlet.optimization import spectral_volume_value_gradient


def _component(frame, spectrum, morphology, *, fixed_morphology=False, center=None):
    if center is None:
        constraint = scarlet.PositivityConstraint()
    else:
        constraint = scarlet.DykstraConstraintChain(
            scarlet.CentroidConstraint(center),
            scarlet.PositivityConstraint(),
            max_iter=10000,
            rtol=1e-12,
            atol=1e-13,
        )
    image = scarlet.Parameter(
        np.asarray(morphology, dtype=float),
        name="image",
        step=scarlet.parameter.relative_step,
        constraint=constraint,
        fixed=fixed_morphology,
    )
    return scarlet.FactorizedComponent(
        frame,
        scarlet.TabulatedSpectrum(frame, np.asarray(spectrum, dtype=float)),
        scarlet.ImageMorphology(frame, image, resizing=False),
    )


class VariableProjectionTest(unittest.TestCase):
    def setUp(self):
        self.shape = (5, 7, 7)
        self.channels = tuple(range(self.shape[0]))
        self.psf = scarlet.DeltaPSF(self.shape[0])
        self.frame = scarlet.Frame(
            self.shape, psf=self.psf, channels=self.channels
        )

    def observation(self, data):
        return scarlet.Observation(
            np.asarray(data, dtype=float),
            psf=self.psf,
            weights=np.ones(self.shape),
            channels=self.channels,
        ).match(self.frame)

    def test_exactly_profiles_nonnegative_spectra(self):
        morphology_a = np.zeros(self.shape[1:])
        morphology_a[1:3, 1:3] = 0.25
        morphology_b = np.zeros(self.shape[1:])
        morphology_b[4:6, 4:6] = 0.25
        truth = np.asarray(
            [[0.4, 1.1], [0.7, 0.9], [1.0, 0.6], [1.3, 0.3], [1.6, 0.1]]
        )
        data = (
            truth[:, 0, None, None] * morphology_a
            + truth[:, 1, None, None] * morphology_b
        )
        sources = [
            _component(
                self.frame, np.ones(self.shape[0]), morphology_a,
                fixed_morphology=True,
            ),
            _component(
                self.frame, np.ones(self.shape[0]), morphology_b,
                fixed_morphology=True,
            ),
        ]
        blend = scarlet.Blend(sources, self.observation(data))

        blend.fit(1, optimizer="variable_projection", e_rel=0)

        recovered = np.stack(
            [source.spectrum.get_model() for source in sources], axis=1
        )
        np.testing.assert_allclose(recovered, truth, atol=2e-12)
        self.assertEqual(len(blend.loss), 1)

    def test_centroid_projection_is_exact_and_objective_is_monotone(self):
        rows, columns = np.indices(self.shape[1:], dtype=float)
        centers = ((2.0, 2.0), (4.0, 4.0))

        def blob(center, scale):
            value = np.exp(-np.hypot(rows - center[0], columns - center[1]) / scale)
            return value / value.sum()

        truth_morphologies = [blob(centers[0], 0.8), blob(centers[1], 0.9)]
        truth_spectra = np.asarray(
            [[0.5, 1.2], [0.8, 1.0], [1.1, 0.7], [1.4, 0.5], [1.7, 0.3]]
        )
        data = sum(
            truth_spectra[:, source, None, None] * morphology
            for source, morphology in enumerate(truth_morphologies)
        )
        sources = [
            _component(
                self.frame,
                np.ones(self.shape[0]),
                blob(center, 1.6),
                center=center,
            )
            for center in centers
        ]
        blend = scarlet.Blend(sources, self.observation(data))

        blend.fit(
            20,
            optimizer="variable_projection",
            e_rel=0,
            project_initial=True,
            normalize_initial_factors=True,
        )

        self.assertTrue(np.all(np.diff(np.asarray(blend.loss)) <= 1e-10))
        for source, center in zip(sources, centers):
            image = np.asarray(source.morphology.get_model())
            measured = np.asarray(
                [np.sum(rows * image), np.sum(columns * image)]
            ) / np.sum(image)
            np.testing.assert_allclose(measured, center, atol=2e-10)

    def test_volume_strength_changes_the_spectral_optimum(self):
        morphology_a = np.zeros(self.shape[1:])
        morphology_a[1:4, 1:4] = 1.0 / 9.0
        morphology_b = np.zeros(self.shape[1:])
        morphology_b[3:6, 3:6] = 1.0 / 9.0
        truth = np.asarray(
            [[0.5, 1.4], [0.8, 1.1], [1.1, 0.8], [1.4, 0.5], [1.7, 0.3]]
        )
        data = (
            truth[:, 0, None, None] * morphology_a
            + truth[:, 1, None, None] * morphology_b
        )

        def fitted(strength):
            sources = [
                _component(
                    self.frame, np.ones(self.shape[0]), morphology_a,
                    fixed_morphology=True,
                ),
                _component(
                    self.frame, np.ones(self.shape[0]), morphology_b,
                    fixed_morphology=True,
                ),
            ]
            blend = scarlet.Blend(sources, self.observation(data))
            blend.fit(
                1,
                optimizer="variable_projection",
                minimum_volume_strength=strength,
                spectral_max_iter=300,
                spectral_tolerance=1e-11,
                e_rel=0,
            )
            spectra = np.stack(
                [source.spectrum.get_model() for source in sources], axis=1
            )
            return blend, spectra

        baseline, baseline_spectra = fitted(0.0)
        regularized, regularized_spectra = fitted(0.1)
        baseline_volume, _ = spectral_volume_value_gradient(
            baseline_spectra, 1.0
        )
        regularized_volume, _ = spectral_volume_value_gradient(
            regularized_spectra, 1.0
        )

        self.assertGreater(np.linalg.norm(regularized_spectra - baseline_spectra), 1e-6)
        self.assertLess(regularized_volume, baseline_volume)
        self.assertLessEqual(regularized.loss[-1], baseline.loss[-1] + 0.1 * baseline_volume)

    def test_volume_gradient_matches_finite_difference(self):
        rng = np.random.RandomState(19)
        spectra = rng.uniform(0.3, 1.8, size=(7, 3))
        direction = rng.normal(size=spectra.shape)
        strength = 0.37
        _, gradient = spectral_volume_value_gradient(spectra, strength)
        epsilon = 1e-6
        above, _ = spectral_volume_value_gradient(
            spectra + epsilon * direction, strength
        )
        below, _ = spectral_volume_value_gradient(
            spectra - epsilon * direction, strength
        )
        numerical = (above - below) / (2.0 * epsilon)
        analytic = float(np.sum(gradient * direction))
        self.assertAlmostEqual(numerical, analytic, places=6)


if __name__ == "__main__":
    unittest.main()

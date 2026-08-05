"""Contracts for wavelength-aware, line-preserving spectral smoothness."""

import itertools
import unittest

import numpy as np
from astropy import units as u

import spaxlet


def _curvature_matrix(wavelength, gap_factor=5):
    wavelength = np.asarray(wavelength, dtype=float)
    size = wavelength.size
    if size < 3:
        return np.zeros((0, size), dtype=float)
    spacing = np.diff(wavelength)
    coordinate = (wavelength - wavelength[0]) / np.median(spacing)
    spacing = np.diff(coordinate)
    gap = spacing > gap_factor
    rows = []
    for center in range(1, size - 1):
        if gap[center - 1] or gap[center]:
            continue
        left, right = spacing[center - 1 : center + 1]
        normalization = 2 / (left + right)
        row = np.zeros(size, dtype=float)
        row[center - 1 : center + 2] = (
            normalization / left,
            -normalization * (1 / left + 1 / right),
            normalization / right,
        )
        rows.append(row)
    return np.asarray(rows, dtype=float).reshape((-1, size))


def _enumerated_nonnegative_quadratic_prox(value, hessian):
    """Reference the tiny bound-constrained solve by enumerating active faces."""

    value = np.asarray(value, dtype=float)
    best = np.zeros_like(value)
    best_objective = 0.5 * float(np.dot(value, value))
    for count in range(1, value.size + 1):
        for support_tuple in itertools.combinations(range(value.size), count):
            support = np.asarray(support_tuple, dtype=int)
            candidate = np.zeros_like(value)
            candidate[support] = np.linalg.solve(
                hessian[np.ix_(support, support)], value[support]
            )
            if np.min(candidate[support]) < -1e-11:
                continue
            gradient = hessian @ candidate - value
            inactive = np.ones(value.size, dtype=bool)
            inactive[support] = False
            if np.min(gradient[inactive], initial=0) < -1e-10:
                continue
            objective = 0.5 * float(candidate @ hessian @ candidate)
            objective -= float(value @ candidate)
            if objective < best_objective:
                best = candidate
                best_objective = objective
    return best


class SpectralSmoothnessConstraintTest(unittest.TestCase):
    def test_matches_enumerated_nonnegative_quadratic_prox(self):
        wavelength = np.asarray([1.0, 1.1, 1.21, 1.31, 1.43, 1.54])
        value = np.asarray([1.2, -0.7, 2.1, 0.1, -0.4, 1.5])
        strength = 1.7
        step = 0.35
        curvature = _curvature_matrix(wavelength)
        hessian = np.eye(value.size) + step * strength * (
            curvature.T @ curvature
        )
        expected = _enumerated_nonnegative_quadratic_prox(value, hessian)
        constraint = spaxlet.SpectralSmoothnessConstraint(
            wavelength,
            strength,
            max_iter=1000,
            rtol=1e-12,
            atol=1e-13,
        )

        actual = constraint(value, step)

        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)
        self.assertGreaterEqual(float(actual.min()), 0)

    def test_preserves_resolved_line_and_rejects_one_channel_spike(self):
        wavelength = np.linspace(3.8, 4.0, 81)
        channel = np.arange(wavelength.size, dtype=float)
        continuum = 1 + 0.003 * channel
        resolved_line = 5 * np.exp(-0.5 * ((channel - 40) / 2.2) ** 2)
        one_channel_spike = np.zeros_like(channel)
        one_channel_spike[20] = 5
        constraint = spaxlet.SpectralSmoothnessConstraint(wavelength, strength=3)

        resolved = constraint(continuum + resolved_line, step=0.2) - continuum
        unresolved = constraint(continuum + one_channel_spike, step=0.2) - continuum

        self.assertGreater(resolved.max(), 0.95 * resolved_line.max())
        self.assertAlmostEqual(
            float(resolved.sum()), float(resolved_line.sum()), delta=0.01
        )
        self.assertLess(unresolved.max(), 0.65 * one_channel_spike.max())
        self.assertGreater(resolved.max() / resolved_line.max(),
                           unresolved.max() / one_channel_spike.max() + 0.3)

    def test_affine_segments_are_unchanged_across_a_gap(self):
        wavelength = np.asarray([1.0, 1.1, 1.2, 2.0, 2.1, 2.2])
        value = np.asarray([1.0, 1.2, 1.4, 4.0, 3.7, 3.4])
        constraint = spaxlet.SpectralSmoothnessConstraint(
            wavelength, strength=100, gap_factor=5
        )

        np.testing.assert_allclose(constraint(value, step=1), value, atol=2e-12)

    def test_reference_scale_makes_amplitude_response_invariant(self):
        wavelength = np.linspace(1, 2, 21)
        value = 1 + np.exp(-0.5 * ((np.arange(21) - 10) / 2) ** 2)
        base = spaxlet.SpectralSmoothnessConstraint(
            wavelength, strength=2, reference_scale=1
        )(value, step=0.2)
        scaled = spaxlet.SpectralSmoothnessConstraint(
            wavelength, strength=2, reference_scale=7
        )(7 * value, step=7 * 0.2)

        np.testing.assert_allclose(scaled, 7 * base, rtol=2e-12, atol=2e-12)

    def test_tabulated_spectrum_accepts_explicit_constraint(self):
        wavelength = np.asarray([1.0, 1.1, 1.2])
        frame = spaxlet.Frame(
            (3, 2, 2), channels=("a", "b", "c"), wavelengths=wavelength * u.um
        )
        constraint = spaxlet.SpectralSmoothnessConstraint(wavelength, strength=1)
        spectrum = spaxlet.TabulatedSpectrum(
            frame, np.ones(3), constraint=constraint
        )

        self.assertIs(spectrum.parameters[0].constraint, constraint)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            spaxlet.SpectralSmoothnessConstraint([1, 1, 2], strength=1)
        with self.assertRaises(ValueError):
            spaxlet.SpectralSmoothnessConstraint([1, 2, 3], strength=-1)
        with self.assertRaises(ValueError):
            spaxlet.SpectralSmoothnessConstraint(
                [1, 2, 3], strength=1, reference_scale=0
            )
        constraint = spaxlet.SpectralSmoothnessConstraint([1, 2, 3], strength=1)
        with self.assertRaises(ValueError):
            constraint(np.ones(4), step=1)


if __name__ == "__main__":
    unittest.main()

"""Independent contracts for exact projections onto constraint intersections."""

import itertools
import unittest

import numpy as np

import scarlet


def _enumerated_nonnegative_centroid_projection(value, center):
    """Solve the tiny reference problem by enumerating every support face."""

    value = np.asarray(value, dtype=float)
    rows, columns = np.indices(value.shape, dtype=float)
    design = np.stack((rows - center[0], columns - center[1]), axis=0).reshape(
        2, -1
    )
    flat = value.reshape(-1)
    best = np.zeros_like(flat)
    best_loss = float(np.dot(best - flat, best - flat))
    for count in range(1, flat.size + 1):
        for support in itertools.combinations(range(flat.size), count):
            support = np.asarray(support, dtype=int)
            local_design = design[:, support]
            local = flat[support]
            gram = local_design @ local_design.T
            projected = local - local_design.T @ np.linalg.pinv(gram) @ (
                local_design @ local
            )
            if np.min(projected) < -1e-11:
                continue
            if np.linalg.norm(local_design @ projected) > 1e-10:
                continue
            candidate = np.zeros_like(flat)
            candidate[support] = projected
            loss = float(np.dot(candidate - flat, candidate - flat))
            if loss < best_loss:
                best = candidate
                best_loss = loss
    return best.reshape(value.shape)


class ExactIntersectionProjectionTest(unittest.TestCase):
    def test_centroid_projection_matches_enumerated_reference(self):
        value = np.array([[1.2, -0.4, 0.7], [0.1, 1.8, -0.6]])
        center = (0.45, 1.1)
        expected = _enumerated_nonnegative_centroid_projection(value, center)
        constraint = scarlet.DykstraConstraintChain(
            scarlet.CentroidConstraint(center),
            scarlet.PositivityConstraint(),
            max_iter=20000,
            rtol=1e-12,
            atol=1e-13,
        )

        actual = constraint(value.copy(), 0)
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
        self.assertGreaterEqual(np.min(actual), -2e-11)
        rows, columns = np.indices(actual.shape, dtype=float)
        self.assertLess(abs(np.sum(actual * (rows - center[0]))), 2e-10)
        self.assertLess(abs(np.sum(actual * (columns - center[1]))), 2e-10)

        alternating = scarlet.ConstraintChain(
            scarlet.CentroidConstraint(center),
            scarlet.PositivityConstraint(),
            repeat=100,
        )(value.copy(), 0)
        alternating_violation = np.linalg.norm(
            scarlet.CentroidConstraint(center)(alternating.copy(), 0) - alternating
        )
        self.assertLess(alternating_violation, 1e-12)
        self.assertGreater(np.linalg.norm(alternating - expected), 0.07)
        self.assertGreater(
            np.linalg.norm(alternating - value),
            np.linalg.norm(actual - value) + 1e-3,
        )

    def test_projection_is_idempotent(self):
        value = np.array([[0.2, 2.0, -1.0], [1.4, -0.3, 0.8]])
        constraint = scarlet.DykstraConstraintChain(
            scarlet.CentroidConstraint((0.4, 0.9)),
            scarlet.PositivityConstraint(),
        )
        once = constraint(value.copy(), 0)
        twice = constraint(once.copy(), 0)
        np.testing.assert_allclose(twice, once, rtol=2e-10, atol=1e-10)

    def test_exact_chain_rejects_heuristic_constraints(self):
        with self.assertRaisesRegex(ValueError, "MonotonicityConstraint"):
            scarlet.DykstraConstraintChain(
                scarlet.MonotonicityConstraint()
            )
        with self.assertRaisesRegex(ValueError, "SymmetryConstraint"):
            scarlet.DykstraConstraintChain(
                scarlet.SymmetryConstraint(strength=0.5)
            )

    def test_nonconvergence_is_not_silent(self):
        constraint = scarlet.DykstraConstraintChain(
            scarlet.CentroidConstraint((0.45, 1.1)),
            scarlet.PositivityConstraint(),
            max_iter=1,
            rtol=0,
            atol=0,
        )
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            constraint(np.array([[1.2, -0.4, 0.7], [0.1, 1.8, -0.6]]), 0)


if __name__ == "__main__":
    unittest.main()

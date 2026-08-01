"""Coordinate-bearing constraints must survive morphology support changes."""

import unittest

import numpy as np

import scarlet
from scarlet.model import UpdateException


def _centroid_constraint(chain):
    return next(
        constraint
        for constraint in chain.constraints
        if isinstance(constraint, scarlet.CentroidConstraint)
    )


class SupportResizeConstraintTest(unittest.TestCase):
    def _morphology(self, shape, center, origin, value, *, grow=False):
        frame = scarlet.Frame((1, 61, 71), channels=[0])
        constraint = scarlet.DykstraConstraintChain(
            scarlet.CentroidConstraint(center),
            scarlet.PositivityConstraint(),
        )
        keywords = {}
        if grow:
            keywords = {
                "m": -np.ones(shape),
                "v": np.ones(shape),
                "vhat": np.ones(shape),
            }
        image = scarlet.Parameter(
            value,
            name="image",
            step=1.0,
            constraint=constraint,
            **keywords
        )
        return scarlet.ImageMorphology(
            frame,
            image,
            bbox=scarlet.Box(shape, origin=origin),
            resizing=True,
        )

    def test_growth_preserves_global_centroid_coordinate(self):
        shape = (21, 21)
        center = (8.0, 12.0)
        origin = (10, 20)
        morphology = self._morphology(
            shape, center, origin, np.ones(shape), grow=True
        )
        global_center = np.add(origin, center)

        with self.assertRaises(UpdateException):
            morphology.update()

        self.assertEqual(morphology.bbox.shape, (31, 31))
        shifted = _centroid_constraint(morphology.get_parameter(0).constraint)
        np.testing.assert_allclose(
            np.add(morphology.bbox.origin, shifted.center), global_center
        )
        self.assertEqual(shifted.center, (13.0, 17.0))

    def test_shrink_preserves_global_centroid_coordinate(self):
        shape = (31, 31)
        center = (13.0, 17.0)
        origin = (5, 15)
        value = np.zeros(shape)
        value[14:17, 14:17] = 1
        morphology = self._morphology(shape, center, origin, value)
        global_center = np.add(origin, center)

        with self.assertRaises(UpdateException):
            morphology.update()

        self.assertEqual(morphology.bbox.shape, (21, 21))
        shifted = _centroid_constraint(morphology.get_parameter(0).constraint)
        np.testing.assert_allclose(
            np.add(morphology.bbox.origin, shifted.center), global_center
        )
        self.assertEqual(shifted.center, (8.0, 12.0))

    def test_nested_explicit_centers_shift_together(self):
        chain = scarlet.ConstraintChain(
            scarlet.MonotonicityConstraint(center=(4, 7)),
            scarlet.SymmetryConstraint(center=(4, 7)),
            scarlet.CenterOnConstraint(center=(4, 7)),
        ).shifted((3, -2))
        self.assertEqual(
            tuple(constraint.center for constraint in chain.constraints),
            ((7, 5), (7, 5), (7, 5)),
        )

    def test_coordinate_free_constraint_is_reused(self):
        constraint = scarlet.PositivityConstraint()
        self.assertIs(constraint.shifted((5, 5)), constraint)

    def test_integer_center_rejects_fractional_grid_shift(self):
        with self.assertRaisesRegex(ValueError, "integer shift"):
            scarlet.CenterOnConstraint(center=(4, 7)).shifted((0.5, 1))


if __name__ == "__main__":
    unittest.main()

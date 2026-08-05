"""Unit tests for target-specific SPT0311 benchmark plumbing."""

import unittest

import numpy as np
from astropy.io.fits import Header

from benchmarks.run_spt0311_deblend import (
    PUBLISHED_OFFSETS_ARCSEC,
    catalog_centers_yx,
    morphology_parameter,
    parse_source_groups,
    selected_channel_indices,
    source_box_size,
    source_constraint_name,
    source_group_box_size,
    source_morphology_box,
)


class SPT0311BenchmarkTest(unittest.TestCase):
    def test_contiguous_wavelength_selection(self):
        wavelengths = np.arange(10, dtype=float) * 0.1 + 1
        np.testing.assert_array_equal(
            selected_channel_indices(wavelengths, 1.2, 1.5), (2, 3, 4, 5)
        )
        with self.assertRaises(ValueError):
            selected_channel_indices(wavelengths, 3, 4)

    def test_catalog_offsets_map_to_expected_pixel_directions(self):
        primary = Header({"TARG_RA": 30.0, "TARG_DEC": -20.0})
        science = Header(
            {
                "NAXIS": 2,
                "NAXIS1": 57,
                "NAXIS2": 57,
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRPIX1": 29.0,
                "CRPIX2": 29.0,
                "CRVAL1": 30.0,
                "CRVAL2": -20.0,
                "CDELT1": -0.1 / 3600,
                "CDELT2": 0.1 / 3600,
                "CUNIT1": "deg",
                "CUNIT2": "deg",
            }
        )
        centers, pixel_scale, origin = catalog_centers_yx(
            primary, science, ("lens", "E")
        )

        self.assertAlmostEqual(pixel_scale, 0.1)
        for name in centers:
            delta_ra, delta_dec = PUBLISHED_OFFSETS_ARCSEC[name]
            np.testing.assert_allclose(
                centers[name],
                origin + np.asarray([delta_dec, -delta_ra]) / pixel_scale,
            )

    def test_source_support_is_odd_centered_and_in_frame(self):
        box = source_morphology_box((20, 30), np.asarray([1.2, 28.4]), 9)
        self.assertEqual(box.shape, (9, 9))
        self.assertEqual(box.origin, (0, 21))
        with self.assertRaises(ValueError):
            source_morphology_box((20, 30), (5, 5), 8)

        self.assertEqual(source_box_size("lens"), 21)
        self.assertEqual(source_box_size("lens", padding=2), 25)
        self.assertEqual(source_box_size("C1", padding=2), 15)
        with self.assertRaises(ValueError):
            source_box_size("C1", padding=-1)

    def test_merged_source_factor_uses_combined_support(self):
        groups = parse_source_groups("lens+W,E+L7,C1", "g395h")
        self.assertEqual(groups["lens+W"], ("lens", "W"))
        centers = {"E": np.asarray((10.0, 10.0)), "L7": np.asarray((11.0, 9.0))}
        self.assertEqual(
            source_group_box_size(
                ("E", "L7"), centers, np.asarray((10.5, 9.5))
            ),
            19,
        )
        with self.assertRaisesRegex(ValueError, "more than one factor"):
            parse_source_groups("E+L7,L7", "g395h")

    def test_hybrid_constraint_selection_is_explicit(self):
        self.assertEqual(source_constraint_name("lens", "hybrid"), "symmetry")
        self.assertEqual(source_constraint_name("E", "hybrid"), "positivity")
        self.assertEqual(
            source_constraint_name("E", "hybrid_centered"), "centroid"
        )
        self.assertEqual(source_constraint_name("E", "monotonic"), "monotonic")

    def test_spatial_smoothness_is_joint_with_identity_constraints(self):
        image = np.zeros((7, 7))
        image[3, 3] = 1
        parameter = morphology_parameter(
            image, (3, 3), "centroid", spatial_smoothness_strength=100
        )
        projected = parameter.constraint(image.copy(), step=0.01)
        rows, columns = np.indices(image.shape, dtype=float)

        self.assertTrue(np.all(projected >= -1e-12))
        self.assertLess(projected[3, 3], 1)
        np.testing.assert_allclose(
            [np.sum(rows * projected), np.sum(columns * projected)],
            3 * projected.sum(),
            atol=1e-9,
        )
        with self.assertRaises(ValueError):
            morphology_parameter(image, (3, 3), "monotonic", 1)


if __name__ == "__main__":
    unittest.main()

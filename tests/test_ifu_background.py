"""Tests for reusable IFU blank-sky and variance calibration."""

import unittest

import numpy as np

import spaxlet


class IFUBackgroundTest(unittest.TestCase):
    def test_isolated_spatial_outliers_preserve_resolved_emission(self):
        data = np.zeros((2, 9, 9), dtype=float)
        variance = np.ones_like(data)
        data[0, 1:3, 1:3] = ((4, 5), (6, 20))
        data[0, 7, 7] = -18
        data[1, 2:7, 2:7] = 4
        data[1, 4, 4] = 30
        valid = np.ones_like(data, dtype=bool)
        valid[0, 7, 7] = False

        mask = spaxlet.isolated_spatial_outlier_mask(
            data,
            variance,
            valid_mask=valid,
            sigma=12,
            grow_sigma=3,
            max_spatial_pixels=12,
        )

        expected = np.zeros_like(mask)
        expected[0, 1:3, 1:3] = True
        np.testing.assert_array_equal(mask, expected)

    def test_invalid_spatial_outlier_contracts_fail(self):
        data = np.ones((2, 3, 4))
        variance = np.ones_like(data)
        invalid_calls = (
            lambda: spaxlet.isolated_spatial_outlier_mask(data[0], variance[0]),
            lambda: spaxlet.isolated_spatial_outlier_mask(
                data, variance[:, :, :-1]
            ),
            lambda: spaxlet.isolated_spatial_outlier_mask(
                data, variance, valid_mask=np.ones_like(data)
            ),
            lambda: spaxlet.isolated_spatial_outlier_mask(
                data, variance, sigma=3, grow_sigma=3
            ),
            lambda: spaxlet.isolated_spatial_outlier_mask(
                data, variance, max_spatial_pixels=0
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()

    def test_channel_background_and_whitened_mad(self):
        residual = np.tile(np.asarray([-4, -2, 0, 2, 4], dtype=float), (5, 1))
        background = np.asarray([12.0, -3.0])
        data = background[:, None, None] + residual[None]
        variance = np.full(data.shape, 4.0)
        source_mask = np.zeros(data.shape[1:], dtype=bool)
        source_mask[2, 2] = True
        data[:, 2, 2] += 1000

        estimate = spaxlet.estimate_ifu_background(
            data,
            variance,
            source_mask=source_mask,
            minimum_noise_scale=0.1,
        )

        np.testing.assert_allclose(estimate.background, background)
        self.assertAlmostEqual(estimate.noise_scale, 1.4826, places=6)
        self.assertEqual(estimate.background_voxels, 48)

    def test_declared_mask_and_noise_floor(self):
        data = np.ones((2, 3, 4))
        variance = np.ones_like(data)
        valid = np.ones_like(data, dtype=bool)
        valid[:, 0] = False

        estimate = spaxlet.estimate_ifu_background(
            data, variance, valid_mask=valid, minimum_noise_scale=2.5
        )

        np.testing.assert_array_equal(estimate.background, (1, 1))
        self.assertEqual(estimate.noise_scale, 2.5)
        self.assertEqual(estimate.background_voxels, 16)

    def test_invalid_shapes_and_masks_fail(self):
        data = np.ones((2, 3, 4))
        variance = np.ones_like(data)
        invalid_calls = (
            lambda: spaxlet.estimate_ifu_background(data[0], variance[0]),
            lambda: spaxlet.estimate_ifu_background(data, variance[:, :, :-1]),
            lambda: spaxlet.estimate_ifu_background(
                data, variance, valid_mask=np.ones_like(data)
            ),
            lambda: spaxlet.estimate_ifu_background(
                data, variance, source_mask=np.zeros((3, 4))
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()

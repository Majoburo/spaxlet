"""IFU PSF preprocessing preserves flux and removes registration shifts."""

import unittest
import warnings

import numpy as np
from scipy.ndimage import shift

import spaxlet


def _gaussian_stack(sigmas, size=9):
    coordinate = np.arange(size, dtype=float) - 0.5 * (size - 1)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    values = np.asarray(
        [np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2) for sigma in sigmas]
    )
    return values / np.sum(values, axis=(1, 2))[:, None, None]


class IFUPSFPreprocessingTest(unittest.TestCase):
    def test_empirical_cube_extraction_removes_background_and_recenters(self):
        size = 17
        yy, xx = np.indices((size, size), dtype=float)
        cube = np.asarray(
            [
                3.0
                + (channel + 1)
                * np.exp(
                    -0.5
                    * (((yy - 8.30) / 1.1) ** 2 + ((xx - 7.65) / 1.1) ** 2)
                )
                for channel in range(5)
            ]
        )
        cube[:, 0, 0] = np.nan

        kernels, removed, peak = spaxlet.empirical_psf_kernels(
            cube,
            channel_indices=np.asarray([1, 3]),
            kernel_size=11,
            spectral_half_width=1,
        )

        self.assertEqual(kernels.shape, (2, 11, 11))
        self.assertEqual(peak, (8, 8))
        np.testing.assert_allclose(np.sum(kernels, axis=(1, 2)), 1)
        self.assertGreater(np.max(np.abs(removed)), 0.2)
        self.assertLess(np.max(np.abs(spaxlet.psf_centroids(kernels))), 0.02)

    def test_crop_reports_retained_flux_and_warns_for_offset_centroid(self):
        kernels = np.zeros((2, 8, 8), dtype=float)
        kernels[:, 3, 3] = 1
        kernels[:, 4, 4] = 1
        kernels[:, 7, 7] = (0.2, 0.4)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cropped, retained = spaxlet.crop_psf_kernels(kernels, 7)

        np.testing.assert_allclose(np.sum(cropped, axis=(1, 2)), 1)
        np.testing.assert_allclose(retained, 2 / np.array([2.2, 2.4]))
        self.assertTrue(any(item.category is RuntimeWarning for item in caught))

    def test_recentering_removes_subpixel_centroids(self):
        base = _gaussian_stack((0.9, 1.1))
        displaced = np.asarray(
            [
                shift(kernel, (0.3, -0.25), order=3, mode="constant", cval=0)
                for kernel in base
            ]
        )
        displaced = np.maximum(displaced, 0)
        displaced /= np.sum(displaced, axis=(1, 2))[:, None, None]

        centered, removed = spaxlet.recenter_psf_kernels(displaced)

        self.assertGreater(np.max(np.abs(removed)), 0.2)
        self.assertLess(np.max(np.abs(spaxlet.psf_centroids(centered))), 0.02)
        np.testing.assert_allclose(np.sum(centered, axis=(1, 2)), 1)

    def test_invalid_psfs_and_crop_sizes_fail_early(self):
        valid = _gaussian_stack((1.0,))
        invalid_calls = (
            lambda: spaxlet.psf_centroids(np.zeros((1, 3, 3))),
            lambda: spaxlet.psf_centroids(np.full((1, 3, 3), np.nan)),
            lambda: spaxlet.crop_psf_kernels(valid, 4),
            lambda: spaxlet.crop_psf_kernels(valid, 11),
            lambda: spaxlet.crop_psf_kernels(valid[:, :, :-1], 7),
            lambda: spaxlet.empirical_psf_kernels(
                np.ones((2, 7, 7)), kernel_size=4
            ),
            lambda: spaxlet.empirical_psf_kernels(
                np.ones((2, 7, 7)), channel_indices=np.asarray([2])
            ),
            lambda: spaxlet.empirical_psf_kernels(np.full((2, 7, 7), np.nan)),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


if __name__ == "__main__":
    unittest.main()

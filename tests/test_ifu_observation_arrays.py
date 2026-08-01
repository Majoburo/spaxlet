"""Mask-safe IFU array ingestion uses measured variance and explicit DQ bits."""

import unittest

import numpy as np
from astropy import units as u

import scarlet


class IFUObservationArraysTest(unittest.TestCase):
    def _arrays(self):
        data = np.arange(18, dtype=float).reshape(3, 2, 3) + 1
        variance = np.full(data.shape, 4.0)
        dq = np.zeros(data.shape, dtype=np.uint32)
        return data, variance, dq

    def test_variance_gaps_nonfinite_data_and_selected_dq_are_masked(self):
        data, variance, dq = self._arrays()
        original = data.copy()
        data[0, 0, 0] = np.nan
        variance[0, 0, 1] = 0
        variance[0, 0, 2] = -1
        variance[1] = np.inf
        dq[2, 0, 0] = 1
        dq[2, 0, 1] = 2

        observation = scarlet.Observation.from_ifu_arrays(
            data,
            np.array([1.0, 1.1, 1.2]) * u.um,
            variance,
            dq=dq,
            dq_bad_bits=2,
            psf=scarlet.DeltaPSF(3),
        )

        self.assertTrue(np.isnan(data[0, 0, 0]))
        np.testing.assert_allclose(data[0, 0, 1:], original[0, 0, 1:])
        self.assertEqual(np.count_nonzero(observation.weights[1]), 0)
        self.assertEqual(observation.weights[2, 0, 0], 0.25)
        self.assertEqual(observation.weights[2, 0, 1], 0)
        self.assertTrue(np.all(np.isfinite(observation.data)))
        self.assertTrue(np.all(np.isfinite(observation.weights)))
        self.assertTrue(np.all(observation.weights >= 0))
        self.assertEqual(observation.ifu_mask_summary["valid"], 8)
        self.assertEqual(observation.ifu_mask_summary["nonfinite_data"], 1)
        self.assertEqual(observation.ifu_mask_summary["invalid_variance"], 8)
        self.assertEqual(observation.ifu_mask_summary["dq_invalid"], 1)

    def test_any_nonzero_dq_is_bad_by_default(self):
        data, variance, dq = self._arrays()
        dq[0, 0, 0] = 1
        dq[0, 0, 1] = 8
        observation = scarlet.Observation.from_ifu_arrays(
            data,
            np.array([1.0, 1.1, 1.2]) * u.um,
            variance,
            dq=dq,
            psf=scarlet.DeltaPSF(3),
        )
        self.assertEqual(np.count_nonzero(observation.weights == 0), 2)

    def test_masked_gap_produces_finite_likelihood(self):
        data, variance, _ = self._arrays()
        data[1] = np.nan
        variance[1] = np.inf
        psf = scarlet.DeltaPSF(3)
        wavelengths = np.array([1.0, 1.1, 1.2]) * u.um
        observation = scarlet.Observation.from_ifu_arrays(
            data, wavelengths, variance, psf=psf
        )
        frame = scarlet.Frame(
            data.shape,
            channels=observation.channels,
            psf=psf,
            wavelengths=wavelengths,
            dtype=observation.dtype,
        )
        observation.match(frame)
        self.assertTrue(np.isfinite(observation.get_log_likelihood(np.zeros(data.shape))))

    def test_invalid_array_contracts_fail_early(self):
        data, variance, dq = self._arrays()
        wavelengths = np.array([1.0, 1.1, 1.2]) * u.um
        invalid_calls = (
            lambda: scarlet.Observation.from_ifu_arrays(
                data[0], wavelengths, variance[0]
            ),
            lambda: scarlet.Observation.from_ifu_arrays(
                data, wavelengths, variance[:, :, :2]
            ),
            lambda: scarlet.Observation.from_ifu_arrays(
                data, wavelengths, variance, dq=dq[:, :, :2]
            ),
            lambda: scarlet.Observation.from_ifu_arrays(
                data, wavelengths, variance, dq=dq.astype(float)
            ),
            lambda: scarlet.Observation.from_ifu_arrays(
                data, wavelengths, variance, dq_bad_bits=1
            ),
            lambda: scarlet.Observation.from_ifu_arrays(
                data, wavelengths, variance, dtype=np.int64
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


if __name__ == "__main__":
    unittest.main()

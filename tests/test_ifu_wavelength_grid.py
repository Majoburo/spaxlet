"""Physical IFU wavelengths must agree, not merely channel-array shapes."""

import unittest

import numpy as np
from autograd import grad
from astropy import units as u

import spaxlet


class IFUWavelengthGridTest(unittest.TestCase):
    def _frame(self, wavelengths=None, channels=None):
        if channels is None:
            channels = ["a", "b", "c"]
        return spaxlet.Frame(
            (len(channels), 5, 7),
            channels=channels,
            psf=spaxlet.DeltaPSF(len(channels)),
            wavelengths=wavelengths,
        )

    def _observation(self, wavelengths=None, channels=None, psf=None):
        if channels is None:
            channels = ["a", "b", "c"]
        if psf is None:
            psf = spaxlet.DeltaPSF(len(channels))
        return spaxlet.Observation(
            np.zeros((len(channels), 5, 7)),
            channels=channels,
            psf=psf,
            wavelengths=wavelengths,
        )

    def test_matching_physical_grids_accept_convertible_units(self):
        frame = self._frame(np.array([1.0, 1.1, 1.2]) * u.um)
        observation = self._observation(np.array([1000, 1100, 1200]) * u.nm)
        self.assertIs(observation.match(frame).model_frame, frame)

    def test_equal_channel_labels_cannot_hide_wavelength_mismatch(self):
        frame = self._frame(np.array([1.0, 1.1, 1.2]) * u.um)
        observation = self._observation(np.array([1.0, 1.11, 1.2]) * u.um)
        with self.assertRaisesRegex(ValueError, "wavelengths disagree"):
            observation.match(frame)

    def test_mapped_subset_uses_corresponding_model_wavelengths(self):
        frame = self._frame(
            np.array([1.0, 1.1, 1.2, 1.3]) * u.um,
            channels=["a", "b", "c", "d"],
        )
        observation = self._observation(
            np.array([1.1, 1.2]) * u.um,
            channels=["b", "c"],
            psf=frame.psf,
        )
        self.assertIs(observation.match(frame).model_frame, frame)

    def test_noncontiguous_subset_maps_model_and_psf_channels(self):
        model_psfs = np.zeros((4, 3, 3), dtype=float)
        observed_psfs = np.zeros((2, 3, 3), dtype=float)
        for index in range(4):
            model_psfs[index, 1, 1] = 1
            model_psfs[index, 1, 0] = 0.05 * index
        for index in range(2):
            observed_psfs[index, 1, 1] = 1
            observed_psfs[index, 0, 1] = 0.1 * (index + 1)

        full_frame = spaxlet.Frame(
            (4, 5, 7),
            channels=["a", "b", "c", "d"],
            psf=spaxlet.ImagePSF(model_psfs.copy()),
            wavelengths=np.array([1.0, 1.1, 1.2, 1.3]) * u.um,
        )
        observation = self._observation(
            np.array([1.0, 1.2]) * u.um,
            channels=["a", "c"],
            psf=spaxlet.ImagePSF(observed_psfs.copy()),
        ).match(full_frame)

        subset_frame = spaxlet.Frame(
            (2, 5, 7),
            channels=["a", "c"],
            psf=spaxlet.ImagePSF(model_psfs[[0, 2]].copy()),
            wavelengths=np.array([1.0, 1.2]) * u.um,
        )
        reference = self._observation(
            np.array([1.0, 1.2]) * u.um,
            channels=["a", "c"],
            psf=spaxlet.ImagePSF(observed_psfs.copy()),
        ).match(subset_frame)

        model = np.arange(4 * 5 * 7, dtype=float).reshape(4, 5, 7)
        np.testing.assert_allclose(
            observation.render(model),
            reference.render(model[[0, 2]]),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            observation.renderer.render_channels(model, 0, 1),
            reference.renderer.render_channels(model[[0, 2]], 0, 1),
            rtol=0,
            atol=1e-12,
        )
        gradient = grad(lambda value: np.sum(observation.render(value)))(model)
        self.assertTrue(np.any(gradient[0] != 0))
        self.assertTrue(np.all(gradient[1] == 0))
        self.assertTrue(np.any(gradient[2] != 0))
        self.assertTrue(np.all(gradient[3] == 0))

    def test_one_sided_wavelength_metadata_is_rejected(self):
        frame = self._frame(np.array([1.0, 1.1, 1.2]) * u.um)
        with self.assertRaisesRegex(ValueError, "both declare wavelengths"):
            self._observation().match(frame)

    def test_broadband_default_remains_compatible(self):
        frame = self._frame()
        self.assertIs(self._observation().match(frame).model_frame, frame)

    def test_invalid_physical_grids_fail_early(self):
        invalid = (
            np.array([1.0, 1.1, 1.2]),
            np.array([1.0, 1.1]) * u.um,
            np.array([1.0, np.nan, 1.2]) * u.um,
            np.array([1.0, 1.0, 1.2]) * u.um,
            np.array([1.0, 1.1, 1.2]) * u.s,
        )
        for wavelengths in invalid:
            with self.subTest(wavelengths=wavelengths):
                with self.assertRaises((TypeError, ValueError)):
                    self._frame(wavelengths)

        with self.assertRaisesRegex(ValueError, "identifiers must be unique"):
            self._frame(
                np.array([1.0, 1.1, 1.2]) * u.um,
                channels=["a", "a", "c"],
            )


if __name__ == "__main__":
    unittest.main()

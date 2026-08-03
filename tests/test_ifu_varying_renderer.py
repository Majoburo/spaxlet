"""The opt-in field-dependent IFU renderer is linear and adjoint-correct."""

import unittest

import numpy as np
from astropy import units as u
from autograd import grad
from scipy.signal import convolve2d, correlate2d

import spaxlet


CHANNELS = ("a", "b", "c")
OBSERVED_CHANNELS = ("a", "c")
SHAPE = (7, 9)
ANCHORS = (np.array([1.0, 5.0]), np.array([2.0, 7.0]))


def _kernel(y_offset, x_offset):
    value = np.zeros((3, 3), dtype=float)
    value[1, 1] = 1
    value[1 + y_offset, 1 + x_offset] = 0.2
    return value / value.sum()


def _field_psfs():
    first = np.stack(
        [_kernel(0, -1), _kernel(-1, 0), _kernel(1, 0), _kernel(0, 1)]
    )
    second = np.stack(
        [_kernel(-1, -1), _kernel(-1, 1), _kernel(1, -1), _kernel(1, 1)]
    )
    return np.stack((first, second))


class IFUVaryingRendererTest(unittest.TestCase):
    def _frames(self, field_psfs=None):
        wavelengths = np.array([1.0, 1.1, 1.2]) * u.um
        frame = spaxlet.Frame(
            (3, *SHAPE),
            channels=CHANNELS,
            psf=spaxlet.DeltaPSF(3),
            wavelengths=wavelengths,
            dtype=np.float64,
        )
        observation = spaxlet.Observation(
            np.zeros((2, *SHAPE), dtype=float),
            channels=OBSERVED_CHANNELS,
            psf=spaxlet.DeltaPSF(2),
            wavelengths=wavelengths[[0, 2]],
        )
        if field_psfs is None:
            field_psfs = _field_psfs()
        renderer = spaxlet.SpatiallyVaryingConvolutionRenderer(
            observation, frame, field_psfs, ANCHORS
        )
        return frame, observation.match(frame, renderer=renderer)

    def test_matches_independent_direct_convolution_and_chunks(self):
        _, observation = self._frames()
        rng = np.random.default_rng(4)
        model = rng.normal(size=(3, *SHAPE))
        weights = spaxlet.spatial_interpolation_weights(ANCHORS, SHAPE)
        psfs = _field_psfs()
        expected = np.zeros((2, *SHAPE))
        for channel, model_channel in enumerate((0, 2)):
            for anchor in range(weights.shape[0]):
                expected[channel] += convolve2d(
                    weights[anchor] * model[model_channel],
                    psfs[channel, anchor],
                    mode="same",
                )

        rendered = observation.render(model)
        np.testing.assert_allclose(rendered, expected, rtol=0, atol=2e-12)
        chunked = np.concatenate(
            [
                observation.renderer.render_channels(model, 0, 1),
                observation.renderer.render_channels(model, 1, 2),
            ]
        )
        np.testing.assert_allclose(chunked, rendered, rtol=0, atol=2e-12)

    def test_autograd_is_the_exact_spatial_adjoint(self):
        _, observation = self._frames()
        rng = np.random.default_rng(7)
        model = rng.normal(size=(3, *SHAPE))
        observed = rng.normal(size=(2, *SHAPE))
        gradient = grad(
            lambda value: np.sum(observation.render(value) * observed)
        )(model)

        weights = spaxlet.spatial_interpolation_weights(ANCHORS, SHAPE)
        psfs = _field_psfs()
        expected = np.zeros_like(model)
        for channel, model_channel in enumerate((0, 2)):
            for anchor in range(weights.shape[0]):
                expected[model_channel] += weights[anchor] * correlate2d(
                    observed[channel], psfs[channel, anchor], mode="same"
                )
        np.testing.assert_allclose(gradient, expected, rtol=0, atol=3e-12)
        self.assertTrue(np.all(gradient[1] == 0))

    def test_uniform_anchor_grid_matches_shift_invariant_renderer(self):
        base = np.stack((_kernel(0, -1), _kernel(1, 0)))
        field_psfs = np.repeat(base[:, None], 4, axis=1)
        frame, varying = self._frames(field_psfs)
        regular = spaxlet.Observation(
            np.zeros((2, *SHAPE), dtype=float),
            channels=OBSERVED_CHANNELS,
            psf=spaxlet.ImagePSF(base.copy()),
            wavelengths=np.array([1.0, 1.2]) * u.um,
        ).match(frame)
        model = np.random.default_rng(10).normal(size=(3, *SHAPE))
        np.testing.assert_allclose(
            varying.render(model), regular.render(model), rtol=0, atol=2e-12
        )

    def test_interpolation_is_a_partition_of_unity(self):
        weights = spaxlet.spatial_interpolation_weights(ANCHORS, SHAPE)
        self.assertEqual(weights.shape, (4, *SHAPE))
        np.testing.assert_allclose(np.sum(weights, axis=0), 1, rtol=0, atol=3e-16)
        np.testing.assert_array_equal(weights[:, 1, 2], [1, 0, 0, 0])
        np.testing.assert_array_equal(weights[:, 5, 7], [0, 0, 0, 1])

    def test_storage_accounting_excludes_duplicate_kernel_cube(self):
        _, observation = self._frames()
        renderer = observation.renderer
        self.assertEqual(
            renderer.storage_bytes,
            renderer.weights.nbytes + renderer.kernel_fft.nbytes,
        )
        self.assertFalse(hasattr(renderer, "field_psfs"))

    def test_non_delta_model_frame_is_rejected(self):
        wavelengths = np.array([1.0, 1.2]) * u.um
        frame = spaxlet.Frame(
            (2, *SHAPE),
            channels=OBSERVED_CHANNELS,
            psf=spaxlet.GaussianPSF(np.array([0.3, 0.3])),
            wavelengths=wavelengths,
            dtype=np.float64,
        )
        observation = spaxlet.Observation(
            np.zeros((2, *SHAPE)),
            channels=OBSERVED_CHANNELS,
            psf=spaxlet.DeltaPSF(2),
            wavelengths=wavelengths,
        )
        with self.assertRaisesRegex(ValueError, "DeltaPSF"):
            spaxlet.SpatiallyVaryingConvolutionRenderer(
                observation, frame, _field_psfs(), ANCHORS
            )

    def test_malformed_field_contracts_fail_early(self):
        frame, observation = self._frames()
        invalid = (
            (_field_psfs()[:, :3], ANCHORS),
            (_field_psfs()[0], ANCHORS),
            (np.zeros_like(_field_psfs()), ANCHORS),
            (_field_psfs(), (np.array([1.0, 1.0]), ANCHORS[1])),
        )
        for psfs, anchors in invalid:
            with self.subTest(shape=psfs.shape, anchors=anchors):
                with self.assertRaises(ValueError):
                    spaxlet.SpatiallyVaryingConvolutionRenderer(
                        observation, frame, psfs, anchors
                    )


if __name__ == "__main__":
    unittest.main()

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from functools import partial
import scarlet


class TestObservation(object):
    def get_psfs(self, sigmas, boxsize):
        psf = scarlet.GaussianPSF(sigmas, boxsize=boxsize)
        return psf

    def test_render_loss(self):
        # model frame with minimal PSF
        shape0 = (3, 13, 13)
        s0 = 0.9
        model_psf = scarlet.GaussianPSF(s0, boxsize=shape0[1])
        model_psf_image = model_psf.get_model()

        shape = (3, 43, 43)
        channels = np.arange(shape[0])
        model_frame = scarlet.Frame(shape, psf=model_psf, channels=channels)

        # insert point source manually into center for model
        origin = (0, shape[1] // 2 - shape0[1] // 2, shape[2] // 2 - shape0[2] // 2)
        bbox = scarlet.Box(shape0, origin=origin)
        model = np.zeros(shape)
        box = np.stack([model_psf_image[0] for c in range(shape[0])], axis=0)
        bbox.insert_into(model, box)

        # generate observation with wider PSFs
        psf = scarlet.GaussianPSF([2.1, 1.1, 3.5], boxsize=shape[1])
        psf_image = psf.get_model()
        images = np.ones(shape)
        observation = scarlet.Observation(images, psf=psf, channels=channels)
        observation.match(model_frame)
        model_ = observation.render(model)
        assert_almost_equal(model_, psf_image)

        # compute the expected loss
        weights = 1
        log_norm = (
            np.prod(images.shape) / 2 * np.log(2 * np.pi)
            + np.sum(np.log(1 / weights)) / 2
        )
        true_loss = log_norm + np.sum(weights * (model_ - images) ** 2) / 2
        # loss is negative logL
        assert_almost_equal(observation.get_log_likelihood(model), -true_loss)

    def test_delta_model_psf_applies_observation_kernel_directly(self):
        channels = np.arange(3)
        shape = (3, 31, 29)
        model_frame = scarlet.Frame(
            shape, psf=scarlet.DeltaPSF(len(channels)), channels=channels
        )
        observation_psf = scarlet.GaussianPSF(
            [0.8, 1.2, 1.7], boxsize=11
        )
        observation = scarlet.Observation(
            np.zeros(shape), psf=observation_psf, channels=channels
        ).match(model_frame)
        model = np.zeros(shape)
        model[:, shape[1] // 2, shape[2] // 2] = 1.0
        rendered = observation.render(model)
        expected = np.zeros(shape)
        kernel = observation_psf.get_model()
        y0 = shape[1] // 2 - kernel.shape[1] // 2
        x0 = shape[2] // 2 - kernel.shape[2] // 2
        expected[:, y0 : y0 + kernel.shape[1], x0 : x0 + kernel.shape[2]] = kernel
        assert_almost_equal(rendered, expected, decimal=7)

    def test_delta_psf_validates_channel_count(self):
        for invalid in (0, -1, 1.5):
            try:
                scarlet.DeltaPSF(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("DeltaPSF accepted invalid channel count")

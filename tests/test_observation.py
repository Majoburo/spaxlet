import numpy as np
from autograd import grad
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

    def test_channel_chunked_likelihood_and_gradient_match_full_render(self):
        rng = np.random.RandomState(13)
        shape = (7, 17, 15)
        channels = np.arange(shape[0])
        model_frame = scarlet.Frame(
            shape,
            psf=scarlet.DeltaPSF(shape[0], dtype=np.float32),
            channels=channels,
            dtype=np.float32,
        )
        observation = scarlet.Observation(
            rng.normal(size=shape).astype(np.float32),
            psf=scarlet.GaussianPSF(
                np.linspace(0.7, 1.6, shape[0]), boxsize=9
            ),
            weights=rng.uniform(0.2, 1.5, size=shape).astype(np.float32),
            channels=channels,
        ).match(model_frame)
        model = rng.normal(size=shape).astype(np.float32)

        full = observation.get_log_likelihood(model)
        chunked = observation.get_log_likelihood(model, channel_chunk_size=3)
        assert_almost_equal(chunked, full, decimal=4)

        full_gradient = grad(
            lambda value: -observation.get_log_likelihood(value)
        )(model)
        chunked_gradient = grad(
            lambda value: -observation.get_log_likelihood(
                value, channel_chunk_size=3
            )
        )(model)
        np.testing.assert_allclose(
            chunked_gradient, full_gradient, rtol=2e-5, atol=2e-5
        )

    def test_parameter_optimality_vanishes_and_is_scale_gauge_invariant(self):
        rng = np.random.RandomState(21)
        shape = (3, 9, 9)
        channels = np.arange(shape[0])
        psf = scarlet.DeltaPSF(shape[0])
        frame = scarlet.Frame(shape, psf=psf, channels=channels)
        truth_spectrum = np.asarray([0.7, 1.1, 1.6])
        truth_morphology = rng.uniform(0.2, 1.0, size=shape[1:])
        data = truth_spectrum[:, None, None] * truth_morphology[None]
        observation = scarlet.Observation(
            data,
            psf=psf,
            weights=np.full(shape, 4.0),
            channels=channels,
        ).match(frame)

        def make_blend(spectrum, morphology):
            source = scarlet.FactorizedComponent(
                frame,
                scarlet.TabulatedSpectrum(frame, np.asarray(spectrum)),
                scarlet.ImageMorphology(
                    frame, np.asarray(morphology), resizing=False
                ),
            )
            return scarlet.Blend([source], observation)

        stationary = make_blend(truth_spectrum, truth_morphology)
        stationary_residual = (
            stationary.parameter_optimization_diagnostics()
            .relative_projected_gradient
        )
        assert stationary_residual < 1e-10

        spectrum = truth_spectrum * np.asarray([1.2, 0.8, 1.1])
        morphology = truth_morphology + 0.05 * rng.normal(size=shape[1:])
        base = make_blend(spectrum, morphology)
        rescaled = make_blend(spectrum / 1e3, morphology * 1e3)
        base_diagnostic = base.parameter_optimization_diagnostics()
        rescaled_diagnostic = rescaled.parameter_optimization_diagnostics()
        assert base_diagnostic.relative_projected_gradient > 1e-3
        np.testing.assert_allclose(
            rescaled_diagnostic.relative_projected_gradient,
            base_diagnostic.relative_projected_gradient,
            rtol=2e-5,
        )
        np.testing.assert_allclose(
            rescaled_diagnostic.morphology_relative_projected_gradient,
            base_diagnostic.morphology_relative_projected_gradient,
            rtol=2e-5,
        )

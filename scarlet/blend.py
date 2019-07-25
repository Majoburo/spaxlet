import autograd.numpy as np
from autograd import grad

from .component import ComponentTree

import logging

logger = logging.getLogger("scarlet.blend")


class Blend(ComponentTree):
    """The blended scene

    The class represents a scene as collection of components, internally as a
    `~scarlet.component.ComponentTree`, and provides the functions to fit it
    to data.

    Attributes
    ----------
    mse: list
        Array of mean squared errors in each iteration
    """

    def __init__(self, sources, observations):
        """Constructor

        Form a blended scene from a collection of `~scarlet.component.Component`s

        Parameters
        ----------
        sources: list of `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
            Intitialized components or sources to fit to the observations
        observations: a `scarlet.Observation` instance or a list thereof
            Data package(s) to fit
        """
        ComponentTree.__init__(self, sources)

        try:
            iter(observations)
        except TypeError:
            observations = (observations,)
        self.observations = observations

    def fit(self, max_iter=200, e_rel=1e-3, step_size=1e-2, b1=0.5, b2=0.999, prox_iter=1):
        """Fit the model for each source to the data

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations if the algorithm doesn't converge.
        e_rel: float
            Relative error for convergence of each component.
        """
        assert b1 >= 0 and b1 < 1
        assert b2 >= 0 and b2 < 1
        assert prox_iter == 1 or (prox_iter > 1 and prox_iter % 2 == 0)

        # dynamically call parameters to allow for addition / fixing
        x = self.parameters
        n_params = len(x)
        m = [np.zeros(x_.shape, x_.dtype) for x_ in x]
        v = [np.zeros(x_.shape, x_.dtype) for x_ in x]
        vhat = [np.zeros(x_.shape, x_.dtype) for x_ in x]
        h = [np.zeros(x_.shape, x_.dtype) for x_ in x]

        priors = [p.prior for p in x if p.prior is not None]
        batch_size = len(priors)
        priors = set(priors)
        if batch_size > 0:
            import tensorflow as tf

            assert len(priors) == 1, "Currently only supports a single morphology prior for all components"
            prior = priors[0]

            inx = tf.placeholder(shape=[batch_size, prior.stamp_size, prior.stamp_size, 1])
            grad_prior = prior.grad(inx)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            self._compute_grad_prior = lambda x: self.run(grad_prior, feed_dict={inx: x})

        # compute the backward gradient tree
        self._grad = grad(self._loss, tuple(range(n_params)))
        step_sizes = self._get_stepsizes(*x, step_size=step_size)
        e_rel2 = e_rel ** 2
        self.mse = []
        converged = False

        for it in range(max_iter):
            g = self._grad(*x)
            gp = self._grad_prior(*x)
            b1t = b1**(it+1)
            b1tm1 = b1**it

            # for convergence test
            x_ = [x_._data.copy() for x_ in x]

            # AdamX gradient updates
            for j in range(n_params):
                ggp = g[j] + gp[j]
                m[j] = (1 - b1t) * ggp + b1t * m[j]
                v[j] = (1 - b2) * ggp**2 + b2 * v[j]
                if it == 0:
                    vhat[j] = v[j]
                else:
                    vhat[j] = np.maximum(v[j], vhat[j] * (1 - b1t)**2 / (1 - b1tm1)**2)

                # inline update from gradients
                x[j] -= step_sizes[type(x[j])] * m[j] / np.sqrt(vhat[j])

                # step size for prox sub-iterations below
                h[j] = np.sqrt(vhat[j])
                x[j].step = 1 / np.max(vhat[j])

            if prox_iter > 1:
                z = [x_._data.copy() for x_ in x]
                for t in range(prox_iter):
                    # subiterations for proxes in accelerated Adam
                    for j in range(n_params):
                        x[j][:] = x[j]._data - x[j].step * h[j] * (x[j]._data - z[j])
                    self.update()
            else:
                self.update()

            # convergence test *after* proximal updates
            converged = True
            for j in range(n_params):
                x[j].converged = np.sum((x_[j] - x[j]._data)**2) <= e_rel2 * np.sum(x[j]._data**2)
                converged &= x[j].converged

            if converged:
                break
            # additionally allow for f value convergence
            elif it > 5:
                prev = np.mean(self.mse[-5:-2])
                current = np.mean(self.mse[-4:])
                if it > 5 and abs(prev - current) < e_rel * current:
                    break

        return self

    def _loss(self, *parameters):
        """Loss function for autograd

        This method combines the seds and morphologies
        into a model that is used to calculate the loss
        function and update the gradient for each
        parameter
        """
        model = self.get_model(*parameters)
        # Caculate the total loss function from all of the observations
        total_loss = 0
        for observation in self.observations:
            total_loss = total_loss + observation.get_loss(model)
        self.mse.append(total_loss._value)
        return total_loss

    def _grad_prior(self, *parameters):
        # TODO: could use collecting identical priors to run on mini-batches
        #return [ p.prior(p.view(np.ndarray)) if p.prior is not None else 0 for p in parameters ]
        batch = []
        for p in parameters:
            if p.prior is not None:
                bbox, padding = p.get_centered_ROI(p.prior.stamp_size)
                roi = np.pad(p[bbox.slices], padding, mode='constant')
                batch.append(roi.reshape((1, p.prior.stamp_size, p.prior.stamp_size, 1)))

        if len(batch) == 0:
            return [0,]*len(parameters)

        # Concatenate stamps and feed them to the network
        batch = self._compute_grad_prior(np.stack(batch, axis=0))

        # Extract the results and interleave 0s for parameters not affected by
        # prior
        grad_prior = []
        ind = 0
        for p in parameters:
            if p.prior is not None:
                gp = np.zeros(p.shape, dtype=p.dtype)
                bbox, padding = p.get_centered_ROI(p.prior.stamp_size)
                (bottom, top), (left, right) = padding
                top = None if top == 0 else -top
                right = None if right == 0 else -right

                gp[bbox.slices] = batch[ind++][bottom:top, left:right]

                grad_prior.append(gp)
            else:
                grad_prior.append(0)

        return grad_prior

    def _get_stepsizes(self, *parameters, step_size=1e-3):
        # pick step as a fraction of the mac value of the parameter
        # since we don't want faint sources to slow down, we set all sources to
        # the same step size
        step_sizes = {}
        for p in parameters:
            t = type(p)
            step_sizes[t] = max(step_size * p._data.max(), step_sizes.get(t, 0))

        return step_sizes

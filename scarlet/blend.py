from collections import namedtuple
from functools import partial

import numpy.ma as ma
import autograd.numpy as np
from autograd import grad
from autograd.extend import defvjp, primitive
import proxmin
import logging

from .bbox import overlapped_slices
from .component import CombinedComponent
from .model import UpdateException

logger = logging.getLogger("scarlet.blend")


ParameterOptimizationDiagnostics = namedtuple(
    "ParameterOptimizationDiagnostics",
    (
        "relative_projected_gradient",
        "spectral_relative_projected_gradient",
        "morphology_relative_projected_gradient",
        "projected_gradient_norm",
        "weighted_data_energy",
        "parameter_relative_projected_gradients",
    ),
)


@primitive
def _add_models(*models, full_model, slices):
    """Insert the models into the full model

    `slices` is a tuple `(full_model_slice, model_slices)` used
    to insert a model into the full_model in the region where the
    two models overlap.
    """
    for i in range(len(models)):
        full_model[slices[i][0]] += models[i][slices[i][1]]
    return full_model


def _grad_add_models(upstream_grad, *models, full_model, slices, index):
    """Gradient for a single model

    The full model is just the sum of the models,
    so the gradient is 1 for each model,
    we just have to slice it appropriately.
    """
    model = models[index]
    full_model_slices = slices[index][0]
    model_slices = slices[index][1]

    def result(upstream_grad):
        _result = np.zeros(model.shape, dtype=model.dtype)
        _result[model_slices] = upstream_grad[full_model_slices]
        return _result

    return result


class Blend(CombinedComponent):
    """The blended scene

    The class represents a scene as collection of and provides the functions to
    fit it to data.

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

        if hasattr(sources, "__iter__"):
            self.sources = sources
        else:
            self.sources = (sources,)

        if hasattr(observations, "__iter__"):
            self.observations = observations
        else:
            self.observations = (observations,)

        super().__init__(self.sources)

        # only for backward compatibility, use log_likelihood instead
        self.loss = []
        self._noise_factor = 0
        self._channel_chunk_size = None

    def fit(
        self,
        max_iter=200,
        e_rel=1e-3,
        min_iter=1,
        noise_factor=0,
        project_initial=False,
        channel_chunk_size=None,
        **alg_kwargs
    ):
        """Fit the model for each source to the data

        Parameters
        ----------
        max_iter: int
            Maximum number of iterations if the algorithm doesn't converge
        e_rel: float
            Relative error for convergence of the loss function
        min_iter: int
            Maximum number of iterations if the algorithm doesn't converge
        alg_kwargs: dict
            Keywords for the `proxmin.adaprox` optimizer
        project_initial: bool
            Project every free starting parameter through its declared
            constraint before evaluating the first model. This is useful for
            caller-supplied factors: an infeasible start can otherwise create
            a large first proximal jump unrelated to the gradient. The default
            is ``False`` for backward compatibility.
        channel_chunk_size: int or None
            Render and score this many observation channels at a time. This
            lowers peak memory for compatible renderers without changing the
            objective.
        """
        if not isinstance(project_initial, (bool, np.bool_)):
            raise TypeError("project_initial must be boolean")
        if channel_chunk_size is not None:
            if not isinstance(channel_chunk_size, (int, np.integer)):
                raise TypeError("channel_chunk_size must be an integer or None")
            if channel_chunk_size <= 0:
                raise ValueError("channel_chunk_size must be positive")
        self.initial_projection_relative_l2 = 0.0
        if project_initial:
            parameters = self.parameters + tuple(
                parameter
                for observation in self.observations
                for parameter in observation.parameters
            )
            change_squared = 0.0
            scale_squared = 0.0
            for parameter in parameters:
                if parameter.fixed or parameter.constraint is None:
                    continue
                original = parameter.copy()
                projected = np.asarray(parameter.constraint(original.copy(), 0))
                if projected.shape != parameter.shape:
                    raise ValueError("an initial projection changed parameter shape")
                if not np.all(np.isfinite(projected)):
                    raise ValueError("an initial projection produced non-finite values")
                parameter[...] = projected
                change_squared += float(np.sum((projected - original) ** 2))
                scale_squared += float(np.sum(original**2))
            self.initial_projection_relative_l2 = np.sqrt(change_squared) / max(
                np.sqrt(scale_squared), np.finfo(float).tiny
            )
        it = 0
        self._noise_factor = noise_factor
        self._channel_chunk_size = channel_chunk_size
        while it < max_iter:
            try:
                X = self.parameters + tuple(
                    p for obs in self.observations for p in obs.parameters
                )

                # compute the backward gradients
                # but only for non-fixed parameters
                require_grad = tuple(k for k, x in enumerate(X) if not x.fixed)

                def expand_grads(*X, func=None):
                    G = func(*X)
                    expanded = [0.0] * len(X)
                    for k, j in enumerate(require_grad):
                        expanded[j] = G[k]
                    return expanded

                grad_logL_func = grad(self._loss_func, require_grad)
                grad_logL = lambda *X: expand_grads(*X, func=grad_logL_func)

                # same for prior. easier her bc we call them independently
                grad_logP = lambda *X: tuple(
                    x.prior(x.view(np.ndarray))
                    if x.prior is not None and not x.fixed
                    else 0
                    for x in X
                )

                # combine for log posterior
                _grad = lambda *X: tuple(
                    l + p for l, p in zip(grad_logL(*X), grad_logP(*X))
                )

                # step sizes
                _step = lambda *X, it: tuple(
                    x.step(x, it=it) if hasattr(x.step, "__call__") else x.step
                    for x in X
                )

                # proxes
                _prox = tuple(x.constraint for x in X)

                # good defaults for adaprox
                scheme = alg_kwargs.pop("scheme", "amsgrad")
                prox_max_iter = alg_kwargs.pop("prox_max_iter", 10)
                callback = partial(
                    self._callback,
                    e_rel=e_rel,
                    callback=alg_kwargs.pop("callback", None),
                    min_iter=min_iter,
                )

                # do we have a current state of the optimizer to warm start?
                for x in X:
                    if x.m is None:
                        x.m = np.zeros(x.shape)
                    if x.v is None:
                        x.v = np.zeros(x.shape)
                    if x.vhat is None:
                        x.vhat = np.zeros(x.shape)
                M = tuple(x.m for x in X)
                V = tuple(x.v for x in X)
                Vhat = tuple(x.vhat for x in X)

                proxmin.adaprox(
                    X,
                    _grad,
                    _step,
                    prox=_prox,
                    max_iter=max_iter - it,
                    e_rel=e_rel,
                    check_convergence=False,
                    scheme=scheme,
                    prox_max_iter=prox_max_iter,
                    callback=callback,
                    M=M,
                    V=V,
                    Vhat=Vhat,
                    **alg_kwargs
                )

                logger.info(
                    "scarlet ran for {0} iterations to logL = {1}".format(
                        len(self.log_likelihood), self.log_likelihood[-1]
                    )
                )

                # set convergence and standard deviation from optimizer
                for p, m, v, vhat in zip(X, M, V, Vhat):
                    p.std = 1 / np.sqrt(
                        ma.masked_equal(v, 0)
                    )  # this is rough estimate!

                return len(self.log_likelihood), self.log_likelihood[-1]

            # model update forces restart
            except UpdateException:
                it = len(self.log_likelihood)

    def get_model(self, *parameters, frame=None):
        """Get the model of the entire blend

        Parameters
        ----------
        parameters: tuple of optimization parameters
        frame:  `scarlet.Frame`
            Alternative Frame to project the model into

        Returns
        -------
        model: array
            (Bands, Height, Width) data cube
        """

        # boxed models of every source
        models = self.get_models_of_children(*parameters, frame=None)

        if frame is None:
            frame = self.frame

        # if this is the model frame then the slices are already cached
        if frame == self.frame:
            slices = tuple(
                (src._model_frame_slices, src._model_slices) for src in self.sources
            )
        else:
            slices = tuple(
                overlapped_slices(frame.bbox, src.bbox) for src in self.sources
            )

        # We have to declare the function that inserts sources
        # into the blend with autograd.
        # This has to be done each time we fit a blend,
        # since the number of components => the number of arguments,
        # which must be linked to the autograd primitive function.
        defvjp(
            _add_models,
            *([partial(_grad_add_models, index=k) for k in range(len(self.sources))])
        )

        full_model = np.zeros(frame.shape, dtype=frame.dtype)
        full_model = _add_models(*models, full_model=full_model, slices=slices)

        return full_model

    @property
    def log_likelihood(self):
        """Log likelihood at each iteration

        The fitting method computes and sums the negative log-likelihood for the each
        observation given the current model as loss function for the optimization.

        Returns
        -------
        log_likelihood: array of (positive) log-likelihood
        """
        return -np.array(self.loss)

    def _loss_func(self, *parameters):
        total_loss = self._objective_func(*parameters)
        self.loss.append(getattr(total_loss, "_value", total_loss))
        return total_loss

    def _objective_func(self, *parameters):
        n_params = len(self.parameters)
        model = self.get_model(*parameters[:n_params], frame=self.frame)

        # Caculate the total loss function from all of the observations
        total_loss = 0
        for observation in self.observations:
            n_obs_params = len(observation.parameters)
            obs_params = parameters[n_params : n_params + n_obs_params]
            total_loss = total_loss - observation.get_log_likelihood(
                model,
                *obs_params,
                noise_factor=self._noise_factor,
                channel_chunk_size=self._channel_chunk_size,
            )
            n_params += n_obs_params

        return total_loss

    def parameter_optimization_diagnostics(self, probe=1e-5):
        """Return a dimensionless proximal first-order residual.

        Each free parameter is probed with a gauge-covariant step
        ``probe * ||x||**2 / E``, where ``E`` is the larger of the weighted
        data energy and current objective magnitude. The resulting projected
        gradient mapping is normalized as ``||r|| ||x|| / E``. Morphology
        mappings have their radial component removed because that direction
        is the exact spectrum--morphology scale gauge; spectral stationarity
        is reported separately so a physical amplitude error is not hidden.

        This diagnostic evaluates the declared constraints but does not alter
        parameters or append to the fit history. It is intended as a KKT-style
        stopping and comparison check, not as an identifiability diagnostic.
        """
        probe = float(probe)
        if not np.isfinite(probe) or probe <= 0:
            raise ValueError("probe must be finite and positive")
        if self._noise_factor > 0:
            raise ValueError(
                "optimization diagnostics require deterministic noise_factor=0"
            )

        parameters = self.parameters + tuple(
            parameter
            for observation in self.observations
            for parameter in observation.parameters
        )
        free_indices = tuple(
            index for index, parameter in enumerate(parameters) if not parameter.fixed
        )
        weighted_data_energy = float(
            sum(
                np.sum(observation.weights * observation.data**2)
                for observation in self.observations
            )
        )
        if not free_indices:
            return ParameterOptimizationDiagnostics(
                0.0, 0.0, 0.0, 0.0, weighted_data_energy, ()
            )

        gradient_function = grad(self._objective_func, free_indices)
        gradients = gradient_function(*parameters)
        if len(free_indices) == 1 and not isinstance(gradients, tuple):
            gradients = (gradients,)
        objective = float(self._objective_func(*parameters))
        scale = max(
            abs(objective), weighted_data_energy, np.finfo(float).tiny
        )

        total_energy = 0.0
        spectral_energy = 0.0
        morphology_energy = 0.0
        absolute_mapping_energy = 0.0
        parameter_residuals = []
        for index, gradient_value in zip(free_indices, gradients):
            parameter = parameters[index]
            gradient_value = np.asarray(gradient_value, dtype=float)
            if parameter.prior is not None:
                gradient_value = gradient_value + parameter.prior(
                    parameter.view(np.ndarray)
                )
            value = np.asarray(parameter.view(np.ndarray), dtype=float)
            value_norm = float(np.linalg.norm(value))
            if value_norm <= np.finfo(float).tiny:
                parameter_residuals.append((parameter.name, 0.0))
                continue
            step = probe * value_norm**2 / scale
            trial = value - step * gradient_value
            if parameter.constraint is not None:
                projected = np.asarray(
                    parameter.constraint(trial.copy(), step), dtype=float
                )
            else:
                projected = trial
            mapping = (value - projected) / step
            if parameter.name in ("image", "coeffs"):
                radial = float(np.vdot(value, mapping).real) / value_norm**2
                mapping = mapping - radial * value
            mapping_norm = float(np.linalg.norm(mapping))
            relative = mapping_norm * value_norm / scale
            energy = relative**2
            total_energy += energy
            absolute_mapping_energy += mapping_norm**2
            if parameter.name == "spectrum":
                spectral_energy += energy
            elif parameter.name in ("image", "coeffs"):
                morphology_energy += energy
            parameter_residuals.append((parameter.name, relative))

        return ParameterOptimizationDiagnostics(
            float(np.sqrt(total_energy)),
            float(np.sqrt(spectral_energy)),
            float(np.sqrt(morphology_energy)),
            float(np.sqrt(absolute_mapping_energy)),
            weighted_data_energy,
            tuple(parameter_residuals),
        )

    def _callback(self, *parameters, it=None, e_rel=1e-3, callback=None, min_iter=1):

        # raises ArithmeticError if some of the parameters have become inf/nan
        for src in self.sources:
            src.check_parameters()

        # raises UpdateException if model updates require optimimzation interruption
        throw_exception = False
        if it > 0 and it % 10 == 0:
            for src in self.sources:
                try:
                    src.update()
                except UpdateException:
                    throw_exception = True
                    pass
            if throw_exception:
                raise UpdateException

        if it > min_iter and abs(self.loss[-1] - self.loss[-2]) < e_rel * np.abs(
            self.loss[-1]
        ):
            raise StopIteration(
                "scarlet.Blend.fit() converged"
            )  # clean return from proxmin

        if callback is not None:
            callback(*parameters, it=it)

    @property
    def bbox(self):
        """Bounding box of the blend
        """
        return self.frame.bbox

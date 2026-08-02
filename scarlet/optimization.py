"""Opt-in optimizers for constrained factorized spectral scenes."""

import numpy as np
from autograd import grad

from .component import FactorizedComponent
from .constraint import PositivityConstraint
from .morphology import ImageMorphology
from .spectrum import TabulatedSpectrum


def _solve_nonnegative_gram(gram, matched):
    """Solve one small non-negative quadratic from its sufficient statistics."""
    gram = np.asarray(gram, dtype=float)
    matched = np.asarray(matched, dtype=float)
    size = matched.size
    scale = np.sqrt(np.maximum(np.diag(gram), 0.0))
    scale[scale == 0.0] = 1.0
    equilibrated = gram / scale[:, None] / scale[None, :]
    target = matched / scale
    unconstrained = np.linalg.lstsq(equilibrated, target, rcond=None)[0]
    tolerance = 100.0 * np.finfo(float).eps * max(
        float(np.max(np.abs(unconstrained))), np.finfo(float).tiny
    )
    if np.all(unconstrained >= -tolerance):
        return np.maximum(unconstrained, 0.0) / scale

    value = np.zeros(size)
    passive = np.zeros(size, dtype=bool)
    dual = target.copy()
    iterations = 0
    dual_tolerance = 10.0 * max(size, 1) * np.finfo(float).eps * max(
        float(np.linalg.norm(target)), np.finfo(float).tiny
    )
    while np.any((~passive) & (dual > dual_tolerance)):
        passive[int(np.argmax(np.where(~passive, dual, -np.inf)))] = True
        while True:
            iterations += 1
            if iterations > max(30 * size, 1):
                raise RuntimeError("spectral NNLS active-set solver did not converge")
            candidate = np.zeros_like(value)
            candidate[passive] = np.linalg.lstsq(
                equilibrated[np.ix_(passive, passive)],
                target[passive],
                rcond=None,
            )[0]
            leaving = passive & (candidate <= 0.0)
            if not np.any(leaving):
                value = candidate
                break
            fraction = float(
                np.min(value[leaving] / (value[leaving] - candidate[leaving]))
            )
            value += fraction * (candidate - value)
            boundary = 10.0 * np.finfo(float).eps * max(
                float(np.max(np.abs(value))), np.finfo(float).tiny
            )
            at_boundary = passive & (value <= boundary)
            passive[at_boundary] = False
            value[at_boundary] = 0.0
        dual = target - equilibrated @ value
    return np.maximum(value, 0.0) / scale


def spectral_volume_value_gradient(spectra, strength):
    """Return normalized log-volume and its gradient for spectral columns."""
    spectra = np.asarray(spectra, dtype=float)
    strength = float(strength)
    gradient = np.zeros_like(spectra)
    if strength == 0.0 or spectra.shape[1] < 2:
        return 0.0, gradient
    norms = np.linalg.norm(spectra, axis=0)
    if np.any(norms <= np.finfo(float).tiny):
        return 0.0, gradient
    normalized = spectra / norms
    gram = normalized.T @ normalized + 1e-12 * np.eye(spectra.shape[1])
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0:
        return 0.0, gradient
    normalized_gradient = 2.0 * normalized @ np.linalg.inv(gram)
    for source in range(spectra.shape[1]):
        direction = normalized[:, source]
        value = normalized_gradient[:, source]
        gradient[:, source] = (
            value - direction * float(np.dot(direction, value))
        ) / norms[source]
    return strength * float(logdet), strength * gradient


def _channel_indices(observation, model_channels):
    mapping = observation.renderer.channel_map
    indices = np.arange(model_channels)
    if mapping is not None:
        indices = indices[mapping]
    indices = np.asarray(indices, dtype=int)
    if indices.shape != (observation.shape[0],):
        raise ValueError(
            "variable projection requires a channel-selection renderer"
        )
    if np.unique(indices).size != indices.size:
        raise ValueError("variable projection requires unique observation channels")
    return indices


def _validate_factorized_problem(blend):
    if not blend.sources:
        raise ValueError("variable projection requires at least one source")
    if any(observation.parameters for observation in blend.observations):
        raise ValueError(
            "variable projection does not support free observation parameters"
        )
    image_parameters = []
    for source in blend.sources:
        if not isinstance(source, FactorizedComponent):
            raise TypeError(
                "variable projection requires FactorizedComponent sources"
            )
        if not isinstance(source.spectrum, TabulatedSpectrum):
            raise TypeError(
                "variable projection requires TabulatedSpectrum spectra"
            )
        if not isinstance(source.morphology, ImageMorphology):
            raise TypeError(
                "variable projection requires ImageMorphology morphologies"
            )
        if source.morphology.shifting:
            raise ValueError(
                "variable projection does not yet support fitted morphology shifts"
            )
        spectrum = source.spectrum.parameters[0]
        image = source.morphology.parameters[0]
        if spectrum.shape != (blend.frame.C,):
            raise ValueError(
                "variable projection requires full-frame tabulated spectra"
            )
        if spectrum.fixed:
            raise ValueError("variable projection requires free spectra")
        if spectrum.prior is not None or image.prior is not None:
            raise ValueError(
                "variable projection does not yet support per-parameter priors"
            )
        if not isinstance(spectrum.constraint, PositivityConstraint):
            raise ValueError(
                "variable projection requires non-negative spectra"
            )
        if not image.fixed:
            if image.constraint is not None and not getattr(
                image.constraint, "is_euclidean_projection", False
            ):
                raise ValueError(
                    "scalar projected steps require exact Euclidean morphology projections"
                )
            image_parameters.append(image)

    parameter_ids = {id(parameter): index for index, parameter in enumerate(blend.parameters)}
    image_indices = tuple(parameter_ids[id(parameter)] for parameter in image_parameters)
    channel_indices = tuple(
        _channel_indices(observation, blend.frame.C)
        for observation in blend.observations
    )
    covered = np.zeros(blend.frame.C, dtype=bool)
    for indices in channel_indices:
        covered[indices] = True
    if not np.all(covered):
        raise ValueError(
            "variable projection requires every model channel to be observed"
        )
    return tuple(image_parameters), image_indices, channel_indices


def _source_basis(source, observation, morphology):
    parameters = [np.asarray(parameter) for parameter in source.parameters]
    for index, parameter in enumerate(source.parameters):
        if parameter.name == "spectrum":
            parameters[index] = np.ones_like(parameter)
        elif parameter.name == "image":
            parameters[index] = morphology
    model = source.get_model(*parameters, frame=source.frame)
    return np.asarray(observation.render(model), dtype=float)


def _spectral_statistics(blend, morphologies, channel_indices):
    n_channel = blend.frame.C
    n_source = len(blend.sources)
    grams = np.zeros((n_channel, n_source, n_source))
    matched = np.zeros((n_channel, n_source))
    constant = 0.0
    for observation, indices in zip(blend.observations, channel_indices):
        bases = np.asarray(
            [
                _source_basis(source, observation, morphology)
                for source, morphology in zip(blend.sources, morphologies)
            ]
        )
        flattened = bases.reshape(n_source, observation.C, -1)
        weights = np.asarray(observation.weights, dtype=float).reshape(
            observation.C, -1
        )
        data = np.asarray(observation.data, dtype=float).reshape(observation.C, -1)
        grams[indices] += np.einsum(
            "cp,kcp,lcp->ckl", weights, flattened, flattened, optimize=True
        )
        matched[indices] += np.einsum(
            "cp,kcp,cp->ck", weights, flattened, data, optimize=True
        )
        constant += float(observation.log_norm) + 0.5 * float(
            np.sum(weights * data**2)
        )
    return grams, matched, constant


def _spectral_objective(spectra, grams, matched, constant, volume_strength):
    quadratic = constant + 0.5 * float(
        np.einsum("ck,ckl,cl->", spectra, grams, spectra, optimize=True)
    ) - float(np.sum(matched * spectra))
    volume, _ = spectral_volume_value_gradient(spectra, volume_strength)
    return quadratic + volume


def _solve_regularized_spectra(
    initial,
    grams,
    matched,
    constant,
    strength,
    max_iter,
    tolerance,
):
    spectra = np.asarray(initial, dtype=float).copy()
    largest = max(
        float(np.max(np.linalg.eigvalsh(grams))), np.finfo(float).tiny
    )
    step = 1.0 / largest
    objective = _spectral_objective(
        spectra, grams, matched, constant, strength
    )
    for _ in range(max_iter):
        _, volume_gradient = spectral_volume_value_gradient(spectra, strength)
        gradient = np.einsum("ckl,cl->ck", grams, spectra) - matched
        gradient += volume_gradient
        accepted = False
        for backtrack in range(31):
            trial_step = step * 0.5**backtrack
            trial = np.maximum(spectra - trial_step * gradient, 0.0)
            trial_objective = _spectral_objective(
                trial, grams, matched, constant, strength
            )
            if trial_objective <= objective:
                change = float(np.linalg.norm(trial - spectra))
                scale = max(float(np.linalg.norm(spectra)), 1.0)
                spectra, objective = trial, trial_objective
                step = 1.5 * trial_step
                accepted = True
                break
        if not accepted or change <= tolerance * scale:
            break
    return spectra, objective


def _profile_spectra(
    blend,
    morphologies,
    channel_indices,
    volume_strength,
    spectral_max_iter,
    spectral_tolerance,
):
    grams, matched, constant = _spectral_statistics(
        blend, morphologies, channel_indices
    )
    spectra = np.asarray(
        [_solve_nonnegative_gram(gram, target) for gram, target in zip(grams, matched)]
    )
    if volume_strength > 0.0:
        spectra, objective = _solve_regularized_spectra(
            spectra,
            grams,
            matched,
            constant,
            volume_strength,
            spectral_max_iter,
            spectral_tolerance,
        )
    else:
        objective = _spectral_objective(spectra, grams, matched, constant, 0.0)
    return spectra, float(objective)


def fit_variable_projection(
    blend,
    *,
    max_iter,
    e_rel,
    min_iter,
    callback,
    projected_step,
    max_backtracks,
    volume_strength,
    spectral_max_iter,
    spectral_tolerance,
):
    """Fit factorized sources with exact spectra and projected morphologies."""
    image_parameters, image_indices, channel_indices = _validate_factorized_problem(
        blend
    )
    if max_backtracks < 0 or spectral_max_iter < 1:
        raise ValueError("optimizer iteration counts must be non-negative")
    if volume_strength < 0.0 or not np.isfinite(volume_strength):
        raise ValueError("volume_strength must be finite and non-negative")
    if spectral_tolerance <= 0.0 or not np.isfinite(spectral_tolerance):
        raise ValueError("spectral_tolerance must be finite and positive")

    morphologies = [
        np.asarray(source.morphology.parameters[0], dtype=float).copy()
        for source in blend.sources
    ]
    spectra, objective = _profile_spectra(
        blend,
        morphologies,
        channel_indices,
        volume_strength,
        spectral_max_iter,
        spectral_tolerance,
    )
    for source, values in zip(blend.sources, spectra.T):
        source.spectrum.parameters[0][...] = values

    if projected_step is None:
        steps = np.asarray(
            [
                parameter.step(parameter, it=0)
                if callable(parameter.step)
                else parameter.step
                for parameter in image_parameters
            ],
            dtype=float,
        )
    else:
        steps = np.full(len(image_parameters), float(projected_step))
    if np.any(~np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("projected morphology steps must be finite and positive")

    gradient_function = (
        grad(blend._objective_func, image_indices) if image_indices else None
    )
    for iteration in range(1, max_iter + 1):
        previous_objective = objective
        if gradient_function is not None:
            parameters = blend.parameters
            gradients = gradient_function(*parameters)
            if len(image_indices) == 1 and not isinstance(gradients, tuple):
                gradients = (gradients,)
            accepted = False
            for backtrack in range(max_backtracks + 1):
                fraction = 0.5**backtrack
                candidate = list(morphologies)
                for parameter, gradient_value, base_step in zip(
                    image_parameters, gradients, steps
                ):
                    source_index = next(
                        source_index
                        for source_index, source in enumerate(blend.sources)
                        if source.morphology.parameters[0] is parameter
                    )
                    trial_step = base_step * fraction
                    projected = np.asarray(
                        parameter.constraint(
                            morphologies[source_index] - trial_step * gradient_value,
                            trial_step,
                        )
                        if parameter.constraint is not None
                        else morphologies[source_index] - trial_step * gradient_value,
                        dtype=float,
                    )
                    total = float(np.sum(projected))
                    if total <= np.finfo(float).tiny:
                        break
                    candidate[source_index] = projected / total
                else:
                    trial_spectra, trial_objective = _profile_spectra(
                        blend,
                        candidate,
                        channel_indices,
                        volume_strength,
                        spectral_max_iter,
                        spectral_tolerance,
                    )
                    tolerance = 10.0 * np.finfo(float).eps * max(
                        abs(objective), 1.0
                    )
                    if trial_objective <= objective + tolerance:
                        morphologies = candidate
                        spectra = trial_spectra
                        objective = trial_objective
                        steps *= 1.5 * fraction
                        accepted = True
                        break
            if not accepted:
                steps *= 0.5 ** (max_backtracks + 1)

        for source, spectrum, morphology in zip(
            blend.sources, spectra.T, morphologies
        ):
            source.spectrum.parameters[0][...] = spectrum
            source.morphology.parameters[0][...] = morphology
        blend.loss.append(float(objective))
        try:
            blend._callback(
                *blend.parameters,
                it=iteration,
                e_rel=e_rel,
                callback=callback,
                min_iter=min_iter,
            )
        except StopIteration:
            break
        if (
            iteration > min_iter
            and abs(previous_objective - objective)
            < e_rel * max(abs(objective), 1.0)
        ):
            break
    return len(blend.log_likelihood), blend.log_likelihood[-1]

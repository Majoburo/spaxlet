from functools import partial

import numpy as np
import proxmin

from . import operator
from .cache import Cache


def _constraint_center(shape, center):
    if center is None:
        return (shape[0] // 2, shape[1] // 2)
    if len(center) != 2 or any(
        not isinstance(value, (int, np.integer)) for value in center
    ):
        raise ValueError("center must contain two integer pixel coordinates")
    center = tuple(int(value) for value in center)
    if any(value < 0 or value >= size for value, size in zip(center, shape)):
        raise ValueError("center must lie inside the morphology")
    return center


class Constraint:
    """Constraint base class

    Constraints encode expected properties of the solution.
    Mathematically, they are the consequence of adding potentially
    non-differentiable penalty functions to the model fitting loss function.

    As we use proximal gradient methods, all constraints act as proxmimal
    operators, i.e. they need to have the following signature:

        f(X, step) -> X'

    where X' is the closest point to X that satisfies the feasibility criterion
    of the penalty function.

    For reference, every operator of the `proxmin` package yields a valid
    `Constraint`.
    """

    is_euclidean_projection = False

    def __init__(self, f=None):
        """Constraint base class

        Parameters
        ----------
        f: proximal mapping
            Signature: f(X, step) -> X'
        """
        self.f = f

    def __call__(self, X, step):
        """Proximal mapping

        Parameters
        ----------
        X: array
            Optimimzation parameter
        step: float or array of same shape as X
            Step size for the proximal mapping

        Returns
        -------
        X': closest feasible match to X
        """
        if self.f is not None:
            return self.f(X, step)
        return X


class ConstraintChain:
    """An ordered list of `Constraint`s.

    Uses the concept of alternating projections onto convex sets to find
    solutions that are feasible according to a list of constraints.

    Parameters
    ----------
    constraints: list of `Constraint`
    repeat: int
        How often the constrain chain is repeated to ensure feasibility
    """

    def __init__(self, *constraints, repeat=1):
        assert isinstance(repeat, int) and repeat >= 1
        self.constraints = constraints
        self.repeat = repeat

    def __call__(self, X, step):
        for r in range(self.repeat):
            for c in self.constraints:
                X = c(X, step)
        return X


class DykstraConstraintChain(ConstraintChain):
    """Project onto the intersection of exact convex constraint sets.

    Unlike :class:`ConstraintChain`, Dykstra's correction terms converge to the
    closest point in the intersection, not merely to a feasible point. Every
    member must explicitly declare that it is an exact Euclidean projection;
    heuristic transforms such as :class:`MonotonicityConstraint` are rejected.

    Parameters
    ----------
    constraints: list of `Constraint`
        Exact Euclidean projections onto closed convex sets.
    max_iter: int
        Maximum number of complete Dykstra sweeps.
    rtol, atol: float
        Relative and absolute convergence tolerances for both iterate change
        and residual constraint violation.
    """

    is_euclidean_projection = True

    def __init__(self, *constraints, max_iter=10000, rtol=1e-10, atol=1e-12):
        if not constraints:
            raise ValueError("a Dykstra chain requires at least one constraint")
        invalid = [
            type(constraint).__name__
            for constraint in constraints
            if not getattr(constraint, "is_euclidean_projection", False)
        ]
        if invalid:
            raise ValueError(
                "Dykstra requires exact Euclidean projections; invalid: "
                + ", ".join(invalid)
            )
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(rtol) or rtol < 0:
            raise ValueError("rtol must be finite and non-negative")
        if not np.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and non-negative")
        super().__init__(*constraints, repeat=1)
        self.max_iter = int(max_iter)
        self.rtol = float(rtol)
        self.atol = float(atol)

    def __call__(self, X, step):
        original = np.asarray(X)
        result = original.copy()
        corrections = [np.zeros_like(result) for _ in self.constraints]
        scale = max(float(np.linalg.norm(original)), 1.0)
        threshold = self.atol + self.rtol * scale

        for _ in range(self.max_iter):
            previous = result.copy()
            for index, constraint in enumerate(self.constraints):
                shifted = result + corrections[index]
                projected = np.asarray(constraint(shifted.copy(), step))
                if projected.shape != original.shape:
                    raise ValueError("constraints in a chain must preserve shape")
                corrections[index] = shifted - projected
                result = projected

            change = float(np.linalg.norm(result - previous))
            if change <= threshold:
                violations = [
                    float(
                        np.linalg.norm(
                            np.asarray(constraint(result.copy(), step)) - result
                        )
                    )
                    for constraint in self.constraints
                ]
                if max(violations, default=0.0) <= threshold:
                    return result

        raise RuntimeError(
            "Dykstra projection did not converge in {} sweeps".format(self.max_iter)
        )


class PositivityConstraint(Constraint):
    """Allow only values not smaller than `zero`.
    """

    is_euclidean_projection = True

    def __init__(self, zero=0):
        self.zero = zero

    def __call__(self, X, step):
        X = np.maximum(X, self.zero)
        return X


class NormalizationConstraint(Constraint):
    def __init__(self, type="sum"):
        """Normalize X to unity.

        Parameters
        ----------
        type: in ['sum', 'max']
            Whether the sum or the maximum is set to unity.
        """
        type = type.lower()
        assert type in ["sum", "max"]
        self.type = type

    def __call__(self, X, step):

        if self.type == "sum":
            X /= X.sum()
        else:
            X /= X.max()
        return X


class L0Constraint(Constraint):
    def __init__(self, thresh, type="absolute"):
        """L0 norm (sparsity) penalty

        Parameters
        ----------
        thresh: float
            regularization strength
        type: ['relative', 'absolute']
            if the penalty is expressed in units of the function value (relative)
            or in units of the variable X (absolute).
        """
        super().__init__(
            partial(proxmin.operators.prox_hard, thresh=thresh, type=type,)
        )


class L1Constraint(Constraint):
    def __init__(self, thresh, type="absolute"):
        """L1 norm (sparsity) penalty

        Parameters
        ----------
        thresh: regularization strength
        type: ['relative', 'absolute']
            if the penalty is expressed in units of the function value (relative)
            or in units of the variable X (absolute).
        """
        super().__init__(partial(proxmin.operators.prox_soft, thresh=thresh, type=type))


class ThresholdConstraint(Constraint):
    """Set a cutoff threshold for pixels below the noise

    Use the log histogram of pixel values to determine when the
    source is fitting noise. This function works well to prevent
    faint sources from growing large footprints but for large
    diffuse galaxies with a wide range of pixel values this
    does not work as well.

    The region that contains flux above the threshold is contained
    in `component.bboxes["thresh"]`.
    """

    def __call__(self, X, step):
        thresh, _bins = self.threshold(X)
        return proxmin.operators.prox_hard_plus(X, step, thresh=thresh, type="absolute")

    def threshold(self, morph):
        """Find the threshold value for a given morphology
        """
        _morph = morph[morph > 0]
        _bins = 50
        # Decrease the bin size for sources with a small number of pixels
        if _morph.size < 500:
            _bins = max(int(_morph.size / 10), 1)
            if _bins == 1:
                return 0, _bins
        hist, bins = np.histogram(np.log10(_morph).reshape(-1), _bins)
        cutoff = np.where(hist == 0)[0]
        # If all of the pixels are used there is no need to threshold
        if len(cutoff) == 0:
            return 0, _bins
        return 10 ** bins[cutoff[-1]], _bins


class MonotonicityConstraint(Constraint):
    """Make morphology monotonically decrease from the center

    See `~scarlet.operator.prox_monotonic`
    for a description of the other parameters.
    """

    def __init__(
        self,
        neighbor_weight="flat",
        min_gradient=0.1,
        use_mask=False,
        fit_center_radius=0,
        center=None,
    ):
        self.neighbor_weight = neighbor_weight
        self.min_gradient = min_gradient
        self.use_mask = use_mask
        self.fit_center = fit_center_radius > 0
        self.fit_center_radius = fit_center_radius
        self.center = center

    def __call__(self, morph, step):
        shape = morph.shape
        center = _constraint_center(shape, self.center)
        if self.fit_center:
            center = operator.get_center(morph, center, radius=self.fit_center_radius)

        # get prox from the cache
        prox_name = "operator.prox_weighted_monotonic"
        key = (shape, center, self.neighbor_weight, self.min_gradient)
        # The creation of this operator is expensive,
        # so load it from memory if possible.
        try:
            prox = Cache.check(prox_name, key)
        except KeyError:
            prox = operator.prox_weighted_monotonic(
                shape,
                neighbor_weight=self.neighbor_weight,
                min_gradient=self.min_gradient,
                center=center,
            )
            Cache.set(prox_name, key, prox)

        # apply the prox
        _morph = morph.copy()
        result = prox(morph, step)
        if self.use_mask:
            valid, _morph, _bounds = operator.prox_monotonic_mask(
                _morph, step, center=center, center_radius=0, variance=0, max_iter=0,
            )
            result[valid] = _morph[valid]

        return result


class MonotonicMaskConstraint(Constraint):
    """Make morphology monotonic by branching from the center
    """

    def __init__(self, center, center_radius=1, variance=0.0, max_iter=3):
        self.center = center
        self.center_radius = center_radius
        self.variance = variance
        self.max_iter = max_iter
        self.prox = partial(
            operator.prox_monotonic_mask,
            center=center,
            center_radius=center_radius,
            variance=variance,
            max_iter=max_iter,
        )

    def __call__(self, morph, step):
        if len(morph.shape) == 2:
            valid, morph, bounds = self.prox(morph, step)
        else:
            morph = np.array([self.prox(morph_, step)[1] for morph_ in morph])
        return morph


class SymmetryConstraint(Constraint):
    """Make the source symmetric about its center

    See `~scarlet.operator.prox_uncentered_symmetry`
    for a description of the parameters.
    """

    def __init__(self, strength=1, center=None):
        self.strength = strength
        self.center = center

    @property
    def is_euclidean_projection(self):
        # An explicit integer center selects an odd finite-support patch whose
        # reflected pairs are averaged exactly. The historical implicit-center
        # operator pads even axes and is therefore not always idempotent.
        return self.strength == 1 and self.center is not None

    def __call__(self, morph, step):
        if self.center is not None:
            center = _constraint_center(morph.shape, self.center)
            return operator.prox_uncentered_symmetry(
                morph,
                step,
                center=center,
                algorithm="soft",
                strength=self.strength,
            )
        return operator.prox_soft_symmetry(morph, step, strength=self.strength)


class CenterOnConstraint(Constraint):
    """Sets the center pixel to a tiny non-zero value
    """

    is_euclidean_projection = True

    def __init__(self, tiny=1e-6, center=None):
        self.tiny = tiny
        self.center = center

    def __call__(self, morph, step):
        shape = morph.shape
        center = _constraint_center(shape, self.center)
        morph[center] = max(morph[center], self.tiny)
        return morph


class CentroidConstraint(Constraint):
    """Project a 2-D morphology onto a fixed flux-weighted centroid.

    For non-zero morphology ``m``, fixing its centroid to ``c`` is equivalent
    to the two linear equations ``sum(m * (row-c_row)) = 0`` and
    ``sum(m * (column-c_column)) = 0``. This class is the exact Euclidean
    projector onto that linear subspace. Combine it with
    :class:`PositivityConstraint` through :class:`DykstraConstraintChain` to
    obtain the closest non-negative morphology with the declared centroid.

    Parameters
    ----------
    center: tuple of float
        Target ``(row, column)`` centroid in morphology pixel coordinates.
    """

    is_euclidean_projection = True

    def __init__(self, center):
        if len(center) != 2 or not np.all(np.isfinite(center)):
            raise ValueError("center must contain two finite coordinates")
        self.center = tuple(float(coordinate) for coordinate in center)

    def __call__(self, morph, step):
        value = np.asarray(morph)
        if value.ndim != 2:
            raise ValueError("centroid constraints require a 2-D morphology")
        if any(
            coordinate < 0 or coordinate > size - 1
            for coordinate, size in zip(self.center, value.shape)
        ):
            raise ValueError("centroid must lie inside the morphology")

        rows, columns = np.indices(value.shape, dtype=float)
        design = np.stack(
            (rows - self.center[0], columns - self.center[1]), axis=0
        ).reshape(2, -1)
        flat = value.reshape(-1)
        gram = design @ design.T
        correction = design.T @ np.linalg.pinv(gram) @ (design @ flat)
        return (flat - correction).reshape(value.shape)


class LeakyConstraint(Constraint):
    """Make a constraint leak the original value with a configurable amount:

    Updates `x = (1-leak) * prox(x, step) + leak * x`
    """

    def __init__(self, constraint, leak=0.05):
        self.constraint = constraint
        self.leak = leak

    def __call__(self, x, step):
        return (1 - self.leak) * self.constraint(x, step) + self.leak * x

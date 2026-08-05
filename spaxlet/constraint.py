from functools import partial

import numpy as np
import proxmin
from scipy.linalg import solve_banded
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve

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


def _shifted_center(center, offset, *, integer):
    if center is None:
        return None
    if len(offset) != len(center) or not np.all(np.isfinite(offset)):
        raise ValueError("a coordinate shift must match the center and be finite")
    if integer:
        resolved = tuple(int(delta) for delta in offset)
        if any(float(delta) != value for delta, value in zip(offset, resolved)):
            raise ValueError("an integer pixel center requires an integer shift")
        return tuple(
            int(coordinate) + delta
            for coordinate, delta in zip(center, resolved)
        )
    return tuple(
        float(coordinate) + float(delta)
        for coordinate, delta in zip(center, offset)
    )


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

    def shifted(self, offset):
        """Return the equivalent constraint after a coordinate-grid shift.

        Constraints without spatial coordinates are invariant and return
        themselves. Coordinate-bearing subclasses rebuild their cached
        geometry around the shifted local center.
        """

        return self


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

    def shifted(self, offset):
        constraints = tuple(
            constraint.shifted(offset)
            if hasattr(constraint, "shifted")
            else constraint
            for constraint in self.constraints
        )
        return ConstraintChain(*constraints, repeat=self.repeat)


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

    def shifted(self, offset):
        constraints = tuple(
            constraint.shifted(offset) for constraint in self.constraints
        )
        return DykstraConstraintChain(
            *constraints,
            max_iter=self.max_iter,
            rtol=self.rtol,
            atol=self.atol
        )


class ProximalDykstraConstraintChain(ConstraintChain):
    """Proximal map of a sum of convex penalties via Dykstra splitting.

    Unlike :class:`DykstraConstraintChain`, members need not be projections:
    each can be the proximal map of a convex penalty. This permits a quadratic
    smoothness penalty to be combined with exact positivity, centroid, or
    symmetry indicators without making their result depend on operator order.
    """

    def __init__(self, *constraints, max_iter=2000, rtol=1e-8, atol=1e-10):
        if not constraints:
            raise ValueError("a proximal Dykstra chain requires constraints")
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
        dtype = original.dtype if np.issubdtype(original.dtype, np.floating) else float
        numerical_floor = 10 * np.finfo(dtype).eps * scale
        threshold = max(self.atol + self.rtol * scale, numerical_floor)
        for _ in range(self.max_iter):
            previous = result.copy()
            for index, constraint in enumerate(self.constraints):
                shifted = result + corrections[index]
                proximal = np.asarray(constraint(shifted.copy(), step))
                if proximal.shape != original.shape:
                    raise ValueError("constraints in a chain must preserve shape")
                corrections[index] = shifted - proximal
                result = proximal
            if float(np.linalg.norm(result - previous)) <= threshold:
                return result
        raise RuntimeError(
            "proximal Dykstra did not converge in {} sweeps".format(self.max_iter)
        )

    def shifted(self, offset):
        return ProximalDykstraConstraintChain(
            *(constraint.shifted(offset) for constraint in self.constraints),
            max_iter=self.max_iter,
            rtol=self.rtol,
            atol=self.atol
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


class SpectralSmoothnessConstraint(Constraint):
    """Non-negative spectrum with quadratic wavelength-curvature penalty.

    The penalty is ``0.5 * strength * ||D2 spectrum||**2``, where ``D2`` is
    the second divided difference on the physical wavelength grid, normalized
    so that a uniformly sampled grid has the familiar ``[1, -2, 1]`` stencil.
    Large wavelength gaps start independent segments and are never smoothed
    across. Positivity and the quadratic penalty are combined with proximal
    Dykstra iterations, yielding the proximal map of their sum rather than an
    order-dependent smoothing-and-clipping heuristic.

    Parameters
    ----------
    wavelengths: one-dimensional array
        Strictly increasing physical wavelengths. Units cancel after the grid
        is normalized by its median spacing.
    strength: float
        Non-negative dimensionless curvature-penalty strength. Zero recovers
        positivity.
    reference_scale: float
        Positive fixed spectral-amplitude scale. The physical quadratic
        penalty weight is ``strength / reference_scale``. Declaring this scale
        makes the proximal response invariant when an entire spectrum is
        multiplied by a constant.
    zero: float
        Lower bound for every spectral coefficient.
    gap_factor: float
        Adjacent wavelength spacings larger than this multiple of the median
        spacing split the regularizer into independent segments.
    max_iter: int
        Maximum proximal Dykstra sweeps.
    rtol, atol: float
        Relative and absolute iterate tolerances.
    """

    def __init__(
        self,
        wavelengths,
        strength,
        reference_scale=1,
        zero=0,
        gap_factor=5,
        max_iter=100,
        rtol=1e-8,
        atol=1e-10,
    ):
        wavelengths = np.asarray(wavelengths, dtype=float)
        if wavelengths.ndim != 1 or wavelengths.size == 0:
            raise ValueError("wavelengths must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(wavelengths)):
            raise ValueError("wavelengths must be finite")
        spacing = np.diff(wavelengths)
        if np.any(spacing <= 0):
            raise ValueError("wavelengths must be strictly increasing")
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("smoothness strength must be finite and non-negative")
        if not np.isfinite(reference_scale) or reference_scale <= 0:
            raise ValueError("reference_scale must be finite and positive")
        if not np.isfinite(zero):
            raise ValueError("spectral lower bound must be finite")
        if not np.isfinite(gap_factor) or gap_factor <= 1:
            raise ValueError("gap_factor must be finite and greater than one")
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(rtol) or rtol < 0:
            raise ValueError("rtol must be finite and non-negative")
        if not np.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and non-negative")

        self.wavelengths = wavelengths
        self.strength = float(strength)
        self.reference_scale = float(reference_scale)
        self.zero = float(zero)
        self.gap_factor = float(gap_factor)
        self.max_iter = int(max_iter)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self._curvature_bands = self._build_curvature_bands()

    def _build_curvature_bands(self):
        size = self.wavelengths.size
        if size < 3:
            return np.zeros((5, size), dtype=float)
        spacing = np.diff(self.wavelengths)
        typical = float(np.median(spacing))
        coordinate = (self.wavelengths - self.wavelengths[0]) / typical
        scaled_spacing = np.diff(coordinate)
        gap = scaled_spacing > self.gap_factor

        rows = []
        columns = []
        values = []
        row = 0
        for center in range(1, size - 1):
            if gap[center - 1] or gap[center]:
                continue
            left = scaled_spacing[center - 1]
            right = scaled_spacing[center]
            normalization = 2 / (left + right)
            coefficients = (
                normalization / left,
                -normalization * (1 / left + 1 / right),
                normalization / right,
            )
            for column, value in zip(
                (center - 1, center, center + 1), coefficients
            ):
                rows.append(row)
                columns.append(column)
                values.append(value)
            row += 1
        if row == 0:
            return np.zeros((5, size), dtype=float)
        curvature = coo_matrix(
            (values, (rows, columns)), shape=(row, size), dtype=float
        ).tocsr()
        gram = (curvature.T @ curvature).tocsr()
        bands = np.zeros((5, size), dtype=float)
        bands[2] = gram.diagonal(0)
        bands[1, 1:] = gram.diagonal(1)
        bands[0, 2:] = gram.diagonal(2)
        bands[3, :-1] = gram.diagonal(-1)
        bands[4, :-2] = gram.diagonal(-2)
        return bands

    @staticmethod
    def _scalar_step(step):
        value = np.asarray(step, dtype=float)
        if value.ndim == 0:
            result = float(value)
        elif value.size and np.all(value == value.flat[0]):
            result = float(value.flat[0])
        else:
            raise ValueError(
                "spectral smoothness requires a scalar proximal step"
            )
        if not np.isfinite(result) or result < 0:
            raise ValueError("proximal step must be finite and non-negative")
        return result

    def _smooth(self, value, step):
        weight = self.strength * step / self.reference_scale
        if weight == 0 or value.size < 3:
            return value.copy()
        system = weight * self._curvature_bands
        system[2] += 1
        return solve_banded(
            (2, 2), system, value, overwrite_ab=True, check_finite=False
        )

    def __call__(self, X, step):
        original = np.asarray(X)
        if original.ndim != 1 or original.shape != self.wavelengths.shape:
            raise ValueError(
                "spectral smoothness expects one value per declared wavelength"
            )
        resolved_step = self._scalar_step(step)
        if resolved_step == 0 or self.strength == 0:
            return np.maximum(original, self.zero)

        result = original.copy()
        smooth_correction = np.zeros_like(result)
        positive_correction = np.zeros_like(result)
        scale = max(float(np.linalg.norm(original)), 1.0)
        threshold = self.atol + self.rtol * scale
        for _ in range(self.max_iter):
            previous = result.copy()
            shifted = result + smooth_correction
            smoothed = self._smooth(shifted, resolved_step)
            smooth_correction = shifted - smoothed

            shifted = smoothed + positive_correction
            result = np.maximum(shifted, self.zero)
            positive_correction = shifted - result
            if float(np.linalg.norm(result - previous)) <= threshold:
                return result
        raise RuntimeError(
            "spectral smoothness proximal map did not converge in {} sweeps".format(
                self.max_iter
            )
        )


class SpatialSmoothnessConstraint(Constraint):
    """Quadratic nearest-neighbor coherence penalty for a 2-D morphology.

    Its penalty is ``0.5 * strength * sum_edges (m_i - m_j)**2``. The exact
    proximal map solves one sparse screened-Poisson system. Positivity and
    identity constraints are intentionally separate and can be combined with
    this penalty through :class:`ProximalDykstraConstraintChain`.

    ``reference_scale`` is a fixed positive morphology amplitude. As with the
    spectral smoothness constraint, dividing the physical penalty weight by
    this scale makes the response invariant when both the morphology and its
    optimizer step are rescaled together.
    """

    def __init__(self, strength, reference_scale=1):
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("smoothness strength must be finite and non-negative")
        if not np.isfinite(reference_scale) or reference_scale <= 0:
            raise ValueError("reference_scale must be finite and positive")
        self.strength = float(strength)
        self.reference_scale = float(reference_scale)
        self._laplacian_cache = {}

    @staticmethod
    def _scalar_step(step):
        value = np.asarray(step, dtype=float)
        if value.ndim == 0:
            result = float(value)
        elif value.size and np.all(value == value.flat[0]):
            result = float(value.flat[0])
        else:
            raise ValueError("spatial smoothness requires a scalar proximal step")
        if not np.isfinite(result) or result < 0:
            raise ValueError("proximal step must be finite and non-negative")
        return result

    def _laplacian(self, shape):
        if shape not in self._laplacian_cache:
            height, width = shape
            index = np.arange(height * width).reshape(shape)
            first = []
            second = []
            if width > 1:
                first.append(index[:, :-1].ravel())
                second.append(index[:, 1:].ravel())
            if height > 1:
                first.append(index[:-1].ravel())
                second.append(index[1:].ravel())
            if not first:
                result = coo_matrix((height * width, height * width), dtype=float)
            else:
                left = np.concatenate(first)
                right = np.concatenate(second)
                rows = np.repeat(np.arange(left.size), 2)
                columns = np.column_stack((left, right)).ravel()
                values = np.tile((1.0, -1.0), left.size)
                difference = coo_matrix(
                    (values, (rows, columns)),
                    shape=(left.size, height * width),
                ).tocsr()
                result = (difference.T @ difference).tocsr()
            self._laplacian_cache[shape] = result
        return self._laplacian_cache[shape]

    def __call__(self, X, step):
        original = np.asarray(X)
        if original.ndim != 2:
            raise ValueError("spatial smoothness expects a 2-D morphology")
        resolved_step = self._scalar_step(step)
        weight = self.strength * resolved_step / self.reference_scale
        if weight == 0 or original.size == 1:
            return original.copy()
        laplacian = self._laplacian(original.shape)
        system = eye(original.size, format="csr") + weight * laplacian
        result = spsolve(system, original.reshape(-1))
        return np.asarray(result).reshape(original.shape)


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

    See `~spaxlet.operator.prox_monotonic`
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

    def shifted(self, offset):
        center = _shifted_center(self.center, offset, integer=True)
        return MonotonicityConstraint(
            neighbor_weight=self.neighbor_weight,
            min_gradient=self.min_gradient,
            use_mask=self.use_mask,
            fit_center_radius=self.fit_center_radius,
            center=center,
        )


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

    def shifted(self, offset):
        center = _shifted_center(self.center, offset, integer=True)
        return MonotonicMaskConstraint(
            center,
            center_radius=self.center_radius,
            variance=self.variance,
            max_iter=self.max_iter,
        )


class SymmetryConstraint(Constraint):
    """Make the source symmetric about its center

    See `~spaxlet.operator.prox_uncentered_symmetry`
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

    def shifted(self, offset):
        center = _shifted_center(self.center, offset, integer=True)
        return SymmetryConstraint(strength=self.strength, center=center)


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

    def shifted(self, offset):
        center = _shifted_center(self.center, offset, integer=True)
        return CenterOnConstraint(tiny=self.tiny, center=center)


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
        self._projection_cache = {}

    def __call__(self, morph, step):
        value = np.asarray(morph)
        if value.ndim != 2:
            raise ValueError("centroid constraints require a 2-D morphology")
        if any(
            coordinate < 0 or coordinate > size - 1
            for coordinate, size in zip(self.center, value.shape)
        ):
            raise ValueError("centroid must lie inside the morphology")

        if value.shape not in self._projection_cache:
            rows, columns = np.indices(value.shape, dtype=float)
            design = np.stack(
                (rows - self.center[0], columns - self.center[1]), axis=0
            ).reshape(2, -1)
            gram_inverse = np.linalg.pinv(design @ design.T)
            self._projection_cache[value.shape] = (design, gram_inverse)
        design, gram_inverse = self._projection_cache[value.shape]
        flat = value.reshape(-1)
        correction = design.T @ gram_inverse @ (design @ flat)
        return (flat - correction).reshape(value.shape)

    def shifted(self, offset):
        return CentroidConstraint(
            _shifted_center(self.center, offset, integer=False)
        )


class LeakyConstraint(Constraint):
    """Make a constraint leak the original value with a configurable amount:

    Updates `x = (1-leak) * prox(x, step) + leak * x`
    """

    def __init__(self, constraint, leak=0.05):
        self.constraint = constraint
        self.leak = leak

    def __call__(self, x, step):
        return (1 - self.leak) * self.constraint(x, step) + self.leak * x

"""Isolate the mechanism behind the converged many-source flux bias.

At the converged budget the broad central ``lens`` factor holds about 6.3%
too much flux and the other nine sources too little.  The noiseless control
converges to truth, so the model and the oracle catalog are not at fault and
the bias is created by the noise.  Two candidates remain, and they have very
different consequences:

``correlated``
    The contract injects spatially and spectrally correlated residuals but the
    fit uses only the diagonal variance.  If this misspecification is the
    cause, the same bias is present in every real reduced cube, because
    reduced IFU data always has correlated noise fitted with a diagonal
    weight.

``positivity``
    The faint neighbours' wings sit against the nonnegativity boundary, and a
    clipped estimator is biased low, which would push the balance into the one
    unclipped broad component.

The discriminating arm replaces the correlated residual field with white noise
of the identical per-voxel variance, seed, and mask.  Everything else, the
truth, the PSF, the supports and the constraints, is untouched.
"""

from __future__ import annotations

import argparse

import numpy as np

import benchmarks.many_source_recovery_contract as contract
import benchmarks.run_synthetic_many_source_recovery as recovery
from benchmarks.run_synthetic_many_source_recovery import (
    RECOVERY_CONVERGED_MAX_ITER,
)
from benchmarks.signed_recovery_metrics import (
    SOURCE_NAMES,
    cancellation_ratio,
    signed_source_metrics,
)


def _white_standard_noise(shape, generator):
    """Return unit-RMS white noise, matching the correlated helper's contract."""

    white = generator.normal(size=shape)
    white -= np.mean(white)
    return white / np.std(white)


class _NoiseArm:
    """Swap the fixture's residual generator for the duration of one fit."""

    def __init__(self, standard_noise):
        self._standard_noise = standard_noise
        self._saved = None

    def __enter__(self):
        self._saved = contract._correlated_standard_noise
        contract._correlated_standard_noise = self._standard_noise
        return self

    def __exit__(self, *exception):
        contract._correlated_standard_noise = self._saved
        return False


class _ScaledNoiseArm:
    """Scale the injected residual and its declared variance together.

    Scaling both keeps the likelihood correctly specified at every noise
    level, so the arm measures how the bias depends on the noise rather than
    how it responds to a mis-stated weight.  A bilinear estimator's
    second-order bias goes as the variance, so halving the noise should divide
    the bias by about four; a first-order effect would only halve it.
    """

    def __init__(self, scale):
        if scale <= 0:
            raise ValueError("noise scale must be positive")
        self._scale = float(scale)
        self._saved = None

    def __enter__(self):
        self._saved = recovery.recovery_noisy_cube
        original, scale = self._saved, self._scale

        def scaled(seed=None):
            data, variance, valid, truth, residual = original(seed=seed)
            scaled_residual = residual * scale
            scaled_data = truth + scaled_residual
            scaled_data[~valid] = 0.0
            return scaled_data, variance * scale**2, valid, truth, scaled_residual

        recovery.recovery_noisy_cube = scaled
        return self

    def __exit__(self, *exception):
        recovery.recovery_noisy_cube = self._saved
        return False


def _positivity_activity(result):
    """Fraction of fitted morphology pixels pinned at zero inside the support."""

    pinned = 0
    total = 0
    for spec, image in zip(contract.RECOVERY_SOURCE_SPECS, result.fitted_morphologies):
        half = spec.support // 2
        window = image[
            spec.center[0] - half : spec.center[0] + half + 1,
            spec.center[1] - half : spec.center[1] + half + 1,
        ]
        pinned += int(np.count_nonzero(window <= 0))
        total += window.size
    return pinned / total


def run_arm(name, standard_noise, max_iter, seed, positivity=True):
    with _NoiseArm(standard_noise):
        # The fixture caches nothing, but the runner imports the noisy-cube
        # helper by name, so both module bindings must see the swap.
        result = recovery.fit_recovery_cube(
            start="A", max_iter=max_iter, seed=seed, positivity=positivity
        )
    signed = signed_source_metrics(result.fitted_spectra)
    lens = SOURCE_NAMES.index("lens")
    return {
        "arm": name,
        "positivity": positivity,
        "chi2_per_valid_voxel": result.chi2_per_valid_voxel,
        "relative_projected_gradient": result.relative_projected_gradient,
        "lens_signed": float(signed["signed_total_relative"][lens]),
        "max_abs_signed": float(np.max(np.abs(signed["signed_total_relative"]))),
        "negative_sources": int(np.count_nonzero(signed["signed_total_relative"] < 0)),
        "cancellation": cancellation_ratio(signed["signed_total_absolute"]),
        "positivity_active_fraction": _positivity_activity(result),
        "signed": signed["signed_total_relative"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=RECOVERY_CONVERGED_MAX_ITER)
    parser.add_argument("--seed", type=int, default=contract.RECOVERY_SEED)
    parser.add_argument("--noise-scales", default="")
    args = parser.parse_args()

    if args.noise_scales:
        scales = tuple(float(v) for v in args.noise_scales.split(","))
        print(
            "{:<8} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12}".format(
                "scale", "chi2/N", "rpg", "lens", "max|d|", "lens/scale", "lens/scale^2"
            )
        )
        for scale in scales:
            with _ScaledNoiseArm(scale):
                entry = run_arm(
                    "white", _white_standard_noise, args.max_iter, args.seed
                )
            print(
                "{:<8.3f} {:>10.6f} {:>10.2e} {:>+10.5f} {:>10.5f} "
                "{:>12.5f} {:>12.5f}".format(
                    scale,
                    entry["chi2_per_valid_voxel"],
                    entry["relative_projected_gradient"],
                    entry["lens_signed"],
                    entry["max_abs_signed"],
                    entry["lens_signed"] / scale,
                    entry["lens_signed"] / scale**2,
                )
            )
        return

    # A 2x2 over the two candidate causes.  Correlated-versus-white isolates
    # the diagonal-variance misspecification; positivity on-versus-off isolates
    # the clipped-estimator effect.  If both together explain the bias, the
    # white/no-positivity cell should be near zero.
    arms = (
        ("corr+pos", contract._correlated_standard_noise, True),
        ("corr-pos", contract._correlated_standard_noise, False),
        ("white+pos", _white_standard_noise, True),
        ("white-pos", _white_standard_noise, False),
    )
    results = [
        run_arm(name, noise, args.max_iter, args.seed, positivity)
        for name, noise, positivity in arms
    ]

    print(
        "{:<12} {:>10} {:>10} {:>10} {:>10} {:>6} {:>10} {:>10}".format(
            "arm", "chi2/N", "rpg", "lens", "max|d|", "neg", "cancel", "posactive"
        )
    )
    for entry in results:
        print(
            "{:<12} {:>10.6f} {:>10.2e} {:>+10.5f} {:>10.5f} {:>6d} "
            "{:>10.4f} {:>10.4f}".format(
                entry["arm"],
                entry["chi2_per_valid_voxel"],
                entry["relative_projected_gradient"],
                entry["lens_signed"],
                entry["max_abs_signed"],
                entry["negative_sources"],
                entry["cancellation"],
                entry["positivity_active_fraction"],
            )
        )

    print()
    print("{:<12} {}".format("source", " ".join("{:>12}".format(e["arm"]) for e in results)))
    for index, name in enumerate(SOURCE_NAMES):
        print(
            "{:<12} {}".format(
                name,
                " ".join("{:>+12.5f}".format(e["signed"][index]) for e in results),
            )
        )


if __name__ == "__main__":
    main()

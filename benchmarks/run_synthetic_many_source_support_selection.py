"""Choose detected support extents without truth, using held-out voxels.

Support extent is the binding error in the detected-catalog arm: detection
recovers every source position exactly, but thresholding a detection image
undershoots the extent by about 2.4 px and the maximum per-source flux error
is 73% against the oracle catalog's 7%.  Dilating overshoots and doubles the
error again, so the sensitivity is steep in both directions and the extent has
to be chosen by fitting rather than by thresholding.

Plain chi-square cannot make that choice safely, because it rewards extra
freedom: a spurious eleventh source lowers it.  Held-out voxels can, since a
component that only absorbs noise cannot improve a residual it never saw.

The scan multiplies every detected support by a common factor.  Truth is read
only after the selection is made, to check whether the truth-free rule picked
the arm truth would have picked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.many_source_recovery_contract import (
    RECOVERY_SEED,
    RECOVERY_SHAPE,
    RECOVERY_SOURCE_SPECS,
    recovery_morphologies,
    recovery_noisy_cube,
    recovery_spectra,
)
from benchmarks.run_synthetic_many_source_detected_catalog import (
    detected_catalog,
    match_to_truth,
)
from benchmarks.run_synthetic_many_source_recovery import (
    RECOVERY_CONVERGED_MAX_ITER,
    CatalogEntry,
    fit_catalog,
    oracle_catalog,
)

DEFAULT_SCALES = (0.6, 0.8, 1.0, 1.3, 1.6, 2.0)
HOLDOUT_STRIDE = 11


def _rescale(entry, scale, limit):
    """Return the entry with an odd support scaled by ``scale``."""

    half = int(round(0.5 * scale * (entry.support - 1)))
    half = max(1, min(half, entry.center[0], entry.center[1],
                      limit - 1 - entry.center[0], limit - 1 - entry.center[1]))
    support = 2 * half + 1
    if support == entry.support:
        return entry
    # The centroid target is expressed in local box coordinates, so it has to
    # move with the box origin or the constraint would silently shift.
    shift = (support - entry.support) // 2
    centroid = (entry.centroid[0] + shift, entry.centroid[1] + shift)
    return CatalogEntry(
        name=entry.name,
        center=entry.center,
        support=support,
        start_width=entry.start_width,
        centroid=centroid,
    )


def scaled_catalog(catalog, scale, limit=RECOVERY_SHAPE[1]):
    return tuple(_rescale(entry, scale, limit) for entry in catalog)


def truth_error(fit, catalog, matches):
    """Score against truth.  Used only to audit the selection, never to make it."""

    truth_spectra = recovery_spectra()
    truth_morphologies = recovery_morphologies()
    signed = []
    morphology = []
    for detection, truth in matches.items():
        fitted = fit["fitted_spectra"][detection]
        reference = truth_spectra[truth]
        signed.append((np.sum(fitted) - np.sum(reference)) / np.sum(reference))
        morphology.append(
            np.linalg.norm(
                fit["fitted_morphologies"][detection] - truth_morphologies[truth]
            )
            / np.linalg.norm(truth_morphologies[truth])
        )
    return {
        "max_abs_signed_flux": float(np.max(np.abs(signed))),
        "max_morphology_relative_l2": float(np.max(morphology)),
        "mean_abs_signed_flux": float(np.mean(np.abs(signed))),
    }


def run(catalog, scales, *, start, max_iter, seed):
    matches, _, _ = match_to_truth(catalog)
    results = []
    for scale in scales:
        arm = scaled_catalog(catalog, scale)
        fit = fit_catalog(
            catalog=arm,
            start=start,
            max_iter=max_iter,
            seed=seed,
            holdout_stride=HOLDOUT_STRIDE,
        )
        supports = [entry.support for entry in arm]
        entry = {
            "scale": scale,
            "supports": supports,
            "mean_support": float(np.mean(supports)),
            "fitted_chi2": fit["chi2_per_valid_voxel"],
            "held_out_chi2": fit["held_out_chi2"],
            "relative_projected_gradient": fit["relative_projected_gradient"],
        }
        entry.update(truth_error(fit, arm, matches))
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", choices=("A", "B", "C"), default="A")
    parser.add_argument("--max-iter", type=int, default=RECOVERY_CONVERGED_MAX_ITER)
    parser.add_argument("--seed", type=int, default=RECOVERY_SEED)
    parser.add_argument(
        "--scales", default=",".join(str(value) for value in DEFAULT_SCALES)
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    scales = tuple(float(value) for value in args.scales.split(","))
    data, variance, valid, _, _ = recovery_noisy_cube(seed=args.seed)
    weights = np.zeros_like(variance)
    weights[valid] = 1.0 / variance[valid]
    catalog, _ = detected_catalog(data, weights)

    results = run(
        catalog, scales, start=args.start, max_iter=args.max_iter, seed=args.seed
    )

    print(
        "{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
            "scale", "meanSupport", "fittedChi2", "heldOutChi2", "maxFluxErr", "maxMorphErr"
        )
    )
    for entry in results:
        print(
            "{:>6.2f} {:>12.2f} {:>12.6f} {:>12.6f} {:>12.5f} {:>12.5f}".format(
                entry["scale"],
                entry["mean_support"],
                entry["fitted_chi2"],
                entry["held_out_chi2"],
                entry["max_abs_signed_flux"],
                entry["max_morphology_relative_l2"],
            )
        )

    selected = min(results, key=lambda entry: entry["held_out_chi2"])
    by_truth = min(results, key=lambda entry: entry["max_abs_signed_flux"])
    by_fitted = min(results, key=lambda entry: entry["fitted_chi2"])
    oracle_supports = float(np.mean([entry.support for entry in oracle_catalog()]))
    truth_supports = float(
        np.mean([spec.support for spec in RECOVERY_SOURCE_SPECS])
    )

    print()
    print("selected by held-out chi2 : scale {:.2f}, mean support {:.2f}, "
          "max flux error {:.5f}".format(
              selected["scale"], selected["mean_support"],
              selected["max_abs_signed_flux"]))
    print("best by truth             : scale {:.2f}, max flux error {:.5f}".format(
        by_truth["scale"], by_truth["max_abs_signed_flux"]))
    print("best by fitted chi2       : scale {:.2f}, max flux error {:.5f}".format(
        by_fitted["scale"], by_fitted["max_abs_signed_flux"]))
    print("oracle mean support       : {:.2f} (truth {:.2f})".format(
        oracle_supports, truth_supports))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "support_selection.json").write_text(
            json.dumps(
                {
                    "start": args.start,
                    "max_iter": args.max_iter,
                    "seed": args.seed,
                    "holdout_stride": HOLDOUT_STRIDE,
                    "selected_scale": selected["scale"],
                    "truth_optimal_scale": by_truth["scale"],
                    "fitted_chi2_optimal_scale": by_fitted["scale"],
                    "oracle_mean_support": oracle_supports,
                    "arms": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

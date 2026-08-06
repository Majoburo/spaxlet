"""Fit the many-source recovery cube from a catalog estimated from the data.

The declared recovery contract is conditional on an oracle catalog: the source
count, every center, every support box and every centroid-constraint target
come from truth.  This benchmark removes that conditioning.  It detects
sources in the cube with ``lisasep.segmentation``, derives the whole catalog
from the detection image, fits with the identical forward model and
constraints, and scores only after matching detections to truth.

It also runs deliberate miscount arms, because a real catalog is not merely
noisier than the oracle -- it can have the wrong number of rows.  The
interesting question is whether the residual notices, since chi-square is the
only quantity available to choose a catalog on a real cube.

``lisasep`` is imported from the sibling checkout; run with

    PYTHONPATH=../lisasep/src python -m benchmarks.run_synthetic_many_source_detected_catalog
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
from benchmarks.run_synthetic_many_source_recovery import (
    RECOVERY_CONVERGED_MAX_ITER,
    CatalogEntry,
    fit_catalog,
    oracle_catalog,
)

try:
    from lisasep.segmentation import segment_sources
except ImportError as error:  # pragma: no cover - environment dependent
    raise SystemExit(
        "lisasep.segmentation is required; run with "
        "PYTHONPATH=<workspace>/lisasep/src"
    ) from error


MAX_SUPPORT = 15


def _odd_span(mask_axis, center, limit):
    """Return an odd support that covers the segment and fits in the frame."""

    indices = np.flatnonzero(mask_axis)
    if indices.size == 0:
        return 3
    reach = max(abs(int(indices[0]) - center), abs(int(indices[-1]) - center))
    reach = min(reach, center, limit - 1 - center, MAX_SUPPORT // 2)
    return max(3, 2 * int(reach) + 1)


def detected_catalog(data, weights, **segmentation_arguments):
    """Build a catalog from detection alone, using no truth quantity."""

    segmentation = segment_sources(data, weights, **segmentation_arguments)
    image = segmentation.detection_image
    height, width = image.shape
    rows, columns = np.indices(image.shape, dtype=float)

    entries = []
    for index, peak in enumerate(segmentation.peaks):
        center = (int(peak[0]), int(peak[1]))
        support_mask = segmentation.supports[index]
        support = min(
            _odd_span(support_mask.any(axis=1), center[0], height),
            _odd_span(support_mask.any(axis=0), center[1], width),
        )
        half = support // 2
        window = (
            slice(center[0] - half, center[0] + half + 1),
            slice(center[1] - half, center[1] + half + 1),
        )
        # The centroid target and the start width both come from the detection
        # image, which is the only spatial information a real pipeline has.
        patch = np.maximum(image[window], 0.0)
        if patch.sum() <= 0:
            continue
        local_rows = rows[window] - (center[0] - half)
        local_columns = columns[window] - (center[1] - half)
        centroid = (
            float(np.sum(local_rows * patch) / patch.sum()),
            float(np.sum(local_columns * patch) / patch.sum()),
        )
        variance_y = np.sum(((local_rows - centroid[0]) ** 2) * patch) / patch.sum()
        variance_x = np.sum(((local_columns - centroid[1]) ** 2) * patch) / patch.sum()
        entries.append(
            CatalogEntry(
                name="det{:02d}".format(index),
                center=center,
                support=support,
                start_width=float(max(np.sqrt(np.sqrt(variance_y * variance_x)), 0.6)),
                centroid=centroid,
            )
        )
    return tuple(entries), segmentation


def match_to_truth(catalog, tolerance=2.5):
    """Greedily match catalog rows to truth sources by proximity.

    Matching exists only for scoring.  Nothing in the fit uses it.
    """

    truth_centers = np.asarray(
        [spec.center for spec in RECOVERY_SOURCE_SPECS], dtype=float
    )
    centers = np.asarray([entry.center for entry in catalog], dtype=float)
    if centers.size == 0:
        return {}, list(range(len(RECOVERY_SOURCE_SPECS))), []

    distance = np.linalg.norm(
        centers[:, None, :] - truth_centers[None, :, :], axis=2
    )
    matches = {}
    used_truth = set()
    for detection, truth in sorted(
        ((d, t) for d in range(len(catalog)) for t in range(len(truth_centers))),
        key=lambda pair: distance[pair],
    ):
        if distance[detection, truth] > tolerance:
            break
        if detection in matches or truth in used_truth:
            continue
        matches[detection] = truth
        used_truth.add(truth)
    missed = [t for t in range(len(truth_centers)) if t not in used_truth]
    spurious = [d for d in range(len(catalog)) if d not in matches]
    return matches, missed, spurious


def score(fit, catalog, matches):
    """Score matched sources against truth, in fitted-catalog order."""

    truth_spectra = recovery_spectra()
    truth_morphologies = recovery_morphologies()
    fitted_spectra = fit["fitted_spectra"]
    fitted_morphologies = fit["fitted_morphologies"]

    rows = []
    for detection, truth in sorted(matches.items(), key=lambda item: item[1]):
        fitted = fitted_spectra[detection]
        reference = truth_spectra[truth]
        signed = float(
            (np.sum(fitted) - np.sum(reference)) / np.sum(reference)
        )
        cosine = float(
            np.dot(fitted, reference)
            / (np.linalg.norm(fitted) * np.linalg.norm(reference))
        )
        morphology_error = float(
            np.linalg.norm(fitted_morphologies[detection] - truth_morphologies[truth])
            / np.linalg.norm(truth_morphologies[truth])
        )
        centre_offset = float(
            np.linalg.norm(
                np.asarray(catalog[detection].center, dtype=float)
                - np.asarray(RECOVERY_SOURCE_SPECS[truth].center, dtype=float)
            )
        )
        rows.append(
            {
                "truth": RECOVERY_SOURCE_SPECS[truth].name,
                "detection": catalog[detection].name,
                "center_offset_px": centre_offset,
                "support": catalog[detection].support,
                "truth_support": RECOVERY_SOURCE_SPECS[truth].support,
                "signed_flux_relative": signed,
                "spectrum_cosine": cosine,
                "morphology_relative_l2": morphology_error,
            }
        )
    return rows


def run_arm(name, catalog, *, start, max_iter, seed):
    fit = fit_catalog(catalog=catalog, start=start, max_iter=max_iter, seed=seed)
    matches, missed, spurious = match_to_truth(catalog)
    rows = score(fit, catalog, matches)
    signed = np.asarray([row["signed_flux_relative"] for row in rows])
    return {
        "arm": name,
        "sources": len(catalog),
        "matched": len(matches),
        "missed": [RECOVERY_SOURCE_SPECS[t].name for t in missed],
        "spurious": [catalog[d].name for d in spurious],
        "chi2_per_valid_voxel": fit["chi2_per_valid_voxel"],
        "relative_projected_gradient": fit["relative_projected_gradient"],
        "max_abs_signed_flux": float(np.max(np.abs(signed))) if signed.size else None,
        "max_morphology_relative_l2": (
            float(np.max([row["morphology_relative_l2"] for row in rows]))
            if rows
            else None
        ),
        "min_spectrum_cosine": (
            float(np.min([row["spectrum_cosine"] for row in rows])) if rows else None
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", choices=("A", "B", "C"), default="A")
    parser.add_argument("--max-iter", type=int, default=RECOVERY_CONVERGED_MAX_ITER)
    parser.add_argument("--seed", type=int, default=RECOVERY_SEED)
    parser.add_argument("--threshold-sigma", type=float, default=5.0)
    parser.add_argument("--min-separation", type=float, default=2.0)
    parser.add_argument("--dilation", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    data, variance, valid, _, _ = recovery_noisy_cube(seed=args.seed)
    weights = np.zeros_like(variance)
    weights[valid] = 1.0 / variance[valid]

    catalog, segmentation = detected_catalog(
        data,
        weights,
        threshold_sigma=args.threshold_sigma,
        min_separation=args.min_separation,
        dilation=args.dilation,
    )
    print(
        "detection: {} peaks at threshold {:.4g}".format(
            len(catalog), segmentation.threshold
        )
    )

    arms = [("oracle", oracle_catalog()), ("detected", catalog)]
    if len(catalog) > 1:
        # Drop the faintest detection to make an under-counted catalog, and add
        # a spurious row in a quiet corner to make an over-counted one.
        brightness = [
            segmentation.detection_image[entry.center] for entry in catalog
        ]
        order = np.argsort(brightness)
        arms.append(
            (
                "undercount",
                tuple(entry for i, entry in enumerate(catalog) if i != order[0]),
            )
        )
        arms.append(
            (
                "overcount",
                catalog
                + (
                    CatalogEntry(
                        name="spurious",
                        center=(4, 28),
                        support=7,
                        start_width=1.2,
                        centroid=(3.0, 3.0),
                    ),
                ),
            )
        )

    results = []
    for name, arm in arms:
        result = run_arm(
            name, arm, start=args.start, max_iter=args.max_iter, seed=args.seed
        )
        results.append(result)
        print(
            "{:<11} n={:<3d} matched={:<3d} chi2/N={:.6f} rpg={:.2e} "
            "max|dS|={} maxMorph={} missed={} spurious={}".format(
                result["arm"],
                result["sources"],
                result["matched"],
                result["chi2_per_valid_voxel"],
                result["relative_projected_gradient"],
                "n/a"
                if result["max_abs_signed_flux"] is None
                else "{:.5f}".format(result["max_abs_signed_flux"]),
                "n/a"
                if result["max_morphology_relative_l2"] is None
                else "{:.5f}".format(result["max_morphology_relative_l2"]),
                ",".join(result["missed"]) or "-",
                ",".join(result["spurious"]) or "-",
            )
        )

    detected = next(r for r in results if r["arm"] == "detected")
    print()
    print(
        "{:<11} {:>10} {:>8} {:>8} {:>14} {:>9}".format(
            "truth", "offset px", "support", "truth", "signed flux", "cos"
        )
    )
    for row in detected["rows"]:
        print(
            "{:<11} {:>10.3f} {:>8d} {:>8d} {:>+14.5f} {:>9.5f}".format(
                row["truth"],
                row["center_offset_px"],
                row["support"],
                row["truth_support"],
                row["signed_flux_relative"],
                row["spectrum_cosine"],
            )
        )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "detected_catalog.json").write_text(
            json.dumps(
                {
                    "start": args.start,
                    "max_iter": args.max_iter,
                    "seed": args.seed,
                    "threshold_sigma": args.threshold_sigma,
                    "min_separation": args.min_separation,
                    "dilation": args.dilation,
                    "detection_threshold": float(segmentation.threshold),
                    "arms": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

"""Test whether a sharper imaging band fixes the detected-catalog failure.

The detected-catalog arm loses a factor of ten in per-source flux accuracy,
and the cause is support extent rather than position.  Support extent is set
by angular resolution, so this benchmark adds a sharper broadband imaging
observation of the same latent scene and separates two distinct effects:

``detect_imaging``
    the catalog is estimated from the sharper data, but only the IFU is fitted.
    This isolates the value of a better catalog.

``joint``
    the same catalog, with both observations fitted against one shared
    morphology per source.  The difference from ``detect_imaging`` is the value
    of the joint constraint itself.

Both observations declare their own channels against one model frame, so every
source keeps a single morphology and one spectrum spanning IFU channels and
imaging bands together.  Scoring uses the IFU channels alone, which keeps every
number comparable with the IFU-only arms.

Run with the sibling checkout on the path for detection:

    PYTHONPATH=../lisasep/src python -m \\
      benchmarks.run_synthetic_many_source_joint_imaging --max-iter 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import spaxlet

from benchmarks.many_source_imaging_contract import (
    IMAGING_CHANNELS,
    imaging_noisy_bands,
    imaging_psfs,
)
from benchmarks.many_source_recovery_contract import (
    RECOVERY_SEED,
    RECOVERY_SHAPE,
    RECOVERY_SOURCE_SPECS,
    recovery_morphologies,
    recovery_noisy_cube,
    recovery_psfs,
    recovery_spectra,
)
from benchmarks.run_synthetic_many_source_detected_catalog import (
    detected_catalog,
    match_to_truth,
)
from benchmarks.run_synthetic_many_source_recovery import (
    RECOVERY_CONVERGED_MAX_ITER,
    START_WIDTH_SCALE,
    _initial_image,
    oracle_catalog,
)

IFU_CHANNELS = tuple("ch{:03d}".format(i) for i in range(RECOVERY_SHAPE[0]))
JOINT_CHANNELS = IFU_CHANNELS + IMAGING_CHANNELS


def _weights(variance, valid):
    weights = np.zeros_like(variance)
    weights[valid] = 1.0 / variance[valid]
    return weights


def _sources(frame, catalog, start, cube, bands, use_imaging):
    """Build one factorized source per catalog row on the joint frame."""

    rows, columns = np.indices(RECOVERY_SHAPE[1:], dtype=float)
    sources = []
    for entry in catalog:
        half = entry.support // 2
        origin = (entry.center[0] - half, entry.center[1] - half)
        box = spaxlet.Box((entry.support, entry.support), origin=origin)
        constraint = spaxlet.DykstraConstraintChain(
            spaxlet.CentroidConstraint(np.asarray(entry.centroid, dtype=float)),
            spaxlet.PositivityConstraint(),
            max_iter=20000,
            rtol=1e-12,
            atol=1e-13,
        )
        image = spaxlet.Parameter(
            _initial_image(entry, START_WIDTH_SCALE[start]),
            name="image",
            step=spaxlet.parameter.relative_step,
            constraint=constraint,
        )
        morphology = spaxlet.ImageMorphology(frame, image, bbox=box, resizing=False)

        aperture_radius = max(2.0, 0.35 * entry.support)
        aperture = np.hypot(
            rows - entry.center[0], columns - entry.center[1]
        ) <= aperture_radius
        cube_start = np.maximum(np.sum(cube[:, aperture], axis=1), 1e-10)
        if use_imaging:
            band_start = np.maximum(np.sum(bands[:, aperture], axis=1), 1e-10)
            spectrum_start = np.concatenate([cube_start, band_start])
        else:
            spectrum_start = cube_start
        spectrum = spaxlet.TabulatedSpectrum(frame, spectrum_start)
        sources.append(spaxlet.FactorizedComponent(frame, spectrum, morphology))
    return sources


def fit_arm(catalog, *, use_imaging, start, max_iter, seed, optimizer):
    """Fit one catalog, with or without the imaging observation."""

    cube, cube_variance, cube_valid, _, _ = recovery_noisy_cube(seed=seed)
    bands, band_variance, band_valid, _ = imaging_noisy_bands(seed=seed)

    channels = JOINT_CHANNELS if use_imaging else IFU_CHANNELS
    shape = (len(channels),) + RECOVERY_SHAPE[1:]
    frame = spaxlet.Frame(
        shape, channels=channels, psf=spaxlet.DeltaPSF(len(channels))
    )
    observations = [
        spaxlet.Observation(
            cube,
            channels=IFU_CHANNELS,
            psf=spaxlet.ImagePSF(recovery_psfs()),
            weights=_weights(cube_variance, cube_valid),
        ).match(frame)
    ]
    if use_imaging:
        observations.append(
            spaxlet.Observation(
                bands,
                channels=IMAGING_CHANNELS,
                psf=spaxlet.ImagePSF(imaging_psfs()),
                weights=_weights(band_variance, band_valid),
            ).match(frame)
        )

    sources = _sources(frame, catalog, start, cube, bands, use_imaging)
    blend = spaxlet.Blend(sources, observations)
    arguments = (
        {"projected_max_backtracks": 20}
        if optimizer == "variable_projection"
        else {"scheme": "amsgrad", "channel_chunk_size": 32}
    )
    iterations, _ = blend.fit(
        max_iter, optimizer=optimizer, e_rel=0, project_initial=True, **arguments
    )

    fitted_spectra = np.asarray(
        [spaxlet.measure.factorization(source).spectrum for source in sources]
    )
    fitted_morphologies = []
    for source in sources:
        factor = spaxlet.measure.factorization(source)
        image = np.zeros(RECOVERY_SHAPE[1:], dtype=float)
        y0, x0 = source.morphology.bbox.origin
        height, width = source.morphology.bbox.shape
        image[y0 : y0 + height, x0 : x0 + width] = factor.morphology
        fitted_morphologies.append(image)

    model = np.asarray(observations[0].render(blend.get_model()), dtype=float)
    residual = cube - model
    weights = _weights(cube_variance, cube_valid)
    chi2 = float(np.sum(weights * residual**2)) / int(np.count_nonzero(cube_valid))
    return {
        # Score on the IFU channels only, so every arm is directly comparable
        # with the IFU-only benchmarks.
        "fitted_spectra": fitted_spectra[:, : RECOVERY_SHAPE[0]],
        "fitted_morphologies": np.asarray(fitted_morphologies),
        "ifu_chi2_per_valid_voxel": chi2,
        "relative_projected_gradient": (
            blend.parameter_optimization_diagnostics().relative_projected_gradient
        ),
        "iterations": int(iterations),
    }


def score(fit, catalog, matches):
    truth_spectra = recovery_spectra()
    truth_morphologies = recovery_morphologies()
    rows = []
    for detection, truth in sorted(matches.items(), key=lambda item: item[1]):
        fitted = fit["fitted_spectra"][detection]
        reference = truth_spectra[truth]
        rows.append(
            {
                "truth": RECOVERY_SOURCE_SPECS[truth].name,
                "support": catalog[detection].support,
                "truth_support": RECOVERY_SOURCE_SPECS[truth].support,
                "signed_flux_relative": float(
                    (np.sum(fitted) - np.sum(reference)) / np.sum(reference)
                ),
                "spectrum_cosine": float(
                    np.dot(fitted, reference)
                    / (np.linalg.norm(fitted) * np.linalg.norm(reference))
                ),
                "morphology_relative_l2": float(
                    np.linalg.norm(
                        fit["fitted_morphologies"][detection]
                        - truth_morphologies[truth]
                    )
                    / np.linalg.norm(truth_morphologies[truth])
                ),
            }
        )
    return rows


def run_arm(name, catalog, *, use_imaging, start, max_iter, seed, optimizer):
    fit = fit_arm(
        catalog,
        use_imaging=use_imaging,
        start=start,
        max_iter=max_iter,
        seed=seed,
        optimizer=optimizer,
    )
    matches, missed, spurious = match_to_truth(catalog)
    rows = score(fit, catalog, matches)
    signed = np.asarray([row["signed_flux_relative"] for row in rows])
    return {
        "arm": name,
        "imaging": use_imaging,
        "sources": len(catalog),
        "matched": len(matches),
        "missed": [RECOVERY_SOURCE_SPECS[t].name for t in missed],
        "ifu_chi2_per_valid_voxel": fit["ifu_chi2_per_valid_voxel"],
        "relative_projected_gradient": fit["relative_projected_gradient"],
        "max_abs_signed_flux": float(np.max(np.abs(signed))) if signed.size else None,
        "max_morphology_relative_l2": float(
            np.max([row["morphology_relative_l2"] for row in rows])
        ),
        "min_spectrum_cosine": float(
            np.min([row["spectrum_cosine"] for row in rows])
        ),
        "mean_support_error": float(
            np.mean([row["support"] - row["truth_support"] for row in rows])
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", choices=("A", "B", "C"), default="A")
    parser.add_argument("--max-iter", type=int, default=RECOVERY_CONVERGED_MAX_ITER)
    parser.add_argument("--seed", type=int, default=RECOVERY_SEED)
    parser.add_argument(
        "--optimizer",
        choices=("variable_projection", "adaprox"),
        default="variable_projection",
    )
    parser.add_argument("--threshold-sigma", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cube, cube_variance, cube_valid, _, _ = recovery_noisy_cube(seed=args.seed)
    bands, band_variance, band_valid, _ = imaging_noisy_bands(seed=args.seed)

    ifu_catalog, ifu_segmentation = detected_catalog(
        cube,
        _weights(cube_variance, cube_valid),
        threshold_sigma=args.threshold_sigma,
    )
    imaging_catalog, imaging_segmentation = detected_catalog(
        bands,
        _weights(band_variance, band_valid),
        threshold_sigma=args.threshold_sigma,
    )
    print(
        "detection: {} peaks from the IFU, {} from imaging".format(
            len(ifu_catalog), len(imaging_catalog)
        )
    )

    arms = (
        ("oracle_ifu", oracle_catalog(), False),
        ("detect_ifu", ifu_catalog, False),
        ("detect_imaging", imaging_catalog, False),
        ("joint", imaging_catalog, True),
        ("joint_oracle", oracle_catalog(), True),
    )
    results = []
    for name, catalog, use_imaging in arms:
        result = run_arm(
            name,
            catalog,
            use_imaging=use_imaging,
            start=args.start,
            max_iter=args.max_iter,
            seed=args.seed,
            optimizer=args.optimizer,
        )
        results.append(result)
        print(
            "{:<15} img={:<5} n={:<3d} ifu_chi2/N={:.6f} rpg={:.2e} "
            "max|dS|={:.5f} maxMorph={:.5f} dSupport={:+.2f}".format(
                result["arm"],
                str(result["imaging"]),
                result["sources"],
                result["ifu_chi2_per_valid_voxel"],
                result["relative_projected_gradient"],
                result["max_abs_signed_flux"],
                result["max_morphology_relative_l2"],
                result["mean_support_error"],
            )
        )

    print()
    print(
        "{:<11} {:>9} {:>9} {:>9} {:>11} {:>11}".format(
            "truth", "trueSupp", "ifuSupp", "imgSupp", "detIFU dS", "joint dS"
        )
    )
    by_arm = {result["arm"]: {row["truth"]: row for row in result["rows"]}
              for result in results}
    for spec in RECOVERY_SOURCE_SPECS:
        detect = by_arm["detect_ifu"].get(spec.name)
        imaging = by_arm["detect_imaging"].get(spec.name)
        joint = by_arm["joint"].get(spec.name)
        print(
            "{:<11} {:>9d} {:>9} {:>9} {:>11} {:>11}".format(
                spec.name,
                spec.support,
                "-" if detect is None else detect["support"],
                "-" if imaging is None else imaging["support"],
                "-" if detect is None else "{:+.4f}".format(detect["signed_flux_relative"]),
                "-" if joint is None else "{:+.4f}".format(joint["signed_flux_relative"]),
            )
        )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "joint_imaging.json").write_text(
            json.dumps(
                {
                    "start": args.start,
                    "max_iter": args.max_iter,
                    "seed": args.seed,
                    "optimizer": args.optimizer,
                    "ifu_detection_threshold": float(ifu_segmentation.threshold),
                    "imaging_detection_threshold": float(
                        imaging_segmentation.threshold
                    ),
                    "arms": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

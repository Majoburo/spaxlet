"""Run the matched soft-constraint ablation on the many-source recovery cube."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.run_synthetic_many_source_recovery import fit_recovery_cube


ARMS = (
    ("unregularized", 0, 0),
    ("spectral_300", 300, 0),
    ("spectral_3000", 3000, 0),
    ("spatial_100", 0, 100),
)


def _summary(result):
    return {
        "chi2_per_valid_voxel": result.chi2_per_valid_voxel,
        "relative_projected_gradient": result.relative_projected_gradient,
        "maximum_spectrum_relative_l2": float(np.max(result.spectrum_relative_l2)),
        "maximum_morphology_relative_l2": float(
            np.max(result.morphology_relative_l2)
        ),
        "maximum_integrated_flux_relative_error": float(
            np.max(result.integrated_flux_relative_error)
        ),
        "maximum_line_flux_relative_error": float(
            np.max(result.line_flux_relative_error)
        ),
        "maximum_line_peak_relative_error": float(
            np.max(result.line_peak_relative_error)
        ),
        "minimum_morphology_identity_margin": float(
            np.min(result.morphology_identity_margin)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", choices=("A", "B", "C"), default="A")
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        fit_recovery_cube(
            start=args.start,
            max_iter=args.max_iter,
            optimizer="adaprox",
            spectral_smoothness_strength=spectral,
            spatial_smoothness_strength=spatial,
        )
        for _, spectral, spatial in ARMS
    ]
    summaries = {
        name: _summary(result) for (name, _, _), result in zip(ARMS, results)
    }
    # The selection is a declared conclusion, so record the evidence for it in
    # the same artifact.  Writing the declaration alone made the report read as
    # a derived result while being a string literal that could not disagree
    # with the numbers beside it.
    lower_is_better = tuple(
        key for key in next(iter(summaries.values())) if not key.startswith("minimum_")
    )
    best_arm = {
        key: min(summaries, key=lambda arm: summaries[arm][key])
        for key in lower_is_better
    }
    best_arm.update(
        {
            key: max(summaries, key=lambda arm: summaries[arm][key])
            for key in next(iter(summaries.values()))
            if key.startswith("minimum_")
        }
    )
    declared = "spectral_300"
    report = {
        "selection": declared,
        "reason": (
            "improves source recovery while retaining line flux and peaks; "
            "spectral_3000 oversmooths peaks and spatial_100 biases source flux"
        ),
        "max_iter": args.max_iter,
        "best_arm_per_metric": best_arm,
        "selection_wins_metrics": sorted(
            key for key, arm in best_arm.items() if arm == declared
        ),
        "arms": summaries,
    }
    (args.output_dir / "constraint_ablation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    labels = [name for name, _, _ in ARMS]
    metrics = (
        ("maximum_spectrum_relative_l2", "max spectrum rel. L2"),
        ("maximum_morphology_relative_l2", "max morphology rel. L2"),
        ("maximum_integrated_flux_relative_error", "max total-flux error"),
        ("maximum_line_flux_relative_error", "max line-flux error"),
        ("maximum_line_peak_relative_error", "max line-peak error"),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(17, 3.8), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        values = [summaries[label][metric] for label in labels]
        colors = ["tab:green" if label == "spectral_300" else "tab:gray" for label in labels]
        axis.bar(np.arange(len(labels)), values, color=colors)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(args.output_dir / "constraint_ablation.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()

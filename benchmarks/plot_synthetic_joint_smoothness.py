"""Truth-test separate and combined spectral/spatial smoothness penalties."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.plot_synthetic_spectral_smoothness import (
    _jsonable,
    _plot_morphologies,
    _plot_residuals,
    _plot_spectra,
)
from benchmarks.run_synthetic_many_source_ifu import fit_joint_model


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spectral-strength", type=float, default=300)
    parser.add_argument("--spatial-strength", type=float, default=300)
    parser.add_argument("--slices", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--start", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--constraint",
        choices=("positivity", "centroid", "selective_symmetry", "global_symmetry"),
        default="selective_symmetry",
    )
    return parser


def main():
    args = _parser().parse_args()
    if args.spectral_strength <= 0 or args.spatial_strength <= 0:
        raise ValueError("comparison strengths must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arms = (
        ("neither", 0, 0),
        ("spectral_only", args.spectral_strength, 0),
        ("spatial_only", 0, args.spatial_strength),
        ("both", args.spectral_strength, args.spatial_strength),
    )
    results = tuple(
        fit_joint_model(
            slices=args.slices,
            support="correct",
            constraint=args.constraint,
            start=args.start,
            max_iter=args.max_iter,
            spectral_smoothness_strength=spectral,
            spatial_smoothness_strength=spatial,
        )
        for _, spectral, spatial in arms
    )
    _plot_spectra(results, args.output_dir / "spectra_truth_comparison.png")
    _plot_morphologies(results, args.output_dir / "morphology_comparison.png")
    _plot_residuals(results, args.output_dir / "residual_comparison.png")
    report = {
        "purpose": "known-truth spectral/spatial smoothness ablation",
        "arms": [
            {"name": name, **_jsonable(result)}
            for (name, _, _), result in zip(arms, results)
        ],
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

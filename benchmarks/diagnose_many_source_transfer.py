"""Signed per-source recovery diagnostics for the many-source contract.

``run_synthetic_many_source_recovery`` reports absolute errors only.  An
absolute error cannot distinguish ten independent errors from a systematic
flux transfer between overlapping factors, which is the failure mode the
two-galaxy benchmark established as the dangerous one.  This module reports
the same quantities with their sign, splits them into continuum and line
parts, and measures whether the signed errors cancel across sources.

The cancellation check is the point.  The data constrain the summed cube, so
a redistribution of flux between overlapping factors leaves the total nearly
unchanged while individual sources move.  If the per-source signed errors sum
to far less than they individually are, the errors are not independent.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from benchmarks.many_source_recovery_contract import RECOVERY_SEED
from benchmarks.run_synthetic_many_source_recovery import (
    RECOVERY_PILOT_MAX_ITER,
    fit_recovery_cube,
)
from benchmarks.signed_recovery_metrics import (
    BIN_COUNT,
    SOURCE_NAMES,
    cancellation_ratio,
    signed_source_metrics,
)


def cancellation(signed_total_absolute, truth_total):
    """Report the cancellation statistic alongside its net and gross parts."""

    net = float(np.sum(signed_total_absolute))
    return {
        "net_absolute": net,
        "gross_absolute": float(np.sum(np.abs(signed_total_absolute))),
        "cancellation_ratio": cancellation_ratio(signed_total_absolute),
        "net_relative_to_scene": net / float(np.sum(truth_total)),
    }


def run(starts, seeds, max_iter):
    """Fit every start/seed combination and collect signed diagnostics."""

    runs = []
    for seed in seeds:
        for start in starts:
            result = fit_recovery_cube(start=start, max_iter=max_iter, seed=seed)
            metrics = signed_source_metrics(result.fitted_spectra)
            runs.append(
                {
                    "start": start,
                    "seed": int(seed),
                    "chi2_per_valid_voxel": result.chi2_per_valid_voxel,
                    "relative_projected_gradient": result.relative_projected_gradient,
                    "metrics": metrics,
                    "cancellation": cancellation(
                        metrics["signed_total_absolute"], metrics["truth_total"]
                    ),
                }
            )
    return runs


def _stack(runs, key):
    return np.asarray([run["metrics"][key] for run in runs])


def report(runs):
    """Print a per-source signed summary across all fitted runs."""

    total = _stack(runs, "signed_total_relative")
    continuum = _stack(runs, "signed_continuum_relative")
    line = _stack(runs, "signed_line_relative")

    print("runs: {}".format(len(runs)))
    for run in runs:
        print(
            "  start {} seed {}: chi2/N={:.4f} rpg={:.2e} "
            "cancellation={:.3f} net/scene={:+.5f}".format(
                run["start"],
                run["seed"],
                run["chi2_per_valid_voxel"],
                run["relative_projected_gradient"],
                run["cancellation"]["cancellation_ratio"],
                run["cancellation"]["net_relative_to_scene"],
            )
        )

    print()
    header = "{:<11} {:>18} {:>18} {:>18} {:>7}"
    print(header.format("source", "signed total", "continuum", "line", "sign"))
    for index, name in enumerate(SOURCE_NAMES):
        signs = np.sign(total[:, index])
        consistent = "yes" if np.all(signs == signs[0]) and signs[0] != 0 else "NO"
        print(
            "{:<11} {:>+9.4f}+-{:<7.4f} {:>+9.4f}+-{:<7.4f} {:>+9.4f}+-{:<7.4f} {:>7}".format(
                name,
                total[:, index].mean(),
                total[:, index].std(),
                continuum[:, index].mean(),
                continuum[:, index].std(),
                line[:, index].mean(),
                line[:, index].std(),
                consistent,
            )
        )

    print()
    print("mean binned signed relative error (8 channel bins)")
    binned = _stack(runs, "binned_signed_relative").mean(axis=0)
    print("{:<11} {}".format("source", " ".join("{:>8d}".format(b) for b in range(BIN_COUNT))))
    for index, name in enumerate(SOURCE_NAMES):
        print(
            "{:<11} {}".format(
                name, " ".join("{:>+8.4f}".format(value) for value in binned[index])
            )
        )


def to_json(runs):
    payload = []
    for run in runs:
        entry = {
            key: run[key]
            for key in ("start", "seed", "chi2_per_valid_voxel", "relative_projected_gradient")
        }
        entry["cancellation"] = run["cancellation"]
        entry["signed"] = {
            name: {
                "total_relative": float(run["metrics"]["signed_total_relative"][index]),
                "continuum_relative": float(
                    run["metrics"]["signed_continuum_relative"][index]
                ),
                "line_relative": float(run["metrics"]["signed_line_relative"][index]),
                "binned_relative": [
                    float(value)
                    for value in run["metrics"]["binned_signed_relative"][index]
                ],
            }
            for index, name in enumerate(SOURCE_NAMES)
        }
        payload.append(entry)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", default="A,B,C")
    parser.add_argument("--seeds", default=str(RECOVERY_SEED))
    parser.add_argument("--max-iter", type=int, default=RECOVERY_PILOT_MAX_ITER)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    starts = tuple(args.starts.split(","))
    seeds = tuple(int(value) for value in args.seeds.split(","))
    runs = run(starts, seeds, args.max_iter)
    report(runs)

    if args.output:
        with open(args.output, "w") as stream:
            json.dump(to_json(runs), stream, indent=2, sort_keys=True)
        print("\nwrote {}".format(args.output))


if __name__ == "__main__":
    main()

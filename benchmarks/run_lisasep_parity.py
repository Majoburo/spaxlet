"""Run the existing cross-repository parity harness from one command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


SCARLET_BASE = "3ce064d714d27f8dcbdb9a77c438272960697d16"
LISASEP_REFERENCE = "8dbafc835fc5712bac10c882810b05a1bdc456de"


def _git(repo, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _run(command, environment):
    print("+ " + " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lisasep", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable, type=Path)
    parser.add_argument(
        "--output-dir", default=Path("benchmark_artifacts/ifu_parity"), type=Path
    )
    parser.add_argument(
        "--operator-only",
        action="store_true",
        help="skip the slower six-channel end-to-end matrix",
    )
    parser.add_argument("--lisasep-max-iter", type=int, default=500)
    parser.add_argument("--spaxlet-max-iter", type=int, default=600)
    args = parser.parse_args()

    spaxlet = Path(__file__).resolve().parents[1]
    lisasep = args.lisasep.resolve()
    operator_script = (
        lisasep / "examples/benchmark_twogalaxy/compare_constraint_operators.py"
    )
    feature_script = (
        lisasep / "examples/benchmark_twogalaxy/compare_deblend_features.py"
    )
    for path in (operator_script, feature_script):
        if not path.is_file():
            parser.error("missing lisasep comparison script: {}".format(path))

    scarlet_base = _git(spaxlet, "merge-base", "HEAD", SCARLET_BASE)
    if scarlet_base != SCARLET_BASE:
        parser.error("Scarlet branch does not descend from {}".format(SCARLET_BASE))
    lisasep_head = _git(lisasep, "rev-parse", "HEAD")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    python_path = [str(spaxlet), str(lisasep / "src")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    print("Scarlet base: {}".format(SCARLET_BASE))
    print("lisasep reference: {}".format(LISASEP_REFERENCE))
    print("lisasep tested HEAD: {}".format(lisasep_head))
    if lisasep_head != LISASEP_REFERENCE:
        print("WARNING: lisasep HEAD differs from the pinned reference")

    _run(
        [
            str(args.python),
            str(operator_script),
            "--output",
            str(args.output_dir / "constraint_operators.json"),
        ],
        environment,
    )
    if not args.operator_only:
        _run(
            [
                str(args.python),
                str(feature_script),
                "--output",
                str(args.output_dir / "deblend_features.json"),
                "--lisasep-max-iter",
                str(args.lisasep_max_iter),
                "--spaxlet-max-iter",
                str(args.scarlet_max_iter),
            ],
            environment,
        )


if __name__ == "__main__":
    main()

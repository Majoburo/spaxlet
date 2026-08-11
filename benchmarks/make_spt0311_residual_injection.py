"""Inject a fitted scene into block-bootstrap realizations of real residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--prism-block-channels", type=int, default=8)
    parser.add_argument("--g395h-block-channels", type=int, default=32)
    return parser


def block_wild_residual(residual, valid, block_channels, generator):
    """Flip complete spectral blocks while preserving their internal covariance."""

    values = np.asarray(residual, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 3 or mask.shape != values.shape:
        raise ValueError("residual and validity mask must share (channel, y, x)")
    if not isinstance(block_channels, (int, np.integer)) or block_channels <= 0:
        raise ValueError("spectral block size must be a positive integer")
    block_count = int(np.ceil(values.shape[0] / block_channels))
    signs = generator.choice(np.asarray([-1.0, 1.0]), size=block_count)
    channel_signs = np.repeat(signs, block_channels)[: values.shape[0]]
    realized = values * channel_signs[:, None, None]
    realized[~mask] = 0
    return realized, signs


def make_injection(product_path, seed, block_channels):
    generator = np.random.default_rng(seed)
    output = {}
    signs = {}
    with np.load(product_path) as product:
        for arm in ("prism", "g395h"):
            model = np.asarray(product[arm + "_model"], dtype=float)
            residual = np.asarray(product[arm + "_residual"], dtype=float)
            valid = np.asarray(product[arm + "_valid_mask"], dtype=bool)
            realized, arm_signs = block_wild_residual(
                residual, valid, block_channels[arm], generator
            )
            data = model + realized
            data[~valid] = 0
            output[arm + "_data"] = data.astype(np.float32)
            signs[arm] = arm_signs.astype(int).tolist()
    metadata = {
        "kind": "real_residual_injection",
        "baseline_product": str(product_path.resolve()),
        "seed": int(seed),
        "spectral_block_channels": block_channels,
        "block_signs": signs,
        "interpretation": (
            "fitted scene injected into block-wild-bootstrap real residuals; "
            "within-block spatial/spectral correlation is preserved"
        ),
    }
    output["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return output


def main():
    args = _parser().parse_args()
    blocks = {
        "prism": args.prism_block_channels,
        "g395h": args.g395h_block_channels,
    }
    output = make_injection(args.product, args.seed, blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()

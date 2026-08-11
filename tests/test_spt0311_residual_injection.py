"""Tests for residual-preserving SPT0311 injection realizations."""

import numpy as np

from benchmarks.make_spt0311_residual_injection import block_wild_residual


def test_block_wild_residual_preserves_values_and_within_block_covariance():
    residual = np.arange(8 * 3 * 3, dtype=float).reshape(8, 3, 3) + 1
    valid = np.ones_like(residual, dtype=bool)
    valid[:, 0, 0] = False
    realized, signs = block_wild_residual(
        residual, valid, 2, np.random.default_rng(4)
    )
    assert signs.shape == (4,)
    np.testing.assert_array_equal(realized[:, 0, 0], 0)
    for start, sign in zip(range(0, 8, 2), signs):
        np.testing.assert_array_equal(
            realized[start : start + 2, 1:, 1:],
            sign * residual[start : start + 2, 1:, 1:],
        )


def test_block_wild_residual_rejects_invalid_block_size():
    values = np.ones((2, 2, 2))
    try:
        block_wild_residual(values, np.ones_like(values, dtype=bool), 0, np.random.default_rng(1))
    except ValueError:
        pass
    else:
        raise AssertionError("zero-sized residual block was accepted")

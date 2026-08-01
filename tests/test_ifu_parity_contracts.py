"""Pinned contracts for the surgical lisasep IFU parity branch."""

import numpy as np
import unittest

import scarlet
from benchmarks.ifu_parity_contracts import (
    CASE_NAMES,
    DEBLEND_SHAPE,
    N_CHANNELS,
    OPERATOR_CENTER,
    OPERATOR_SHAPE,
    deblend_cases,
    noisy_cube,
    operator_morphologies,
)


def _relative_l2(value, reference):
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def _project(feature, value):
    if feature == "symmetry":
        return scarlet.operator.prox_uncentered_symmetry(
            value.copy(),
            0.0,
            center=OPERATOR_CENTER,
            algorithm="soft",
            strength=1.0,
        )
    prox = scarlet.operator.prox_weighted_monotonic(
        OPERATOR_SHAPE,
        center=OPERATOR_CENTER,
        neighbor_weight=feature,
        min_gradient=0.0,
    )
    return np.asarray(prox(value.copy(), 0.0)).reshape(OPERATOR_SHAPE)


class IFUParityContracts(unittest.TestCase):
    def test_operator_fixture_contract(self):
        cases = operator_morphologies()
        self.assertEqual(
            tuple(cases),
            ("compatible", "secondary_peak", "clumpy_misspecified"),
        )
        self.assertTrue(
            all(value.shape == OPERATOR_SHAPE for value in cases.values())
        )
        self.assertTrue(all(np.all(value >= 0) for value in cases.values()))
        self.assertAlmostEqual(cases["compatible"][OPERATOR_CENTER], 1.0)
        self.assertGreater(
            cases["secondary_peak"][8, 8],
            cases["secondary_peak"][OPERATOR_CENTER],
        )

    def test_clumpy_negative_control_pins_prior_distortion(self):
        expected = {
            "symmetry": 0.20866072478762615,
            "flat": 0.07784435198218645,
            "angle": 0.06449289718163626,
            "nearest": 0.046566174091379094,
        }
        value = operator_morphologies()["clumpy_misspecified"]
        for feature, expected_change in expected.items():
            with self.subTest(feature=feature):
                projected = _project(feature, value)
                self.assertAlmostEqual(
                    _relative_l2(projected, value), expected_change, places=10
                )

    def test_compatible_operator_null(self):
        value = operator_morphologies()["compatible"]
        for feature in ("symmetry", "flat", "angle", "nearest"):
            with self.subTest(feature=feature):
                self.assertLess(_relative_l2(_project(feature, value), value), 1e-14)

    def test_six_channel_deblend_contract(self):
        cases = deblend_cases()
        self.assertEqual(tuple(cases), CASE_NAMES)
        for case_name, case in cases.items():
            self.assertEqual(len(case["centers"]), 2)
            self.assertEqual(len(case["morphologies"]), 2)
            self.assertEqual(len(case["starts"]), 2)
            self.assertTrue(
                all(value.shape == DEBLEND_SHAPE for value in case["morphologies"])
            )
            self.assertTrue(
                all(value.shape == DEBLEND_SHAPE for value in case["starts"])
            )
            for value in case["morphologies"]:
                self.assertAlmostEqual(value.sum(), 1.0)
            data, noise, truth = noisy_cube(case_name)
            self.assertEqual(data.shape, (N_CHANNELS,) + DEBLEND_SHAPE)
            self.assertEqual(truth.shape, data.shape)
            self.assertGreater(noise, 0)
            self.assertTrue(np.all(np.isfinite(data)))

        null_separation = np.subtract(*cases["compatible_null"]["centers"])
        close_separation = np.subtract(*cases["close_blend_helpful"]["centers"])
        self.assertAlmostEqual(np.linalg.norm(null_separation), 8.0)
        self.assertAlmostEqual(np.linalg.norm(close_separation), 4.0)

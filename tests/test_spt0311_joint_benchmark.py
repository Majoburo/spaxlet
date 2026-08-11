"""Focused tests for the two-observation SPT0311 benchmark plumbing."""

import unittest
from collections import OrderedDict

import numpy as np

from benchmarks.run_spt0311_joint_deblend import (
    common_centers,
    factor_observables,
    joint_channel_layout,
    parse_morphology_constraint_overrides,
    prism_resolving_power,
    shared_spectral_layout,
    start_sigma_scale,
)


class SPT0311JointBenchmarkTest(unittest.TestCase):
    def test_joint_channels_are_unique_contiguous_segments(self):
        channels, slices = joint_channel_layout([2, 3], [10, 11, 12])
        self.assertEqual(
            channels,
            ("prism:0002", "prism:0003", "g395h:0010", "g395h:0011", "g395h:0012"),
        )
        self.assertEqual(channels[slices["prism"]], channels[:2])
        self.assertEqual(channels[slices["g395h"]], channels[2:])

    def test_common_centers_average_registered_observations(self):
        per_observation = OrderedDict(
            prism=OrderedDict(A=np.asarray((2.0, 3.0)), B=np.asarray((5.0, 7.0))),
            g395h=OrderedDict(A=np.asarray((2.2, 2.8)), B=np.asarray((5.0, 7.0))),
        )
        centers, disagreement = common_centers(per_observation)
        np.testing.assert_allclose(centers["A"], (2.1, 2.9))
        self.assertAlmostEqual(disagreement["A"], np.sqrt(0.02))
        self.assertEqual(disagreement["B"], 0)

    def test_per_source_morphology_constraint_overrides(self):
        overrides = parse_morphology_constraint_overrides(
            ["W=monotonic,lens=symmetry"], ("lens", "W", "E")
        )
        self.assertEqual(overrides, {"W": "monotonic", "lens": "symmetry"})
        with self.assertRaisesRegex(ValueError, "unknown morphology override source"):
            parse_morphology_constraint_overrides(["missing=monotonic"], ("W",))
        with self.assertRaisesRegex(ValueError, "duplicate morphology override source"):
            parse_morphology_constraint_overrides(
                ["W=monotonic", "W=centroid"], ("W",)
            )

    def test_predeclared_starts_exchange_foreground_and_high_z_widths(self):
        self.assertEqual(start_sigma_scale(("lens",), "A"), 1.0)
        self.assertEqual(start_sigma_scale(("W",), "A"), 1.0)
        self.assertEqual(start_sigma_scale(("lens",), "B"), 1.5)
        self.assertEqual(start_sigma_scale(("W",), "B"), 0.7)
        self.assertEqual(start_sigma_scale(("lens",), "C"), 0.7)
        self.assertEqual(start_sigma_scale(("W",), "C"), 1.5)

    def test_factor_observables_use_integrated_unit_morphology_gauge(self):
        wavelength = np.asarray([2.92, 2.94, 2.96, 2.98])
        spectrum = np.asarray([1.0, 2.0, 3.0, 4.0])
        morphology = np.asarray([[0.0, 1.0], [0.0, 1.0]])
        result = factor_observables(
            wavelength, spectrum, morphology, origin_yx=(10, 20)
        )
        np.testing.assert_allclose(result["centroid_yx"], [10.5, 21.0])
        self.assertAlmostEqual(result["effective_morphology_pixels"], 2.0)
        self.assertAlmostEqual(result["integrated_spectrum_l1"], 20.0)
        self.assertIsNotNone(
            result["window_flux_density_integrals"]["oii"]
        )

    def test_shared_layout_uses_fine_grid_only_in_overlap(self):
        prism = np.asarray([0.5, 0.75, 1.25, 1.75, 2.25, 2.5])
        g395h = np.asarray([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        channels, latent, responses, layout = shared_spectral_layout(
            prism, g395h, np.float32
        )
        np.testing.assert_allclose(
            latent, [0.5, 0.75, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5]
        )
        self.assertEqual(len(channels), latent.size)
        np.testing.assert_array_equal(layout["prism_low_indices"], [0, 1])
        np.testing.assert_array_equal(layout["prism_high_indices"], [4, 5])
        np.testing.assert_array_equal(
            responses["g395h"].indices[:, 0], np.arange(2, 8)
        )
        for response in responses.values():
            np.testing.assert_allclose(response.weights.sum(axis=1), 1)

    def test_calibrated_prism_response_is_broader_than_top_hat(self):
        prism = np.arange(2.8, 3.21, 0.005)
        g395h = np.arange(2.87, 3.201, 0.000665)
        _, _, binned, _ = shared_spectral_layout(prism, g395h, np.float32)
        _, _, broadened, _ = shared_spectral_layout(
            prism, g395h, np.float32, "official-gaussian"
        )
        overlap = (prism >= g395h[0]) & (prism <= g395h[-1])
        binned_support = np.count_nonzero(binned["prism"].weights[overlap], axis=1)
        broadened_support = np.count_nonzero(
            broadened["prism"].weights[overlap], axis=1
        )
        self.assertTrue(np.all(broadened_support > binned_support))
        self.assertAlmostEqual(prism_resolving_power([3.0])[0], 101.221345)


if __name__ == "__main__":
    unittest.main()

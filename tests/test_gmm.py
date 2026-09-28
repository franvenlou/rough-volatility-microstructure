"""Regression checks for rolling features and the exploratory GMM selector."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from src.gmm_latent_space import generate_latent_space, plot_gmm_3d, rolling_latent_features
from src.regime_gmm import LatentRegimeIsolator


class RollingFeatureTests(unittest.TestCase):
    def test_complete_windows_include_final_window_and_known_features(self):
        # Linear log-return magnitudes have a first-moment scaling exponent of one.
        returns = np.exp(np.arange(8, dtype=float)) * 1e-5
        volumes = np.ones(8)
        features = rolling_latent_features(returns, volumes, window_size=6, max_lag=3)
        self.assertEqual(features.shape, (3, 3))
        assert_allclose(features[:, 0], 1.0, atol=1e-12)
        assert_allclose(features[:, 1], np.log(6), atol=1e-12)
        expected_means = np.array([returns[start : start + 6].mean() * 1e4 for start in range(3)])
        assert_allclose(features[:, 2], expected_means)

    def test_exactly_one_complete_window_is_retained(self):
        returns = np.exp(np.arange(6, dtype=float)) * 1e-5
        self.assertEqual(rolling_latent_features(returns, np.ones(6), 6, 3).shape, (1, 3))

    def test_constant_windows_do_not_fabricate_half_hurst(self):
        with self.assertWarnsRegex(UserWarning, "Omitted 3 rolling windows"):
            features = rolling_latent_features(np.ones(8) * 1e-4, np.ones(8), 6, 3)
        self.assertEqual(features.shape, (0, 3))

    def test_invalid_alignment_and_insufficient_observations_fail(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            rolling_latent_features(np.ones(8), np.ones(7), 6, 3)
        with self.assertRaisesRegex(ValueError, "fewer returns"):
            rolling_latent_features(np.ones(5), np.ones(5), 6, 3)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            rolling_latent_features(np.zeros(8), np.ones(8), 6, 3)

    def test_parquet_features_align_volume_with_return_endpoint(self):
        magnitudes = np.exp(np.arange(8, dtype=float)) * 1e-5
        mid_prices = 100 * np.exp(np.r_[0, np.cumsum(magnitudes)])
        # The first snapshot has no incoming return and must not affect entropy.
        volumes = np.r_[1000.0, np.ones(8)]
        snapshots = pd.DataFrame({
            "bid_price": mid_prices - 0.01,
            "ask_price": mid_prices + 0.01,
            "bid_vol": volumes / 2,
            "ask_vol": volumes / 2,
        })
        with TemporaryDirectory() as directory:
            path = Path(directory) / "snapshots.parquet"
            snapshots.to_parquet(path, index=False)
            features = generate_latent_space(path, window_size=6, max_lag=3)
        self.assertEqual(features.shape, (3, 3))
        assert_allclose(features[:, 0], 1.0, atol=1e-8)
        assert_allclose(features[:, 1], np.log(6), atol=1e-12)

    def test_gmm_plot_saves_output_and_reports_feature_units(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(40, 3)) + [0.4, 3.5, 20]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "gmm.png"
            fig, model = plot_gmm_3d(features, output=output, show=False)
            try:
                self.assertTrue(output.is_file())
                self.assertEqual(model.n_components, 2)
                self.assertIn("nats", fig.axes[0].get_ylabel())
                self.assertEqual(fig.axes[0].get_zlabel(), "Mean absolute return (bps)")
            finally:
                plt.close(fig)

    def test_gmm_rejects_insufficient_valid_windows(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            plot_gmm_3d(np.empty((0, 3)), show=False)


class ExploratoryRegimeTests(unittest.TestCase):
    def make_isolator(self):
        rng = np.random.default_rng(123)
        first = rng.normal([0.25, 0.2, 15], [0.02, 0.02, 1], size=(100, 3))
        second = rng.normal([0.50, 0.8, 85], [0.02, 0.02, 1], size=(100, 3))
        values = np.vstack([first, second])
        return LatentRegimeIsolator(*values.T, initial_precision=np.eye(3))

    def test_bic_selects_two_separated_synthetic_groups(self):
        isolator = self.make_isolator()
        isolator.fit_optimal_gmm((1, 2))
        self.assertEqual(isolator.optimal_k, 2)
        self.assertLess(isolator.bic_scores_[2], isolator.bic_scores_[1])
        selected = isolator.select_centroid(entropy_upper_bound=0.4)
        # The closer H=0.5 group is ineligible under the entropy bound.
        self.assertAlmostEqual(selected.hurst_mean, 0.25, delta=0.01)
        self.assertLess(selected.entropy_mean, 0.4)
        self.assertFalse(hasattr(selected, "is_tradable_bs"))
        with self.assertRaisesRegex(RuntimeError, "No component"):
            isolator.select_centroid(entropy_upper_bound=-1.0)

    def test_selection_requires_fit(self):
        with self.assertRaisesRegex(ValueError, "Fit a GMM"):
            self.make_isolator().select_centroid()

    def test_initial_precision_must_match_feature_coordinates_and_be_positive_definite(self):
        values = np.arange(6, dtype=float)
        for precision in (np.eye(4), np.diag([1, -1, 1]), np.ones((3, 3))):
            with self.subTest(precision=precision):
                with self.assertRaises(ValueError):
                    LatentRegimeIsolator(values, values, values, precision)

    def test_feature_lengths_and_component_range_are_validated(self):
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            LatentRegimeIsolator(np.ones(3), np.ones(4), np.ones(3), np.eye(3))
        isolator = self.make_isolator()
        for bounds in ((0, 2), (3, 2), (2, 201)):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    isolator.fit_optimal_gmm(bounds)


if __name__ == "__main__":
    unittest.main()

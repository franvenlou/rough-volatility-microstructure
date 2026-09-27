"""Closed-form and seeded synthetic checks; these do not validate market data."""

import unittest

import numpy as np

from src.microstruct.core_math import (
    calculate_hurst_variogram,
    gaussian_kernel,
    nadaraya_watson_smoother,
    shannon_entropy,
)


class HurstVariogramTests(unittest.TestCase):
    def test_linear_series_has_exact_unit_scaling(self):
        # The q-th increment moment of a linear path is proportional to lag**q.
        # This detects the old sample-covariance/population-variance mismatch.
        series = 3.0 * np.arange(100) + 7.0
        for order in (1, 2):
            for max_lag in (2, 5, 20):
                with self.subTest(order=order, max_lag=max_lag):
                    self.assertAlmostEqual(
                        calculate_hurst_variogram(series, max_lag, order), 1.0, places=12
                    )

    def test_default_is_first_absolute_moment(self):
        series = np.array([0.0, 0.3, -2.0, 1.0, 1.2, 4.0, 2.0, -1.0])
        lags = np.arange(1, 5)
        for order in (1, 2):
            moments = [np.mean(np.abs(series[k:] - series[:-k]) ** order) for k in lags]
            expected = np.polyfit(np.log(lags), np.log(moments), 1)[0] / order
            self.assertAlmostEqual(calculate_hurst_variogram(series, 4, order), expected)
        self.assertEqual(
            calculate_hurst_variogram(series, 4), calculate_hurst_variogram(series, 4, 1)
        )

    def test_seeded_brownian_path_recovers_half(self):
        brownian = np.cumsum(np.random.default_rng(2026).normal(size=100_000))
        for order in (1, 2):
            with self.subTest(order=order):
                self.assertAlmostEqual(
                    calculate_hurst_variogram(brownian, 50, order), 0.5, delta=0.04
                )

    def test_fractional_brownian_ensemble_recovers_known_h(self):
        # Unit-grid fractional Gaussian noise covariance is the second difference
        # of |k|**(2H). Its cumulative sum gives fBm sampled at integer times.
        hurst = 0.2
        length = 512
        indices = np.arange(length)
        distance = np.abs(indices[:, None] - indices[None, :])
        covariance = 0.5 * (
            (distance + 1) ** (2 * hurst)
            - 2 * distance ** (2 * hurst)
            + np.abs(distance - 1) ** (2 * hurst)
        )
        innovations = np.random.default_rng(1701).normal(size=(length, 24))
        paths = np.cumsum(np.linalg.cholesky(covariance) @ innovations, axis=0)
        for order in (1, 2):
            estimates = [calculate_hurst_variogram(path, 20, order) for path in paths.T]
            with self.subTest(order=order):
                self.assertAlmostEqual(float(np.mean(estimates)), hurst, delta=0.04)

    def test_rejects_undefined_or_invalid_fits(self):
        cases = [
            (np.ones(30), 20, 1),
            (np.tile([0.0, 1.0], 20), 4, 1),
            (np.arange(10.0), 1, 1),
            (np.arange(10.0), 10, 1),
            (np.arange(10.0), 2.5, 1),
            (np.arange(10.0), True, 1),
            (np.arange(10.0), 3, 3),
            (np.arange(10.0), 3, 1.5),
            (np.array([0.0, 1.0, np.nan, 4.0]), 2, 1),
            (np.array([0.0, 1.0, np.inf, 4.0]), 2, 1),
            (np.ones((4, 4)), 2, 1),
            (np.array([]), 2, 1),
        ]
        for series, lag, order in cases:
            with self.subTest(series=series, lag=lag, order=order):
                with self.assertRaises(ValueError):
                    calculate_hurst_variogram(series, lag, order)


class VolumeEntropyTests(unittest.TestCase):
    def test_uniform_volume_has_log_number_of_categories_entropy(self):
        self.assertAlmostEqual(shannon_entropy(np.full(7, 3.0)), np.log(7.0))

    def test_nonuniform_volume_matches_closed_form_and_is_scale_invariant(self):
        expected = -0.25 * np.log(0.25) - 0.75 * np.log(0.75)
        self.assertAlmostEqual(shannon_entropy(np.array([0.0, 1.0, 3.0])), expected)
        self.assertAlmostEqual(shannon_entropy(np.array([0.0, 100.0, 300.0])), expected)
        self.assertAlmostEqual(shannon_entropy(np.array([0.0, 1e308, 1e308])), np.log(2))

    def test_empty_zero_and_single_category_have_zero_entropy(self):
        for values in ([], [0.0, 0.0], [0.0, 8.0, 0.0]):
            self.assertEqual(shannon_entropy(np.array(values)), 0.0)

    def test_rejects_invalid_volume_weights(self):
        for values in ([1.0, -1.0], [np.nan], [np.inf], [[1.0, 2.0]]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    shannon_entropy(np.array(values))


class SmootherTests(unittest.TestCase):
    def test_gaussian_normalization_and_symmetry(self):
        self.assertAlmostEqual(gaussian_kernel(0.0), 1.0 / np.sqrt(2.0 * np.pi))
        self.assertAlmostEqual(gaussian_kernel(-2.0), gaussian_kernel(2.0))

    def test_constant_values_remain_constant_on_irregular_grid(self):
        coordinates = np.array([0.0, 0.1, 0.1, 1.5, 4.0])
        fitted = nadaraya_watson_smoother(coordinates, np.full(5, 3.5), 0.7)
        np.testing.assert_allclose(fitted, 3.5)

    def test_two_point_smoother_matches_closed_form(self):
        weight = np.exp(-0.5)
        expected = np.array([2.0 * weight, 2.0]) / (1.0 + weight)
        actual = nadaraya_watson_smoother(np.array([0.0, 1.0]), np.array([0.0, 2.0]), 1.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_small_constant_is_preserved_exactly_before_log_scaling(self):
        values = np.full(100, 1e-8)
        fitted = nadaraya_watson_smoother(np.arange(100.0), values, 2.0)
        np.testing.assert_array_equal(fitted, values)
        with self.assertRaises(ValueError):
            calculate_hurst_variogram(np.log(fitted), 20)

    def test_small_bandwidth_recovers_distinct_observations(self):
        values = np.array([1.0, 5.0, -2.0])
        fitted = nadaraya_watson_smoother(np.arange(3.0), values, 1e-4)
        np.testing.assert_allclose(fitted, values)

    def test_empty_input_returns_empty_vector(self):
        self.assertEqual(nadaraya_watson_smoother(np.array([]), np.array([]), 1.0).shape, (0,))

    def test_rejects_invalid_bandwidth_and_misaligned_input(self):
        for bandwidth in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(bandwidth=bandwidth):
                with self.assertRaises(ValueError):
                    nadaraya_watson_smoother(np.arange(3.0), np.arange(3.0), bandwidth)
        with self.assertRaises(ValueError):
            nadaraya_watson_smoother(np.arange(3.0), np.arange(4.0), 1.0)
        with self.assertRaises(ValueError):
            nadaraya_watson_smoother(np.arange(3.0), np.array([0.0, np.nan, 1.0]), 1.0)


if __name__ == "__main__":
    unittest.main()

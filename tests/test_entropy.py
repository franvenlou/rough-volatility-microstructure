"""Exact small-distribution checks for imbalance and discrete information measures."""

import unittest

import numpy as np

from src.microstruct.entropy import (
    compute_volume_imbalance,
    discretize_imbalance,
    estimate_mutual_information,
    estimate_shannon_entropy,
)


class ImbalanceTests(unittest.TestCase):
    def test_requested_level_and_empty_depth(self):
        lob = np.array([
            [10.0, 2.0, 11.0, 2.0, 9.0, 3.0, 12.0, 1.0],
            [10.0, 0.0, 11.0, 0.0, 9.0, 0.0, 12.0, 1.0],
            [10.0, 1e-300, 11.0, 0.0, 9.0, 1e308, 12.0, 1e308],
        ])
        np.testing.assert_allclose(compute_volume_imbalance(lob), [0.0, 0.0, 1.0])
        np.testing.assert_allclose(compute_volume_imbalance(lob, level=2), [0.5, -1.0, 0.0])

    def test_discretization_includes_endpoints_and_assigns_boundaries(self):
        values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        np.testing.assert_array_equal(discretize_imbalance(values, bins=4), [0, 1, 2, 3, 3])
        np.testing.assert_array_equal(discretize_imbalance(values, bins=1), [0, 0, 0, 0, 0])

    def test_invalid_depth_inputs_are_rejected(self):
        for lob, level in ((np.ones((2, 5)), 1), (np.ones((2, 4)), 0), (np.ones((2, 4)), 2)):
            with self.subTest(level=level, shape=lob.shape):
                with self.assertRaises(ValueError):
                    compute_volume_imbalance(lob, level)
        for volume in (-1.0, np.inf, np.nan):
            with self.assertRaises(ValueError):
                compute_volume_imbalance(np.array([[1.0, volume, 2.0, 1.0]]))

    def test_invalid_discretization_inputs_are_rejected(self):
        for values, bins in (([1.01], 4), ([-1.01], 4), ([np.nan], 4), ([0.0], 0), ([0.0], 2.5)):
            with self.subTest(values=values, bins=bins):
                with self.assertRaises(ValueError):
                    discretize_imbalance(np.array(values), bins)


class DiscreteEntropyTests(unittest.TestCase):
    def test_uniform_states_and_arbitrary_logarithm_bases(self):
        states = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        self.assertAlmostEqual(estimate_shannon_entropy(states), 2.0)
        self.assertAlmostEqual(estimate_shannon_entropy(states, base=np.e), np.log(4))
        self.assertAlmostEqual(estimate_shannon_entropy(states, base=10), np.log10(4))

    def test_sparse_labels_use_frequencies_instead_of_label_magnitudes(self):
        expected = -0.25 * np.log2(0.25) - 0.75 * np.log2(0.75)
        self.assertAlmostEqual(estimate_shannon_entropy(np.array([3, 10**9, 10**9, 10**9])), expected)

    def test_constant_and_empty_sequences_have_zero_entropy(self):
        self.assertEqual(estimate_shannon_entropy(np.array([5, 5, 5])), 0.0)
        self.assertEqual(estimate_shannon_entropy(np.array([], dtype=int)), 0.0)

    def test_invalid_states_and_logarithm_bases_are_rejected(self):
        for sequence in ([-1, 0], [0.5, 1.0], [[0, 1]], [np.nan]):
            with self.assertRaises(ValueError):
                estimate_shannon_entropy(np.array(sequence))
        for base in (0.0, 1.0, -2.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                estimate_shannon_entropy(np.array([0, 1]), base)


class MutualInformationTests(unittest.TestCase):
    def test_identical_balanced_states_share_one_bit(self):
        sequence = np.array([0, 1, 0, 1])
        mutual_information, normalized = estimate_mutual_information(sequence, sequence, 2)
        self.assertAlmostEqual(mutual_information, 1.0)
        self.assertAlmostEqual(normalized, 1.0)

    def test_balanced_independent_pairs_have_zero_mutual_information(self):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])
        np.testing.assert_allclose(estimate_mutual_information(x, y, 2), [0.0, 0.0], atol=1e-14)

    def test_partial_dependence_uses_joint_entropy_normalization(self):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 0, 0, 1])
        h_y = -0.25 * np.log2(0.25) - 0.75 * np.log2(0.75)
        expected_mi = 1.0 + h_y - 1.5
        actual = estimate_mutual_information(x, y, 2)
        np.testing.assert_allclose(actual, [expected_mi, expected_mi / 1.5], rtol=1e-12)

    def test_empty_and_constant_pairs_return_zero(self):
        empty = np.array([], dtype=int)
        self.assertEqual(estimate_mutual_information(empty, empty, 2), (0.0, 0.0))
        self.assertEqual(estimate_mutual_information(np.zeros(4, dtype=int), np.ones(4, dtype=int), 2), (0.0, 0.0))

    def test_invalid_pair_alignment_and_bin_bounds_are_rejected(self):
        for x, y, bins in (([0, 1], [0], 2), ([0, 2], [0, 1], 2), ([0], [-1], 2), ([0], [0], 0)):
            with self.subTest(x=x, y=y, bins=bins):
                with self.assertRaises(ValueError):
                    estimate_mutual_information(np.array(x), np.array(y), bins)


if __name__ == "__main__":
    unittest.main()

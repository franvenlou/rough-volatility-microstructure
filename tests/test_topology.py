"""Graph invariants and regression tests for topology-based portfolio experiments."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
from numpy.testing import assert_allclose

from src.topology.network_filters import TopologicalFilter
from src.topology.pmfg_filter import build_pmfg
from src.topology.portfolio import calculate_standard_weights, calculate_topological_weights
from src.topology.visualize_pmfg import plot_pmfg


class PMFGTests(unittest.TestCase):
    def test_pmfg_is_planar_and_saturates_edge_bound(self):
        returns = np.random.default_rng(42).normal(size=(8, 100))
        correlations = np.corrcoef(returns)
        graph = build_pmfg(correlations)
        self.assertTrue(nx.check_planarity(graph)[0])
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(graph.number_of_edges(), 3 * (8 - 2))
        for left, right, data in graph.edges(data=True):
            self.assertEqual(data["weight"], correlations[left, right])

    def test_small_graph_keeps_negative_and_zero_correlations(self):
        correlations = np.array([[1, -0.4, 0], [-0.4, 1, 0.2], [0, 0.2, 1]])
        graph = build_pmfg(correlations)
        self.assertEqual(graph.number_of_edges(), 3)
        self.assertEqual(graph[0][1]["weight"], -0.4)
        self.assertEqual(graph[0][2]["weight"], 0)
        for n_assets in (1, 2):
            small_graph = build_pmfg(np.eye(n_assets))
            self.assertEqual(small_graph.number_of_nodes(), n_assets)
            self.assertEqual(small_graph.number_of_edges(), n_assets - 1)

    def test_pmfg_ranks_signed_correlations_not_absolute_magnitudes(self):
        correlations = np.full((5, 5), 0.05)
        np.fill_diagonal(correlations, 1.0)
        correlations[0, 1] = correlations[1, 0] = -0.4
        graph = build_pmfg(correlations)
        # K5 minus one edge is planar, so the last (most negative) edge is omitted.
        self.assertEqual(graph.number_of_edges(), 9)
        self.assertFalse(graph.has_edge(0, 1))

    def test_invalid_correlation_shapes_and_values_raise(self):
        for matrix in (np.ones(3), np.ones((2, 3)), np.empty((0, 0)), np.array([[np.nan]])):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                build_pmfg(matrix)

    def test_signed_correlations_can_be_plotted(self):
        graph = build_pmfg(np.array([[1, -0.4, 0], [-0.4, 1, 0.2], [0, 0.2, 1]]))
        try:
            with patch("matplotlib.pyplot.show"):
                plot_pmfg(graph)
            self.assertTrue(plt.get_fignums())
        finally:
            plt.close("all")

    def test_edge_colors_match_fixed_colorbar_for_signed_and_uniform_weights(self):
        for weights in ((-0.4, 0.0, 0.2), (0.2, 0.2, 0.2)):
            with self.subTest(weights=weights):
                graph = nx.complete_graph(3)
                nx.set_edge_attributes(graph, dict(zip(graph.edges, weights)), "weight")
                figure = plot_pmfg(graph, show=False)
                try:
                    figure.canvas.draw()
                    edge_collection = next(
                        collection for collection in figure.axes[0].collections
                        if isinstance(collection, LineCollection)
                    )
                    assert_allclose(figure.axes[-1].get_ylim(), [-1.0, 1.0])
                    expected_colors = plt.cm.viridis((np.array(weights) + 1.0) / 2.0)
                    assert_allclose(edge_collection.get_colors()[:, :3], expected_colors[:, :3])
                finally:
                    plt.close(figure)

    def test_plot_can_save_without_showing_a_window(self):
        graph = build_pmfg(np.eye(3))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "plots" / "pmfg.png"
            with patch("matplotlib.pyplot.show") as show:
                figure = plot_pmfg(graph, output=output, show=False)
            try:
                show.assert_not_called()
                self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            finally:
                plt.close(figure)


class PortfolioTests(unittest.TestCase):
    def test_diagonal_covariance_has_inverse_variance_weights(self):
        weights = calculate_standard_weights(np.diag([1.0, 4.0, 9.0]))
        assert_allclose(weights, np.array([36, 9, 4]) / 49)
        self.assertAlmostEqual(weights.sum(), 1.0)

    def test_graph_mask_retains_nonpositive_edge_weights(self):
        covariance = np.array([[1.0, -0.5], [-0.5, 4.0]])
        # The two-asset GMVP has w0 = (variance1 - covariance) / total variance.
        expected_first_weight = (4.0 + 1e-4 + 0.5) / (1.0 + 4.0 + 2e-4 + 1.0)
        for edge_weight in (-0.5, 0.0):
            graph = nx.Graph()
            graph.add_edge(0, 1, weight=edge_weight)
            with self.subTest(edge_weight=edge_weight):
                weights = calculate_topological_weights(covariance, graph)
                assert_allclose(weights, [expected_first_weight, 1 - expected_first_weight])
        assert_allclose(covariance, [[1.0, -0.5], [-0.5, 4.0]])

    def test_masking_can_destroy_positive_definiteness_despite_ridge(self):
        covariance = np.full((3, 3), 0.9)
        np.fill_diagonal(covariance, 1.0)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0)
        with self.assertRaisesRegex(ValueError, "positive-definite"):
            calculate_topological_weights(covariance, nx.path_graph(3))

    def test_standard_weights_reject_indefinite_covariance(self):
        with self.assertRaisesRegex(ValueError, "positive-definite"):
            calculate_standard_weights(np.array([[1.0, 2.0], [2.0, 1.0]]))


class NetworkFilterTests(unittest.TestCase):
    def setUp(self):
        generator = np.random.default_rng(42)
        self.returns = generator.normal(size=(200, 8)) + generator.normal(size=(200, 1))

    def test_mst_fits_once_and_matches_reference_total_distance(self):
        network_filter = TopologicalFilter(self.returns)
        self.assertIsNone(network_filter.distance_matrix)
        with patch.object(
            network_filter, "apply_ledoit_wolf_shrinkage",
            wraps=network_filter.apply_ledoit_wolf_shrinkage,
        ) as fit:
            graph = network_filter.extract_mst_kruskal()
            network_filter.extract_tmfg()
            network_filter.get_masked_precision_matrix()
            self.assertEqual(fit.call_count, 1)
        self.assertTrue(nx.is_tree(graph))
        complete_graph = nx.from_numpy_array(network_filter.distance_matrix)
        expected = nx.minimum_spanning_tree(complete_graph)
        self.assertAlmostEqual(graph.size(weight="weight"), expected.size(weight="weight"))

    def test_face_insertion_graph_is_planar_chordal_and_maximal(self):
        graph = TopologicalFilter(self.returns).extract_tmfg()
        self.assertTrue(nx.check_planarity(graph)[0])
        self.assertTrue(nx.is_chordal(graph))
        self.assertEqual(graph.number_of_edges(), 3 * (8 - 2))

    def test_face_insertion_supports_fewer_than_four_assets(self):
        for n_assets in (1, 2, 3, 4):
            with self.subTest(n_assets=n_assets):
                graph = TopologicalFilter(self.returns[:, :n_assets]).extract_tmfg()
                self.assertEqual(graph.number_of_nodes(), n_assets)
                self.assertEqual(graph.number_of_edges(), n_assets * (n_assets - 1) // 2)

    def test_precision_mask_preserves_zero_distance_edges(self):
        network_filter = TopologicalFilter(self.returns)
        network_filter.apply_ledoit_wolf_shrinkage()
        precision = np.linalg.inv(network_filter.covariance_matrix)
        self.assertNotEqual(precision[0, 1], 0)
        graph = nx.empty_graph(network_filter.n_assets)
        graph.add_edge(0, 1, weight=0.0)
        with patch.object(network_filter, "extract_tmfg", return_value=graph):
            masked = network_filter.get_masked_precision_matrix()
        self.assertEqual(masked[0, 1], precision[0, 1])
        self.assertEqual(masked[1, 0], precision[1, 0])
        self.assertEqual(masked[0, 2], 0)
        assert_allclose(np.diag(masked), np.diag(precision))

    def test_invalid_returns_and_zero_variance_raise(self):
        for returns in (np.ones(3), np.ones((1, 3)), np.ones((2, 0)), np.full((2, 3), np.nan)):
            with self.subTest(returns=returns), self.assertRaises(ValueError):
                TopologicalFilter(returns)
        with self.assertRaisesRegex(ValueError, "positive variances"):
            TopologicalFilter(np.ones((5, 3))).extract_mst_kruskal()


if __name__ == "__main__":
    unittest.main()

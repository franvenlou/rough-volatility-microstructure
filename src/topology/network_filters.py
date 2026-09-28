"""Exploratory shrinkage, spanning-tree, and face-insertion network filters.

These utilities are separate from the greedy PMFG implementation. They have not
been validated as a covariance estimator or portfolio method on market data.
"""

from itertools import combinations

import networkx as nx
import numpy as np
import numpy.typing as npt
from sklearn.covariance import LedoitWolf

Tensor2D = npt.NDArray[np.float64]


class TopologicalFilter:
    """Build distance-weighted networks from observation-by-asset returns."""

    def __init__(self, raw_returns: Tensor2D) -> None:
        raw_returns = np.asarray(raw_returns, dtype=float)
        if raw_returns.ndim != 2 or raw_returns.shape[0] < 2 or raw_returns.shape[1] < 1:
            raise ValueError("Returns must have at least two observations and one asset.")
        if not np.all(np.isfinite(raw_returns)):
            raise ValueError("Returns must contain finite values.")
        self.raw_returns = raw_returns.copy()
        self.n_assets = raw_returns.shape[1]
        self.assets_indices = list(range(self.n_assets))
        # Unfitted state is explicit: never build a graph from uninitialized memory.
        self.covariance_matrix: Tensor2D | None = None
        self.correlation_matrix: Tensor2D | None = None
        self.distance_matrix: Tensor2D | None = None

    def apply_ledoit_wolf_shrinkage(self) -> None:
        """Estimate covariance, then convert correlations to chord distances.

        Distances are sqrt(2 * (1 - correlation)); clipping handles rounding at
        the correlation bounds. Zero variance prevents this normalization.
        """
        covariance = LedoitWolf().fit(self.raw_returns).covariance_
        variances = np.diag(covariance)
        if not np.all(np.isfinite(covariance)) or np.any(variances <= 0):
            raise ValueError("Shrinkage covariance must have finite, positive variances.")
        standard_deviations = np.sqrt(variances)
        correlations = covariance / np.outer(standard_deviations, standard_deviations)
        correlations = np.clip(correlations, -1.0, 1.0)
        self.covariance_matrix = covariance
        self.correlation_matrix = correlations
        self.distance_matrix = np.sqrt(2.0 * (1.0 - correlations))

    def _fitted_matrices(self) -> tuple[Tensor2D, Tensor2D]:
        if self.covariance_matrix is None or self.distance_matrix is None:
            self.apply_ledoit_wolf_shrinkage()
        assert self.covariance_matrix is not None and self.distance_matrix is not None
        return self.covariance_matrix, self.distance_matrix

    def extract_mst_kruskal(self) -> nx.Graph:
        """Build a minimum spanning tree of the complete distance graph."""
        _, distances = self._fitted_matrices()
        mst = nx.Graph()
        mst.add_nodes_from(self.assets_indices)
        edges = [
            (distances[i, j], i, j)
            for i, j in combinations(self.assets_indices, 2)
        ]
        edges.sort(key=lambda edge: edge[0])
        parent = {i: i for i in self.assets_indices}

        def find(node: int) -> int:
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        def union(left: int, right: int) -> bool:
            root_left, root_right = find(left), find(right)
            if root_left == root_right:
                return False
            parent[root_left] = root_right
            return True

        for weight, left, right in edges:
            if union(left, right):
                mst.add_edge(left, right, weight=weight)
                if mst.number_of_edges() == self.n_assets - 1:
                    break
        return mst

    def extract_tmfg(self) -> nx.Graph:
        """Build an exploratory TMFG-style graph by triangular-face insertion.

        The first four columns seed K4; the seed is not optimized and the result
        depends on asset order. Each subsequent insertion minimizes the sum of
        distances to a face. This restricted construction preserves planarity and
        chordality but is not a full implementation of TMFG move/seed selection.
        For fewer than four assets, return the complete graph.
        """
        _, distances = self._fitted_matrices()
        tmfg = nx.Graph()
        tmfg.add_nodes_from(self.assets_indices)
        initial_clique = self.assets_indices[:4]
        for left, right in combinations(initial_clique, 2):
            tmfg.add_edge(left, right, weight=distances[left, right])
        if self.n_assets <= 4:
            return tmfg

        faces = list(combinations(initial_clique, 3))
        uninserted_nodes = set(self.assets_indices[4:])
        while uninserted_nodes:
            best_node = -1
            best_face_idx = -1
            minimum_distance_sum = float("inf")
            for node in sorted(uninserted_nodes):
                for face_idx, face in enumerate(faces):
                    distance_sum = sum(distances[node, vertex] for vertex in face)
                    if distance_sum < minimum_distance_sum:
                        minimum_distance_sum = distance_sum
                        best_node = node
                        best_face_idx = face_idx

            target_face = faces.pop(best_face_idx)
            uninserted_nodes.remove(best_node)
            for vertex in target_face:
                tmfg.add_edge(best_node, vertex, weight=distances[best_node, vertex])
            faces.extend([
                (best_node, target_face[0], target_face[1]),
                (best_node, target_face[1], target_face[2]),
                (best_node, target_face[0], target_face[2]),
            ])
        return tmfg

    def get_masked_precision_matrix(self) -> Tensor2D:
        """Zero inverse-covariance entries outside the face-insertion graph.

        This exploratory Hadamard mask is not a fitted graphical model and need
        not remain positive definite. Do not interpret it as a valid precision
        matrix or as conditional independence without additional validation.
        Singular covariance raises numpy.linalg.LinAlgError during inversion.
        """
        covariance, _ = self._fitted_matrices()
        graph = self.extract_tmfg()
        # A zero distance still represents an edge, so ignore numerical weights.
        mask = nx.to_numpy_array(graph, nodelist=self.assets_indices, weight=None, dtype=bool)
        np.fill_diagonal(mask, True)
        return np.linalg.inv(covariance) * mask

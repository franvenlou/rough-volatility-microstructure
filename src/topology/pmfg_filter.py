"""Greedy planar filtering of a correlation matrix."""

import networkx as nx
import numpy as np


def build_pmfg(correlation_matrix: np.ndarray) -> nx.Graph:
    """Keep edges in descending signed-correlation order while preserving planarity.

    This is a Planar Maximally Filtered Graph (PMFG): the greedy procedure
    produces a maximal planar graph, not a global optimum over planar graphs.
    Planarity alone does not establish that retained correlations are signal.
    """
    correlation_matrix = np.asarray(correlation_matrix, dtype=float)
    if (
        correlation_matrix.ndim != 2
        or correlation_matrix.shape[0] != correlation_matrix.shape[1]
        or correlation_matrix.shape[0] == 0
    ):
        raise ValueError("The correlation matrix must be nonempty and square.")
    if not np.all(np.isfinite(correlation_matrix)):
        raise ValueError("The correlation matrix must contain finite values.")
    if not np.allclose(correlation_matrix, correlation_matrix.T):
        raise ValueError("The correlation matrix must be symmetric.")

    n_nodes = correlation_matrix.shape[0]
    i_idx, j_idx = np.triu_indices(n_nodes, k=1)
    correlations = correlation_matrix[i_idx, j_idx]
    sorted_indices = np.argsort(correlations)[::-1]

    pmfg = nx.Graph()
    pmfg.add_nodes_from(range(n_nodes))
    max_edges = 3 * (n_nodes - 2) if n_nodes >= 3 else n_nodes * (n_nodes - 1) // 2

    for idx in sorted_indices:
        u, v = i_idx[idx], j_idx[idx]
        pmfg.add_edge(u, v, weight=correlations[idx])
        is_planar, _ = nx.check_planarity(pmfg)
        if not is_planar:
            pmfg.remove_edge(u, v)
        elif pmfg.number_of_edges() == max_edges:
            break

    return pmfg


if __name__ == "__main__":
    print("Building a PMFG from synthetic returns...")
    np.random.seed(42)
    random_returns = np.random.randn(10, 1000)
    empirical_corr = np.corrcoef(random_returns)
    pmfg_graph = build_pmfg(empirical_corr)

    print(f"Assets: {pmfg_graph.number_of_nodes()}")
    print(f"Retained edges: {pmfg_graph.number_of_edges()}")
    print(f"Planar edge bound for this universe: {3 * (10 - 2)}")
    print("This synthetic example checks graph structure, not statistical noise removal.")

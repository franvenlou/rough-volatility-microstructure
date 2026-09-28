"""Unconstrained minimum-variance weights with optional PMFG covariance masking."""

import networkx as nx
import numpy as np

from src.topology.pmfg_filter import build_pmfg


def _validate_covariance(cov_matrix: np.ndarray) -> np.ndarray:
    covariance = np.asarray(cov_matrix, dtype=float)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or covariance.shape[0] == 0
    ):
        raise ValueError("The covariance matrix must be nonempty and square.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("The covariance matrix must contain finite values.")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("The covariance matrix must be symmetric.")
    return covariance


def _minimum_variance_weights(covariance: np.ndarray) -> np.ndarray:
    # Only interpret the covariance-based objective as GMVP when it is definite.
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Minimum-variance weights require a positive-definite covariance matrix; "
            "topological masking and a fixed ridge do not guarantee this."
        ) from exc
    weights = np.linalg.solve(covariance, np.ones(covariance.shape[0]))
    return weights / weights.sum()


def calculate_topological_weights(cov_matrix: np.ndarray, pmfg: nx.Graph) -> np.ndarray:
    """Mask covariance by graph edges, add a fixed ridge, and compute GMVP weights.

    The diagonal is retained and short positions are allowed. The historical
    ridge of 1e-4 is an untuned default in covariance units; masking can still
    produce an indefinite matrix, in which case this function raises ValueError.
    This experiment does not establish portfolio robustness or out-of-sample gains.
    """
    covariance = _validate_covariance(cov_matrix)
    n_assets = covariance.shape[0]
    if pmfg.is_directed() or set(pmfg.nodes) != set(range(n_assets)):
        raise ValueError("The graph must be undirected with nodes 0 through n_assets - 1.")

    # Edge presence, rather than correlation sign, defines the topological mask.
    mask = nx.to_numpy_array(pmfg, nodelist=range(n_assets), weight=None, dtype=bool)
    np.fill_diagonal(mask, True)
    filtered_covariance = covariance * mask
    filtered_covariance += 1e-4 * np.eye(n_assets)
    return _minimum_variance_weights(filtered_covariance)


def calculate_standard_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """Compute fully invested GMVP weights with short positions allowed."""
    return _minimum_variance_weights(_validate_covariance(cov_matrix))


if __name__ == "__main__":
    np.random.seed(42)
    n_assets = 10
    print("Simulating returns with a common factor...")
    returns = np.random.randn(n_assets, 1000)
    returns += np.random.randn(1000) * 0.5
    empirical_covariance = np.cov(returns)
    pmfg = build_pmfg(np.corrcoef(returns))

    topological_weights = calculate_topological_weights(empirical_covariance, pmfg)
    standard_weights = calculate_standard_weights(empirical_covariance)
    print(f"{'Asset':<10} | {'Sample covariance':<20} | {'PMFG mask + ridge':<20}")
    for i in range(n_assets):
        print(f"{i:<10} | {standard_weights[i] * 100:>18.2f}% | {topological_weights[i] * 100:>18.2f}%")

    print(f"\nL2 weight norm (sample covariance): {np.linalg.norm(standard_weights):.4f}")
    print(f"L2 weight norm (PMFG mask + ridge): {np.linalg.norm(topological_weights):.4f}")
    print("Weight norms describe concentration; this example does not test stability or future risk.")

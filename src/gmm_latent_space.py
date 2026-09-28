"""Cluster rolling snapshot features with an exploratory Gaussian mixture model.

The features are a Hurst scaling estimate of log absolute returns, Shannon
entropy of total snapshot volumes across time (nats), and mean absolute return
(basis points). They are not validated trading signals or tests of a price model.
"""

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from src.ingestion.snapshots import load_snapshots, snapshot_features
from src.microstruct.core_math import calculate_hurst_variogram, shannon_entropy


def rolling_latent_features(
    absolute_returns: np.ndarray,
    volumes: np.ndarray,
    window_size: int = 50,
    max_lag: int = 5,
) -> np.ndarray:
    """Return valid rolling feature rows, including the last complete window.

    Returns and volumes must already be aligned, with strictly positive absolute
    returns, as produced by ``snapshot_features``. Windows whose Hurst scaling
    estimate is undefined are omitted and their count is reported in a warning.
    The defaults of 50 observations and five lags are exploratory, not calibrated.
    Features retain their original units; no standardization is applied.
    """
    returns = np.asarray(absolute_returns, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if returns.ndim != 1 or volumes.ndim != 1 or returns.shape != volumes.shape:
        raise ValueError("Returns and volumes must be aligned one-dimensional arrays.")
    if not np.all(np.isfinite(returns)) or np.any(returns <= 0):
        raise ValueError("Absolute returns must be finite and strictly positive.")
    if not np.all(np.isfinite(volumes)) or np.any(volumes < 0):
        raise ValueError("Volumes must be finite and nonnegative.")
    if (
        not isinstance(max_lag, (int, np.integer))
        or isinstance(max_lag, bool)
        or max_lag < 2
    ):
        raise ValueError("max_lag must be an integer of at least two.")
    if (
        not isinstance(window_size, (int, np.integer))
        or isinstance(window_size, bool)
        or window_size <= max_lag
    ):
        raise ValueError("window_size must be an integer greater than max_lag.")
    if len(returns) < window_size:
        raise ValueError("There are fewer returns than the requested window size.")

    rows = []
    omitted = 0
    for start in range(len(returns) - window_size + 1):
        window_returns = returns[start : start + window_size]
        window_volumes = volumes[start : start + window_size]
        try:
            hurst = calculate_hurst_variogram(np.log(window_returns), max_lag)
        except ValueError:
            omitted += 1
            continue
        if not np.isfinite(hurst):
            omitted += 1
            continue
        rows.append(
            (hurst, shannon_entropy(window_volumes), np.mean(window_returns) * 1e4)
        )

    if omitted:
        warnings.warn(
            f"Omitted {omitted} rolling windows with undefined Hurst estimates.",
            UserWarning,
            stacklevel=2,
        )
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def generate_latent_space(
    parquet_path: str | Path, window_size: int = 50, max_lag: int = 5
) -> np.ndarray:
    """Read validated top-of-book snapshots and calculate rolling features."""
    returns, volumes = snapshot_features(load_snapshots(parquet_path))
    return rolling_latent_features(returns, volumes[1:], window_size, max_lag)


def plot_gmm_3d(
    X: np.ndarray,
    n_components: int = 2,
    *,
    output: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, GaussianMixture]:
    """Fit and plot raw feature coordinates; cluster IDs have no trading meaning."""
    features = np.asarray(X, dtype=float)
    if (
        features.ndim != 2
        or features.shape[1] != 3
        or not np.all(np.isfinite(features))
    ):
        raise ValueError("X must be a finite matrix with three feature columns.")
    if (
        not isinstance(n_components, (int, np.integer))
        or isinstance(n_components, bool)
        or n_components < 1
    ):
        raise ValueError("n_components must be a positive integer.")
    if len(features) < max(2, n_components):
        raise ValueError("GMM fitting needs at least two rows and one row per component.")

    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=42
    )
    labels = gmm.fit_predict(features)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        features[:, 0], features[:, 1], features[:, 2],
        c=labels, cmap="viridis", marker="o", alpha=0.7,
    )
    ax.set_xlabel("Hurst scaling estimate (H)")
    ax.set_ylabel("Temporal volume entropy (nats)")
    ax.set_zlabel("Mean absolute return (bps)")
    ax.set_title("Exploratory GMM of rolling snapshot features")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, ticks=np.arange(n_components))
    cbar.set_label("Cluster ID")
    fig.tight_layout()
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    return fig, gmm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/lob_snapshot.parquet"))
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    args = parser.parse_args()
    features = generate_latent_space(args.input, args.window_size)
    fig, gmm = plot_gmm_3d(
        features, args.components, output=args.output, show=not args.no_show
    )
    print(f"Fitted {gmm.n_components} GMM components to {len(features)} rolling windows.")
    print(f"Optimizer converged: {gmm.converged_}. Cluster IDs are exploratory diagnostics.")
    if args.output is not None:
        print(f"Saved plot to {args.output}.")
    plt.close(fig)


if __name__ == "__main__":
    main()

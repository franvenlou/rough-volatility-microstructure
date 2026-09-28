"""Exploratory BIC selection for supplied Hurst, entropy, and ESG feature series.

The snapshot pipeline does not provide an ESG feature. Callers must supply and
justify that series independently. Selecting a centroid near H=0.5 under an
entropy threshold does not identify a Black-Scholes model or establish tradability.
The standalone demo uses unconstrained Gaussian feature coordinates, so its
entropy coordinate can be negative and is not a computed Shannon entropy.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class RegimeCentroid:
    """Feature means of a component selected by an explicit heuristic."""

    cluster_id: int
    hurst_mean: float
    entropy_mean: float
    esg_mean: float


class LatentRegimeIsolator:
    """Fit GMM candidates and inspect their centroids without model-validity claims."""

    def __init__(
        self,
        hurst_ts: np.ndarray,
        entropy_ts: np.ndarray,
        esg_ts: np.ndarray,
        initial_precision: np.ndarray,
    ) -> None:
        """Accept aligned features and a positive-definite 3x3 initial precision.

        Precision must describe these three feature coordinates, rather than an
        asset-return network. It only initializes fitting: EM does not preserve
        its sparsity or impose a PMFG constraint. Features are not standardized.
        """
        series = [
            np.asarray(values, dtype=float) for values in (hurst_ts, entropy_ts, esg_ts)
        ]
        if any(values.ndim != 1 for values in series):
            raise ValueError("All feature series must be one-dimensional.")
        if len({len(values) for values in series}) != 1 or len(series[0]) < 2:
            raise ValueError("Feature series must have matching lengths of at least two.")
        self.Z = np.column_stack(series)
        if not np.all(np.isfinite(self.Z)):
            raise ValueError("All feature values must be finite.")
        self.n_samples = len(self.Z)
        precision = np.asarray(initial_precision, dtype=float)
        if precision.shape != (3, 3) or not np.all(np.isfinite(precision)):
            raise ValueError("Initial precision must be a finite 3x3 matrix.")
        if not np.allclose(precision, precision.T):
            raise ValueError("Initial precision must be symmetric.")
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Initial precision must be positive definite.") from exc
        self.base_precision = precision.copy()
        self.optimal_gmm: GaussianMixture | None = None
        self.optimal_k = 0
        self.bic_scores_: dict[int, float] = {}

    def fit_optimal_gmm(self, k_range: tuple[int, int] = (2, 7)) -> None:
        """Choose the lowest-BIC candidate over an inclusive component-count range.

        BIC compares the fitted candidates; it does not validate the feature
        definitions or the independent-observation assumption for rolling data.
        Scikit-learn convergence warnings remain visible.
        """
        if (
            len(k_range) != 2
            or any(
                not isinstance(k, (int, np.integer)) or isinstance(k, bool)
                for k in k_range
            )
            or not 1 <= k_range[0] <= k_range[1] <= self.n_samples
        ):
            raise ValueError("k_range must satisfy 1 <= minimum <= maximum <= sample count.")
        lowest_bic = np.inf
        best_gmm = None
        scores = {}
        for k in range(k_range[0], k_range[1] + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                precisions_init=np.repeat(self.base_precision[None, :, :], k, axis=0),
                random_state=42,
                max_iter=500,
            )
            gmm.fit(self.Z)
            current_bic = gmm.bic(self.Z)
            scores[k] = current_bic
            if np.isfinite(current_bic) and current_bic < lowest_bic:
                lowest_bic = current_bic
                best_gmm = gmm
        if best_gmm is None:
            raise RuntimeError("No candidate produced a finite BIC.")
        self.optimal_gmm = best_gmm
        self.optimal_k = best_gmm.n_components
        self.bic_scores_ = scores

    def select_centroid(
        self, entropy_upper_bound: float = 0.4, target_hurst: float = 0.5
    ) -> RegimeCentroid:
        """Select the nearest H centroid whose entropy is below the supplied bound.

        The defaults reproduce the original exploratory rule. Neither the entropy
        cutoff nor target has been calibrated, and the result is not a trading
        recommendation or a test of diffusion or semimartingale assumptions.
        Entropy is expected in nats when computed by the canonical estimator.
        """
        if self.optimal_gmm is None:
            raise ValueError("Fit a GMM with fit_optimal_gmm() before selecting a centroid.")
        if not np.isfinite(entropy_upper_bound) or not np.isfinite(target_hurst):
            raise ValueError("The entropy bound and Hurst target must be finite.")
        means = self.optimal_gmm.means_
        candidates = np.flatnonzero(means[:, 1] < entropy_upper_bound)
        if len(candidates) == 0:
            raise RuntimeError("No component satisfies the supplied entropy threshold.")
        selected = int(
            candidates[np.argmin(np.abs(means[candidates, 0] - target_hurst))]
        )
        return RegimeCentroid(
            cluster_id=selected,
            hurst_mean=float(means[selected, 0]),
            entropy_mean=float(means[selected, 1]),
            esg_mean=float(means[selected, 2]),
        )


def draw_covariance_ellipse(
    gmm: GaussianMixture, cluster_id: int, ax: plt.Axes, **kwargs
) -> None:
    """Draw the radius-two covariance contour of the Hurst/entropy marginal.

    This is a component dispersion contour, not a confidence interval for the
    estimated mean. Radius two in two dimensions is not a 95% probability contour.
    """
    covariance = gmm.covariances_[cluster_id][:2, :2]
    mean = gmm.means_[cluster_id][:2]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 4 * np.sqrt(eigenvalues)
    ax.add_patch(Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs))


def plot_regime_projection(
    isolator: LatentRegimeIsolator,
    selected_centroid: RegimeCentroid | None = None,
    *,
    output: str | Path | None = None,
    show: bool = True,
    title: str = "Exploratory GMM projection: supplied Hurst, entropy, and ESG features",
) -> plt.Figure:
    """Project fitted components into Hurst/entropy coordinates for inspection."""
    gmm = isolator.optimal_gmm
    if gmm is None:
        raise ValueError("Fit the GMM before plotting.")
    labels = gmm.predict(isolator.Z)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, isolator.optimal_k))
    ax.scatter(
        isolator.Z[:, 0], isolator.Z[:, 1], c=colors[labels],
        s=15, alpha=0.6, edgecolors="none",
    )
    for k in range(isolator.optimal_k):
        selected = selected_centroid is not None and k == selected_centroid.cluster_id
        draw_covariance_ellipse(
            gmm, k, ax, facecolor="none", edgecolor="black" if selected else colors[k],
            lw=2 if selected else 1, ls="--" if selected else "-",
        )
        ax.scatter(
            *gmm.means_[k][:2], color="black" if selected else colors[k],
            marker="*" if selected else "X", s=180 if selected else 100, zorder=5,
            label=f"Component {k}" + (" (selected by heuristic)" if selected else ""),
        )
    ax.set_title(title)
    ax.set_xlabel("Hurst scaling estimate (H)")
    ax.set_ylabel("Supplied entropy feature (nats)")
    ax.legend(loc="best")
    fig.tight_layout()
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a synthetic ESG-feature GMM demonstration."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/synthetic_regime_gmm.png")
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    n_samples = 5000
    # These Gaussian feature samples are not derived from LOB or ESG observations.
    hurst = rng.normal(0.3, 0.15, n_samples)
    entropy = rng.normal(0.6, 0.2, n_samples)
    esg = rng.normal(50, 15, n_samples)
    injected = rng.choice(n_samples, 800, replace=False)
    hurst[injected] = rng.normal(0.49, 0.03, 800)
    entropy[injected] = rng.normal(0.2, 0.05, 800)
    esg[injected] = rng.normal(85, 5, 800)
    isolator = LatentRegimeIsolator(hurst, entropy, esg, np.eye(3))
    isolator.fit_optimal_gmm(k_range=(2, 6))
    selected = isolator.select_centroid()
    fig = plot_regime_projection(
        isolator, selected, output=args.output, show=not args.no_show,
        title="Synthetic feature clusters (not market or ESG observations)",
    )
    print("Synthetic demonstration with a manually injected feature cluster; no empirical validation.")
    print("Gaussian entropy coordinates can be negative and are not measured Shannon entropies.")
    print(
        f"Lowest-BIC candidate: K={isolator.optimal_k}; "
        f"optimizer converged: {isolator.optimal_gmm.converged_}."
    )
    print(
        f"Heuristic-selected component {selected.cluster_id}: "
        f"H={selected.hurst_mean:.3f}, entropy={selected.entropy_mean:.3f} nats."
    )
    print(f"Saved plot to {args.output}.")
    plt.close(fig)


if __name__ == "__main__":
    main()

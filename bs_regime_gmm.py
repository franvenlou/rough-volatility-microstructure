import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from dataclasses import dataclass
from typing import Tuple, List, Optional

# Type aliases for strict static analysis
Tensor2D = npt.NDArray[np.float64]
Tensor1D = npt.NDArray[np.float64]

@dataclass
class RegimeCentroid:
    """Structure to hold the evaluated parameters of a learned market regime."""
    cluster_id: int
    hurst_mean: float
    entropy_mean: float
    esg_mean: float
    is_tradable_bs: bool

class LatentRegimeIsolator:
    """
    Abstract controller for unsupervised GMM clustering over a 3D topological latent space.
    Isolates the Black-Scholes continuous semimartingale regime from rough microstructural chaos.
    """
    
    def __init__(self, hurst_ts: Tensor1D, entropy_ts: Tensor1D, esg_ts: Tensor1D, pmfg_precision: Tensor2D) -> None:
        """
        Concatenates the time-series flows and initializes the baseline precision topology.
        """
        self.n_samples = len(hurst_ts)
        # Design matrix Z combining Hurst, LOB Entropy, and ESG Flow
        self.Z: Tensor2D = np.column_stack([hurst_ts, entropy_ts, esg_ts])
        self.base_precision: Tensor2D = pmfg_precision
        
        self.optimal_gmm: Optional[GaussianMixture] = None
        self.optimal_k: int = 0
        
    def fit_optimal_gmm(self, k_range: Tuple[int, int] = (2, 7)) -> None:
        """
        Dynamically trains the GMM across K clusters, structurally penalizing model 
        overfitting by minimizing the Bayesian Information Criterion (BIC).
        Initializes the precision matrices using the static PMFG topological mask.
        """
        lowest_bic = np.inf
        best_gmm = None
        best_k = 0
        
        for k in range(k_range[0], k_range[1] + 1):
            # Tile the PMFG precision matrix for K components to seed the topological prior
            precisions_init = np.array([self.base_precision] * k)
            
            gmm = GaussianMixture(
                n_components=k, 
                covariance_type='full', 
                precisions_init=precisions_init,
                random_state=42,
                max_iter=500
            )
            gmm.fit(self.Z)
            
            # Evaluate BIC
            current_bic = gmm.bic(self.Z)
            if current_bic < lowest_bic:
                lowest_bic = current_bic
                best_gmm = gmm
                best_k = k
                
        self.optimal_gmm = best_gmm
        self.optimal_k = best_k
        print(f"Convergence achieved at K={self.optimal_k} clusters (BIC: {lowest_bic:.2f}).")

    def isolate_diffusive_regime(self, entropy_upper_bound: float = 0.4) -> RegimeCentroid:
        """
        Analytically interrogates the learned centroids (mean vector \mu) to identify the 
        Black-Scholes (BS) regime. The BS regime is mathematically defined by a Hurst 
        exponent H approaching 0.5 (Standard Brownian Motion)[cite: 2, 4, 11] and constrained 
        by a probabilistic upper threshold of LOB entropy[cite: 2, 8].
        """
        if self.optimal_gmm is None:
            raise ValueError("GMM has not been fitted. Call fit_optimal_gmm() first.")
            
        means = self.optimal_gmm.means_
        best_cluster = -1
        min_hurst_distance = np.inf
        target_hurst = 0.5
        
        for cluster_id in range(self.optimal_k):
            mu_hurst, mu_entropy, mu_esg = means[cluster_id]
            hurst_dist = abs(mu_hurst - target_hurst)
            
            # Condition: Minimal distance to H=0.5 AND Entropy strictly below toxic threshold
            if hurst_dist < min_hurst_distance and mu_entropy < entropy_upper_bound:
                min_hurst_distance = hurst_dist
                best_cluster = cluster_id
                
        if best_cluster == -1:
            raise RuntimeError("System Failure: No component satisfies the Markovian boundary conditions.")
            
        bs_centroid = RegimeCentroid(
            cluster_id=best_cluster,
            hurst_mean=means[best_cluster][0],
            entropy_mean=means[best_cluster][1],
            esg_mean=means[best_cluster][2],
            is_tradable_bs=True
        )
        return bs_centroid

def draw_confidence_ellipsoid(gmm: GaussianMixture, cluster_id: int, ax: plt.Axes, **kwargs) -> None:
    """Calculates and draws the 2D Gaussian confidence ellipsoid for a given GMM component."""
    # Slice the 3D covariance matrix for the 2D projection (Hurst [0] and Entropy [1])
    cov_2d = gmm.covariances_[cluster_id][:2, :2]
    mean_2d = gmm.means_[cluster_id][:2]
    
    # Calculate eigenvalues and eigenvectors for ellipsoid geometry
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    
    # 2 standard deviations (~95% confidence interval)
    width, height = 2 * np.sqrt(eigenvalues) * 2 
    ellip = Ellipse(xy=mean_2d, width=width, height=height, angle=angle, **kwargs)
    
    ax.add_patch(ellip)

def generate_github_portfolio_visualization(isolator: LatentRegimeIsolator, bs_centroid: RegimeCentroid) -> None:
    """
    Renders the definitive 2D phase-space visualization. Projects the Gaussian ellipsoids 
    to dramatically delineate the rough microstructural chaos vs. the Black-Scholes islet.
    """
    gmm = isolator.optimal_gmm
    labels = gmm.predict(isolator.Z)
    
    # Extract responsibilities - conditional expectations measuring cluster membership strength
    responsibilities = gmm.predict_proba(isolator.Z)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot background scatter
    scatter = ax.scatter(
        isolator.Z[:, 0], isolator.Z[:, 1], 
        c=labels, cmap='viridis', s=15, alpha=0.6, edgecolors='none'
    )
    
    # Plot geometric confidence ellipsoids
    colors = sns.color_palette('viridis', isolator.optimal_k)
    for k in range(isolator.optimal_k):
        if k == bs_centroid.cluster_id:
            draw_confidence_ellipsoid(gmm, k, ax, facecolor='none', edgecolor='cyan', lw=3, ls='--')
            ax.scatter(*gmm.means_[k][:2], color='cyan', marker='*', s=300, zorder=5, label='Black-Scholes Islet (H ≈ 0.5)')
        else:
            draw_confidence_ellipsoid(gmm, k, ax, facecolor='none', edgecolor=colors[k], lw=1.5, alpha=0.5)
            ax.scatter(*gmm.means_[k][:2], color=colors[k], marker='X', s=150, zorder=5)

    # Topological boundary aesthetics
    ax.axvline(x=0.5, color='white', linestyle=':', alpha=0.5, label='Theoretical Semimartingale (H=0.5)')
    ax.axhline(y=0.4, color='red', linestyle=':', alpha=0.5, label='Toxic Entropy Boundary')
    
    ax.set_title("Topological Phase Space: Rough Volatility Chaos vs. Black-Scholes Regime", fontsize=16, pad=20)
    ax.set_xlabel("Empirical Hurst Exponent (H)", fontsize=12)
    ax.set_ylabel("Limit Order Book Shannon Entropy (Bits)", fontsize=12)
    ax.legend(loc='upper left', frameon=True, facecolor='black', edgecolor='white')
    
    plt.tight_layout()
    plt.savefig("topological_regime_isolation.png", dpi=300)
    print("GitHub asset 'topological_regime_isolation.png' generated successfully.")

if __name__ == "__main__":
    # Simulated pipeline ingestion
    np.random.seed(42)
    n_samples = 5000
    
    # Mocking the 3D Topological Flow (Hurst, Entropy, ESG)
    mock_hurst = np.random.normal(0.3, 0.15, n_samples)
    mock_entropy = np.random.normal(0.6, 0.2, n_samples)
    mock_esg = np.random.normal(50, 15, n_samples)
    
    # Injecting the artificial Black-Scholes Subspace (H ~ 0.5, Low Entropy, High ESG)
    bs_indices = np.random.choice(n_samples, 800, replace=False)
    mock_hurst[bs_indices] = np.random.normal(0.49, 0.03, 800)
    mock_entropy[bs_indices] = np.random.normal(0.2, 0.05, 800)
    mock_esg[bs_indices] = np.random.normal(85, 5, 800)
    
    # Mock PMFG Precision Matrix (3x3 Identity for demonstration)
    mock_precision = np.eye(3)
    
    # Execution
    engine = LatentRegimeIsolator(mock_hurst, mock_entropy, mock_esg, mock_precision)
    engine.fit_optimal_gmm(k_range=(2, 6))
    
    try:
        bs_regime = engine.isolate_diffusive_regime(entropy_upper_bound=0.4)
        print(f"Black-Scholes Regime Isolated at Cluster {bs_regime.cluster_id}")
        print(f"Centroid -> Hurst: {bs_regime.hurst_mean:.3f}, Entropy: {bs_regime.entropy_mean:.3f}")
        
        generate_github_portfolio_visualization(engine, bs_regime)
    except RuntimeError as e:
        print(e)
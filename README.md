# Non-Parametric Specification Tests for Rough Volatility Dynamics & Topological Portfolio Optimization

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Quant Architecture](https://img.shields.io/badge/architecture-Production--Grade-orange)

## Abstract
This repository houses an advanced quantitative finance pipeline designed to empirically challenge the classical diffusive assumptions of the Black-Scholes framework. By leveraging high-frequency Limit Order Book (LOB) microstructural data, the architecture isolates latent market regimes using Rough Volatility dynamics, Shannon Entropy, and Gaussian Mixture Models (GMM). 

Furthermore, it circumvents the inherent instability of Markowitz portfolio optimization—often referred to as the "Curse of Dimensionality"—by applying rigorous network topology via the Planar Maximally Filtered Graph (PMFG) to construct a robust precision matrix.

---

## Architectural Breakdown: What It Does & How It Works

The codebase is strictly modularized into three core domains: Asynchronous Ingestion, Stochastic Inference, and Topological Network Optimization.

### 1. High-Frequency Asynchronous Ingestion
**File:** `src/ingestion/lob_streamer.py`

*   **What it does:** Captures real-time, highly granular microstructural order flow data from digital asset exchanges.
*   **How it works:** It establishes an asynchronous WebSocket connection utilizing Python's `asyncio` and `websockets` libraries to stream Level-2 Depth snapshots at millisecond resolution. To optimize memory overhead and subsequent backtesting I/O speeds, it instantly serializes these multidimensional arrays into the highly compressed, columnar Apache Parquet format via `pyarrow`.

### 2. The Stochastic Core Engine
**File:** `src/microstruct/core_math.py`

*   **What it does:** Performs the heavy mathematical lifting required to extract non-parametric volatility proxies and liquidity imbalances while bypassing Python's Global Interpreter Lock (GIL).
*   **How it works:** Every function is strictly typed and compiled Just-In-Time (JIT) using the `Numba` `@njit(nogil=True, fastmath=True)` decorator. 
    *   **Nadaraya-Watson Kernel Regression:** Smooths the empirical volatility surface using a Gaussian kernel to eradicate spurious bid-ask bounce noise.
    *   **Hurst Exponent ($H$):** Computes log-linear variograms of the smoothed variance to detect fractional Brownian motion (roughness / anti-persistence).
    *   **Shannon Entropy ($S$):** Evaluates the probabilistic distribution of volume across LOB tiers to quantify directional toxicity.

### 3. Latent Space Mapping & Unsupervised Clustering
**File:** `src/gmm_latent_space.py` & `notebooks/1_Latent_Space_GMM.ipynb`

*   **What it does:** Segregates chaotic, toxic market regimes from tradable, diffusive environments without human supervisory bias.
*   **How it works:** It implements a rolling temporal window that projects the stochastic triad $(H_t, S_t, \log(\sigma_t))$ into a 3D topological latent space. It then calibrates a Gaussian Mixture Model (GMM) via the Expectation-Maximization algorithm. By maximizing the log-likelihood function, it isolates the exact multidimensional clusters where the market exhibits extreme rough volatility versus classical Brownian behavior.

### 4. Topological Network Filtering (PMFG)
**Files:** `src/topology/pmfg_filter.py` & `src/topology/visualize_pmfg.py`

*   **What it does:** Eradicates spurious empirical correlations from the covariance matrix, preserving only the most structurally significant systemic linkages and sectorial cliques.
*   **How it works:** It sorts the empirical correlation coefficients in descending order and iteratively attempts to add edges to an empty graph. It applies Kuratowski's theorem programmatically: an edge is only retained if its insertion does not violate genus-$0$ planarity (no intersecting lines). The algorithm converges exactly at Euler's theoretical limit $E = 3(V-2)$, yielding a sparse, information-rich topological network. Visual rendering is achieved via Kamada-Kawai energy minimization layouts.

### 5. Robust Portfolio Optimization
**File:** `src/topology/markowitz_topologico.py`

*   **What it does:** Computes the Global Minimum Variance Portfolio (GMVP) weights, solving the historical instability of classical Markowitz matrices.
*   **How it works:** It extracts the adjacency matrix of the calculated PMFG and applies it as a structural Hadamard mask over the empirical covariance matrix $\Sigma$. After injecting infinitesimal Tikhonov (Ridge) regularization to guarantee positive definiteness, it inverts the masked matrix to derive a highly stable Precision Matrix $\Theta$. The resulting weights mathematically preclude the extreme leverage and aggressive short positions historically triggered by sample noise.

$$ w^* = \frac{\Theta \mathbf{1}}{\mathbf{1}^T \Theta \mathbf{1}} $$

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/rough-volatility-microstructure.git](https://github.com/your-username/rough-volatility-microstructure.git)
   cd rough-volatility-microstructure

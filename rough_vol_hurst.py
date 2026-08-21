import numpy as np
from numba import njit

@njit(cache=True, fastmath=True, nogil=True)
def compute_hurst_variogram(volatility_series: np.ndarray, max_lags: int) -> float:
    """
    Estimates the Hurst exponent (H) of a stochastic volatility time series via 
    a robust variogram (2nd-order absolute moments) on a continuous fill asymptotic grid.
    
    This function evaluates the quadratic variation across multiple scale lags (q) 
    and applies an algebraic Ordinary Least Squares (OLS) regression in the 
    dual-logarithmic domain: \log(q) vs \log(Variation_q).
    
    Theoretical Context:
    Under the rough volatility paradigm, log-volatility behaves as a fractional 
    Brownian motion (fBm) rather than a standard diffusion[cite: 11]. 
    - If H = 0.5: The process is a standard Brownian motion (continuous semimartingale).
    - If H \approx 0.1: This denotes the massive presence of rough clusters and 
      fractional memory in the volatility surface. It fundamentally violates the 
      semimartingale assumption, exhibiting highly anti-persistent micro-increments 
      and infinite quadratic variation[cite: 7, 14].
      
    Args:
        volatility_series (np.ndarray): 1D continuous tensor of localized volatility states.
        max_lags (int): The maximum lag q to compute the quadratic scaling.
        
    Returns:
        float: The empirical Hurst exponent (H).
    """
    n_obs = volatility_series.shape[0]
    
    # Pre-allocate scalar accumulators for OLS regression to strictly avoid 
    # dynamic object instantiation, preserving L1/L2 cache harmony and nogil constraints.
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    
    # Compute the 2nd-order moment for each lag q
    for q in range(1, max_lags + 1):
        var_q = 0.0
        valid_pairs = n_obs - q
        
        # Hardware-aligned unrolled loop over the 1D tensor
        for i in range(valid_pairs):
            diff = volatility_series[i + q] - volatility_series[i]
            var_q += diff * diff
            
        # Normalize the variance by the number of valid overlapping pairs
        var_q /= valid_pairs
        
        # Dual-logarithmic mapping: \log(q) -> \log(Variation_q)
        log_q = np.log(float(q))
        log_var_q = np.log(var_q)
        
        # Iterative OLS accumulation
        sum_x += log_q
        sum_y += log_var_q
        sum_xx += log_q * log_q
        sum_xy += log_q * log_var_q
        
    # Algebraic OLS Slope computation (Slope = 2H)
    n_lags = float(max_lags)
    numerator = (n_lags * sum_xy) - (sum_x * sum_y)
    denominator = (n_lags * sum_xx) - (sum_x * sum_x)
    
    # Isolate the slope and compute H
    slope = numerator / denominator
    hurst_exponent = slope / 2.0
    
    return hurst_exponent

if __name__ == "__main__":
    # Example test-bench execution
    # Simulating a highly anti-persistent fractional noise series (Rough Volatility proxy)
    np.random.seed(42)
    # Note: In production, input represents \log(\sigma_t) extracted from TSRV or Options Implied Vol
    dummy_volatility = np.cumsum(np.random.randn(100_000)) 
    
    # Evaluate across 100 micro-structural lags
    H_estimate = compute_hurst_variogram(dummy_volatility, max_lags=100)
    print(f"Empirical Hurst Exponent (H): {H_estimate:.4f}")
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True, nogil=True)
def calculate_hurst_variogram(log_prices: np.ndarray, max_lag: int) -> float:
    """Estima el exponente de Hurst usando variogramas de primer orden."""
    n = len(log_prices)
    lags = np.arange(1, max_lag + 1)
    variances = np.zeros(max_lag)
    
    for i, lag in enumerate(lags):
        diffs = np.abs(log_prices[lag:] - log_prices[:-lag])
        variances[i] = np.mean(diffs)
        
    # Regresión lineal en espacio log-log (evitando np.polyfit por compatibilidad Numba)
    log_lags = np.log(lags)
    log_vars = np.log(variances)
    
    cov = np.cov(log_lags, log_vars)[0, 1]
    var_lags = np.var(log_lags)
    hurst_exponent = cov / var_lags
    
    return hurst_exponent

@njit(cache=True, fastmath=True, nogil=True)
def shannon_entropy(volumes: np.ndarray) -> float:
    """Calcula la entropía de Shannon para los volúmenes del LOB."""
    total_vol = np.sum(volumes)
    if total_vol == 0:
        return 0.0
    probs = volumes / total_vol
    probs = probs[probs > 0] # Evitar log(0)
    return -np.sum(probs * np.log(probs))

# Para compilarlo: python -m py_compile src/microstruct/core_math.py
# Para ejecutarlo: python src/microstruct/core_math.py

# Añadir al final de src/microstruct/core_math.py

@njit(cache=True, fastmath=True, nogil=True)
def gaussian_kernel(x: float) -> float:
    """Kernel Gaussiano estandarizado."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

@njit(cache=True, fastmath=True, nogil=True)
def nadaraya_watson_smoother(t: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    """
    Estimador de regresión no paramétrica de Nadaraya-Watson.
    Aplica suavizado espacial sobre la serie temporal y para aislar la señal del ruido microestructural.
    """
    n = len(t)
    y_hat = np.zeros(n)
    
    for i in range(n):
        weights = np.zeros(n)
        for j in range(n):
            weights[j] = gaussian_kernel((t[i] - t[j]) / h)
        
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            y_hat[i] = np.sum(weights * y) / sum_weights
        else:
            y_hat[i] = y[i]  # Fallback si el peso se desvanece
            
    return y_hat

# python -m py_compile src/microstruct/core_math.py
# python src/microstruct/core_math.py
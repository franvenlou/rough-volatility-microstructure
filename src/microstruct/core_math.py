"""Moment-scaling diagnostics and Gaussian kernel smoothing.

These numerical routines do not establish a market regime or validate a
stochastic-volatility model. Inputs to the scaling estimator should be observed
on an equally spaced grid; lag lengths are measured in observations.
"""

from numbers import Integral

import numpy as np
from numba import njit


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Convert a numeric vector to float64 and reject invalid observations."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return values


@njit(cache=True, nogil=True)
def _hurst_variogram(series: np.ndarray, max_lag: int, moment_order: int) -> float:
    log_lags = np.log(np.arange(1, max_lag + 1, dtype=np.float64))
    log_moments = np.empty(max_lag, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        differences = np.abs(series[lag:] - series[:-lag])
        if moment_order == 2:
            differences = differences * differences
        moment = np.mean(differences)
        if not np.isfinite(moment) or moment <= 0.0:
            raise ValueError("Every lag must have a positive, finite increment moment")
        log_moments[lag - 1] = np.log(moment)

    centered_lags = log_lags - np.mean(log_lags)
    centered_moments = log_moments - np.mean(log_moments)
    # Matching centered sums avoid mixing sample covariance and population variance.
    slope = np.sum(centered_lags * centered_moments) / np.sum(centered_lags**2)
    return slope / moment_order


def calculate_hurst_variogram(
    log_prices: np.ndarray, max_lag: int, moment_order: int = 1
) -> float:
    """Estimate H from the lag scaling of absolute increment moments.

    For lags 1 through ``max_lag``, regress
    ``log(mean(abs(X[t + lag] - X[t])**q))`` on ``log(lag)`` with an
    intercept and return the slope divided by q. The default q=1 preserves
    the pipeline's first absolute moment estimator; q=2 exposes the squared
    increment estimator formerly in ``rough_vol_hurst.py``.

    ``log_prices`` is the historical parameter name, but the input can be any
    finite scalar series, including log-volatility. This scaling estimate is
    interpretable as H only under appropriate process assumptions. No clipping,
    significance test, or automatic lag selection is applied.

    ``max_lag`` must be an integer with ``2 <= max_lag < len(log_prices)``.
    Zero moments (including constant input) and nonfinite moments raise
    ``ValueError`` because their logarithms cannot be fitted.
    """
    series = _finite_vector(log_prices, "log_prices")
    if (
        isinstance(max_lag, bool)
        or not isinstance(max_lag, Integral)
        or not 2 <= max_lag < len(series)
    ):
        raise ValueError("max_lag must be an integer between 2 and len(log_prices) - 1")
    if (
        isinstance(moment_order, bool)
        or not isinstance(moment_order, Integral)
        or moment_order not in (1, 2)
    ):
        raise ValueError("moment_order must be 1 or 2")
    return float(_hurst_variogram(series, int(max_lag), int(moment_order)))


def shannon_entropy(volumes: np.ndarray) -> float:
    """Return entropy in nats after normalizing nonnegative volume weights.

    Each array element is one category. In the main pipeline these categories
    are snapshots, so this measures concentration of total volume across time,
    not discrete order-flow state uncertainty or bid/ask depth entropy.
    Empty arrays and all-zero volumes return zero by convention.
    """
    volumes = _finite_vector(volumes, "volumes")
    if np.any(volumes < 0.0):
        raise ValueError("volumes must be nonnegative")
    if volumes.size == 0 or np.max(volumes) == 0.0:
        return 0.0
    # Rescaling preserves probabilities and avoids overflow in the total volume.
    weights = volumes / np.max(volumes)
    probabilities = weights[weights > 0.0] / np.sum(weights)
    return float(-np.sum(probabilities * np.log(probabilities)))


@njit(cache=True, nogil=True)
def gaussian_kernel(x: float) -> float:
    """Evaluate the standard normal density at x."""
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


@njit(cache=True, nogil=True)
def _nadaraya_watson(t: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    n = len(t)
    fitted = np.empty(n, dtype=np.float64)
    for i in range(n):
        numerator = 0.0
        denominator = 0.0
        for j in range(n):
            weight = gaussian_kernel((t[i] - t[j]) / h)
            numerator += weight * y[j]
            denominator += weight
        # Every observation includes its own positive kernel weight.
        fitted[i] = numerator / denominator
    return fitted


def nadaraya_watson_smoother(t: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    """Smooth y at its observed coordinates using normalized Gaussian weights.

    The bandwidth h must be positive and finite, and has the same units as t.
    There is no bandwidth selection or boundary correction. This implementation
    uses all observations for each fitted value (quadratic time), including
    observations later in time; it is an offline smoother, not a causal filter.
    Empty, equally sized inputs return an empty array.
    """
    t = _finite_vector(t, "t")
    y = _finite_vector(y, "y")
    if len(t) != len(y):
        raise ValueError("t and y must have the same length")
    if not np.isscalar(h) or not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be a positive, finite scalar")
    # Preserve the exact constant-function identity: rounding in weighted sums
    # otherwise creates tiny increments that can yield a spurious Hurst fit.
    if y.size and np.all(y == y[0]):
        return y.copy()
    return _nadaraya_watson(t, y, float(h))

"""Depth imbalance, discrete state entropy, and empirical mutual information.

These exploratory statistics describe observed frequencies. They do not
identify algorithmic trading, institutional participation, or causality.
"""

from numbers import Integral

import numpy as np

from .core_math import _finite_vector, shannon_entropy


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _discrete_sequence(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError(f"{name} must contain nonnegative integer states")
    return values


def _log_base(base: float) -> float:
    if not np.isscalar(base) or not np.isfinite(base) or base <= 1.0:
        raise ValueError("base must be a finite scalar greater than 1")
    return float(np.log(base))


def compute_volume_imbalance(lob_tensor: np.ndarray, level: int = 1) -> np.ndarray:
    """Return (bid volume - ask volume) / total volume at a one-based level.

    Input columns repeat ``[bid price, bid volume, ask price, ask volume]``
    for each level; the number of columns must be a positive multiple of four.
    Only volumes at the requested level are used. They must be finite and
    nonnegative. An empty level (both volumes zero) has imbalance zero.
    """
    lob = np.asarray(lob_tensor, dtype=np.float64)
    level = _positive_integer(level, "level")
    if lob.ndim != 2 or lob.shape[1] == 0 or lob.shape[1] % 4 != 0:
        raise ValueError("lob_tensor must have shape (observations, 4 * levels)")
    if level > lob.shape[1] // 4:
        raise ValueError("level exceeds the available book depth")
    offset = 4 * (level - 1)
    bid = _finite_vector(lob[:, offset + 1], "bid volumes")
    ask = _finite_vector(lob[:, offset + 3], "ask volumes")
    if np.any(bid < 0.0) or np.any(ask < 0.0):
        raise ValueError("book volumes must be nonnegative")

    # Scaling avoids overflow; explicit masking leaves nonempty depths unbiased.
    scale = np.maximum(bid, ask)
    bid = np.divide(bid, scale, out=np.zeros_like(bid), where=scale > 0.0)
    ask = np.divide(ask, scale, out=np.zeros_like(ask), where=scale > 0.0)
    return np.divide(bid - ask, bid + ask, out=np.zeros_like(bid), where=scale > 0.0)


def discretize_imbalance(imbalance_tensor: np.ndarray, bins: int) -> np.ndarray:
    """Map values in [-1, 1] to equal-width bins numbered 0 through bins - 1.

    Internal boundaries belong to the higher bin. The endpoints -1 and +1
    belong to the first and last bins, respectively.
    """
    values = _finite_vector(imbalance_tensor, "imbalance_tensor")
    bins = _positive_integer(bins, "bins")
    if np.any((values < -1.0) | (values > 1.0)):
        raise ValueError("imbalance values must lie in [-1, 1]")
    edges = np.linspace(-1.0, 1.0, bins + 1)
    return np.digitize(values, edges[1:-1]).astype(np.int64)


def estimate_shannon_entropy(discrete_sequence: np.ndarray, base: float = 2.0) -> float:
    """Return plug-in entropy of nonnegative integer states (bits by default).

    Frequencies, rather than the numeric magnitude of state labels, define the
    probabilities. No finite-sample bias correction is applied. Empty input
    returns zero. Any finite logarithm base greater than one is supported.
    """
    sequence = _discrete_sequence(discrete_sequence, "discrete_sequence")
    log_base = _log_base(base)
    _, counts = np.unique(sequence, return_counts=True)
    return shannon_entropy(counts) / log_base


def _paired_sequences(
    seq_x: np.ndarray, seq_y: np.ndarray, num_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    x = _discrete_sequence(seq_x, "seq_x")
    y = _discrete_sequence(seq_y, "seq_y")
    num_bins = _positive_integer(num_bins, "num_bins")
    if x.size != y.size:
        raise ValueError("paired sequences must have the same length")
    if np.any(x >= num_bins) or np.any(y >= num_bins):
        raise ValueError("states must be smaller than num_bins")
    return x, y


def _joint_entropy(x: np.ndarray, y: np.ndarray, log_base: float) -> float:
    _, counts = np.unique(np.column_stack((x, y)), axis=0, return_counts=True)
    return shannon_entropy(counts) / log_base


def _estimate_joint_entropy(
    seq_x: np.ndarray, seq_y: np.ndarray, num_bins: int, base: float = 2.0
) -> float:
    """Return empirical entropy of aligned state pairs, with validated bounds."""
    x, y = _paired_sequences(seq_x, seq_y, num_bins)
    return _joint_entropy(x, y, _log_base(base))


def estimate_mutual_information(
    seq_l1: np.ndarray, seq_l5: np.ndarray, num_bins: int
) -> tuple[float, float]:
    """Return empirical MI in bits and MI divided by joint entropy.

    Sequence names retain the original level-1/level-5 example, but any two
    aligned discrete state sequences can be used. Both must have states in
    ``[0, num_bins)`` and equal length. The normalization is ``I(X;Y)/H(X,Y)``;
    it is not an independently validated market index. Empty sequences and
    zero joint entropy return (0, 0). No finite-sample bias correction is used.
    """
    x, y = _paired_sequences(seq_l1, seq_l5, num_bins)
    h_x = estimate_shannon_entropy(x)
    h_y = estimate_shannon_entropy(y)
    h_joint = _joint_entropy(x, y, float(np.log(2.0)))
    if h_joint == 0.0:
        return 0.0, 0.0
    # Empirical MI is nonnegative; remove only floating-point cancellation below 0.
    mutual_information = max(0.0, h_x + h_y - h_joint)
    return mutual_information, mutual_information / h_joint

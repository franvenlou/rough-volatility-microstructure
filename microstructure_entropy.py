import numpy as np
import numpy.typing as npt
from numba import njit
from typing import Tuple

# Strict type aliases for static analysis
Tensor2D = npt.NDArray[np.float64]
Tensor1D = npt.NDArray[np.float64]
Discrete1D = npt.NDArray[np.int64]

def compute_volume_imbalance(lob_tensor: Tensor2D, level: int = 1) -> Tensor1D:
    """
    Vectorized extraction of the Volume Imbalance Ratio for a specific LOB topological layer.
    
    Expects a 2D tensor of shape (N, 20) representing 5 levels of depth:
    Columns per level: [Bid Price, Bid Volume, Ask Price, Ask Volume]
    
    The Depth Imbalance is formalized as the difference between the bid and ask depths 
    normalized by the total depth at that level[cite: 1].
    """
    # Offset calculation based on the required LOB level (1-indexed)
    col_idx = (level - 1) * 4
    
    bid_volumes = lob_tensor[:, col_idx + 1]
    ask_volumes = lob_tensor[:, col_idx + 3]
    
    total_depth = bid_volumes + ask_volumes
    # Suppress divide-by-zero warnings in perfectly empty levels by adding machine epsilon
    imbalance = (bid_volumes - ask_volumes) / (total_depth + np.finfo(float).eps)
    
    return imbalance

def discretize_imbalance(imbalance_tensor: Tensor1D, bins: int) -> Discrete1D:
    """
    Quantizes (coarse-grains) the continuous imbalance [-1, 1] domain into B discrete, 
    parametric bins to prepare the data for information-theoretic evaluations.
    """
    # Create linearly spaced bin edges between -1.0 and 1.0
    bin_edges = np.linspace(-1.0, 1.0, bins + 1)
    
    # np.digitize returns 1-indexed bins; subtract 1 for 0-indexed integer arrays
    discrete_sequence = np.digitize(imbalance_tensor, bin_edges[1:-1])
    return discrete_sequence.astype(np.int64)

@njit(cache=True, fastmath=True, nogil=True)
def estimate_shannon_entropy(discrete_sequence: np.ndarray, base: float = 2.0) -> float:
    """
    Estimates the empirical Shannon Entropy of a quantized ring buffer sequence.
    By quantifying the amount of uncertainty, a lower entropy indicates a collapse in randomness, 
    signaling algorithmic determinism[cite: 2].
    """
    n_obs = len(discrete_sequence)
    if n_obs == 0:
        return 0.0
        
    # Dynamically size the frequency counter based on the maximum bin index
    max_bin = np.max(discrete_sequence)
    frequencies = np.zeros(max_bin + 1, dtype=np.float64)
    
    # Compute empirical absolute frequencies (avoiding python object instantiation)
    for i in range(n_obs):
        frequencies[discrete_sequence[i]] += 1.0
        
    entropy = 0.0
    for i in range(len(frequencies)):
        if frequencies[i] > 0.0:
            p_i = frequencies[i] / n_obs
            # Accumulate entropy: -\sum p_i \log(p_i)[cite: 2]
            entropy -= p_i * np.log(p_i)
            
    # Convert from nats to bits if base=2
    if base == 2.0:
        entropy /= np.log(2.0)
        
    return entropy

@njit(cache=True, fastmath=True, nogil=True)
def _estimate_joint_entropy(seq_x: np.ndarray, seq_y: np.ndarray, num_bins: int, base: float = 2.0) -> float:
    """Helper function to calculate joint entropy H(X, Y) without GIL overhead."""
    n_obs = len(seq_x)
    joint_freqs = np.zeros((num_bins, num_bins), dtype=np.float64)
    
    for i in range(n_obs):
        joint_freqs[seq_x[i], seq_y[i]] += 1.0
        
    joint_entropy = 0.0
    for i in range(num_bins):
        for j in range(num_bins):
            if joint_freqs[i, j] > 0.0:
                p_ij = joint_freqs[i, j] / n_obs
                joint_entropy -= p_ij * np.log(p_ij)
                
    if base == 2.0:
        joint_entropy /= np.log(2.0)
        
    return joint_entropy

@njit(cache=True, fastmath=True, nogil=True)
def estimate_mutual_information(seq_l1: np.ndarray, seq_l5: np.ndarray, num_bins: int) -> Tuple[float, float]:
    """
    Calculates the Mutual Information I(L_1; L_5) between the superficial limit order book 
    layer and the deep institutional layers. 
    Returns the absolute Mutual Information and the Thermodynamic Market Index (Normalized MI).
    """
    h_l1 = estimate_shannon_entropy(seq_l1, base=2.0)
    h_l5 = estimate_shannon_entropy(seq_l5, base=2.0)
    h_joint = _estimate_joint_entropy(seq_l1, seq_l5, num_bins, base=2.0)
    
    # I(X; Y) = H(X) + H(Y) - H(X, Y)
    mutual_information = h_l1 + h_l5 - h_joint
    
    # Thermodynamic Market Index: Normalizing I(X;Y) by the joint entropy to bound [0, 1]
    # Represents the percentage of shared information structure across the LOB depth.
    if h_joint > 0.0:
        thermo_index = mutual_information / h_joint
    else:
        thermo_index = 0.0
        
    return mutual_information, thermo_index

if __name__ == "__main__":
    # Test-bench execution mimicking N=10,000 tick snapshot tensor with 5 levels (20 cols)
    np.random.seed(42)
    dummy_lob_data = np.random.uniform(1.0, 100.0, size=(10_000, 20))
    
    # 1. Preprocess Vectorized Imbalances
    imbalance_l1 = compute_volume_imbalance(dummy_lob_data, level=1)
    imbalance_l5 = compute_volume_imbalance(dummy_lob_data, level=5)
    
    # 2. Discretization into B parametric bins
    B_BINS = 10
    discrete_l1 = discretize_imbalance(imbalance_l1, bins=B_BINS)
    discrete_l5 = discretize_imbalance(imbalance_l5, bins=B_BINS)
    
    # 3 & 4. Estimate Shannon Entropy & Mutual Information bounds
    h_l1 = estimate_shannon_entropy(discrete_l1, base=2.0)
    mi, thermo_idx = estimate_mutual_information(discrete_l1, discrete_l5, num_bins=B_BINS)
    
    print(f"L1 Shannon Entropy: {h_l1:.4f} bits")
    print(f"L1-L5 Mutual Information: {mi:.4f} bits")
    print(f"Thermodynamic Market Index (Norm MI): {thermo_idx:.4f}")
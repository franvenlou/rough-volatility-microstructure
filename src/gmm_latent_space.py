import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
import warnings

# Suprimir warnings matemáticos menores de divisiones por cero en ventanas sin volumen
warnings.filterwarnings('ignore')

def rolling_hurst(log_vol: np.ndarray, max_lag: int = 5) -> float:
    """Calcula el exponente de Hurst en una ventana pequeña mediante variograma."""
    if len(log_vol) < max_lag + 2:
        return 0.5
    lags = np.arange(1, max_lag + 1)
    variances = np.array([np.mean(np.abs(log_vol[lag:] - log_vol[:-lag])) for lag in lags])
    
    # Evitar log(0)
    variances = np.where(variances == 0, 1e-8, variances)
    
    log_lags = np.log(lags)
    log_vars = np.log(variances)
    
    if np.var(log_lags) == 0:
        return 0.5
        
    cov = np.cov(log_lags, log_vars)[0, 1]
    hurst_exponent = cov / np.var(log_lags)
    return hurst_exponent

def rolling_entropy(volumes: np.ndarray) -> float:
    """Calcula la entropía de Shannon local."""
    total_vol = np.sum(volumes)
    if total_vol == 0:
        return 0.0
    probs = volumes / total_vol
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def generate_latent_space(parquet_path: str, window_size: int = 50):
    print("1. Procesando tensores desde LOB Parquet...")
    df = pd.read_parquet(parquet_path)
    df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2.0
    df['total_volume'] = df['bid_vol'] + df['ask_vol']
    
    returns = np.abs(np.diff(np.log(df['mid_price'].to_numpy())))
    returns = np.where(returns == 0, 1e-8, returns)
    volumes = df['total_volume'].to_numpy()[1:]
    
    n_points = len(returns) - window_size
    
    hurst_series = np.zeros(n_points)
    entropy_series = np.zeros(n_points)
    vol_series = np.zeros(n_points)
    
    print(f"2. Generando espacio latente mediante ventana móvil (N={window_size})...")
    for i in range(n_points):
        window_ret = returns[i:i+window_size]
        window_vol = volumes[i:i+window_size]
        
        hurst_series[i] = rolling_hurst(np.log(window_ret))
        entropy_series[i] = rolling_entropy(window_vol)
        vol_series[i] = np.mean(window_ret) * 1e4 # Escalado a basis points
        
    # Limpiar posibles NaNs o infinitos inducidos por el logaritmo
    valid_idx = np.isfinite(hurst_series) & np.isfinite(entropy_series) & np.isfinite(vol_series)
    X = np.vstack([hurst_series[valid_idx], entropy_series[valid_idx], vol_series[valid_idx]]).T
    
    return X

def plot_gmm_3d(X: np.ndarray, n_components: int = 2):
    print(f"3. Calibrando Gaussian Mixture Model (K={n_components}) vía Expectation-Maximization...")
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(X)
    
    print("4. Renderizando topología del espacio latente 3D...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o', alpha=0.7)
    
    ax.set_xlabel('Exponente de Hurst ($H$)')
    ax.set_ylabel('Entropía del LOB ($S$)')
    ax.set_zlabel('Volatilidad Local (bps)')
    ax.set_title('Clustering Topológico del Mercado (GMM sobre Espacio Latente)')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Régimen de Mercado (Cluster ID)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_latent = generate_latent_space('data/lob_snapshot.parquet', window_size=50)
    plot_gmm_3d(X_latent, n_components=2)

# python -m py_compile src/gmm_latent_space.py
# python src/gmm_latent_space.py
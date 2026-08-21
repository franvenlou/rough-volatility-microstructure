import pandas as pd
import numpy as np
from microstruct.core_math import calculate_hurst_variogram, shannon_entropy, nadaraya_watson_smoother

def run_microstructure_analysis(parquet_path: str):
    print("1. Cargando snapshots del LOB desde Parquet...")
    df = pd.read_parquet(parquet_path)
    
    # Calcular el Mid-Price
    df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2.0
    df['total_volume'] = df['bid_vol'] + df['ask_vol']
    
    # 2. Extracción de Volatilidad Instantánea
    print("2. Calculando retornos y proxy de volatilidad microestructural...")
    mid_price_array = df['mid_price'].to_numpy(dtype=np.float64)
    log_prices = np.log(mid_price_array)
    
    # Retornos logarítmicos (diferencias de primer orden)
    returns = np.diff(log_prices)
    
    # Proxy de volatilidad (valor absoluto de los retornos)
    inst_vol = np.abs(returns)
    # Reemplazamos ceros estrictos por un epsilon mínimo para evitar log(0)
    inst_vol = np.where(inst_vol == 0, 1e-8, inst_vol)
    
    # 3. Aplicar Suavizado Kernel a la VOLATILIDAD, no al precio
    # Reducimos el ancho de banda para no destruir la señal fractal (h=2.0)
    print("3. Aplicando regresión de Nadaraya-Watson a la superficie de volatilidad...")
    t_array = np.arange(len(inst_vol), dtype=np.float64)
    bandwidth = 2.0 
    smoothed_vol = nadaraya_watson_smoother(t_array, inst_vol, bandwidth)
    
    # Transformación logarítmica de la varianza suavizada
    log_vol = np.log(smoothed_vol)
    
    # 4. Estimar el Exponente de Hurst de la Volatilidad y la Entropía del Flujo
    print("4. Evaluando Exponente de Hurst y Entropía de Shannon...")
    max_lag = 20 
    h_exponent = calculate_hurst_variogram(log_vol, max_lag)
    
    volume_array = df['total_volume'].to_numpy(dtype=np.float64)
    entropy = shannon_entropy(volume_array)
    
    print("\n" + "="*60)
    print("RESULTADOS DEL ANÁLISIS DE VOLATILIDAD RUGOSA:")
    print("="*60)
    print(f"Exponente de Hurst (H) de la Volatilidad : {h_exponent:.5f}")
    
    if 0.01 <= h_exponent <= 0.30:
        print("  -> Confirmado empíricamente: Régimen de Volatilidad RUGOSA.")
        print("  -> Conclusión: El modelo de Black-Scholes subestimará sistemáticamente el riesgo de cola a corto plazo.")
    elif 0.40 <= h_exponent <= 0.60:
        print("  -> Dinámica Browniana Clásica detectada. Volatilidad difusiva.")
    else:
        print("  -> Comportamiento anómalo o persistencia extrema.")
        
    print(f"Entropía de Shannon (S) del LOB          : {entropy:.5f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_microstructure_analysis('data/lob_snapshot.parquet')
import numpy as np
import networkx as nx
from scipy.linalg import inv
# Importamos tu algoritmo previamente validado
from pmfg_filter import build_pmfg

def calculate_topological_weights(cov_matrix: np.ndarray, pmfg: nx.Graph) -> np.ndarray:
    """
    Calcula los pesos óptimos de Markowitz utilizando la matriz de precisión 
    derivada del Planar Maximally Filtered Graph (PMFG).
    """
    n = cov_matrix.shape[0]
    
    # 1. Crear máscara topológica (Adyacencia del PMFG)
    mask = nx.to_numpy_array(pmfg, nodelist=range(n))
    
    # La diagonal principal (varianza intrínseca) siempre debe preservarse
    np.fill_diagonal(mask, 1.0)
    
    # 2. Filtrado Estructural (Producto de Hadamard)
    # Retenemos únicamente las covarianzas que superaron el test de planaridad
    filtered_cov = cov_matrix * (mask > 0)
    
    # 3. Regularización de Tikhonov (Ridge)
    # Forzamos matemáticamente que la matriz enmascarada sea definida positiva 
    # añadiendo un factor infinitesimal a la diagonal para garantizar su invertibilidad.
    epsilon = 1e-4
    filtered_cov += epsilon * np.eye(n)
    
    # 4. Inversión para obtener la Matriz de Precisión Topológica
    precision_matrix = inv(filtered_cov)
    
    # 5. Ecuación de Mínima Varianza Global de Markowitz
    ones = np.ones(n)
    weights = precision_matrix @ ones
    optimal_weights = weights / (ones.T @ precision_matrix @ ones)
    
    return optimal_weights

def calculate_standard_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """Markowitz Clásico para comparar la estabilidad."""
    n = cov_matrix.shape[0]
    precision_matrix = inv(cov_matrix)
    ones = np.ones(n)
    weights = precision_matrix @ ones
    return weights / (ones.T @ precision_matrix @ ones)

if __name__ == "__main__":
    np.random.seed(42)
    n_activos = 10
    
    print("1. Simulando universo de activos financieros...")
    returns = np.random.randn(n_activos, 1000)
    # Introducimos un factor sistémico para dotar de estructura a la matriz
    returns += np.random.randn(1000) * 0.5 
    
    cov_empirica = np.cov(returns)
    
    print("2. Construyendo la topología PMFG...")
    grafo_pmfg = build_pmfg(np.corrcoef(returns))
    
    print("3. Optimizando Cartera Topológica vs Clásica...\n")
    w_topologico = calculate_topological_weights(cov_empirica, grafo_pmfg)
    w_clasico = calculate_standard_weights(cov_empirica)
    
    print("="*60)
    print("COMPARATIVA DE PESOS ÓPTIMOS (MARKOWITZ)")
    print("="*60)
    print(f"{'Activo':<10} | {'Clásico (Ruidoso)':<20} | {'Topológico (PMFG)':<20}")
    print("-" * 60)
    
    for i in range(n_activos):
        # Multiplicamos por 100 para mostrar en porcentaje
        print(f"Activo {i:<3} | {w_clasico[i]*100:>17.2f}% | {w_topologico[i]*100:>17.2f}%")
    
    print("="*60)
    
    # Medición objetiva de la estabilidad (Norma L2 del vector de pesos)
    # Una norma menor implica menos apalancamiento/cortos extremos y más estabilidad
    norma_clasica = np.linalg.norm(w_clasico)
    norma_topo = np.linalg.norm(w_topologico)
    
    print(f"\nNorma L2 (Clásica)   : {norma_clasica:.4f} -> Elevada susceptibilidad a shocks.")
    print(f"Norma L2 (Topológica): {norma_topo:.4f} -> Cartera robusta y estructuralmente estable.")

# python -m py_compile src/topology/markowitz_topologico.py
# python src/topology/markowitz_topologico.py
# endprogram
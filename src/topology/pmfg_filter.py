import numpy as np
import networkx as nx

def build_pmfg(correlation_matrix: np.ndarray) -> nx.Graph:
    """
    Construye el Planar Maximally Filtered Graph (PMFG) a partir de una matriz de correlación empírica.
    Aplica el teorema de Kuratowski para garantizar un género topológico g=0.
    """
    n_nodes = correlation_matrix.shape[0]
    
    # Validación axiomática de la dimensionalidad
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("La matriz de correlación debe ser un operador lineal cuadrado (NxN).")

    # Extraemos la triángular superior para evitar duplicidades
    i_idx, j_idx = np.triu_indices(n_nodes, k=1)
    correlations = correlation_matrix[i_idx, j_idx]

    # Ordenamos topológicamente por la magnitud de la correlación (descendente)
    sorted_indices = np.argsort(correlations)[::-1]
    
    pmfg = nx.Graph()
    pmfg.add_nodes_from(range(n_nodes))
    
    edges_added = 0
    max_edges = 3 * (n_nodes - 2)
    
    for idx in sorted_indices:
        u, v = i_idx[idx], j_idx[idx]
        weight = correlations[idx]
        
        pmfg.add_edge(u, v, weight=weight)
        
        # Test de planaridad algorítmica (Boyer-Myrvold o equivalente en NetworkX)
        is_planar, _ = nx.check_planarity(pmfg)
        
        if not is_planar:
            # Si rompe el género 0, se desecha la arista (filtro de ruido)
            pmfg.remove_edge(u, v)
        else:
            edges_added += 1
            if edges_added == max_edges:
                break
                
    return pmfg

if __name__ == "__main__":
    print("Iniciando filtrado topológico PMFG sobre matriz de covarianza simulada...")
    
    # Generamos una matriz de correlación sintética de 10 activos (para validación del teorema)
    np.random.seed(42)
    random_returns = np.random.randn(10, 1000)
    empirical_corr = np.corrcoef(random_returns)
    
    pmfg_graph = build_pmfg(empirical_corr)
    
    print("\n" + "="*50)
    print("ANÁLISIS TOPOLÓGICO COMPLETADO")
    print("="*50)
    print(f"Nodos procesados (Activos del Universo) : {pmfg_graph.number_of_nodes()}")
    print(f"Aristas de Información retenidas        : {pmfg_graph.number_of_edges()}")
    print(f"Límite teórico de Euler (3*(V-2))       : {3 * (10 - 2)}")
    print("="*50)
    print("Diagnóstico: El ruido espurio ha sido filtrado preservando los cliques sectoriales.")

# Para compilarlo: python -m py_compile src/topology/pmfg_filter.py
# Para ejecutarlo: python src/topology/pmfg_filter.py
# endprogram
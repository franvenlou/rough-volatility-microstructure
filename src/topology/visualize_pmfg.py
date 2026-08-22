import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pmfg_filter import build_pmfg

def plot_pmfg(G: nx.Graph):
    """
    Renderiza la estructura topológica del PMFG.
    Aplica el algoritmo de energía de Kamada-Kawai para optimizar la visualización planar
    y mapea el grado de centralidad de los nodos a una escala de colores termodinámica.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Algoritmo de posicionamiento basado en minimización de energía (óptimo para grafos filtrados)
    pos = nx.kamada_kawai_layout(G)
    
    # Extraemos la centralidad de grado para ajustar el tamaño y color de los nodos (Hubs)
    degrees = dict(nx.degree(G))
    node_sizes = [v * 180 for v in degrees.values()]
    node_colors = list(degrees.values())
    
    # Extraemos los pesos de las aristas (correlaciones empíricas) para la paleta de enlaces
    edges = G.edges(data=True)
    edge_colors = [data['weight'] for u, v, data in edges]
    
    # Renderizamos los nodos (Activos financieros)
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        edgecolors='white',
        linewidths=1.5,
        ax=ax
    )
    
    # Renderizamos las aristas (Enlaces de información mutua)
    edges_drawn = nx.draw_networkx_edges(
        G, pos,
        width=2.5,
        edge_color=edge_colors,
        edge_cmap=plt.cm.viridis,
        alpha=0.8,
        ax=ax
    )
    
    # Renderizamos las etiquetas (IDs de los activos)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="black", ax=ax)
    
    # Incorporamos barras de color para la interpretación rigurosa del analista
    cbar_nodes = plt.colorbar(nodes, ax=ax, fraction=0.03, pad=0.02)
    cbar_nodes.set_label('Centralidad de Grado (Dominancia de Mercado)')
    
    # Workaround para matplotlib: Solo añadir colorbar de aristas si la colección es mapeable
    if hasattr(edges_drawn, 'set_array'):
        cbar_edges = plt.colorbar(edges_drawn, ax=ax, fraction=0.03, pad=0.07)
        cbar_edges.set_label('Magnitud de la Correlación Empírica')
        
    ax.set_title("Estructura Topológica PMFG (Planar Maximally Filtered Graph)", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("1. Generando matriz de covarianza estocástica sintética de 30 activos...")
    np.random.seed(42)
    
    # Simulamos 30 activos con 1000 observaciones temporales (e.g., retornos diarios)
    random_returns = np.random.randn(30, 1000)
    
    # Inyectamos un 'Factor Común de Mercado' a todos los retornos para forzar 
    # la aparición empírica de correlaciones positivas y cliques, imitando a un mercado real.
    market_factor = np.random.randn(1000)
    random_returns += market_factor * 0.7 
    
    empirical_corr = np.corrcoef(random_returns)
    
    print("2. Construyendo topología PMFG y filtrando ruido de la matriz de covarianza...")
    G_pmfg = build_pmfg(empirical_corr)
    
    print(f"   -> Topología extraída matemáticamente: {G_pmfg.number_of_nodes()} Nodos, {G_pmfg.number_of_edges()} Aristas.")
    print("3. Renderizando el grafo planar. Revise la ventana emergente de Matplotlib...")
    
    plot_pmfg(G_pmfg)

# Para compilarlo: python -m py_compile src/topology/visualize_pmfg.py
# Para ejecutarlo: python src/topology/visualize_pmfg.py
# endprogram
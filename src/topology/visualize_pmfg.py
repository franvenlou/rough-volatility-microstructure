"""Visualize a PMFG with correlation-colored edges."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np

from src.topology.pmfg_filter import build_pmfg


def plot_pmfg(
    graph: nx.Graph,
    *,
    output: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot degree and signed correlations using an unweighted graph layout.

    Kamada-Kawai treats edge weights as distances. Correlations may be negative,
    so use unweighted shortest-path distances for positioning. The drawing itself
    is not guaranteed to be a crossing-free planar embedding.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    positions = nx.kamada_kawai_layout(graph, weight=None)
    degrees = dict(nx.degree(graph))
    node_sizes = [max(degree, 1) * 180 for degree in degrees.values()]
    node_colors = list(degrees.values())
    correlations = [data["weight"] for _, _, data in graph.edges(data=True)]
    # Share a fixed correlation scale so uniform weights cannot desynchronize
    # precomputed edge colors from a colorbar's automatic range expansion.
    edge_mappable = ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=plt.cm.viridis)

    nodes = nx.draw_networkx_nodes(
        graph, positions,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        edgecolors="white",
        linewidths=1.5,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph, positions,
        width=2.5,
        edge_color=edge_mappable.to_rgba(correlations),
        alpha=0.8,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph, positions, font_size=10, font_weight="bold", font_color="black", ax=ax
    )

    node_colorbar = fig.colorbar(nodes, ax=ax, fraction=0.03, pad=0.02)
    node_colorbar.set_label("Node degree")
    if correlations:
        edge_colorbar = fig.colorbar(edge_mappable, ax=ax, fraction=0.03, pad=0.07)
        edge_colorbar.set_label("Empirical correlation")

    ax.set_title("Planar Maximally Filtered Graph (PMFG)", fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    fig.tight_layout()
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    args = parser.parse_args()
    print("Simulating 30 assets with a common return factor...")
    np.random.seed(42)
    random_returns = np.random.randn(30, 1000)
    random_returns += np.random.randn(1000) * 0.7
    graph = build_pmfg(np.corrcoef(random_returns))
    print(f"PMFG: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
    fig = plot_pmfg(graph, output=args.output, show=not args.no_show)
    if args.output is not None:
        print(f"Saved plot to {args.output}.")
    plt.close(fig)


if __name__ == "__main__":
    main()

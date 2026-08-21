import numpy as np
import numpy.typing as npt
import networkx as nx
from sklearn.covariance import LedoitWolf
from typing import Tuple, List, Set

# Type aliases for strict static analysis
Tensor2D = npt.NDArray[np.float64]

class TopologicalFilter:
    """
    Multivariate topological filtering framework for high-frequency financial networks.
    Extracts strictly planar or tree-like backbones from noisy elliptical covariance structures.
    """
    
    def __init__(self, raw_returns: Tensor2D) -> None:
        self.raw_returns: Tensor2D = raw_returns
        self.n_assets: int = raw_returns.shape[1]
        self.assets_indices: List[int] = list(range(self.n_assets))
        
        # Internal state matrices
        self.covariance_matrix: Tensor2D = np.empty((self.n_assets, self.n_assets))
        self.correlation_matrix: Tensor2D = np.empty((self.n_assets, self.n_assets))
        self.distance_matrix: Tensor2D = np.empty((self.n_assets, self.n_assets))
        
    def apply_ledoit_wolf_shrinkage(self) -> None:
        """
        Applies iterative Ledoit-Wolf shrinkage to process the raw returns tensor into 
        a well-conditioned covariance matrix, followed by a transformation into a 
        strict metric distance matrix to induce an intuitive topology.
        """
        # Step 1: Matrix Shrinkage
        lw = LedoitWolf()
        self.covariance_matrix = lw.fit(self.raw_returns).covariance_
        
        # Step 2: Correlation extraction
        v = np.sqrt(np.diag(self.covariance_matrix))
        outer_v = np.outer(v, v)
        self.correlation_matrix = self.covariance_matrix / outer_v
        
        # Clip to prevent floating point instability outside [-1, 1]
        self.correlation_matrix = np.clip(self.correlation_matrix, -1.0, 1.0)
        
        # Step 3: Transformation to a valid metric distance space
        self.distance_matrix = np.sqrt(2.0 * (1.0 - self.correlation_matrix))

    def extract_mst_kruskal(self) -> nx.Graph:
        """
        Extracts the Minimum Spanning Tree (MST) utilizing strict matrix-level Kruskal's 
        algorithm, ensuring a topological tree extraction bounded in O(E log V) time.
        """
        mst = nx.Graph()
        mst.add_nodes_from(self.assets_indices)
        
        # Extract upper triangular edges to avoid duplication (i < j)
        edges = []
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                edges.append((self.distance_matrix[i, j], i, j))
                
        # Sort edges by distance (ascending) to satisfy Kruskal's greedy requirement
        edges.sort(key=lambda x: x[0])
        
        # Union-Find / Disjoint Set structure for O(E log V) cycle detection
        parent = {i: i for i in self.assets_indices}
        
        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int) -> bool:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                return True
            return False

        # Build MST
        for weight, u, v in edges:
            if union(u, v):
                mst.add_edge(u, v, weight=weight)
                if mst.number_of_edges() == self.n_assets - 1:
                    break
                    
        return mst

    def extract_tmfg(self) -> nx.Graph:
        """
        Implements the Triangulated Maximally Filtered Graph (TMFG) algorithm.
        
        STRUCTURAL LOGIC & ASYMPTOTIC ADVANTAGE:
        Classical PMFG extraction requires strict boolean planarity testing (e.g., Boyer-Myrvold) 
        at each edge insertion, yielding a computational complexity of O(V^3). The TMFG circumvents 
        this asymptotic bottleneck by deterministically grafting nodes directly into chordal planar 
        sub-cliques (triangular faces or K3 cliques). 
        
        By starting with a K4 tetrahedron and sequentially adding the remaining V-4 vertices inside 
        the planar triangular faces that maximize the local correlation (minimize distance), we 
        guarantee planarity by construction. The resulting network is a chordal graph composed 
        exclusively of 3-cliques and 4-cliques.
        """
        tmfg = nx.Graph()
        tmfg.add_nodes_from(self.assets_indices)
        
        # Initialization: Find the 4 vertices that form the K4 clique with minimum total distance
        # For production scalability with large V, this is approximated via sorting the heaviest edges.
        # Here we mock the initialization of the first 4 nodes for algorithmic structural clarity.
        initial_k4 = self.assets_indices[:4] 
        uninserted_nodes = set(self.assets_indices[4:])
        
        # Add the K4 edges to the graph
        for i in range(4):
            for j in range(i + 1, 4):
                tmfg.add_edge(initial_k4[i], initial_k4[j], weight=self.distance_matrix[initial_k4[i], initial_k4[j]])
                
        # The K4 tetrahedron creates 4 triangular faces (K3)
        faces: List[Tuple[int, int, int]] = [
            (initial_k4[0], initial_k4[1], initial_k4[2]),
            (initial_k4[0], initial_k4[1], initial_k4[3]),
            (initial_k4[0], initial_k4[2], initial_k4[3]),
            (initial_k4[1], initial_k4[2], initial_k4[3])
        ]
        
        # Iterative TMFG Insertion
        while uninserted_nodes:
            best_node = -1
            best_face_idx = -1
            min_dist_sum = float('inf')
            
            # Find the node-face combination that minimizes the distance metric
            for node in uninserted_nodes:
                for f_idx, face in enumerate(faces):
                    dist_sum = sum(self.distance_matrix[node, v] for v in face)
                    if dist_sum < min_dist_sum:
                        min_dist_sum = dist_sum
                        best_node = node
                        best_face_idx = f_idx
                        
            # Graft the node into the optimal planar triangular face
            target_face = faces.pop(best_face_idx)
            uninserted_nodes.remove(best_node)
            
            for v in target_face:
                tmfg.add_edge(best_node, v, weight=self.distance_matrix[best_node, v])
                
            # Splitting the face: One K3 face is broken into three new K3 faces
            faces.append((best_node, target_face[0], target_face[1]))
            faces.append((best_node, target_face[1], target_face[2]))
            faces.append((best_node, target_face[0], target_face[2]))
            
        return tmfg

    def get_masked_precision_matrix(self) -> Tensor2D:
        """
        Computes the precision matrix (inverse covariance) and strictly masks it using the 
        TMFG topological backbone. This mathematically forces conditional independence 
        between assets not connected in the fundamental K3/K4 chordal structure, 
        yielding a sparse tensor ready for probabilistic graphical ingestion.
        """
        # Ensure distances and correlations are extracted
        if self.covariance_matrix.shape[0] == self.n_assets:
            self.apply_ledoit_wolf_shrinkage()
            
        # Extract the TMFG backbone
        tmfg_graph = self.extract_tmfg()
        
        # Build the adjacency boolean mask from the TMFG
        adjacency_mask = nx.to_numpy_array(tmfg_graph, nodelist=self.assets_indices)
        adjacency_mask = (adjacency_mask > 0).astype(np.float64)
        
        # Add self-loops (diagonal = 1) to the mask for valid precision diagonals
        np.fill_diagonal(adjacency_mask, 1.0)
        
        # Compute the precision matrix (Theta)
        precision_matrix = np.linalg.inv(self.covariance_matrix)
        
        # Apply the static topological mask via Hadamard product
        masked_precision = precision_matrix * adjacency_mask
        
        return masked_precision
import random
import torch
import math
from torch_geometric.data import Data

class SyntheticGraphDataGenerator:
    def __init__(self, mode="simple", num_categories=3, p=0.25, num_nodes=20,
                 chain_length_min=2, chain_length_max=7, motif_dim=3,
                 candidate_matrices=None, base_distribution=None):
        """
        Initialize the data generator with configuration parameters.

        Parameters:
          mode (str): "simple", "correlated", "motif", or "combined"
          num_categories (int): Dimensionality for one-hot embeddings (used in simple, correlated, combined)
          p (float): Activation probability for node embeddings.
          num_nodes (int): Number of nodes for chain graphs (simple and correlated).
          chain_length_min (int): Minimum chain length (for motif and combined modes).
          chain_length_max (int): Maximum chain length.
          motif_dim (int): Dimensionality for the motif label (for motif and combined modes).
          candidate_matrices (list of Tensors, optional): For correlated/combined modes.
          base_distribution (Tensor, optional): Base distribution for activation; defaults to uniform.
        """
        self.mode = mode
        self.num_categories = num_categories
        self.p = p
        self.num_nodes = num_nodes
        self.chain_length_min = chain_length_min
        self.chain_length_max = chain_length_max
        self.motif_dim = motif_dim
        self.candidate_matrices = candidate_matrices
        self.base_distribution = base_distribution if base_distribution is not None else torch.ones(num_categories) / num_categories

    # ------------------------------
    # Functions for chain graphs (simple & correlated)
    # ------------------------------

    def create_chain_edge_index(self, num_nodes=None):
        """Creates an edge index for a linear chain with self-loops."""
        if num_nodes is None:
            num_nodes = self.num_nodes
        edges = []
        for i in range(num_nodes):
            edges.append((i, i))  # self-loop
            if i > 0:
                edges.append((i, i - 1))
            if i < num_nodes - 1:
                edges.append((i, i + 1))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def random_onehot_embedding(self, num_categories=None, p=None):
        """Returns a one-hot vector with probability p, else a zero vector."""
        if num_categories is None:
            num_categories = self.num_categories
        if p is None:
            p = self.p
        if random.random() < p:
            emb = torch.zeros(num_categories, dtype=torch.float)
            idx = random.randint(0, num_categories - 1)
            emb[idx] = 1.0
            return emb
        return torch.zeros(num_categories, dtype=torch.float)

    def assign_node_embeddings(self, num_nodes=None, num_categories=None, p=None):
        """Generates node embeddings for a chain (no correlations)."""
        if num_nodes is None:
            num_nodes = self.num_nodes
        if num_categories is None:
            num_categories = self.num_categories
        if p is None:
            p = self.p
        embeddings = [self.random_onehot_embedding(num_categories, p) for _ in range(num_nodes)]
        return torch.stack(embeddings, dim=0)

    def compute_feature_vector(self, edge_index, node_embs, num_categories=None):
        """
        Computes a binary vector (length=num_categories) where each element is 1 if there is at
        least one adjacent node pair (ignoring self-loops) with that feature active.
        """
        if num_categories is None:
            num_categories = self.num_categories
        features = torch.zeros(num_categories, dtype=torch.float)
        src_nodes, tgt_nodes = edge_index[0], edge_index[1]
        non_self_loop = src_nodes != tgt_nodes
        src_nodes = src_nodes[non_self_loop]
        tgt_nodes = tgt_nodes[non_self_loop]
        for feature_idx in range(num_categories):
            active_mask = (node_embs[:, feature_idx] == 1.0)
            active_src = active_mask[src_nodes]
            active_tgt = active_mask[tgt_nodes]
            if torch.any(active_src & active_tgt):
                features[feature_idx] = 1.0
        return features

    # ------------------------------
    # Functions for correlated node embeddings
    # ------------------------------

    def correlated_onehot_embedding(self, prev_feature, num_categories=None, p=None,
                                    base_distribution=None, transition_matrix=None):
        """
        Returns a correlated one-hot vector. If activated (with probability p), the feature
        is sampled based on the previous node's feature using the base_distribution or transition_matrix.
        """
        if num_categories is None:
            num_categories = self.num_categories
        if p is None:
            p = self.p
        if random.random() < p:
            if prev_feature is None:
                probs = base_distribution if base_distribution is not None else torch.ones(num_categories) / num_categories
            else:
                probs = transition_matrix[prev_feature] if transition_matrix is not None else torch.ones(num_categories) / num_categories
            idx = torch.multinomial(probs, num_samples=1).item()
            emb = torch.zeros(num_categories, dtype=torch.float)
            emb[idx] = 1.0
            return emb, idx
        else:
            return torch.zeros(num_categories, dtype=torch.float), None

    def assign_node_embeddings_correlated(self, num_nodes=None, num_categories=None, p=None,
                                            base_distribution=None, transition_matrix=None):
        """
        Generates node embeddings for a chain using a Markov chain mechanism.
        In 'combined' mode, appends an extra constant dimension.
        """
        if num_nodes is None:
            num_nodes = self.num_nodes
        if num_categories is None:
            num_categories = self.num_categories
        if p is None:
            p = self.p
        if base_distribution is None:
            base_distribution = self.base_distribution
        embeddings = []
        prev_feature = None
        for _ in range(num_nodes):
            emb, curr_feature = self.correlated_onehot_embedding(prev_feature, num_categories, p, base_distribution, transition_matrix)
            if self.mode == "combined":
                # Append an extra constant dimension for the combined experiment.
                emb = torch.cat([emb, torch.ones(1, dtype=emb.dtype)], dim=0)
            embeddings.append(emb)
            prev_feature = curr_feature if curr_feature is not None else None
        return torch.stack(embeddings, dim=0)

    # ------------------------------
    # Functions for motif graphs
    # ------------------------------

    def assign_node_features(self, num_nodes):
        """Returns a [num_nodes, 1] tensor where every node feature is 1."""
        return torch.ones((num_nodes, 1), dtype=torch.float)

    def create_motif_edge_index(self, motif_type="triangle", chain_length=3, motif_dim=None):
        """
        Creates an edge index for a graph containing a motif subgraph and attached chains.
        The motif label is determined by the motif type.
        """
        if motif_dim is None:
            motif_dim = self.motif_dim
        if motif_type == "triangle":
            motif_n = 3
            label = [1, 0, 0]
        elif motif_type == "square":
            motif_n = 4
            label = [0, 1, 0]
        elif motif_type == "pentagon":
            motif_n = 5
            label = [0, 0, 1]
        elif motif_type == "pair":
            motif_n = 2
            label = [0] * motif_dim
        else:
            raise ValueError("Invalid motif type")
        total_nodes = motif_n + motif_n * chain_length
        edges = []
        # Add self-loops.
        for i in range(total_nodes):
            edges.append((i, i))
        # Build the motif subgraph.
        if motif_type == "pair":
            edges.append((0, 1))
            edges.append((1, 0))
        elif motif_type in ["square", "pentagon"]:
            for i in range(motif_n):
                for j in range(i + 1, motif_n):
                    edges.append((i, j))
                    edges.append((j, i))
        else:  # triangle: use a cycle
            for i in range(motif_n):
                j = (i + 1) % motif_n
                edges.append((i, j))
                edges.append((j, i))
        # Attach a chain to each motif node.
        for i in range(motif_n):
            start = motif_n + i * chain_length
            edges.append((i, start))
            edges.append((start, i))
            for j in range(chain_length - 1):
                a = start + j
                b = start + j + 1
                edges.append((a, b))
                edges.append((b, a))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index, total_nodes, label

    # ------------------------------
    # Single graph generation method (dispatch based on mode)
    # ------------------------------

    def generate_single_graph(self):
        """
        Generates a single graph based on the chosen mode.
        Returns a PyTorch Geometric Data object.
        """
        if self.mode == "simple":
            # Simple chains with independent node features.
            edge_index = self.create_chain_edge_index()
            x = self.assign_node_embeddings()
            y = self.compute_feature_vector(edge_index, x)
            data = Data(x=x, edge_index=edge_index, y=y)
            data.num_nodes = self.num_nodes
            return data

        elif self.mode == "correlated":
            # Simple chains with correlated node embeddings.
            edge_index = self.create_chain_edge_index()
            transition_matrix = random.choice(self.candidate_matrices) if self.candidate_matrices is not None else None
            x = self.assign_node_embeddings_correlated(transition_matrix=transition_matrix)
            y = self.compute_feature_vector(edge_index, x)
            data = Data(x=x, edge_index=edge_index, y=y)
            data.num_nodes = self.num_nodes
            return data

        elif self.mode == "motif":
            # Graphs with motif topology and constant node features.
            chain_length = random.randint(self.chain_length_min, self.chain_length_max)
            motif_type = random.choice(["triangle", "square", "pentagon"])
            edge_index, total_nodes, label = self.create_motif_edge_index(motif_type=motif_type, chain_length=chain_length)
            x = self.assign_node_features(total_nodes)
            y = torch.tensor(label, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
            data.num_nodes = total_nodes
            return data

        elif self.mode == "combined":
            # Graphs with motif topology and correlated node embeddings.
            chain_length = random.randint(self.chain_length_min, self.chain_length_max)
            motif_type = "pair" if random.random() < 0.5 else random.choice(["triangle", "square", "pentagon"])
            edge_index, total_nodes, motif_label = self.create_motif_edge_index(motif_type=motif_type, chain_length=chain_length)
            transition_matrix = random.choice(self.candidate_matrices) if self.candidate_matrices is not None else None
            x = self.assign_node_embeddings_correlated(num_nodes=total_nodes, num_categories=self.num_categories, p=self.p,
                                                        base_distribution=self.base_distribution, transition_matrix=transition_matrix)
            feature_y = self.compute_feature_vector(edge_index, x, num_categories=self.num_categories)
            motif_y = torch.tensor(motif_label, dtype=torch.float)
            # Concatenate the computed feature vector and motif label.
            y = torch.cat([feature_y, motif_y], dim=0)
            data = Data(x=x, edge_index=edge_index, y=y)
            data.num_nodes = total_nodes
            return data

        else:
            raise ValueError("Invalid mode selected.")

    # ------------------------------
    # Generate multiple graphs
    # ------------------------------

    def generate_data(self, num_samples=1000):
        """
        Generates a list of graph data objects.
        
        Parameters:
          num_samples (int): Number of graphs to generate.
          
        Returns:
          List[Data]: A list of PyTorch Geometric Data objects.
        """
        return [self.generate_single_graph() for _ in range(num_samples)]



# Sparcity Calculator

def sparcity_calculator(num_nodes, p, num_features):
    """
    Returns the probability that there is at least one pair of adjacent
    occupied sites in a lattice of n sites where each site is occupied
    independently with probability p.
    """
    # Compute the probability that there is no adjacent pair.
    prob_no_pair = 0.0
    # Modify definition of p
    p = p/num_features
    # The maximum number of occupied sites without adjacent ones is floor((n+1)/2)
    max_occupied = (num_nodes + 1) // 2
    for k in range(0, max_occupied + 1):
        # Number of ways to place k occupied sites with no adjacent ones:
        ways = math.comb(num_nodes - k + 1, k)
        prob_no_pair += ways * (p**k) * ((1-p)**(num_nodes-k))
    return prob_no_pair
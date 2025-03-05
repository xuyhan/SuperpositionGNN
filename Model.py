import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

# ----------------------------
# Helper functions for final layer initialization
# ----------------------------
def equiangular_frame(out_dim, hidden_dim):
    """
    Returns a fixed weight matrix with an equiangular configuration for some special cases.

    This helps in setting up the final linear layer in a specific configuration.
    """
    if out_dim == 3 and hidden_dim == 2:
        return torch.tensor([
            [1.0, 0.0],
            [-0.5, math.sqrt(3)/2],
            [-0.5, -math.sqrt(3)/2]
        ])
    elif out_dim == 4 and hidden_dim == 2:
        return torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0]
        ])
    elif out_dim == 5 and hidden_dim == 2:
        return torch.tensor([
            [math.cos(0 * 2*math.pi/5), math.sin(0 * 2*math.pi/5)],
            [math.cos(1 * 2*math.pi/5), math.sin(1 * 2*math.pi/5)],
            [math.cos(2 * 2*math.pi/5), math.sin(2 * 2*math.pi/5)],
            [math.cos(3 * 2*math.pi/5), math.sin(3 * 2*math.pi/5)],
            [math.cos(4 * 2*math.pi/5), math.sin(4 * 2*math.pi/5)]
        ])
    elif out_dim == 6 and hidden_dim == 2:
        return torch.tensor([
            [math.cos(0 * 2*math.pi/6), math.sin(0 * 2*math.pi/6)],
            [math.cos(1 * 2*math.pi/6), math.sin(1 * 2*math.pi/6)],
            [math.cos(2 * 2*math.pi/6), math.sin(2 * 2*math.pi/6)],
            [math.cos(3 * 2*math.pi/6), math.sin(3 * 2*math.pi/6)],
            [math.cos(4 * 2*math.pi/6), math.sin(4 * 2*math.pi/6)],
            [math.cos(5 * 2*math.pi/6), math.sin(5 * 2*math.pi/6)]
        ])
    elif out_dim == 4 and hidden_dim == 3:
        return torch.tensor([
            [1.0, 0.0, -math.sqrt(0.5)],
            [-1.0, 0.0, -math.sqrt(0.5)],
            [0.0, 1.0, math.sqrt(0.5)],
            [0.0, -1.0, math.sqrt(0.5)]
        ])
    elif out_dim == 6 and hidden_dim == 3:
        # Return the 6 vertices of a regular octahedron in 3D.
        return torch.tensor([
            [1.0,  0.0,  0.0],
            [-1.0, 0.0,  0.0],
            [0.0,  1.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0,  1.0],
            [0.0,  0.0, -1.0]
        ])
    else:
        raise ValueError("Equiangular frame not implemented for (out_dim={}, hidden_dim={}).".format(out_dim, hidden_dim))

def initialize_output_weights(W, out_dim, hidden_dim):
    """
    Initializes the weight matrix W using an equiangular frame if available;
    otherwise falls back on orthogonal initialization.
    """
    try:
        eq_frame = equiangular_frame(out_dim, hidden_dim)
        W.data.copy_(eq_frame.to(W.device).type_as(W))
    except ValueError:
        nn.init.orthogonal_(W)

# ----------------------------
# Unified GNN Model Class
# ----------------------------
class GNNModel(nn.Module):
    def __init__(self, model_type="GCN", in_dim=3, hidden_dims=[3, 3], out_dim=3, freeze_final=True):
        """
        Constructs a flexible GNN model.
        
        Parameters:
          model_type (str): Choose "GCN" or "GIN" to decide which convolution type to use.
          in_dim (int): Input dimension for node features.
          hidden_dims (list of int): List of hidden layer dimensions.
          out_dim (int): Number of output features (e.g. number of classes or target dimensions).
          freeze_final (bool): If True, freezes the weight (but not the bias) of the final linear layer.
        """
        super(GNNModel, self).__init__()
        self.model_type = model_type

        if self.model_type == "GCN":
            self.convs = nn.ModuleList()
            prev_dim = in_dim
            # Build GCNConv layers
            for hdim in hidden_dims:
                self.convs.append(GCNConv(prev_dim, hdim, add_self_loops=True))
                prev_dim = hdim

        elif self.model_type == "GIN":
            self.convs = nn.ModuleList()
            prev_dim = in_dim
            # Build GINConv layers, each wrapping an MLP.
            for hdim in hidden_dims:
                mlp = nn.Sequential(
                    nn.Linear(prev_dim, hdim),
                    nn.ReLU(),
                    nn.Linear(hdim, hdim)
                )
                self.convs.append(GINConv(mlp, train_eps=True))
                prev_dim = hdim

        else:
            raise ValueError("Unsupported model_type. Choose 'GCN' or 'GIN'.")

        # Final linear layer (applied after global pooling)
        self.lin_out = nn.Linear(prev_dim, out_dim, bias=True)
        # Initialize weights using our custom initializer.
        initialize_output_weights(self.lin_out.weight, out_dim, hidden_dims[-1])

        if freeze_final:
            # Freeze only the weight so that during Phase 1 training it remains fixed.
            self.lin_out.weight.requires_grad = False
            if self.lin_out.bias is not None:
                self.lin_out.bias.requires_grad = True

    def forward(self, x, edge_index, batch):
        """
        Forward pass: Applies the convolutional layers, then global mean pooling,
        then the final linear layer to produce raw logits.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        graph_repr = global_mean_pool(x, batch)
        logits = self.lin_out(graph_repr)
        return logits

    def get_graph_repr(self, x, edge_index, batch):
        """
        Returns the graph-level representation (after global pooling).
        Useful for analysis and visualization.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return global_mean_pool(x, batch)

    def get_hidden_embeddings(self, x, edge_index, batch):
        """
        Returns the node-level hidden embeddings (output from the final conv layer).
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


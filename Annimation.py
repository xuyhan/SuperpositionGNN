import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Slider, Button

# PyG imports
from torch_geometric.data import Data

# -----------------------------
# Import model and data generator classes
# -----------------------------
from Model import GNNModel  # This file defines both GCN and GIN variants.
from GraphGeneration import SyntheticGraphDataGenerator  # Contains the motif generator.

# -----------------------------
# Helper functions
# -----------------------------
def load_experiment_results(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def choose_best_model_key(data):
    """
    Chooses the best model key from the summary (lowest average loss).
    Assumes data["summary"] maps a key string to a list where the first element is avg_loss.
    """
    summary = data["summary"]
    best_key = None
    best_loss = float('inf')
    for k, v in summary.items():
        avg_loss = v[0]  # take the first element as the average loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_key = k
    return best_key

def assign_weights_from_dict(model, weight_dict):
    """
    Assigns weights from the dictionary to the model.
    The weight_dict keys must match the model.state_dict() keys.
    """
    state_dict = model.state_dict()
    for key, value in weight_dict.items():
        if key in state_dict:
            # Convert list to tensor using the same type and device.
            tensor_value = torch.tensor(value, dtype=state_dict[key].dtype, device=state_dict[key].device)
            state_dict[key].copy_(tensor_value)
        else:
            print(f"Warning: Key {key} not found in model state_dict.")
    model.load_state_dict(state_dict)

# -----------------------------
# Graph construction for "simple" mode
# -----------------------------
def linear_graph(vectors):
    """
    Constructs a simple linear graph (with self-loops) from a list of vectors.
    Returns: PyG Data, a NetworkX graph, and node positions.
    """
    num_nodes = len(vectors)
    x = torch.tensor(vectors, dtype=torch.float)
    edges = []
    # Add self-loops and chain edges.
    for i in range(num_nodes):
        edges.append((i, i))
        if i > 0:
            edges.append((i, i - 1))
            edges.append((i - 1, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.zeros(num_nodes, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    # Use a simple path graph (which does not have self-loops) for visualization.
    G = nx.path_graph(num_nodes)
    pos = nx.spring_layout(G, seed=42)
    return data, G, pos

# -----------------------------
# Compute hidden embeddings layer-by-layer
# -----------------------------
def compute_hidden_embeddings(model, data):
    """
    Computes node hidden embeddings for each conv layer.
    For GIN layers, note that the returned value is the final output after the internal MLP.
    """
    x = data.x.clone()
    embeddings_by_layer = [x.detach().cpu().numpy()]
    for i, conv in enumerate(model.convs):
        x = conv(x, data.edge_index)
        if i < len(model.convs) - 1:
            x = F.relu(x)
        embeddings_by_layer.append(x.detach().cpu().numpy())
    return embeddings_by_layer

# -----------------------------
# Animation code: show arrows for 2D embeddings (except for layer 0)
# -----------------------------
def animate_graph(embeddings_by_layer, G, default_pos):
    """
    Animates the evolution of node embeddings.
    
    For layer 0 (input features), text annotations are shown.
    For later layers, if the embeddings are 2D, an arrow (in red) is drawn for each node.
    The arrow originates from the fixed node position (default_pos) and is scaled for visibility.
    
    Self-loops are not drawn.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)
    
    def draw_graph(current_layer):
        ax.clear()
        # Draw graph edges and nodes using the fixed layout.
        nx.draw_networkx_edges(G, default_pos, ax=ax)
        nx.draw_networkx_nodes(G, default_pos, ax=ax, node_color='lightblue', node_size=500)
        ax.set_title(f"Layer {current_layer}")
        ax.axis('off')
        
        layer_emb = embeddings_by_layer[current_layer]
        # For the first layer, we show text annotations.
        if current_layer == 0:
            for i, (pos_x, pos_y) in default_pos.items():
                text_str = "(" + ", ".join(f"{float(v):.2f}" for v in layer_emb[i]) + ")"
                ax.text(pos_x, pos_y, text_str, fontsize=9, ha='right', va='bottom')
        else:
            # If the embeddings are 2D, draw an arrow for each node.
            if layer_emb.ndim == 2 and layer_emb.shape[1] == 2:
                scale = 0.03  # Scale factor for arrow length.
                for i, (pos_x, pos_y) in default_pos.items():
                    vec = layer_emb[i]
                    dx = vec[0] * scale
                    dy = vec[1] * scale
                    ax.arrow(pos_x, pos_y, dx, dy, head_width=0.05, head_length=0.1, fc='red', ec='red')
            else:
                # Otherwise, fallback to text annotations.
                for i, (pos_x, pos_y) in default_pos.items():
                    text_str = "(" + ", ".join(f"{float(v):.2f}" for v in layer_emb[i]) + ")"
                    ax.text(pos_x, pos_y, text_str, fontsize=9, ha='right', va='bottom')
    
    draw_graph(0)
    num_layers = len(embeddings_by_layer)
    
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    layer_slider = Slider(slider_ax, 'Layer', 0, num_layers - 1, valinit=0, valstep=1)
    
    def update(val):
        current_layer = int(layer_slider.val)
        draw_graph(current_layer)
        fig.canvas.draw_idle()
    
    layer_slider.on_changed(update)
    
    button_ax = plt.axes([0.85, 0.025, 0.1, 0.04])
    next_button = Button(button_ax, 'Next')
    
    def next_layer(event):
        current = int(layer_slider.val)
        new_val = (current + 1) % num_layers
        layer_slider.set_val(new_val)
    next_button.on_clicked(next_layer)
    
    plt.show()

# -----------------------------
# Main script: auto-detect mode and GNN type from file path
# -----------------------------
def main():
    # Example file path; change this value as needed.
    file_path = "experiment_results/exp_count_GCN_3cats_2hidden_20250310_111122.json"
    
    # Automatically detect mode and GNN type from the file path.
    mode = "motif" if "motif" in file_path.lower() else ("count" if "count" in file_path.lower() else 
                    ("simple" if "simple" in file_path.lower() else "unknown"))
    gnn_type = "GIN" if "gin" in file_path.lower() else "GCN"
    print("Detected mode:", mode)
    print("Detected GNN type:", gnn_type)
    
    # Load experiment results and select the best model.
    data_json = load_experiment_results(file_path)
    best_model_key = choose_best_model_key(data_json)
    print("Selected best model key:", best_model_key)
    weight_dict = data_json["model summary"][best_model_key]
    
    # Get experiment configuration.
    exp_config = data_json["experiment_config"]
    # For motif mode, we assume constant node features (dim=1) and output is motif_dim.
    if mode == "motif":
        in_dim = exp_config.get("in_dim", 1)
        out_dim = exp_config.get("motif_dim", 3)
    elif mode == "simple":
        in_dim = exp_config["in_dim"]
        out_dim = exp_config["num_categories"] 
    elif mode == "count":
        in_dim = exp_config.get("in_dim", 1)
        out_dim = exp_config.get("num_categories", 3)
    
    # Instantiate the model with the detected GNN type.
    model = GNNModel(model_type=gnn_type,
                     in_dim=in_dim,
                     hidden_dims=exp_config["hidden_dims"],
                     out_dim=out_dim,
                     freeze_final=True,
                     pooling=exp_config["pooling"])
    
    # Load weights.
    assign_weights_from_dict(model, weight_dict)
    model.eval()
    
    # Create a graph.
    if mode == "motif":
        # Use the SyntheticGraphDataGenerator for motif graphs.
        generator = SyntheticGraphDataGenerator(
            mode="motif",
            num_categories=exp_config.get("num_categories", 3),
            p=exp_config.get("p", 0.25),
            num_nodes=exp_config.get("num_nodes", 20),
            chain_length_min=exp_config.get("chain_length_min", 2),
            chain_length_max=exp_config.get("chain_length_max", 7),
            motif_dim=exp_config.get("motif_dim", 3)
        )
        data = generator.generate_single_graph()
        # Convert the edge index to a NetworkX graph.
        G = nx.Graph()
        edge_index = data.edge_index.numpy()
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
        # Remove self-loops.
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        # Use a spring layout as a default.
        default_pos = nx.spring_layout(G, seed=42)
    elif mode == "simple":
        # For simple mode, use a predefined linear graph.
        vectors = ((0, 0, 0, 0),
                   (0, 0, 0, 0), 
                   (1, 0, 0, 0), 
                   (1, 0, 0, 0), 
                   (0, 0, 0, 0), 
                   (0, 0, 0, 0))
        data, G, default_pos = linear_graph(vectors)
    elif mode == "count":
        # Use the SyntheticGraphDataGenerator for count graphs.
        generator = SyntheticGraphDataGenerator(
            mode="count",
            num_nodes=exp_config.get("num_nodes", 20),
            p_count=exp_config.get("p_count", 0.9)
        )
        data = generator.generate_single_graph()
        # Convert the edge index to a NetworkX graph.
        G = nx.Graph()
        edge_index = data.edge_index.numpy()
        edges = list(zip(edge_index[0], edge_index[1]))
        G.add_edges_from(edges)
        # Remove self-loops.
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        # Use a spring layout as a default.
        default_pos = nx.spring_layout(G, seed=42)
    
    # Compute the hidden embeddings layer-by-layer.
    embeddings_by_layer = compute_hidden_embeddings(model, data)
    
    # Animate the evolution of node embeddings.
    animate_graph(embeddings_by_layer, G, default_pos)

if __name__ == "__main__":
    main()
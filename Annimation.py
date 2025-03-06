import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Slider, Button
from sklearn.decomposition import PCA

# Import your GNN model.
from Model import GNNModel  # Adjust the import according to your file structure

# PyG imports
from torch_geometric.data import Data

# -----------------------------
# Helper functions
# -----------------------------
def load_experiment_results(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def choose_best_model_key(data):
    """
    Choose the best model key from the summary (lowest average loss).
    Assumes data["summary"] maps a key string to [avg_loss, std_loss].
    """
    summary = data["summary"]
    best_key = None
    best_loss = float('inf')
    for k, (avg_loss, _) in summary.items():
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_key = k
    return best_key
    

def assign_weights_from_dict(model, weight_dict):
    """
    Given a GNN model and a dictionary of weights (as stored in the JSON file),
    assign the weights to the model. Assumes the keys in weight_dict match
    the keys of model.state_dict() (or at least the conv and linear layer parameters).
    """
    state_dict = model.state_dict()
    for key, value in weight_dict.items():
        if key in state_dict:
            # Convert the value (list) into a tensor with the proper type and device.
            tensor_value = torch.tensor(value, dtype=state_dict[key].dtype, device=state_dict[key].device)
            state_dict[key].copy_(tensor_value)
        else:
            print(f"Warning: Key {key} not found in model state_dict.")
    model.load_state_dict(state_dict)

# -----------------------------
# Function for toy graph generation
# -----------------------------

def linear_graph(vectors):
    """
    Creates a linear chain graph from a list or array of vectors.
    
    Parameters:
        vectors (list or np.ndarray): A list or array of vectors. Each vector is assigned to a node.
    
    Returns:
        data (torch_geometric.data.Data): Graph data containing node features and edge index.
        G (networkx.Graph): A NetworkX graph representing the same linear chain.
        pos (dict): A dictionary of positions for nodes (useful for visualization).
    """

    # Determine number of nodes and convert vectors to a tensor.
    num_nodes = len(vectors)
    x = torch.tensor(np.array(vectors), dtype=torch.float)
    
    # Create edge_index for a linear chain: (0 -> 1, 1 -> 2, ..., (n-2) -> (n-1))
    if num_nodes > 1:
        edge_list = [[i, i+1] for i in range(num_nodes - 1)]
        # Convert to tensor and transpose to shape [2, num_edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # For an undirected graph, add reverse edges.
        reverse_edge_index = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Assume all nodes belong to a single graph.
    batch = torch.zeros(num_nodes, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Create a NetworkX graph for visualization.
    G = nx.path_graph(num_nodes)
    # Use a spring layout for positioning (or choose a layout that suits a linear graph).
    pos = nx.spring_layout(G, seed=42)
    
    return data, G, pos


# -----------------------------
# Compute hidden embeddings using message passing
# -----------------------------
def compute_hidden_embeddings(model, data):
    """
    Performs a forward pass through the GNN's conv layers (with ReLU in between)
    to compute the node hidden embeddings layer-by-layer.
    Returns a list of embeddings (numpy arrays) for each layer (starting with input features).
    """
    x = data.x.clone()
    embeddings_by_layer = [x.detach().cpu().numpy()]
    
    # Pass through each convolution layer
    for i, conv in enumerate(model.convs):
        x = conv(x, data.edge_index)
        if i < len(model.convs) - 1:
            x = F.relu(x)
        embeddings_by_layer.append(x.detach().cpu().numpy())
        
    return embeddings_by_layer

# -----------------------------
# Animation with interactive slider
# -----------------------------
def animate_graph(embeddings_by_layer, G, pos):
    """
    Creates an interactive animation showing the test graph with node embeddings.
    Each node is drawn as a dot with its embedding printed next to it.
    A slider and a "Next" button let the user step through the layers.
    """
    num_layers = len(embeddings_by_layer)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)

    # Draw graph edges and nodes.
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
    
    # Create text annotations for each node (using full embedding as tuple)
    texts = {}
    for n, (xx, yy) in pos.items():
        arr = embeddings_by_layer[0][n]
        # Build a string with exactly two decimal places per value:
        text_str = "(" + ", ".join(f"{float(v):.2f}" for v in arr) + ")"  # <-- CHANGED
        texts[n] = ax.text(xx, yy, text_str, fontsize=9, ha='right', va='bottom')
    
    ax.set_title("Layer 0")
    ax.axis('off')

    # Slider setup.
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    layer_slider = Slider(slider_ax, 'Layer', 0, num_layers - 1, valinit=0, valstep=1)

    def update(val):
        layer = int(layer_slider.val)
        ax.set_title(f"Layer {layer}")
        for n in G.nodes():
            arr = embeddings_by_layer[layer][n]
            # Again, format each element to two decimals:
            text_str = "(" + ", ".join(f"{float(v):.2f}" for v in arr) + ")"  # <-- CHANGED
            texts[n].set_text(text_str)
        fig.canvas.draw_idle()

    layer_slider.on_changed(update)

    # "Next" button to step layers.
    button_ax = plt.axes([0.85, 0.025, 0.1, 0.04])
    next_button = Button(button_ax, 'Next')

    def next_layer(event):
        current = int(layer_slider.val)
        new_val = (current + 1) % num_layers
        layer_slider.set_val(new_val)
    next_button.on_clicked(next_layer)

    plt.show()

# -----------------------------
# Main script
# -----------------------------
def main():
    # --- Load the trained model weights from file ---
    file_path = "experiment_results/exp_simple_GCN_4cats_2hidden_20250306_151331.json"  
    data_json = load_experiment_results(file_path)
    
    # Choose the best model key from the JSON file.
    best_model_key = choose_best_model_key(data_json)
    print("Selected best model:", best_model_key)
    
    # Extract the weight dictionary.
    weight_dict = data_json["model summary"][best_model_key]
    
    # --- Instantiate the GNN model using experiment configuration ---
    exp_config = data_json["experiment_config"]
    model = GNNModel(model_type=exp_config["model_type"],
                     in_dim=exp_config["in_dim"],
                     hidden_dims=exp_config["hidden_dims"],
                     out_dim=exp_config["num_categories"] + exp_config.get("motif_dim", 0),
                     freeze_final=True)
    
    # Load the weights from the JSON file into the model.
    assign_weights_from_dict(model, weight_dict)
    model.eval()
    
    # --- Create a test graph ---
    # vectors all 4 dimensional
    vectors = ((0, 0, 0, 0),
               (0, 0, 0, 0), 
               (1, 0, 0, 0), 
               (1, 0, 0, 0), 
               (0, 0, 0, 0), 
               (0, 0, 0, 0))

    data, G, pos = linear_graph(vectors)

    
    # --- Compute hidden embeddings layer-by-layer using message passing ---
    embeddings_by_layer = compute_hidden_embeddings(model, data)
    
    # --- Animate the evolution of node embeddings ---
    animate_graph(embeddings_by_layer, G, pos)

main()
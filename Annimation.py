import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# Global visualization parameters
# -----------------------------
FONT_SIZE = 24  # Set this to adjust all text and tick sizes
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE
})

# PyG imports
from torch_geometric.data import Data

# -----------------------------
# Import model and data generator classes
# -----------------------------
from Model import GNNModel
from GraphGeneration import SyntheticGraphDataGenerator

# -----------------------------
# Helper functions
# -----------------------------
def load_experiment_results(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def choose_best_model_key(data):
    summary = data["summary"]
    best_key = None
    best_loss = float('inf')
    for k, v in summary.items():
        avg_loss = v[0]
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_key = k
    return best_key


def assign_weights_from_dict(model, weight_dict):
    state_dict = model.state_dict()
    for key, value in weight_dict.items():
        if key in state_dict:
            tensor_value = torch.tensor(value, dtype=state_dict[key].dtype, device=state_dict[key].device)
            state_dict[key].copy_(tensor_value)
        else:
            print(f"Warning: Key {key} not found in model state_dict.")
    model.load_state_dict(state_dict)

# -----------------------------
# Graph construction for "simple" mode
# -----------------------------
def linear_graph(vectors):
    num_nodes = len(vectors)
    x = torch.tensor(vectors, dtype=torch.float)
    edges = []
    for i in range(num_nodes):
        edges.append((i, i))
        if i > 0:
            edges.append((i, i - 1))
            edges.append((i - 1, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.zeros(num_nodes, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    G = nx.path_graph(num_nodes)
    pos = nx.spring_layout(G, seed=42)
    return data, G, pos

# -----------------------------
# Compute hidden embeddings layer-by-layer
# -----------------------------
def compute_hidden_embeddings(model, data):
    x = data.x.clone()
    embeddings_by_layer = [x.detach().cpu().numpy()]
    for i, conv in enumerate(model.convs):
        x = conv(x, data.edge_index)
        if i < len(model.convs) - 1:
            x = F.relu(x)
        embeddings_by_layer.append(x.detach().cpu().numpy())
    return embeddings_by_layer

# -----------------------------
# Plot all layers in subplots for a single graph
# -----------------------------
def plot_all_layers(embeddings_by_layer, G, default_pos, ax_row=None, scale=0.01):
    num_layers = len(embeddings_by_layer)
    if ax_row is None:
        fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))
        if num_layers == 1:
            axes = [axes]
    else:
        axes = ax_row
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        emb = embeddings_by_layer[layer_idx]
        nx.draw_networkx_edges(G, default_pos, ax=ax)
        nx.draw_networkx_nodes(G, default_pos, ax=ax, node_color='gray', node_size=150)
        ax.set_title(f"L{layer_idx}")
        ax.axis('off')
        if emb.ndim == 2 and emb.shape[1] == 2:
            for i, (x_pos, y_pos) in default_pos.items():
                dx, dy = emb[i] * scale
                ax.arrow(x_pos, y_pos, dx, dy, head_width=0.05, head_length=0.1, fc='black', ec='black', linewidth=3)

# -----------------------------
# Plot grid for count mode: one row per graph type
# -----------------------------
def plot_count_mode_grid(model, generator):
    sample = generator.generate_single_graph()
    num_classes = sample.y.shape[-1]
    collected = {}
    while len(collected) < num_classes:
        data = generator.generate_single_graph()
        label = int(torch.argmax(data.y).item())
        if label not in collected:
            collected[label] = data
    embeddings_list, Gs, poses = [], [], []
    for label in sorted(collected.keys()):
        data = collected[label]
        embeddings = compute_hidden_embeddings(model, data)
        embeddings_list.append(embeddings)
        G = nx.Graph()
        idxs = data.edge_index.numpy()
        G.add_edges_from(list(zip(idxs[0], idxs[1])))
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        pos = nx.spring_layout(G, seed=42)
        Gs.append(G)
        poses.append(pos)
    num_rows = len(embeddings_list)
    num_layers = len(embeddings_list[0])
    fig, axes = plt.subplots(num_rows, num_layers, figsize=(4 * num_layers, 4 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)
    if num_layers == 1:
        axes = np.expand_dims(axes, 1)
    for r in range(num_rows):
        for c in range(num_layers):
            graph_type = ["More Red", "More Blue", "Same Red and Blue"]
            ax = axes[r, c]
            emb = embeddings_list[r][c]
            plot_all_layers([emb], Gs[r], poses[r], ax_row=[ax])
            ax.set_title(f"{graph_type[r]}, GNN Layer {c}")
    plt.tight_layout()
    plt.savefig("count_mode_grid.png", dpi=300)
    plt.show()

# -----------------------------
# Plot grid for motif mode: one row per motif type + pooled vector arrow
# -----------------------------
def plot_motif_mode_grid(model, generator, pooled_head_widths=None):
    # Default arrow head widths for each motif row
    if pooled_head_widths is None:
        pooled_head_widths = [0.1, 0.2, 0.3]

    scales = [0.1, 0.4, 0.01]
    motif_name = ["Triangle", "Square", "Pentagon"]

    sample = generator.generate_single_graph()
    num_types = sample.y.shape[-1]
    collected = {}
    while len(collected) < num_types:
        data = generator.generate_single_graph()
        label = int(torch.argmax(data.y).item())
        if label not in collected:
            collected[label] = data

    embeddings_list, Gs, poses = [], [], []
    for label in sorted(collected.keys()):
        data = collected[label]
        embeddings = compute_hidden_embeddings(model, data)
        embeddings_list.append(embeddings)
        G = nx.Graph()
        idxs = data.edge_index.numpy()
        G.add_edges_from(list(zip(idxs[0], idxs[1])))
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        pos = nx.spring_layout(G, seed=42)
        Gs.append(G)
        poses.append(pos)

    num_rows = len(embeddings_list)
    num_layers = len(embeddings_list[0])
    fig, axes = plt.subplots(
        num_rows, num_layers + 1,
        figsize=(4 * (num_layers + 1), 4 * num_rows)
    )
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    for r in range(num_rows):
        for c in range(num_layers):
            ax = axes[r, c]
            emb = embeddings_list[r][c]
            plot_all_layers([emb], Gs[r], poses[r], ax_row=[ax], scale=scales[c])
            ax.set_title(f"{motif_name[r]}, Layer {c}")

        # Pooled vector arrow in last column
        pooled = embeddings_list[r][-1].mean(axis=0)
        ax_pool = axes[r, num_layers]
        ax_pool.axhline(0, color='black', linewidth=2)
        ax_pool.axvline(0, color='black', linewidth=2)

        # Draw arrow with custom head width
        hw = pooled_head_widths[r]
        ax_pool.arrow(
            0, 0, pooled[0], pooled[1],
            head_width=hw, head_length=0.1,
            fc='black', ec='black',
            linewidth=2, width=0.01,
            length_includes_head=True
        )

        lim = max(abs(pooled[0]), abs(pooled[1])) * 1.2
        ax_pool.set_xlim(-lim, lim)
        ax_pool.set_ylim(-lim, lim)
        ax_pool.set_aspect('equal')
        ax_pool.set_title(f"{motif_name[r]}, Pooled")

    plt.tight_layout()
    plt.savefig("count_mode_grid.png", dpi=300)
    plt.show()
# -----------------------------
# Main script: auto-detect mode and GNN type
# -----------------------------
def main():
    file_path = "experiment_results/motif/GIN/mean/2/exp_motif_GIN_0cats_2hidden_20250507_162229.json" 
    # file_path = "experiment_results/evo/count/GCN/mean/2/exp_count_GCN_3cats_2hidden_20250507_151931.json"
    # file_path = "experiment_results/evo/count/GIN/mean/2/exp_count_GIN_3cats_2hidden_20250507_151241.json"
    mode = (
        "motif" if "motif" in file_path.lower() else
        ("count" if "count" in file_path.lower() else
        ("simple" if "simple" in file_path.lower() else "unknown"))
    )
    gnn_type = "GIN" if "gin" in file_path.lower() else "GCN"
    print("Detected mode:", mode)
    print("Detected GNN type:", gnn_type)
    data_json = load_experiment_results(file_path)
    best_key = choose_best_model_key(data_json)
    print("Selected best model key:", best_key)
    weight_dict = data_json["model summary"][best_key]
    exp_config = data_json["experiment_config"]
    if mode == "motif":
        in_dim = exp_config.get("in_dim", 1)
        out_dim = exp_config.get("motif_dim", 3)
    elif mode == "simple":
        in_dim = exp_config["in_dim"]
        out_dim = exp_config["num_categories"]
    elif mode == "count":
        in_dim = exp_config.get("in_dim", 1)
        out_dim = exp_config.get("num_categories", 3)
    model = GNNModel(
        model_type=gnn_type,
        in_dim=in_dim,
        hidden_dims=exp_config["hidden_dims"],
        out_dim=out_dim,
        freeze_final=True,
        pooling=exp_config["pooling"]
    )
    assign_weights_from_dict(model, weight_dict)
    model.eval()
    if mode == "motif":
        generator = SyntheticGraphDataGenerator(
            mode="motif",
            num_categories=exp_config.get("num_categories", 3),
            p=exp_config.get("p", 0.25),
            num_nodes=exp_config.get("num_nodes", 20),
            chain_length_min=exp_config.get("chain_length_min", 2),
            chain_length_max=3, # For plotting purposes
            motif_dim=exp_config.get("motif_dim", 3)
        )
        plot_motif_mode_grid(model, generator)
    elif mode == "simple":
        vectors = ((0, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        data, G, default_pos = linear_graph(vectors)
        embeddings_by_layer = compute_hidden_embeddings(model, data)
        plot_all_layers(embeddings_by_layer, G, default_pos)
    elif mode == "count":
        generator = SyntheticGraphDataGenerator(
            mode="count",
            num_nodes=exp_config.get("num_nodes", 20),
            p_count=exp_config.get("p_count", 0.9)
        )
        plot_count_mode_grid(model, generator)

if __name__ == "__main__":
    main()
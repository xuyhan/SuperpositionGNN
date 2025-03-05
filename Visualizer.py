import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import networkx as nx
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
import itertools

class Visualizer:
    @staticmethod
    def plot_avg_hidden_embeddings_3d(avg_embeddings, colors, markers, keys_to_plot):
        """
        Plots average hidden embeddings as vectors in 3D.
        
        Parameters:
          avg_embeddings (dict): Mapping from a target key (e.g. a tuple) to a 3D torch.Tensor.
          colors (list): List of colors, one per key.
          markers (list): List of marker symbols, one per key.
          keys_to_plot (list): The list of keys (targets) to plot.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        origin = [0, 0, 0]
        
        for idx, key in enumerate(keys_to_plot):
            if key not in avg_embeddings:
                continue
            vec = avg_embeddings[key]
            if vec.size(0) != 3:
                print(f"Cannot plot target {key} in 3D; hidden_dim != 3.")
                continue
            x, y, z = vec.tolist()
            ax.quiver(*origin, x, y, z, color=colors[idx], arrow_length_ratio=0.1, linewidth=2, label=str(key))
            ax.scatter(x, y, z, color=colors[idx], marker=markers[idx], s=100)
            ax.text(x, y, z, str(key), size=12, zorder=1, color='k')
        
        pts = [avg_embeddings[k].tolist() for k in keys_to_plot if k in avg_embeddings]
        if pts:
            pts_tensor = torch.tensor(pts)
            max_val = torch.abs(pts_tensor).max().item() + 0.1
            ax.set_xlim([-max_val, max_val])
            ax.set_ylim([-max_val, max_val])
            ax.set_zlim([-max_val, max_val])
        
        ax.set_title('Average Hidden Embeddings (Pure Only) - 3D')
        ax.set_xlabel('Hidden Dim 1')
        ax.set_ylabel('Hidden Dim 2')
        ax.set_zlabel('Hidden Dim 3')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Targets')
        plt.show()

    @staticmethod
    def plot_avg_hidden_embeddings_2d(avg_embeddings, colors, markers, keys_to_plot):
        """
        Plots average hidden embeddings as vectors in 2D.
        
        Parameters:
          avg_embeddings (dict): Mapping from a target key (e.g. a tuple) to a 2D torch.Tensor.
          colors (list): List of colors.
          markers (list): List of markers.
          keys_to_plot (list): Keys (targets) to plot.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        origin = [0, 0]
        
        for idx, key in enumerate(keys_to_plot):
            if key not in avg_embeddings:
                continue
            vec = avg_embeddings[key]
            if vec.size(0) != 2:
                print(f"Cannot plot target {key} in 2D; hidden_dim != 2.")
                continue
            x, y = vec.tolist()
            ax.quiver(*origin, x, y, angles='xy', scale_units='xy', scale=1,
                      color=colors[idx], width=0.005, label=str(key))
            ax.scatter(x, y, color=colors[idx], marker=markers[idx], s=100)
            ax.text(x, y, str(key), size=12, zorder=1, color='k')
        
        pts = [avg_embeddings[k].tolist() for k in keys_to_plot if k in avg_embeddings]
        if pts:
            pts_tensor = torch.tensor(pts)
            max_val = torch.abs(pts_tensor).max().item() + 0.5
            ax.set_xlim([-max_val, max_val])
            ax.set_ylim([-max_val, max_val])
        
        ax.set_title('Average Hidden Embeddings (Pure Only) - 2D')
        ax.set_xlabel('Hidden Dim 1')
        ax.set_ylabel('Hidden Dim 2')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Targets')
        ax.grid(True)
        plt.show()

    @staticmethod
    def plot_graph_with_embeddings(model, sample_data_list, avg_embeddings, specified_types, colors, markers):
        """
        Visualizes sample graphs with nodes colored according to their final hidden embeddings.
        
        For each graph:
          - Node embeddings are extracted via model.get_hidden_embeddings.
          - Cosine similarity is computed between each node embedding and the average embedding for each type.
          - Nodes are colored according to the type with the highest similarity.
          - Node sizes are scaled based on the norm (activation strength) of the node embedding.
        
        Parameters:
          model: A trained GNN model that provides a get_hidden_embeddings() method.
          sample_data_list (list): List of PyG Data objects representing graphs.
          avg_embeddings (dict): Dictionary mapping a target type to its average hidden embedding (torch.Tensor).
          specified_types (list): Ordered list of target types (keys) to consider.
          colors (list): List of colors corresponding to each specified type.
          markers (list): List of markers corresponding to each specified type.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        with torch.no_grad():
            n_samples = len(sample_data_list)
            if n_samples > 1:
                fig, axs = plt.subplots(1, n_samples, figsize=(5 * n_samples, 5))
            else:
                fig, axs = plt.subplots(figsize=(6, 6))
                axs = [axs]

            for i, data in enumerate(sample_data_list):
                data = data.to(device)
                # Obtain node-level embeddings.
                node_embeddings = model.get_hidden_embeddings(data.x, data.edge_index, data.batch)
                num_nodes, hidden_dim = node_embeddings.size()

                # Normalize node embeddings.
                node_norm = node_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
                node_embeddings_normed = node_embeddings / node_norm
                avg_matrix = torch.stack([avg_embeddings[gtype] for gtype in specified_types]).to(device)
                avg_norm = avg_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
                avg_matrix_normed = avg_matrix / avg_norm

                similarities = torch.matmul(node_embeddings_normed, avg_matrix_normed.t())
                pred_classes = torch.argmax(similarities, dim=1)

                strength = node_embeddings.norm(dim=1)
                G = nx.Graph()
                edges = data.edge_index.t().cpu().numpy()
                edges = [edge for edge in edges if edge[0] != edge[1]]
                G.add_edges_from(edges)
                pos = nx.spring_layout(G, seed=42)
                node_colors = [colors[pred_classes[i].item()] for i in range(num_nodes)]
                strength = strength.cpu().numpy()
                scaler = MinMaxScaler()
                strength_scaled = scaler.fit_transform(strength.reshape(-1, 1)).flatten() * 1500

                ax = axs[i]
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=0.5)
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=strength_scaled, ax=ax, alpha=0.7)
                ax.set_title(f"Graph {i+1}")
                ax.axis('off')

                for j in range(num_nodes):
                    ax.text(pos[j][0], pos[j][1], str(pred_classes[j].item()), size=8,
                            horizontalalignment='center', verticalalignment='center')

                custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[j], markersize=10)
                                for j in range(len(specified_types))]
                ax.legend(custom_lines, [f"Type {specified_types[j]}" for j in range(len(specified_types))],
                          loc='upper right', title='Graph Types')
            plt.tight_layout()
            plt.show()


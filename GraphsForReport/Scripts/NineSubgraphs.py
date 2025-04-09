import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Only needed for 3D plots

def plot_vectors_in_subplot(ax, vectors_dict):
    """
    Plot 2D or 3D vectors in the provided subplot axis (ax),
    without labeling axes. The scaling of the vector geometry is preserved.
    
    Parameters:
      ax (matplotlib.axes.Axes or Axes3D): The subplot axis.
      vectors_dict (dict): Dictionary with keys as labels and values as lists (2D or 3D vectors).
    """
    # Determine the dimensionality from the first vector
    first_vec = next(iter(vectors_dict.values()))
    dim = len(first_vec)
    
    cmap = plt.get_cmap("tab10")
    labels = list(vectors_dict.keys())
    origin = np.zeros(dim)
    
    if dim == 2:
        for i, label in enumerate(labels):
            vec = np.array(vectors_dict[label])
            ax.quiver(origin[0], origin[1],
                      vec[0], vec[1],
                      angles='xy', scale_units='xy', scale=1,
                      color=cmap(i), width=0.005)
        # Set axis limits based on vector values
        all_vals = np.array(list(vectors_dict.values()))
        min_val = all_vals.min() - 0.3
        max_val = all_vals.max() + 0.3
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        # Force equal aspect ratio so vector geometry is preserved
        ax.set_aspect('equal')
        # Remove ticks and labels for a clean look
        ax.set_xticks([])
        ax.set_yticks([])
        
    elif dim == 3:
        for i, label in enumerate(labels):
            vec = np.array(vectors_dict[label])
            ax.quiver(origin[0], origin[1], origin[2],
                      vec[0], vec[1], vec[2],
                      arrow_length_ratio=0.1, color=cmap(i), linewidth=2)
        all_vals = np.array(list(vectors_dict.values()))
        min_val = all_vals.min() - 1
        max_val = all_vals.max() + 1
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_zlim([min_val, max_val])
        # For 3D, equal aspect ratio is more challenging;
        # we remove ticks to avoid misinterpretation.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        raise ValueError("Vectors must be either 2D or 3D.")


def plot_lines_from_folder(ax, folder_path, experiment_label, metric_name="Loss"):
    """
    Plot all CSV files in a given folder on one subplot. Each CSV file is expected
    to contain 'Step' and 'Value' columns. All lines in the subplot correspond to
    the same experiment (folder) but are plotted with unique colors.
    
    Parameters:
      ax (matplotlib.axes.Axes): The subplot axis to draw on.
      folder_path (str): Path to the folder containing CSV files.
      experiment_label (str): Label to use for identifying the experiment.
      metric_name (str): Y-axis label (e.g., "Loss").
    """
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        raise ValueError(f"No CSV files found in folder: {folder_path}")
    
    cmap = plt.get_cmap("tab10")
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        epochs = df["Step"]
        values = df["Value"]
        ax.plot(epochs, values, label=f"{experiment_label} - file {i+1}", linewidth=2, color=cmap(i))
    
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True)

def create_combined_figure(vectors_list, csv_folders, csv_labels, metric_name="Loss", output_file="combined_figure.pdf"):
    """
    Create a 3x3 grid of subplots:
      - Top 6 subplots display vector plots from the provided list of dictionaries.
      - Bottom 3 subplots display line plots, each corresponding to one folder containing CSV files.
    
    The overall figure is adjusted to have a wide aspect ratio (roughly twice as wide as high)
    so that it fits nicely as a figure in a paper, without altering the intrinsic vector scaling.
    
    Parameters:
      vectors_list (list of dict): List of 6 dictionaries with vector data.
      csv_folders (list of str): List of 3 folder paths, each containing CSV files.
      csv_labels (list of str): List of 3 labels corresponding to each folder.
      metric_name (str): Metric name for the y-axis in the line plots.
      output_file (str): File name for the output PDF.
    """
    if len(vectors_list) != 6:
        raise ValueError("Please provide exactly 6 dictionaries for the vector plots.")
    if len(csv_folders) != 3 or len(csv_labels) != 3:
        raise ValueError("Please provide exactly 3 CSV folder paths and 3 corresponding labels.")
    
    # Set the figure to be twice as wide as it is high.
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 8))
    
    # First two rows: vector plots
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        plot_vectors_in_subplot(ax, vectors_list[i])
    
    # Last row: line plots from CSV folders
    for idx in range(3):
        ax = axes[2, idx]
        plot_lines_from_folder(ax, csv_folders[idx], csv_labels[idx], metric_name=metric_name)
    
    plt.tight_layout()
    plt.savefig(output_file, format="pdf")
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Example vector dictionaries for the first 6 subplots (replace with your actual data)
    vectors_example = [
        {
            "(1, 0, 0)": [0.9102323055267334, -0.28037720918655396],
            "(0, 1, 0)": [0.008074425160884857, 0.9957887530326843],
            "(0, 0, 1)": [-0.740456759929657, -0.3839344382286072]
        },
        {
            "(0, 0, 1, 0)": [-1.1671184301376343, -0.06306133419275284],
            "(0, 0, 0, 1)": [0.0007617459050379694, -1.1354484558105469],
            "(0, 1, 0, 0)": [0.08999386429786682, 0.8271757364273071],
            "(1, 0, 0, 0)": [1.0018384456634521, -0.23353418707847595]
        },
        {
            "(0, 0, 0, 0, 1)": [0.02192610688507557, -2.0216991901397705],
            "(0, 0, 0, 1, 0)": [-1.173049807548523, -0.26479917764663696],
            "(0, 0, 1, 0, 0)": [0.17389777302742004, -0.2835731506347656],
            "(1, 0, 0, 0, 0)": [0.8847442269325256, -0.3996647894382477],
            "(0, 1, 0, 0, 0)": [0.14500993490219116, 1.0995112657546997]
        },
        {
            "(0, 0, 1)": [-0.33713048696517944, 0.5424337387084961],
            "(0, 1, 0)": [0.898219108581543, 1.3243781328201294],
            "(1, 0, 0)": [0.8060863018035889, -0.3235540986061096]
        },
        {
            "(0, 0, 1, 0)": [-0.9523501992225647, -0.05801386013627052],
            "(1, 0, 0, 0)": [0.48521849513053894, -0.9395773410797119],
            "(0, 0, 0, 1)": [-0.024101126939058304, -0.26713791489601135],
            "(0, 1, 0, 0)": [0.6478540897369385, 0.811893105506897]
        },
        {
            "(0, 0, 0, 0, 1)": [0.02192610688507557, -2.0216991901397705],
            "(0, 0, 0, 1, 0)": [-1.173049807548523, -0.26479917764663696],
            "(0, 0, 1, 0, 0)": [0.17389777302742004, -0.2835731506347656],
            "(1, 0, 0, 0, 0)": [0.8847442269325256, -0.3996647894382477],
            "(0, 1, 0, 0, 0)": [0.14500993490219116, 1.0995112657546997]
        },
    ]
    
    # List of 3 folders, each containing CSV files for one experiment.
    script_dir = os.path.dirname(__file__)
    csv_folder1 = os.path.join(script_dir, "..", "Data", "NinePlots", "3")
    csv_folder2 = os.path.join(script_dir, "..", "Data", "NinePlots", "4")
    csv_folder3 = os.path.join(script_dir, "..", "Data", "NinePlots", "5")
    csv_folders = [os.path.normpath(csv_folder1), os.path.normpath(csv_folder2), os.path.normpath(csv_folder3)]
    
    # Labels corresponding to the 3 experiments/folders
    csv_labels = ["Input dim. 3", "Input dim. 4", "Input dim. 5"]
    
    # Output file path for the combined PDF
    output_pdf = os.path.join(script_dir, "..", "Graphs", "NinePlots", "combined_figure.pdf")
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    
    create_combined_figure(
        vectors_list=vectors_example,
        csv_folders=csv_folders,
        csv_labels=csv_labels,
        metric_name="Loss",
        output_file=output_pdf
    )
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Only needed for 3D plots


# --------------------------------------------------------------------------
#  Vector‑plot helper
# --------------------------------------------------------------------------
def plot_vectors_in_subplot(ax, vectors_dict):
    """
    Draw one set of 2‑D or 3‑D arrows on a single axis (ax) with no ticks
    and preserving aspect ratio.
    """
    first_vec = next(iter(vectors_dict.values()))
    dim       = len(first_vec)

    cmap   = plt.get_cmap("tab10")
    labels = list(vectors_dict.keys())
    origin = np.zeros(dim)

    if dim == 2:
        for i, label in enumerate(labels):
            vec = np.array(vectors_dict[label])
            ax.quiver(origin[0], origin[1],
                      vec[0], vec[1],
                      angles="xy", scale_units="xy", scale=1,
                      color=cmap(i), width=0.005)

        vals    = np.array(list(vectors_dict.values()))
        padding = 0.3
        ax.set_xlim(vals.min() - padding, vals.max() + padding)
        ax.set_ylim(vals.min() - padding, vals.max() + padding)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    elif dim == 3:
        for i, label in enumerate(labels):
            vec = np.array(vectors_dict[label])
            ax.quiver(origin[0], origin[1], origin[2],
                      vec[0], vec[1], vec[2],
                      arrow_length_ratio=0.1,
                      color=cmap(i), linewidth=2)

        vals    = np.array(list(vectors_dict.values()))
        padding = 1
        ax.set_xlim(vals.min() - padding, vals.max() + padding)
        ax.set_ylim(vals.min() - padding, vals.max() + padding)
        ax.set_zlim(vals.min() - padding, vals.max() + padding)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    else:
        raise ValueError("Vectors must be either 2‑D or 3‑D.")


# --------------------------------------------------------------------------
#  CSV‑loss‑curve helper
# --------------------------------------------------------------------------
def plot_lines_from_folder(ax, folder_path, experiment_label, metric_name="Loss"):
    """
    Plot all CSVs in *folder_path* (must have ‘Step’ and ‘Value’ cols) on ax.
    """
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}")

    cmap = plt.get_cmap("tab10")
    for i, csv_file in enumerate(csv_files):
        label_file = ["Top config", "Bottom config"]
        df = pd.read_csv(csv_file)
        ax.plot(df["Step"], df["Value"],
                label=f"{label_file[i]}",
                linewidth=2, color=cmap(i))

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True)


# --------------------------------------------------------------------------
#  NEW: combined figure with a single “shift” knob for the top rows
# --------------------------------------------------------------------------
def create_combined_figure(vectors_list,
                           csv_folders,
                           csv_labels,
                           metric_name="Loss",
                           vector_annotations=None,
                           output_file="combined_figure.pdf",
                           top_horizontal_shift=0.03):
    """
    Draw a 3×3 grid (2 rows of vectors + 1 row of loss curves).

    Parameters
    ----------
    vectors_list : list[dict]            – 6 dictionaries with vector coords
    csv_folders  : list[str]             – 3 folders with loss CSVs
    csv_labels   : list[str]             – labels for those folders
    vector_annotations : list[dict]      – 6 dicts with
        {'num_features', 'num_active_features', 'final_loss'}
    top_horizontal_shift : float
        How far to move the **top six** axes horizontally
        (fraction of total figure width; positive → right, negative → left)
    """
    if len(vectors_list) != 6:
        raise ValueError("Need exactly 6 vector dictionaries.")
    if len(csv_folders) != 3 or len(csv_labels) != 3:
        raise ValueError("Need exactly 3 CSV folders and 3 labels.")
    if vector_annotations is None or len(vector_annotations) != 6:
        raise ValueError("Need exactly 6 annotation dicts.")

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 8))

    # --- top 2 rows: vectors + labels ---------------------------------------
    for i in range(6):
        row, col = divmod(i, 3)
        ax       = axes[row, col]

        # slide ax horizontally while keeping width and height unchanged
        pos     = ax.get_position()
        ax.set_position([pos.x0 + top_horizontal_shift,
                         pos.y0,
                         pos.width,
                         pos.height])

        plot_vectors_in_subplot(ax, vectors_list[i])

        ann   = vector_annotations[i]
        label = (f"Num features: {ann['num_features']}\n"
                 f"Num active features: {ann['num_active_features']}\n"
                 f"Final loss: {ann['final_loss']}")
        ax.text(1.02, 0.5, label,
                transform=ax.transAxes,
                ha="left", va="center", fontsize=10)

    # --- bottom row: loss curves --------------------------------------------
    for idx in range(3):
        plot_lines_from_folder(axes[2, idx],
                               csv_folders[idx],
                               csv_labels[idx],
                               metric_name=metric_name)

    plt.tight_layout()
    # plt.savefig(output_file, format="pdf")
    plt.show()


# --------------------------------------------------------------------------
#  Example usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # ---------- dummy vectors (replace with real data) ----------
    vectors_example = [
        {"(1, 0, 0)": [0.91, -0.28],
         "(0, 1, 0)": [0.01, 0.99],
         "(0, 0, 1)": [-0.74, -0.38]},
        {"(0, 0, 1, 0)": [-1.17, -0.06],
         "(0, 0, 0, 1)": [0.00, -1.13],
         "(0, 1, 0, 0)": [0.09, 0.83],
         "(1, 0, 0, 0)": [1.00, -0.23]},
        {"(0, 0, 0, 0, 1)": [0.50806558, -2.10432434],
         "(0, 0, 0, 1, 0)": [-1.51411092, -0.84883428],
         "(0, 0, 1, 0, 0)": [-0.71743095, 1.29326510],
         "(0, 1, 0, 0, 0)": [0.73846961, 2.54762864],
         "(1, 0, 0, 0, 0)": [2.05495167, 0.02076192]},
        {"(0, 0, 1)": [-0.34, 0.54],
         "(0, 1, 0)": [0.90, 1.32],
         "(1, 0, 0)": [0.81, -0.32]},
        {"(0, 0, 1, 0)": [-0.95, -0.06],
         "(1, 0, 0, 0)": [0.49, -0.94],
         "(0, 0, 0, 1)": [-0.02, -0.27],
         "(0, 1, 0, 0)": [0.65, 0.81]},
        {"(0, 0, 0, 0, 1)": [0.02, -2.02],
         "(0, 0, 0, 1, 0)": [-1.17, -0.26],
         "(0, 0, 1, 0, 0)": [0.17, -0.28],
         "(1, 0, 0, 0, 0)": [0.88, -0.40],
         "(0, 1, 0, 0, 0)": [0.15, 1.10]},
    ]

    # ---------- annotation blocks ----------
    vector_annotations = [
        {"num_features": 3, "num_active_features": 3, "final_loss": 0.398},
        {"num_features": 4, "num_active_features": 4, "final_loss": 0.550},
        {"num_features": 5, "num_active_features": 5, "final_loss": 0.569},
        {"num_features": 3, "num_active_features": 3, "final_loss": 0.856},
        {"num_features": 4, "num_active_features": 3, "final_loss": 0.743},
        {"num_features": 5, "num_active_features": 4, "final_loss": 0.790},
    ]

    # ---------- CSV folders & labels (adjust paths as needed) ----------
    script_dir  = os.path.dirname(__file__)
    csv_folders = [os.path.join(script_dir, "..", "Data", "NinePlots", d)
                   for d in ("3", "4", "5")]
    csv_labels  = ["Input dim. 3", "Input dim. 4", "Input dim. 5"]

    output_pdf  = os.path.join(script_dir, "..", "Graphs", "NinePlots",
                               "combined_figure.pdf")
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    # ---------- call the plotter ----------
    create_combined_figure(
        vectors_list=vectors_example,
        csv_folders=csv_folders,
        csv_labels=csv_labels,
        metric_name="Loss",
        vector_annotations=vector_annotations,
        output_file=output_pdf,
        top_horizontal_shift=0.0   # tweak this until alignment looks perfect
    )
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

#########################################
# Functions for CSV reading and plotting
#########################################

def read_csv_file(filepath):
    """
    Read the CSV file (using pandas) and return a DataFrame.
    Expects columns "Step" and "Value".
    """
    return pd.read_csv(filepath)


def read_all_files_from_folder(folder, file_pattern="*.csv"):
    """
    Reads all CSV files from the given folder (sorted order).
    """
    files = sorted(glob.glob(os.path.join(folder, file_pattern)))
    if not files:
        raise ValueError(f"No files matching {file_pattern} found in folder {folder}")
    return [read_csv_file(f) for f in files]


def plot_multiline_from_folder(ax, folder, file_pattern="*.csv", metric_name="Metric"):
    """
    Reads all CSV files from a folder and plots each file's data (Step vs. Value)
    on the given axis, filtered to epoch <= 50, all in black.
    """
    data_frames = read_all_files_from_folder(folder, file_pattern)
    for i, df in enumerate(data_frames):
        df = df[df["Step"] <= 50]
        ax.plot(
            df["Step"],
            df["Value"],
            label=f"SV {i+1}",
            linewidth=2,
            color="black"
        )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} vs. Epoch", fontsize=14)
    ax.set_xlim(0, 50)
    ax.grid(True)

#########################################
# Main: Two Plots in One Figure
#########################################
if __name__ == "__main__":
    # Set your two folder paths here:
    folder_sv_nonleaky = os.path.normpath(os.path.join(
        "GraphsForReport", "Data", "LeakyReLU", "NonLeaky"))
    folder_sv_leaky = os.path.normpath(os.path.join(
        "GraphsForReport", "Data", "LeakyReLU", "Leaky"))
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(ncols=2, figsize=(18, 6))
    fig.suptitle("Evolution of Rank (Leaky vs NonLeaky ReLU)", fontsize=16, y=0.97)
    
    # Plot each folder’s CSVs in its own subplot, epoch<=50
    plot_multiline_from_folder(axes[0], folder_sv_nonleaky, metric_name="Rank")
    axes[0].set_title("Non-Leaky", fontsize=14)
    
    plot_multiline_from_folder(axes[1], folder_sv_leaky, metric_name="Rank")
    axes[1].set_title("Leaky", fontsize=14)

    # Synchronize y-axis limits across both subplots
    y_lims = [ax.get_ylim() for ax in axes]
    min_y = min(lim[0] for lim in y_lims)
    max_y = max(lim[1] for lim in y_lims)
    for ax in axes:
        ax.set_ylim(min_y, max_y)
    
    plt.tight_layout()
    plt.show()

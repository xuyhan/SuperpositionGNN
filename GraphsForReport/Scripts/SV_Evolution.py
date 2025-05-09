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
    
    Parameters:
      folder (str): Folder path.
      file_pattern (str): Pattern to match CSV files.
      
    Returns:
      list of DataFrame: List containing the DataFrame for each CSV file.
    """
    files = sorted(glob.glob(os.path.join(folder, file_pattern)))
    if not files:
        raise ValueError(f"No files matching {file_pattern} found in folder {folder}")
    return [read_csv_file(f) for f in files]

def plot_multiline_from_folder(ax, folder, file_pattern="*.csv", metric_name="Metric"):
    """
    Reads all CSV files from a folder and plots each file's data (Step vs. Value)
    on the given axis, all in black.
    
    Parameters:
      ax (matplotlib.axes.Axes): Axis to plot on.
      folder (str): Folder path containing the CSV files.
      file_pattern (str): Pattern for file matching (default "*.csv").
      metric_name (str): Name for the metric (for y-axis label).
    """
    data_frames = read_all_files_from_folder(folder, file_pattern)
    for i, df in enumerate(data_frames):
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
    ax.grid(True)

#########################################
# Main: Three Plots in One Figure
#########################################
if __name__ == "__main__":
    # Set your three folder paths here:
    folder_sv6  = os.path.normpath(os.path.join("GraphsForReport", "Data", "SV_Evolution", "6"))
    folder_sv10 = os.path.normpath(os.path.join("GraphsForReport", "Data", "SV_Evolution", "10"))
    folder_sv14 = os.path.normpath(os.path.join("GraphsForReport", "Data", "SV_Evolution", "14"))
    
    # Create figure with three subplots side by side
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
    fig.suptitle("Evolution of Singular Values", fontsize=16, y=0.97)
    
    # Plot each folder’s CSVs in its own subplot, all lines black
    plot_multiline_from_folder(axes[0], folder_sv6,  metric_name="Singular Values")
    axes[0].set_title("Feature number = 6",  fontsize=14)
    
    plot_multiline_from_folder(axes[1], folder_sv10, metric_name="Singular Values")
    axes[1].set_title("Feature number = 10", fontsize=14)
    
    plot_multiline_from_folder(axes[2], folder_sv14, metric_name="Singular Values")
    axes[2].set_title("Feature number = 14", fontsize=14)
    
    plt.tight_layout()
    plt.show()
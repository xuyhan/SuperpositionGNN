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

def read_single_file_from_folder(folder, file_pattern="*.csv"):
    """
    Reads the first CSV file in the folder (sorted order).
    
    Parameters:
      folder (str): Folder path.
      file_pattern (str): Pattern to match CSV files.
      
    Returns:
      DataFrame: Contents of the CSV file.
    """
    files = sorted(glob.glob(os.path.join(folder, file_pattern)))
    if not files:
        raise ValueError(f"No files matching {file_pattern} found in folder {folder}")
    return read_csv_file(files[0])

def read_all_files_from_folder(folder, file_pattern="*.csv"):
    """
    Reads all CSV files from the given folder.
    
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

#########################################
# Plot functions
#########################################
def plot_loss_accuracy_dual(ax, folder_loss, folder_accuracy, file_pattern="*.csv", loss_label="Loss", acc_label="Accuracy"):
    """
    Plots two lines (loss and accuracy) on the same x-axis using a dual y-axis.
    
    Parameters:
      ax (matplotlib.axes.Axes): Primary axis for plotting the loss.
      folder_loss (str): Folder containing the CSV file for loss.
      folder_accuracy (str): Folder containing the CSV file for accuracy.
      file_pattern (str): Pattern for file matching (default "*.csv").
      loss_label (str): Label for the loss line (left axis).
      acc_label (str): Label for the accuracy line (right axis).
    """
    # Read one file from each folder.
    df_loss = read_single_file_from_folder(folder_loss, file_pattern)
    df_acc  = read_single_file_from_folder(folder_accuracy, file_pattern)
    
    # Plot loss on the primary (left) y-axis.
    ax.plot(df_loss["Step"], df_loss["Value"], label=loss_label, linewidth=2, color="tab:blue")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(loss_label, fontsize=12, color="black")
    ax.tick_params(axis="y", labelcolor="black")
    
    # Create a second y-axis that shares the same x-axis.
    ax2 = ax.twinx()
    ax2.plot(df_acc["Step"], df_acc["Value"], label=acc_label, linewidth=2, color="tab:orange")
    ax2.set_ylabel(acc_label, fontsize=12, color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    
    # Optionally, add legends manually.
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10, loc="upper right")
    ax.grid(True)

def plot_multiline_from_folder(ax, folder, file_pattern="*.csv", metric_name="Metric"):
    """
    Reads all CSV files from a folder and plots each file's data (Step vs. Value)
    on the given axis.
    
    Parameters:
      ax (matplotlib.axes.Axes): Axis to plot on.
      folder (str): Folder path containing the CSV files.
      file_pattern (str): Pattern for file matching (default "*.csv").
      metric_name (str): Name for the metric (for y-axis label).
    """
    data_frames = read_all_files_from_folder(folder, file_pattern)
    for i, df in enumerate(data_frames):
        # Use file index in the label; you could also use os.path.basename of the file.
        ax.plot(df["Step"], df["Value"], label=f"File {i+1}", linewidth=2, color="black")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} vs. Epoch", fontsize=14)
    ax.grid(True)
    ax.set_yscale("log")

#########################################
# Main Example Usage: Two Graphs in One Figure
#########################################
if __name__ == "__main__":
    # Set folder paths (you can use relative or absolute paths)
    # For the dual-axis plot: one folder for loss and one folder for accuracy
    folder_loss = os.path.normpath(os.path.join("GraphsForReport", "Data", "Rank", "Stuck", "Loss"))
    folder_accuracy = os.path.normpath(os.path.join("GraphsForReport", "Data", "Rank", "Stuck", "Accuracy"))
    
    # For the multi-line plot: a folder with several CSV files (for example, additional metrics)
    folder_extra = os.path.normpath(os.path.join("GraphsForReport", "Data", "Rank", "Stuck", "SingularValues"))
    
    # Create a figure with two subplots (side by side)
    fig, axes = plt.subplots(ncols=2, figsize=(16, 6))
    
    # Left subplot: dual-axis plot (loss and accuracy)
    plot_loss_accuracy_dual(axes[0], folder_loss, folder_accuracy, file_pattern="*.csv", loss_label="Loss", acc_label="Accuracy")
    axes[0].set_title("Training Loss & Accuracy", fontsize=14)
    
    # Right subplot: multi-line plot from folder_extra
    plot_multiline_from_folder(axes[1], folder_extra, file_pattern="*.csv", metric_name="Singular Values")
    axes[1].set_title("Singular Values Evolution", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("combined_figure.pdf", format="pdf")
    plt.show()
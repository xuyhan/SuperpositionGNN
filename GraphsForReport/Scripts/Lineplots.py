import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_tensorboard_csvs(folder_path, labels, metric_name, output_file="figure.pdf"):
    """
    Plots TensorBoard CSV data on a single figure.
    
    Parameters:
      folder_path (str): Path to the folder containing CSV files.
      labels (list of str): A list of labels to assign to each CSV file.
      metric_name (str): The name of the quality/metric being plotted (used for y-axis label).
      output_file (str): The file name for the saved figure.
    """
    # Retrieve and sort all CSV files in the folder
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    
    if len(csv_files) != len(labels):
        raise ValueError("Number of CSV files does not match the number of provided labels.")
    
    # Set up a high-quality figure (adjust figsize as needed)
    plt.figure(figsize=(8, 6))
    plt.style.use("seaborn-whitegrid")  # Use a clean style for scientific figures
    
    # Loop over CSV files and labels, and plot the data
    for csv_file, label in zip(csv_files, labels):
        # Read the CSV file using pandas
        df = pd.read_csv(csv_file)
        
        # Extract the relevant columns ("step" as epoch and "value" as the metric)
        epochs = df["Step"]
        values = df["Value"]
        
        # Plot the data with a thicker line for clarity
        plt.plot(epochs, values, label=label, linewidth=2)
    
    # Label the axes and title the figure
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.title(f"{metric_name} vs. Epoch", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the figure in a vector format (PDF) for publication quality
    plt.savefig(output_file, format="pdf")
    plt.show()

# Example usage:
if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)        
    folder = os.path.join(script_dir, '..', 'Data/LossLandscape') 
    
    # Normalize the path (optional, but makes it platform-independent)
    folder = os.path.normpath(folder)

    # Set the plot output folder (Folder4, relative to Folder2)
    plot_folder = os.path.join(script_dir, '..', 'Graphs/LossLandscape')
    plot_folder = os.path.normpath(plot_folder)
    
    # Create Folder4 if it doesn't exist
    os.makedirs(plot_folder, exist_ok=True)
    
    # Define the full output file path for the saved plot
    output_file = os.path.join(plot_folder, "figure.pdf")
    
    label_list = ["Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4"]
    metric = "Loss"  # Change to the metric you want to plot
    
    plot_tensorboard_csvs(folder, label_list, metric, output_file)
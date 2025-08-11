import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad


#########################################
# Functions for reading file data
#########################################
def build_folder_path(path_parts):
    """
    Takes an iterable of path components and returns an absolute path
    using os.path.join and os.path.abspath.

    Parameters:
      path_parts (iterable): Iterable of path components (e.g., ("..", "Graphs", "Folder1"))

    Returns:
      str: Absolute path
    """
    relative_path = os.path.join(*path_parts)
    return os.path.abspath(relative_path)


def read_array_from_folder(folder, file_pattern="*.txt"):
    """
    Reads a file matching file_pattern from the specified folder (provided
    as a relative path or a tuple of path components) and returns an array
    of numbers. The file is expected to have the form:

        [num1, num2, num3, ...]

    Parameters:
      folder (str or iterable): The folder path as a string or a tuple/list of strings.
      file_pattern (str): Pattern of file name(s) to search for. Default is "*.txt".

    Returns:
      np.ndarray: Array of numbers read from the file.
    """
    # If folder is provided as tuple/list, build an absolute path.
    if isinstance(folder, (tuple, list)):
        folder = build_folder_path(folder)
    else:
        folder = os.path.abspath(folder)

    files = glob.glob(os.path.join(folder, file_pattern))
    if not files:
        raise ValueError(f"No files matching {file_pattern} found in folder {folder}")

    # Open the first matching file and read its contents.
    with open(files[0], "r") as f:
        content = f.read().strip()

    # Remove the surrounding square brackets if they exist.
    if content.startswith('[') and content.endswith(']'):
        content = content[1:-1]

    # Split the content by commas and convert each piece to a float.
    try:
        arr = np.array([float(item.strip()) for item in content.split(',') if item.strip() != ''])
    except Exception as e:
        raise ValueError(f"Error parsing numbers from file {files[0]}: {e}")
    return arr


#########################################
# Functions for plotting histogram using KDE
#########################################
def plot_smoothed_normalized_histogram_from_folders(folders, file_pattern="*.txt", bw=0.3, ax=None):
    """
    For each folder, reads a file to get an array of numbers, computes a
    smoothed normalized histogram (using a Gaussian KDE), and plots the resulting curves.

    Parameters:
      folders (list): List of folder paths to process. Each folder may be a string
                      or an iterable of path components.
      file_pattern (str): Pattern for file matching in each folder.
      bw (float): Bandwidth multiplier for the Gaussian KDE.
      ax (matplotlib.axes.Axes, optional): An axis to plot on. If None, creates a new figure.
    """
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    cmap = plt.get_cmap("tab10")

    for i, folder in enumerate(folders):
        # Read the array from the file in the folder.
        data = read_array_from_folder(folder, file_pattern)

        # Compute the Gaussian KDE (which is normalized so that the area under the curve is 1).
        kde = gaussian_kde(data, bw_method=bw)
        x_min, x_max = data.min(), data.max()
        xs = np.linspace(x_min, x_max, 200)
        ys = kde(xs)

        # Construct a label from the folder path.
        if isinstance(folder, (tuple, list)):
            folder_path = build_folder_path(folder)
        else:
            folder_path = os.path.abspath(folder)
        label = os.path.basename(os.path.normpath(folder_path))

        ax.plot(xs, ys, label=f"Components for: p = {label}", linewidth=2, color=cmap(i))

    # Note: Axes labels, title, legend and grid are set in the combined plot function.


#########################################
# Functions for the normalized basis function
#########################################
def basis_function(x, n, a=1.0):
    """
    Compute the basis function (1-x^2)^((n-3)/2) for a given x and parameter n.
    """
    exponent = (n - 3) / 2.0
    return (1 - (x / a) ** 2) ** exponent


def normalized_basis(n, a, left_edge, right_edge, num_points=400):
    """
    Computes the normalized basis function over the interval [left_edge, right_edge].

    Parameters:
        n (int or float): Parameter in the function f(x) = (1-x^2)^((n-3)/2).
        left_edge (float): Left boundary of the domain.
        right_edge (float): Right boundary of the domain.
        num_points (int): Number of points used for plotting.

    Returns:
        xs (np.ndarray): The x values.
        ys (np.ndarray): The normalized function values so that the integral over [left_edge, right_edge] is 1.
    """
    xs = np.linspace(left_edge, right_edge, num_points)

    # Compute the area under the unnormalized function using numerical integration.
    area, _ = quad(basis_function, left_edge, right_edge, args=(n, a))

    # Normalize the function values.
    ys = basis_function(xs, n, a) / area
    return xs, ys


def plot_normalized_basis(n, a, left_edge, right_edge, ax=None):
    """
    Plots the normalized basis function on the specified axis.

    Parameters:
        n (int or float): Parameter in the function f(x) = (1-x^2)^((n-3)/2).
        left_edge (float): Left boundary of the domain.
        right_edge (float): Right boundary of the domain.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new figure.
    """
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    xs, ys = normalized_basis(n, a, left_edge, right_edge)
    ax.plot(xs, ys, lw=2, color="darkred", label=f"Components for: Randomly alligned")


#########################################
# Combined Plot on the Same GraphVa
#########################################
if __name__ == "__main__":
    # Example folder paths provided as tuples of path components.
    folders = [
        ("experiment_results", "FromCSD3", "all_elements", "1.0"),
        ("experiment_results", "FromCSD3", "all_elements", "4.0"),
        ("experiment_results", "FromCSD3", "all_elements", "16.0")
    ]

    # Set parameters for the normalized basis function.
    n = 5
    a = 2.0
    left_edge = -2.0  # For the basis function; adjust as necessary.
    right_edge = 2.0  # For the basis function.

    # Create a single figure and axis for both plots.
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the KDE smoothed normalized histograms from the data files.
    plot_smoothed_normalized_histogram_from_folders(folders, file_pattern="*.txt", bw=0.3, ax=ax)

    # Plot the normalized basis function on the same axis.
    plot_normalized_basis(n, a, left_edge, right_edge, ax=ax)

    # Set common labels, title, grid, and legend.
    ax.set_xlabel("Component Value", fontsize=18)
    ax.set_ylabel("Normalized Frequency", fontsize=18)
    ax.set_title("Distribution of Embedding Vector Components (5D)", fontsize=22)
    ax.grid(True)
    ax.legend(fontsize=16)

    plt.tight_layout()
    plt.show()
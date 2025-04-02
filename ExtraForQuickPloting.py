import matplotlib.pyplot as plt
import numpy as np

vectors = {
            "(0, 1, 0, 0, 0)": [
                -0.6953477263450623,
                1.6549696922302246
            ],
            "(0, 0, 0, 0, 1)": [
                -0.7391722798347473,
                -1.9371166229248047
            ],
            "(1, 0, 0, 0, 0)": [
                1.5772455930709839,
                -0.8344478011131287
            ],
            "(0, 0, 1, 0, 0)": [
                -0.6943793892860413,
                -0.8380476236343384
            ],
            "(0, 0, 0, 1, 0)": [
                -1.985925555229187,
                -0.8117438554763794
            ]
}

def plot_vectors(vectors):
    """
    Plot a dictionary of vectors, automatically handling 2D or 3D.
    
    Parameters:
        vectors (dict): A dictionary where keys are labels and values are lists or arrays.
                        All vectors must be either 2D or 3D.
    """
    # Determine the dimension by looking at the first vector in the dictionary.
    any_vector = next(iter(vectors.values()))
    dim = len(any_vector)
    
    if dim not in (2, 3):
        raise ValueError("Vectors must be 2D or 3D.")
    
    # Use a colormap to assign different colors automatically.
    cmap = plt.get_cmap("tab10")
    labels = list(vectors.keys())
    
    if dim == 2:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        origin = np.array([0, 0])
        
        for i, label in enumerate(labels):
            vec = np.array(vectors[label])
            ax.quiver(origin[0], origin[1], vec[0], vec[1],
                      angles='xy', scale_units='xy', scale=1,
                      color=cmap(i), label=label, width=0.005)
        
        # Determine axis limits.
        all_values = np.array(list(vectors.values()))
        min_val = all_values.min() - 1
        max_val = all_values.max() + 1
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D Vector Plot")
        ax.legend()
        ax.grid(True)
        plt.show()
    
    elif dim == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        origin = np.array([0, 0, 0])
        
        for i, label in enumerate(labels):
            vec = np.array(vectors[label])
            ax.quiver(origin[0], origin[1], origin[2],
                      vec[0], vec[1], vec[2],
                      arrow_length_ratio=0.1, color=cmap(i), label=label, linewidth=2)
        
        # Determine axis limits.
        all_values = np.array(list(vectors.values()))
        min_val = all_values.min() - 1
        max_val = all_values.max() + 1
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_zlim([min_val, max_val])
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Vector Plot")
        ax.legend()
        plt.show()


# Call the function with either 2D or 3D vectors:
plot_vectors(vectors)

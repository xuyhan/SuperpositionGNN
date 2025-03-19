import matplotlib.pyplot as plt
import numpy as np

vectors = {
            "(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)": [
                2.4889931678771973,
                -1.4148719310760498,
                0.2939084768295288
            ],
            "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0)": [
                0.5458265542984009,
                -0.8263402581214905,
                -0.9894341230392456
            ],
            "(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)": [
                -2.125486135482788,
                -3.2531015872955322,
                -2.6660633087158203
            ],
            "(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)": [
                0.8043816089630127,
                -1.1592965126037598,
                -1.1499987840652466
            ],
            "(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)": [
                2.590268135070801,
                -2.324655294418335,
                -2.204134941101074
            ],
            "(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)": [
                0.6464748978614807,
                -1.067886233329773,
                -1.1657023429870605
            ],
            "(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)": [
                0.7676016092300415,
                -1.1398062705993652,
                -1.2780572175979614
            ],
            "(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)": [
                2.867479085922241,
                1.708311915397644,
                -2.187119483947754
            ],
            "(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)": [
                0.7632727026939392,
                -1.2494500875473022,
                -1.130967140197754
            ],
            "(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)": [
                0.7213864326477051,
                -4.487110137939453,
                0.19688323140144348
            ],
            "(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)": [
                -0.6435221433639526,
                0.9690316915512085,
                -4.369259834289551
            ],
            "(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)": [
                0.6677554845809937,
                -1.0449868440628052,
                -1.025333046913147
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

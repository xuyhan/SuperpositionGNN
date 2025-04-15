import numpy as np
import matplotlib.pyplot as plt

def random_unit_vector(n):
    """
    Generate a random unit vector of dimension n.
    """
    vec = np.random.normal(0, 1, n)
    return vec / np.linalg.norm(vec)

def one_hot_vector(n):
    """
    Generate a one-hot vector of dimension n.
    Randomly choose one index to be 1, others 0.
    """
    vec = np.zeros(n)
    index = np.random.randint(0, n)
    vec[index] = 1.0
    return vec

def compute_angle(v1, v2):
    """
    Compute the angle between two vectors v1 and v2.
    v1 is assumed to be a unit vector.
    """
    dot_product = np.dot(v1, v2)
    norm_v2 = np.linalg.norm(v2)
    # Clip cosine to valid range to avoid numerical issues
    cos_theta = np.clip(dot_product / norm_v2, -1, 1)
    return np.arccos(cos_theta)

# Define the range of dimensions (n) and s values (std deviation for Gaussian)
dims = range(2, 21)  # Dimensionalities from 2 to 20
s_values = np.linspace(0.001, 0.35, 50)  # 50 values between 0.001 and 0.2 for s

n_runs = 100  # Number of experiments for each (n, s) combination

# Arrays to store the average angles
results_random = np.zeros((len(s_values), len(dims)))
results_onehot = np.zeros((len(s_values), len(dims)))

# Loop over each combination of s and n
for i, s in enumerate(s_values):
    for j, n in enumerate(dims):
        angles_random = []
        angles_onehot = []
        for _ in range(n_runs):
            # --- Case 1: Original vector is random ---
            orig_random = random_unit_vector(n)
            new_random = np.empty(n)
            for k in range(n):
                gaussian_list = np.random.normal(0, s, 20)
                max_gaussian = max(gaussian_list, key=abs)  # number with max abs value
                if abs(orig_random[k]) >= abs(max_gaussian):
                    new_random[k] = orig_random[k]
                else:
                    new_random[k] = max_gaussian
            angle_random = compute_angle(orig_random, new_random)
            angles_random.append(angle_random)
            
            # --- Case 2: Original vector is one-hot ---
            orig_onehot = one_hot_vector(n)
            new_onehot = np.empty(n)
            for k in range(n):
                gaussian_list = np.random.normal(0, s, 20)
                max_gaussian = max(gaussian_list, key=abs)
                if abs(orig_onehot[k]) >= abs(max_gaussian):
                    new_onehot[k] = orig_onehot[k]
                else:
                    new_onehot[k] = max_gaussian
            angle_onehot = compute_angle(orig_onehot, new_onehot)
            angles_onehot.append(angle_onehot)
        
        # Average the angles over n_runs experiments.
        results_random[i, j] = np.mean(angles_random)
        results_onehot[i, j] = np.mean(angles_onehot)

# Determine global min and max for the color scale
global_min = min(results_random.min(), results_onehot.min())
global_max = max(results_random.max(), results_onehot.max())

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Adjust space on the right so the color bar doesn't overlap the second subplot
fig.subplots_adjust(right=0.85)

# Plot for random unit vectors, with common vmin/vmax
im1 = ax1.imshow(results_random, aspect='auto', origin='lower',
                 extent=[min(dims), max(dims), min(s_values), max(s_values)],
                 vmin=global_min, vmax=global_max)
ax1.set_title('Average Angle Change: Random Unit Vector')
ax1.set_xlabel('Dimensionality (n)')
ax1.set_ylabel('Standard Deviation (s)')

# Plot for one-hot vectors, also with common vmin/vmax
im2 = ax2.imshow(results_onehot, aspect='auto', origin='lower',
                 extent=[min(dims), max(dims), min(s_values), max(s_values)],
                 vmin=global_min, vmax=global_max)
ax2.set_title('Average Angle Change: One-Hot Vector')
ax2.set_xlabel('Dimensionality (n)')
ax2.set_ylabel('Standard Deviation (s)')

# Add a new set of axes for the color bar on the right
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label('Average angle (radians)')

fig.suptitle('Comparison of Angle Changes after Max Pooling (Common Color Scale)', y=0.98)
plt.show()
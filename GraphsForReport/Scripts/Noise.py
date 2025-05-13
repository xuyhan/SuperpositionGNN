import numpy as np
import matplotlib.pyplot as plt

# --- (Re-run simulation for completeness) ---
def random_unit_vector(n):
    vec = np.random.normal(0, 1, n)
    return vec / np.linalg.norm(vec)

def one_hot_vector(n):
    vec = np.zeros(n)
    index = np.random.randint(0, n)
    vec[index] = 1.0
    return vec

def compute_angle(v1, v2):
    cos_theta = np.clip(np.dot(v1, v2) / np.linalg.norm(v2), -1, 1)
    return np.arccos(cos_theta)

dims = range(2, 21)
s_values = np.linspace(0.001, 0.35, 50)
n_runs = 100

results_random = np.zeros((len(s_values), len(dims)))
results_onehot = np.zeros((len(s_values), len(dims)))

for i, s in enumerate(s_values):
    for j, n in enumerate(dims):
        angles_r, angles_o = [], []
        for _ in range(n_runs):
            orig_r = random_unit_vector(n)
            new_r = np.array([max(np.random.normal(0, s, 20), key=abs) 
                              if abs(orig_r[k]) < max(np.random.normal(0, s, 20), key=abs) 
                              else orig_r[k] for k in range(n)])
            angles_r.append(compute_angle(orig_r, new_r))
            
            orig_o = one_hot_vector(n)
            new_o = np.array([max(np.random.normal(0, s, 20), key=abs) 
                              if abs(orig_o[k]) < max(np.random.normal(0, s, 20), key=abs) 
                              else orig_o[k] for k in range(n)])
            angles_o.append(compute_angle(orig_o, new_o))
        results_random[i, j] = np.mean(angles_r)
        results_onehot[i, j] = np.mean(angles_o)

vmin = min(results_random.min(), results_onehot.min())
vmax = max(results_random.max(), results_onehot.max())

# --- Plot with adjusted tick fontsize and suptitle layout ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(right=0.85, top=0.84)  # Increase top margin for suptitle

# Random unit vectors heatmap
im1 = ax1.imshow(results_random, aspect='auto', origin='lower',
                 extent=[min(dims), max(dims), min(s_values), max(s_values)],
                 vmin=vmin, vmax=vmax)
ax1.set_title('Random Unit Vector', fontsize=22)
ax1.set_xlabel('Dimensionality (n)', fontsize=18)
ax1.set_ylabel('Noise (s)', fontsize=18)
ax1.tick_params(axis='both', labelsize=16)

# One-hot vectors heatmap
im2 = ax2.imshow(results_onehot, aspect='auto', origin='lower',
                 extent=[min(dims), max(dims), min(s_values), max(s_values)],
                 vmin=vmin, vmax=vmax)
ax2.set_title('One-Hot Vector', fontsize=22)
ax2.set_xlabel('Dimensionality (n)', fontsize=18)
ax2.set_ylabel('Noise (s)', fontsize=18)
ax2.tick_params(axis='both', labelsize=16)

# Colorbar
cbar_ax = fig.add_axes([0.88, 0.13, 0.02, 0.7])
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label('Average change (radians)', fontsize=18)
cbar.ax.tick_params(labelsize=16)

# Adjust suptitle
fig.suptitle('Effect of Max Pooling on Vector Orientation', fontsize=24)
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Publication-quality settings tailored to 3.5x2.5" figure
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.3
})

# Parameters
separation = np.array([0, 0, 0.3])
num_vectors = 5
angle_x = np.radians(30)
angle_y = np.radians(45)

# Rotation matrices
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(angle_x), -np.sin(angle_x)],
    [0, np.sin(angle_x),  np.cos(angle_x)]
])
R_y = np.array([
    [np.cos(angle_y), 0, np.sin(angle_y)],
    [0, 1, 0],
    [-np.sin(angle_y), 0, np.cos(angle_y)]
])
R = R_y @ R_x

# Base vectors
angles = np.linspace(0, 2*np.pi, num_vectors, endpoint=False)
base_vectors = np.stack([np.cos(angles), np.sin(angles), np.zeros(num_vectors)], axis=1)

# Plot setup
fig = plt.figure(figsize=(3.5, 2.5), dpi=300)
ax = fig.add_subplot(111, projection="3d")
ax.tick_params(pad=3)
ax.grid(True, alpha=0.3)

# Plot arrows at reduced length
for sign, color in [(1, "C0"), (-1, "C1")]:
    for base in base_vectors:
        noised = base + sign * separation + np.random.uniform(-0.05, 0.05, size=3)
        vec = R @ noised
        ax.quiver(0, 0, 0, *vec,
                  length=0.7,
                  arrow_length_ratio=0.05,
                  linewidth=1.0,
                  color=color)

# Hyperplane
grid_lim = 1.0
grid = np.linspace(-grid_lim, grid_lim, 8)
X, Y = np.meshgrid(grid, grid)
Z = np.zeros_like(X)
coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
rot = R @ coords
Xr = rot[0].reshape(X.shape)
Yr = rot[1].reshape(X.shape)
Zr = rot[2].reshape(X.shape)
ax.plot_surface(Xr, Yr, Zr, alpha=0.2, color='gray', edgecolor='none')

# Legend via proxy artists
legend_handles = [
    Line2D([0], [0], color="C0", lw=1.5, label="Embeddings"),
    Line2D([0], [0], color="C1", lw=1.5, label="Output weights")
]
ax.legend(handles=legend_handles,
          loc="upper right",
          frameon=False,
          fontsize=8)

# Labels and limits
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)
ax.set_title("Hyperplane between embeddings and output weights",
             pad=2,
             fontsize=8)

# Camera angle
ax.view_init(elev=25, azim=30)

plt.tight_layout()
plt.show()
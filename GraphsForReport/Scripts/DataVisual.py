import matplotlib.pyplot as plt
import random

# ======================
# Left subplot: Example Graph → Target Vector
# ======================
# 1) Embeddings (10 nodes × 6 dims; nodes 2&3 share dim=2)
embedding_list = [
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,1,0,0,0],
    [0,0,1,0,0,0],  # shared 1
    [0,0,0,0,0,0],
    [1,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,1,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
]

num_nodes     = len(embedding_list)
embedding_dim = len(embedding_list[0])

# 2) Detect the shared adjacent pair dims
active_pairs = []
for i in range(num_nodes-1):
    for d in range(embedding_dim):
        if embedding_list[i][d] and embedding_list[i+1][d]:
            active_pairs += [(i, d), (i+1, d)]

# 3) Build the target vector
target = [0]*embedding_dim
for _, d in active_pairs:
    target[d] = 1

# 4) Create combined figure
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# -- Left axis --
ax = axes[0]
ax.axis('off')

# Draw edges + nodes
xs     = list(range(num_nodes))
y_node = 0.0
for i in range(num_nodes-1):
    ax.plot([xs[i], xs[i+1]], [y_node, y_node], 'k-', lw=1)
ax.scatter(xs, [y_node]*num_nodes, s=200, color='k', zorder=2)

# Embedding stack params
y_step  = 0.35
y_start = y_node + (embedding_dim-1)*y_step/2
x_off   = -0.3

# Draw each node's vertical embedding
for i, emb in enumerate(embedding_list):
    for d, bit in enumerate(emb):
        x = xs[i] + x_off
        y = y_start - d*y_step
        if (i, d) in active_pairs:
            c = 'C1'
        elif bit == 1:
            c = 'k'
        else:
            c = 'gray'
        ax.text(x, y, str(bit), ha='center', va='center',
                fontsize=14, color=c)

# Arrow → target
tx = xs[-1] + 1.5
ax.annotate(
    "",
    xy=(tx-0.2, y_node),            # arrow head
    xytext=(xs[-1]+0.2, y_node),    # arrow tail
    arrowprops=dict(arrowstyle="->", lw=2)
)

# Draw target vector
for d, bit in enumerate(target):
    y = y_start - d*y_step
    c = 'C1' if bit else 'gray'
    ax.text(tx, y, str(bit), ha='center', va='center',
            fontsize=14, color=c)

# Labels
label_y = y_start + y_step
ax.text(xs[num_nodes//2], label_y, "Pairwise Interaction Example Graph",
        ha='center', va='bottom', fontsize=16)
ax.text(tx, label_y, "Target Vector",
        ha='center', va='bottom', fontsize=16)

# Reapply original limits
half_h = (embedding_dim-1)*y_step/2
ax.set_xlim(xs[0] + x_off - 0.5, tx + 0.5)
ax.set_ylim(y_node - half_h - 0.5, y_node + half_h + 0.5)


# ======================
# Right subplot: Square Clique with Randomized Chains
# ======================
ax2 = axes[1]
ax2.axis('off')

# Central clique nodes (4 nodes in a square)
central_pos = [(0,0), (1,0), (1,1), (0,1)]

# Collect positions and edges
positions = central_pos.copy()
edges = []

# Add clique edges
for i in range(len(central_pos)):
    for j in range(i+1, len(central_pos)):
        edges.append((central_pos[i], central_pos[j]))

# Attach randomized 4-node chains (including the central node)
directions = [(0,-1), (1,0), (0,1), (-1,0)]
chain_length = 4  # total nodes in each chain, including the central node
jitter_scale = 0.3  # maximum jitter

for (cx, cy), (dx, dy) in zip(central_pos, directions):
    prev_pos = (cx, cy)
    for step in range(1, chain_length):
        # Ideal position along grid
        ideal = (cx + dx*step, cy + dy*step)
        # Apply random jitter
        jittered = (
            ideal[0] + random.uniform(-jitter_scale, jitter_scale),
            ideal[1] + random.uniform(-jitter_scale, jitter_scale)
        )
        edges.append((prev_pos, jittered))
        positions.append(jittered)
        prev_pos = jittered

# Draw edges and nodes
for p1, p2 in edges:
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=1)
xs2, ys2 = zip(*positions)
ax2.scatter(xs2, ys2, s=200, color='k', zorder=2)

# Label the right subplot
mid_x = sum(x for x, _ in central_pos) / 4
max_y = max(y for _, y in positions)
ax2.text(mid_x, max_y + 0.5, "Motifs Example Graph",
         ha='center', va='bottom', fontsize=16)

# Adjust spacing
fig.subplots_adjust(wspace=0.4)
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

# 1) Embeddings (10 nodes × 6 dims; nodes 3&4 share dim=2)
embedding_list = [
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,0,0,0],
    [0,0,1,0,0,0],
    [0,0,1,0,0,0],
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,0,1,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1],
]

num_nodes     = len(embedding_list)
embedding_dim = len(embedding_list[0])

# 2) Find the one shared dimension in the adjacent pair
active_pairs = []
for i in range(num_nodes-1):
    for d in range(embedding_dim):
        if embedding_list[i][d] and embedding_list[i+1][d]:
            active_pairs.append((i, d))

# 3) Build the target vector
target = [0]*embedding_dim
for _, d in active_pairs:
    target[d] = 1

# 4) Plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

xs     = list(range(num_nodes))
y_node = 0.0

# Draw edges and nodes
for i in range(num_nodes-1):
    ax.plot([xs[i], xs[i+1]], [y_node, y_node], 'k-', lw=1)
ax.scatter(xs, [y_node]*num_nodes, s=200, color='k', zorder=2)

# Vertical‐stack params
y_step  = 0.35
y_start = y_node + (embedding_dim-1)*y_step/2
x_off   = -0.5

# Draw embeddings
for i, emb in enumerate(embedding_list):
    for d, bit in enumerate(emb):
        x = xs[i] + x_off
        y = y_start - d*y_step
        if (i,d) in active_pairs:
            c = 'C1'
        elif bit == 1:
            c = 'k'
        else:
            c = 'gray'
        ax.text(x, y, str(bit), ha='center', va='center', fontsize=14, color=c)

# Arrow → target
tx = xs[-1] + 1.5
ax.annotate("", xy=(xs[-1]+0.2, y_node), xytext=(tx-0.2, y_node),
            arrowprops=dict(arrowstyle="->", lw=2))

# Draw target
for d, bit in enumerate(target):
    y = y_start - d*y_step
    c = 'C1' if bit else 'gray'
    ax.text(tx, y, str(bit), ha='center', va='center', fontsize=14, color=c)

# ─── THIS IS THE CRUCIAL PART ─────────────────────
# Expand the axis so all of your vertical stacks are inside the view!
half_h = (embedding_dim-1)*y_step/2
ax.set_xlim(xs[0] + x_off - 0.5, tx + 0.5)
ax.set_ylim(y_node - half_h - 0.5, y_node + half_h + 0.5)
# ──────────────────────────────────────────────────

plt.tight_layout()
plt.show()
import numpy as np
from itertools import combinations

# the full set of vectors
data = {"(1, 0, 0)":[
                0.9102323055267334,
                -0.28037720918655396
            ],
            "(0, 1, 0)": [
                0.008074425160884857,
                0.9957887530326843
            ],
            "(0, 0, 1)": [
                -0.740456759929657,
                -0.3839344382286072
            ]
        }

# convert to numpy arrays
vecs = {k: np.array(v) for k, v in data.items()}

# helper to compute angle between two vectors
def angle(u, v):
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# build a dict keyed by the unordered pair of vector-keys
angles = {
    frozenset((k1, k2)): angle(vecs[k1], vecs[k2])
    for k1, k2 in combinations(vecs, 2)
}

# manually identified “cluster” of 4 near-colinear vectors
cluster = [

]

# filter out those four
remaining = [k for k in vecs if k not in cluster]

# collect all remaining pairwise angles and average
rem_angles = [
    angles[frozenset((a, b))]
    for a, b in combinations(remaining, 2)
]
print(rem_angles)
avg_angle = np.mean(rem_angles)

print(f"Average angle among the remaining {len(remaining)} vectors: {avg_angle:.2f}°")
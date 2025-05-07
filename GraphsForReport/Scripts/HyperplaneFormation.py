import numpy as np
import matplotlib.pyplot as plt

# Original data
input_size = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
sv_ratio    = [1.38, 1.31, 1.41, 1.23, 1.38, 1.28, 1.40, 1.63, 1.52, 1.62, 1.84, 1.63]

# Define a moving-average that adapts window at the edges
def moving_average(data, window_size):
    half = window_size // 2
    smoothed = []
    n = len(data)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        smoothed.append(np.mean(data[start:end]))
    return smoothed

# Apply smoothing
window_size = 3  # still a 3-point average
y_smooth = moving_average(sv_ratio, window_size)

# Plot
plt.figure()
# plt.plot(input_size, sv_ratio, marker='o', linestyle='-', label='Original')
plt.plot(input_size, y_smooth, linestyle='-', label=f'{window_size}-point MA (adaptive edges)')
plt.xlabel('Feature Number')
plt.ylabel('Singular Value Rattio')
plt.title('Singular Value Ratio vs Feature Number')
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import pandas as pd
from itertools import product

# Define your categories and values
feature_num = [5, 12]
type_ = ['large', 'same', 'small_direct', 'small_compression']
pooling = ['mean', 'max']
architecture = ['GIN', 'GCN']
depth = [2, 3, 1]
loss = ['BCE', 'MSE']

# Generate combinations in the specified order
combinations = list(product(loss, depth, architecture, pooling, type_, feature_num))

# Create DataFrame
df = pd.DataFrame(combinations, columns=['Loss', 'Depth', 'Architecture', 'Pooling', 'Type', 'Feature_num'])

# Export to Excel
df.to_excel('combinations.xlsx', index=False)

print(df.head())
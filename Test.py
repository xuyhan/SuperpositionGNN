import pandas as pd
import torch


true = True
if true == True:
    base_config = {
        "mode": "simple",           # Options: "simple", "motif", "correlated", "combined"
        "num_categories": 12,
        "p": 0.8,
        "p_count": 0.9,
        "num_nodes": 20,
        "motif_dim": 0,             # 0 for simple experiments (no motif features)
        "chain_length_min": 2,
        "chain_length_max": 7,
        "num_train_samples": 5000,
        "num_test_samples": 1500,
        "batch_size": 16,
        "in_dim": 12,
        "hidden_dims": [18, 18],
        "lr": 0.01,
        "use_weighting": True,
        "importance": (15.0, 10.0),
        "phase1_epochs": 0,
        "phase2_epochs": 50,
        "num_epochs": 12,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "model_type": "GIN",         # e.g. "GCN" or "GIN"
        "pooling": "max",
        "log_dir": "runs/GIN/simple/large/max/12",
        "add_graph": False,
        "track_embeddings": False,
        "save": True
    }

# Define specific rows to iterate over (replace with actual row indices)
specific_rows = [2]  # example rows, can use index as in excel file. 
specific_rows = [i - 2 for i in specific_rows]
df = pd.read_excel('ExperimentList/combinations.xlsx')
# Create a list of configurations to iterate over
configs = []

for idx in specific_rows:
    row = df.iloc[idx]
    config = base_config.copy()

    # Set parameters from Excel
    config['loss'] = row['Loss']
    config['model_type'] = row['Architecture']
    config['pooling'] = row['Pooling']
    config['num_categories'] = row['Feature_num']
    config['in_dim'] = row['Feature_num']

    # Set hidden_dims based on depth, feature_num, and type as per specified logic
    if row['Depth'] == 1:
        hidden_dim_lookup = {
            5: {'large': [8], 'same': [5], 'small_direct': [2], 'small_compression': [2]},
            12: {'large': [18], 'same': [12], 'small_direct': [6], 'small_compression': [6]}
        }
    elif row['Depth'] == 2:
        hidden_dim_lookup = {
            5: {'large': [8, 8], 'same': [5, 5], 'small_direct': [2, 2], 'small_compression': [5, 2]},
            12: {'large': [18, 18], 'same': [12, 12], 'small_direct': [6, 6], 'small_compression': [12, 6]}
        }
    elif row['Depth'] == 3:
        hidden_dim_lookup = {
            5: {'large': [8, 8, 8], 'same': [5, 5, 5], 'small_direct': [2, 2, 2], 'small_compression': [5, 5, 2]},
            12: {'large': [18, 18, 18], 'same': [12, 12, 12], 'small_direct': [6, 6, 6], 'small_compression': [12, 12, 6]}
        }

    hidden_dim_value = hidden_dim_lookup[row['Feature_num']][row['Type']]
    config['hidden_dims'] = hidden_dim_value 

    configs.append(config)


import os
import json
import torch
from datetime import datetime
from Runner import run_multiple_experiments
from Trainer import Trainer
from GraphGeneration import sparcity_calculator
import numpy as np
import pandas as pd

def convert_keys_to_str(obj):
    """
    Recursively convert dictionary keys to strings and convert
    non-serializable objects (like torch.device, torch.Tensor, and NumPy types)
    to serializable types.
    """
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    else:
        return obj

def run_single_experiment(experiment_config):
    # Check for required keys.
    required_keys = ["hidden_dims", "mode", "model_type", "num_categories"]
    for key in required_keys:
        if key not in experiment_config:
            raise ValueError(f"Missing required key '{key}' in experiment_config.")

    # Calculate the sparsity of the initial embedding features.
    sparcity = sparcity_calculator(
        experiment_config["num_nodes"], 
        experiment_config["p"], 
        experiment_config["in_dim"]
    )

    print(f"\nRunning experiment: mode={experiment_config['mode']} | model_type={experiment_config['model_type']}")
    
    # Run the experiments.
    results, all_model_params, all_average_embeddings = run_multiple_experiments(experiment_config, num_experiments=1)
    print(f"Results: {results}")

    # Process and enhance the experiment results.
    readable_results = []
    keys = [
        "Num of features",         # index 0
        "Num of active features",  # index 1
        "Num of accurate feature", # index 2
        "Geometry",                # index 3
        "Collapsed",               # index 4
        "Loss"                     # index 5
    ]
    for result_entry, embedding in zip(results, all_average_embeddings):
        # Compute SVD analysis for the embeddings.
        rank, singular_values = Trainer.svd_analysis(embedding)
        entry_dict = { key: result_entry[i] for i, key in enumerate(keys) }
        entry_dict["Rank"] = rank
        entry_dict["Singular values"] = singular_values.tolist()  # convert for JSON serialization
        readable_results.append(entry_dict)
    
    # Perform additional geometry analysis and summarization.
    config_losses, model_params, average_embeddings = Trainer.geometry_analysis(results, all_model_params, all_average_embeddings)
    summary, model_summary, average_embeddings_summary = Trainer.summarize_config_losses(config_losses, model_params, average_embeddings)
    
    # Dynamically build a descriptive file name.
    mode = experiment_config["mode"]
    model_type = experiment_config["model_type"]
    num_categories = experiment_config["num_categories"]
    hidden_dims = experiment_config["hidden_dims"]
    final_hidden_dim = hidden_dims[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"exp_{mode}_{model_type}_{num_categories}cats_{final_hidden_dim}hidden_{timestamp}.json"
    
    # Define the folder structure; you could even vary this per config.
    file_path = experiment_config.get("file_path", "experiment_results")
    folder = os.path.join("experiment_results", file_path)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file_name)
    
    # Prepare the final output dictionary.
    output = {
        "experiment_config": experiment_config,
        "sparcity": sparcity,
        "summary format:": "Key: (Num of active features, Num of accurate feature, Geometry, Collapsed). Loss, s.d. Loss, Count",
        "summary": summary,
        "results": readable_results,
        "average embeddings summary": average_embeddings_summary,
        "model summary": model_summary
    }
    
    # Convert keys/objects to strings as necessary.
    output_str = convert_keys_to_str(output)
    
    # Save results if requested.
    if experiment_config.get("save", False):
        with open(file_path, "w") as f:
            json.dump(output_str, f, indent=4)
        print(f"Experiment results saved to {file_path}")

    return output_str

def main(specific_rows):
    # Choose MODE
    Mode = "motif"  # Options: "simple", "motif", "correlated", "combined"
    # Base configuration used for all experiments.
    base_config_simple = {
        "mode": "simple",           # Options: "simple", "motif", "correlated", "combined"
        "num_categories": 12,
        "p": 0.8,
        "num_nodes": 20,
        "motif_dim": 0,             # 0 for simple experiments (no motif features)
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
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "model_type": "GIN",         # e.g. "GCN" or "GIN"
        "loss": "BCE",
        "pooling": "max",
        "log_dir": "runs/GIN/simple/large/max/12",
        "file_path": "GIN/simple/large/max/12",
        "add_graph": False,
        "track_embeddings": False,
        "save": True
    }

    base_config_motif = {
        "mode": "motif",           # REQUIRED: Options: "simple", "motif", "correlated", "combined"
        "num_categories": 0,        # REQUIRED motif does not contibute to the number of categories
        "p": 0.3,
        "num_nodes": 20,
        "motif_dim": 3,             # 0 for simple experiments (no motif features)
        "chain_length_min": 2,
        "chain_length_max": 7,
        "num_train_samples": 5000,
        "num_test_samples": 1500,
        "batch_size": 4,
        "in_dim": 1,
        "hidden_dims": [6, 3],      # REQUIRED: List of hidden layer dimensions
        "lr": 0.01,
        "use_weighting": True,
        "importance": (15.0, 10.0),
        "phase1_epochs": 0,
        "phase2_epochs": 50,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "model_type": "GIN",         # REQUIRED: e.g. "GCN" or "GIN"
        "loss": "BCE",
        "pooling": "max",
        "log_dir": "runs/GIN/simple/large/max/12",
        "file_path": "GIN/simple/large/max/12",
        "add_graph": False,
        "track_embeddings": False,
        "save": True
    }

    # Define specific rows to iterate over (replace with actual row indices)
    specific_rows = [i - 2 for i in specific_rows]

    # Create a list of configurations to iterate over
    configs = []

    # Setup configurations based on the mode
    if Mode == "simple":
        df = pd.read_excel('ExperimentList/combinations.xlsx')

        for idx in specific_rows:
            row = df.iloc[idx]
            config = base_config_simple.copy()

            # Set parameters from Excel
            config['loss'] = row['Loss']
            config['model_type'] = row['Architecture']
            config['pooling'] = row['Pooling']
            config['num_categories'] = row['Feature_num']
            if row['Feature_num'] == 5:
                config['p'] = 0.3
            config['in_dim'] = row['Feature_num']
            config['log_dir'] = f"runs/{row['Loss']}/{row['Depth']}/{row['Architecture']}/{row['Type']}/{row['Pooling']}/{row['Feature_num']}"
            config['file_path'] = (f"{row['Loss']}/{row['Depth']}/{row['Architecture']}/{row['Type']}/{row['Pooling']}/{row['Feature_num']}")

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

    elif Mode == "motif":
        df = pd.read_excel('ExperimentList/motif_combinations.xlsx')

        for idx in specific_rows:
            row = df.iloc[idx]
            config = base_config_motif.copy()

            # Set parameters from Excel
            config['model_type'] = row['Architecture']
            config['pooling'] = row['Pooling']
            config['log_dir'] = f"runs/motif/{row['Architecture']}/{row['Pooling']}/{row['Hidden']}"
            config['file_path'] = (f"motif/{row['Architecture']}/{row['Pooling']}/{row['Hidden']}")

            if row['Hidden'] == 1:
                config['hidden_dims'] = [2]
            elif row['Hidden'] == 2:
                config['hidden_dims'] = [2, 2]
            elif row['Hidden'] == 3:
                config['hidden_dims'] = [3, 2]
            elif row['Hidden'] == 4:
                config['hidden_dims'] = [4, 2]
            elif row['Hidden'] == 5:
                config['hidden_dims'] = [2, 2, 2]
            elif row['Hidden'] == 6:
                config['hidden_dims'] = [3, 3, 2]
            elif row['Hidden'] == 7:
                config['hidden_dims'] = [4, 4, 2]

            configs.append(config)

    # Loop through each configuration and run the corresponding experiment.
    for config in configs:
        run_single_experiment(config)


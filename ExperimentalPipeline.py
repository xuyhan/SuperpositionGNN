import os
import json
import torch
from datetime import datetime
from Runner import run_multiple_experiments
from Trainer import Trainer
from GraphGeneration import sparcity_calculator
import numpy as np

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

def main():
    # Define the experiment configuration.
    experiment_config = {
        "mode": "simple",           # REQUIRED: Options: "simple", "motif", "correlated", "combined"
        "num_categories": 12,        # REQUIRED motif does not contibute to the number of categories
        "p": 0.8,
        "p_count": 0.9,             # Probability for edges in count mode
        "num_nodes": 20,
        "motif_dim": 0,             # 0 for simple experiments (no motif features)
        "chain_length_min": 2,
        "chain_length_max": 7,
        "num_train_samples": 5000,
        "num_test_samples": 1500,
        "batch_size": 16,
        "in_dim": 12,
        "hidden_dims": [12, 6],      # REQUIRED: List of hidden layer dimensions
        "lr": 0.01,
        "use_weighting": True,
        "importance": (15.0, 10.0),
        "phase1_epochs": 0,
        "phase2_epochs": 50,
        "num_epochs": 12,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "model_type": "GIN",         # REQUIRED: e.g. "GCN" or "GIN"
        "pooling": "max",
        "log_dir": "runs/GIN/simple/small/max/12",
        "add_graph": False,
        "track_embeddings": False,  # In TensorBoard, track the embeddings of the last layer.
        "save": True
    }
    
    # Check for required keys.
    required_keys = ["hidden_dims", "mode", "model_type", "num_categories"]
    for key in required_keys:
        if key not in experiment_config:
            raise ValueError(f"Missing required key '{key}' in experiment_config.")
    
    # Calculate the sparcity of the initial embedding features
    sparcity = sparcity_calculator(experiment_config["num_nodes"], experiment_config["p"], experiment_config["in_dim"])

    print("Running experiments...")
    results, all_model_params, all_average_embeddings = run_multiple_experiments(experiment_config, num_experiments=50)
    print(f"Results: {results}")

    # Make results more readable and add SVD results for each experiment
    readable_results = []
    # Expected keys for each result entry.
    keys = [
        "Num of features",         # index 0 in original list
        "Num of active features",  # index 1
        "Num of accurate feature", # index 2
        "Geometry",                # index 3
        "Collapsed",               # index 4
        "Loss"                     # index 5
    ]

    # Loop over each experiment result and corresponding embeddings.
    for result_entry, embedding in zip(results, all_average_embeddings):
        # Compute SVD analysis for the given embeddings.
        rank, singular_values = Trainer.svd_analysis(embedding)
        # Build a dictionary from the result_entry.
        entry_dict = { key: result_entry[i] for i, key in enumerate(keys) }
        # Add SVD analysis results.
        entry_dict["Rank"] = rank
        entry_dict["Singular values"] = singular_values.tolist()  # convert to list for JSON serialization
        readable_results.append(entry_dict)
    
    # Perform geometry analysis on the results.
    config_losses, model_params, average_embeddings = Trainer.geometry_analysis(results, all_model_params, all_average_embeddings)
    summary, model_summary, average_embeddings_summary = Trainer.summarize_config_losses(config_losses, model_params, average_embeddings)
    
    # Dynamically build a descriptive file name.
    mode = experiment_config["mode"]
    model_type = experiment_config["model_type"]
    num_categories = experiment_config["num_categories"]
    hidden_dims = experiment_config["hidden_dims"]
    final_hidden_dim = hidden_dims[-1]  # Dynamically uses the last element of hidden_dims.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"exp_{mode}_{model_type}_{num_categories}cats_{final_hidden_dim}hidden_{timestamp}.json"
    
    # Specify the folder where you want to save the file.
    folder = "experiment_results"
    # Create the folder if it doesn't exist.
    os.makedirs(folder, exist_ok=True)
    
    # Combine folder and file name into a full path.
    file_path = os.path.join(folder, file_name)
    
    # Prepare output dictionary.
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
    
    # Save the experiment configuration and summary into the JSON file.
    if experiment_config["save"] == True:
        with open(file_path, "w") as f:
            json.dump(output_str, f, indent=4)
    
        print(f"\nExperiment results saved to {file_path}")


############################################################################################################
#######################   Here are examples of the required parameters for each mode   #####################


def main_combined():
    # Define the experiment configuration.
    experiment_config = {
        "mode": "combined",           # REQUIRED: Options: "simple", "motif", "correlated", "combined"
        "num_categories": 3,        # REQUIRED motif does not contibute to the number of categories
        "p": 0.25,
        "num_nodes": 20,
        "motif_dim": 3,             # 0 for simple experiments (no motif features)
        "chain_length_min": 2,
        "chain_length_max": 7,
        "num_train_samples": 10000,
        "num_test_samples": 3000,
        "batch_size": 16,
        "in_dim": 4,
        "hidden_dims": [6, 6, 3],      # REQUIRED: List of hidden layer dimensions
        "lr": 0.01,
        "use_weighting": True,
        "importance": (15.0, 10.0),
        "phase1_epochs": 5,
        "phase2_epochs": 10,
        "num_epochs": 5,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "model_type": "GIN"         # REQUIRED: e.g. "GCN" or "GIN"
    }


def main_motif():
    # Define the experiment configuration.
    experiment_config = {
        "mode": "combined",           # REQUIRED: Options: "simple", "motif", "correlated", "combined"
        "num_categories": 0,        # REQUIRED motif does not contibute to the number of categories
        "p": 0.25,
        "num_nodes": 20,
        "motif_dim": 3,             # 0 for simple experiments (no motif features)
        "chain_length_min": 2,
        "chain_length_max": 7,
        "num_train_samples": 10000,
        "num_test_samples": 3000,
        "batch_size": 4,
        "in_dim": 1,
        "hidden_dims": [6, 6, 3],      # REQUIRED: List of hidden layer dimensions
        "lr": 0.01,
        "use_weighting": True,
        "importance": (15.0, 10.0),
        "phase1_epochs": 5,
        "phase2_epochs": 10,
        "num_epochs": 5,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "model_type": "GIN"         # REQUIRED: e.g. "GCN" or "GIN"
    }

def main_simple():
    # Define the experiment configuration.
    experiment_config = {
        "mode": "simple",           # REQUIRED: Options: "simple", "motif", "correlated", "combined"
        "num_categories": 3,        # REQUIRED motif does not contibute to the number of categories
        "p": 0.25,
        "num_nodes": 20,
        "motif_dim": 0,             # 0 for simple experiments (no motif features)
        "chain_length_min": 2,
        "chain_length_max": 7,
        "num_train_samples": 10000,
        "num_test_samples": 3000,
        "batch_size": 4,
        "in_dim": 3,
        "hidden_dims": [6, 6, 3],      # REQUIRED: List of hidden layer dimensions
        "lr": 0.01,
        "use_weighting": True,
        "importance": (15.0, 10.0),
        "phase1_epochs": 5,
        "phase2_epochs": 10,
        "num_epochs": 5,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "model_type": "GCN"         # REQUIRED: e.g. "GCN" or "GIN"
    }
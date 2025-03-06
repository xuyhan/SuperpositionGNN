import os
import json
import torch
from datetime import datetime
from Runner import run_multiple_experiments
from Trainer import Trainer
from GraphGeneration import sparcity_calculator

def convert_keys_to_str(obj):
    """
    Recursively convert dictionary keys to strings and convert
    non-serializable objects (like torch.device) to strings.
    """
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(item) for item in obj]
    elif isinstance(obj, torch.device):
        return str(obj)
    else:
        return obj

def main():
    # Define the experiment configuration.
    experiment_config = {
        "mode": "simple",           # REQUIRED: Options: "simple", "motif", "correlated", "combined"
        "num_categories": 3,        # REQUIRED motif does not contibute to the number of categories
        "p": 0.2,
        "num_nodes": 20,
        "motif_dim": 0,             # 0 for simple experiments (no motif features)
        "chain_length_min": 2,
        "chain_length_max": 7,
        "num_train_samples": 10000,
        "num_test_samples": 3000,
        "batch_size": 16,
        "in_dim": 3,
        "hidden_dims": [3, 3],      # REQUIRED: List of hidden layer dimensions
        "lr": 0.01,
        "use_weighting": True,
        "importance": (15.0, 10.0),
        "phase1_epochs": 5,
        "phase2_epochs": 5,
        "num_epochs": 5,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "model_type": "GCN"         # REQUIRED: e.g. "GCN" or "GIN"
    }
    
    # Check for required keys.
    required_keys = ["hidden_dims", "mode", "model_type", "num_categories"]
    for key in required_keys:
        if key not in experiment_config:
            raise ValueError(f"Missing required key '{key}' in experiment_config.")
    
    # Calculate the sparcity of the initial embedding features
    sparcity = sparcity_calculator(experiment_config["num_nodes"], experiment_config["p"], experiment_config["in_dim"])

    print("Running experiments...")
    results, all_model_params = run_multiple_experiments(experiment_config, num_experiments=4)
    print(f"Results: {results}")
    
    # Perform geometry analysis on the results.
    config_losses, model_params = Trainer.geometry_analysis(results, all_model_params)
    summary, model_summary = Trainer.summarize_config_losses(config_losses, model_params)
    
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
        "summary": summary,
        "model summary": model_summary,
        "results": results,
    }
    
    # Convert keys/objects to strings as necessary.
    output_str = convert_keys_to_str(output)
    
    # Save the experiment configuration and summary into the JSON file.
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
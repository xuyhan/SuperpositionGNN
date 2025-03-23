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
    results, all_model_params, all_average_embeddings = run_multiple_experiments(experiment_config, num_experiments=50)
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
    folder = os.path.join("experiment_results", model_type, mode)
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

def main():
    # Base configuration used for all experiments.
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

    # Create a list of configurations to iterate over.
    # For example, we vary the "mode" and adjust other parameters accordingly.
    configs = []
    for mode in ["simple", "motif", "correlated"]:
        # Make a shallow copy of the base config.
        config = base_config.copy()
        config["mode"] = mode
        # Optionally adjust other parameters for each configuration.
        if mode == "motif":
            config["motif_dim"] = 5  # example change for motif mode
        configs.append(config)
    
    # Loop through each configuration and run the corresponding experiment.
    for config in configs:
        run_single_experiment(config)

if __name__ == "__main__":
    main()
import os
import json
import torch
from datetime import datetime
from Runner import run_multiple_experiments
from Trainer import Trainer
from GraphGeneration import sparcity_calculator
import numpy as np
import pandas as pd
from PipelineUtils import *

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
    results, all_model_params, all_average_embeddings, empty_graph_stats, avg_predictions = run_multiple_experiments(experiment_config, num_experiments=1)
    print(f"Average predictions: {avg_predictions}")
    # Condense empty graph stats
    empty_graph_stats = mean_std_global(empty_graph_stats)
    print(f"Results: {results}")

    # Convert the raw results into a human-readable format.
    readable_results = make_readable_results(results, all_average_embeddings, Trainer)
    sv_ratio = mean_singular_values(readable_results)

    # Determine the percentage of collapsed embeddings.
    collapsed_percentage = determine_percentage_of_collapsed(readable_results)

    # Perform additional geometry analysis and summarization.
    config_losses, model_params, average_embeddings = Trainer.geometry_analysis(results, all_model_params, all_average_embeddings)
    summary, model_summary, average_embeddings_summary = Trainer.summarize_config_losses(config_losses, model_params, average_embeddings)
    # Saves all elements in an extra file
    get_all_elements(experiment_config, average_embeddings)
    
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
        "empty graph stats": empty_graph_stats,
        "singular value ratio": sv_ratio,
        "summary format:": "Key: (Num of active features, Num of accurate feature, Geometry, Collapsed). Loss, s.d. Loss, Count",
        "collapsed percentage": collapsed_percentage,
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

def main(specific_rows, Mode):
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
        "gm_p": 1.0,               # Generalized mean pooling parameter
        "log_dir": "runs/GIN/simple/large/max/12",
        "file_path": "GIN/simple/large/max/12",
        "add_graph": False,
        "track_embeddings": False,
        "track_singular_values": True,
        "get_elements": False,      # To get all the elements of the hidden embedding (best configuration)
        "save": True
    }

    base_config_motif = {
        "mode": "motif",           # REQUIRED: Options: "simple", "motif", "correlated", "combined"
        "num_categories": 0,       # REQUIRED: motif does not contribute to the number of categories
        "p": 0.3,
        "num_nodes": 20,
        "motif_dim": 3,            # 0 for simple experiments (no motif features)
        "chain_length_min": 2,
        "chain_length_max": 7,
        "num_train_samples": 5000,
        "num_test_samples": 1500,
        "batch_size": 4,
        "in_dim": 1,
        "hidden_dims": [6, 3],     # REQUIRED: List of hidden layer dimensions
        "lr": 0.01,
        "use_weighting": True,
        "importance": (15.0, 10.0),
        "phase1_epochs": 0,
        "phase2_epochs": 50,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "model_type": "GIN",       # REQUIRED: e.g. "GCN" or "GIN"
        "loss": "BCE",
        "pooling": "max",
        "gm_p": 1.0,               # Generalized mean pooling parameter
        "log_dir": "runs/GIN/simple/large/max/12",
        "file_path": "GIN/simple/large/max/12",
        "add_graph": False,
        "track_embeddings": False,
        "track_singular_values": True,
        "save": True
    }

    base_config_count = {
        "mode": "count",           
        "num_categories": 3,        # REQUIRED: motif does not contribute to the number of categories
        "p": 0.3,
        "p_count": 0.9,  
        "num_nodes": 10,
        "num_train_samples": 5000,
        "num_test_samples": 1500,
        "batch_size": 4,
        "in_dim": 2,
        "hidden_dims": [6, 3],      # REQUIRED: List of hidden layer dimensions
        "lr": 0.01,
        "use_weighting": True,
        "importance": (15.0, 10.0),
        "phase1_epochs": 0,
        "phase2_epochs": 15,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "model_type": "GIN",       # REQUIRED: e.g. "GCN" or "GIN"
        "loss": "BCE",
        "pooling": "max",
        "gm_p": 1.0,               # Generalized mean pooling parameter
        "log_dir": "runs/GIN/count",
        "file_path": "GIN/count",
        "add_graph": False,
        "track_embeddings": False,
        "track_singular_values": True,
        "save": True
    }

    base_config_tox21 = {
        "mode": "tox21",            # NEW
        "root": "data/Tox21",       # download location
        "batch_size": 32,
        "train_split": 0.8,
        "mask_missing": True,
        "use_weighting": False,
        "importance": (100.0, 100.0),

        # Network-level params (tune freely in Excel)
        "hidden_dims": [64, 64],
        "num_categories": 12, 
        "in_dim": 9,           # Tox21 has 9 features
        "lr": 1e-3,
        "model_type": "GCN",        # GCN also works
        "loss": "BCE",
        "pooling": "mean",
        "gm_p": 1.0,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "phase1_epochs": 0,
        "phase2_epochs": 50,

        # bookkeeping
        "log_dir": "runs/tox21",
        "file_path": "tox21",
        "save": True,

        # Avoid silly errors
        "num_nodes": 20,          # Not used in Tox21
        "p": 0.8,                 # Not used in Tox21

    }

    # Adjust specific_rows according to your logic.
    specific_rows = [i - 2 for i in specific_rows]
    configs = []

    if Mode == "simple":
        df = pd.read_excel('ExperimentList/combinations.xlsx')
        for idx in specific_rows:
            row = df.iloc[idx]
            config = base_config_simple.copy()

            type_lower = row['Type'].strip().lower()
            feature_num_str = str(row['Feature_num'])

            config['loss'] = row['Loss']
            config['model_type'] = row['Architecture']
            config['pooling'] = row['Pooling']

            if type_lower == "specify" or ("," in feature_num_str):
                # For the specify case, call helper function to get a tuple: (feature, hidden_dims)
                feature_val, hidden_dims = get_hidden_dims("simple",
                                                           feature_num=row['Feature_num'],
                                                           depth=int(row['Depth']),
                                                           type_str=row['Type'])
                config['num_categories'] = feature_val
                config['in_dim'] = feature_val
                config['hidden_dims'] = hidden_dims
                if feature_val < 18:
                    config['p'] = feature_val/18
                else:
                    config['p'] = 1.0

                # Use a default naming convention for specify.
                config['log_dir'] = (f"runs/SV/{row['Loss']}/{int(row['Depth'])}/"
                                     f"{row['Architecture']}/{row['Type']}/{row['Feature_num']}/specified")
                config['file_path'] = (f"SV/{row['Loss']}/{int(row['Depth'])}/"
                                       f"{row['Architecture']}/{row['Type']}/{row['Feature_num']}/specified")
                configs.append(config.copy())
            else:
                # Regular case using the lookup.
                config['num_categories'] = int(feature_num_str)
                config['in_dim'] = int(feature_num_str)
                config['hidden_dims'] = get_hidden_dims("simple",
                                                       feature_num=row['Feature_num'],
                                                       depth=int(row['Depth']),
                                                       type_str=row['Type'])
                if int(row['Feature_num']) == 5:
                    probs = [0.3, 0.6, 0.8]
                    labels = ['high', 'medium', 'low']
                    for i in range(1):  # Example: iterate over one probability option.
                        config['p'] = probs[i]
                        if row['Pooling'] == 'gm':
                            config['gm_p'] = row['Power']
                            config['log_dir'] = (f"runs/T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                                 f"{row['Type']}/p_gm={row['Power']}/{int(row['Feature_num'])}/{labels[i]}")
                            config['file_path'] = (f"T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                                   f"{row['Type']}/p_gm={row['Power']}/{row['Feature_num']}/{labels[i]}")
                        else:
                            config['log_dir'] = (f"runs/T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                                 f"{row['Type']}/{row['Pooling']}/{int(row['Feature_num'])}/{labels[i]}")
                            config['file_path'] = (f"T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                                   f"{row['Type']}/{row['Pooling']}/{int(row['Feature_num'])}/{labels[i]}")
                        configs.append(config.copy())
                elif int(row['Feature_num']) == 12:
                    if row['Pooling'] == 'gm':
                        config['log_dir'] = (f"runs/T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                             f"{row['Type']}/p_gm={row['Power']}/{int(row['Feature_num'])}/high")
                        config['file_path'] = (f"T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                               f"{row['Type']}/p_gm={row['Power']}/{int(row['Feature_num'])}/high")
                        configs.append(config)
                    else:
                        config['log_dir'] = (f"runs/T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                             f"{row['Type']}/{row['Pooling']}/{int(row['Feature_num'])}/high")
                        config['file_path'] = (f"T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                               f"{row['Type']}/{row['Pooling']}/{int(row['Feature_num'])}/high")
                        configs.append(config)
                elif int(row['Feature_num']) == 3:
                    config['p'] = 0.18
                    config['log_dir'] = (f"runs/T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                         f"{row['Type']}/{row['Pooling']}/{int(row['Feature_num'])}")
                    config['file_path'] = (f"T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                           f"{row['Type']}/{row['Pooling']}/{int(row['Feature_num'])}")
                    configs.append(config.copy())
                elif int(row['Feature_num']) == 4:
                    config['p'] = 0.24
                    config['log_dir'] = (f"runs/T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                         f"{row['Type']}/{row['Pooling']}/{int(row['Feature_num'])}")
                    config['file_path'] = (f"T/{row['Loss']}/{int(row['Depth'])}/{row['Architecture']}/"
                                           f"{row['Type']}/{row['Pooling']}/{int(row['Feature_num'])}")
                    configs.append(config.copy())

    elif Mode == "motif":
        df = pd.read_excel('ExperimentList/motif_combinations.xlsx')
        for idx in specific_rows:
            row = df.iloc[idx]
            config = base_config_motif.copy()
            
            config['model_type'] = row['Architecture']
            config['pooling'] = row['Pooling']
            config['log_dir'] = f"runs/motif/{row['Architecture']}/{row['Pooling']}/{row['Hidden']}"
            config['file_path'] = f"motif/{row['Architecture']}/{row['Pooling']}/{row['Hidden']}"
            
            # Use the helper function for motif mode.
            config['hidden_dims'] = get_hidden_dims("motif", hidden=row['Hidden'])
            configs.append(config)

    elif Mode == "count":
        df = pd.read_excel('ExperimentList/count_combinations.xlsx')
        for idx in specific_rows:
            row = df.iloc[idx]
            config = base_config_count.copy()
            
            config['model_type'] = row['Architecture']
            config['pooling'] = row['Pooling']
            config['log_dir'] = f"runs/evo/count/{row['Architecture']}/{row['Pooling']}/{row['Hidden']}"
            config['file_path'] = f"evo/count/{row['Architecture']}/{row['Pooling']}/{row['Hidden']}"
            
            # Use the helper function for count mode.
            config['hidden_dims'] = get_hidden_dims("count", hidden=row['Hidden'])
            configs.append(config)

    elif Mode == "tox21":
        df = pd.read_excel('ExperimentList/tox21_combinations.xlsx')
        for idx in specific_rows:
            row = df.iloc[idx]
            config = base_config_tox21.copy()

            # let you vary architecture / pooling / hidden dims via Excel
            config["model_type"] = row["Architecture"]
            config["pooling"]    = row["Pooling"]
            __, hidden_dims = get_hidden_dims("tox21",
                                                    feature_num=row['Feature_num'],
                                                    depth=int(row['Depth']),
                                                    type_str=row['Type'])
            config["hidden_dims"] = hidden_dims

            config["log_dir"]  = f"runs/tox21/{row['Architecture']}/{row['Pooling']}/{config['hidden_dims']}"
            config["file_path"]= f"tox21/{row['Architecture']}/{row['Pooling']}/{config['hidden_dims']}"
            configs.append(config)
    
    # Loop through each configuration and run the corresponding experiment.
    for config in configs:
        run_single_experiment(config)


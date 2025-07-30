import os
import json
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Any
import itertools

def convert_keys_to_str(obj):
    """
    Recursively convert dictionary keys to strings and convert non-serializable objects
    (such as torch.device, torch.Tensor, and NumPy types) to serializable formats.
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

def get_all_elements(experimental_config, average_embeddings):
    """
    If enabled in the experimental configuration, extract all elements from the average embeddings
    and save them to a file.
    """
    get_elements = experimental_config.get("get_elements", False)
    if get_elements:
        all_elements = []
        try:
            elements_list = average_embeddings[(5, 5, 5, 0)]
        except KeyError:
            print("Key not found, skipping")
            elements_list = []
        for elements_dict in elements_list:
            for tensor_vector in elements_dict.values():
                for number in tensor_vector:
                    all_elements.append(number.item())
        
        output_dir = f"experiment_results/all_elements/{experimental_config['gm_p']}"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "all_elements.txt")
        with open(output_file_path, "w") as file:
            file.write(str(all_elements))
        return all_elements
    return None

def make_readable_results(results, all_average_embeddings, trainer):
    """
    Convert raw experiment results into a human-readable format by computing
    extra analyses (e.g., performing SVD on embeddings).

    Parameters:
        results (list): List of raw result entries.
        all_average_embeddings (list): List of corresponding average embeddings.
        trainer (module): The Trainer module which provides an 'svd_analysis' function.

    Returns:
        list: A list of dictionaries with human-readable result information.
    """
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
        rank, singular_values = trainer.svd_analysis(embedding)
        entry_dict = { key: result_entry[i] for i, key in enumerate(keys) }
        entry_dict["Rank"] = rank
        entry_dict["Singular values"] = singular_values.tolist()
        readable_results.append(entry_dict)
    
    return readable_results


def determine_percentage_of_collapsed(readable_results):
    """
    Compute the percentage of runs where the computed rank is less than the number of accurate features.
    
    Parameters:
        readable_results (list of dict): Each dictionary should contain at least the following keys:
            - "Rank": an integer representing the computed rank,
            - "Num of accurate feature": an integer representing the number of accurate features.
    
    Returns:
        float: The percentage of runs where the rank is less than the number 
               of accurate features.
    """
    if not readable_results:
        return 0.0  # Avoid division by zero.
    
    count = 0
    total_runs = len(readable_results)
    
    for result in readable_results:
        rank = result.get("Rank", 0)
        accurate_features = result.get("Num of accurate feature", 0)
        if rank < accurate_features:
            count += 1
            
    percentage = (count / total_runs) * 100
    return percentage


def get_hidden_dims(mode, **kwargs):
    """
    Returns the hidden dimension configuration based on the mode and provided parameters.

    For mode "simple":
        Expected keyword arguments:
            - feature_num:
                * If type_str (from kwargs) is "specify", then feature_num should be a string 
                  containing comma‐separated numbers (e.g., "16, 10, 10"). In this case, the first number
                  is used for 'num_categories' and 'in_dim', and the remaining numbers become the hidden dimensions.
                * Otherwise, feature_num must be a number (or string convertible to int) representing
                  the feature number used to index the internal lookup.
            - depth (int): The network depth (1, 2, or 3).
            - type_str (str): One of "large", "same", "small_direct", "small_compression", or "specify".

    For modes "motif" or "count":
        Expected keyword arguments:
            - hidden (int): The hidden configuration level (1 to 7).

    Parameters:
        mode (str): One of "simple", "motif", or "count".
        **kwargs: Parameters required for the specific mode.

    Returns:
        For "simple" mode and type "specify":
            tuple: (feature_number, list_of_hidden_dims)
        For "simple" mode with a lookup type:
            list: The corresponding list of hidden dimensions.
        For "motif" or "count":
            list: The corresponding list of hidden dimensions.

    Raises:
        ValueError: If required parameters are missing or values are unsupported.
    """
    if mode in ["simple", "tox21"]:
        type_str = kwargs['type_str']
        if type_str.strip().lower() == "specify":
            # Parse the comma-separated string.
            dims_str = str(kwargs['feature_num'])
            try:
                dims = [int(x.strip()) for x in dims_str.split(',') if x.strip()]
            except Exception as e:
                raise ValueError("Error parsing 'feature_num' for specify mode: " + str(e))
            if len(dims) < 2:
                raise ValueError("For specify type, you must provide at least two numbers: "
                                 "the first for the feature number and at least one hidden dimension.")
            feature_val = dims[0]
            hidden_dims = dims[1:]
            return (feature_val, hidden_dims)
        else:
            # Not specifying hidden dimensions directly: use a lookup.
            feature_num = int(kwargs['feature_num'])
            depth = int(kwargs['depth'])
            if depth == 1:
                hidden_dim_lookup = {
                    5: {'large': [8], 'same': [5], 'small_direct': [2], 'small_compression': [2]},
                    12: {'large': [18], 'same': [12], 'small_direct': [6], 'small_compression': [6]}
                }
            elif depth == 2:
                hidden_dim_lookup = {
                    3: {'small_compression': [3, 2]},
                    4: {'small_compression': [4, 2]},
                    5: {'large': [8, 8], 'same': [5, 5], 'small_direct': [2, 2], 'small_compression': [5, 2]},
                    12: {'large': [18, 18], 'same': [12, 12], 'small_direct': [6, 6], 'small_compression': [12, 6]}
                }
            elif depth == 3:
                hidden_dim_lookup = {
                    5: {'large': [8, 8, 8], 'same': [5, 5, 5], 'small_direct': [2, 2, 2], 'small_compression': [5, 5, 2]},
                    12: {'large': [18, 18, 18], 'same': [12, 12, 12], 'small_direct': [6, 6, 6], 'small_compression': [12, 12, 6]}
                }
            else:
                raise ValueError("Unsupported depth: {}.".format(depth))
            
            if feature_num not in hidden_dim_lookup:
                raise ValueError("Unsupported feature number: {} for depth {}.".format(feature_num, depth))
            
            if type_str not in hidden_dim_lookup[feature_num]:
                raise ValueError("Unsupported type: {} for feature number {} and depth {}."
                                 .format(type_str, feature_num, depth))
            
            return hidden_dim_lookup[feature_num][type_str]

    elif mode in ("motif", "count"):
        try:
            hidden_val = int(kwargs['hidden'])
        except KeyError:
            raise ValueError("Missing required parameter 'hidden' for mode '{}'.".format(mode))
        
        motif_lookup = {
            1: [2],
            2: [2, 2],
            3: [3, 2],
            4: [4, 2],
            5: [2, 2, 2],
            6: [3, 3, 2],
            7: [4, 4, 2]
        }
        if hidden_val not in motif_lookup:
            raise ValueError("Unsupported hidden value: {} for mode '{}'.".format(hidden_val, mode))
        return motif_lookup[hidden_val]
    
    else:
        raise ValueError("Unsupported mode: {}. Use 'simple', 'motif', or 'count'.".format(mode))
    

    
def mean_std_global(stats: List[Dict[str, List[float]]]) -> float:
    """
    Compute the overall mean of all floats in all 'std' vectors.
    
    Args:
        stats: List of dicts, each with a 'std' key whose value is a list of floats.
        
    Returns:
        A single float: the mean of all std values.
    """
    # Flatten all std lists into one sequence
    all_values = list(itertools.chain.from_iterable(entry['std'] for entry in stats))
    if not all_values:
        return 0.0
    return sum(all_values) / len(all_values)



def mean_singular_values(results: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Compute the mean of the highest and second-highest singular values,
    but only for entries satisfying:
      - if Num of features > 6:
            (Num of features - 4) < Num of active features
      - else:
            (Num of features - 3) < Num of active features

    Args:
        results: List of dicts, each expected to have at least:
                   - "Num of features": int
                   - "Num of active features": int
                   - "Singular values": List[float]

    Returns:
        (mean_highest, mean_second)
    """
    highest = []
    second  = []

    for entry in results:
        n_feat = entry.get("Num of features", 0)
        n_active = entry.get("Num of active features", 0)

        # choose the correct threshold
        threshold = n_feat - 4 if n_feat > 6 else n_feat - 3

        if threshold < n_active:
            sv = entry.get("Singular values", [])
            if len(sv) >= 1:
                highest.append(sv[0])
            if len(sv) >= 2:
                second.append(sv[1])

    if not highest:
        return 0.0, 0.0

    mean_highest = sum(highest) / len(highest)
    mean_second  = sum(second)  / len(second) if second else 0.0

    return mean_highest, mean_second


def alignment_index(embeddings: Dict[Any, torch.Tensor]) -> Tuple[float, Tuple[float, float]]:
    """Compute the Alignment Index (AI) for a set of embedding vectors.

    The AI measures how axis‑aligned a collection of vectors is. It is defined
    as the mean over vectors of ``max(|v_j|) / ||v||_2``.

    Parameters
    ----------
    embeddings : Dict[Any, torch.Tensor]
        Mapping from keys to 1‑D embedding tensors. The values may also be
        NumPy arrays or lists, in which case they are converted automatically.

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        ``(ai, (lower, upper))`` where ``ai`` is the mean Alignment Index and
        ``(lower, upper)`` is the 95% confidence interval computed using the
        standard error ``1.96 * sd / sqrt(n)``.
    """

    # Convert to a list of 1-D numpy arrays
    vecs = []
    for v in embeddings.values():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        vecs.append(np.asarray(v, dtype=float))

    if not vecs:
        return 0.0, (0.0, 0.0)

    ratios = []
    for v in vecs:
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        ratios.append(np.max(np.abs(v)) / norm)

    if not ratios:
        return 0.0, (0.0, 0.0)

    ratios = np.array(ratios)
    ai = ratios.mean()
    sd = ratios.std(ddof=1) if len(ratios) > 1 else 0.0
    se = sd / np.sqrt(len(ratios)) if len(ratios) > 0 else 0.0
    delta = 1.96 * se
    return ai, (ai - delta, ai + delta)


def alignment_index_list(embeddings_list: List[Dict[Any, torch.Tensor]]) -> Tuple[float, Tuple[float, float]]:
    """Compute the Alignment Index over multiple embedding dictionaries.

    Parameters
    ----------
    embeddings_list : list of dict
        Each element is passed to :func:`alignment_index`.

    Returns
    -------
    Tuple[float, Tuple[float, float]]
        Mean AI across the list and its 95% confidence interval.
    """
    if not embeddings_list:
        return 0.0, (0.0, 0.0)

    values = []
    for emb in embeddings_list:
        ai, _ = alignment_index(emb)
        values.append(ai)

    values = np.array(values)
    mean_ai = values.mean()
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    se = sd / np.sqrt(len(values))
    delta = 1.96 * se
    return mean_ai, (mean_ai - delta, mean_ai + delta)
# datasets/tox21_loader.py
import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader        # quiets the deprecation warning

import torch

def compute_pos_weights(data_list, n_tasks=12):
    pos = torch.zeros(n_tasks)
    neg = torch.zeros(n_tasks)

    for d in data_list:
        y = d.y
        if y.dim() > 1:                  # squash [1,12] → [12]
            y = y.view(-1)

        pos += (y == 1).float()
        neg += (y == 0).float()

    pos = torch.clamp(pos, min=1.)       # avoid div-0
    neg = torch.clamp(neg, min=1.)
    return neg / pos                     # tensor length-12

def get_tox21_loaders(root: str,
                      batch_size: int = 32,
                      train_split: float = 0.8,
                      shuffle: bool = True,
                      mask_missing: bool = True):

    dataset = MoleculeNet(root=root, name="Tox21")

    in_dim = dataset.num_node_features          # ← store BEFORE any filtering
    # Compute positive weights for loss function
    pos_weights = compute_pos_weights(dataset)

    if mask_missing:                            # optional: still drop –1 graphs
        dataset = [d for d in dataset if (d.y >= 0).all()]

    for d in dataset:          # make GCN happy
        d.x = d.x.to(torch.float32)

    n = len(dataset)
    split = int(train_split * n)
    train_data, test_data = dataset[:split], dataset[split:]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, in_dim, pos_weights     # ← now safe
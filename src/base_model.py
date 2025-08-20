import math
from pathlib import Path
from typing import List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GINConv
from torch_scatter import scatter_mean
from tqdm import trange

from src.eval import MultiLabelEvaluator
from src.logger import CustomLogger
from src.utils import Constants


def equiangular_frame(out_dim, hidden_dim):
    """
    Returns a fixed weight matrix with an equiangular configuration for some special cases.

    This helps in setting up the final linear layer in a specific configuration.
    """
    if out_dim == 3 and hidden_dim == 2:
        return torch.tensor([[1.0, 0.0], [-0.5, math.sqrt(3) / 2], [-0.5, -math.sqrt(3) / 2]])
    elif out_dim == 4 and hidden_dim == 2:
        return torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    elif out_dim == 5 and hidden_dim == 2:
        return torch.tensor([[math.cos(0 * 2 * math.pi / 5), math.sin(0 * 2 * math.pi / 5)],
                             [math.cos(1 * 2 * math.pi / 5), math.sin(1 * 2 * math.pi / 5)],
                             [math.cos(2 * 2 * math.pi / 5), math.sin(2 * 2 * math.pi / 5)],
                             [math.cos(3 * 2 * math.pi / 5), math.sin(3 * 2 * math.pi / 5)],
                             [math.cos(4 * 2 * math.pi / 5), math.sin(4 * 2 * math.pi / 5)]])
    elif out_dim == 6 and hidden_dim == 2:
        return torch.tensor([[math.cos(0 * 2 * math.pi / 6), math.sin(0 * 2 * math.pi / 6)],
                             [math.cos(1 * 2 * math.pi / 6), math.sin(1 * 2 * math.pi / 6)],
                             [math.cos(2 * 2 * math.pi / 6), math.sin(2 * 2 * math.pi / 6)],
                             [math.cos(3 * 2 * math.pi / 6), math.sin(3 * 2 * math.pi / 6)],
                             [math.cos(4 * 2 * math.pi / 6), math.sin(4 * 2 * math.pi / 6)],
                             [math.cos(5 * 2 * math.pi / 6), math.sin(5 * 2 * math.pi / 6)]])
    elif out_dim == 4 and hidden_dim == 3:
        return torch.tensor([[1.0, 0.0, -math.sqrt(0.5)], [-1.0, 0.0, -math.sqrt(0.5)], [0.0, 1.0, math.sqrt(0.5)],
                             [0.0, -1.0, math.sqrt(0.5)]])
    elif out_dim == 6 and hidden_dim == 3:
        # Return the 6 vertices of a regular octahedron in 3D.
        return torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    elif out_dim == 3 and hidden_dim == 3:
        # Return the 3 vertices of a regular tetrahedron in 3D.
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], ])
    else:
        raise ValueError(
            "Equiangular frame not implemented for (out_dim={}, hidden_dim={}).".format(out_dim, hidden_dim))


def initialize_output_weights(W, out_dim, hidden_dim):
    """
    Initializes the weight matrix W using an equiangular frame if available;
    otherwise falls back on orthogonal initialization.
    """
    try:
        eq_frame = equiangular_frame(out_dim, hidden_dim)
        W.data.copy_(eq_frame.to(W.device).type_as(W))
    except ValueError:
        nn.init.orthogonal_(W)


def global_generalized_mean_pool(x, batch, p, eps=1e-6):
    """
    Generalized mean pooling that preserves the sign of each element with numerical stability.

    For each element in x:
      - Compute: sign(x) * (|x| + eps)^p
      - Pool these values using scatter_mean.
      - Apply the inverse transformation: sign(pooled) * (|pooled| + eps)^(1/p)

    Args:
        x (Tensor): Node features of shape [num_nodes, feature_dim].
        batch (Tensor): Batch vector of shape [num_nodes] indicating graph assignment.
        p (float): Generalized mean parameter.
        eps (float): A small constant to prevent numerical issues.

    Returns:
        Tensor: Graph-level pooled representations, shape [num_graphs, feature_dim].
    """
    # Transform each element while preserving its sign and ensuring numerical stability.
    x_transformed = torch.sign(x) * ((torch.abs(x) + eps) ** p)

    pooled = scatter_mean(x_transformed, batch, dim=0)

    # Apply the inverse transformation with epsilon for stability.
    return torch.sign(pooled) * ((torch.abs(pooled) + eps) ** (1.0 / p))


class BaseModel(nn.Module):

    def __init__(self, identifier: str, model_type: str, in_dim: int, hidden_dims: List[int], out_dim: int,
                 num_ff: int,
                 pooling: str, criterion: Callable, logger: CustomLogger, gm_p: float = 1.0):

        super().__init__()

        self.identifier = identifier
        self.model_type = model_type
        self.pooling = pooling
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_ff = num_ff

        self.p = gm_p
        self.criterion = criterion
        self.logger = logger

        self.device = None

        self.convs = nn.ModuleList()
        self.ffs = nn.ModuleList()

        self._init_layers(in_dim, hidden_dims, out_dim, num_ff)

    def _init_layers(self, in_dim: int, hidden_dims: List[int], out_dim: int, num_ff: int):

        if self.model_type == "GCN":

            prev_dim = in_dim

            for hdim in hidden_dims:
                self.convs.append(GCNConv(prev_dim, hdim))
                prev_dim = hdim

        elif self.model_type == "GIN":

            prev_dim = in_dim

            for hdim in hidden_dims:
                mlp = nn.Sequential(nn.Linear(prev_dim, hdim), nn.ReLU(), nn.Linear(hdim, hdim))
                self.convs.append(GINConv(mlp, train_eps=True))
                prev_dim = hdim

        elif self.model_type == "GAT":

            prev_dim = in_dim

            for hdim in hidden_dims:
                self.convs.append(GATv2Conv(prev_dim, hdim))
                prev_dim = hdim

        else:
            raise ValueError("Unsupported model_type.")

        for idx in range(num_ff):
            if idx == num_ff - 1:
                self.ffs.append(nn.Linear(prev_dim, out_dim, bias=True))
            else:
                self.ffs.append(nn.Linear(prev_dim, prev_dim, bias=True))

        # initialize_output_weights(self.ffs[-1].weight, out_dim, hidden_dims[-1])

        #
        # self.lin_out1 = nn.Linear(prev_dim, prev_dim, bias=True)
        #
        # self.lin_out = nn.Linear(prev_dim, out_dim, bias=True)

        # if freeze_final:
        #     initialize_output_weights(self.lin_out.weight, out_dim, hidden_dims[-1])
        #     self.lin_out.weight.requires_grad = False
        #     if self.lin_out.bias is not None:
        #         self.lin_out.bias.requires_grad = True

    def set_device(self, device: torch.device):
        self.to(device)
        self.device = device

    def forward(self, x, edge_index, batch, return_repr: bool = False):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)

        if self.pooling == "mean":
            graph_repr = global_mean_pool(x, batch)
        elif self.pooling == "max":
            graph_repr = global_max_pool(x, batch)
        elif self.pooling == "gm":
            graph_repr = global_generalized_mean_pool(x, batch, p=self.p)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        graph_repr = F.relu(graph_repr)
        x = graph_repr

        for i, ff in enumerate(self.ffs):
            x = ff(x)
            if i < len(self.ffs) - 1:
                x = F.relu(x)

        return (x, graph_repr) if return_repr else x

    def _get_path(self) -> Path:

        return Constants.MODEL_SAVE_DIR / f"{self.identifier}.pt"

    def _save(self, optimizer):

        path = self._get_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "pytorch_version": torch.__version__,
        }

        torch.save(state, path)

    def _restore(self, optimizer):

        path = self._get_path()
        state = torch.load(path, weights_only=False, map_location=self.device)
        self.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])

    def _one_epoch(self, train_loader: DataLoader, optimizer):

        self.train()

        evaluator = MultiLabelEvaluator(self.out_dim, device=self.device)

        total_loss = 0.0

        for data in train_loader:
            data = data.to(self.device)

            optimizer.zero_grad()
            logits = self.forward(data.x, data.edge_index, data.batch)  # [batch_size, out_dim]

            batch_size, out_dim = logits.size()
            targets = data.y.float().view(batch_size, out_dim)

            evaluator.update(logits, targets)

            loss = self.criterion(logits, targets)  # shape: [batch_size, out_dim]

            mean_loss = loss.mean()
            mean_loss.backward()
            optimizer.step()
            total_loss += mean_loss.item()

        acc_total, prec_total, rec_total, auc_total, _ = evaluator.compute()

        return total_loss / len(train_loader), acc_total, prec_total, rec_total, auc_total

    def fit(self, train_loader: DataLoader, optimizer, num_epochs: int, use_cached: bool):

        path = self._get_path()

        if path.exists() and use_cached:
            self.logger.info(f"Using cached model at: {path}")
            self._restore(optimizer)
        else:
            pbar = trange(1, num_epochs + 1, desc="Training", unit="epoch")
            for epoch in pbar:
                epoch_loss, acc_total, prec_total, rec_total, auc_total = self._one_epoch(train_loader, optimizer)

                pbar.set_description(f"Epoch {epoch}/{num_epochs}")
                pbar.set_postfix(train_loss=f"{epoch_loss:.3f}",
                                 train_acc=f"{acc_total:.3f}",
                                 train_prec=f"{prec_total:.3f}",
                                 train_rec=f"{rec_total:.3f}",
                                 train_auc=f"{auc_total:.3f}")

            self._save(optimizer)

    def evaluate(self, test_loader: DataLoader):
        self.eval()

        evaluator = MultiLabelEvaluator(self.out_dim, device=self.device)

        total_loss = torch.zeros((), device=self.device)
        total_elems = 0

        sum_preds = None  # shape [C, C]  (avg predicted vector per one-hot class)
        sum_repr = None  # shape [C, D]  (avg embedding per one-hot class)
        count = None  # shape [C]

        with torch.inference_mode():
            for data in test_loader:
                data = data.to(self.device, non_blocking=True)

                logits, graph_repr = self.forward(data.x, data.edge_index, data.batch, return_repr=True)
                B, C = logits.shape
                probs = torch.sigmoid(logits)
                targets = data.y.float().view(B, C)

                evaluator.update(logits, targets)

                loss = self.criterion(logits, targets)
                total_loss += loss.sum()
                total_elems += loss.numel()

                if sum_preds is None:
                    D = graph_repr.size(1)
                    sum_preds = torch.zeros(C, C, device=self.device)
                    sum_repr = torch.zeros(C, D, device=self.device)
                    count = torch.zeros(C, dtype=torch.long, device=self.device)

                pure_mask = targets.sum(dim=1) == 1
                if pure_mask.any():
                    cls_idx = targets[pure_mask].argmax(dim=1)  # [M]
                    preds = (probs[pure_mask] > 0.5).float()  # [M, C]

                    sum_preds.index_add_(0, cls_idx, preds)  # [C, C]
                    sum_repr.index_add_(0, cls_idx, graph_repr[pure_mask])  # [C, D]
                    count.index_add_(0, cls_idx, torch.ones_like(cls_idx, dtype=count.dtype))

        avg_loss = (total_loss / total_elems).item()

        avg_predictions, avg_embeddings = {}, {}
        if count is not None:
            nonzero = (count > 0).nonzero(as_tuple=False).squeeze(1)
            if nonzero.numel() > 0:
                avg_pred_mat = sum_preds[nonzero] / count[nonzero].unsqueeze(1)
                avg_repr_mat = sum_repr[nonzero] / count[nonzero].unsqueeze(1)
                C = sum_preds.size(1)
                for idx_in_list, c in enumerate(nonzero.tolist()):
                    key = tuple(1 if i == c else 0 for i in range(C))
                    avg_predictions[key] = avg_pred_mat[idx_in_list].detach().cpu()
                    avg_embeddings[key] = avg_repr_mat[idx_in_list].detach().cpu()

        acc_total, prec_total, rec_total, auc_total, per_class_results = evaluator.compute()

        return avg_embeddings, avg_predictions, avg_loss, acc_total, prec_total, rec_total, auc_total, per_class_results

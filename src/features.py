from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Set

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from src.base_model import BaseModel


# -------------------------------
# Concept-family base class
# -------------------------------
class ConceptFamily(ABC):

    @abstractmethod
    def get_id(self) -> str:
        ...

    @abstractmethod
    def names(self, data_sample: Data) -> List[str]:
        """Return list of concept names for this family, given a sample Data
        (can use data.x.shape[1] to infer cardinality)."""
        ...

    @abstractmethod
    def detect(self, data: Data) -> torch.Tensor:
        """Return node-level concept labels as a (num_nodes, num_concepts) float/bool tensor."""
        ...


# -------------------------------
# is(k): node-type indicators
# Assumes data.x is one-hot (or multi-hot); threshold binarizes if needed.
# -------------------------------
class IsTypeFamily(ConceptFamily):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def get_id(self) -> str:
        return f"{self.__class__.__name__} {self.threshold}"

    def names(self, data_sample: Data) -> List[str]:
        D = data_sample.x.size(-1)
        return [f"is({k})" for k in range(D)]

    def detect(self, data: Data) -> torch.Tensor:
        x = data.x
        return (x > self.threshold).to(torch.float32)


# -------------------------------
# adjacent_to(k): a node is positive for k if ANY neighbor has type k
# Works for directed or undirected graphs; we symmetrize.
# -------------------------------
class AdjacentToFamily(ConceptFamily):
    def __init__(self, threshold: float = 0.5, symmetrize: bool = True, ignore_self_loops: bool = True):
        self.threshold = threshold
        self.symmetrize = symmetrize
        self.ignore_self_loops = ignore_self_loops

    def get_id(self) -> str:
        return f"{self.__class__.__name__} {self.threshold} {self.symmetrize} {self.ignore_self_loops}"

    def names(self, data_sample: Data) -> List[str]:
        D = data_sample.x.size(-1)
        return [f"adjacent_to({k})" for k in range(D)]

    def detect(self, data: Data) -> torch.Tensor:
        x = (data.x > self.threshold).to(torch.float32)  # (N, F)
        N, F = x.shape
        src, dst = data.edge_index  # (2, E)

        if self.ignore_self_loops:
            mask = src != dst
            src = src[mask]
            dst = dst[mask]

        nb_sum = x.new_zeros((N, F))
        # aggregate src -> dst
        nb_sum.index_add_(0, dst, x[src])

        if self.symmetrize:
            # and dst -> src (use the same filtered pairs)
            nb_sum.index_add_(0, src, x[dst])

        # “exists a neighbor with type k” → binary
        return (nb_sum > 0).to(torch.float32)


class GraphConceptFamily(ABC):

    def __init__(self):
        self.mode = None

    @abstractmethod
    def get_id(self) -> str:
        ...

# -----------------------------------------
# Graph-level concept family (labels)
# -----------------------------------------
class GraphLabelFamily(GraphConceptFamily):
    """
    Graph-level ‘concepts’ are the K class labels.
    You can choose the label source:
      - 'gt'         : ground-truth multi-hot labels
      - 'pred>0.5'   : model sigmoid(logits) > threshold (default 0.5)
      - 'inter'      : intersection of GT and Pred>threshold (logical AND)
    Optionally keep only one-hot rows when computing targets (one_hot_only=True).
    """
    def __init__(self,
                 label_source: str = "gt",       # "gt" | "pred>0.5" | "inter"
                 pred_threshold: float = 0.5,
                 one_hot_only: bool = False):
        super().__init__()

        assert label_source in {"gt", "pred>0.5", "inter"}
        self.label_source = label_source
        self.pred_threshold = pred_threshold
        self.one_hot_only = one_hot_only
        self.mode = "logits"

    def get_id(self) -> str:
        return f"{self.__class__.__name__} {self.label_source} {self.pred_threshold} {self.one_hot_only}"

    def names(self, num_classes: int) -> List[str]:
        return [f"class_{i}" for i in range(num_classes)]

    def targets(self, logits: torch.Tensor, y: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
          logits: [N, C] tensor of model logits (on CPU or GPU).
          y     : [N, C] tensor of ground-truth multi-hot labels (same N,C).
        Returns:
          T_np     : [N', C] numpy array of binary targets after filtering
          row_mask : [N] boolean numpy array indicating which rows kept
        """
        assert logits.shape == y.shape
        with torch.no_grad():
            P = torch.sigmoid(logits)

            if self.label_source == "gt":
                T = (y > 0.5).to(torch.int64)
            elif self.label_source == "pred>0.5":
                T = (P > self.pred_threshold).to(torch.int64)
            else:  # "inter"
                T = ((y > 0.5) & (P > self.pred_threshold)).to(torch.int64)

            if self.one_hot_only:
                row_mask = (T.sum(dim=1) == 1)
            else:
                row_mask = torch.ones(T.size(0), dtype=torch.bool, device=T.device)

            T_np = T[row_mask].cpu().numpy()
            mask_np = row_mask.cpu().numpy()
            return T_np, mask_np



# Utilities (cycle handling)
# =========================

def _build_undirected_adj(
    edge_index: torch.Tensor,
    num_nodes: int,
    ignore_self_loops: bool = True,
) -> Tuple[List[Set[int]], Set[frozenset]]:
    """
    Return (adj, undirected_edge_set).
      adj[u] = set of neighbors v (undirected, no self-loops if requested)
      undirected_edge_set = { frozenset({u,v}) } for fast membership tests
    """
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    E = edge_index.shape[1]
    for e in range(E):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])
        if ignore_self_loops and u == v:
            continue
        if u == v:
            continue
        # undirected
        adj[u].add(v)
        adj[v].add(u)
    undirected_edges: Set[frozenset] = set()
    for u in range(num_nodes):
        for v in adj[u]:
            if u < v:
                undirected_edges.add(frozenset((u, v)))
    return adj, undirected_edges


def _canon_cycle(nodes: List[int]) -> Tuple[int, ...]:
    """
    Canonicalize a cycle node list up to rotation and reversal:
      - rotate so the smallest node id comes first
      - choose orientation (forward/backward) that is lexicographically smaller
    """
    L = len(nodes)
    # forward rotation
    m = min(nodes)
    i0 = nodes.index(m)
    fwd = nodes[i0:] + nodes[:i0]
    # backward rotation
    bwd_list = list(reversed(nodes))
    j0 = bwd_list.index(m)
    bwd = bwd_list[j0:] + bwd_list[:j0]
    return tuple(fwd) if tuple(fwd) <= tuple(bwd) else tuple(bwd)


def _is_chordless(cyc: Tuple[int, ...], undirected_edges: Set[frozenset]) -> bool:
    """
    A cycle is chordless if no non-consecutive pair in cyc shares an edge.
    """
    L = len(cyc)
    on_cycle = {frozenset((cyc[i], cyc[(i + 1) % L])) for i in range(L)}
    for i in range(L):
        for j in range(i + 1, L):
            e = frozenset((cyc[i], cyc[j]))
            # consecutive (including last-first) edges are allowed
            if e in on_cycle:
                continue
            # a present edge between non-consecutive nodes is a chord
            if e in undirected_edges:
                return False
    return True


def _enumerate_cycles_by_length(
    edge_index: torch.Tensor,
    num_nodes: int,
    lengths: Tuple[int, ...],
    *,
    ignore_self_loops: bool = True,
    chordless_only: bool = True,
) -> Dict[int, List[Tuple[int, ...]]]:
    """
    Enumerate simple cycles (lengths in `lengths`) exactly.
    Designed for small graphs / short cycles (e.g., <= 6 or 7).
    Returns: dict L -> list of canonicalized cycles (tuples of node ids).
    """
    if len(lengths) == 0 or num_nodes == 0:
        return {L: [] for L in lengths}
    Lmax = max(lengths)
    adj, undirected_edges = _build_undirected_adj(edge_index, num_nodes, ignore_self_loops)

    seen: Set[Tuple[int, ...]] = set()
    out: Dict[int, List[Tuple[int, ...]]] = {L: [] for L in lengths}

    # DFS with pruning: only explore paths whose minimum node is the start node
    for s in range(num_nodes):
        stack: List[Tuple[int, List[int]]] = [(s, [s])]
        while stack:
            u, path = stack.pop()
            if len(path) > Lmax:
                continue
            for v in adj[u]:
                if v == s and len(path) >= 3:
                    L = len(path)
                    if L in out:
                        cyc = _canon_cycle(path)
                        if cyc not in seen:
                            if (not chordless_only) or _is_chordless(cyc, undirected_edges):
                                seen.add(cyc)
                                out[L].append(cyc)
                    continue
                if v in path:
                    continue
                # prune duplicates: never allow nodes smaller than start
                if v < s:
                    continue
                if len(path) + 1 <= Lmax:
                    stack.append((v, path + [v]))
    return out


# ==================================
# Node-level motif concept family
# ==================================
class IsInsideMotifFamily(ConceptFamily):
    """
    Node-level concepts: for each L in cycle_lengths, mark nodes that lie on
    at least one (optionally chordless) L-cycle.
    """
    def __init__(
        self,
        cycle_lengths: Tuple[int, ...] = (3, 4, 5, 6),
        *,
        chordless_only: bool = True,
        ignore_self_loops: bool = True,
        name_prefix: str = "inside"
    ):
        self.cycle_lengths = tuple(int(L) for L in cycle_lengths)
        self.chordless_only = bool(chordless_only)
        self.ignore_self_loops = bool(ignore_self_loops)
        self.name_prefix = name_prefix

    def get_id(self) -> str:
        return f"{self.__class__.__name__} {self.cycle_lengths} {self.chordless_only} {self.ignore_self_loops}"

    def names(self, data_sample: Data) -> List[str]:
        return [f"{self.name_prefix}(C{L})" for L in self.cycle_lengths]

    def detect(self, data: Data) -> torch.Tensor:
        """
        Returns: [N, K] float tensor, K=len(cycle_lengths).
        Entry (u, j) == 1.0 iff node u is on some (chordless) C_{cycle_lengths[j]}.
        """
        N = int(data.num_nodes) if hasattr(data, "num_nodes") else int(data.x.size(0))
        if N == 0:
            return torch.zeros(0, len(self.cycle_lengths), dtype=torch.float32)

        cycles = _enumerate_cycles_by_length(
            data.edge_index, N, self.cycle_lengths,
            ignore_self_loops=self.ignore_self_loops,
            chordless_only=self.chordless_only,
        )
        out = torch.zeros(N, len(self.cycle_lengths), dtype=torch.float32)
        idx = {L: j for j, L in enumerate(self.cycle_lengths)}
        for L, cyc_list in cycles.items():
            j = idx[L]
            for cyc in cyc_list:
                for u in cyc:
                    out[u, j] = 1.0
        return out


from torch_geometric.utils import subgraph

def _filter_subgraph(data: Data, node_mask: torch.Tensor) -> Data:
    """
    Return an induced subgraph on the nodes where node_mask == True.
    Keeps undirectedness as-is (edges are filtered).
    """
    node_idx = torch.nonzero(node_mask, as_tuple=False).view(-1)
    ei, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    sub = Data(
        x=(data.x[node_idx] if getattr(data, "x", None) is not None else None),
        edge_index=ei,
    )
    sub.num_nodes = int(node_idx.numel())
    return sub


class HasMotifFamily(GraphConceptFamily):
    """
    Graph-level concepts from structure (e.g., presence of chordless C_L).
    Provides:
      - mode = "data"
      - names(data_sample)
      - targets_from_data(data) -> [B, C] float tensor of per-graph labels
    """
    def __init__(self,
                 cycle_lengths: Tuple[int, ...] = (3, 4, 6),
                 *,
                 chordless_only: bool = True,
                 ignore_self_loops: bool = True,
                 name_prefix: str = "has"):
        super().__init__()
        self.cycle_lengths = tuple(int(L) for L in cycle_lengths)
        self.chordless_only = bool(chordless_only)
        self.ignore_self_loops = bool(ignore_self_loops)
        self.name_prefix = name_prefix
        self.mode = "data"

    def get_id(self) -> str:
        return f"{self.__class__.__name__} {self.cycle_lengths} {self.chordless_only} {self.ignore_self_loops}"

    def names(self, data_sample: Data) -> List[str]:
        return [f"{self.name_prefix}(C{L})" for L in self.cycle_lengths]

    # ---- Use your cycle detector from earlier (not repeated here) ----
    def targets_from_data(self, data: Data) -> torch.Tensor:
        """
        Returns: [B, C] float, where C=len(cycle_lengths).
        If `data` is a single-graph Data (no batch), returns [1, C].
        """
        device = data.edge_index.device
        lengths = self.cycle_lengths

        def _single_graph_targets(graph: Data) -> torch.Tensor:
            N = int(graph.num_nodes)
            if N == 0:
                return torch.zeros(len(lengths), dtype=torch.float32, device=device).unsqueeze(0)
            cycles = _enumerate_cycles_by_length(
                graph.edge_index, N, lengths,
                ignore_self_loops=self.ignore_self_loops,
                chordless_only=self.chordless_only,
            )
            present = [1.0 if len(cycles.get(L, [])) > 0 else 0.0 for L in lengths]
            return torch.tensor(present, dtype=torch.float32, device=device).unsqueeze(0)

        if getattr(data, "batch", None) is None:
            return _single_graph_targets(data)  # [1, C]

        B = int(data.batch.max().item() + 1)
        out = torch.zeros((B, len(lengths)), dtype=torch.float32, device=device)
        # split nodes per-graph and build subgraphs; reuse earlier helper if you have one
        for b in range(B):
            mask = (data.batch == b)
            sub = Data(
                x=data.x[mask] if getattr(data, "x", None) is not None else None,
                edge_index=data.edge_index,  # edge filtering is inside the helper
            )
            sub.num_nodes = int(mask.sum().item())
            # The helper that builds the subgraph should filter edges to endpoints in 'mask'
            out[b:b+1] = _single_graph_targets(_filter_subgraph(data, mask))
        return out


def fast_hash(data_loader):
    num_graphs = len(data_loader)
    total_nodes = sum([g.x.shape[0] for g in data_loader])
    return hash(num_graphs * total_nodes)

class FeatureExtractor:

    CONCEPT_CACHE = {}

    def __init__(self, model: BaseModel):

        self.model = model

    def _collect_embeddings_logits_targets(self, loader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          H: [N, D] graph embeddings
          Z: [N, C] logits
          Y: [N, C] ground-truth {0,1}
        """
        self.model.eval()
        Hs, Zs, Ys = [], [], []
        with torch.inference_mode():
            for data in loader:
                data = data.to(self.model.device, non_blocking=True)
                logits, graph_repr = self.model.forward(
                    data.x, data.edge_index, data.batch, return_repr=True
                )
                B, C = logits.shape
                y = data.y.float().view(B, C)
                Hs.append(graph_repr.detach().cpu())
                Zs.append(logits.detach().cpu())
                Ys.append(y.detach().cpu())
        H = torch.vstack(Hs) if Hs else torch.zeros(0, 0)
        Z = torch.vstack(Zs) if Zs else torch.zeros(0, 0)
        Y = torch.vstack(Ys) if Ys else torch.zeros(0, 0)
        return H, Z, Y

    def compute_soft_centroid_features(
            self,
            loader,
            *,
            weight_source: str = "pred_prob",  # "pred_prob" | "gt" | "pred_x_gt"
            variant: str = "pos",  # "pos" | "delta"
            min_mass: float = 1.0,
            eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          F: [C, D] feature matrix (rows may be zero if effective mass < min_mass)
          mass: [C] effective positive mass used per class
        """
        H, Z, Y = self._collect_embeddings_logits_targets(loader)  # [N,D], [N,C], [N,C]
        if H.numel() == 0:
            return torch.zeros(0, 0), torch.zeros(0)

        P = torch.sigmoid(Z)  # [N, C]
        N, D = H.shape
        C = P.shape[1]

        if weight_source == "pred_prob":
            Wpos = P
        elif weight_source == "gt":
            Wpos = Y
        elif weight_source == "pred_x_gt":
            # soft intersection (can also use Y * (P > 0.5) if you prefer hard)
            Wpos = P * Y
        else:
            raise ValueError(f"Unknown weight_source: {weight_source}")

        # positive centroid
        num_pos = Wpos.T @ H  # [C, D]
        den_pos = Wpos.sum(dim=0).clamp_min(eps).unsqueeze(1)  # [C,1]
        mu_pos = num_pos / den_pos

        if variant == "pos":
            F = mu_pos
            mass = den_pos.squeeze(1)
        elif variant == "delta":
            Wneg = (1.0 - Wpos)
            num_neg = Wneg.T @ H
            den_neg = Wneg.sum(dim=0).clamp_min(eps).unsqueeze(1)
            mu_neg = num_neg / den_neg
            F = mu_pos - mu_neg
            mass = den_pos.squeeze(1)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # zero out features with too little effective support
        mask = (mass >= min_mass).float().unsqueeze(1)  # [C,1]
        F = F * mask
        return F.to(self.model.device), mass.to(self.model.device)


    # ------------------------------------------
    # Helper: compute node embeddings at a layer
    # layer_idx: -1 (default) means after last conv (pre-pooling)
    # pre_relu: if True, capture before ReLU in hidden layers (< last)
    # ------------------------------------------
    def _node_embeddings_from_batch(
        self,
        data: Data,
        *,
        layer_idx: int = -1,
        pre_relu: bool = False,
    ) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        # run conv stack, optionally capturing at layer_idx
        for i, conv in enumerate(self.model.convs):
            x = conv(x, edge_index)
            is_last = (i == len(self.model.convs) - 1)
            if (i == layer_idx) and pre_relu and not is_last:
                # capture pre-ReLU at this layer
                return x
            if not is_last:
                x = torch.nn.functional.relu(x)
            if i == layer_idx:
                return x
        # if layer_idx == -1 or out of range, return final node embeddings (pre-pooling)
        return x

    # ---------------------------------------------------------
    # Node-level concepts → linear probe features (normals)
    # Returns:
    #   W: [C, D] probe normals in original embedding space
    #   b: [C]    intercepts
    #   active_mask: List[bool] concepts kept (prevalence & AUROC gate)
    #   auroc_by_concept: Dict[int, float] (for transparency)
    #
    # Notes:
    # - We standardize node embeddings once (global scaler).
    # - Class weighting="balanced" to handle imbalanced concepts.
    # - We gate by (min_pos, min_neg, auroc_thresh).
    # ---------------------------------------------------------
    def compute_node_concept_features(
        self,
        loader: DataLoader,
        concept_family: ConceptFamily,
        *,
        layer_idx: int = -1,
        pre_relu: bool = False,
        auroc_thresh: float = 0.80,
        min_pos: int = 100,
        min_neg: int = 100,
        C_reg: float = 1.0,
        class_weight: Optional[str] = "balanced",
        max_iter: int = 500,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[bool]]:
        self.model.eval()

        node_embeddings: List[torch.Tensor] = []
        concept_labels: List[torch.Tensor] = []
        sample_data_for_names: Optional[Data] = None

        loader_hash = fast_hash(loader)
        if loader_hash in FeatureExtractor.CONCEPT_CACHE:
            cached = FeatureExtractor.CONCEPT_CACHE[loader_hash]
        else:
            FeatureExtractor.CONCEPT_CACHE[loader_hash] = {}
            cached = FeatureExtractor.CONCEPT_CACHE[loader_hash]

        concept_family_id = concept_family.get_id()

        self.model.logger.debug(f"loader_hash: {loader_hash}")
        self.model.logger.debug(f"concept_family_id: {concept_family_id}")

        if concept_family_id in cached:
            node_embeddings, concept_labels, sample_data_for_names = cached[concept_family_id]
            self.model.logger.info(f"Loaded cached concepts for {concept_family_id}.")
        else:
            with torch.inference_mode():
                for data in loader:
                    data = data.to(self.model.device, non_blocking=True)
                    # stash sample for naming / concept count
                    if sample_data_for_names is None:
                        sample_data_for_names = data

                    # compute concept labels for this batch
                    T_batch = concept_family.detect(data)  # (N_b, C_f)

                    # compute node embeddings at requested layer (pre-pooling)
                    H_batch = self._node_embeddings_from_batch(
                        data, layer_idx=layer_idx, pre_relu=pre_relu
                    )  # (N_b, D)

                    # sanity check sizes
                    if H_batch.size(0) != T_batch.size(0):
                        raise RuntimeError(
                            f"Node count mismatch: H[{H_batch.size(0)}] vs labels[{T_batch.size(0)}]"
                        )
                    node_embeddings.append(H_batch.detach().cpu())
                    concept_labels.append(T_batch.detach().cpu())

            cached[concept_family_id] = node_embeddings, concept_labels, sample_data_for_names

        if not node_embeddings:
            # empty loader
            return (
                torch.zeros(0, 0, device=self.model.device),
                torch.zeros(0, device=self.model.device),
                [],
                [],
            )

        X = torch.cat(node_embeddings, dim=0).numpy()  # (N, D)
        X[np.abs(X) < 1e-6] = 0

        T = torch.cat(concept_labels, dim=0).numpy()   # (N, C_f)
        N, D = X.shape
        C_f = T.shape[1]

        if N == 0:
            self.model.logger.info(f"No concepts found for: {self.model} {concept_family}")
            return (
                torch.zeros(0, 0, device=self.model.device),
                torch.zeros(0, device=self.model.device),
                [],
                [],
            )

        # Standardize embeddings globally
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(X)
        std = scaler.scale_.astype(np.float64)
        std[std == 0.0] = 1.0
        mean = scaler.mean_.astype(np.float64)

        W_orig = np.zeros((C_f, D), dtype=np.float64)
        b_orig = np.zeros((C_f,), dtype=np.float64)
        active_mask: List[bool] = [False] * C_f
        auroc_by_concept: List[float] = [float("nan") for _ in range(C_f)]

        # One-vs-rest logistic for each concept
        for c in range(C_f):
            y = T[:, c].astype(np.int32)
            pos = int(y.sum())
            neg = int(len(y) - pos)
            # Prevalence gate
            if pos < min_pos or neg < min_neg:
                continue

            # Fit
            clf = LogisticRegression(
                penalty="l2",
                C=C_reg,
                solver="liblinear",
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=0,
            )
            clf.fit(Xs, y)

            # Compute AUROC from calibrated scores (logit)
            # z = w_std^T * Xs + b_std
            scores = clf.decision_function(Xs)
            try:
                auc = roc_auc_score(y, scores)
            except ValueError:
                auc = float("nan")
            auroc_by_concept[c] = float(auc)

            if not np.isfinite(auc) or auc < auroc_thresh:
                continue  # not active

            # Map weights back to ORIGINAL embedding space:
            # logit = w_std^T * ((x - mean)/std) + b_std
            #       = (w_std/std)^T x + (b_std - (w_std/std)^T mean)
            w_std = clf.coef_.ravel().astype(np.float64)
            b_std = float(clf.intercept_.ravel()[0])
            w_o = w_std / std
            b_o = b_std - np.dot(w_o, mean)
            W_orig[c] = w_o
            b_orig[c] = b_o
            active_mask[c] = True

        W_t = torch.from_numpy(W_orig).float().to(self.model.device)  # [C_f, D]
        b_t = torch.from_numpy(b_orig).float().to(self.model.device)  # [C_f]
        return W_t, b_t, auroc_by_concept, active_mask

    def compute_graph_concept_features(
            self,
            loader,
            concept_family,  # GraphLabelFamily (mode="logits") OR HasMotifFamily (mode="data")
            *,
            C_reg: float = 1.0,
            class_weight: Optional[str] = "balanced",
            max_iter: int = 500,
            auroc_thresh: float = 0.65,
            min_pos: int = 20,
            min_neg: int = 20,
            standardize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[bool]]:
        """
        Fit one-vs-rest linear probes on graph embeddings H for a graph-level concept family.

        Returns:
          W_orig : [C, D] probe weight vectors in ORIGINAL embedding space
          b      : [C]    intercepts in ORIGINAL space
          aucs   : List[float] AUROC per concept (np.nan if degenerate)
          active : List[bool] (auroc >= auroc_thresh and min_pos/min_neg satisfied)
        """
        device = self.model.device
        self.model.eval()

        H_rows: List[torch.Tensor] = []
        Z_rows: List[torch.Tensor] = []
        Y_rows: List[torch.Tensor] = []
        T_rows: List[torch.Tensor] = []  # used only for mode="data"

        loader_hash = fast_hash(loader)
        if loader_hash in FeatureExtractor.CONCEPT_CACHE:
            cached = FeatureExtractor.CONCEPT_CACHE[loader_hash]
        else:
            FeatureExtractor.CONCEPT_CACHE[loader_hash] = {}
            cached = FeatureExtractor.CONCEPT_CACHE[loader_hash]

        concept_family_id = concept_family.get_id()

        self.model.logger.debug(f"loader_hash: {loader_hash}")
        self.model.logger.debug(f"concept_family_id: {concept_family_id}")

        if concept_family_id in cached:
            H_rows, Z_rows, Y_rows, T_rows = cached[concept_family_id]
            self.model.logger.info(f"Loaded cached concepts for {concept_family_id}.")
        else:

            with torch.inference_mode():
                for data in loader:
                    data = data.to(device, non_blocking=True)
                    logits, graph_repr = self.model.forward(
                        data.x, data.edge_index, data.batch, return_repr=True
                    )  # logits: [B, C_task], graph_repr: [B, D]
                    H_rows.append(graph_repr.detach().cpu())
                    Z_rows.append(logits.detach().cpu())
                    if getattr(data, "y", None) is not None:
                        y = data.y.float().view(logits.size(0), -1)
                        Y_rows.append(y.detach().cpu())
                    else:
                        # If your loader doesn't carry GT for certain concepts, fill zeros of the same row-count
                        Y_rows.append(torch.zeros_like(logits.detach().cpu()))

                    # If the concept family needs graph structure, compute per-batch targets now
                    if concept_family.mode == "data":
                        T_b = concept_family.targets_from_data(data)  # [B, C_concepts]
                        T_rows.append(T_b.detach().cpu())

            cached[concept_family_id] = H_rows, Z_rows, Y_rows, T_rows

        # stack
        H = torch.vstack(H_rows)  # [N, D]
        Z = torch.vstack(Z_rows)  # [N, C_task]
        Y = torch.vstack(Y_rows)  # [N, C_task]

        # Build targets + row mask
        if concept_family.mode == "logits":
            T_np, row_mask_np = concept_family.targets(Z, Y)  # [N', C_concepts], [N]
            row_mask_t = torch.from_numpy(row_mask_np.astype(bool))
            H_use = H[row_mask_t]
        else:
            # data-based: T computed during the loop to match order
            if len(T_rows) == 0:
                # empty loader
                return (torch.zeros(0, 0, device=device),
                        torch.zeros(0, device=device),
                        [],
                        [])
            T = torch.vstack(T_rows)  # [N, C_concepts]
            row_mask_np = np.ones(T.shape[0], dtype=bool)
            T_np = T.numpy()
            H_use = H

        if H_use.numel() == 0 or T_np.shape[0] == 0:
            return (torch.zeros(0, 0, device=device),
                    torch.zeros(0, device=device),
                    [],
                    [])

        # Prepare X, targets
        X = H_use.numpy()
        C_concepts = T_np.shape[1]

        # Standardize embeddings (important for well-scaled weights)
        if standardize:
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xs = scaler.fit_transform(X)  # (H - mean)/std
            mean = scaler.mean_.astype(np.float64)
            std = scaler.scale_.astype(np.float64)
            std[std == 0.0] = 1.0
        else:
            Xs = X
            mean = np.zeros(X.shape[1], dtype=np.float64)
            std = np.ones(X.shape[1], dtype=np.float64)

        W_orig = np.zeros((C_concepts, X.shape[1]), dtype=np.float64)
        b_orig = np.zeros((C_concepts,), dtype=np.float64)
        aucs: List[float] = []
        active: List[bool] = []

        # Fit one-vs-rest logistic probes
        for c in range(C_concepts):
            y = T_np[:, c].astype(np.int32)
            pos = int(y.sum())
            neg = int(len(y) - pos)

            # Degenerate or too few positives/negatives for a stable probe
            if pos == 0 or pos == len(y) or pos < min_pos or neg < min_neg:
                aucs.append(float("nan"))
                active.append(False)
                continue

            clf = LogisticRegression(
                penalty="l2", C=C_reg, solver="liblinear",
                class_weight=class_weight, max_iter=max_iter, random_state=0
            )
            clf.fit(Xs, y)

            # AUC on the same split (reporting metric; not for model selection)
            try:
                proba = clf.predict_proba(Xs)[:, 1]
                auc = roc_auc_score(y, proba)
            except Exception:
                auc = float("nan")

            # Map weights back to the ORIGINAL embedding space:
            #   logit = w_std^T * ((x - mean) / std) + b_std
            #        = (w_std/std)^T x + [b_std - (w_std/std)^T mean]
            w_std = clf.coef_.ravel().astype(np.float64)
            b_std = float(clf.intercept_.ravel()[0])
            w = w_std / std
            b = b_std - np.dot(w, mean)

            W_orig[c, :] = w
            b_orig[c] = b
            aucs.append(float(auc))
            active.append(bool(auc >= auroc_thresh))

        W_t = torch.from_numpy(W_orig).float().to(device)
        b_t = torch.from_numpy(b_orig).float().to(device)
        return W_t, b_t, aucs, active

    def compute_centroid_features(
            self,
            test_loader: DataLoader,
            centroid_mode: str,  # "pred" | "gt" | "inter"
            one_hot_only: bool,
            active_thresh: float = 0.5,
            min_n: int = 30
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[bool]
    ]:
        self.model.eval()

        C = self.model.out_dim

        embeddings_by_centroid: Dict[int, List[torch.Tensor]] = {c_n : [] for c_n in range(C)}
        logits_by_centroid: Dict[int, List[torch.Tensor]] = {c_n : [] for c_n in range(C)}
        labels_by_centroid: Dict[int, List[torch.Tensor]] = {c_n : [] for c_n in range(C)}

        centroids = None  # [C, D]
        centroid_sums = None  # [C, D]
        counts = None  # [C]

        with torch.inference_mode():
            for data in test_loader:
                data = data.to(self.model.device, non_blocking=True)
                logits, graph_repr = self.model.forward(
                    data.x, data.edge_index, data.batch, return_repr=True
                )  # logits: [B,C], graph_repr: [B,D]

                B, C = logits.shape
                D = graph_repr.size(1)
                targets = data.y.float().view(B, C)
                preds = (torch.sigmoid(logits) > 0.5).to(targets.dtype)

                if centroids is None:
                    centroids = torch.zeros(C, D, device=self.model.device)
                    centroid_sums = torch.zeros(C, D, device=self.model.device)
                    counts = torch.zeros(C, dtype=torch.long, device=self.model.device)

                # helpers to append aligned rows into dicts on CPU
                def _append_rows(c_idx_tensor: torch.Tensor, emb_rows: torch.Tensor,
                                 logit_rows: torch.Tensor, label_rows: torch.Tensor):
                    # c_idx_tensor: [M] class indices for each row in emb_rows/logit_rows/label_rows
                    for c_idx in c_idx_tensor.unique():
                        sel = (c_idx_tensor == c_idx)
                        c = int(c_idx.item())
                        embeddings_by_centroid[c].append(emb_rows[sel].detach().cpu())
                        logits_by_centroid[c].append(logit_rows[sel].detach().cpu())
                        labels_by_centroid[c].append(label_rows[sel].detach().cpu())

                if centroid_mode == "pr":
                    if one_hot_only:
                        mask = preds.sum(dim=1) == 1
                        if mask.any():
                            idx = preds[mask].argmax(dim=1)  # [M]
                            emb = graph_repr[mask]  # [M,D]
                            log = logits[mask]  # [M,C]
                            lab = targets[mask]  # [M,C]
                            centroid_sums.index_add_(0, idx, emb)
                            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
                            _append_rows(idx, emb, log, lab)
                    else:
                        # multi-label: add each embedding to all predicted classes
                        for c in range(C):
                            m = preds[:, c] == 1
                            if m.any():
                                e = graph_repr[m]
                                centroid_sums[c].add_(e.sum(dim=0))
                                counts[c] += m.sum()
                                # replicate class index for each selected row
                                idx = torch.full((int(m.sum().item()),), c, device=self.model.device, dtype=torch.long)
                                _append_rows(idx, e, logits[m], targets[m])

                elif centroid_mode == "gt":
                    if one_hot_only:
                        pure_mask = targets.sum(dim=1) == 1
                        if pure_mask.any():
                            idx = targets[pure_mask].argmax(dim=1)  # [M]
                            emb = graph_repr[pure_mask]  # [M,D]
                            log = logits[pure_mask]  # [M,C]
                            lab = targets[pure_mask]  # [M,C]
                            centroid_sums.index_add_(0, idx, emb)
                            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
                            _append_rows(idx, emb, log, lab)
                    else:
                        for c in range(C):
                            m = targets[:, c] == 1
                            if m.any():
                                e = graph_repr[m]
                                centroid_sums[c].add_(e.sum(dim=0))
                                counts[c] += m.sum()
                                idx = torch.full((int(m.sum().item()),), c, device=self.model.device, dtype=torch.long)
                                _append_rows(idx, e, logits[m], targets[m])

                elif centroid_mode == "in":
                    if one_hot_only:
                        gt_one = targets.sum(dim=1) == 1
                        pr_one = preds.sum(dim=1) == 1
                        both_one = gt_one & pr_one
                        if both_one.any():
                            gt_idx = targets[both_one].argmax(dim=1)  # [M]
                            pr_idx = preds[both_one].argmax(dim=1)  # [M]
                            agree = (gt_idx == pr_idx)
                            if agree.any():
                                idx = gt_idx[agree]  # agreed class indices
                                sel = torch.zeros_like(both_one)
                                sel[both_one] = agree
                                emb = graph_repr[sel]
                                log = logits[sel]
                                lab = targets[sel]
                                centroid_sums.index_add_(0, idx, emb)
                                counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
                                _append_rows(idx, emb, log, lab)
                    else:
                        # add to classes where both GT and Pred are positive (multi-label allowed)
                        for c in range(C):
                            m = (targets[:, c] == 1) & (preds[:, c] == 1)
                            if m.any():
                                e = graph_repr[m]
                                centroid_sums[c].add_(e.sum(dim=0))
                                counts[c] += m.sum()
                                idx = torch.full((int(m.sum().item()),), c, device=self.model.device, dtype=torch.long)
                                _append_rows(idx, e, logits[m], targets[m])

                else:
                    raise ValueError(f"Unknown centroid mode: {centroid_mode}")

        grouped_embeddings, grouped_logits, grouped_labels = [], [], []
        for c_n in range(C):
            if len(embeddings_by_centroid[c_n]) == 0:
                grouped_embeddings.append([])
                grouped_logits.append([])
                grouped_labels.append([])
            else:
                grouped_embeddings.append(torch.vstack(embeddings_by_centroid[c_n]))
                grouped_logits.append(torch.vstack(logits_by_centroid[c_n]))
                grouped_labels.append(torch.vstack(labels_by_centroid[c_n]))

        if centroid_mode == "gt":
            active_mask = []

            for class_idx, M in enumerate(grouped_logits):

                preds = (torch.sigmoid(M) > 0.5).float()
                mean_preds = preds.mean(axis=0)

                is_active = True

                for k, v in enumerate(mean_preds):
                    if k == class_idx and v < 1 - active_thresh:
                        is_active = False
                    if k != class_idx and v >= active_thresh:
                        is_active = False

                active_mask.append(is_active)
        else:
            active_mask = [len(M) > min_n for M in grouped_embeddings]

        if centroids is None:
            return None, grouped_embeddings, grouped_logits, grouped_labels, active_mask

        denom = counts.clamp_min(1).to(centroids.dtype).unsqueeze(1)  # [C,1]
        centroids = centroid_sums / denom

        return centroids, grouped_embeddings, grouped_logits, grouped_labels, active_mask

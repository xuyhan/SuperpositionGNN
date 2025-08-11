import hashlib
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import torch
from torch_geometric.data import Data


class SyntheticDataset(ABC):

    def __init__(self, n_train: int, n_test: int):
        self.n_train = n_train
        self.n_test = n_test

        self.train_data: List[Data] | None = None
        self.test_data: List[Data] | None = None

    def generate(self):
        self.train_data, self.test_data = self._generate_and_split()

    @abstractmethod
    def _generate_and_split(self, max_tries: Optional[int] = None) -> Tuple[List[Data], List[Data]]:
        pass


class SyntheticChainDataset(SyntheticDataset):
    """
    Generates chain graphs where y[k] = 1 iff there exists an edge (u,v)
    such that x[u,k] == x[v,k] == 1.
    """

    def __init__(self, num_categories: int = 3, p: float = 0.25, num_nodes: int = 20, seed: Optional[int] = None,
                 **kwargs):

        super().__init__(**kwargs)

        self.num_categories = num_categories
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0,1].")
        self.num_categories = int(num_categories)
        self.p = float(p)
        self.num_nodes = int(num_nodes)

        # Reproducibility
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        self._seed = seed

        # Precompute fixed chain edges (with self-loops, undirected as two directed edges).
        self._edge_index = self._create_chain_edge_index(self.num_nodes)

    # ------------------------------
    # Core graph construction
    # ------------------------------

    @staticmethod
    def _create_chain_edge_index(num_nodes: int) -> torch.Tensor:
        edges = []
        for i in range(num_nodes):
            edges.append((i, i))  # self-loop
            if i > 0:
                edges.append((i, i - 1))
            if i < num_nodes - 1:
                edges.append((i, i + 1))
        return torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]

    def _sample_node_features(self) -> torch.Tensor:
        """
        Returns:
            x: [N, C] float tensor with rows either all-zeros or one-hot.
        """
        N, C, p = self.num_nodes, self.num_categories, self.p
        x = torch.zeros((N, C), dtype=torch.float32)

        # Which nodes get activated?
        active = torch.rand(N) < p
        num_active = int(active.sum().item())
        if num_active > 0:
            # Assign a random category for each active node
            cats = torch.randint(low=0, high=C, size=(num_active,))
            x[active, :] = 0.0
            x[active, cats] = 1.0
        return x

    def _compute_label(self, edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of y in [0,1]^C as described.
        """
        src, dst = edge_index[0], edge_index[1]
        non_self = src != dst
        src, dst = src[non_self], dst[non_self]

        # For each edge, a [C]-vector marking categories present on both ends
        both_active = (x[src].bool() & x[dst].bool())  # [E, C]
        y = both_active.any(dim=0).float()  # [C]
        return y

    # ------------------------------
    # Public API
    # ------------------------------

    def generate_unique_dataset(self, num_samples: int, max_tries: Optional[int] = None) -> List[Data]:
        """
        Generates 'num_samples' unique graphs (no duplicate node-feature patterns).

        Dedup key = SHA1 of x (after converting to uint8). Edge pattern is fixed.
        """
        if max_tries is None:
            max_tries = max(num_samples * 20, 10_000)  # generous but finite

        edge_index = self._edge_index
        seen = set()
        out: List[Data] = []

        tries = 0
        while len(out) < num_samples and tries < max_tries:
            tries += 1
            x = self._sample_node_features()
            key = hashlib.sha1(x.to(torch.uint8).contiguous().numpy().tobytes()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            y = self._compute_label(edge_index, x)
            data = Data(x=x, edge_index=edge_index, y=y)
            data.num_nodes = self.num_nodes
            out.append(data)

        if len(out) < num_samples:
            raise RuntimeError(f"Could not generate {num_samples} unique samples after {tries} tries. "
                               f"Try decreasing p, num_samples, or increasing max_tries.")
        return out

    @staticmethod
    def _train_test_split(dataset: List[Data], test_ratio: float = 0.33, seed: Optional[int] = None) -> Tuple[
        List[Data], List[Data]]:
        """
        Splits one dataset into train/test with no overlap.
        """
        if not (0.0 < test_ratio < 1.0):
            raise ValueError("test_ratio must be in (0,1).")
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        idx = torch.randperm(len(dataset)).tolist()
        cut = int(round(len(dataset) * (1.0 - test_ratio)))
        train_idx, test_idx = idx[:cut], idx[cut:]
        train = [dataset[i] for i in train_idx]
        test = [dataset[i] for i in test_idx]
        return train, test

    def _generate_and_split(self, max_tries: Optional[int] = None) -> Tuple[List[Data], List[Data]]:

        full = self.generate_unique_dataset(self.n_train + self.n_test, max_tries=max_tries)
        test_ratio = self.n_test / (self.n_train + self.n_test)
        return self._train_test_split(full, test_ratio=test_ratio, seed=self._seed)


class SharedDataset:
    PAIRWISE_12 = SyntheticChainDataset(num_categories=12, p=0.9, num_nodes=20, seed=42, n_train=10000, n_test=5000)


SharedDataset.PAIRWISE_12.generate()

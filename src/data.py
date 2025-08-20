import hashlib
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Sequence, Literal

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree, add_self_loops

try:
    import networkx as nx
    from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl_hash
except Exception as e:
    raise ImportError(
        "This dataset requires networkx (>=2.6) for WL hashing. "
        "Install with `pip install networkx`."
    ) from e


# ---------- utilities ----------

def _dense_adj_from_edge_index(edge_index: torch.Tensor,
                               num_nodes: int,
                               add_sl: bool = True,
                               make_undirected: bool = True) -> torch.Tensor:
    """
    Build a (num_nodes x num_nodes) dense adjacency from edge_index.
    Assumes unweighted edges. Optionally add self-loops and symmetrize.
    """
    ei = edge_index
    if make_undirected:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    if add_sl:
        ei, _ = add_self_loops(ei, num_nodes=num_nodes)
    A = to_dense_adj(ei, max_num_nodes=num_nodes).squeeze(0).to(torch.float32)  # [N, N]
    A.fill_diagonal_(1.0 if add_sl else A.diagonal())  # ensure exact SLs
    return A


# ---------- (1) Degree features ----------

def degree_features(edge_index: torch.Tensor,
                    num_nodes: int,
                    add_sl: bool = True,
                    normalize: Optional[Literal["log", "sqrt", "none"]] = "none") -> torch.Tensor:
    """
    Returns X_deg: [N, 1] with degree (incl. self-loop if add_sl=True).
    """
    row = edge_index[0]
    if add_sl:
        row = torch.cat([row, torch.arange(num_nodes, device=row.device)])
    deg = degree(row, num_nodes=num_nodes).to(torch.float32)  # [N]

    if normalize == "log":
        x = torch.log1p(deg).unsqueeze(1)
    elif normalize == "sqrt":
        x = torch.sqrt(deg.clamp_min(0)).unsqueeze(1)
    else:
        x = deg.unsqueeze(1)
    return x


# ---------- (2) RWPE (return probabilities) ----------

def rwpe_diag_return(edge_index: torch.Tensor,
                     num_nodes: int,
                     K: int = 4,
                     add_sl: bool = True) -> torch.Tensor:
    """
    Random-Walk Positional Encoding: K channels per node as the diagonal
    of P^t (return probability after t steps), t=1..K, where P is row-stochastic.
    Implementation uses dense matrices; fine for small synthetic graphs.
    """
    A = _dense_adj_from_edge_index(edge_index, num_nodes, add_sl=add_sl, make_undirected=True)  # [N, N]
    degs = A.sum(dim=1).clamp_min(1e-8)  # includes SLs
    P = A / degs.unsqueeze(1)            # row-normalized

    N = num_nodes
    X = torch.zeros(N, K, dtype=torch.float32, device=A.device)
    M = P.clone()
    for t in range(K):
        X[:, t] = torch.diagonal(M)      # diag(P^t)
        M = M @ P
    return X  # [N, K]


# ---------- (3) Laplacian PE ----------

def lappe(edge_index: torch.Tensor,
          num_nodes: int,
          dim: int = 4,
          add_sl: bool = True) -> torch.Tensor:
    """
    LapPE: first `dim` non-trivial eigenvectors of normalized Laplacian.
    Handles multiple zero eigenvalues (disconnected graphs) by skipping them.
    Returns X: [N, dim]. If not enough non-trivial eigenvectors, pads with zeros.
    """
    A = _dense_adj_from_edge_index(edge_index, num_nodes, add_sl=add_sl, make_undirected=True)
    D = A.sum(dim=1).clamp_min(1e-8)
    Dmh = torch.diag(1.0 / torch.sqrt(D))
    An = Dmh @ A @ Dmh
    L = torch.eye(num_nodes, device=A.device, dtype=torch.float32) - An  # normalized Laplacian

    # eigh is symmetric eigendecomposition; safer than svd here
    evals, evecs = torch.linalg.eigh(L)  # ascending order

    # skip all ~zero eigenvalues (connected components)
    tol = 1e-6
    nontrivial = (evals > tol).nonzero(as_tuple=False).squeeze(-1)
    X = torch.zeros(num_nodes, dim, dtype=torch.float32, device=A.device)
    if nontrivial.numel() == 0:
        return X  # totally empty after removing components; pad zeros

    take = min(dim, nontrivial.numel())
    cols = nontrivial[:take]
    V = evecs[:, cols]  # [N, take]

    # sign disambiguation: flip each vector so that its largest-magnitude entry is positive
    idx = V.abs().argmax(dim=0)                  # [take]
    signs = torch.sign(V.gather(0, idx.unsqueeze(0)).squeeze(0)).clamp(min=1e-8)
    V = V * signs
    X[:, :take] = V
    return X  # [N, dim]


# ---------- Dispatcher ----------

def structural_node_features(edge_index: torch.Tensor,
                             num_nodes: int,
                             kind: Literal["deg", "rwpe", "lappe", "rand"] = "deg",
                             *,
                             add_self_loops: bool = True,
                             deg_norm: Optional[str] = "none",
                             rwpe_K: int = 4,
                             lappe_dim: int = 4,
                             random_dim: int = 4) -> torch.Tensor:
    """
    Main entry point. Returns node feature matrix X: [N, D_struct].
    """
    if kind == "deg":
        return degree_features(edge_index, num_nodes, add_sl=add_self_loops, normalize=deg_norm)
    elif kind == "rwpe":
        return rwpe_diag_return(edge_index, num_nodes, K=rwpe_K, add_sl=add_self_loops)
    elif kind == "lappe":
        return lappe(edge_index, num_nodes, dim=lappe_dim, add_sl=add_self_loops)
    elif kind == "rand":
        return torch.randn(num_nodes, random_dim, device=edge_index.device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown structural feature kind: {kind}")


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

    def __init__(self, num_categories: int = 3, p: float = 0.25, num_nodes: int = 20, seed: Optional[int] = None, add_self_loops: bool = False,
                 **kwargs):

        super().__init__(**kwargs)

        self.num_categories = num_categories
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0,1].")
        self.num_categories = int(num_categories)
        self.p = float(p)
        self.num_nodes = int(num_nodes)
        self.add_self_loops = add_self_loops

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
            if add_self_loops:
                edges.append((i, i))
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

# ---------------------------------------------------------
# Conjunction dataset with *class* & *concept* level priors
# ---------------------------------------------------------
class SyntheticConjunctionCyclesDataset(SyntheticDataset):
    """
    Targets:
      A := (has at least one C3) AND (has at least one C6)
      B := (has at least one C4) AND (has at least one C5)

    Label categories (multilabel):
      'none'   -> [0,0]
      'A_only' -> [1,0]
      'B_only' -> [0,1]
      'both'   -> [1,1]

    Two-level sampling:
      1) Choose category from class_prior.
      2) Sample the *counts* of cycles (C3,C4,C5,C6) from concept priors,
         *conditioned on the chosen category* (enforce constraints).
      3) Build a concrete graph (cycles + random bridges + whiskers) and
         WL-dedup with bucket-specific rejection sampling.

    This prevents WL-uniqueness from skewing concept frequencies.
    """

    def __init__(
        self,
        *,
        n_train: int,
        n_test: int,
        seed: Optional[int] = None,

        # -------- class-level balance
        class_prior: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        ensure_unique: bool = True,
        max_tries_per_bucket: int = 60_000,

        # -------- concept-level priors (extra copies beyond minima)
        # mode: "bernoulli" = each concept independently included extras with prob p_extra[len]
        #       "poisson"   = extras ~ Poisson(lambda_extra[len])
        concept_prior_mode: str = "bernoulli",
        p_extra: Optional[Dict[int, float]] = None,          # for "bernoulli"
        lambda_extra: Optional[Dict[int, float]] = None,     # for "poisson"
        max_extra_copies_per_concept: int = 3,

        # whether singletons of target concepts are allowed in 'none':

        # -------- variety knobs
        connect_components: bool = True,
        bridge_len_range: Tuple[int, int] = (1, 3),          # inclusive
        whiskers_per_anchor_range: Tuple[int, int] = (0, 2), # inclusive
        whisker_len_range: Tuple[int, int] = (1, 3),         # inclusive
        decoy_cycle_lengths: Tuple[int, ...] = (7, 8),       # extra cycles not in {3,4,5,6}
        decoy_bernoulli_p: float = 0.2,                      # chance to add each decoy length

        # -------- self-loops
        add_self_loops_flag: bool = False,

        # -------- node embeddings
        node_feat_kind: Literal["deg", "rwpe", "lappe", "const", "rand"] = "deg",
        node_feat_kwargs: Optional[Dict] = None,
    ):
        super().__init__(n_train=n_train, n_test=n_test)

        self.seed = seed
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.class_prior = self._normalize_prior(class_prior)
        self.ensure_unique = bool(ensure_unique)
        self.max_tries_per_bucket = int(max_tries_per_bucket)

        # concept prior config
        self.concept_prior_mode = concept_prior_mode
        self.max_extra_copies_per_concept = int(max_extra_copies_per_concept)

        # default symmetric priors if none given
        self.p_extra = {3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5} if p_extra is None else dict(p_extra)
        self.lambda_extra = {3: 0.6, 4: 0.6, 5: 0.6, 6: 0.6} if lambda_extra is None else dict(lambda_extra)

        # topology variety
        self.connect_components = bool(connect_components)
        self.bridge_len_range = bridge_len_range
        self.whiskers_per_anchor_range = whiskers_per_anchor_range
        self.whisker_len_range = whisker_len_range
        self.decoy_cycle_lengths = tuple(int(x) for x in decoy_cycle_lengths)
        self.decoy_bernoulli_p = float(decoy_bernoulli_p)

        self.add_self_loops_flag = bool(add_self_loops_flag)

        # fixed pairs
        self.A_pair = (3, 6)
        self.B_pair = (4, 5)

        self.node_feat_kind = node_feat_kind
        self.node_feat_kwargs = node_feat_kwargs or {}

    # --------------------- public API ---------------------

    def _generate_and_split(self, max_tries: Optional[int] = None) -> Tuple[List[Data], List[Data]]:
        total = self.n_train + self.n_test
        targets = self._target_counts_from_prior(total, self.class_prior)

        # fill each bucket with unique graphs (class+concept fixed before WL-reject)
        buckets = self._fill_buckets_with_uniques(targets)

        # per-bucket split preserves class balance
        train, test = self._per_bucket_split(buckets, train_total=self.n_train, test_total=self.n_test)
        return train, test

    # ---------------- bucket filling ---------------------

    def _fill_buckets_with_uniques(self, targets: Dict[str, int]) -> Dict[str, List[Data]]:
        buckets = {k: [] for k in targets}
        seen_hashes: set[str] = set() if self.ensure_unique else set()

        fill_order = ["none", "A_only", "B_only", "both"]  # deterministic order
        for cat in fill_order:
            need = targets[cat]
            tries = 0
            while len(buckets[cat]) < need and tries < self.max_tries_per_bucket:
                tries += 1

                # (1) sample concept counts conditioned on category
                counts = self._sample_concept_counts_conditioned(cat)

                # (2) materialize a concrete graph (cycles + bridges + whiskers + optional decoys)
                data = self._build_graph_from_counts(counts, category=cat)

                # (3) WL-dedup
                h = self._wl_hash(data.edge_index, data.num_nodes)
                if (not self.ensure_unique) or (h not in seen_hashes):
                    buckets[cat].append(data)
                    seen_hashes.add(h)

            if len(buckets[cat]) < need:
                raise RuntimeError(
                    f"Bucket '{cat}': could not reach {need} uniques after {tries} proposals. "
                    f"Loosen priors or range knobs."
                )
        return buckets

    # -------------- concept count sampler ----------------

    def _sample_concept_counts_conditioned(self, category: str) -> Dict[int, int]:

        req = {3: 0, 4: 0, 5: 0, 6: 0}

        if category == "A_only":
            req[3] = 1
            req[6] = 1

            # r = random.random()
            # if r < 1/3:
            #     req[4] = 1
            # elif r < 2/3:
            #     req[5] = 1
        elif category == "B_only":
            req[4] = 1
            req[5] = 1

            # r = random.random()
            # if r < 1/3:
            #     req[3] = 1
            # elif r < 2/3:
            #     req[6] = 1
        elif category == "both":
            req[3] = 1
            req[4] = 1
            req[5] = 1
            req[6] = 1
        else:
            r = random.random()

            if r < 0.25:
                req[3] = 1
                req[5] = 1
            elif r < 0.5:
                req[4] = 1
                req[6] = 1
            elif r < 0.75:
                req[3] = 1
                req[4] = 1
            else:
                req[5] = 1
                req[6] = 1

        # sample extras
        counts = dict(req)

        for L in (3, 4, 5, 6):
            if counts[L] == 0:
                continue
            p = self.p_extra.get(L, 0.3)
            for _ in range(self.max_extra_copies_per_concept):
                if random.random() < p:
                    counts[L] += 1

        return counts  # dict {3:int,4:int,5:int,6:int}

    # -------------- graph construction -------------------

    def _build_graph_from_counts(self, counts: Dict[int, int], category: str) -> Data:
        edges: List[Tuple[int, int]] = []
        next_node = 0

        def add_edge(u: int, v: int):
            edges.append((u, v));
            edges.append((v, u))

        def add_cycle(L: int) -> int:
            nonlocal next_node
            nodes = list(range(next_node, next_node + L))
            next_node += L
            for i in range(L):
                u, v = nodes[i], nodes[(i + 1) % L]
                add_edge(u, v)
            return nodes[0]  # anchor

        def add_whiskers(anchor: int):
            nonlocal next_node
            wmin, wmax = self.whiskers_per_anchor_range
            lmin, lmax = self.whisker_len_range
            n_wh = random.randint(wmin, wmax)
            for _ in range(n_wh):
                length = random.randint(lmin, lmax)
                prev = anchor
                for _ in range(length):
                    cur = next_node;
                    next_node += 1
                    add_edge(prev, cur);
                    prev = cur

        def connect_with_path(u: int, v: int, length: int):
            nonlocal next_node
            if length <= 0:
                add_edge(u, v);
                return
            prev = u
            for _ in range(max(0, length - 1)):
                cur = next_node;
                next_node += 1
                add_edge(prev, cur);
                prev = cur
            add_edge(prev, v)

        anchors: List[int] = []

        # target cycles
        for L in (3, 4, 5, 6):
            for _ in range(counts[L]):
                a = add_cycle(L)
                anchors.append(a)
                add_whiskers(a)

        # decoy cycles (do not form target pairs)
        for Ld in self.decoy_cycle_lengths:
            if random.random() < self.decoy_bernoulli_p:
                a = add_cycle(Ld)
                anchors.append(a)
                add_whiskers(a)

        # If still no structure, force a tiny neutral structure (2-node path)
        if next_node == 0:
            u, v = 0, 1
            next_node = 2
            add_edge(u, v)

        # Connect components if requested
        if self.connect_components and len(anchors) >= 2:
            random.shuffle(anchors)
            for i in range(len(anchors) - 1):
                u, v = anchors[i], anchors[i + 1]
                connect_with_path(u, v, random.randint(*self.bridge_len_range))

        # Optional self-loops
        if self.add_self_loops_flag:
            for i in range(next_node):
                edges.append((i, i))

        # Build a well-formed edge_index of shape [2, E]
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        if self.node_feat_kind == "const":
            x = torch.ones(next_node, 1, dtype=torch.float32)
        else:
            x = structural_node_features(
                edge_index=edge_index,
                num_nodes=next_node,
                kind=self.node_feat_kind,            # "deg" | "rwpe" | "lappe"
                **self.node_feat_kwargs,             # e.g., rwpe_K=4, lappe_dim=4, add_self_loops=True
            )

        # Label from counts (ground-truth semantics)
        A = int(counts[3] >= 1 and counts[6] >= 1)
        B = int(counts[4] >= 1 and counts[5] >= 1)
        y = torch.tensor([float(A), float(B)], dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = next_node
        return data

    # ----------------- split & helpers -------------------

    def _per_bucket_split(self, buckets: Dict[str, List[Data]], train_total: int, test_total: int) -> Tuple[List[Data], List[Data]]:
        # proportional per-bucket split to preserve class balance
        total = sum(len(v) for v in buckets.values())
        if total != (train_total + test_total):
            raise RuntimeError(f"Internal mismatch: {total} != {train_total + test_total}")

        rng = random.Random(self.seed)
        train, test = [], []
        for cat, graphs in buckets.items():
            idx = list(range(len(graphs)))
            rng.shuffle(idx)
            t_count = round(train_total * (len(graphs) / total))
            t_idx, s_idx = idx[:t_count], idx[t_count:]
            train.extend([graphs[i] for i in t_idx])
            test.extend([graphs[i] for i in s_idx])

        # fix rounding drift
        if len(train) != train_total:
            diff = train_total - len(train)
            if diff > 0:
                test_to_train = test[:diff]
                train += test_to_train
                test = test[diff:]
            else:
                train_to_test = train[diff:]
                test = train_to_test + test
                train = train[:diff]
        return train, test

    @staticmethod
    def _normalize_prior(prior: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        s = sum(prior)
        if s <= 0:
            raise ValueError("class_prior must sum to positive.")
        return tuple(p / s for p in prior)

    @staticmethod
    def _target_counts_from_prior(total: int, prior: Tuple[float, float, float, float]) -> Dict[str, int]:
        raw = [p * total for p in prior]
        rounded = [int(round(x)) for x in raw]
        diff = total - sum(rounded)
        if diff != 0:
            fracs = [x - int(x) for x in raw]
            order = sorted(range(4), key=lambda i: fracs[i], reverse=(diff > 0))
            i = 0
            while diff != 0:
                idx = order[i % 4]
                if diff > 0:
                    rounded[idx] += 1; diff -= 1
                else:
                    if rounded[idx] > 0:
                        rounded[idx] -= 1; diff += 1
                i += 1
        keys = ["none", "A_only", "B_only", "both"]
        return {k: rounded[i] for i, k in enumerate(keys)}

    @staticmethod
    def _wl_hash(edge_index: torch.Tensor, num_nodes: int, iters: int = 3) -> str:
        # Guard: empty graph
        if num_nodes <= 0:
            return hashlib.sha1(b"wl-empty-graph").hexdigest()

        # Normalize edge_index view: allow empty/1-D tensors → treat as 0 edges
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.size(0) != 2:
            E = 0
            src = dst = None
        else:
            E = int(edge_index.size(1))
            src = edge_index[0]
            dst = edge_index[1]

        # Build adjacency
        adj = [[] for _ in range(num_nodes)]
        if E > 0:
            for e in range(E):
                u = int(src[e]);
                v = int(dst[e])
                if 0 <= u < num_nodes and 0 <= v < num_nodes:
                    adj[u].append(v)

        # 1-WL refinement
        colors = [len(adj[v]) for v in range(num_nodes)]
        for _ in range(iters):
            new_colors = []
            for v in range(num_nodes):
                neigh = sorted(colors[w] for w in adj[v])
                new_colors.append(hash((colors[v], tuple(neigh))))
            # compress
            uniq = {c: i for i, c in enumerate(sorted(set(new_colors)))}
            colors = [uniq[c] for c in new_colors]

        hist = {}
        for c in colors:
            hist[c] = hist.get(c, 0) + 1
        raw = repr(sorted(hist.items())).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()



# =========================================================
# New: Motifs (C3, C4, C5) with label+N-first sampling
# =========================================================

LabelTuple = Tuple[int, int, int]  # (has C3, has C4, has C5)

class SyntheticMotifsDataset(SyntheticDataset):
    """
    Motif family: cycles of lengths 3, 4, 5.
    Multi-label y ∈ {0,1}^3 indicates presence of each motif.
    Generation protocol:
      1) Sample a label y from either a joint prior over 8 tuples, or independent marginals.
      2) Sample a total node budget N from a label-independent prior.
      3) Materialize a graph that (a) satisfies y, (b) uses exactly N nodes
         (add neutral fillers if short; reject if we overshoot), and
         (c) avoids duplicates by 1-WL hash within each (y, N) bucket.
    """

    def __init__(
        self,
        *,
        n_train: int,
        n_test: int,
        seed: Optional[int] = None,

        # ----- label prior -----
        # Either specify 'joint_prior' over all 8 label tuples (dict mapping 3-bit tuples to probs),
        # OR leave it None and use 'marginal_p' (independent Bernoulli per motif).
        joint_prior: Optional[Dict[LabelTuple, float]] = None,
        marginal_p: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        allow_all_zero: bool = True,   # if False, resample until at least one motif present

        # ----- node-count prior (independent of label) -----
        num_nodes_range: Tuple[int, int] = (24, 48),  # inclusive range for N
        # or provide a discrete list via num_nodes_support; if given it overrides the range
        num_nodes_support: Optional[Sequence[int]] = None,

        # ----- variety knobs -----
        connect_components: bool = True,
        bridge_len_range: Tuple[int, int] = (1, 4),          # inclusive
        whiskers_per_anchor_range: Tuple[int, int] = (0, 3), # inclusive
        whisker_len_range: Tuple[int, int] = (1, 4),         # inclusive
        decoy_cycle_lengths: Tuple[int, ...] = (7, 8),       # non-target cycle lengths
        decoy_bernoulli_p: float = 0.3,                      # chance to add each decoy length

        ensure_unique: bool = True,
        max_tries_per_bucket: int = 60_000,

        # ----- self-loops & node features -----
        add_self_loops_flag: bool = False,
        node_feat_kind: Literal["deg", "rwpe", "lappe", "const"] = "deg",
        node_feat_kwargs: Optional[Dict] = None,
    ):
        super().__init__(n_train=n_train, n_test=n_test)
        self.seed = seed
        if seed is not None:
            random.seed(seed); torch.manual_seed(seed)

        # label prior
        self.joint_prior = None if joint_prior is None else self._normalize_joint(joint_prior)
        self.marginal_p = tuple(float(p) for p in marginal_p)
        self.allow_all_zero = bool(allow_all_zero)

        # N prior
        if num_nodes_support is not None:
            self.N_support = list(int(n) for n in num_nodes_support)
            if not self.N_support:
                raise ValueError("num_nodes_support cannot be empty.")
        else:
            a, b = num_nodes_range
            if not (isinstance(a, int) and isinstance(b, int) and 4 <= a <= b):
                raise ValueError("num_nodes_range must be integers with 4 <= min <= max.")
            self.N_support = list(range(a, b + 1))

        # variety
        self.connect_components = bool(connect_components)
        self.bridge_len_range = tuple(int(x) for x in bridge_len_range)
        self.whiskers_per_anchor_range = tuple(int(x) for x in whiskers_per_anchor_range)
        self.whisker_len_range = tuple(int(x) for x in whisker_len_range)
        self.decoy_cycle_lengths = tuple(int(x) for x in decoy_cycle_lengths)
        self.decoy_bernoulli_p = float(decoy_bernoulli_p)

        self.ensure_unique = bool(ensure_unique)
        self.max_tries_per_bucket = int(max_tries_per_bucket)

        self.add_self_loops_flag = bool(add_self_loops_flag)
        self.node_feat_kind = node_feat_kind
        self.node_feat_kwargs = node_feat_kwargs or {}

    # ---------------- public API ----------------

    def _generate_and_split(self) -> Tuple[List[Data], List[Data]]:
        total = self.n_train + self.n_test
        # sample label list and N list independently, then zip them
        labels = self._sample_labels(total)
        Ns     = self._sample_node_counts(total)

        # Fill buckets keyed by (label, N)
        buckets: Dict[Tuple[LabelTuple, int], List[Data]] = {}
        seen: set[str] = set() if self.ensure_unique else set()
        tries = { }  # diagnostics

        for y, N in zip(labels, Ns):
            key = (y, N)
            buckets.setdefault(key, [])
            # propose until you get a WL-unique graph that meets (y,N)
            t = 0
            while t < self.max_tries_per_bucket:
                t += 1
                data = self._sample_graph_with_label_and_nodes(y, N)
                h = SyntheticConjunctionCyclesDataset._wl_hash(data.edge_index, data.num_nodes)
                if (not self.ensure_unique) or (h not in seen):
                    buckets[key].append(data)
                    seen.add(h)
                    break
            tries[key] = t

        # Flatten & split preserving global order
        graphs = [g for lst in buckets.values() for g in lst]
        if len(graphs) != total:
            raise RuntimeError(f"Internal mismatch: generated {len(graphs)} != requested {total} "
                               f"(increase max_tries or adjust priors).")

        # simple split (random)
        idx = torch.randperm(total).tolist()
        train_idx, test_idx = idx[:self.n_train], idx[self.n_train:]
        train = [graphs[i] for i in train_idx]
        test  = [graphs[i] for i in test_idx]
        return train, test

    # ---------------- sampling primitives ----------------

    def _sample_labels(self, n: int) -> List[LabelTuple]:
        out: List[LabelTuple] = []
        if self.joint_prior is not None:
            # categorical over 8 tuples
            tuples = list(self.joint_prior.keys())
            probs  = torch.tensor([self.joint_prior[t] for t in tuples], dtype=torch.float)
            probs  = probs / probs.sum()
            idx = torch.multinomial(probs, n, replacement=True).tolist()
            out = [tuples[i] for i in idx]
        else:
            p3, p4, p5 = self.marginal_p
            for _ in range(n):
                while True:
                    y = (int(random.random() < p3),
                         int(random.random() < p4),
                         int(random.random() < p5))
                    if self.allow_all_zero or (y != (0, 0, 0)):
                        out.append(y); break
        return out

    def _sample_node_counts(self, n: int) -> List[int]:
        # common N distribution independent of label
        return [random.choice(self.N_support) for _ in range(n)]

    # ---------------- graph construction -----------------

    def _sample_graph_with_label_and_nodes(self, y: LabelTuple, N_target: int) -> Data:
        """
        Build a graph that contains at least one C3 if y[0]=1, etc.,
        and uses exactly N_target nodes by adding neutral filler chains if needed.
        If we overshoot N_target during construction, we restart.
        """
        while True:
            edges: List[Tuple[int, int]] = []
            next_node = 0

            def add_edge(u: int, v: int):
                edges.append((u, v)); edges.append((v, u))

            def add_cycle(L: int) -> int:
                nonlocal next_node
                nodes = list(range(next_node, next_node + L))
                next_node += L
                for i in range(L):
                    add_edge(nodes[i], nodes[(i + 1) % L])
                return nodes[0]  # anchor

            def add_whiskers(anchor: int):
                nonlocal next_node
                wmin, wmax = self.whiskers_per_anchor_range
                lmin, lmax = self.whisker_len_range
                n_wh = random.randint(wmin, wmax)
                for _ in range(n_wh):
                    L = random.randint(lmin, lmax)
                    prev = anchor
                    for _ in range(L):
                        cur = next_node; next_node += 1
                        add_edge(prev, cur); prev = cur

            def connect_path(u: int, v: int, length: int):
                nonlocal next_node
                if length <= 0:
                    add_edge(u, v); return
                prev = u
                for _ in range(max(0, length - 1)):
                    cur = next_node; next_node += 1
                    add_edge(prev, cur); prev = cur
                add_edge(prev, v)

            anchors: List[int] = []

            # Required target cycles according to y
            want = {3: y[0], 4: y[1], 5: y[2]}
            for L in (3, 4, 5):
                if want[L]:
                    a = add_cycle(L); anchors.append(a); add_whiskers(a)

            # Optionally sprinkle a few decoy cycles (do NOT use 3/4/5 here)
            for Ld in self.decoy_cycle_lengths:
                if random.random() < self.decoy_bernoulli_p:
                    a = add_cycle(Ld); anchors.append(a); add_whiskers(a)

            # Ensure connectedness via bridges
            if self.connect_components and len(anchors) >= 2:
                random.shuffle(anchors)
                for i in range(len(anchors) - 1):
                    u, v = anchors[i], anchors[i + 1]
                    Lb = random.randint(*self.bridge_len_range)
                    connect_path(u, v, Lb)

            # If empty (possible if y=(0,0,0) and no decoys fired), create a tiny path
            if next_node == 0:
                add_edge(0, 1); next_node = 2

            # Now adjust to hit exactly N_target by adding neutral fillers (chains)
            def add_filler_nodes(k: int):
                nonlocal next_node
                # Attach each filler chain to a random existing node; use length-1 increments.
                for _ in range(k):
                    attach = random.randrange(next_node)
                    newn = next_node; next_node += 1
                    add_edge(attach, newn)

            if next_node > N_target:
                # overshoot -> restart (rare; controlled by your ranges)
                continue
            elif next_node < N_target:
                add_filler_nodes(N_target - next_node)

            # Self-loops (optional)
            if self.add_self_loops_flag:
                for i in range(N_target):
                    edges.append((i, i))

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty(2, 0, dtype=torch.long)

            # Node features
            if self.node_feat_kind == "const":
                x = torch.ones(N_target, 1)
            else:
                x = structural_node_features(
                    edge_index=edge_index, num_nodes=N_target,
                    kind=self.node_feat_kind, **self.node_feat_kwargs
                )

            y_t = torch.tensor([float(want[3]), float(want[4]), float(want[5])])
            data = Data(x=x, edge_index=edge_index, y=y_t)
            data.num_nodes = N_target
            return data

    # ---------------- helpers ----------------

    @staticmethod
    def _normalize_joint(joint: Dict[LabelTuple, float]) -> Dict[LabelTuple, float]:
        # fill missing tuples with 0, normalize to sum 1
        all_tuples = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        vals = torch.tensor([float(joint.get(t, 0.0)) for t in all_tuples], dtype=torch.float)
        if vals.sum() <= 0:
            raise ValueError("joint_prior must have positive mass.")
        vals = (vals / vals.sum()).tolist()
        return {t: v for t, v in zip(all_tuples, vals)}


class SharedDataset:
    PAIRWISE_12 = SyntheticChainDataset(num_categories=12, p=0.9, num_nodes=20, seed=42, n_train=2000, n_test=1000)
    PAIRWISE_16 = SyntheticChainDataset(num_categories=16, p=0.9, num_nodes=20, seed=42, n_train=2000, n_test=1000)
    CONJUNCTION =  SyntheticConjunctionCyclesDataset(
        n_train=3000, n_test=1000, seed=42,
        class_prior=(0.25, 0.25, 0.25, 0.25),

        concept_prior_mode="bernoulli",
        p_extra={3:5, 4:0.5, 5:0.5, 6:0.5},  # symmetric → triangles not suppressed
        max_extra_copies_per_concept=8,

        connect_components=True,
        bridge_len_range=(1,2),
        whiskers_per_anchor_range=(0,2),
        whisker_len_range=(1,2),
        decoy_cycle_lengths=(7, 8),
        decoy_bernoulli_p=0,

        add_self_loops_flag=False,

        node_feat_kind="deg"
    )
    MOTIFS = SyntheticMotifsDataset(
        n_train=3000, n_test=2400, seed=42,
        marginal_p=(0.4, 0.4, 0.4),     # co-activations allowed
        allow_all_zero=False,           # at least one motif present
        num_nodes_range=(20, 24),
        connect_components=True,
        bridge_len_range=(1, 4),
        whiskers_per_anchor_range=(0, 2),
        whisker_len_range=(1, 3),
        decoy_cycle_lengths=(7, 8),
        decoy_bernoulli_p=0.25,
        add_self_loops_flag=False,
        node_feat_kind="deg",
        node_feat_kwargs=dict(deg_norm="none")
    )


SharedDataset.CONJUNCTION.generate()
# SharedDataset.MOTIFS.generate()
# SharedDataset.PAIRWISE_12.generate()
SharedDataset.PAIRWISE_16.generate()

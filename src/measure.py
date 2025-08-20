import torch
from typing import Optional, Tuple

# ---------------------------
# Utilities
# ---------------------------
def _check_matrix(C: torch.Tensor) -> torch.Tensor:
    """Ensure C is 2-D (k × d); if empty, return (0,0) CPU float tensor."""
    if isinstance(C, list):  # safety fallback
        C = torch.stack(C, 0)
    if C.ndim != 2:
        raise ValueError(f"Expected 2-D tensor, got shape {tuple(C.shape)}")
    if C.numel() == 0:
        return torch.zeros(0, 0)
    return C

def _pairwise_mean_cos2(C: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Mean of squared cosines between all distinct row pairs.
    Works even if rows are not pre-normalised by dividing by norms per pair.
    """
    k, d = C.shape
    if k <= 1 or d == 0:
        return C.new_tensor(0.0)
    # Row norms and Gram
    n = C.norm(dim=1).clamp_min(eps)          # (k,)
    G = C @ C.t()                              # (k,k)
    # cos^2(i,j) = (G_ij / (||ci|| ||cj||))^2
    denom = torch.outer(n, n)                  # (k,k)
    cos2 = (G / denom).pow(2)
    iu = torch.triu_indices(k, k, offset=1)
    return cos2[iu[0], iu[1]].mean()

def _entropy_effrank(C: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Entropy-based effective rank: exp( H(p) ), p_i = s_i / sum_j s_j, s = singular values.
    """
    if C.numel() == 0:
        return 0.0
    s = torch.linalg.svdvals(C)
    ssum = s.sum()
    if ssum <= eps:
        return 0.0
    p = s / (ssum + eps)
    H = -(p * (p + eps).log()).sum()
    return float(torch.exp(H))

def _project_out_pc1(C: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Project out the top right-singular vector (PC1) from columns of C: return (C_perp, v1).
    """
    if C.numel() == 0:
        return C, None
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    if S.numel() == 0 or S[0] <= eps:
        return C, None
    v1 = Vh[0]  # (d,)
    P = torch.eye(C.shape[1], dtype=C.dtype, device=C.device) - torch.outer(v1, v1)
    return C @ P, v1

def _pc1_energy(C: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Fraction of spectral energy in PC1 (computed on RAW C, no centering):
      s1^2 / sum_i s_i^2
    """
    if C.numel() == 0:
        return 0.0
    s = torch.linalg.svdvals(C)
    denom = (s * s).sum()
    if denom <= eps:
        return 0.0
    return float((s[0] * s[0]) / denom)

# ---------------------------
# Superposition Index (SI)
# ---------------------------
def superposition_index(
    C: torch.Tensor,
    *,
    d_override: Optional[int] = None,
    center_cols_for_effrank: bool = True,
    norm_d: bool = False,
    eps: float = 1e-12,
) -> Tuple[float, float, float, float]:
    """
    SI = min(k_a, d) / EffRank, where EffRank can be computed with or without COM-centering.

    Args:
      C: (k_a × d) feature matrix (e.g., class centroids or probe normals).
      d_override: optional ambient dimension to use instead of C.shape[1].
      center_cols_for_effrank: True for centroids, False for probe normals.
      eps: stability.

    Returns:
      (SI, SI_raw, EffRank_centered, EffRank_raw)

      - EffRank_centered: EffRank of (C - mean_col), if center_cols_for_effrank=True; else EffRank(C).
      - SI uses the 'centered' EffRank if flag True, else uses raw EffRank.
      - SI_raw always uses raw EffRank(C) for debugging.
    """
    C = _check_matrix(C)
    k_a, d_C = C.shape
    d_eff = int(d_override if d_override is not None else d_C)

    # Centered EffRank (for centroids) vs raw EffRank (for probe normals)
    if center_cols_for_effrank:
        C_com = C - C.mean(0, keepdim=True) if k_a > 0 else C
        eff_c = _entropy_effrank(C_com, eps)
    else:
        eff_c = _entropy_effrank(C, eps)

    eff_raw = _entropy_effrank(C, eps)

    if norm_d:
        SI = (min(k_a, d_eff) / eff_c) if eff_c > eps else float("inf")
        SI_raw = (min(k_a, d_eff) / eff_raw) if eff_raw > eps else float("inf")
    else:
        SI = k_a / eff_c if eff_c > eps else float("inf")
        SI_raw = k_a / eff_raw if eff_raw > eps else float("inf")

    return SI, SI_raw, eff_c, eff_raw

# ---------------------------
# WNO (ambient & intrinsic) with configurable normalisation & PC1 removal
# ---------------------------
def wno_ambient(
    C: torch.Tensor,
    *,
    remove_pc1: bool = True,
    skip_pc1_if_d_leq: int = 2,
    eps: float = 1e-12,
) -> Tuple[int, float, float, float, float, float]:
    """
    Ambient WNO with option to remove PC1 (typical for centroids).
    Uses cosine geometry (i.e., effectively row-normalised cosines).

    Returns:
      (d_eff_used, mean_cos2, mu2_best, WNO, WNO_raw, PC1_energy_raw)
        - WNO is your redefined version: 0=Welch-optimal, 1=random, >1 worse than random.
        - WNO_raw is computed with NO PC1 removal.
    """
    C = _check_matrix(C)
    k, d = C.shape
    pc1_e = _pc1_energy(C, eps=eps)

    if k <= 1 or d == 0:
        return d, 0.0, 0.0, float("nan"), float("nan"), pc1_e

    # Raw (no PC removal)
    mean_raw = float(_pairwise_mean_cos2(C, eps))
    mu2_raw = 0.0 if k <= d else (k - d) / (d * (k - 1))
    rand_raw = 1.0 / d
    denom_raw = (rand_raw - mu2_raw)
    WNO_raw = 1.0 - ((rand_raw - mean_raw) / denom_raw) if denom_raw > eps else float("nan")

    # Decide whether to remove PC1
    if (not remove_pc1) or (d <= skip_pc1_if_d_leq):
        return d, mean_raw, mu2_raw, WNO_raw, WNO_raw, pc1_e

    # Remove PC1
    C_perp, _ = _project_out_pc1(C, eps)
    d_eff = d - 1
    mean = float(_pairwise_mean_cos2(C_perp, eps))
    mu2 = 0.0 if k <= d_eff else (k - d_eff) / (d_eff * (k - 1))
    rand = 1.0 / d_eff
    denom = (rand - mu2)
    WNO = 1.0 - ((rand - mean) / denom) if denom > eps else float("nan")
    return d_eff, mean, mu2, WNO, WNO_raw, pc1_e

def wno_intrinsic(
    C: torch.Tensor,
    *,
    r: Optional[int] = None,
    remove_pc1: bool = True,
    effrank_for_r_center_cols: bool = True,   # use centered EffRank to pick r (good for centroids)
    skip_pc1_if_d_leq: int = 2,
    eps: float = 1e-12,
) -> Tuple[int, float, float, float, float, float]:
    """
    Intrinsic WNO computed in the top-r subspace actually used by the features.
    Optionally remove PC1 first (for centroids). For probe normals, set remove_pc1=False.

    Steps:
      - If r is None, pick r = round(EffRank(C_centered)) if effrank_for_r_center_cols=True,
        else r = round(EffRank(C)).
      - RAW intrinsic (no PC1 removal): project C to top-r right singular vectors of C, compute WNO.
      - If remove_pc1 and d > skip_pc1_if_d_leq: project out PC1 first, recompute SVD on C_perp, then intrinsic WNO in top-r.

    Returns:
      (r_used, mean_cos2_r, mu2_best_r, WNO_r, WNO_r_raw, PC1_energy_raw)
    """
    C = _check_matrix(C)
    k, d = C.shape
    pc1_e = _pc1_energy(C, eps=eps)

    if k <= 1 or d == 0:
        return 0, 0.0, 0.0, float("nan"), float("nan"), pc1_e

    # --- Raw intrinsic (no PC1 removal) ---
    U_raw, S_raw, Vh_raw = torch.linalg.svd(C, full_matrices=False)
    if r is None:
        if effrank_for_r_center_cols and k > 0:
            r_pick = int(max(1, round(_entropy_effrank(C - C.mean(0, keepdim=True), eps))))
        else:
            r_pick = int(max(1, round(_entropy_effrank(C, eps))))
    else:
        r_pick = int(max(1, min(r, d)))
    Vr_raw = Vh_raw[:r_pick].t()           # (d, r_pick)
    Cr_raw = C @ Vr_raw                    # (k, r_pick)
    mean_raw = float(_pairwise_mean_cos2(Cr_raw, eps))
    mu2_raw = 0.0 if k <= r_pick else (k - r_pick) / (r_pick * (k - 1))
    rand_raw = 1.0 / r_pick
    denom_raw = (rand_raw - mu2_raw)
    WNO_r_raw = 1.0 - ((rand_raw - mean_raw) / denom_raw) if denom_raw > eps else float("nan")

    # If not removing PC1 (or low dim), return RAW intrinsic
    if (not remove_pc1) or (d <= skip_pc1_if_d_leq):
        return r_pick, mean_raw, mu2_raw, WNO_r_raw, WNO_r_raw, pc1_e

    # --- PC1-removed intrinsic ---
    C_perp, _ = _project_out_pc1(C, eps)
    U, S, Vh = torch.linalg.svd(C_perp, full_matrices=False)
    d_eff = d - 1
    if r is None:
        if effrank_for_r_center_cols and k > 0:
            r_used = int(max(1, min(round(_entropy_effrank(C - C.mean(0, keepdim=True), eps)), d_eff)))
        else:
            r_used = int(max(1, min(round(_entropy_effrank(C, eps)), d_eff)))
    else:
        r_used = int(max(1, min(r, d_eff)))

    Vr = Vh[:r_used].t()                   # (d, r_used) in the PC1-removed space
    Cr = C_perp @ Vr                       # (k, r_used)
    mean = float(_pairwise_mean_cos2(Cr, eps))
    mu2 = 0.0 if k <= r_used else (k - r_used) / (r_used * (k - 1))
    rand = 1.0 / r_used
    denom = (rand - mu2)
    WNO_r = 1.0 - ((rand - mean) / denom) if denom > eps else float("nan")
    return r_used, mean, mu2, WNO_r, WNO_r_raw, pc1_e

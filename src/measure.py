from typing import Tuple, Optional

import torch


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _check_matrix(C: torch.Tensor) -> torch.Tensor:
    """
    Ensure C is a 2-D tensor (k × d).  If empty, returns
    shape (0,0) float tensor on CPU for safe downstream ops.
    """
    if isinstance(C, list):  # fallback guard
        C = torch.stack(C, 0)
    if C.ndim != 2:
        raise ValueError(f"Expected 2-D tensor, got {C.shape}")
    if C.numel() == 0:
        return torch.zeros(0, 0)
    return C


def _unit_rows(C: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return C / C.norm(dim=1, keepdim=True).clamp_min(eps)


def _pairwise_mean_cos2(Cu: torch.Tensor) -> torch.Tensor:
    k = Cu.shape[0]
    if k <= 1:
        return Cu.new_tensor(0.0)
    G = Cu @ Cu.t()
    iu = torch.triu_indices(k, k, offset=1)
    return (G[iu[0], iu[1]].pow(2)).mean()


def _entropy_effrank(C: torch.Tensor, eps: float = 1e-12) -> float:
    if C.numel() == 0:
        return 0.0
    s = torch.linalg.svdvals(C)
    if s.sum() <= eps:
        return 0.0
    p = s / (s.sum() + eps)
    H = -(p * (p + eps).log()).sum()
    return float(torch.exp(H))


def _project_out_pc1(C: torch.Tensor, eps: float = 1e-12):
    """Project out top right-singular vector (PC1)."""
    if C.numel() == 0:
        return C, None
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    if S[0] <= eps:
        return C, None
    v1 = Vh[0]  # (d,)
    P = torch.eye(C.shape[1], dtype=C.dtype, device=C.device) - torch.outer(v1, v1)
    return C @ P, v1


# ------------------------------------------------------------
# 1. Superposition Index
# ------------------------------------------------------------
def superposition_index(
        C: torch.Tensor,
        *,
        d: Optional[int] = None,
        eps: float = 1e-12,
) -> Tuple[float, float, float, float]:
    """
    C : 2-D tensor of centroids (k_a × d).
    Returns (SI, SI_raw, EffRank_COM, EffRank_raw).
    """
    C = _check_matrix(C)
    k_a, d_vec = C.shape
    if d is None:
        d = d_vec

    # COM-centred EffRank
    C_com = C - C.mean(0, keepdim=True) if k_a else C
    eff_com = _entropy_effrank(C_com, eps)
    eff_raw = _entropy_effrank(C, eps)

    si = (min(k_a, d) / eff_com) if eff_com > eps else float("inf")
    si_raw = (min(k_a, d) / eff_raw) if eff_raw > eps else float("inf")
    return si, si_raw, eff_com, eff_raw


# ------------------------------------------------------------
# PC1 energy helper
# ------------------------------------------------------------
def _pc1_energy(C: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Fraction of spectral energy in the top right-singular direction of C:
      PC1Energy = s1^2 / sum_i s_i^2
    Computed on the RAW centroid matrix (no COM-centering), to match
    the PC1 used by WNO's projection step.
    """
    if C.numel() == 0:
        return 0.0
    s = torch.linalg.svdvals(C)
    denom = (s * s).sum()
    if denom <= eps:
        return 0.0
    return float((s[0] * s[0]) / denom)


# ------------------------------------------------------------
# 2. WNO (ambient & intrinsic) — now also returns PC1 energy
# ------------------------------------------------------------
def wno_ambient_pc1_removed(
        C: torch.Tensor, eps: float = 1e-12
) -> Tuple[int, float, float, float, float, float]:
    """
    Ambient WNO after PC1 removal (skipped if d<=2).

    Returns:
        (d_eff, mean_cos2, mu2_best, WNO, WNO_raw, PC1_energy_raw)
    """
    C = _check_matrix(C)
    k, d = C.shape
    pc1_e = _pc1_energy(C, eps=eps)

    if k <= 1 or d == 0:
        return d, 0.0, 0.0, float("nan"), float("nan"), pc1_e

    # --- raw WNO (no COM/PC removal) ---
    Cu_raw = _unit_rows(C)
    mean_raw = float(_pairwise_mean_cos2(Cu_raw))
    mu2_best_raw = 0.0 if k <= d else (k - d) / (d * (k - 1))
    rand_raw = 1.0 / d
    WNO_raw = 1 - (rand_raw - mean_raw) / (rand_raw - mu2_best_raw) if k > 1 else float("nan")

    if d <= 2:  # nothing to project out
        return d, mean_raw, mu2_best_raw, WNO_raw, WNO_raw, pc1_e

    # --- PC1 removal ---
    C_perp, _ = _project_out_pc1(C, eps)
    d_eff = d - 1
    Cu = _unit_rows(C_perp, eps)
    mean = float(_pairwise_mean_cos2(Cu))
    mu2_best = 0.0 if k <= d_eff else (k - d_eff) / (d_eff * (k - 1))
    rand = 1.0 / d_eff
    WNO = 1 - (rand - mean) / (rand - mu2_best) if k > 1 else float("nan")
    return d_eff, mean, mu2_best, WNO, WNO_raw, pc1_e


def wno_intrinsic_pc1_removed(
        C: torch.Tensor,
        *,
        r: Optional[int] = None,
        eps: float = 1e-12,
) -> Tuple[int, float, float, float, float, float]:
    """
    Intrinsic WNO after PC1 removal (skipped if d<=2).

    Returns:
        (r_used, mean_cos2_r, mu2_best_r, WNO_r, WNO_r_raw, PC1_energy_raw)
    """
    C = _check_matrix(C)
    k, d = C.shape
    pc1_e = _pc1_energy(C, eps=eps)

    if k <= 1 or d == 0:
        return 0, 0.0, 0.0, float("nan"), float("nan"), pc1_e

    # ---------- RAW intrinsic ----------
    U_raw, S_raw, Vh_raw = torch.linalg.svd(C, full_matrices=False)
    if r is None:
        r_raw = int(max(1, round(_entropy_effrank(C, eps))))
    else:
        r_raw = int(max(1, min(r, d)))
    Cr_raw = C @ Vh_raw[:r_raw].t()
    Cru_raw = _unit_rows(Cr_raw, eps)
    mean_raw = float(_pairwise_mean_cos2(Cru_raw))
    mu2_best_r_raw = 0.0 if k <= r_raw else (k - r_raw) / (r_raw * (k - 1))
    rand_r_raw = 1.0 / r_raw
    WNO_r_raw = 1 - (rand_r_raw - mean_raw) / (rand_r_raw - mu2_best_r_raw) if k > 1 else float("nan")

    if d <= 2:
        return r_raw, mean_raw, mu2_best_r_raw, WNO_r_raw, WNO_r_raw, pc1_e

    # ---------- PC1-removed intrinsic ----------
    C_perp, _ = _project_out_pc1(C, eps)
    eff_com = _entropy_effrank(C - C.mean(0, keepdim=True), eps)
    r = int(max(1, min(r or round(eff_com), d - 1)))
    Cr = C_perp @ torch.linalg.svd(C_perp, full_matrices=False).Vh[:r].t()
    Cru = _unit_rows(Cr, eps)
    mean = float(_pairwise_mean_cos2(Cru))
    mu2_best_r = 0.0 if k <= r else (k - r) / (r * (k - 1))
    rand_r = 1.0 / r
    WNO_r = 1 - (rand_r - mean) / (rand_r - mu2_best_r) if k > 1 else float("nan")
    return r, mean, mu2_best_r, WNO_r, WNO_r_raw, pc1_e

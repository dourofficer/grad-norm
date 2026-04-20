import torch
from typing import Callable, Literal

DistMetric = Literal["l1", "l2", "cosine"]


# ═══════════════════════════════════════════════════════════════════
# Shared distance helper
# ═══════════════════════════════════════════════════════════════════

def _row_distance(
    G: torch.Tensor,
    ref: torch.Tensor,
    metric: DistMetric,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute distance between each row of G and a reference vector.

    Parameters
    ----------
    G      : (N, d) matrix of row vectors.
    ref    : (d,) reference vector.
    metric : "l1"     → Σ|gᵢ - ref|
             "l2"     → ‖gᵢ - ref‖₂
             "cosine" → 1 - cos(gᵢ, ref)  (in [0, 2]; 0 = identical direction)
    """
    diff = G - ref.unsqueeze(0)

    if metric == "l1":
        return diff.abs().sum(dim=1)
    elif metric == "l2":
        return torch.norm(diff, dim=1)
    elif metric == "cosine":
        G_norms   = torch.norm(G,   dim=1, keepdim=True) + eps
        ref_norm  = torch.norm(ref) + eps
        cos_sim   = (G / G_norms) @ (ref / ref_norm)   # (N,)
        return 1.0 - cos_sim
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Choose 'l1', 'l2', or 'cosine'.")


# ═══════════════════════════════════════════════════════════════════
# Family 1: Central-tendency references
#   Score = distance to a single summary vector.
#   Higher score = more anomalous for all metrics.
# ═══════════════════════════════════════════════════════════════════

def mean_distance(
    G: torch.Tensor,
    metric: DistMetric = "l2",
) -> torch.Tensor:
    """Distance of each row to the arithmetic mean of G.

    Reference : arithmetic mean μ = (1/N) Σ gᵢ.
    Breakdown : 1/N (non-robust; a single large outlier shifts μ).
    Metric    : l1 | l2 (default) | cosine.
    """
    G_f = G.float()
    ref = G_f.mean(dim=0)
    return _row_distance(G_f, ref, metric).to(G.dtype)


def coordinate_median(
    G: torch.Tensor,
    metric: DistMetric = "l2",
) -> torch.Tensor:
    """Distance of each row to the coordinate-wise median of G.

    Reference : coordinate-wise median med_j = median({gᵢⱼ}ᵢ) per dimension j.
    Breakdown : ~50% per coordinate, but not rotation-invariant.
    Metric    : l1 | l2 (default) | cosine.
    """
    G_f = G.float()
    ref = G_f.median(dim=0).values
    return _row_distance(G_f, ref, metric).to(G.dtype)


def geometric_median(
    G: torch.Tensor,
    metric: DistMetric = "l2",
    eps: float = 1e-8,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Distance of each row to the geometric median of G (Weiszfeld iteration).

    Reference : geometric median μ* = argmin_μ Σ ‖gᵢ − μ‖₂,
                computed via Weiszfeld iteration warm-started at the mean.
    Breakdown : ~50%, rotation-invariant. Robust drop-in for the mean.
    Metric    : l1 | l2 (default) | cosine.
    """
    G_f = G.float()
    mu  = G_f.mean(dim=0)

    for _ in range(max_iter):
        d      = torch.norm(G_f - mu, dim=1) + eps
        w      = 1.0 / d
        mu_new = (w.unsqueeze(1) * G_f).sum(dim=0) / w.sum()
        if torch.norm(mu_new - mu) < tol:
            mu = mu_new
            break
        mu = mu_new

    return _row_distance(G_f, mu, metric).to(G.dtype)


# ═══════════════════════════════════════════════════════════════════
# Family 2: Spectral / subspace-based
#   Score = projection onto (or residual from) an SVD subspace.
# ═══════════════════════════════════════════════════════════════════

def _run_svd(G: torch.Tensor, c: int) -> torch.Tensor:
    """Top-c right singular vectors of G, shape (d, c)."""
    _, _, V = torch.svd_lowrank(G, q=c, niter=10)
    return V


def projection_svd(
    G: torch.Tensor,
    c: int = 1,
    centered: bool = True,
) -> torch.Tensor:
    """Mean squared projection of each row onto the top-c right singular vectors.

        τᵢ = (1/c) Σⱼ ⟨g̃ᵢ, vⱼ⟩²

    Reference : top-c singular subspace of G (optionally mean-centered).
    SAL interpretation (Du et al. 2024): the leading singular direction
    aligns with the outlier direction, so a large projection flags OOD rows.
    Higher score = more anomalous.
    """
    G_f = G.float()
    if centered:
        G_f = G_f - G_f.mean(dim=0)
    V = _run_svd(G_f, c)
    return (G_f @ V).square().mean(dim=1).to(G.dtype)


def reconstruction_svd(
    G: torch.Tensor,
    c: int = 5,
    centered: bool = True,
) -> torch.Tensor:
    """Residual L2 norm after projecting each row onto the top-c SVD subspace.

    Reference : top-c singular subspace of G (optionally mean-centered).
    Rows well-explained by the dominant subspace have small residuals;
    rows off this subspace (structural outliers) have large residuals.
    Higher score = more anomalous.
    """
    G_f = G.float()
    G_c = G_f - G_f.mean(dim=0) if centered else G_f
    V     = _run_svd(G_c, c)
    G_rec = (G_c @ V) @ V.T
    return torch.norm(G_c - G_rec, dim=1).to(G.dtype)


# ═══════════════════════════════════════════════════════════════════
# Family 3: Neighborhood-based
#   Score = isolation from nearby rows (no global reference).
# ═══════════════════════════════════════════════════════════════════

def knn_distance(
    G: torch.Tensor,
    k: int = 5,
    normalize: bool = True,
) -> torch.Tensor:
    """Distance to the k-th nearest neighbor within G (excluding self).

    Reference : k-th nearest neighbor among all other rows in G.
    Follows Sun et al. 2022. With normalize=True, rows are L2-normalised
    before computing distances, making L2 monotone in cosine distance.
    Isolated rows (far from their neighbours) receive large scores.
    Higher score = more anomalous.
    """
    G_f   = G.float()
    N     = G_f.shape[0]
    k_eff = max(1, min(k, N - 1))

    if normalize:
        G_f = G_f / (torch.norm(G_f, dim=1, keepdim=True) + 1e-8)

    D = torch.cdist(G_f, G_f, p=2)
    D.fill_diagonal_(float("inf"))
    kth, _ = D.kthvalue(k_eff, dim=1)
    return kth.to(G.dtype)


# ═══════════════════════════════════════════════════════════════════
# Role-based orchestration
# ═══════════════════════════════════════════════════════════════════

def group_role(role: str) -> str:
    if role.startswith("Orchestrator (->"):
        return "Orchestrator (-> Agent)"
    return role


def split_by_role(G: torch.Tensor, index: list) -> dict:
    ROLES = set(group_role(idx.role) for idx in index)
    role_Gs = {}
    for role in ROLES:
        role_idxs = [idx.row for idx in index if group_role(idx.role) == role]
        role_mask = torch.tensor(
            [group_role(idx.role) == role for idx in index],
            device=G.device,
        )
        role_Gs[role] = {"idx": role_idxs, "G": G[role_mask]}
    return role_Gs


def compute_split_scores(G: torch.Tensor, index: list, scoring: Callable) -> torch.Tensor:
    role_Gs = split_by_role(G, index)
    scores  = torch.zeros(G.shape[0], device=G.device, dtype=G.dtype)
    for role_data in role_Gs.values():
        scores[role_data["idx"]] = scoring(role_data["G"])
    return scores


def compute_scores(G: torch.Tensor, index: list, scoring: Callable) -> torch.Tensor:
    return scoring(G)


def make_scoring_fn(index, scoring, name: str = "scoring_by_role"):
    def scoring_fn(G: torch.Tensor) -> torch.Tensor:
        return compute_scores(G, index, scoring)
    scoring_fn.__name__ = name
    return scoring_fn


# ═══════════════════════════════════════════════════════════════════
# Factories
# ═══════════════════════════════════════════════════════════════════

# Family 1: central-tendency
def make_mean_distance_scoring(metric: DistMetric = "l2"):
    def scoring(G): return mean_distance(G, metric=metric)
    return scoring

def make_coordinate_median_scoring(metric: DistMetric = "l2"):
    def scoring(G): return coordinate_median(G, metric=metric)
    return scoring

def make_geometric_median_scoring(metric: DistMetric = "l2", max_iter: int = 100, tol: float = 1e-6):
    def scoring(G): return geometric_median(G, metric=metric, max_iter=max_iter, tol=tol)
    return scoring

# Family 2: spectral
def make_projection_scoring(c: int = 1, centered: bool = True):
    def scoring(G): return projection_svd(G, c=c, centered=centered)
    return scoring

def make_reconstruction_scoring(c: int = 5, centered: bool = True):
    def scoring(G): return reconstruction_svd(G, c=c, centered=centered)
    return scoring

# Family 3: neighborhood
def make_knn_scoring(k: int = 5, normalize: bool = True):
    def scoring(G): return knn_distance(G, k=k, normalize=normalize)
    return scoring
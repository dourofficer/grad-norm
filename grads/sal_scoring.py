"""sal_scoring.py — SAL scoring functions compatible with evaluate_weights."""

import torch


def _run_svd(G: torch.Tensor, c: int) -> torch.Tensor:
    """Return top-c right singular vectors, shape (d, c)."""
    _, _, V = torch.svd_lowrank(G, q=c, niter=10)
    return V


def _sal(G: torch.Tensor, c: int, centered: bool) -> torch.Tensor:
    """Score each row of G by mean squared projection onto top-c SVD vectors.

    τ_i = (1/c) * Σ_{j=1}^{c} ⟨g̃_i, v_j⟩²
    """
    G_f = G.float()
    if centered:
        G_f = G_f - G_f.mean(dim=0)
    V = _run_svd(G_f, c)          # (d, c)
    return (G_f @ V).square().mean(dim=1)   # (T,)


def make_sal_scoring_fn(c: int = 1, centered: bool = True):
    """Factory: returns a scoring_fn(G: Tensor) -> (T,) for evaluate_weights.

    Args:
        c:        number of singular vectors (1–5).
        centered: if True, subtract the mean gradient (SAL w/ ref);
                  otherwise use raw gradients (SAL w/o ref).
    """
    def scoring_fn(G: torch.Tensor) -> torch.Tensor:
        return _sal(G, c=c, centered=centered)
    scoring_fn.__name__ = f"sal_{'wref' if centered else 'noref'}_c{c}"
    return scoring_fn


# All 10 pre-built variants (sal_wref_c1..5, sal_noref_c1..5)
SAL_SCORING_FNS: dict[str, callable] = {
    f"sal_{'wref' if centered else 'noref'}_c{c}": make_sal_scoring_fn(c, centered)
    for centered in (True, False)
    for c in range(1, 6)
}
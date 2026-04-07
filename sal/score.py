"""
sal.score — Compute all SAL and GradNorm scoring variants for failure attribution.

Loads the .pt gradient files produced by the extraction phase, computes 14
scoring variants, and saves per-trajectory JSON files containing all scores.

Scoring variants (14 total):
  SAL (10):    sal_wref_c{1..5}, sal_noref_c{1..5}
  GradNorm (4): gradnorm_l1_centered, gradnorm_l1_uncentered,
                gradnorm_l2_centered, gradnorm_l2_uncentered

Usage:
python -m sal.score \
    --subset hand-crafted \
    --config /home/hoangpham/exchange/v/35 \
    --output outputs/sal/scores/qwen3-8b/hand-crafted/v/35
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepIndex:
    """Row-level metadata for one entry in the stacked gradient matrix."""
    row:        int     # row index in G
    traj_idx:   int     # 1-based index into the loaded data list
    step_idx:   int     # step index within the trajectory
    role:       str     # e.g. "WebSurfer", "Orchestrator (thought)"
    is_mistake: bool    # whether this is the gold mistake step


@dataclass
class GradientStore:
    """All gradient data stacked into a single matrix with index mappings.

    Attributes
    ----------
    G           : (T, d) float16 tensor — all raw gradient vectors.
    index       : list[StepIndex] of length T — per-row metadata.
    lookup      : dict[(traj_idx, step_idx) → row] — fast reverse lookup.
    traj_meta   : list[dict] — per-trajectory metadata dicts.
    traj_ranges : list[tuple[int, int]] — (start_row, end_row) for each traj.
    device      : torch device where G lives.
    """
    G:           torch.Tensor
    index:       list[StepIndex]
    lookup:      dict[tuple[int, int], int]
    traj_meta:   list[dict]
    traj_ranges: list[tuple[int, int]]
    device:      torch.device


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_stack(
    input_dir: Path,
    metadata_dir: Path,
    device: torch.device,
) -> GradientStore:
    """Load .pt gradient files and matching JSONs, stack into (T, d) matrix.

    Parameters
    ----------
    input_dir    : directory containing per-trajectory .pt gradient files.
    metadata_dir : directory containing the matching raw .json files
                   (e.g. ww/hand-crafted/) whose ``history`` entries
                   give us the role for each step index.
    device       : target torch device.

    Returns
    -------
    GradientStore with 1-based traj_idx.
    """
    files = sorted(input_dir.glob("*.pt"))
    files = [f for f in files if f.name != "config.pt"]
    if not files:
        raise FileNotFoundError(f"No .pt gradient files in {input_dir}")

    # Peek at first file for gradient dimension
    first = torch.load(files[0], map_location="cpu", weights_only=False)
    d_final = next(iter(first["gradients"].values())).shape[0]
    del first

    # Pre-allocate oversized buffer
    max_rows = len(files) * 150
    G = torch.empty(max_rows, d_final, dtype=torch.float16)

    index: list[StepIndex] = []
    lookup: dict[tuple[int, int], int] = {}
    traj_meta: list[dict] = []
    traj_ranges: list[tuple[int, int]] = []

    row = 0
    for file_idx, fp in enumerate(tqdm(files, desc="Loading .pt files")):
        traj_idx = file_idx + 1  # 1-based

        payload = torch.load(fp, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        gradients = payload["gradients"]
        mistake_step = int(metadata["mistake_step"])

        # Load matching JSON for step roles
        json_path = metadata_dir / fp.with_suffix(".json").name
        with open(json_path) as f:
            raw = json.load(f)
        history = raw["history"]

        traj_meta.append(metadata)
        start_row = row

        for step_idx in sorted(int(k) for k in gradients.keys()):
            G[row] = gradients[step_idx]
            role = history[step_idx]["role"]
            index.append(StepIndex(
                row=row,
                traj_idx=traj_idx,
                step_idx=step_idx,
                role=role,
                is_mistake=(step_idx == mistake_step),
            ))
            lookup[(traj_idx, step_idx)] = row
            row += 1

        traj_ranges.append((start_row, row))
        del payload, gradients

    G = G[:row].to(device)
    mem_gb = G.element_size() * G.numel() / 1e9
    print(f"G on {device}: {mem_gb:.2f} GB  ({row} × {d_final:,})")

    return GradientStore(
        G=G, index=index, lookup=lookup,
        traj_meta=traj_meta, traj_ranges=traj_ranges,
        device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reference gradient
# ─────────────────────────────────────────────────────────────────────────────

def compute_reference_gradient(G: torch.Tensor) -> torch.Tensor:
    """Compute ∇̄ = mean of all gradient rows.

    Returns (d,) float32 tensor.
    """
    return G.float().mean(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# SVD
# ─────────────────────────────────────────────────────────────────────────────

def run_svd(
    G: torch.Tensor,
    n_components: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-c singular vectors via randomised SVD.

    Parameters
    ----------
    G            : (T, d) matrix (centered or raw).
    n_components : number of singular vectors to compute.

    Returns
    -------
    V : (d, n_components) — top singular vectors.
    S : (n_components,)   — corresponding singular values.
    """
    U, S, V = torch.svd_lowrank(G.float(), q=n_components, niter=5)
    return V, S


# ─────────────────────────────────────────────────────────────────────────────
# SAL scores
# ─────────────────────────────────────────────────────────────────────────────

def compute_sal_scores(
    G: torch.Tensor,
    V: torch.Tensor,
    n_components: int,
) -> torch.Tensor:
    """Compute SAL filtering scores for a given number of singular vectors.

    For c singular vectors, the score for each step is the average of
    squared projections (Appendix K of SAL paper):

        τ_i = (1/c) * Σ_{j=1}^{c} ⟨g̃_i, v_j⟩²

    Parameters
    ----------
    G            : (T, d) — gradient matrix (centered or raw).
    V            : (d, c) — singular vectors.
    n_components : number of singular vectors used.

    Returns
    -------
    scores : (T,) tensor.
    """
    G_f = G.float()
    if V.dim() == 1:
        V = V.unsqueeze(1)

    # (T, d) @ (d, c) → (T, c)
    projections = G_f @ V.float()
    # average of squared projections
    scores = projections.square().mean(dim=1)  # (T,)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# GradNorm scores
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradnorm_scores(
    G: torch.Tensor,
    ref_grad: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute all 4 GradNorm variants in one pass.

    Returns dict with keys:
        gradnorm_l1_uncentered, gradnorm_l2_uncentered,
        gradnorm_l1_centered,   gradnorm_l2_centered
    Each value is a (T,) tensor.
    """
    G_f = G.float()
    G_centered = G_f - ref_grad.unsqueeze(0)

    return {
        "gradnorm_l1_uncentered": G_f.abs().sum(dim=1),
        "gradnorm_l2_uncentered": G_f.norm(p=2, dim=1),
        "gradnorm_l1_centered":   G_centered.abs().sum(dim=1),
        "gradnorm_l2_centered":   G_centered.norm(p=2, dim=1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: compute all 14 scoring variants
# ─────────────────────────────────────────────────────────────────────────────

MAX_COMPONENTS = 5

def compute_all_scores(store: GradientStore) -> dict[str, torch.Tensor]:
    """Compute all 14 scoring variants.

    Returns dict mapping score_name → (T,) tensor:
        sal_wref_c1..c5, sal_noref_c1..c5,
        gradnorm_l1_centered, gradnorm_l1_uncentered,
        gradnorm_l2_centered, gradnorm_l2_uncentered
    """
    all_scores: dict[str, torch.Tensor] = {}

    # ── Reference gradient (computed on raw G) ───────────────────────
    print("\n── Computing reference gradient ──")
    ref_grad = compute_reference_gradient(store.G)
    print(f"  ‖∇̄‖₁ = {ref_grad.abs().sum().item():.4f}")

    # ── GradNorm (4 variants) ────────────────────────────────────────
    print("\n── Computing GradNorm scores (4 variants) ──")
    gn_scores = compute_gradnorm_scores(store.G, ref_grad)
    all_scores.update(gn_scores)
    for name, s in gn_scores.items():
        print(f"  {name}: mean={s.mean().item():.6f}, "
              f"std={s.std().item():.6f}")

    # ── SAL w/ ref (5 variants) ──────────────────────────────────────
    print("\n── Computing SAL w/ ref (5 variants) ──")
    G_centered = store.G.float() - ref_grad.unsqueeze(0)

    V_ref, S_ref = run_svd(G_centered, n_components=MAX_COMPONENTS)
    print(f"  Singular values: {[f'{s:.4f}' for s in S_ref.tolist()]}")

    for c in range(1, MAX_COMPONENTS + 1):
        V_c = V_ref[:, :c]
        scores = compute_sal_scores(G_centered, V_c, n_components=c)
        key = f"sal_wref_c{c}"
        all_scores[key] = scores
        print(f"  {key}: mean={scores.mean().item():.6f}, "
              f"std={scores.std().item():.6f}")

    del G_centered

    # ── SAL wo/ ref (5 variants) ─────────────────────────────────────
    print("\n── Computing SAL wo/ ref (5 variants) ──")
    G_raw = store.G.float()

    V_noref, S_noref = run_svd(G_raw, n_components=MAX_COMPONENTS)
    print(f"  Singular values: {[f'{s:.4f}' for s in S_noref.tolist()]}")

    for c in range(1, MAX_COMPONENTS + 1):
        V_c = V_noref[:, :c]
        scores = compute_sal_scores(G_raw, V_c, n_components=c)
        key = f"sal_noref_c{c}"
        all_scores[key] = scores
        print(f"  {key}: mean={scores.mean().item():.6f}, "
              f"std={scores.std().item():.6f}")

    del G_raw

    print(f"\n── Done: {len(all_scores)} scoring variants computed ──")
    return all_scores


# ─────────────────────────────────────────────────────────────────────────────
# Save: per-trajectory JSON files
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    store: GradientStore,
    all_scores: dict[str, torch.Tensor],
    out_dir: Path,
    subset: str,
    config: str,
):
    """Write one JSON file per trajectory with all 14 score variants.

    Output: {out_dir}/{traj_idx}.json   (1-based)

    Schema:
    {
      "config": { "subset": ..., "weight_config": ... },
      "metadata": { ... },
      "steps": [
        {
          "step_idx": 0,
          "role": "...",
          "is_mistake": false,
          "scores": {
            "sal_wref_c1": ...,
            ...
            "gradnorm_l2_uncentered": ...
          }
        },
        ...
      ]
    }
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-convert all scores to CPU lists
    score_names = sorted(all_scores.keys())
    scores_cpu: dict[str, list[float]] = {
        name: all_scores[name].cpu().tolist() for name in score_names
    }

    n_trajs = len(store.traj_meta)
    for traj_local_idx in range(n_trajs):
        start, end = store.traj_ranges[traj_local_idx]
        traj_idx = store.index[start].traj_idx  # 1-based

        metadata = store.traj_meta[traj_local_idx]

        steps = []
        for row in range(start, end):
            entry = store.index[row]
            step_scores = {
                name: scores_cpu[name][row] for name in score_names
            }
            steps.append({
                "step_idx":   entry.step_idx,
                "role":       entry.role,
                "is_mistake": entry.is_mistake,
                "scores":     step_scores,
            })

        doc = {
            "config": {
                "subset":        subset,
                "weight_config": config,
            },
            "metadata": metadata,
            "steps":    steps,
        }

        filepath = out_dir / f"{traj_idx}.json"
        with open(filepath, "w") as f:
            json.dump(doc, f, indent=2)

    print(f"Saved {n_trajs} trajectory files to {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute SAL and GradNorm scores for failure attribution.",
    )
    p.add_argument(
        "--subset", required=True,
        choices=["algorithm-generated", "hand-crafted"],
        help="Which Who&When subset to use.",
    )
    p.add_argument(
        "--config", required=True,
        help="Weight config / gradient directory (e.g. v/35, q/34).",
    )
    p.add_argument(
        "--output", required=True,
        help="Output directory for per-trajectory JSON score files.",
    )
    p.add_argument(
        "--device", default=None,
        help="Torch device (default: cpu).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if args.device else "cpu")
    input_dir = Path(args.config)
    metadata_dir = Path("ww") / args.subset
    out_dir = Path(args.output)

    print(f"Subset:       {args.subset}")
    print(f"Config:       {args.config}")
    print(f"Input dir:    {input_dir}")
    print(f"Metadata dir: {metadata_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Device:       {device}")

    # ── Load ────────────────────────────────────────────────────────
    store = load_and_stack(input_dir, metadata_dir, device)
    print(f"Loaded {len(store.traj_meta)} trajectories, "
          f"{store.G.shape[0]} total steps.")

    # ── Compute all 14 variants ─────────────────────────────────────
    all_scores = compute_all_scores(store)

    # ── Save ────────────────────────────────────────────────────────
    save_results(store, all_scores, out_dir, args.subset, args.config)

    print("\nDone.")


if __name__ == "__main__":
    main()
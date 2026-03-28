"""
sal_score.py — Phase 2 of SAL-Step: compute reference gradient, SVD, and scores.

Loads the .pt files produced by sal_extract.py, then:
  1. Computes the reference gradient (mean over all steps or pre-mistake steps).
  2. Centers all gradient vectors.
  3. Computes the top singular vector via randomised SVD.
  4. Computes per-step SAL scores: τ_t = ⟨g_t, v⟩².
  5. Saves results as JSON files compatible with evaluate_gradnorm.py.

Usage:
python -m cli.sal_score \
    --input sal_grads/qwen3-8b/hand-crafted \
    --output sal_grads/qwen3-8b/ \
    --reference all
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gradient_files(input_dir: Path) -> list[dict]:
    """Load all .pt gradient files from input_dir.

    Returns a list of dicts, each containing:
        "metadata":  dict with filename, mistake_step, etc.
        "gradients": dict[int, Tensor]  (step_idx -> gradient vector)
    """
    files = sorted(input_dir.glob("*.pt"))
    # Exclude config.pt
    files = [f for f in files if f.name != "config.pt"]
    if not files:
        raise FileNotFoundError(f"No .pt gradient files found in {input_dir}")

    data = []
    for fp in tqdm(files, desc="Loading gradient files"):
        payload = torch.load(fp, map_location="cpu", weights_only=False)
        data.append(payload)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Reference gradient computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_reference_gradient(
    data: list[dict],
    strategy: str = "all",
) -> torch.Tensor:
    """Compute the reference gradient ∇̄ using the specified strategy.

    Parameters
    ----------
    data      : list of loaded gradient file dicts.
    strategy  : "all"        — mean over all steps (default).
                "pre_mistake" — mean over steps before the gold mistake step.

    Returns
    -------
    Tensor of shape (d,) in float32.
    """
    accum = None
    count = 0

    for item in data:
        mistake_step = int(item["metadata"]["mistake_step"])
        gradients = item["gradients"]

        for step_idx, grad_vec in gradients.items():
            step_idx = int(step_idx)

            if strategy == "pre_mistake" and step_idx >= mistake_step:
                continue

            g = grad_vec.float()
            if accum is None:
                accum = torch.zeros_like(g)
            accum += g
            count += 1

    if count == 0:
        raise RuntimeError(
            f"No steps found for reference gradient (strategy='{strategy}')."
        )

    ref_grad = accum / count
    print(f"  Reference gradient: strategy={strategy}, "
          f"n_steps={count}, ‖∇̄‖₁={ref_grad.abs().sum().item():.4f}")

    return ref_grad


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 & 3: Build centered gradient matrix and compute top singular vector
# ─────────────────────────────────────────────────────────────────────────────

def compute_top_singular_vector(
    data:     list[dict],
    ref_grad: torch.Tensor,
    n_components: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center all gradient vectors, stack into G, and compute top singular vector.

    Parameters
    ----------
    data         : list of loaded gradient file dicts.
    ref_grad     : reference gradient ∇̄, shape (d,).
    n_components : number of singular vectors to compute (default 1).

    Returns
    -------
    (v, singular_values)
        v               : top singular vector(s), shape (d,) or (d, n_components).
        singular_values : corresponding singular values.
    """
    # ── Collect all centered vectors ─────────────────────────────────
    centered_list = []
    index_map = []  # (traj_idx, step_idx) for each row

    for traj_idx, item in enumerate(data):
        for step_idx, grad_vec in item["gradients"].items():
            g = grad_vec.float() - ref_grad
            centered_list.append(g)
            index_map.append((traj_idx, int(step_idx)))

    T = len(centered_list)
    d = ref_grad.shape[0]
    print(f"  Gradient matrix G: T={T} steps × d={d:,} params")

    # ── Stack into G (T × d) and compute SVD ─────────────────────────
    # Use torch.svd_lowrank for memory efficiency (randomised SVD)
    G = torch.stack(centered_list, dim=0)  # (T, d)

    print(f"  G memory: {G.element_size() * G.numel() / 1e9:.2f} GB")
    print(f"  Running randomised SVD (q={n_components}) ...")

    U, S, V = torch.svd_lowrank(G, q=n_components, niter=5)
    # V is (d, n_components), S is (n_components,)

    print(f"  Top {n_components} singular value(s): {S.tolist()}")

    if n_components == 1:
        return V.squeeze(1), S

    return V, S


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Compute SAL-Step scores
# ─────────────────────────────────────────────────────────────────────────────

def compute_sal_scores(
    data:     list[dict],
    ref_grad: torch.Tensor,
    v:        torch.Tensor,
) -> list[dict]:
    """Compute τ_t = ⟨g_t, v⟩² for each step and format as evaluation-ready dicts.

    Output format matches the JSON structure expected by evaluate_gradnorm.py:
    {
        "metadata": { ... },
        "steps": [ {"step_idx": int, "role": str, "content": str}, ... ],
        "logs": [
            {
                "step_idx": int,
                "sal_score": float,
                "l1_norm":   { "sal": float },
                "l2_norm":   { "sal": float },
            },
            ...
        ]
    }

    We store the SAL score under the "sal" key in l1_norm/l2_norm dicts
    so that evaluate_gradnorm.py can pick it up as a "layer" called "sal".
    The score convention: HIGHER = more anomalous (predicted error step).
    """
    results = []

    for item in data:
        metadata = item["metadata"]
        gradients = item["gradients"]

        logs = []
        for step_idx, grad_vec in gradients.items():
            step_idx = int(step_idx)
            g = grad_vec.float() - ref_grad

            # SAL score: squared projection onto top singular vector
            projection = torch.dot(g, v).item()
            sal_score = projection ** 2

            logs.append({
                "step_idx":  step_idx,
                "sal_score": sal_score,
                # Compatibility with evaluate_gradnorm.py:
                # store as a "layer" so the eval code can discover it
                "l1_norm":   {"sal": sal_score},
                "l2_norm":   {"sal": sal_score},
            })

        # Sort logs by step_idx
        logs.sort(key=lambda x: x["step_idx"])

        results.append({
            "metadata": metadata,
            "steps":    [],  # we don't have step content in the gradient files
            "logs":     logs,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def print_diagnostics(
    data:     list[dict],
    ref_grad: torch.Tensor,
    v:        torch.Tensor,
):
    """Print diagnostic info: correlation between SAL scores and raw GradNorm."""
    sal_scores = []
    raw_norms = []
    is_mistake = []

    for item in data:
        mistake_step = int(item["metadata"]["mistake_step"])
        for step_idx, grad_vec in item["gradients"].items():
            step_idx = int(step_idx)
            g_raw = grad_vec.float()
            g_centered = g_raw - ref_grad

            sal = torch.dot(g_centered, v).item() ** 2
            raw_l1 = g_raw.abs().sum().item()

            sal_scores.append(sal)
            raw_norms.append(raw_l1)
            is_mistake.append(1.0 if step_idx == mistake_step else 0.0)

    sal_arr = np.array(sal_scores)
    raw_arr = np.array(raw_norms)
    mistake_arr = np.array(is_mistake)

    # Correlation between SAL and raw GradNorm
    corr = np.corrcoef(sal_arr, raw_arr)[0, 1]
    print(f"\n  ── Diagnostics ──")
    print(f"  Pearson(SAL, raw_L1):     {corr:.4f}")

    # Mean SAL score for mistake vs normal steps
    mask = mistake_arr.astype(bool)
    if mask.any() and (~mask).any():
        mean_mistake = sal_arr[mask].mean()
        mean_normal = sal_arr[~mask].mean()
        print(f"  Mean SAL (mistake steps): {mean_mistake:.6f}")
        print(f"  Mean SAL (normal steps):  {mean_normal:.6f}")
        print(f"  Ratio (mistake/normal):   {mean_mistake / max(mean_normal, 1e-12):.2f}")

    # Quick Acc@1: for each trajectory, is argmax(SAL) == mistake_step?
    correct = 0
    total = 0
    for item in data:
        mistake_step = int(item["metadata"]["mistake_step"])
        gradients = item["gradients"]
        if not gradients:
            continue

        best_step = max(
            gradients.keys(),
            key=lambda s: (gradients[int(s)].float() - ref_grad).dot(v).item() ** 2,
        )
        if int(best_step) == mistake_step:
            correct += 1
        total += 1

    print(f"  Quick Acc@1 (argmax):     {correct}/{total} = "
          f"{100 * correct / max(total, 1):.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAL-Step Phase 2: compute SVD and SAL scores."
    )
    p.add_argument("--input",      required=True,
                   help="Directory containing .pt files from sal_extract.")
    p.add_argument("--output",     required=True,
                   help="Output directory for JSON result files.")
    p.add_argument("--reference",  choices=["all", "pre_mistake"],
                   default="all",
                   help="Strategy for computing reference gradient.")
    p.add_argument("--n_components", type=int, default=1,
                   help="Number of singular vectors to compute.")
    p.add_argument("--save_artifacts", action="store_true",
                   help="Save ref_grad, singular vector, and singular values.")
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ─────────────────────────────────────────────────────
    config_path = input_dir / "config.pt"
    if config_path.exists():
        config = torch.load(config_path, map_location="cpu", weights_only=False)
        print(f"Config: target={config['target_param']}, d={config['d']:,}, "
              f"proj_dim={config.get('proj_dim')}")
    else:
        print("Warning: no config.pt found, proceeding without config.")

    # ── Load gradient files ─────────────────────────────────────────────
    data = load_gradient_files(input_dir)
    print(f"Loaded {len(data)} trajectory files.")

    total_steps = sum(len(item["gradients"]) for item in data)
    print(f"Total steps: {total_steps}")

    # ── Step 1: Reference gradient ──────────────────────────────────────
    print("\n── Computing reference gradient ──")
    ref_grad = compute_reference_gradient(data, strategy=args.reference)

    # ── Step 2-3: SVD ───────────────────────────────────────────────────
    print("\n── Computing top singular vector ──")
    v, singular_values = compute_top_singular_vector(
        data, ref_grad, n_components=args.n_components,
    )

    # ── Diagnostics ─────────────────────────────────────────────────────
    print_diagnostics(data, ref_grad, v)

    # ── Step 4: SAL scores ──────────────────────────────────────────────
    print("\n── Computing SAL-Step scores ──")
    results = compute_sal_scores(data, ref_grad, v)

    # ── Save results ────────────────────────────────────────────────────
    for result in results:
        filename = result["metadata"]["filename"]
        out_path = out_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(f"\nSaved {len(results)} result files to {out_dir}")

    # ── Optionally save artifacts ───────────────────────────────────────
    if args.save_artifacts:
        artifacts = {
            "ref_grad":        ref_grad,
            "singular_vector": v,
            "singular_values": singular_values,
            "reference_strategy": args.reference,
        }
        artifact_path = out_dir / "sal_artifacts.pt"
        torch.save(artifacts, artifact_path)
        print(f"Saved artifacts to {artifact_path}")


if __name__ == "__main__":
    main()
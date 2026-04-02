"""
sal_score.py — Phase 2 of SAL-Step: compute reference gradient, SVD, and scores.

Loads the .pt files produced by sal_extract.py, then:
  1. Stacks ALL gradient vectors into a single matrix on GPU.
  2. Computes the reference gradient (mean over all steps or pre-mistake steps).
  3. Centers the matrix in-place and computes the top singular vector via SVD.
  4. Computes per-step SAL scores: τ_t = ⟨g_t, v⟩² in one batched matmul.
  5. Saves results as JSON files compatible with evaluate_gradnorm.py.

Usage:
python -m cli.sal_score \
    --input sal_outputs/grads/qwen3-8b/hand-crafted \
    --output sal_outputs/results/qwen3-8b/hand-crafted \
    --data ww/hand-crafted \
    --reference all
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
import json

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
    traj_idx:   int     # index into the loaded data list
    step_idx:   int     # step index within the trajectory
    is_mistake: bool    # whether this is the gold mistake step


@dataclass
class GradientStore:
    """All gradient data stacked into a single matrix with index mappings.

    Attributes
    ----------
    G           : (T, d) float32 tensor — all raw gradient vectors.
    index       : list[StepIndex] of length T — per-row metadata.
    traj_meta   : list[dict] — per-trajectory metadata dicts.
    traj_ranges : list[tuple[int,int]] — (start_row, end_row) for each traj.
    device      : torch device where G lives.
    """
    G:           torch.Tensor
    index:       list[StepIndex]
    traj_meta:   list[dict]
    traj_ranges: list[tuple[int, int]]
    device:      torch.device


# ─────────────────────────────────────────────────────────────────────────────
# Loading: files → single stacked matrix on GPU
# ─────────────────────────────────────────────────────────────────────────────

def load_and_stack(input_dir: Path, device: torch.device) -> GradientStore:
    """Load all .pt gradient files and stack into one (T, d) matrix on device.

    Returns a GradientStore with the stacked matrix, index mapping, and
    per-trajectory metadata.
    """
    files = sorted(input_dir.glob("*.pt"))
    files = [f for f in files if f.name != "config.pt"]
    # files = files[:2]
    if not files:
        raise FileNotFoundError(f"No .pt gradient files in {input_dir}")

    # ── First pass: collect all gradient tensors + build index ────────
    all_grads: list[torch.Tensor] = []
    index: list[StepIndex] = []
    traj_meta: list[dict] = []
    traj_ranges: list[tuple[int, int]] = []

    row = 0
    for traj_idx, fp in enumerate(tqdm(files, desc="Loading .pt files")):
        payload = torch.load(fp, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        gradients = payload["gradients"]
        mistake_step = int(metadata["mistake_step"])

        traj_meta.append(metadata)
        start_row = row

        for step_idx in sorted(int(k) for k in gradients.keys()):
            grad_vec = gradients[step_idx]   # (d,) float16
            all_grads.append(grad_vec)
            index.append(StepIndex(
                row=row,
                traj_idx=traj_idx,
                step_idx=step_idx,
                is_mistake=(step_idx == mistake_step),
            ))
            row += 1

        traj_ranges.append((start_row, row))

    # ── Stack into one matrix and move to GPU ────────────────────────
    T = len(all_grads)
    d = all_grads[0].shape[0]
    print(f"Stacking {T} vectors × d={d:,} → ({T}, {d})")

    G = torch.stack(all_grads, dim=0).to(dtype=torch.float32, device=device)
    mem_gb = G.element_size() * G.numel() / 1e9
    print(f"  G on {device}: {mem_gb:.2f} GB")

    return GradientStore(
        G=G, index=index, traj_meta=traj_meta,
        traj_ranges=traj_ranges, device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Reference gradient (vectorised on GPU)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reference_gradient(
    store:    GradientStore,
    strategy: str = "all",
) -> torch.Tensor:
    """Compute ∇̄ on GPU using boolean masking for the chosen strategy.

    Returns (d,) float32 tensor on the same device as store.G.
    """
    if strategy == "all":
        ref_grad = store.G.mean(dim=0)
        n = store.G.shape[0]
    elif strategy == "pre_mistake":
        mask = torch.tensor(
            [entry.step_idx < int(
                store.traj_meta[entry.traj_idx]["mistake_step"]
            ) for entry in store.index],
            dtype=torch.bool, device=store.device,
        )
        n = mask.sum().item()
        if n == 0:
            raise RuntimeError("No pre-mistake steps found.")
        ref_grad = store.G[mask].mean(dim=0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"  Reference gradient: strategy={strategy}, "
          f"n_steps={n}, ‖∇̄‖₁={ref_grad.abs().sum().item():.4f}")
    return ref_grad


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 & 3: Center in-place + SVD (on GPU)
# ─────────────────────────────────────────────────────────────────────────────

def center_and_svd(
    store:        GradientStore,
    ref_grad:     torch.Tensor,
    n_components: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract ref_grad from G in-place, then compute top singular vector(s).

    After this call, store.G contains centered vectors.

    Returns
    -------
    (V, S)
        V : (d,) or (d, n_components) — top singular vector(s).
        S : (n_components,)           — corresponding singular values.
    """
    # ── Center in-place ──────────────────────────────────────────────
    store.G -= ref_grad.unsqueeze(0)  # broadcast (T, d) - (1, d)
    print(f"  Centered G in-place.")

    # ── Randomised SVD on GPU ────────────────────────────────────────
    print(f"  Running torch.svd_lowrank(q={n_components}, niter=5) "
          f"on {store.device} ...")
    U, S, V = torch.svd_lowrank(store.G, q=n_components, niter=5)
    # V: (d, n_components),  S: (n_components,)

    print(f"  Top {n_components} singular value(s): "
          f"{[f'{s:.4f}' for s in S.tolist()]}")

    if n_components == 1:
        return V.squeeze(1), S
    return V, S


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Compute SAL scores (single batched matmul)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sal_scores(
    store: GradientStore,
    v:     torch.Tensor,
) -> torch.Tensor:
    """Compute τ = (G @ v)² for all steps in one operation.

    Assumes store.G is already centered.

    Returns (T,) tensor of SAL scores on the same device.
    """
    projections = store.G @ v        # (T,)
    scores = projections.square()    # (T,)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Format results for evaluate_gradnorm.py
# ─────────────────────────────────────────────────────────────────────────────

def format_results(
    store:  GradientStore,
    scores: torch.Tensor,
) -> list[dict]:
    """Convert SAL scores into per-trajectory JSON dicts.

    Output format matches evaluate_gradnorm.py expectations:
    each trajectory gets a dict with "metadata", "steps", "logs".
    The SAL score appears under l1_norm/l2_norm with key "sal".
    """
    scores_cpu = scores.cpu().tolist()
    n_trajs = len(store.traj_meta)
    results = []

    for traj_idx in range(n_trajs):
        start, end = store.traj_ranges[traj_idx]
        metadata = store.traj_meta[traj_idx]

        logs = []
        for row in range(start, end):
            entry = store.index[row]
            sc = scores_cpu[row]
            logs.append({
                "step_idx":  entry.step_idx,
                "sal_score": sc,
                "l1_norm":   {"sal": sc},
                "l2_norm":   {"sal": sc},
            })

        results.append({
            "metadata": metadata,
            "steps":    [],
            "logs":     logs,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics (fully vectorised on GPU)
# ─────────────────────────────────────────────────────────────────────────────

def print_diagnostics(
    store:    GradientStore,
    scores:   torch.Tensor,
    ref_grad: torch.Tensor,
):
    """Print diagnostic info using vectorised GPU ops.

    Assumes store.G is already centered. Reconstructs raw norms via
    un-centering on the fly.
    """
    T = scores.shape[0]

    # ── Raw L1 norms (un-center to get original gradients) ───────────
    raw_l1 = (store.G + ref_grad.unsqueeze(0)).abs().sum(dim=1)   # (T,)

    # ── Move to numpy ────────────────────────────────────────────────
    sal_np = scores.cpu().numpy()
    raw_np = raw_l1.cpu().numpy()

    # ── Mistake mask (vectorised) ────────────────────────────────────
    mistake_mask = np.array([e.is_mistake for e in store.index], dtype=bool)

    # ── Pearson correlation ──────────────────────────────────────────
    corr = np.corrcoef(sal_np, raw_np)[0, 1]
    print(f"\n  ── Diagnostics ──")
    print(f"  Pearson(SAL, raw_L1):     {corr:.4f}")

    # ── Mean SAL for mistake vs normal ───────────────────────────────
    if mistake_mask.any() and (~mistake_mask).any():
        mean_mistake = sal_np[mistake_mask].mean()
        mean_normal  = sal_np[~mistake_mask].mean()
        print(f"  Mean SAL (mistake steps): {mean_mistake:.6f}")
        print(f"  Mean SAL (normal steps):  {mean_normal:.6f}")
        print(f"  Ratio (mistake/normal):   "
              f"{mean_mistake / max(mean_normal, 1e-12):.2f}")

    # ── Quick Acc@1 per trajectory (vectorised via slicing) ──────────
    correct = 0
    total = 0
    for traj_idx, (start, end) in enumerate(store.traj_ranges):
        if start == end:
            continue
        total += 1
        best_local = scores[start:end].argmax().item()
        predicted_step = store.index[start + best_local].step_idx
        mistake_step = int(store.traj_meta[traj_idx]["mistake_step"])
        if predicted_step == mistake_step:
            correct += 1

    print(f"  Quick Acc@1 (argmax):     {correct}/{total} = "
          f"{100 * correct / max(total, 1):.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: step and agent Acc@k
# ─────────────────────────────────────────────────────────────────────────────

def step_at_k(results: list[dict], k: int, reverse: bool = True) -> float:
    """Fraction of trajectories where the gold mistake step is in the top-k.

    reverse=True  → rank by descending score (highest = most anomalous).
    reverse=False → rank by ascending score  (lowest  = most anomalous).
    """
    correct = 0
    count = 0
    for example in results:
        mistake_step = int(example["metadata"]["mistake_step"])
        ranked = sorted(example["logs"],
                        key=lambda x: x["sal_score"], reverse=reverse)[:k]
        ranked_idxs = [x["step_idx"] for x in ranked]
        correct += int(mistake_step in ranked_idxs)
        count += 1
    return correct / count


def agent_at_k(
    results:      list[dict],
    trajectories: list,
    k:            int,
    reverse:      bool = True,
) -> float:
    """Fraction of trajectories where the gold mistake agent is in the top-k.

    Uses case-insensitive substring matching (e.g. "Orchestrator" matches
    "Orchestrator (thought)").
    """
    # Build filename → {step_idx: role} mapping
    role_map: dict[str, dict[int, str]] = {}
    for traj in trajectories:
        roles = {i: step.get("role", "") for i, step in enumerate(traj.history)}
        role_map[traj.filename] = roles

    correct = 0
    count = 0
    for example in results:
        true_agent = example["metadata"]["mistake_agent"]
        filename = example["metadata"]["filename"]
        step_roles = role_map.get(filename, {})
        if not step_roles:
            continue

        ranked = sorted(example["logs"],
                        key=lambda x: x["sal_score"], reverse=reverse)[:k]
        top_agents = [step_roles.get(x["step_idx"], "") for x in ranked]
        correct += int(any(
            true_agent.lower() in agent.lower() for agent in top_agents
        ))
        count += 1
    return correct / count


def print_evaluation(
    results:      list[dict],
    trajectories: list | None = None,
    ks:           list[int] = [1, 3, 5, 10],
):
    """Print step@k and agent@k for both ascending and descending rankings."""
    print(f"\n  ── Evaluation ({len(results)} trajectories) ──")

    for k in ks:
        s_desc = step_at_k(results, k, reverse=True)
        s_asc  = step_at_k(results, k, reverse=False)
        print(f"  step@{k}  desc: {s_desc:.4f}   asc: {s_asc:.4f}")

        if trajectories is not None:
            a_desc = agent_at_k(results, trajectories, k, reverse=True)
            a_asc  = agent_at_k(results, trajectories, k, reverse=False)
            print(f"  agent@{k} desc: {a_desc:.4f}   asc: {a_asc:.4f}")

        print("  " + "--" * 20)


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
    p.add_argument("--device",     default=None,
                   help="Device for computation (default: cuda if available).")
    p.add_argument("--save_artifacts", action="store_true",
                   help="Save ref_grad, singular vector, and singular values.")
    p.add_argument("--data",       default=None,
                   help="Original dataset directory (e.g. ww/hand-crafted) "
                        "for loading step→agent roles. Required for evaluation.")
    p.add_argument("--subset",     default=None,
                   help="Subset filter for load_dataset (e.g. hand-crafted).")
    p.add_argument("--ks",         nargs="+", type=int, default=[1, 5, 10],
                   help="k values for Acc@k evaluation. Default: 1 5 10.")
    return p.parse_args()


def main():
    args = parse_args()

    # if args.device is None:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # else:
    #     device = torch.device(args.device)

    device = torch.device("cpu")
    input_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ─────────────────────────────────────────────────────
    config_path = input_dir / "config.json"
    if config_path.exists():
        # config = torch.load(config_path, map_location="cpu", weights_only=False)
        config = json.load(open(config_path))
        print(f"Config: target={config['target_param']}, d={config['d']:,}, "
              f"proj_dim={config.get('proj_dim')}")
    else:
        print("Warning: no config.json found.")

    # ── Load and stack all gradients onto GPU ───────────────────────────
    store = load_and_stack(input_dir, device)
    print(f"Loaded {len(store.traj_meta)} trajectories, "
          f"{store.G.shape[0]} total steps.")

    # ── Step 1: Reference gradient ──────────────────────────────────────
    print("\n── Step 1: Reference gradient ──")
    ref_grad = compute_reference_gradient(store, strategy=args.reference)

    # ── Step 2-3: Center + SVD ──────────────────────────────────────────
    print("\n── Step 2-3: Center + SVD ──")
    v, singular_values = center_and_svd(
        store, ref_grad, n_components=args.n_components,
    )

    # ── Step 4: SAL scores (single matmul) ──────────────────────────────
    print("\n── Step 4: SAL scores ──")
    scores = compute_sal_scores(store, v)
    print(f"  Computed {scores.shape[0]} SAL scores.")

    # ── Diagnostics ─────────────────────────────────────────────────────
    print_diagnostics(store, scores, ref_grad)

    # ── Format results ──────────────────────────────────────────────────
    results = format_results(store, scores)

    # ── Evaluation ──────────────────────────────────────────────────────
    trajectories = None
    if args.data is not None:
        from core.data import load_dataset

        data_path = Path(args.data)
        if args.subset:
            base_path, subset = str(data_path), args.subset
        else:
            base_path, subset = str(data_path.parent), data_path.name
        trajectories = load_dataset(base_path, subset=subset)

    print_evaluation(results, trajectories, ks=args.ks)

    # ── Save results ────────────────────────────────────────────────────
    print("\n── Saving results ──")
    for result in results:
        filename = result["metadata"]["filename"]
        out_path = out_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(f"Saved {len(results)} result files to {out_dir}")

    # ── Optionally save artifacts ───────────────────────────────────────
    if args.save_artifacts:
        artifacts = {
            "ref_grad":           ref_grad.cpu(),
            "singular_vector":    v.cpu(),
            "singular_values":    singular_values.cpu(),
            "reference_strategy": args.reference,
        }
        artifact_path = out_dir / "sal_artifacts.pt"
        torch.save(artifacts, artifact_path)
        print(f"Saved artifacts to {artifact_path}")


if __name__ == "__main__":
    main()
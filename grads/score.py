from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable
from .sal_scoring import make_sal_scoring_fn

from concurrent.futures import ThreadPoolExecutor


try:
    from safetensors.torch import save_file
    from safetensors import safe_open
except ImportError:
    print("Please install safetensors: pip install safetensors")


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
    """
    Stores gradients for multiple weight names across the same set of steps.
    """
    # Maps: weight_name -> (T, d) matrix
    Gs: dict[str, torch.Tensor] 
    
    # Metadata is shared across all layers because step indices are identical
    index:       list[StepIndex]
    lookup:      dict[tuple[int, int], int]
    traj_meta:   dict[dict]
    traj_ranges: list[tuple[int, int]]
    device:      torch.device

    def __getitem__(self, weight_name: str) -> torch.Tensor:
        return self.Gs[weight_name]

    @property
    def layer_names(self) -> list[str]:
        return list(self.Gs.keys())


def get_all_weight_names(fp: Path):
    with safe_open(fp, framework="pt") as f:
        return sorted({k.split(".", 1)[1] for k in f.keys()})
        
def load_and_stack(
    model: str,
    subset: str,
    weight_names: list[str],  # List of layers to load, e.g. ["down/31", "up/31"]
    data_dir: Path,
    device: torch.device,
    grad_dir: Path,
):
    input_dir = grad_dir / model / subset
    files = sorted(input_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    
    # Initialize containers for each requested layer
    if weight_names == "all":
        weight_names = get_all_weight_names(files[0])
        
    layer_collections = {name: [] for name in weight_names}
    index: list[StepIndex] = []
    lookup: dict[tuple[int, int], int] = {}
    traj_meta: dict[dict] = {}
    traj_ranges: list[tuple[int, int]] = []

    row = 0
    for file_idx, fp in enumerate(tqdm(files, desc="Loading Multi-Layers")):
        traj_idx = int(fp.stem) # /path/to/1.safetensors -> 1
        
        with safe_open(fp, framework="pt", device="cpu") as f:
            # Load metadata
            header = f.metadata()
            metadata = json.loads(header.get("payload_metadata", "{}"))
            mistake_step = int(metadata.get("mistake_step", -1))
            traj_meta[traj_idx] = metadata
            
            # Use the first requested layer to determine step indices 
            # (assuming all layers exist for all steps)
            first_layer = weight_names[0]
            step_keys = [k for k in f.keys() if k.endswith(f".{first_layer}")]
            step_indices = sorted([int(k.split(".")[0]) for k in step_keys])
            
            # Load matching JSON for roles
            with open(data_dir / fp.with_suffix(".json").name) as jf:
                traj_data = json.load(jf)
                history = traj_data['history']
                
            start_row = row
            for step_idx in step_indices:
                # 1. Collect tensors for EVERY requested layer at this step
                for name in weight_names:
                    key = f"{step_idx}.{name}"
                    layer_collections[name].append(f.get_tensor(key))
                
                # 2. Record index (only once per step)
                index.append(StepIndex(
                    row=row, traj_idx=traj_idx, step_idx=step_idx,
                    role=history[step_idx]["role"],
                    is_mistake=(step_idx == mistake_step),
                ))
                lookup[(traj_idx, step_idx)] = row
                row += 1

            traj_ranges.append((start_row, row))

    # Convert lists to stacked matrices and move to device
    Gs = {
        name: torch.stack(tensors).to(device) 
        for name, tensors in layer_collections.items()
    }

    return GradientStore(
        Gs=Gs, index=index, lookup=lookup,
        traj_meta=traj_meta, traj_ranges=traj_ranges,
        device=device,
    )

def standardize_role(role: str) -> str:
    if "orchestrator" in role.lower(): return "Orchestrator"
    else: return role

def compute_metrics(
    scores: np.ndarray,
    store: GradientStore,
    mistake_indices: list[int | None],  # absolute step_idx in history
    mistake_roles: list[str | None],
    ks: list[int],
    direction: str,
) -> dict:
    ascending    = (direction == "asc")
    total_trajs  = len(store.traj_ranges)
    step_hits    = {k: 0 for k in ks}
    agent_hits   = {k: 0 for k in ks}

    for (start, end), mistake_step, mistake_role in zip(
        store.traj_ranges, mistake_indices, mistake_roles
    ):
        if mistake_step is None:
            continue

        # Pair each entry with its score, then rank by score
        traj_entries = store.index[start:end]
        traj_scores  = scores[start:end]
        step_scores  = [(entry.step_idx, entry.role, score) 
                        for entry, score in zip(traj_entries, traj_scores)]
        step_scores.sort(key=lambda x: x[2], reverse=not ascending)

        ranked_steps  = [step_idx for step_idx, _, _ in step_scores]
        ranked_roles  = [standardize_role(role) for _, role, _ in step_scores]
        mistake_rank  = ranked_steps.index(mistake_step) + 1  # 1-based ranking.

        for k in ks:
            if mistake_rank <= k:
                step_hits[k] += 1
            if mistake_role in ranked_roles[:k]:
                agent_hits[k] += 1

    return {
        **{f"step@{k}_{direction}":  step_hits[k]  / total_trajs for k in ks},
        **{f"agent@{k}_{direction}": agent_hits[k] / total_trajs for k in ks},
    }

def evaluate_weights(
    store: GradientStore,
    scoring_fn: Callable[[torch.Tensor], torch.Tensor],
    ks: list[int] = [1, 3, 5],
) -> pd.DataFrame:

    # --- Phase 1: Compute scores for all weights ---
    all_scores: dict[str, np.ndarray] = {}
    for weight_name, G in tqdm(store.Gs.items(), desc="Scoring"):
        all_scores[weight_name] = scoring_fn(G).cpu().numpy()

    # --- Precompute trajectory metadata ---
    mistake_indices: list[int | None] = []
    mistake_roles:   list[str | None] = []

    for start, end in store.traj_ranges:
        traj_index  = store.index[start:end]
        mistake_entry = next((e for e in traj_index if e.is_mistake), None)
        mistake_role = store.traj_meta[mistake_entry.traj_idx]['mistake_agent']
        mistake_idx = mistake_entry.step_idx

        mistake_roles.append(mistake_role)
        mistake_indices.append(mistake_idx)
        # mistake_roles.append(mistake_entry.role if mistake_entry else None)

    # --- Phase 2: Evaluate predictions (parallelized over weights) ---
    results = []
    for weight_name, scores in tqdm(all_scores.items(), desc="Predicting"):
        row = {"weight": weight_name}
        for direction in ["asc", "desc"]:
            row |= compute_metrics(
                scores, 
                store, 
                mistake_indices, 
                mistake_roles, 
                ks, 
                direction
            )
        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values("step@1_asc", ascending=False).reset_index(drop=True)
    return df

def save_results(df: pd.DataFrame, out_dir: Path, subset: str, ks: list[int]) -> None:
    """
    Splits the wide evaluation DataFrame into per-(k, direction) TSV files.

    Output: {out_dir}/metrics/{subset}_k{k}_{direction}.tsv
    Columns: weight, step_acc, agent_acc
    """
    metrics_dir = out_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for k in ks:
        for direction in ["asc", "desc"]:
            out = df[["weight"]].copy()
            out["step_acc"]  = df[f"step@{k}_{direction}"]
            out["agent_acc"] = df[f"agent@{k}_{direction}"]
            out = out.sort_values("step_acc", ascending=False).reset_index(drop=True)

            path = metrics_dir / f"{subset}_k{k}_{direction}.tsv"
            out.to_csv(path, sep="\t", index=False)
            print(f"Saved {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate gradient norms for failure attribution.")
    p.add_argument("--model",    required=True, help="Model tag, e.g. llama-3.1-8b")
    p.add_argument("--subset",   required=True, help="Subset name, e.g. hand-crafted")
    p.add_argument("--grad_dir", default="outputs/grads",  help="Root gradient directory")
    p.add_argument("--data_dir", default=None, help="JSON data dir (defaults to ww/{subset})")
    p.add_argument("--out_dir",  default=None, help="Output dir (defaults to outputs/grads/{model}/{subset})")
    p.add_argument("--weights",  default="all", nargs="+", help="Weight names to load, or 'all'")
    p.add_argument("--ks",       default=[1, 3, 5], nargs="+", type=int)
    p.add_argument("--norm",     choices=["l1", "l2"], default="l1")
    p.add_argument("--device",   default="cuda")
    return p.parse_args()

def sweep():
    MODELS = ["llama-3.1-8b", "qwen3-8b"]
    SUBSETS = ["hand-crafted", "algorithm-generated"]
    SVD_FUNCTIONS = {
        f"sal_{'wref' if centered else 'noref'}_c{c}": make_sal_scoring_fn(c, centered)
        for centered in (True, False)
        for c in range(1, 11)
    }
    GRADNORM_FUNCTIONS = dict(
        gradnorm_l1=(lambda G: G.float().norm(p=1, dim=1)),
        gradnorm_l2=(lambda G: G.float().norm(p=2, dim=1)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ks = [1, 3, 5, 10]

    for model in MODELS:
        for subset in SUBSETS:
            print(f"\n=== {model} / {subset} ===")

            store = load_and_stack(
                model=model, subset=subset,
                weight_names="all",
                data_dir=Path("ww") / subset,
                device=device,
                grad_dir=Path("outputs/grads"),
            )
            out_dir = Path("outputs/grads") / model / "metrics"

            for name, fn in {**GRADNORM_FUNCTIONS, **SVD_FUNCTIONS}.items():
                print(f"  scoring: {name}")
                df = evaluate_weights(store, scoring_fn=fn, ks=ks)
                save_results(df, out_dir=out_dir / name, subset=subset, ks=ks)

            del store

if __name__ == "__main__":
    sweep()

# if __name__ == "__main__":
#     args = parse_args()

#     data_dir = Path(args.data_dir) if args.data_dir else Path("ww") / args.subset
#     out_dir  = Path(args.out_dir)  if args.out_dir  else Path("outputs/grads") / args.model / args.subset / "metrics"
#     device   = torch.device(args.device)

#     print(f"Model:   {args.model}")
#     print(f"Subset:  {args.subset}")
#     print(f"Device:  {device}")

#     store = load_and_stack(
#         model=args.model, subset=args.subset,
#         weight_names=args.weights if args.weights != ["all"] else "all",
#         data_dir=data_dir, device=device, grad_dir=Path(args.grad_dir),
#     )

#     scoring_fn = (lambda G: G.float().norm(p=1, dim=1)) if args.norm == "l1" \
#             else (lambda G: G.float().norm(p=2, dim=1))

#     df = evaluate_weights(store, scoring_fn=scoring_fn, ks=args.ks)
#     save_results(df, out_dir=out_dir, subset=args.subset, ks=args.ks)

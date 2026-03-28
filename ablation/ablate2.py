"""
Ablation sweep (v2): earliest-among-top-k evaluation.

Instead of checking whether the gold step appears anywhere in the top-k
lowest-scoring steps, this variant selects the single step with the
**lowest step_idx** among the top-k candidates and compares that against
the gold step.

Motivation: the gradient-norm method has a systematic late-prediction bias
(longer contexts → mechanically lower scores).  Tie-breaking toward the
earliest candidate counteracts this confound.

python -m ablation.ablate2
--> much lower results as k increases.
"""

from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from .core import (
    NORM_TYPES,
    build_strategies, load_trajectories,
    get_param_names_and_sizes, discover_n_layers,
    CompiledConfigs, score_step, _resolve_agent,
)


# ── per-trajectory evaluation (v2) ───────────────────────────────────────────

def evaluate_trajectory_v2(
    traj:      dict,
    cc:        CompiledConfigs,
    norm_type: str,
    k:         int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Evaluate one trajectory with earliest-among-top-k selection.

    1. Score all valid steps.
    2. Pick the top-*k* steps with the lowest scores.
    3. Among those *k* candidates, select the one with the smallest
       ``step_idx``.
    4. Compare that single predicted step (and its agent) against the
       ground truth.

    Returns
    -------
    ``(step_correct, agent_correct)`` — each of shape ``(C,)`` with
    values 0.0 or 1.0 — or ``None`` if the trajectory has no valid logs.
    """
    valid_logs = [log for log in traj["logs"] if log.get("statistics")]
    if not valid_logs:
        return None

    # Ground truth
    meta          = traj["metadata"]
    mistake_step  = int(meta["mistake_step"])
    mistake_agent = meta["mistake_agent"]
    step_roles    = {s["step_idx"]: s["role"] for s in traj["steps"]}

    # Score and rank
    score_matrix = np.stack([
        score_step(log, cc, norm_type) for log in valid_logs
    ])  # (S, C)
    step_indices = np.array(
        [int(log["step_idx"]) for log in valid_logs],
        dtype=np.int64,
    )

    # Top-k lowest-scoring step indices per config: shape (k, C)
    top_k_rows = np.argsort(score_matrix, axis=0)[:k]          # (k, C)
    top_k_steps = step_indices[top_k_rows]                      # (k, C)

    # Among the k candidates, pick the one with the smallest step_idx
    predicted_steps = top_k_steps[np.argmin(top_k_steps, axis=0),
                                  np.arange(top_k_steps.shape[1])]  # (C,)

    # Step-level accuracy
    step_correct = (predicted_steps == mistake_step).astype(np.float64)

    # Agent-level accuracy
    n_configs = cc.mask.shape[0]
    agent_correct = np.empty(n_configs, dtype=np.float64)
    for c in range(n_configs):
        pred_agent = _resolve_agent(
            step_roles.get(int(predicted_steps[c]), "unknown")
        )
        agent_correct[c] = float(mistake_agent == pred_agent)

    return step_correct, agent_correct


# ── multi-trajectory evaluation (v2) ─────────────────────────────────────────

def evaluate_trajectories_v2(
    trajectories: list[dict],
    cc:           CompiledConfigs,
    norm_type:    str,
    k:            int,
) -> pd.DataFrame:
    """Evaluate all trajectories with earliest-among-top-k and return
    per-config accuracy.

    Returns
    -------
    pd.DataFrame
        Columns: ``config``, ``step_acc``, ``agent_acc``.
    """
    n_configs         = len(cc.names)
    step_correct_sum  = np.zeros(n_configs, dtype=np.float64)
    agent_correct_sum = np.zeros(n_configs, dtype=np.float64)
    n_total           = 0

    for traj in trajectories:
        result = evaluate_trajectory_v2(traj, cc, norm_type, k)
        if result is None:
            continue
        step_correct, agent_correct = result
        step_correct_sum  += step_correct
        agent_correct_sum += agent_correct
        n_total           += 1

    denom = max(n_total, 1)
    return pd.DataFrame({
        "config":    cc.names,
        "step_acc":  step_correct_sum  / denom,
        "agent_acc": agent_correct_sum / denom,
    })


# ── sweep ─────────────────────────────────────────────────────────────────────

def sweep(
    model:     str,
    subset:    str,
    norm_type: str,
    k:         int,
    verbose:   bool = False,
) -> pd.DataFrame:
    results_dir  = Path("ablation") / model / subset
    trajectories = load_trajectories(results_dir)
    param_names, param_sizes = get_param_names_and_sizes(trajectories)
    n_layers     = discover_n_layers(param_names)
    if verbose:
        print(f"Discovered {n_layers} layers.")

    strategies = build_strategies(n_layers)

    out_dir  = Path(f"ablation/{model}/results-v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / f"{subset}_k{k}_{norm_type}.tsv"

    strategy_dfs = {}
    for name, config_dict in strategies.items():
        if verbose:
            print(f"Running strategy: {name} ({len(config_dict)} configs)...")
        cc = CompiledConfigs.compile(config_dict, param_names, param_sizes)
        df = evaluate_trajectories_v2(trajectories, cc, norm_type, k)
        strategy_dfs[name] = df

    agg = (
        pd.concat(
            [df.assign(strategy=name) for name, df in strategy_dfs.items()],
            ignore_index=True,
        )
        .sort_values("step_acc", ascending=False)
        .reset_index(drop=True)
        [["strategy", "config", "step_acc", "agent_acc"]]
    )

    agg.to_csv(agg_path, sep="\t", index=False, float_format="%.4f")
    return agg


def _sweep_unpacked(args):
    norm_type, k, model, subset = args
    return args, sweep(model=model, subset=subset, norm_type=norm_type, k=k)


KS     = [1, 3, 5, 10]
MODELS = ["llama-3.1-8b", "qwen3-8b"]
SUBSETS = ["algorithm-generated", "hand-crafted"]


if __name__ == "__main__":
    combos = list(product(NORM_TYPES, KS, MODELS, SUBSETS))
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_sweep_unpacked, combo): combo for combo in combos}
        results = {}
        for future in tqdm(as_completed(futures), total=len(combos)):
            args, agg = future.result()
            results[args] = agg
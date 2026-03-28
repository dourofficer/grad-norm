"""
Ablation sweep variant: filter out steps with no successors before scoring.

For hand-crafted trajectories, many steps (ledger updates, next-speaker logs,
etc.) are leaf nodes in the dependency graph — no downstream step depends on
them.  These steps add noise to the argmin comparison.  This variant removes
them before ranking, so only steps that actually influence later computation
compete for the "predicted mistake" slot.

Algorithm-generated trajectories have a flat sequential structure, so no
filtering is applied.

python -m ablation.ablate3
--> horible results
"""

from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.graph import derive_llm_inputs, get_dependency_dict

from .core import (
    NORM_TYPES,
    build_strategies, load_trajectories,
    get_param_names_and_sizes, discover_n_layers,
    CompiledConfigs, score_step, _resolve_agent,
)


# ── successor filtering ──────────────────────────────────────────────────────

def get_steps_with_successors(history: list[dict]) -> set[int]:
    """Return the set of step indices that have at least one successor.

    A step *s* has a successor if any other step lists *s* in its
    dependency inputs.  Steps not in this set are leaf nodes.
    """
    deps = get_dependency_dict(derive_llm_inputs(history))
    has_successor: set[int] = set()
    for inputs in deps.values():
        has_successor.update(inputs)
    return has_successor


def is_handcrafted(traj: dict) -> bool:
    """Detect whether a trajectory is from the hand-crafted subset."""
    return any(
        s.get("role", "").startswith("Orchestrator")
        for s in traj["steps"]
    )


# ── per-trajectory evaluation (with filtering) ────────────────────────────────

def evaluate_trajectory_filtered(
    traj:      dict,
    cc:        CompiledConfigs,
    norm_type: str,
    k:         int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Like ``core.evaluate_trajectory``, but drops leaf steps first.

    For hand-crafted trajectories, only steps that have at least one
    successor in the dependency graph are kept.  For algorithm-generated
    trajectories, all steps are kept (no filtering).
    """
    valid_logs = [log for log in traj["logs"] if log.get("statistics")]
    if not valid_logs:
        return None

    # Filter leaf steps for hand-crafted trajectories
    if is_handcrafted(traj):
        keep = get_steps_with_successors(traj["steps"])
        valid_logs = [log for log in valid_logs if int(log["step_idx"]) in keep]
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
    ])
    step_indices = np.array(
        [int(log["step_idx"]) for log in valid_logs], dtype=np.int64,
    )

    pred_step_matrix = step_indices[np.argsort(score_matrix, axis=0)[:k]]

    # argmax
    # pred_step_matrix = step_indices[np.argsort(score_matrix, axis=0)[::-1][:k]]

    step_correct = np.any(
        pred_step_matrix == mistake_step, axis=0,
    ).astype(np.float64)

    n_configs = cc.mask.shape[0]
    agent_correct = np.empty(n_configs, dtype=np.float64)
    for c in range(n_configs):
        predicted_agents = [
            _resolve_agent(step_roles.get(int(idx), "unknown"))
            for idx in pred_step_matrix[:, c]
        ]
        agent_correct[c] = float(mistake_agent in predicted_agents)

    return step_correct, agent_correct


# ── multi-trajectory evaluation ───────────────────────────────────────────────

def evaluate_trajectories_filtered(
    trajectories: list[dict],
    cc:           CompiledConfigs,
    norm_type:    str,
    k:            int,
) -> pd.DataFrame:
    """Evaluate all trajectories with leaf-step filtering."""
    n_configs         = len(cc.names)
    step_correct_sum  = np.zeros(n_configs, dtype=np.float64)
    agent_correct_sum = np.zeros(n_configs, dtype=np.float64)
    n_total           = 0

    for traj in trajectories:
        result = evaluate_trajectory_filtered(traj, cc, norm_type, k)
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
    strategies   = build_strategies(n_layers)

    if verbose:
        print(f"Discovered {n_layers} layers.")

    out_dir  = Path(f"ablation/{model}/aggregated-results-v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / f"{subset}_k{k}_{norm_type}_filtered.tsv"

    strategy_dfs = {}
    for name, config_dict in strategies.items():
        if verbose:
            print(f"Running strategy: {name} ({len(config_dict)} configs)...")
        cc = CompiledConfigs.compile(config_dict, param_names, param_sizes)
        df = evaluate_trajectories_filtered(trajectories, cc, norm_type, k)
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


KS      = [1, 3, 5, 10]
MODELS  = ["llama-3.1-8b", "qwen3-8b"]
SUBSETS = ["algorithm-generated", "hand-crafted"]


if __name__ == "__main__":
    combos = list(product(NORM_TYPES, KS, MODELS, SUBSETS))
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_sweep_unpacked, combo): combo for combo in combos}
        results = {}
        for future in tqdm(as_completed(futures), total=len(combos)):
            args, agg = future.result()
            results[args] = agg
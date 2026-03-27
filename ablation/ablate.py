import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ── strategy definitions ──────────────────────────────────────────────────────
def build_strategies(L: int) -> dict[str, dict[str, str]]:
    def per_layer(prefix, suffix=""):
        return {f"{prefix}/{i}": rf"model\.layers\.{i}\.{suffix}" for i in range(L)}

    return {
        "layer": {
            **per_layer("layer"),
            "lm_head":     r"lm_head\.",
            "embed_tokens": r"model\.embed_tokens\.",
        },
        "mlp":         per_layer("mlp",  r"mlp\."),
        "mlp_weights": {
            **per_layer("gate", r"mlp\.gate_proj\."),
            **per_layer("up",   r"mlp\.up_proj\."),
            **per_layer("down", r"mlp\.down_proj\."),
        },
        "attn":        per_layer("attn", r"self_attn\."),
        "attn_weights": {
            **per_layer("q", r"self_attn\.q_proj\."),
            **per_layer("k", r"self_attn\.k_proj\."),
            **per_layer("v", r"self_attn\.v_proj\."),
            **per_layer("o", r"self_attn\.o_proj\."),
        },
    }


# ── data loading ──────────────────────────────────────────────────────────────
def load_trajectories(results_dir: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(results_dir.glob("*.json"))]


def get_param_names_and_sizes(trajectories: list[dict]) -> tuple[list[str], np.ndarray]:
    sample_stats = next(
        log["statistics"]
        for data in trajectories
        for log in data["logs"]
        if log.get("statistics")
    )
    param_names = list(sample_stats.keys())
    param_sizes = np.array([sample_stats[p]["n_params"] for p in param_names], dtype=np.float64)
    return param_names, param_sizes


def discover_n_layers(param_names: list[str]) -> int:
    """Infer number of layers from the highest layer index in param names."""
    indices = [
        int(m.group(1))
        for p in param_names
        if (m := re.search(r"model\.layers\.(\d+)\.", p))
    ]
    if not indices:
        raise ValueError("Could not discover n_layers: no 'model.layers.N.' params found.")
    return max(indices) + 1


# ── scoring ───────────────────────────────────────────────────────────────────
def score_log(
    log: dict,
    param_names: list[str],
    mask_matrix: np.ndarray,
    n_params_per_config: np.ndarray,
    norm_type: str,
) -> np.ndarray:
    """
    Compute a normalised gradient norm score for each parameter group in one log entry.

    For each group, aggregates per-parameter norms across all parameters in the group,
    then normalises by the group's total parameter count:
      - l1_norm: mean absolute gradient  = sum(|w|)  / N
      - l2_norm: RMS gradient norm       = sqrt(sum(w²)) / N

    Args:
        log:                 A single log entry containing a "statistics" dict keyed by
                             parameter name, each with "l1_norm" or "l2_norm_sq" fields.
        param_names:         Ordered list of parameter names matching the stats keys.
        mask_matrix:         Boolean matrix of shape (n_configs, n_params) where entry
                             [i, j] is True if parameter j belongs to group i.
        n_params_per_config: Total parameter count per group, shape (n_configs,).
        norm_type:           One of "l1_norm" or "l2_norm".

    Returns:
        Normalised scores of shape (n_configs,). Groups with no parameters are NaN.
    """
    stats = log["statistics"]
    safe_counts = np.where(n_params_per_config > 0, n_params_per_config, np.nan)

    with np.errstate(invalid="ignore"):
        if norm_type == "l2_norm":
            vals = np.array([stats[p]["l2_norm_sq"] for p in param_names], dtype=np.float64)
            return np.sqrt((mask_matrix @ vals)) / safe_counts
        else:
            vals = np.array([stats[p]["l1_norm"] for p in param_names], dtype=np.float64)
            return (mask_matrix @ vals) / safe_counts


# ── per-trajectory evaluation ─────────────────────────────────────────────────
def evaluate_trajectory(
    data: dict,
    param_names: list[str],
    mask_matrix: np.ndarray,
    n_params_per_config: np.ndarray,
    norm_type: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (step_correct, agent_correct) of shape (n_configs,), or None if no valid logs."""
    meta          = data["metadata"]
    mistake_step  = int(meta["mistake_step"])
    mistake_agent = meta["mistake_agent"]
    step_roles    = {s["step_idx"]: s["role"] for s in data["steps"]}

    valid_logs = [log for log in data["logs"] if log.get("statistics")]
    if not valid_logs:
        return None

    score_matrix     = np.stack([
        score_log(log, param_names, mask_matrix, n_params_per_config, norm_type)
        for log in valid_logs
    ])                                                        # (n_logs, n_configs)
    step_indices     = np.array([log["step_idx"] for log in valid_logs])
    pred_step_matrix = step_indices[np.argsort(score_matrix, axis=0)[:k]]  # (k, n_configs)

    step_correct  = np.any(pred_step_matrix == mistake_step, axis=0).astype(float)
    agent_correct = np.array([
        float(mistake_agent in [
            "Orchestrator" if "orchestrator" in step_roles.get(idx, "").lower() else step_roles.get(idx, "unknown")
            for idx in pred_step_matrix[:, c]
        ])
        for c in range(mask_matrix.shape[0])
    ])
    return step_correct, agent_correct


# ── strategy sweep ────────────────────────────────────────────────────────────
def run_strategy(
    trajectories: list[dict],
    config_dict: dict[str, str],
    param_names: list[str],
    param_sizes: np.ndarray,
    norm_type: str,
    k: int,
) -> pd.DataFrame:
    """Sweep all trajectories for one strategy. Returns DataFrame: config, step_acc, agent_acc."""
    config_names    = list(config_dict.keys())
    config_patterns = list(config_dict.values())
    n_configs       = len(config_names)

    mask_matrix = np.array(
        [[bool(re.search(pat, p)) for p in param_names] for pat in config_patterns],
        dtype=np.float64,
    )
    n_params_per_config = mask_matrix @ param_sizes

    step_correct_sum  = np.zeros(n_configs)
    agent_correct_sum = np.zeros(n_configs)
    n_total           = 0

    for data in trajectories:
        result = evaluate_trajectory(
            data, param_names, mask_matrix, n_params_per_config, norm_type, k
        )
        if result is None:
            continue
        step_correct, agent_correct = result
        step_correct_sum  += step_correct
        agent_correct_sum += agent_correct
        n_total           += 1

    denom = n_total if n_total > 0 else 1
    return pd.DataFrame({
        "config":    config_names,
        "step_acc":  step_correct_sum  / denom,
        "agent_acc": agent_correct_sum / denom,
    })


# ── main ──────────────────────────────────────────────────────────────────────
def sweep(
    model:       str,
    subset:      str,
    norm_type:   str,
    k:           int,
    verbose: bool = False,
):
    results_dir  = Path("ablation") / model / subset
    trajectories = load_trajectories(results_dir)
    param_names, param_sizes = get_param_names_and_sizes(trajectories)
    n_layers     = discover_n_layers(param_names)
    if verbose: print(f"Discovered {n_layers} layers.")
    strategies   = build_strategies(n_layers)

    out_dir = Path(f"ablation/{model}/aggregated-results")
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / f"{subset}_k{k}_{norm_type}.tsv"

    strategy_dfs = {}
    for name, config_dict in strategies.items():
        if verbose: print(f"Running strategy: {name} ({len(config_dict)} configs)...")
        df = run_strategy(trajectories, config_dict, param_names, param_sizes, norm_type, k)
        strategy_dfs[name] = df

    agg = (
        pd.concat([df.assign(strategy=name) for name, df in strategy_dfs.items()], ignore_index=True)
        .sort_values("step_acc", ascending=False)
        .reset_index(drop=True)
        [["strategy", "config", "step_acc", "agent_acc"]]
    )

    agg.to_csv(agg_path, sep="\t", index=False, float_format="%.4f")
    return agg

def sweep_unpacked(args):
    norm_type, k, model, subset = args
    return args, sweep(model=model, subset=subset, norm_type=norm_type, k=k)

NORM_TYPES   = ["l1_norm", "l2_norm"]
KS          = [1, 3, 5, 10]
MODELS       = ["llama-3.1-8b", "qwen3-8b"]
SUBSETS      = ["algorithm-generated", "hand-crafted"]


if __name__ == "__main__":
    combos = list(product(NORM_TYPES, KS, MODELS, SUBSETS))
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(sweep_unpacked, combo): combo for combo in combos}
        results = {}
        for future in tqdm(as_completed(futures), total=len(combos)):
            args, agg = future.result()
            results[args] = agg
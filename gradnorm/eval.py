"""
Core scoring and evaluation functions for gradient-norm ablation studies.

This module owns the math: scoring steps, scoring trajectories, and evaluating
predictions against ground-truth annotations.  It knows nothing about file paths,
strategy construction, or plotting.

Typical usage
-------------
>>> from ablation.core import build_strategies, load_trajectories, \
...     get_param_names_and_sizes, discover_n_layers
>>> from ablation.core import CompiledConfigs, score_step, evaluate_trajectories
>>>
>>> trajs = load_trajectories(results_dir)
>>> param_names, param_sizes = get_param_names_and_sizes(trajs)
>>> strategies = build_strategies(discover_n_layers(param_names))
>>>
>>> cc = CompiledConfigs.compile(strategies["layer"], param_names, param_sizes)
>>> df = evaluate_trajectories(trajs, cc, "l1_norm", k=1)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ── constants ─────────────────────────────────────────────────────────────────

STRATEGIES = ["layer", "mlp", "attn", "mlp_weights", "attn_weights"]
NORM_TYPES = ["l1_norm", "l2_norm"]


# ── strategy definitions ──────────────────────────────────────────────────────

def build_strategies(L: int) -> dict[str, dict[str, str]]:
    def per_layer(prefix, suffix=""):
        return {f"{prefix}/{i}": rf"model\.layers\.{i}\.{suffix}" for i in range(L)}

    return {
        "layer": {
            **per_layer("layer"),
            "lm_head":      r"lm_head\.",
            "embed_tokens":  r"model\.embed_tokens\.",
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
    param_sizes = np.array(
        [sample_stats[p]["n_params"] for p in param_names], dtype=np.float64,
    )
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


# ── compiled config representation ────────────────────────────────────────────

@dataclass(frozen=True)
class CompiledConfigs:
    """Pre-compiled parameter-group masks ready for vectorised scoring.

    Attributes
    ----------
    names : list[str]
        Human-readable config names (e.g. ``["layer/0", "layer/1", …]``).
    param_names : list[str]
        Ordered parameter names matching the ``statistics`` dicts in log
        entries.  Length *P*.
    mask : np.ndarray
        Boolean matrix of shape ``(C, P)`` where ``mask[i, j]`` is True if
        parameter *j* belongs to config *i*.  Stored as ``float64`` so it
        can be used directly in matrix multiplications.
    n_params : np.ndarray
        Total parameter count per config, shape ``(C,)``.  Derived as
        ``mask @ param_sizes``.
    """

    names:       list[str]
    param_names: list[str]
    mask:        np.ndarray   # (C, P)  float64
    n_params:    np.ndarray   # (C,)    float64

    @classmethod
    def compile(
        cls,
        configs:     dict[str, str],
        param_names: list[str],
        param_sizes: np.ndarray,
    ) -> CompiledConfigs:
        """Build a ``CompiledConfigs`` from a raw strategy config dict.

        Parameters
        ----------
        configs : dict[str, str]
            Mapping from config name to a regex pattern that matches the
            parameter names belonging to that group (e.g.
            ``{"layer/0": r"model\\.layers\\.0\\.", …}``).
        param_names : list[str]
            Ordered list of all parameter names in the model.
        param_sizes : np.ndarray
            Number of scalar parameters per entry in *param_names*, shape
            ``(P,)``.
        """
        names    = list(configs.keys())
        patterns = list(configs.values())

        mask = np.array(
            [[bool(re.search(pat, p)) for p in param_names] for pat in patterns],
            dtype=np.float64,
        )
        n_params = mask @ param_sizes

        return cls(
            names=names,
            param_names=param_names,
            mask=mask,
            n_params=n_params,
        )


# ── step-level scoring ────────────────────────────────────────────────────────

def score_step(
    step_log:  dict,
    cc:        CompiledConfigs,
    norm_type: str,
) -> np.ndarray:
    """Score a single log entry for every config in *cc*.

    Parameters
    ----------
    step_log : dict
        A single log entry whose ``"statistics"`` dict is keyed by parameter
        name.  Each value must contain ``"l1_norm"`` and ``"l2_norm_sq"``
        fields.
    cc : CompiledConfigs
        Pre-compiled config masks.
    norm_type : ``"l1_norm"`` | ``"l2_norm"``
        Which norm to aggregate.

    Returns
    -------
    np.ndarray
        Scores of shape ``(C,)`` — one per config.  Configs with zero
        parameters produce ``NaN``.
    """
    stats       = step_log["statistics"]
    safe_counts = np.where(cc.n_params > 0, cc.n_params, np.nan)

    with np.errstate(invalid="ignore"):
        if norm_type == "l2_norm":
            vals = np.array(
                [stats[p]["l2_norm_sq"] for p in cc.param_names],
                dtype=np.float64,
            )
            return np.sqrt(cc.mask @ vals) / safe_counts
        else:
            vals = np.array(
                [stats[p]["l1_norm"] for p in cc.param_names],
                dtype=np.float64,
            )
            return (cc.mask @ vals) / safe_counts


# ── trajectory-level scoring ──────────────────────────────────────────────────

def score_trajectory(
    traj:      dict,
    cc:        CompiledConfigs,
    norm_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every valid step in a trajectory.

    Parameters
    ----------
    traj : dict
        A single trajectory dict with ``"logs"`` (list of log entries).
    cc : CompiledConfigs
        Pre-compiled config masks.
    norm_type : ``"l1_norm"`` | ``"l2_norm"``

    Returns
    -------
    score_matrix : np.ndarray
        Shape ``(S, C)`` where *S* is the number of valid log entries and
        *C* is the number of configs.
    step_indices : np.ndarray
        Shape ``(S,)`` — the ``step_idx`` of each valid log, aligned with
        the rows of *score_matrix*.

    Raises
    ------
    ValueError
        If the trajectory contains no valid log entries (none with
        ``"statistics"``).
    """
    valid_logs = [log for log in traj["logs"] if log.get("statistics")]
    if not valid_logs:
        raise ValueError("Trajectory contains no valid log entries.")

    score_matrix = np.stack([
        score_step(log, cc, norm_type) for log in valid_logs
    ])  # (S, C)
    step_indices = np.array(
        [int(log["step_idx"]) for log in valid_logs],
        dtype=np.int64,
    )
    return score_matrix, step_indices


# ── per-trajectory evaluation ─────────────────────────────────────────────────

def _resolve_agent(role: str) -> str:
    """Normalise a step role string for agent-level matching.

    The Who&When hand-crafted subset uses ``"Orchestrator (thought)"``,
    ``"Orchestrator (-> WebSurfer)"``, etc.  We collapse all Orchestrator
    variants to ``"Orchestrator"`` so they match the ground-truth label.
    """
    if "orchestrator" in role.lower():
        return "Orchestrator"
    return role


def evaluate_trajectory(
    traj:      dict,
    cc:        CompiledConfigs,
    norm_type: str,
    k:         int,
    ascending: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Evaluate top-*k* predictions for one trajectory.

    The predicted mistake step is the step with the **lowest** score
    (``argmin``).  When *k* > 1 the top-*k* lowest-scoring steps are
    considered; a prediction is correct if the ground-truth step or agent
    appears among them.

    Parameters
    ----------
    traj : dict
        Trajectory dict with ``"logs"``, ``"metadata"`` (containing
        ``"mistake_step"`` and ``"mistake_agent"``), and ``"steps"``
        (each with ``"step_idx"`` and ``"role"``).
    cc : CompiledConfigs
        Pre-compiled config masks.
    norm_type : ``"l1_norm"`` | ``"l2_norm"``
    k : int
        Number of top predictions to consider.

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

    # Top-k scoring (ascending/descending) step indices per config: shape (k, C)
     # pred_step_matrix = step_indices[np.argsort(score_matrix, axis=0)[:k]]
    ranked = np.argsort(score_matrix, axis=0) # sort ascending as default
    if ascending: pred_step_matrix = step_indices[ranked[:k]]
    else:         pred_step_matrix = step_indices[ranked[::-1][:k]]
   
    # Step-level accuracy: is the ground-truth step among the top-k?
    step_correct = np.any(
        pred_step_matrix == mistake_step, axis=0,
    ).astype(np.float64)

    # Agent-level accuracy: is the ground-truth agent among the predicted
    # steps' agents?
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

def evaluate_trajectories(
    trajectories: list[dict],
    cc:           CompiledConfigs,
    norm_type:    str,
    k:            int,
    ascending:    bool = True,
) -> pd.DataFrame:
    """Evaluate all trajectories and return per-config accuracy.

    Parameters
    ----------
    trajectories : list[dict]
        List of trajectory dicts.
    cc : CompiledConfigs
        Pre-compiled config masks.
    norm_type : ``"l1_norm"`` | ``"l2_norm"``
    k : int
        Number of top predictions per trajectory.

    Returns
    -------
    pd.DataFrame
        Columns: ``config``, ``step_acc``, ``agent_acc``.  One row per
        config in *cc*.
    """
    n_configs         = len(cc.names)
    step_correct_sum  = np.zeros(n_configs, dtype=np.float64)
    agent_correct_sum = np.zeros(n_configs, dtype=np.float64)
    n_total           = 0

    for traj in trajectories:
        result = evaluate_trajectory(traj, cc, norm_type, k, ascending)
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


# ── helpers for *_dist.py scripts ─────────────────────────────────────────────

def load_top_configs(
    agg_dir:  Path,
    subset:   str,
    k_sweep:  int,
    norm_type: str,
    k_top:    int,
) -> dict[str, list[str]]:
    """Load aggregated TSV and pick the top-*k_top* configs per strategy.

    Parameters
    ----------
    agg_dir : Path
        Directory containing aggregated result TSVs.
    subset : str
        Subset name (e.g. ``"hand-crafted"``).
    k_sweep : int
        The *k* value used during the sweep that produced the TSV.
    norm_type : str
        ``"l1_norm"`` or ``"l2_norm"``.
    k_top : int
        Number of top configs to select per strategy.

    Returns
    -------
    dict[str, list[str]]
        ``{strategy_name: [config_name, …]}`` with *k_top* best configs
        ranked by ``step_acc``.
    """
    tsv_path = agg_dir / f"{subset}_k{k_sweep}_{norm_type}.tsv"
    df = pd.read_csv(tsv_path, sep="\t")
    return {
        strat: (
            df[df["strategy"] == strat]
            .sort_values("step_acc", ascending=False)
            .head(k_top)["config"]
            .tolist()
        )
        for strat in STRATEGIES
    }


def compile_top_configs(
    top_config_names: dict[str, list[str]],
    all_strategies:   dict[str, dict[str, str]],
    param_names:      list[str],
    param_sizes:      np.ndarray,
) -> dict[str, CompiledConfigs]:
    """Compile a :class:`CompiledConfigs` for each strategy's selected configs.

    Parameters
    ----------
    top_config_names : dict[str, list[str]]
        ``{strategy_name: [config_name, …]}`` — output of
        :func:`load_top_configs`.
    all_strategies : dict[str, dict[str, str]]
        Full strategy dict from :func:`build_strategies`.
    param_names : list[str]
        Ordered parameter names.
    param_sizes : np.ndarray
        Parameter counts, shape ``(P,)``.

    Returns
    -------
    dict[str, CompiledConfigs]
        ``{strategy_name: CompiledConfigs}`` ready for scoring.
    """
    compiled: dict[str, CompiledConfigs] = {}
    for strat, cfg_names in top_config_names.items():
        sub_configs = {name: all_strategies[strat][name] for name in cfg_names}
        compiled[strat] = CompiledConfigs.compile(sub_configs, param_names, param_sizes)
    return compiled
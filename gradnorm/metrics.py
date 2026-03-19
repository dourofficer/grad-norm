"""
metrics.py ― Accuracy@k metrics for step and agent prediction.

All functions operate on a single trajectory's result dict (as returned by
gradnorm.score_trajectory) or over a list of such dicts.

Public API
----------
acc_at_k()          step Acc@k for one trajectory
agent_acc_at_k()    agent Acc@k for one trajectory
compute_metrics()   aggregate Acc@k over a list of trajectory results
"""
from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory metrics
# ─────────────────────────────────────────────────────────────────────────────

def step_at_k(
    scores:    dict[int, float],
    true_step: int,
    k:         int,
) -> int:
    """Step Acc@k for a single trajectory.

    A trajectory is a hit at rank k if the ground-truth mistake step is
    among the k highest-scored steps.

    Parameters
    ----------
    scores    : dict mapping step_idx → GradNorm score.
    true_step : ground-truth mistake step index.
    k         : rank cutoff (k=1 means the top-scored step must be correct).

    Returns
    -------
    1 if true_step is in the top-k, 0 otherwise.
    Returns 0 if true_step is not in scores (unscored / skipped step).
    """
    if true_step not in scores:
        return 0
    # Sort steps by score descending; ties broken by step index (arbitrary but stable).
    ranked = sorted(scores, key=lambda idx: (scores[idx], -idx), reverse=True)
    return int(true_step in ranked[:k])


def agent_at_k(
    scores:      dict[int, float],
    step_agents: dict[int, str],
    true_agent:  str,
    k:           int,
) -> int:
    """Agent Acc@k for a single trajectory.

    A trajectory is a hit at rank k if the ground-truth mistake agent is
    among the agents of the k highest-scored steps.

    Parameters
    ----------
    scores      : dict mapping step_idx → GradNorm score.
    step_agents : dict mapping step_idx → agent role string.
    true_agent  : ground-truth mistake agent name.
    k           : rank cutoff.

    Returns
    -------
    1 if true_agent appears in the agents of the top-k scored steps, 0 otherwise.

    Notes
    -----
    Agent matching is an *exact string match* on the role field.  If the
    dataset uses variant spellings (e.g. "WebSurfer" vs "Web Surfer"), normalise
    before calling this function.
    """
    ranked = sorted(scores, key=lambda idx: (scores[idx], -idx), reverse=True)
    top_k_agents = {step_agents.get(idx, "") for idx in ranked[:k]}
    return int(true_agent in top_k_agents)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    results: list[dict[str, Any]],
    ks:      list[int] | tuple[int, ...] = (1, 2, 3, 5, 10),
) -> dict[str, float]:
    """Aggregate step and agent Acc@k over a list of trajectory result dicts.

    Parameters
    ----------
    results : list of dicts as returned by gradnorm.score_trajectory().
              Each dict must contain:
                  "scores"      : dict[int, float]
                  "true_step"   : int
                  "true_agent"  : str
                  "step_agents" : dict[int, str]
    ks      : k values to evaluate.

    Returns
    -------
    dict[str, float]
        Keys follow the pattern "step_acc@{k}" and "agent_acc@{k}".
        Example: {"step_acc@1": 0.43, "agent_acc@1": 0.51, "step_acc@3": 0.65, ...}
    """
    n = len(results)
    if n == 0:
        return {}

    totals: dict[str, int] = {}
    for k in ks:
        totals[f"step_acc@{k}"]  = 0
        totals[f"agent_acc@{k}"] = 0

    for res in results:
        for k in ks:
            totals[f"step_acc@{k}"] += step_at_k(
                res["scores"], res["true_step"], k
            )
            totals[f"agent_acc@{k}"] += agent_at_k(
                res["scores"], res["step_agents"], res["true_agent"], k
            )

    return {key: val / n for key, val in totals.items()}

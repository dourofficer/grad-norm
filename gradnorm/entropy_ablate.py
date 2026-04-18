"""
gradnorm.entropy_eval — Accuracy evaluation for entropy-scored trajectories.

Reads per-trajectory JSONs written by ``gradnorm.entropy``, ranks steps by
entropy score, and reports step- and agent-level accuracy@k.

Output: one TSV per (model, subset, order) at
    <base_dir>/<model>/<subset>/accuracy_<order>.tsv

Columns: k  step_acc  agent_acc  n_total

Usage
-----
python -m gradnorm.entropy_ablate \
    --base_dir outputs/entropy \
    --models   llama-3.1-8b \
    --subsets  algorithm-generated \
    --ks       1 3 5 10 \
    --orders   ascending descending
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import product
from pathlib import Path

import pandas as pd


# ── agent normalisation (mirrors gradnorm.eval._resolve_agent) ─────────────────

def _resolve_agent(role: str) -> str:
    if re.search(r"orchestrator", role, re.IGNORECASE):
        return "Orchestrator"
    return role


# ── per-trajectory accuracy ────────────────────────────────────────────────────

def _eval_one(traj: dict, k: int, ascending: bool) -> tuple[bool, bool] | None:
    """Return (step_correct, agent_correct) or None if trajectory is unusable."""
    logs = [e for e in traj.get("logs", []) if e.get("entropy") is not None]
    if not logs:
        return None

    meta          = traj["metadata"]
    mistake_step  = str(meta["mistake_step"])
    mistake_agent = meta["mistake_agent"]

    step_roles = {s["step_idx"]: s["role"] for s in traj.get("steps", [])}

    sorted_logs = sorted(logs, key=lambda e: e["entropy"], reverse=not ascending)
    top_k       = sorted_logs[:k]

    step_correct  = mistake_step  in {str(e["step_idx"]) for e in top_k}
    agent_correct = mistake_agent in {
        _resolve_agent(step_roles.get(e["step_idx"], "")) for e in top_k
    }
    return step_correct, agent_correct


# ── directory-level sweep ──────────────────────────────────────────────────────

def evaluate(result_dir: Path, ks: list[int], orders: list[str]) -> pd.DataFrame:
    """Compute accuracy@k for every (k, order) pair in *result_dir*.

    Parameters
    ----------
    result_dir : Path
        Directory containing per-trajectory JSONs
        (i.e. ``<base_dir>/<model>/<subset>/``).
    ks : list[int]
    orders : list[str]  — ``"ascending"`` or ``"descending"``

    Returns
    -------
    pd.DataFrame with columns ``order``, ``k``, ``step_acc``, ``agent_acc``,
    ``n_total``.
    """
    files = sorted(result_dir.glob("*.json"))
    trajs = [json.loads(f.read_text()) for f in files]

    rows = []
    for order in orders:
        ascending = order == "ascending"
        for k in ks:
            step_sum = agent_sum = n = 0
            for traj in trajs:
                res = _eval_one(traj, k, ascending)
                if res is None:
                    continue
                step_correct, agent_correct = res
                step_sum  += step_correct
                agent_sum += agent_correct
                n         += 1
            rows.append({
                "order":      order,
                "k":          k,
                "step_acc":   step_sum  / max(n, 1),
                "agent_acc":  agent_sum / max(n, 1),
                "n_total":    n,
            })

    return pd.DataFrame(rows)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entropy accuracy sweep (step & agent accuracy@k).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_dir",  default="outputs/entropy",
                   help="Root dir containing <model>/<subset>/ result folders.")
    p.add_argument("--models",  nargs="+", required=True,
                   help="Model directory names under base_dir/.")
    p.add_argument("--subsets", nargs="+", default=["algorithm-generated", "hand-crafted"])
    p.add_argument("--ks",      nargs="+", type=int, default=[1, 3, 5, 10], metavar="K")
    p.add_argument("--orders",  nargs="+", default=["ascending", "descending"],
                   choices=["ascending", "descending"])
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    base  = Path(args.base_dir)

    for model, subset in product(args.models, args.subsets):
        result_dir = base / model / subset / "metrics"
        if not result_dir.exists():
            print(f"[skip] {result_dir} not found")
            continue

        df = evaluate(result_dir, args.ks, args.orders)

        for order, grp in df.groupby("order"):
            out_path = result_dir / f"accuracy_{order}.tsv"
            (grp.drop(columns="order")
                .sort_values("k")
                .to_csv(out_path, sep="\t", index=False, float_format="%.4f"))
            print(f"[saved] {out_path}")
            print(grp.drop(columns="order").sort_values("k").to_string(index=False))
            print()


if __name__ == "__main__":
    main()
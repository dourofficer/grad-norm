"""
cli/evaluate_gradnorm.py ― Evaluate per-layer GradNorm scores from cli.gradnorm outputs.

Reads the JSON files produced by cli.gradnorm (one per trajectory), computes
step-level and agent-level Acc@k for every (subset, layer, norm_type) combination,
and writes a TSV table that matches the layout:

    Subset | GradNorm Layer | Step@1 | Step@5 | Step@10 | Agent@1 | Agent@5 | Agent@10

Each metric cell contains "L1_pct / L2_pct" (e.g. "34.48 / 27.59").

Usage
-----
python -m cli.evaluate_gradnorm \
    --input   outputs/llama-3.1-8b/grad-norm/hand-crafted \
    --output  results/gradnorm_eval.tsv

# Evaluate multiple subset dirs at once (one --input per subset):
python -m cli.evaluate_gradnorm \
    --input   outputs/qwen3-8b/grad-norm/hand-crafted \
              outputs/qwen3-8b/grad-norm/algorithm-generated \
    --output  results/gradnorm_qwen.tsv \
    --ks 1 5 10

python -m cli.evaluate_gradnorm \
    --input   outputs/llama-3.1-8b/grad-norm/hand-crafted \
              outputs/llama-3.1-8b/grad-norm/algorithm-generated \
    --output  results/gradnorm_llama.tsv \
    --ks 1 5 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers  (mirrors gradnorm/metrics.py but operates on the on-disk
#                  JSON schema produced by cli.gradnorm)
# ─────────────────────────────────────────────────────────────────────────────

def _step_at_k(scores: dict[int, float], true_step: int, k: int) -> int:
    """1 if true_step is among the k highest-scored steps, else 0."""
    if true_step not in scores:
        return 0
    ranked = sorted(scores, key=lambda idx: (scores[idx], -idx), reverse=False)
    return int(true_step in ranked[:k])


def _agent_at_k(
    scores:      dict[int, float],
    step_agents: dict[int, str],
    true_agent:  str,
    k:           int,
) -> int:
    """1 if true_agent appears in the agents of the k highest-scored steps, else 0."""
    ranked = sorted(scores, key=lambda idx: (scores[idx], -idx), reverse=False)
    top_k_agents = [step_agents.get(idx, "") for idx in ranked[:k]]
    # if k == 5 and true_agent.lower() == "orchestrator":
    #     import pdb; pdb.set_trace()

    return int(any([true_agent.lower() in agent.lower() for agent in top_k_agents]))

    # return int(true_agent in top_k_agents)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_results(input_dir: Path) -> list[dict[str, Any]]:
    """Load all *.json result files from a cli.gradnorm output directory."""
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")
    results = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def _discover_layers(results: list[dict[str, Any]]) -> list[str]:
    """
    Return an ordered list of layer names found across all log entries.

    Layer names come from the keys of the l1_norm / l2_norm dicts.
    Ordering: numeric layers first (layer_0, layer_1, ...) then any others
    (e.g. lm_head) in the order they were first encountered.
    """
    seen: dict[str, None] = {}  # insertion-ordered set
    for result in results:
        for log in result.get("logs", []):
            for key in log.get("l1_norm", {}):
                seen[key] = None
    # Sort: keys of the form "layer_<N>" numerically, rest appended after
    numeric, rest = [], []
    for name in seen:
        parts = name.split("_")
        if len(parts) == 2 and parts[0] == "layer" and parts[1].isdigit():
            numeric.append((int(parts[1]), name))
        else:
            rest.append(name)
    numeric.sort()
    return [name for _, name in numeric] + rest


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory score extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_scores(
    result: dict[str, Any],
    layer:  str,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Build (l1_scores, l2_scores) dicts mapping step_idx -> score for one layer.

    Steps whose norm dict is empty (degenerate / skipped) are omitted,
    so they won't appear in the ranking.
    """
    l1: dict[int, float] = {}
    l2: dict[int, float] = {}
    for log in result.get("logs", []):
        idx   = int(log["step_idx"])
        l1_val = log.get("l1_norm", {}).get(layer)
        l2_val = log.get("l2_norm", {}).get(layer)
        if l1_val is not None:
            l1[idx] = float(l1_val)
        if l2_val is not None:
            l2[idx] = float(l2_val)
    return l1, l2


def _step_agents(result: dict[str, Any]) -> dict[int, str]:
    """Map step_idx -> role string from the 'steps' list."""
    return {int(s["step_idx"]): s["role"] for s in result.get("steps", [])}


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_layer_metrics(
    results:  list[dict[str, Any]],
    layer:    str,
    ks:       list[int],
) -> dict[str, float]:
    """
    Aggregate step- and agent-level Acc@k over *results* for one layer.

    Returns a dict with keys like "step_l1@1", "step_l2@1",
    "agent_l1@1", "agent_l2@1", ... for each k in ks.
    Accuracy is expressed as a percentage (0-100).
    """
    n = len(results)
    if n == 0:
        return {}

    totals: dict[str, int] = {
        f"{metric}_{norm}@{k}": 0
        for metric in ("step", "agent")
        for norm in ("l1", "l2")
        for k in ks
    }

    for result in results:
        meta       = result["metadata"]
        true_step  = int(meta["mistake_step"])
        true_agent = meta["mistake_agent"]
        agents     = _step_agents(result)
        l1, l2     = _extract_scores(result, layer)

        for k in ks:
            totals[f"step_l1@{k}"]  += _step_at_k(l1, true_step, k)
            totals[f"step_l2@{k}"]  += _step_at_k(l2, true_step, k)
            totals[f"agent_l1@{k}"] += _agent_at_k(l1, agents, true_agent, k)
            totals[f"agent_l2@{k}"] += _agent_at_k(l2, agents, true_agent, k)

    return {key: (val / n) * 100 for key, val in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Table building
# ─────────────────────────────────────────────────────────────────────────────

def build_table(
    subset_results: dict[str, list[dict[str, Any]]],
    ks:             list[int],
) -> list[dict[str, str]]:
    """
    Return a list of row dicts, one row per (subset, layer) combination.

    Each row has keys:
        "subset", "layer",
        "step@{k}", "agent@{k}"  for each k  (value = "L1_pct / L2_pct")
    """
    rows = []
    for subset, results in subset_results.items():
        layers = _discover_layers(results)
        for layer in layers:
            metrics = compute_layer_metrics(results, layer, ks)
            row: dict[str, str] = {"subset": subset, "layer": layer}
            for k in ks:
                l1_s = metrics.get(f"step_l1@{k}",  0.0)
                l2_s = metrics.get(f"step_l2@{k}",  0.0)
                l1_a = metrics.get(f"agent_l1@{k}", 0.0)
                l2_a = metrics.get(f"agent_l2@{k}", 0.0)
                row[f"step@{k}"]  = f"{l1_s:.2f} / {l2_s:.2f}"
                row[f"agent@{k}"] = f"{l1_a:.2f} / {l2_a:.2f}"
            rows.append(row)
    return rows


def write_tsv(rows: list[dict[str, str]], ks: list[int], output_path: Path) -> None:
    """Write the table to a TSV file."""
    col_order = (
        ["subset", "layer"]
        + [f"step@{k}"  for k in ks]
        + [f"agent@{k}" for k in ks]
    )
    header = "\t".join(col_order)
    lines  = [header]
    for row in rows:
        lines.append("\t".join(row.get(col, "") for col in col_order))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} rows -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helper
# ─────────────────────────────────────────────────────────────────────────────

def print_table(rows: list[dict[str, str]], ks: list[int]) -> None:
    """Print a human-readable version of the table to stdout."""
    col_order = (
        ["subset", "layer"]
        + [f"step@{k}"  for k in ks]
        + [f"agent@{k}" for k in ks]
    )
    widths = {col: len(col) for col in col_order}
    for row in rows:
        for col in col_order:
            widths[col] = max(widths[col], len(row.get(col, "")))

    sep  = "  ".join("-" * widths[c] for c in col_order)
    head = "  ".join(c.ljust(widths[c]) for c in col_order)
    print(sep)
    print(head)
    print(sep)

    prev_subset = None
    for row in rows:
        if row["subset"] != prev_subset and prev_subset is not None:
            print(sep)
        prev_subset = row["subset"]
        print("  ".join(row.get(c, "").ljust(widths[c]) for c in col_order))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate per-layer GradNorm scores and produce a Step/Agent Acc@k table.\n"
            "Reads JSONs written by cli.gradnorm; each JSON must contain 'metadata', "
            "'steps', and 'logs' fields."
        )
    )
    p.add_argument(
        "--input", nargs="+", required=True, metavar="DIR",
        help=(
            "One or more directories of cli.gradnorm output JSONs. "
            "The subset label is inferred from the directory name "
            "(e.g. 'hand-crafted', 'algorithm-generated') unless --subsets is given."
        ),
    )
    p.add_argument(
        "--output", required=True, metavar="PATH",
        help="Path for the output TSV file (e.g. results/gradnorm_eval.tsv).",
    )
    p.add_argument(
        "--ks", nargs="+", type=int, default=[1, 5, 10],
        metavar="K",
        help="k values for Acc@k. Default: 1 5 10.",
    )
    p.add_argument(
        "--subsets", nargs="+", default=None, metavar="NAME",
        help=(
            "Override subset labels for the --input dirs (one label per dir). "
            "If omitted, the last component of each input path is used as the label."
        ),
    )
    p.add_argument(
        "--no_print", action="store_true",
        help="Suppress the pretty-printed table on stdout.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dirs = [Path(d) for d in args.input]

    if args.subsets:
        if len(args.subsets) != len(input_dirs):
            raise ValueError(
                f"--subsets has {len(args.subsets)} entries but "
                f"--input has {len(input_dirs)}."
            )
        labels = args.subsets
    else:
        labels = [d.name for d in input_dirs]

    subset_results: dict[str, list[dict[str, Any]]] = {}
    for label, d in zip(labels, input_dirs):
        print(f"Loading subset '{label}' from {d} ...")
        subset_results[label] = _load_results(d)
        print(f"  {len(subset_results[label])} trajectories loaded.")

    rows = build_table(subset_results, ks=args.ks)

    if not args.no_print:
        print()
        print_table(rows, ks=args.ks)
        print()
        print("(Each cell: L1_acc% / L2_acc%)")
        print()

    write_tsv(rows, ks=args.ks, output_path=Path(args.output))


if __name__ == "__main__":
    main()
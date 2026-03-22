"""
python -m cli.evaluate --dir 'outputs/qwen3-8b/hand-crafted' --ks 1 5 10

python -m cli.evaluate \
    --sweep \
    --configs outputs/qwen3-8b/all-at-once/hand-crafted \
              outputs/qwen3-8b/step-by-step-full/hand-crafted \
              outputs/qwen3-8b/step-by-step-partial/hand-crafted \
              outputs/qwen3-8b/all-at-once/algorithm-generated \
              outputs/qwen3-8b/step-by-step-full/algorithm-generated \
              outputs/qwen3-8b/step-by-step-partial/algorithm-generated \
    --ks 1 5 10 \
    --save outputs/qwen3-8b/sweep_results.tsv

# Auto-discover configs and run prediction + evaluation in one shot:
python -m cli.evaluate --sweep --base_output outputs/qwen3-8b --predict_first --ks 1 5 10
python -m cli.evaluate --sweep --base_output outputs/qwen3-8b --ks 1 5 10
python -m cli.evaluate --sweep --base_output outputs/qwen3-8b --ks 1 5 10 --by_length --n_bins 3
"""
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from utils.common import _get_sorted_json_files, _load_json_data

# Maps folder names → display labels in the results table
STRATEGY_LABELS = {
    'all-at-once':           'all-at-once',
    'step-by-step-full':     'step-by-step (k)',
    'step-by-step-partial':  'step-by-step (n)',
}
STRATEGY_ORDER = list(STRATEGY_LABELS.values())
SUBSET_ORDER   = ['hand-crafted', 'algorithm-generated']

DEFAULT_KS = [1, 3, 5, 10, 20]


# ---------------------------------------------------------------------------
# Shared data loader
# ---------------------------------------------------------------------------

def _load_result_dir(dir: str) -> list[dict]:
    """Load and validate all result JSONs from *dir*.

    Raises AssertionError with a clear message if predictions are missing,
    so callers don't need to repeat the check.
    """
    result_dir = Path(dir)
    data = []
    for filename in _get_sorted_json_files(result_dir):
        file_data = _load_json_data(result_dir / filename)
        file_data['metadata']['filename'] = filename
        data.append(file_data)

    assert data and "predictions" in data[0], (
        f"No predictions found in {dir}. Run cli.predict first "
        f"(or pass --predict_first to this command)."
    )
    return data


# ---------------------------------------------------------------------------
# Core accuracy helpers
# ---------------------------------------------------------------------------

def compute_acc(dir: str, k: int = 1) -> tuple[float, float]:
    """Compute accuracy@k for agent and step predictions.

    Returns
    -------
    (agent_acc, step_acc)  — both as percentages.
    """
    data = _load_result_dir(dir)

    correct_agent, correct_step = 0, 0
    correct_files, failed_files = [], []

    for entry in data:
        top_k    = entry["predictions"][:k]
        label    = entry["metadata"]
        filename = label["filename"]

        agent_match = label["mistake_agent"] in [p["role"]          for p in top_k]
        step_match  = label["mistake_step"]  in [str(p["step_idx"]) for p in top_k]

        if agent_match: correct_agent += 1
        if step_match:  correct_step  += 1

        if step_match:  correct_files.append(filename)
        else:           failed_files.append(filename)

    total     = len(data)
    agent_acc = (correct_agent / total) * 100
    step_acc  = (correct_step  / total) * 100

    print(f"\n--- Accuracy@{k} for {dir} ---")
    print(f"Total: {total}")
    print(f"Agent: {correct_agent}/{total} ({agent_acc:.2f}%)")
    print(f"Step:  {correct_step}/{total}  ({step_acc:.2f}%)")

    return agent_acc, step_acc


def compute_acc_by_trajectory_length(
    dir: str,
    k: int = 1,
    n_bins: int = 5,
    save_path: str | None = None,
) -> list[dict]:
    """Compute accuracy@k grouped by trajectory-length bins, with an overall row."""
    data   = _load_result_dir(dir)
    lengths = [len([x for x in e["steps"] if x["role"] != "loss"]) for e in data]
    edges   = np.unique(np.percentile(lengths, np.linspace(0, 100, n_bins + 1))).tolist()
    edges[-1] = float('inf')

    def bin_label(n):
        for lo, hi in zip(edges, edges[1:]):
            if lo <= n < hi:
                return f"{int(lo)}-{int(hi)-1}" if hi != float('inf') else f"{int(lo)}+"

    groups: dict[str, list] = defaultdict(list)
    for entry in data:
        groups[bin_label(len(entry["steps"]))].append(entry)

    def _acc_row(label, entries):
        top_ks = [e["predictions"][:k] for e in entries]
        labels = [e["metadata"]        for e in entries]
        ca = sum(l["mistake_agent"] in [p["role"]          for p in t] for l, t in zip(labels, top_ks))
        cs = sum(l["mistake_step"]  in [str(p["step_idx"]) for p in t] for l, t in zip(labels, top_ks))
        n  = len(entries)
        return {"trajectory_length": label, "k": k, "total": n,
                "agent_acc": round(ca / n * 100, 2), "step_acc": round(cs / n * 100, 2)}

    rows = []
    print(f"\n--- Accuracy@{k} by Trajectory Length ({dir}) ---")
    print(f"{'Length':<12} {'Total':>6} {'Agent':>10} {'Step':>10}")

    for length, entries in sorted(groups.items(),
                                  key=lambda x: int(x[0].split('-')[0].replace('+', ''))):
        row = _acc_row(length, entries)
        rows.append(row)
        print(f"{row['trajectory_length']:<12} {row['total']:>6} "
              f"{row['agent_acc']:>9.2f}% {row['step_acc']:>9.2f}%")

    all_row = _acc_row("all", data)
    rows.append(all_row)
    print(f"{'all':<12} {all_row['total']:>6} "
          f"{all_row['agent_acc']:>9.2f}% {all_row['step_acc']:>9.2f}%")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved to {save_path}")

    return rows


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------

def _discover_configs(base_output: str) -> list[str]:
    """Auto-discover all leaf dirs under base_output that match
    <base_output>/<strategy>/<subset>.
    """
    base = Path(base_output)
    configs = sorted([
        str(p) for p in base.glob("*/*")
        if p.is_dir() and any(p.iterdir())
    ])
    if not configs:
        raise ValueError(f"No output directories found under {base_output}")
    print(f"Discovered {len(configs)} config(s) under {base_output}:")
    for c in configs:
        print(f"  {c}")
    return configs


# ---------------------------------------------------------------------------
# Optional predict phase (--predict_first)
# ---------------------------------------------------------------------------

def _run_predict_phase(configs: list[str]) -> None:
    """Run cli.predict over every config dir, inferring the method from the folder name.

    This replaces the Phase 1 loop that used to live in sweep_eval.sh.
    The FOLDER_TO_METHOD map in cli.predict is the single source of truth.
    """
    # Import here to keep the top-level import graph clean and avoid
    # circular imports if predict ever imports evaluate.
    from cli.predict import populate_predictions, infer_method

    print("=" * 60)
    print("Phase 1: Populating predictions")
    print("=" * 60)

    for dir_path in configs:
        method = infer_method(dir_path)
        print(f"\n--- predict: method={method}  dir={dir_path} ---")
        populate_predictions(output_dir=dir_path, method=method)

    print("\nPhase 1 complete.\n")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep(
    configs:   list[str],
    ks:        list[int],
    save_path: str | None = None,
    by_length: bool = False,
    n_bins:    int  = 5,
) -> pd.DataFrame:
    """Sweep over configs × ks and print a results table.

    Table layout:
        Subset          | Strategy           | Step@1 | Step@5 | … | Agent@1 | Agent@5 | …
        hand-crafted    | all-at-once        |  ...
                        | step-by-step (k)   |  ...
        alg-generated   | …
    """
    rows = []

    for dir_path in configs:
        parts        = Path(dir_path).parts
        strategy_raw = parts[-2] if len(parts) >= 2 else ""
        subset       = parts[-1] if len(parts) >= 1 else ""
        strategy     = STRATEGY_LABELS.get(strategy_raw, strategy_raw)

        for k in ks:
            if by_length:
                for row in compute_acc_by_trajectory_length(dir_path, k=k, n_bins=n_bins):
                    rows.append({"subset": subset, "strategy": strategy, **row})
            else:
                agent_acc, step_acc = compute_acc(dir_path, k=k)
                rows.append({
                    "subset":    subset,
                    "strategy":  strategy,
                    "k":         k,
                    "agent_acc": round(agent_acc, 2),
                    "step_acc":  round(step_acc,  2),
                })

    df = pd.DataFrame(rows)

    if by_length:
        pivot = df.pivot_table(
            index=["subset", "strategy", "trajectory_length"],
            columns="k",
            values=["step_acc", "agent_acc"],
            aggfunc="first",
        )
    else:
        pivot = df.pivot_table(
            index=["subset", "strategy"],
            columns="k",
            values=["step_acc", "agent_acc"],
            aggfunc="first",
        )

    # Flatten and rename columns: step_acc_1 → Step@1, agent_acc_1 → Agent@1
    pivot.columns = [
        f"Step@{k}" if metric == "step_acc" else f"Agent@{k}"
        for metric, k in pivot.columns
    ]

    # Order columns: all Step@k first, then Agent@k, both sorted by k
    step_cols  = sorted([c for c in pivot.columns if c.startswith("Step")],
                        key=lambda c: int(c.split("@")[1]))
    agent_cols = sorted([c for c in pivot.columns if c.startswith("Agent")],
                        key=lambda c: int(c.split("@")[1]))
    pivot = pivot[step_cols + agent_cols]

    # Sort rows by canonical subset / strategy order
    def _sort_key(idx):
        subset   = idx[0] if isinstance(idx, tuple) else idx
        strategy = idx[1] if isinstance(idx, tuple) and len(idx) > 1 else ""
        s_rank   = SUBSET_ORDER.index(subset)    if subset   in SUBSET_ORDER   else 99
        b_rank   = STRATEGY_ORDER.index(strategy) if strategy in STRATEGY_ORDER else 99
        return (s_rank, b_rank)

    pivot = pivot.loc[sorted(pivot.index, key=_sort_key)]

    print("\n" + "=" * 70)
    print("Sweep Results")
    print("=" * 70)
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    if save_path:
        pivot.to_csv(save_path, sep="\t")
        print(f"\nSaved to {save_path}")

    return pivot


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # Single-dir mode
    parser.add_argument("--dir",       help="Directory with prediction files (single eval)")
    parser.add_argument("--ks",        type=int, nargs="+", default=DEFAULT_KS,
                        help="One or more k values, e.g. --ks 1 5 10")
    parser.add_argument("--n_bins",    type=int, default=5)
    parser.add_argument("--save",      help="Path to save results (.tsv)")
    parser.add_argument("--by_length", action="store_true")

    # Sweep mode
    parser.add_argument("--sweep",        action="store_true",
                        help="Run sweep over multiple configs")
    parser.add_argument("--configs",      nargs="+",
                        help="Explicit list of output directories to sweep")
    parser.add_argument("--base_output",  help="Auto-discover configs under this directory")

    # Combined predict + evaluate in one command
    parser.add_argument(
        "--predict_first",
        action="store_true",
        help=(
            "Run cli.predict on every discovered config before evaluating. "
            "The prediction method is inferred from the strategy folder name "
            "using FOLDER_TO_METHOD in cli.predict. "
            "Replaces the Phase 1 loop from sweep_eval.sh."
        ),
    )

    args = parser.parse_args()

    if args.sweep:
        if args.configs:
            configs = args.configs
        elif args.base_output:
            configs = _discover_configs(args.base_output)
        else:
            parser.error("--sweep requires either --configs or --base_output")

        if args.predict_first:
            _run_predict_phase(configs)

        sweep(
            configs=configs,
            ks=args.ks,
            save_path=args.save or (
                "sweep_by_length.tsv" if args.by_length else "sweep_results.tsv"
            ),
            by_length=args.by_length,
            n_bins=args.n_bins,
        )

    else:
        assert args.dir, "--dir is required when not using --sweep"
        for k in args.ks:
            if args.by_length:
                compute_acc_by_trajectory_length(
                    args.dir, k=k, n_bins=args.n_bins, save_path=args.save
                )
            else:
                compute_acc(args.dir, k=k)


if __name__ == "__main__":
    main()
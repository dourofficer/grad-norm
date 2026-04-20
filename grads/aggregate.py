"""
grads/aggregate.py — Aggregate sweep results across scoring families.

For each family, reads all per-scoring-config TSV files, ranks them by
step_acc, and writes top-K results for each (subset, direction) pair.
Then writes global top-K across all families per (subset, direction).

Output layout
-------------
outputs/grads/{model}/aggregated/{family}/{subset}_k{k}_{direction}.tsv
outputs/grads/{model}/aggregated/{subset}_top-configs_{direction}.tsv

Usage
-----
python -m grads.aggregate
python -m grads.aggregate --models qwen3-8b --top-family 20 --top-global 50
python -m grads.aggregate --stratify   # quota sampling across families
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_MODELS  = ["llama-3.1-8b", "qwen3-8b"]
KNOWN_SUBSETS = ["hand-crafted", "algorithm-generated"]
KNOWN_KS      = [1, 3, 5, 10]

FAMILIES = {
    "central":  ["mean_dist", "coord_median", "geom_median"],
    "svd":      ["proj", "recon"],
    "gradnorm": ["gradnorm"],
    "knn":      ["knn"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def belongs_to_family(scoring_name: str, family: str) -> bool:
    return any(scoring_name.startswith(prefix) for prefix in FAMILIES[family])


def load_all_results(
    metrics_root: Path,
    subset: str,
    k: int,
    direction: str,
) -> pd.DataFrame:
    """Collect step_acc / agent_acc across every scoring config into one DataFrame."""
    rows = []
    filename = f"{subset}_k{k}_{direction}.tsv"

    for scoring_dir in sorted(metrics_root.iterdir()):
        tsv = scoring_dir / filename
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t")
        df.insert(0, "scoring", scoring_dir.name)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df.insert(1, "family", df["scoring"].apply(assign_family))
    return df.sort_values("step_acc", ascending=False).reset_index(drop=True)


def assign_family(scoring_name: str) -> str:
    for family in FAMILIES:
        if belongs_to_family(scoring_name, family):
            return family
    return "unknown"


def top_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df.head(k).reset_index(drop=True)


def top_k_stratified(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Take top k//n_families rows per family, then sort the result by step_acc."""
    n = max(1, k // len(FAMILIES))
    frames = [
        df[df["family"] == family].head(n)
        for family in FAMILIES
        if (df["family"] == family).any()
    ]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("step_acc", ascending=False)
        .reset_index(drop=True)
    )


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    print(f"  Saved {path}  ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_model(
    model: str,
    base_dir: Path,
    subsets: list[str],
    ks: list[int],
    top_family: int,
    top_global: int,
    stratify: bool,
) -> None:
    metrics_root = base_dir / model / "metrics"
    agg_root     = base_dir / model / "aggregated"

    print(f"\n{'━'*55}")
    print(f"  Model: {model}  |  metrics: {metrics_root}")
    print(f"{'━'*55}")

    global_frames: dict[tuple, pd.DataFrame] = {}

    for subset in subsets:
        for k in ks:
            for direction in ["asc", "desc"]:
                key = (subset, k, direction)
                all_df = load_all_results(metrics_root, subset, k, direction)
                if all_df.empty:
                    print(f"  [skip] no results for {key}")
                    continue

                global_frames[key] = all_df

                # ── Per-family top-K ──────────────────────────────────────
                for family in FAMILIES:
                    family_df = all_df[all_df["family"] == family]
                    if family_df.empty:
                        continue
                    out_path = agg_root / family / f"{subset}_k{k}_{direction}.tsv"
                    save_tsv(top_k(family_df, top_family), out_path)

    # ── Global top-K per (subset, direction) ─────────────────────────────────
    select = top_k_stratified if stratify else top_k

    for subset in subsets:
        for direction in ["asc", "desc"]:
            df = global_frames.get((subset, ks[0], direction))
            if df is None or df.empty:
                continue
            out_path = agg_root / "global" / f"{subset}_top-configs_{direction}.tsv"
            save_tsv(select(df, top_global), out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate sweep results across scoring families.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models",   nargs="+", default=KNOWN_MODELS,  metavar="MODEL")
    parser.add_argument("--subsets",  nargs="+", default=KNOWN_SUBSETS, metavar="SUBSET")
    parser.add_argument("--ks",       nargs="+", type=int, default=KNOWN_KS, metavar="K")
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/grads"), metavar="DIR")
    parser.add_argument("--top-family", type=int, default=20, metavar="N",
                        help="Top-N configs to keep per family.")
    parser.add_argument("--top-global", type=int, default=60, metavar="N",
                        help="Top-N configs to keep in the global ranking.")
    parser.add_argument("--stratify", action="store_true",
                        help="Sample top_global // n_families configs per family, then sort.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for model in args.models:
        aggregate_model(
            model=model,
            base_dir=args.base_dir,
            subsets=args.subsets,
            ks=args.ks,
            top_family=args.top_family,
            top_global=args.top_global,
            stratify=args.stratify,
        )
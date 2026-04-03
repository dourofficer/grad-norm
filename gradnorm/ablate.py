"""
Ablation sweep: evaluate every config in every strategy across all trajectories.

python -m gradnorm.ablate --base_dir "outputs/gradnorm-v2" \
                          --models qwen3-8b-full llama-3.1-8b-full \
                          --subsets hand-crafted algorithm-generated \
                          --ks 1 3 5 10

python -m gradnorm.ablate --base_dir "outputs/gradnorm-losses" \
                          --models qwen3-8b-kl_uniform \
                          --subsets hand-crafted \
                          --ks 1 3 5 10
"""

import argparse
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .eval import (
    STRATEGIES, NORM_TYPES,
    build_strategies, load_trajectories,
    get_param_names_and_sizes, discover_n_layers,
    CompiledConfigs, evaluate_trajectories,
)

def sweep(
    model:     str,
    subset:    str,
    norm_type: str,
    k:         int,
    base_dir:  str,
    ascending: bool,
    verbose:   bool = False,
) -> pd.DataFrame:
    base         = Path(base_dir)
    results_dir  = base / model / subset
    order = "ascending" if ascending else "descending"
    trajectories = load_trajectories(results_dir)
    param_names, param_sizes = get_param_names_and_sizes(trajectories)
    n_layers     = discover_n_layers(param_names)
    if verbose:
        print(f"Discovered {n_layers} layers.")
        print(f"Raning scores in {order} order.")

    strategies = build_strategies(n_layers)

    out_dir  = base / model / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / f"{subset}_k{k}_{norm_type}_{order}.tsv"

    strategy_dfs = {}
    for name, config_dict in strategies.items():
        if verbose:
            print(f"Running strategy: {name} ({len(config_dict)} configs)...")
        cc = CompiledConfigs.compile(config_dict, param_names, param_sizes)
        df = evaluate_trajectories(trajectories, cc, norm_type, k, ascending)
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
    norm_type, k, model, subset, base_dir, ascending = args
    return args, sweep(
        model=model, subset=subset, norm_type=norm_type, k=k,
        base_dir=base_dir, ascending=ascending,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablation sweep: evaluate every config × strategy across all trajectories.",
    )
    p.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help=f"Root directory that contains <model>/<subset>/ result folders.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        # default=["qwen3-8b-full", "llama-3.1-8b-full"],
        help="Model directory names under base_dir/",
    )
    p.add_argument(
        "--subsets",
        nargs="+",
        default=["algorithm-generated", "hand-crafted"],
        help="Subset names (default: algorithm-generated hand-crafted).",
    )
    p.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        metavar="K",
        help="Top-K values to evaluate (default: 1 3 5 10).",
    )
    p.add_argument(
        "--norm-types",
        nargs="+",
        default=NORM_TYPES,
        choices=NORM_TYPES,
        metavar="NORM",
        help=f"Norm types to evaluate (default: {' '.join(NORM_TYPES)}).",
    )
    p.add_argument(
        "--orders",
        nargs="+",
        default=["ascending", "descending"],
        choices=["ascending", "descending"],
        metavar="ORDER",
        help="Ranking orders to evaluate (default: ascending descending).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Max parallel workers for ProcessPoolExecutor (default: 16).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-strategy progress within each sweep call.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    combos = list(product(
        args.norm_types, 
        args.ks, 
        args.models, 
        args.subsets, 
        args.orders
    ))
    combos = [
        (nt, k, m, s, args.base_dir, order == "ascending")
        for nt, k, m, s, order in combos
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_sweep_unpacked, combo): combo for combo in combos}
        results = {}
        for future in tqdm(as_completed(futures), total=len(combos)):
            combo, agg = future.result()
            results[combo] = agg
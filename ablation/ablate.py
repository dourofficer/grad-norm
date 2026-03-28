"""
Ablation sweep: evaluate every config in every strategy across all trajectories.

python -m ablation.ablate
"""

from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .core import (
    STRATEGIES, NORM_TYPES,
    build_strategies, load_trajectories,
    get_param_names_and_sizes, discover_n_layers,
    CompiledConfigs, evaluate_trajectories,
)


# ── main ──────────────────────────────────────────────────────────────────────

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

    out_dir  = Path(f"ablation/{model}/aggregated-results")
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / f"{subset}_k{k}_{norm_type}.tsv"

    strategy_dfs = {}
    for name, config_dict in strategies.items():
        if verbose:
            print(f"Running strategy: {name} ({len(config_dict)} configs)...")
        cc = CompiledConfigs.compile(config_dict, param_names, param_sizes)
        df = evaluate_trajectories(trajectories, cc, norm_type, k)
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
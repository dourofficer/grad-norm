"""
python -m ablation.score_dist --model qwen3-8b --subset hand-crafted
python -m ablation.score_dist --model qwen3-8b --subset algorithm-generated
python -m ablation.score_dist --model llama-3.1-8b --subset hand-crafted
python -m ablation.score_dist --model llama-3.1-8b --subset algorithm-generated
"""

import argparse
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
from .ablate import build_strategies, load_trajectories, get_param_names_and_sizes, discover_n_layers

# ── constants (never change) ──────────────────────────────────────────────────
KDE_POINTS = 400
STRATEGIES = ["layer", "mlp", "attn", "mlp_weights", "attn_weights"]
NORM_TYPES = ["l1_norm", "l2_norm"]


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot top-K config score distributions (normal vs mistake steps)."
    )
    p.add_argument("--model",        required=True,  help="Model name, e.g. qwen3-8b")
    p.add_argument("--subset",       required=True,  help="Subset name, e.g. hand-crafted")
    p.add_argument("--k-sweep",      type=int, default=1,
                   help="k used when running sweep() — determines which TSV to load (default: 1)")
    p.add_argument("--k-top",        type=int, default=3,
                   help="Number of top configs to display per strategy (default: 3)")
    p.add_argument("--ablation-dir", type=Path, default=Path("ablation"),
                   help="Root directory for ablation outputs (default: ablation/)")
    return p.parse_args()


# ── score helper ──────────────────────────────────────────────────────────────
def score_log_for_configs(
    log: dict,
    param_names: list[str],
    mask_matrix: np.ndarray,
    n_params_per_config: np.ndarray,
    norm_type: str,
) -> np.ndarray:
    """Score one log entry for each config in mask_matrix."""
    stats = log["statistics"]
    safe_counts = np.where(n_params_per_config > 0, n_params_per_config, np.nan)
    with np.errstate(invalid="ignore"):
        if norm_type == "l2_norm":
            vals = np.array([stats[p]["l2_norm_sq"] for p in param_names], dtype=np.float64)
            return np.sqrt(mask_matrix @ vals) / safe_counts
        else:
            vals = np.array([stats[p]["l1_norm"] for p in param_names], dtype=np.float64)
            return (mask_matrix @ vals) / safe_counts


# ── step 1: load aggregated TSV and pick top-K configs per strategy ───────────
def load_top_configs(
    agg_dir: Path,
    subset: str,
    k_sweep: int,
    norm_type: str,
    k_top: int,
) -> dict[str, list[str]]:
    """Returns {strategy_name: [config_name, ...]} with k_top best configs."""
    tsv_path = agg_dir / f"{subset}_k{k_sweep}_{norm_type}.tsv"
    df = pd.read_csv(tsv_path, sep="\t")

    top_configs: dict[str, list[str]] = {}
    for strat in STRATEGIES:
        sub = (
            df[df["strategy"] == strat]
            .sort_values("step_acc", ascending=False)
            .head(k_top)
        )
        top_configs[strat] = sub["config"].tolist()
    return top_configs


# ── step 2: collect per-step scores for selected configs ─────────────────────
def collect_scores(
    trajectories: list[dict],
    param_names: list[str],
    param_sizes: np.ndarray,
    all_strategies: dict[str, dict[str, str]],
    top_configs_by_norm: dict[str, dict[str, list[str]]],
) -> dict:
    """
    Returns nested dict:
      norm_type → strategy → config → {"mistake": [...], "normal": [...]}
    """
    score_store: dict = {
        nt: {
            strat: {cfg: {"mistake": [], "normal": []} for cfg in top_configs_by_norm[nt][strat]}
            for strat in STRATEGIES
        }
        for nt in NORM_TYPES
    }

    # Pre-compute mask matrices once — reused across all trajectories
    mask_cache: dict = {}
    for nt in NORM_TYPES:
        for strat in STRATEGIES:
            selected_cfgs = top_configs_by_norm[nt][strat]
            patterns      = [all_strategies[strat][c] for c in selected_cfgs]
            mask_matrix   = np.array(
                [[bool(re.search(pat, p)) for p in param_names] for pat in patterns],
                dtype=np.float64,
            )
            mask_cache[(nt, strat)] = (mask_matrix, selected_cfgs, mask_matrix @ param_sizes)

    for data in trajectories:
        mistake_step = int(data["metadata"]["mistake_step"])
        for log in data["logs"]:
            if not log.get("statistics"):
                continue
            kind = "mistake" if int(log["step_idx"]) == mistake_step else "normal"
            for nt in NORM_TYPES:
                for strat in STRATEGIES:
                    mask_matrix, cfg_names, n_params = mask_cache[(nt, strat)]
                    scores = score_log_for_configs(log, param_names, mask_matrix, n_params, nt)
                    for i, cfg in enumerate(cfg_names):
                        val = scores[i]
                        if not np.isnan(val):
                            score_store[nt][strat][cfg][kind].append(float(val))

    return score_store


# ── step 3: KDE plot helper ───────────────────────────────────────────────────
def plot_kde(ax: plt.Axes, values: list[float], label: str, color: str) -> None:
    if len(values) < 2:
        return
    arr = np.array(values)
    kde = gaussian_kde(arr)
    xs  = np.linspace(arr.min(), arr.max(), KDE_POINTS)
    ax.plot(xs, kde(xs), label=label, color=color, linewidth=1.2)
    ax.fill_between(xs, kde(xs), alpha=0.18, color=color)


# ── step 4: build the figure ──────────────────────────────────────────────────
def make_figure(
    score_store: dict,
    top_configs_by_norm: dict,
    model: str,
    subset: str,
    k_top: int,
) -> plt.Figure:
    n_strat    = len(STRATEGIES)
    total_rows = 2 * n_strat
    fig, axes  = plt.subplots(
        nrows=total_rows, ncols=k_top,
        figsize=(k_top * 3.4, total_rows * 2.4),
        squeeze=False,
    )

    half_offsets = {"l1_norm": 0, "l2_norm": n_strat}
    half_labels  = {"l1_norm": "Ranked by L1-norm", "l2_norm": "Ranked by L2-norm"}

    for nt in NORM_TYPES:
        row_offset = half_offsets[nt]

        for si, strat in enumerate(STRATEGIES):
            row      = row_offset + si
            cfg_list = top_configs_by_norm[nt][strat]

            for ci, cfg in enumerate(cfg_list):
                ax = axes[row, ci]
                plot_kde(ax, score_store[nt][strat][cfg]["normal"],  "normal",  "steelblue")
                plot_kde(ax, score_store[nt][strat][cfg]["mistake"], "mistake", "tomato")
                ax.set_title(f"#{ci+1} {cfg}", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.set_xlabel("score", fontsize=6)
                if ci == 0:
                    ax.set_ylabel(f"[{strat}]\ndensity", fontsize=6)

            for ci in range(len(cfg_list), k_top):
                axes[row, ci].set_visible(False)

        # Block label on the left of the middle strategy row
        mid_row = row_offset + n_strat // 2
        axes[mid_row, 0].annotate(
            half_labels[nt],
            xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=8, fontweight="bold", rotation=90,
            va="center", ha="center", color="dimgray",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.8)

    line_y = 1.0 - n_strat / total_rows
    fig.add_artist(plt.Line2D(
        [0.01, 0.99], [line_y, line_y],
        transform=fig.transFigure, color="gray", linewidth=1.2, linestyle="--",
    ))

    fig.suptitle(
        f"Grad-norm KDE: normal vs mistake  |  model={model}  subset={subset}\n"
        f"Top-{k_top} configs per strategy (ranked by step accuracy)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    results_dir = args.ablation_dir / args.model / args.subset
    agg_dir     = args.ablation_dir / args.model / "aggregated-results"

    print("Loading trajectories …")
    trajectories             = load_trajectories(results_dir)
    param_names, param_sizes = get_param_names_and_sizes(trajectories)
    n_layers                 = discover_n_layers(param_names)
    print(f"  {len(trajectories)} trajectories, {n_layers} layers discovered.")

    all_strategies = build_strategies(n_layers)

    print("Loading top configs from aggregated TSVs …")
    top_configs_by_norm = {
        nt: load_top_configs(agg_dir, args.subset, args.k_sweep, nt, args.k_top)
        for nt in NORM_TYPES
    }
    for nt in NORM_TYPES:
        for strat in STRATEGIES:
            print(f"  [{nt}] {strat}: {top_configs_by_norm[nt][strat]}")

    score_store = collect_scores(
        trajectories, param_names, param_sizes,
        all_strategies, top_configs_by_norm,
    )

    fig = make_figure(score_store, top_configs_by_norm, args.model, args.subset, args.k_top)

    out_path = agg_dir / f"{args.subset}_score-distribution.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
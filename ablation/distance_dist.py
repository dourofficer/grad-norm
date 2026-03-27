"""
python -m ablation.distance_dist --model qwen3-8b --subset hand-crafted
python -m ablation.distance_dist --model qwen3-8b --subset algorithm-generated
python -m ablation.distance_dist --model llama-3.1-8b --subset hand-crafted
python -m ablation.distance_dist --model llama-3.1-8b --subset algorithm-generated
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .ablate import build_strategies, load_trajectories, get_param_names_and_sizes, discover_n_layers

# ── constants ─────────────────────────────────────────────────────────────────
STRATEGIES = ["layer", "mlp", "attn", "mlp_weights", "attn_weights"]
NORM_TYPES = ["l1_norm", "l2_norm"]



# ── args ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot predicted_step - mistake_step distance distributions."
    )
    p.add_argument("--model",        required=True)
    p.add_argument("--subset",       required=True)
    p.add_argument("--k-sweep",      type=int, default=1)
    p.add_argument("--k-top",        type=int, default=3)
    p.add_argument("--ablation-dir", type=Path, default=Path("ablation"))
    return p.parse_args()


# ── TSV loader (identical to other scripts) ───────────────────────────────────
def load_top_configs(
    agg_dir: Path, subset: str, k_sweep: int, norm_type: str, k_top: int,
) -> dict[str, list[str]]:
    df = pd.read_csv(agg_dir / f"{subset}_k{k_sweep}_{norm_type}.tsv", sep="\t")
    return {
        strat: (
            df[df["strategy"] == strat]
            .sort_values("step_acc", ascending=False)
            .head(k_top)["config"]
            .tolist()
        )
        for strat in STRATEGIES
    }


# ── scoring helper (identical to other scripts) ───────────────────────────────
def score_log_for_configs(
    log: dict,
    param_names: list[str],
    mask_matrix: np.ndarray,
    n_params_per_config: np.ndarray,
    norm_type: str,
) -> np.ndarray:
    stats       = log["statistics"]
    safe_counts = np.where(n_params_per_config > 0, n_params_per_config, np.nan)
    with np.errstate(invalid="ignore"):
        if norm_type == "l2_norm":
            vals = np.array([stats[p]["l2_norm_sq"] for p in param_names], dtype=np.float64)
            return np.sqrt(mask_matrix @ vals) / safe_counts
        else:
            vals = np.array([stats[p]["l1_norm"] for p in param_names], dtype=np.float64)
            return (mask_matrix @ vals) / safe_counts


# ── data collection ───────────────────────────────────────────────────────────
def collect_distances(
    trajectories:        list[dict],
    param_names:         list[str],
    param_sizes:         np.ndarray,
    all_strategies:      dict[str, dict[str, str]],
    top_configs_by_norm: dict[str, dict[str, list[str]]],
) -> dict:
    """
    Returns:
      norm → strat → config → [distance, ...]
    where distance = predicted_step_idx − mistake_step_idx  (one value per trajectory).
    The predicted step is the step whose score is highest (argmax over valid logs).
    """
    distances: dict = {
        nt: {
            strat: {cfg: [] for cfg in top_configs_by_norm[nt][strat]}
            for strat in STRATEGIES
        }
        for nt in NORM_TYPES
    }

    # Pre-build mask matrices once
    mask_cache: dict = {}
    for nt in NORM_TYPES:
        for strat in STRATEGIES:
            cfgs     = top_configs_by_norm[nt][strat]
            patterns = [all_strategies[strat][c] for c in cfgs]
            mm       = np.array(
                [[bool(re.search(pat, p)) for p in param_names] for pat in patterns],
                dtype=np.float64,
            )
            mask_cache[(nt, strat)] = (mm, cfgs, mm @ param_sizes)

    for data in trajectories:
        mistake_step = int(data["metadata"]["mistake_step"])

        valid_logs = [log for log in data["logs"] if log.get("statistics")]
        if not valid_logs:
            continue

        step_indices = np.array([int(log["step_idx"]) for log in valid_logs])

        for nt in NORM_TYPES:
            for strat in STRATEGIES:
                mm, cfg_names, n_params = mask_cache[(nt, strat)]

                # score matrix: shape (n_logs, n_configs)
                score_matrix = np.stack([
                    score_log_for_configs(log, param_names, mm, n_params, nt)
                    for log in valid_logs
                ])  # (n_logs, n_configs)

                # argmin over steps for each config → predicted step index
                best_log_idx  = np.nanargmin(score_matrix, axis=0)   # (n_configs,)
                predicted_steps = step_indices[best_log_idx]          # (n_configs,)

                for i, cfg in enumerate(cfg_names):
                    dist = int(predicted_steps[i]) - mistake_step
                    distances[nt][strat][cfg].append(dist)

    return distances


# ── figure ─────────────────────────────────────────────────────────────────────
def make_figure(
    distances:           dict,
    top_configs_by_norm: dict,
    model:               str,
    subset:              str,
    k_top:               int,
) -> plt.Figure:
    n_strat = len(STRATEGIES)
    n_cols  = len(NORM_TYPES) * k_top
    n_rows  = n_strat

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols * 3, n_rows * 2.6),
        squeeze=False,
    )

    norm_labels = {"l1_norm": "L1", "l2_norm": "L2"}

    for ci, nt in enumerate(NORM_TYPES):
        for ki in range(k_top):
            col = ci * k_top + ki

            # Block header above the top row
            if ki == 0:
                axes[0, col].annotate(
                    f"── {norm_labels[nt]}-norm ranked ──",
                    xy=(k_top / 2 - 0.5, 1.18),
                    xycoords=("axes fraction", "axes fraction"),
                    fontsize=8, fontweight="bold", color="dimgray",
                    ha="center", va="bottom", annotation_clip=False,
                )

            for ri, strat in enumerate(STRATEGIES):
                ax  = axes[ri, col]
                cfg = (
                    top_configs_by_norm[nt][strat][ki]
                    if ki < len(top_configs_by_norm[nt][strat])
                    else None
                )

                if cfg is None:
                    ax.set_visible(False)
                    continue

                dists = np.array(distances[nt][strat][cfg])
                if len(dists) == 0:
                    ax.set_visible(False)
                    continue

                # ── stacked dot plot ──────────────────────────────────────
                # For each unique x value, stack dots upward at y=0,1,2,...
                from collections import Counter
                counts   = Counter(dists.tolist())
                xs_stack = []
                ys_stack = []
                cs_stack = []

                for x_val, count in sorted(counts.items()):
                    color = (
                        "gold"      if x_val == 0  else
                        "steelblue" if x_val <  0  else
                        "tomato"
                    )
                    for level in range(count):
                        xs_stack.append(x_val)
                        ys_stack.append(level)
                        cs_stack.append(color)

                ax.scatter(xs_stack, ys_stack, c=cs_stack, s=22, alpha=0.85,
                           edgecolors="none", zorder=3)

                # vertical reference at 0
                ax.axvline(0, color="black", linewidth=1.0, linestyle="--",
                           alpha=0.6, zorder=2)

                # annotation: pct exact (distance == 0)
                pct_exact = (dists == 0).mean() * 100
                ax.text(
                    0.97, 0.93, f"exact: {pct_exact:.0f}%",
                    transform=ax.transAxes,
                    fontsize=5.5, ha="right", va="top", color="dimgray",
                )

                ax.set_title(f"#{ki+1} {cfg}", fontsize=6.5)
                ax.set_yticks([])
                ax.tick_params(axis="x", labelsize=5)
                ax.set_xlabel("predicted − mistake (steps)", fontsize=6)
                if col == 0:
                    ax.set_ylabel(f"[{strat}]", fontsize=6)

    # vertical divider between the two norm blocks
    mid = k_top / n_cols
    fig.add_artist(plt.Line2D(
        [mid, mid], [0.02, 0.98],
        transform=fig.transFigure, color="gray", linewidth=1.2, linestyle="--",
    ))

    # legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gold",
               markersize=7, label="exact (distance = 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=7, label="predicted too early (< 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
               markersize=7, label="predicted too late (> 0)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        f"Prediction distance: predicted_step − mistake_step  "
        f"|  model={model}  subset={subset}\n"
        f"Each dot = one trajectory    "
        f"●gold = exact    ●blue = too early    ●red = too late",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    results_dir = args.ablation_dir / args.model / args.subset
    agg_dir     = args.ablation_dir / args.model / "aggregated-results"

    trajectories             = load_trajectories(results_dir)
    param_names, param_sizes = get_param_names_and_sizes(trajectories)
    all_strategies           = build_strategies(discover_n_layers(param_names))

    top_configs_by_norm = {
        nt: load_top_configs(agg_dir, args.subset, args.k_sweep, nt, args.k_top)
        for nt in NORM_TYPES
    }

    distances = collect_distances(
        trajectories, param_names, param_sizes,
        all_strategies, top_configs_by_norm,
    )

    fig      = make_figure(distances, top_configs_by_norm, args.model, args.subset, args.k_top)
    out_path = args.ablation_dir / args.model / f"{args.subset}_distance-chart.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
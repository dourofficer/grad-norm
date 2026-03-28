"""
Plot predicted_step − mistake_step distance distributions.

python -m ablation.distance_dist --model qwen3-8b --subset hand-crafted
python -m ablation.distance_dist --model qwen3-8b --subset algorithm-generated
python -m ablation.distance_dist --model llama-3.1-8b --subset hand-crafted
python -m ablation.distance_dist --model llama-3.1-8b --subset algorithm-generated
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .core import (
    STRATEGIES, NORM_TYPES,
    build_strategies, load_trajectories,
    get_param_names_and_sizes, discover_n_layers,
    CompiledConfigs, score_trajectory,
    load_top_configs, compile_top_configs,
)


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot predicted_step - mistake_step distance distributions.",
    )
    p.add_argument("--model",        required=True)
    p.add_argument("--subset",       required=True)
    p.add_argument("--k-sweep",      type=int, default=1)
    p.add_argument("--k-top",        type=int, default=3)
    p.add_argument("--ablation-dir", type=Path, default=Path("ablation"))
    return p.parse_args()


# ── data collection ───────────────────────────────────────────────────────────

def collect_distances(
    trajectories:     list[dict],
    compiled_by_norm: dict[str, dict[str, CompiledConfigs]],
) -> dict:
    """Compute prediction distance for each trajectory.

    Returns
    -------
    dict
        ``norm → strat → config → [distance, …]``
        where ``distance = predicted_step − mistake_step``.
        The predicted step is the one with the lowest score (argmin).
    """
    distances: dict = {
        nt: {
            strat: {cfg: [] for cfg in cc.names}
            for strat, cc in strat_ccs.items()
        }
        for nt, strat_ccs in compiled_by_norm.items()
    }

    for traj in trajectories:
        mistake_step = int(traj["metadata"]["mistake_step"])

        for nt, strat_ccs in compiled_by_norm.items():
            for strat, cc in strat_ccs.items():
                try:
                    score_matrix, step_indices = score_trajectory(traj, cc, nt)
                except ValueError:
                    continue

                # argmin over steps for each config → predicted step index
                best_log_idx    = np.nanargmin(score_matrix, axis=0)   # (C,)
                predicted_steps = step_indices[best_log_idx]           # (C,)

                for i, cfg in enumerate(cc.names):
                    distances[nt][strat][cfg].append(
                        int(predicted_steps[i]) - mistake_step
                    )

    return distances


# ── plotting ──────────────────────────────────────────────────────────────────

def make_figure(
    distances:        dict,
    compiled_by_norm: dict[str, dict[str, CompiledConfigs]],
    model:  str,
    subset: str,
    k_top:  int,
) -> plt.Figure:
    n_strat    = len(STRATEGIES)
    total_rows = 2 * n_strat
    fig, axes  = plt.subplots(
        nrows=total_rows, ncols=k_top,
        figsize=(k_top * 3.6, total_rows * 2.0),
        squeeze=False,
    )

    half_offsets = {"l1_norm": 0, "l2_norm": n_strat}
    half_labels  = {"l1_norm": "Ranked by L1-norm", "l2_norm": "Ranked by L2-norm"}

    for nt in NORM_TYPES:
        row_offset = half_offsets[nt]

        for si, strat in enumerate(STRATEGIES):
            row      = row_offset + si
            cfg_list = compiled_by_norm[nt][strat].names

            for ci, cfg in enumerate(cfg_list):
                ax   = axes[row, ci]
                vals = np.array(distances[nt][strat][cfg])
                if len(vals) == 0:
                    ax.set_visible(False)
                    continue

                colors = np.where(vals == 0, "gold",
                         np.where(vals < 0,  "steelblue", "tomato"))

                # Stack points with the same distance vertically
                y_pos = np.zeros(len(vals), dtype=np.float64)
                counts: dict[int, int] = {}
                for j, v in enumerate(vals):
                    rank = counts.get(int(v), 0)
                    y_pos[j] = rank
                    counts[int(v)] = rank + 1

                ax.scatter(vals, y_pos, c=colors, s=18, alpha=0.7, edgecolors="none")
                ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

                exact_pct = 100.0 * np.sum(vals == 0) / len(vals)
                ax.text(
                    0.97, 0.93, f"exact: {exact_pct:.0f}%",
                    transform=ax.transAxes, fontsize=7,
                    ha="right", va="top", color="dimgray",
                )
                ax.set_title(f"#{ci+1} {cfg}", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.set_xlabel("predicted − mistake (steps)", fontsize=6)
                ax.set_yticks([])
                if ci == 0:
                    ax.set_ylabel(f"[{strat}]", fontsize=6)

            for ci in range(len(cfg_list), k_top):
                axes[row, ci].set_visible(False)

        mid_row = row_offset + n_strat // 2
        axes[mid_row, 0].annotate(
            half_labels[nt],
            xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=8, fontweight="bold", rotation=90,
            va="center", ha="center", color="dimgray",
        )

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

    compiled_by_norm: dict[str, dict[str, CompiledConfigs]] = {}
    for nt in NORM_TYPES:
        top_names = load_top_configs(agg_dir, args.subset, args.k_sweep, nt, args.k_top)
        compiled_by_norm[nt] = compile_top_configs(
            top_names, all_strategies, param_names, param_sizes,
        )

    distances = collect_distances(trajectories, compiled_by_norm)
    fig       = make_figure(distances, compiled_by_norm, args.model, args.subset, args.k_top)

    out_path = args.ablation_dir / args.model / "aggregated-results" / f"fig_{args.subset}_distance-chart.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
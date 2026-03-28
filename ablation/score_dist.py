"""
Plot top-K config score distributions (normal vs mistake steps).

python -m ablation.score_dist --model qwen3-8b --subset hand-crafted
python -m ablation.score_dist --model qwen3-8b --subset algorithm-generated
python -m ablation.score_dist --model llama-3.1-8b --subset hand-crafted
python -m ablation.score_dist --model llama-3.1-8b --subset algorithm-generated
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from .core import (
    STRATEGIES, NORM_TYPES,
    build_strategies, load_trajectories,
    get_param_names_and_sizes, discover_n_layers,
    CompiledConfigs, score_step,
    load_top_configs, compile_top_configs,
)

KDE_POINTS = 400


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot top-K config score distributions (normal vs mistake steps).",
    )
    p.add_argument("--model",        required=True)
    p.add_argument("--subset",       required=True)
    p.add_argument("--k-sweep",      type=int, default=1)
    p.add_argument("--k-top",        type=int, default=3)
    p.add_argument("--ablation-dir", type=Path, default=Path("ablation"))
    return p.parse_args()


# ── data collection ───────────────────────────────────────────────────────────

def collect_scores(
    trajectories:        list[dict],
    compiled_by_norm:    dict[str, dict[str, CompiledConfigs]],
) -> dict:
    """Collect per-step scores partitioned by normal/mistake.

    Returns
    -------
    dict
        ``norm → strat → config → {"normal": [float, …], "mistake": [float, …]}``
    """
    score_store: dict = {
        nt: {
            strat: {
                cfg: {"normal": [], "mistake": []}
                for cfg in cc.names
            }
            for strat, cc in strat_ccs.items()
        }
        for nt, strat_ccs in compiled_by_norm.items()
    }

    for traj in trajectories:
        mistake_step = int(traj["metadata"]["mistake_step"])

        for log in traj["logs"]:
            if not log.get("statistics"):
                continue
            kind = "mistake" if int(log["step_idx"]) == mistake_step else "normal"

            for nt, strat_ccs in compiled_by_norm.items():
                for strat, cc in strat_ccs.items():
                    scores = score_step(log, cc, nt)
                    for i, cfg in enumerate(cc.names):
                        val = scores[i]
                        if not np.isnan(val):
                            score_store[nt][strat][cfg][kind].append(float(val))

    return score_store


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_kde(ax: plt.Axes, values: list[float], label: str, color: str) -> None:
    if len(values) < 2:
        return
    arr = np.array(values)
    kde = gaussian_kde(arr)
    xs  = np.linspace(arr.min(), arr.max(), KDE_POINTS)
    ax.plot(xs, kde(xs), label=label, color=color, linewidth=1.2)
    ax.fill_between(xs, kde(xs), alpha=0.18, color=color)


def make_figure(
    score_store:         dict,
    compiled_by_norm:    dict[str, dict[str, CompiledConfigs]],
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
            cfg_list = compiled_by_norm[nt][strat].names

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
    all_strategies           = build_strategies(n_layers)
    print(f"  {len(trajectories)} trajectories, {n_layers} layers discovered.")

    print("Loading top configs from aggregated TSVs …")
    compiled_by_norm: dict[str, dict[str, CompiledConfigs]] = {}
    for nt in NORM_TYPES:
        top_names = load_top_configs(agg_dir, args.subset, args.k_sweep, nt, args.k_top)
        compiled_by_norm[nt] = compile_top_configs(
            top_names, all_strategies, param_names, param_sizes,
        )
        for strat in STRATEGIES:
            print(f"  [{nt}] {strat}: {compiled_by_norm[nt][strat].names}")

    score_store = collect_scores(trajectories, compiled_by_norm)
    fig = make_figure(score_store, compiled_by_norm, args.model, args.subset, args.k_top)

    out_path = args.ablation_dir / args.model / "aggregated-results" / f"fig_{args.subset}_score-distribution.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
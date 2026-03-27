"""
python -m ablation.length_dist --model qwen3-8b --subset hand-crafted
python -m ablation.length_dist --model qwen3-8b --subset algorithm-generated
python -m ablation.length_dist --model llama-3.1-8b --subset hand-crafted
python -m ablation.length_dist --model llama-3.1-8b --subset algorithm-generated
"""

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .ablate import build_strategies, load_trajectories, get_param_names_and_sizes, discover_n_layers
from core.data import _serialize_turns, select_context
from scipy.stats import gaussian_kde

# ── constants ─────────────────────────────────────────────────────────────────
STRATEGIES = ["layer", "mlp", "attn", "mlp_weights", "attn_weights"]
NORM_TYPES = ["l1_norm", "l2_norm"]
BINS       = list(range(0, 8192 + 128, 128))  # [0, 128, 256, …, 8192]
KDE_POINTS = 400


# ── args ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True)
    p.add_argument("--subset",       required=True)
    p.add_argument("--k-sweep",      type=int, default=1)
    p.add_argument("--k-top",        type=int, default=3)
    p.add_argument("--ablation-dir", type=Path, default=Path("ablation"))
    return p.parse_args()


# ── length proxy ──────────────────────────────────────────────────────────────
def context_word_count(data: dict, step_idx: int) -> int:
    """
    Mirrors the actual GradNorm context build:
      select_context   → pick which history turns go into the user slot
      _serialize_turns → flatten them to plain text
    Word count is used as a cheap proxy for token count.
    """
    history     = data["steps"]
    ctx_indices = select_context(history, step_idx)
    ctx_text    = _serialize_turns(history, ctx_indices)
    return len(ctx_text.split())


def bin_length(n_words: int) -> int:
    """Snap a word count to the left edge of its 128-word bin (max 8192)."""
    return min((n_words // 128) * 128, 8192)


# ── scoring (same as plot_topk_distributions.py) ──────────────────────────────
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


# ── data collection ───────────────────────────────────────────────────────────
def collect_points(
    trajectories: list[dict],
    param_names:  list[str],
    param_sizes:  np.ndarray,
    all_strategies: dict[str, dict[str, str]],
    top_configs_by_norm: dict[str, dict[str, list[str]]],
) -> dict:
    """
    Returns:
      norm → strat → config → {"lengths": [...], "scores": [...], "is_mistake": [...]}
    One entry per (step, trajectory).
    """
    points: dict = {
        nt: {
            strat: {cfg: {"lengths": [], "scores": [], "is_mistake": []}
                    for cfg in top_configs_by_norm[nt][strat]}
            for strat in STRATEGIES
        }
        for nt in NORM_TYPES
    }

    # pre-compute mask matrices once
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

        for log in data["logs"]:
            if not log.get("statistics"):
                continue
            step_idx   = int(log["step_idx"])
            is_mistake = step_idx == mistake_step

            # word-count length of the context fed into this step's gradient pass
            binned = bin_length(context_word_count(data, step_idx))

            for nt in NORM_TYPES:
                for strat in STRATEGIES:
                    mm, cfg_names, n_params = mask_cache[(nt, strat)]
                    scores = score_log_for_configs(log, param_names, mm, n_params, nt)
                    for i, cfg in enumerate(cfg_names):
                        val = scores[i]
                        if not np.isnan(val):
                            points[nt][strat][cfg]["lengths"].append(binned)
                            points[nt][strat][cfg]["scores"].append(float(val))
                            points[nt][strat][cfg]["is_mistake"].append(is_mistake)

    return points

def collect_lengths(trajectories: list[dict]) -> dict[str, list[int]]:
    """Collect binned context lengths for all steps, keyed by normal/mistake."""
    lengths: dict[str, list[int]] = {"normal": [], "mistake": []}
    for data in trajectories:
        mistake_step = int(data["metadata"]["mistake_step"])
        for log in data["logs"]:
            if not log.get("statistics"):
                continue
            step_idx = int(log["step_idx"])
            kind     = "mistake" if step_idx == mistake_step else "normal"
            lengths[kind].append(bin_length(context_word_count(data, step_idx)))
    return lengths


def make_figure(
    points: dict,
    top_configs_by_norm: dict,
    all_lengths: dict[str, list[int]],   # ← new
    model: str,
    subset: str,
    k_top: int,
) -> plt.Figure:
    n_strat   = len(STRATEGIES)
    n_cols    = len(NORM_TYPES) * k_top
    n_rows    = n_strat + 1              # +1 for the KDE header row

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols * 3, n_rows * 3),
        squeeze=False,
    )

    norm_labels = {"l1_norm": "L1", "l2_norm": "L2"}
    dot_color   = "steelblue"
    star_color  = "tomato"

    # ── row 0: length KDE (same content in every column) ─────────────────────
    for col in range(n_cols):
        ax = axes[0, col]
        for kind, color in [("normal", dot_color), ("mistake", star_color)]:
            vals = np.array(all_lengths[kind])
            if len(vals) >= 2:
                kde = gaussian_kde(vals)
                xs  = np.linspace(vals.min(), vals.max(), KDE_POINTS)
                ax.plot(xs, kde(xs), color=color, linewidth=1.2,
                        label=kind if col == 0 else None)
                ax.fill_between(xs, kde(xs), alpha=0.18, color=color)
        ax.set_xlim(0, 8192)
        ax.set_xticks(range(0, 8192 + 1, 128 * 16))
        ax.tick_params(axis="x", rotation=45, labelsize=5)
        ax.tick_params(axis="y", labelsize=5)
        ax.set_xlabel("length (words)", fontsize=6)
        if col == 0:
            ax.set_ylabel("density", fontsize=6)
        ax.set_title("length distribution", fontsize=6.5, color="dimgray")

    # ── rows 1…n_strat: scatter dot charts ───────────────────────────────────
    for ci, nt in enumerate(NORM_TYPES):
        for ki in range(k_top):
            col = ci * k_top + ki

            if ki == 0:
                axes[1, col].annotate(
                    f"── {norm_labels[nt]}-norm ranked ──",
                    xy=(k_top / 2 - 0.5, 1.18), xycoords=("axes fraction", "axes fraction"),
                    fontsize=8, fontweight="bold", color="dimgray",
                    ha="center", va="bottom", annotation_clip=False,
                )

            for ri, strat in enumerate(STRATEGIES):
                ax  = axes[ri + 1, col]   # +1 to skip KDE row
                cfg = top_configs_by_norm[nt][strat][ki] \
                      if ki < len(top_configs_by_norm[nt][strat]) else None

                if cfg is None:
                    ax.set_visible(False)
                    continue

                d          = points[nt][strat][cfg]
                lengths    = np.array(d["lengths"])
                scores     = np.array(d["scores"])
                is_mistake = np.array(d["is_mistake"])

                ax.scatter(
                    lengths[~is_mistake], scores[~is_mistake],
                    s=8, alpha=0.25, color=dot_color,
                    label="normal" if (ri == 0 and col == 0) else None,
                )
                ax.scatter(
                    lengths[is_mistake], scores[is_mistake],
                    s=60, alpha=0.9, color=star_color, marker="*",
                    edgecolors="black", linewidths=0.5,
                    label="mistake" if (ri == 0 and col == 0) else None,
                )

                ax.set_title(f"#{ki+1} {cfg}", fontsize=6.5)
                ax.tick_params(labelsize=5)
                ax.set_xticks(range(0, 8192 + 1, 128 * 16))
                ax.tick_params(axis="x", rotation=45)
                ax.set_xlabel("length (words)", fontsize=6)
                if col == 0:
                    ax.set_ylabel(f"[{strat}]\nscore", fontsize=6)

    # vertical divider between the two norm blocks
    mid = k_top / n_cols
    fig.add_artist(plt.Line2D(
        [mid, mid], [0.02, 0.98],
        transform=fig.transFigure, color="gray", linewidth=1.2, linestyle="--",
    ))

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        f"Context length vs grad-norm score  |  model={model}  subset={subset}\n"
        f"★ = decisive error step    ·  = normal step",
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

    points = collect_points(
        trajectories, param_names, param_sizes,
        all_strategies, top_configs_by_norm,
    )
    all_lengths = collect_lengths(trajectories)

    fig      = make_figure(points, top_configs_by_norm, all_lengths,
                           args.model, args.subset, args.k_top)
    out_path = args.ablation_dir / args.model / f"{args.subset}_length-score-chart.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
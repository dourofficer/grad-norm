"""
main.py ― GradNorm evaluation runner for Who&When.

Usage
-----
    python main.py \\
        --dataset   /path/to/whoandwhen.json \\
        --model     meta-llama/Meta-Llama-3-8B-Instruct \\
        --layer     lm_head \\
        --strategy  standard \\
        --subset    handcrafted \\
        --output    results/llama_lmhead_handcrafted.json

All six (model × layer) GradNorm combinations in one sweep:
    for LAYER in lm_head out_proj final_layer; do
      for MODEL in meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen3-8B; do
        python main.py --dataset ... --model $MODEL --layer $LAYER --subset all \\
                       --output results/${MODEL##*/}_${LAYER}.json
      done
    done
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_dataset, build_context, custom_build_context
from gradnorm import score_trajectory
from metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run GradNorm anomaly scoring on the Who&When dataset."
    )
    p.add_argument(
        "--dataset", required=True,
        help="Path to the Who&When JSON file."
    )
    p.add_argument(
        "--model", required=True,
        help="HuggingFace model name or local path "
             "(e.g. meta-llama/Meta-Llama-3-8B-Instruct)."
    )
    p.add_argument(
        "--layer", choices=["lm_head", "out_proj", "final_layer"],
        default="lm_head",
        help="Which weight matrix W to differentiate against (default: lm_head)."
    )
    p.add_argument(
        "--strategy", choices=["standard", "split"],
        default="standard",
        help="Gradient computation strategy (default: standard)."
    )
    p.add_argument(
        "--subset", choices=["algo", "handcrafted", "all"],
        default="all",
        help="Dataset subset to evaluate (default: all)."
    )
    p.add_argument(
        "--output", default=None,
        help="Path to save per-trajectory results as JSON. "
             "Prints metrics to stdout if omitted."
    )
    p.add_argument(
        "--device", default=None,
        help="Torch device (default: 'cuda' if available, else 'cpu')."
    )
    p.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
        help="Model dtype (default: bfloat16)."
    )
    p.add_argument(
        "--ks", nargs="+", type=int, default=[1, 2, 3, 5, 10],
        help="k values for Acc@k (default: 1 2 3 5 10)."
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-step scores during evaluation."
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_metrics(metrics: dict, prefix: str = "") -> None:
    """Pretty-print a metrics dict to stdout."""
    step_keys  = sorted(k for k in metrics if k.startswith("step_acc"))
    agent_keys = sorted(k for k in metrics if k.startswith("agent_acc"))
    print(f"\n{prefix}")
    print(f"  {'Metric':<18} {'Value':>8}")
    print(f"  {'-'*28}")
    for key in step_keys + agent_keys:
        print(f"  {key:<18} {metrics[key]:>8.4f}")


def _split_by_subset(
    results: list[dict],
    trajectories: list,
) -> dict[str, list[dict]]:
    """Group results by subset label for per-subset reporting."""
    by_subset: dict[str, list[dict]] = {"algo": [], "handcrafted": [], "all": results}
    subset_map = {t.question_id: t.subset for t in trajectories}
    for res in results:
        s = subset_map.get(res["question_id"], "unknown")
        by_subset.setdefault(s, []).append(res)
    return by_subset


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = getattr(torch, args.dtype)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\nLoading tokeniser: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model ({args.dtype}) → {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype  = dtype,
        device_map   = device,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f}B parameters loaded.\n")

    # ── Load dataset ────────────────────────────────────────────────────────
    subset_filter = None if args.subset == "all" else args.subset
    trajectories  = load_dataset(args.dataset, subset=subset_filter)
    print(f"Loaded {len(trajectories)} trajectories  [subset={args.subset}]")

    # ── Score ───────────────────────────────────────────────────────────────
    # Swap context_builder to custom_build_context here if needed.
    context_builder = build_context

    all_results: list[dict] = []
    t0 = time.perf_counter()

    for i, traj in enumerate(trajectories):
        elapsed = time.perf_counter() - t0
        print(
            f"[{i+1:3d}/{len(trajectories)}]  {traj.question_id}"
            f"  steps={len(traj.history)}"
            f"  subset={traj.subset}"
            f"  elapsed={elapsed:.0f}s",
            flush=True,
        )

        try:
            result = score_trajectory(
                trajectory      = traj,
                model           = model,
                tokenizer       = tokenizer,
                layer_variant   = args.layer,
                strategy        = args.strategy,
                context_builder = context_builder,
                device          = device,
                verbose         = args.verbose,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue

        all_results.append(result)

    total_time = time.perf_counter() - t0
    print(f"\nScoring complete in {total_time:.1f}s  ({total_time/max(len(all_results),1):.1f}s/traj)")

    # ── Metrics ─────────────────────────────────────────────────────────────
    # Report per-subset and combined.
    by_subset = _split_by_subset(all_results, trajectories)

    all_metrics: dict[str, dict] = {}
    for subset_name, subset_results in by_subset.items():
        if not subset_results:
            continue
        m = compute_metrics(subset_results, ks=args.ks)
        all_metrics[subset_name] = m
        _print_metrics(
            m,
            prefix=(
                f"=== {subset_name.upper()}  |  "
                f"model={args.model}  layer={args.layer}  "
                f"strategy={args.strategy}  "
                f"n={len(subset_results)} ==="
            ),
        )

    # ── Save ────────────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args":        vars(args),
            "metrics":     all_metrics,
            "results":     all_results,
            "total_time_s": total_time,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()

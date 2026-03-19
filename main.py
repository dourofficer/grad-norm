"""
main.py ― GradNorm evaluation runner for Who&When.

Usage
-----
CUDA_VISIBLE_DEVICES=7 python main.py \
    --dataset   ww \
    --model     /data/hoang/resources/models/Qwen/Qwen3-8B \
    --layer     lm_head \
    --strategy  split \
    --subset    hand-crafted \
    --output    outputs/qwen_lmhead.json \
    --verbose

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
from tqdm import tqdm
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gradnorm.data import load_dataset, build_context, custom_build_context
from gradnorm.gradnorm import score_trajectory
from gradnorm.metrics import compute_metrics


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
        "--max_len", type=int,
        default=16000,
        help="Max context length"
    )
    p.add_argument(
        "--strategy", choices=["standard", "split"],
        default="standard",
        help="Gradient computation strategy (default: standard)."
    )
    p.add_argument(
        "--subset", choices=["algorithm-generated", "hand-crafted"],
        default="hand-crafted",
        help="Dataset subset to evaluate"
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
        "--ks", nargs="+", type=int, default=[1, 3, 5, 10],
        help="k values for Acc@k (default: 1 3 5 10)."
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = getattr(torch, args.dtype)

    # ── Load dataset ────────────────────────────────────────────────────────
    trajectories  = load_dataset(args.dataset, subset=args.subset)
    # trajectories = trajectories[1:20]
    print(f"Loaded {len(trajectories)} trajectories  [subset={args.subset}]")

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

    # ── Score ───────────────────────────────────────────────────────────────
    # Swap context_builder to custom_build_context here if needed.
    context_builder = build_context

    all_results: list[dict] = []
    t0 = time.perf_counter()

    # for i, traj in tqdm(enumerate(trajectories)):
    pbar = tqdm(trajectories)
    for traj in pbar:
        # pbar.set_description(traj.filename)
        pbar.set_postfix(traj=traj.filename, steps=len(traj.history), subset=traj.subset)
        elapsed = time.perf_counter() - t0
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
    all_metrics = compute_metrics(all_results, ks=args.ks)
    _print_metrics(
        all_metrics,
        prefix=(
            f"model={args.model} | layer={args.layer}  "
            f"strategy={args.strategy}  "
            f"n={len(all_results)} ==="
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

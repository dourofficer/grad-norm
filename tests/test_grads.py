"""
tests/test_grads.py — Verify that extract_gradient_hooked matches
                      extract_gradient_standard on real model inputs.

For each tested step the script compares the two gradient vectors
(after identical reduction) along four axes:
  1. Cosine similarity          (expect ≈ 1.0)
  2. Max absolute difference    (expect close to 0)
  3. Relative L2 difference     (expect close to 0)
  4. L1 / L2 norm ratio         (cross-check; should match)

Usage
-----
# Single parameter
python -m tests.test_grads \
    --model  "/data/hoang/resources/models/Qwen/Qwen3-4B" \
    --input  ww --subset hand-crafted \
    --target_params q/31 \
    --loss ntp \
    --traj_idx 0 --step_idx 1

# Multiple parameters
python -m tests.test_grads \
    --model  "/data/hoang/resources/models/Qwen/Qwen3-4B" \
    --input  ww --subset hand-crafted \
    --target_params v/5 gate/5 in_norm/5 \
    --traj_idx 0 --step_idx 1

# All parameters (slow — use a small model / few steps)
python -m tests.test_grads \
    --model  "/data/hoang/resources/models/Qwen/Qwen3-4B" \
    --input  ww --subset hand-crafted \
    --target_params all \
    --traj_idx 0 --step_idx 1
"""
from __future__ import annotations

import argparse
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gradnorm.data import load_dataset, iter_scoreable_steps
from grads.core import (
    extract_gradient_hooked,
    extract_gradient_standard,
    shorthand_to_param,
)
from grads.extract import build_context   # uses the same custom context builder
from gradnorm.core import LOSSES

# ─────────────────────────────────────────────────────────────────────────────
# Comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compare_pair(sh: str, g_std: torch.Tensor, g_hook: torch.Tensor) -> dict:
    """Compute numeric agreement metrics between two gradient vectors."""
    g_std  = g_std.float()
    g_hook = g_hook.float()

    cos_sim  = F.cosine_similarity(g_std.unsqueeze(0), g_hook.unsqueeze(0)).item()
    max_diff = (g_std - g_hook).abs().max().item()
    rel_diff = (g_std - g_hook).norm() / (g_std.norm() + 1e-12)
    l1_std,  l1_hook  = g_std.norm(p=1).item(), g_hook.norm(p=1).item()
    l2_std,  l2_hook  = g_std.norm(p=2).item(), g_hook.norm(p=2).item()

    return dict(
        shorthand = sh,
        shape     = tuple(g_std.shape),
        cos_sim   = cos_sim,
        max_diff  = max_diff,
        rel_diff  = rel_diff.item(),
        l1_std    = l1_std,
        l1_hook   = l1_hook,
        l2_std    = l2_std,
        l2_hook   = l2_hook,
    )


def print_comparison(metrics: dict) -> bool:
    """Pretty-print metrics; return True if the pair passes basic sanity checks."""
    ok = metrics["cos_sim"] > 0.999 and metrics["rel_diff"] < 1e-2
    status = "✓ PASS" if ok else "✗ FAIL"
    print(
        f"  [{status}] {metrics['shorthand']}  shape={metrics['shape']}\n"
        f"         cos_sim={metrics['cos_sim']:.6f}  "
        f"max_diff={metrics['max_diff']:.3e}  "
        f"rel_diff={metrics['rel_diff']:.3e}\n"
        f"         L1  std={metrics['l1_std']:.4f}  hook={metrics['l1_hook']:.4f}  "
        f"L2  std={metrics['l2_std']:.4f}  hook={metrics['l2_hook']:.4f}"
    )
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main test
# ─────────────────────────────────────────────────────────────────────────────

def test_correctness(
    model,
    tokenizer,
    traj,
    step_idx:      int,
    target_params: list[str] | Literal["all"],
    loss_func:     Callable,
    device:        str,
    max_tokens:    int = 8192,
) -> bool:
    """Run hooked and standard extraction on one step; compare and report.

    Returns True if all parameters pass, False otherwise.
    """
    # breakpoint()
    encoded = build_context(traj.history, step_idx, tokenizer, max_tokens=max_tokens)
    input_ids      = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    ctx_len = encoded["ctx_len"]

    print(f"\n── Step {step_idx}  seq_len={input_ids.shape[1]}  ctx_len={ctx_len} ──")

    # Resolve shorthands → full names for standard pass
    if target_params == "all":
        full_params: list[str] | str = "all"
    else:
        full_params = [shorthand_to_param(sh) for sh in target_params]

    print("  Running standard backward …")
    grads_std = extract_gradient_standard(
        model, input_ids, attention_mask, ctx_len, full_params, loss_func,
    )

    print("  Running hooked backward …")
    grads_hook = extract_gradient_hooked(
        model, input_ids, attention_mask, ctx_len, full_params, loss_func,
    )

    # Both dicts are keyed by shorthand
    keys = sorted(set(grads_std) | set(grads_hook))
    all_pass = True
    for sh in keys:
        if sh not in grads_std:
            print(f"  [WARN] '{sh}' missing from standard output.")
            all_pass = False
            continue
        if sh not in grads_hook:
            print(f"  [WARN] '{sh}' missing from hooked output.")
            all_pass = False
            continue
        m = _compare_pair(sh, grads_std[sh], grads_hook[sh])
        if not print_comparison(m):
            all_pass = False

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare hooked vs standard gradient extraction."
    )
    p.add_argument("--model",         required=True)
    p.add_argument("--input",         required=True)
    p.add_argument("--subset",        default=None)
    p.add_argument(
        "--target_params", required=True, nargs="+",
        help="Shorthands like 'v/5' 'gate/5', or 'all'.",
    )
    p.add_argument("--traj_idx",   type=int, default=0,
                   help="Which trajectory to test (default: 0).")
    p.add_argument("--step_idx",   type=int, default=None,
                   help="Which step to test. Default: first scoreable step.")
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--device",     default=None)
    p.add_argument("--dtype",      choices=["float32", "bfloat16", "float16"],
                   default="bfloat16")
    p.add_argument("--loss",       choices=LOSSES.keys(), default="ntp")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    loss_func = LOSSES[args.loss]

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model → {device} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map={"": device},
    )
    model.eval()

    # ── Load trajectory ───────────────────────────────────────────────────────
    from pathlib import Path
    input_path = Path(args.input)
    if args.subset:
        base_path, subset = str(input_path), args.subset
    else:
        base_path, subset = str(input_path.parent), input_path.name

    trajectories = load_dataset(base_path, subset=subset)
    traj = trajectories[args.traj_idx]
    print(f"\nTrajectory: {traj.filename}  ({len(traj.history)} steps)")

    scoreable = iter_scoreable_steps(traj)
    step_idx  = args.step_idx if args.step_idx is not None else scoreable[0]
    print(f"Testing step_idx={step_idx}  (scoreable steps: {scoreable})")

    # ── Resolve target_params ─────────────────────────────────────────────────
    if len(args.target_params) == 1 and args.target_params[0] == "all":
        target_params: list[str] | str = "all"
    else:
        target_params = args.target_params

    # ── Run test ──────────────────────────────────────────────────────────────
    passed = test_correctness(
        model, tokenizer, traj, step_idx,
        target_params, loss_func, device, args.max_tokens,
    )

    print(f"\n{'All checks passed ✓' if passed else 'Some checks FAILED ✗'}")


if __name__ == "__main__":
    main()
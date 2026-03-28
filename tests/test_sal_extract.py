"""
test_sal_extract.py — Verify that extract_gradient produces correct gradients.

Compares the hooked extraction (with gradient checkpointing + hook that
captures one param and clears the rest) against a plain backward pass
(no hooks, no checkpointing) where we just read param.grad directly.

Usage:
python -m tests.test_sal_extract \
    --model "/data/hoang/resources/models/Qwen/Qwen3-4B" \
    --input ww --subset hand-crafted \
    --target_param model.layers.5.self_attn.v_proj.weight

The test checks:
  1. Element-wise max absolute difference
  2. Relative difference (normalised by norm)
  3. L1/L2 norm agreement (cross-check with gradnorm_hooked_all)
  4. Cosine similarity (should be ~1.0)
"""
from __future__ import annotations

import argparse
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.data import build_context, load_dataset, iter_scoreable_steps
from core.gradnorm import _ntp_loss
from cli.sal_extract import extract_gradient


# ─────────────────────────────────────────────────────────────────────────────
# Standard (no-hook) gradient extraction — the ground truth
# ─────────────────────────────────────────────────────────────────────────────

def extract_gradient_standard(
    model,
    input_ids,
    attention_mask,
    ctx_len: int,
    target_param: str,
) -> torch.Tensor:
    """Plain forward+backward, then read .grad from the target parameter.

    No gradient checkpointing, no hooks. This is the simplest possible
    implementation and serves as ground truth.
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    # Make sure all params accumulate gradients
    for p in model.parameters():
        p.requires_grad_(True)

    logits = model(input_ids, attention_mask, use_cache=False).logits
    loss = _ntp_loss(logits, input_ids, ctx_len)
    loss.backward()

    # Read the gradient
    for name, p in model.named_parameters():
        if name == target_param:
            if p.grad is None:
                raise RuntimeError(f"No gradient computed for {target_param}")
            grad = p.grad.float().flatten().clone().cpu()
            break
    else:
        raise ValueError(f"Parameter '{target_param}' not found.")

    # Cleanup
    model.zero_grad(set_to_none=True)

    return grad


# ─────────────────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_gradients(
    grad_standard: torch.Tensor,
    grad_hooked:   torch.Tensor,
    label: str = "",
):
    """Print detailed comparison between two gradient vectors."""
    # Upcast hooked from float16 to float32 for comparison
    grad_hooked_f32 = grad_hooked.float()

    diff = (grad_standard - grad_hooked_f32).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    norm_std = grad_standard.norm().item()
    norm_hook = grad_hooked_f32.norm().item()
    rel_diff = (grad_standard - grad_hooked_f32).norm().item() / max(norm_std, 1e-12)

    cosine = F.cosine_similarity(
        grad_standard.unsqueeze(0),
        grad_hooked_f32.unsqueeze(0),
    ).item()

    l1_std = grad_standard.abs().sum().item()
    l1_hook = grad_hooked_f32.abs().sum().item()
    l1_rel = abs(l1_std - l1_hook) / max(l1_std, 1e-12)

    print(f"\n{'═' * 60}")
    print(f"  Gradient Comparison {label}")
    print(f"{'═' * 60}")
    print(f"  Shape:              {tuple(grad_standard.shape)}")
    print(f"  ‖grad‖₂ (standard): {norm_std:.6e}")
    print(f"  ‖grad‖₂ (hooked):   {norm_hook:.6e}")
    print(f"  ‖grad‖₁ (standard): {l1_std:.6e}")
    print(f"  ‖grad‖₁ (hooked):   {l1_hook:.6e}")
    print(f"  ─────────────────────────────")
    print(f"  Max |diff|:         {max_abs_diff:.6e}")
    print(f"  Mean |diff|:        {mean_abs_diff:.6e}")
    print(f"  Relative L2 diff:   {rel_diff:.6e}")
    print(f"  L1 relative diff:   {l1_rel:.6e}")
    print(f"  Cosine similarity:  {cosine:.8f}")
    print(f"{'═' * 60}")

    # Verdict
    # float16 has ~3.3 decimal digits of precision, so relative error
    # up to ~1e-3 is expected from the float32→float16 cast alone.
    if cosine > 0.999 and rel_diff < 0.01:
        print(f"  ✅ PASS — gradients match (differences from float16 quantisation)")
    elif cosine > 0.99:
        print(f"  ⚠️  WARN — cosine high but relative diff notable: {rel_diff:.4e}")
    else:
        print(f"  ❌ FAIL — gradients do NOT match!")

    return {
        "max_abs_diff": max_abs_diff,
        "rel_diff":     rel_diff,
        "cosine":       cosine,
        "l1_rel":       l1_rel,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Also test random projection
# ─────────────────────────────────────────────────────────────────────────────

def test_random_projection(
    model, input_ids, attention_mask, ctx_len,
    target_param, grad_standard,
    proj_dim=256, seed=42,
):
    """Verify that the projected gradient = R^T @ full_gradient."""
    d = grad_standard.shape[0]
    rng = torch.Generator().manual_seed(seed)
    proj_matrix = torch.randn(d, proj_dim, generator=rng, dtype=torch.float32)
    proj_matrix /= math.sqrt(proj_dim)

    # Expected: project the standard gradient
    expected = (grad_standard @ proj_matrix).half()

    # Actual: extract with projection
    actual = extract_gradient(
        model, input_ids, attention_mask, ctx_len,
        target_param, proj_matrix,
    )

    diff = (expected.float() - actual.float()).abs()
    max_diff = diff.max().item()
    cosine = F.cosine_similarity(
        expected.float().unsqueeze(0),
        actual.float().unsqueeze(0),
    ).item()

    print(f"\n── Random Projection Test (d={d} → {proj_dim}) ──")
    print(f"  Max |diff|:        {max_diff:.6e}")
    print(f"  Cosine similarity: {cosine:.8f}")

    if cosine > 0.999:
        print(f"  ✅ PASS")
    else:
        print(f"  ❌ FAIL")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Test sal_extract gradient correctness.")
    p.add_argument("--model",        required=True)
    p.add_argument("--input",        required=True)
    p.add_argument("--subset",       default=None)
    p.add_argument("--target_param", required=True)
    p.add_argument("--traj_idx",     type=int, default=0,
                   help="Which trajectory to test on.")
    p.add_argument("--step_idx",     type=int, default=None,
                   help="Which step to test. None = first scoreable step.")
    p.add_argument("--max_tokens",   type=int, default=4096)
    p.add_argument("--device",       default=None)
    p.add_argument("--dtype",        default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--test_proj",    action="store_true",
                   help="Also test random projection.")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]

    # ── Load model ──────────────────────────────────────────────────
    print(f"Loading model: {args.model} → {device} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map={"": device},
    )
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

    # Verify target param
    for name, p in model.named_parameters():
        if name == args.target_param:
            print(f"  Target: {name}  shape={tuple(p.shape)}  "
                  f"d={p.numel():,}")
            break
    else:
        print(f"ERROR: '{args.target_param}' not found.")
        print("Available v_proj params:")
        for name, p in model.named_parameters():
            if "v_proj" in name:
                print(f"  {name}  {tuple(p.shape)}")
        return

    # ── Load data ───────────────────────────────────────────────────
    from pathlib import Path
    input_path = Path(args.input)
    if args.subset:
        base_path, subset = str(input_path), args.subset
    else:
        base_path, subset = str(input_path.parent), input_path.name

    trajectories = load_dataset(base_path, subset=subset)
    traj = trajectories[args.traj_idx]
    print(f"\n  Trajectory: {traj.filename} ({len(traj.history)} steps)")

    # Pick step
    scoreable = iter_scoreable_steps(traj)
    if args.step_idx is not None:
        step_idx = args.step_idx
    else:
        step_idx = scoreable[0]
    print(f"  Testing step_idx={step_idx}")

    # ── Tokenise ────────────────────────────────────────────────────
    encoded = build_context(
        traj.history, step_idx, tokenizer, max_tokens=args.max_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    ctx_len = encoded["ctx_len"]
    print(f"  seq_len={input_ids.shape[1]}, ctx_len={ctx_len}")

    # ── Method 1: Standard backward (ground truth) ──────────────────
    print("\n── Standard backward (no hooks, no checkpointing) ──")
    grad_standard = extract_gradient_standard(
        model, input_ids, attention_mask, ctx_len, args.target_param,
    )

    # ── Method 2: Hooked extraction (sal_extract) ───────────────────
    print("\n── Hooked extraction (sal_extract.extract_gradient) ──")
    grad_hooked = extract_gradient(
        model, input_ids, attention_mask, ctx_len, args.target_param,
    )

    # ── Compare ─────────────────────────────────────────────────────
    compare_gradients(grad_standard, grad_hooked, label=f"(step {step_idx})")

    # ── Optional: test random projection ────────────────────────────
    if args.test_proj:
        test_random_projection(
            model, input_ids, attention_mask, ctx_len,
            args.target_param, grad_standard,
        )

if __name__ == "__main__":
    main()
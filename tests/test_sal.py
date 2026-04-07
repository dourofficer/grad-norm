"""
tests/test_gradient.py — Correctness test for sal.gradient.extract_gradient.

Compares the hooked multi-param extraction (gradient checkpointing + hooks
that capture target grads and zero the rest) against a plain backward pass
where we just read .grad directly after loss.backward().

The two should agree up to float16 quantisation error (relative diff < 1 %,
cosine similarity > 0.999).

Usage
-----
# Single target param (shorthand):
python -m tests.test_sal \
    --model "/data/hoang/resources/models/Qwen/Qwen3-8B" \
    --target_params "v/2" \
    --seq_len 256 --ctx_len 64

# Multiple params + custom device:
python -m tests.test_sal \
    --model "/data/hoang/resources/models/Qwen/Qwen3-8B" \
    --target_params "v/35" "q/35" \
    --seq_len 256 --ctx_len 64 --dtype bfloat16
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sal.gradient import expand_param_name, extract_gradient
from gradnorm.losses import _ntp_loss


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth: plain backward, no hooks, no checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def standard_backward(model, input_ids, attention_mask, ctx_len: int,
                       target_params: list[str]) -> dict[str, torch.Tensor]:
    """Plain forward+backward; read .grad directly. Returns float32 CPU tensors."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    logits = model(input_ids, attention_mask, use_cache=False).logits
    _ntp_loss(logits, input_ids, ctx_len).backward()

    result = {}
    named  = dict(model.named_parameters())
    for full in target_params:
        p = named[full]
        if p.grad is None:
            raise RuntimeError(f"No gradient for '{full}'")
        result[full] = p.grad.float().flatten().clone().cpu()

    model.zero_grad(set_to_none=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Comparison helper
# ─────────────────────────────────────────────────────────────────────────────

def compare(ref: torch.Tensor, got: torch.Tensor, label: str) -> bool:
    """Print stats and return True if the test passes."""
    ref_f = ref.float()
    got_f = got.float()

    cosine   = F.cosine_similarity(ref_f.unsqueeze(0), got_f.unsqueeze(0)).item()
    rel_diff = (ref_f - got_f).norm().item() / max(ref_f.norm().item(), 1e-12)

    print(f"\n  [{label}]")
    print(f"    shape      : {tuple(ref_f.shape)}")
    print(f"    ‖ref‖₂     : {ref_f.norm().item():.6e}")
    print(f"    ‖got‖₂     : {got_f.norm().item():.6e}")
    print(f"    max |diff| : {(ref_f - got_f).abs().max().item():.6e}")
    print(f"    rel diff   : {rel_diff:.6e}")
    print(f"    cosine     : {cosine:.8f}")

    passed = cosine > 0.999 and rel_diff < 0.01
    print(f"    {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test sal.gradient.extract_gradient correctness.")
    p.add_argument("--model",         required=True,
                   help="HF model name or local path.")
    p.add_argument("--target_params", required=True, nargs="+",
                   help="Shorthands (e.g. 'v/35') or full dotted param names.")
    p.add_argument("--seq_len",  type=int, default=128,
                   help="Synthetic sequence length (tokens).")
    p.add_argument("--ctx_len",  type=int, default=32,
                   help="Number of context tokens (NTP loss ignores these).")
    p.add_argument("--device",   default=None)
    p.add_argument("--dtype",    choices=["float32", "bfloat16", "float16"],
                   default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()

    device      = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {"float32": torch.float32,
                   "bfloat16": torch.bfloat16,
                   "float16": torch.float16}[args.dtype]

    # ── Resolve shorthands ───────────────────────────────────────────────
    try:
        full_params = [expand_param_name(sh) for sh in args.target_params]
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model: {args.model} → {device} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map={"": device},
    )
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

    # Verify all target params exist
    named = dict(model.named_parameters())
    for sh, full in zip(args.target_params, full_params):
        if full not in named:
            print(f"ERROR: '{full}' not found in model.", file=sys.stderr); sys.exit(1)
        print(f"  {sh!r:12s} → {full}  shape={tuple(named[full].shape)}")

    # ── Build synthetic input ────────────────────────────────────────────
    vocab_size = model.config.vocab_size
    input_ids  = torch.randint(0, vocab_size, (1, args.seq_len), device=device)
    ctx_len    = args.ctx_len
    print(f"\nSynthetic input: seq_len={args.seq_len}, ctx_len={ctx_len}, vocab={vocab_size}")

    # ── Reference: plain backward ────────────────────────────────────────
    print("\n── Standard backward (ground truth) ──")
    ref_grads = standard_backward(model, input_ids, None, ctx_len, full_params)

    # ── Under test: hooked extraction ────────────────────────────────────
    print("\n── Hooked extraction (sal.gradient.extract_gradient) ──")
    got_grads = extract_gradient(model, input_ids, None, ctx_len, full_params)

    # ── Compare ──────────────────────────────────────────────────────────
    print("\n── Results ──")
    all_pass = True
    for sh, full in zip(args.target_params, full_params):
        if full not in got_grads:
            print(f"  ❌ FAIL [{sh}]: gradient not captured"); all_pass = False; continue
        passed   = compare(ref_grads[full], got_grads[full], sh)
        all_pass = all_pass and passed

    print(f"\n{'✅ ALL PASS' if all_pass else '❌ SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
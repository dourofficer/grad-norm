"""
sal_extract.py — Phase 1 of SAL-Step: extract and save raw gradient vectors.

For each scoreable step in each trajectory, computes the NTP-loss gradient
w.r.t. a single target weight parameter (e.g. model.layers.35.self_attn.v_proj.weight)
and saves the flattened gradient vector in float16.

Output: one .pt file per trajectory under --output, containing:
    {
        "metadata": { ... },          # same as gradnorm output
        "gradients": {                # step_idx -> flattened gradient (float16)
            1: Tensor(d,),
            3: Tensor(d,),
            ...
        }
    }

Usage:
python -m cli.sal_extract \
    --model "/data/hoang/resources/models/Qwen/Qwen3-8B" \
    --input ww/hand-crafted \
    --output sal_outputs/grads/qwen3-8b/hand-crafted \
    --target_param model.layers.35.self_attn.v_proj.weight \
    --max_tokens 8192 \
    --start_idx 4 --end_idx 5
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from core.data import (
    Trajectory,
    # build_context,
    _serialize_turns,
    iter_scoreable_steps,
    load_dataset,
)
from core.gradnorm import _ntp_loss

# ─────────────────────────────────────────────────────────────────────────────
# Temporary custom build_context
# ─────────────────────────────────────────────────────────────────────────────

from utils.graph import get_dependency_dict, derive_llm_inputs
def select_context(history: list[dict], step_idx: int) -> list[int]:
    is_handcrafted = any([m.get("role").startswith("Orchestrator") for m in history])
    if is_handcrafted:
        deps = get_dependency_dict(derive_llm_inputs(history))
        return deps[step_idx]
    else:
        return list(range(step_idx))

def build_context(
    history:    list[dict],
    step_idx:   int,
    tokenizer:  None,
    max_tokens: int | None = None,
) -> dict[str, Any]:

    ctx_indices  = select_context(history, step_idx)
    # assert ctx_indices == list(range(step_idx)), "taking full context, no graph"
    step_content = history[step_idx].get("content", "").strip()
    step_content = _serialize_turns(history, [step_idx])
    assistant_msg = {"role": "assistant", "content": step_content}
 
    def _apply(indices: list[int]) -> tuple:
        """Tokenise [user_msg, assistant_msg] and the user-only prefix."""
        user_msg = {"role": "user", "content": _serialize_turns(history, indices)}
        full_ids = tokenizer.apply_chat_template(
            [user_msg, assistant_msg],
            tokenize              = True,
            add_generation_prompt = False,
            return_tensors        = "pt",
        )
        prefix_ids = tokenizer.apply_chat_template(
            [user_msg],
            tokenize              = True,
            add_generation_prompt = True,
            return_tensors        = "pt",
        )
        return full_ids, prefix_ids
 
    full_ids, prefix_ids = _apply(ctx_indices)
 
    # ── Truncate context if full sequence exceeds max_tokens ─────────────
    # Drop the oldest context turns one by one until the total fits.
    # The step content is always preserved; only ctx_indices shrinks.
    if max_tokens is not None:
        while (
            full_ids["input_ids"].shape[1] > max_tokens
            and len(ctx_indices) > 0
        ):
            ctx_indices = ctx_indices[1:]   # drop oldest turn
            full_ids, prefix_ids = _apply(ctx_indices)

        if full_ids["input_ids"].shape[1] > max_tokens:
            step_len = full_ids["input_ids"].shape[1] - prefix_ids["input_ids"].shape[1]
            full_ids["input_ids"] = full_ids["input_ids"][:, -max_tokens:]
            ctx_len = max(0, max_tokens - step_len)
            return {"input_ids": full_ids["input_ids"], "ctx_len": ctx_len}
 
    ctx_len = prefix_ids["input_ids"].shape[1]
 
    return {"input_ids": full_ids["input_ids"], "ctx_len": ctx_len}

# ─────────────────────────────────────────────────────────────────────────────
# Gradient extraction for a single parameter
# ─────────────────────────────────────────────────────────────────────────────

def extract_gradient(
    model:        PreTrainedModel,
    input_ids:    Tensor,
    attention_mask: Tensor | None,
    ctx_len:      int,
    target_param: str,
    proj_matrix:  Tensor | None = None,
) -> Tensor:
    """Run forward+backward and return the flattened gradient for target_param.

    Uses gradient checkpointing + a hook to capture only the target gradient,
    immediately clearing all others to minimise VRAM.

    Parameters
    ----------
    model         : HuggingFace causal LM.
    input_ids     : (1, seq_len) token ids.
    attention_mask: (1, seq_len) or None.
    ctx_len       : number of context tokens (NTP loss masks these).
    target_param  : full dotted parameter name, e.g.
                    "model.layers.35.self_attn.v_proj.weight"
    proj_matrix   : optional (d, proj_dim) random projection matrix.
                    If provided, returns projected gradient of shape (proj_dim,).

    Returns
    -------
    Tensor of shape (d,) or (proj_dim,) in float16 on CPU.
    """
    was_training = model.training
    model.train()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # ── Locate the target parameter ──────────────────────────────────
    target_p = None
    for name, p in model.named_parameters():
        if name == target_param:
            target_p = p
            break
    if target_p is None:
        raise ValueError(f"Parameter '{target_param}' not found in model.")

    # ── Hook: capture target grad, clear everything else ─────────────
    captured_grad = {}
    handles = []
    hooked_params = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        hooked_params.add(p)

        if name == target_param:
            def make_capture_hook(store: dict):
                def hook(param):
                    if param.grad is not None:
                        with torch.no_grad():
                            g = param.grad.float().flatten()
                            if proj_matrix is not None:
                                g = g @ proj_matrix
                            store["grad"] = g.half().cpu()
                        param.grad = None
                return hook
            h = p.register_post_accumulate_grad_hook(make_capture_hook(captured_grad))
        else:
            def clear_hook(param):
                param.grad = None
            h = p.register_post_accumulate_grad_hook(clear_hook)
        handles.append(h)

    # ── Forward + backward ───────────────────────────────────────────
    model.zero_grad(set_to_none=True)

    logits = model(input_ids, attention_mask, use_cache=False).logits
    loss = _ntp_loss(logits, input_ids, ctx_len)
    loss.backward()

    # ── Cleanup ──────────────────────────────────────────────────────
    for h in handles:
        h.remove()

    model.gradient_checkpointing_disable()
    if not was_training:
        model.eval()

    model.zero_grad(set_to_none=True)

    if "grad" not in captured_grad:
        raise RuntimeError(f"Gradient for '{target_param}' was not captured.")

    return captured_grad["grad"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def extract_trajectory(
    traj:         Trajectory,
    model:        PreTrainedModel,
    tokenizer,
    max_tokens:   int,
    device:       str,
    target_param: str,
    proj_matrix:  Tensor | None = None,
    pbar=None,
) -> dict[int, Tensor]:
    """Extract gradient vectors for all scoreable steps in a trajectory."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    gradients: dict[int, Tensor] = {}

    for step_idx in iter_scoreable_steps(traj):
        encoded = build_context(
            traj.history, step_idx, tokenizer, max_tokens=max_tokens,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        seq_len = input_ids.shape[1]
        ctx_len = encoded["ctx_len"]

        if pbar is not None:
            pbar.set_postfix(OrderedDict([
                ("file", traj.filename),
                ("seq_len", seq_len),
                ("ctx_len", ctx_len),
                ("step_idx", step_idx),
                ("n_steps", len(traj.history)),
            ]))

        # Skip degenerate cases
        if input_ids.shape[1] <= ctx_len:
            continue

        grad_vec = extract_gradient(
            model, input_ids, attention_mask, ctx_len,
            target_param, proj_matrix,
        )
        gradients[step_idx] = grad_vec

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return gradients


def _extract_metadata(traj: Trajectory) -> dict:
    return {
        "filename":      traj.filename,
        "question_id":   traj.question_id,
        "mistake_agent": traj.mistake_agent,
        "mistake_step":  str(traj.mistake_step),
        "level":         traj.level,
        "subset":        traj.subset,
        "question":      traj.question,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAL-Step Phase 1: extract raw gradient vectors."
    )
    p.add_argument("--model",        required=True, help="HF model name or path.")
    p.add_argument("--input",        required=True, help="Dataset directory.")
    p.add_argument("--output",       required=True, help="Output directory for .pt files.")
    p.add_argument("--target_param", required=True,
                   help="Full parameter name, e.g. model.layers.35.self_attn.v_proj.weight")
    p.add_argument("--max_tokens",   type=int, default=8192)
    p.add_argument("--proj_dim",     type=int, default=None,
                   help="Random projection dimension. None = full gradient (default).")
    p.add_argument("--proj_seed",    type=int, default=42,
                   help="RNG seed for the random projection matrix.")
    p.add_argument("--start_idx",    type=int, default=0)
    p.add_argument("--end_idx",      type=int, default=None)
    p.add_argument("--device",       default=None)
    p.add_argument("--dtype",        choices=["float32", "bfloat16", "float16"],
                   default="bfloat16")
    p.add_argument("--subset",       default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype_map = {
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
    }
    torch_dtype = dtype_map[args.dtype]

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model → {device} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map={"": device},
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f}B parameters loaded.")

    # ── Verify target param exists and print its shape ──────────────────
    target_shape = None
    for name, p in model.named_parameters():
        if name == args.target_param:
            target_shape = p.shape
            break
    if target_shape is None:
        print(f"ERROR: '{args.target_param}' not found.", file=sys.stderr)
        print("Available parameters containing 'v_proj':")
        for name, p in model.named_parameters():
            if "v_proj" in name:
                print(f"  {name}  {tuple(p.shape)}")
        sys.exit(1)

    d = math.prod(target_shape)
    print(f"  Target: {args.target_param}  shape={tuple(target_shape)}  d={d:,}")

    # ── Random projection matrix (optional) ─────────────────────────────
    proj_matrix = None
    if args.proj_dim is not None:
        print(f"  Random projection: {d:,} → {args.proj_dim:,}")
        rng = torch.Generator().manual_seed(args.proj_seed)
        proj_matrix = torch.randn(d, args.proj_dim, generator=rng, dtype=torch.float32)
        proj_matrix /= math.sqrt(args.proj_dim)
        # keep on CPU — projection happens in the hook after .float()

    # ── Load data ───────────────────────────────────────────────────────
    input_path = Path(args.input)
    if args.subset:
        base_path, subset = str(input_path), args.subset
    else:
        base_path, subset = str(input_path.parent), input_path.name

    trajectories = load_dataset(base_path, subset=subset)
    end_idx = args.end_idx if args.end_idx is not None else len(trajectories)
    trajectories = trajectories[args.start_idx:end_idx]
    print(f"  {len(trajectories)} trajectories [{args.start_idx}:{end_idx}]")

    # ── Output dir ──────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save extraction config for reproducibility
    config = {
        "model":        args.model,
        "target_param": args.target_param,
        "target_shape": list(target_shape),
        "d":            d,
        "proj_dim":     args.proj_dim,
        "proj_seed":    args.proj_seed,
        "max_tokens":   args.max_tokens,
        "dtype":        args.dtype,
        "subset":       subset,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    # torch.save(config, out_dir / "config.pt")

    # ── Extract ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    pbar = tqdm(trajectories)

    for traj in pbar:
        out_path = out_dir / traj.filename.replace(".json", ".pt")
        if out_path.exists():
            print(f"  skip: {out_path}")
            continue

        pbar.set_postfix(file=traj.filename, n_steps=len(traj.history))

        gradients = extract_trajectory(
            traj, model, tokenizer, args.max_tokens, device,
            args.target_param, proj_matrix, pbar,
        )

        payload = {
            "metadata":  _extract_metadata(traj),
            "gradients": gradients,
        }
        torch.save(payload, out_path)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s  "
          f"({elapsed / max(len(trajectories), 1):.1f}s/traj)")


if __name__ == "__main__":
    main()
"""
sal/gradient.py — Extract NTP-loss gradients w.r.t. target weight parameters.

For each scoreable step in each trajectory, runs one forward+backward pass and
captures the flattened float16 gradient for every requested target parameter.

Shorthand notation
------------------
  v/35  →  model.layers.35.self_attn.v_proj.weight
  q/35  →  model.layers.35.self_attn.q_proj.weight
  k/35  →  model.layers.35.self_attn.k_proj.weight
  o/35  →  model.layers.35.self_attn.o_proj.weight
  gate/35, up/35, down/35  →  mlp projections
Full dotted names are also accepted unchanged.

Output layout
-------------
  {output}/{shorthand}/{traj}.pt          e.g.  outputs/.../v/35/traj_001.pt

Each .pt file contains:
  {
      "metadata":  { ... },
      "gradients": { step_idx: Tensor(d,) float16 CPU, ... }
  }

Usage
-----
v/35
up/35
gate/32
down/35
k/21

python -m sal.gradient \
    --model "/data/hoang/resources/models/Qwen/Qwen3-8B" \
    --input ww/hand-crafted \
    --output outputs/sal/grads/qwen3-8b/hand-crafted \
    --target_params q/34 k/34 v/35 k/21 v/24 \
    --max_tokens 8192 \
    --start_idx 4 --end_idx 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from gradnorm.data import build_context, iter_scoreable_steps, load_dataset, Trajectory
from gradnorm.losses import _ntp_loss


# ─────────────────────────────────────────────────────────────────────────────
# Shorthand → full parameter name
# ─────────────────────────────────────────────────────────────────────────────

_PROJ_MAP = {
    "q":    "self_attn.q_proj.weight",
    "k":    "self_attn.k_proj.weight",
    "v":    "self_attn.v_proj.weight",
    "o":    "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight",
    "up":   "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}

def expand_param_name(shorthand: str) -> str:
    """'v/35' → 'model.layers.35.self_attn.v_proj.weight'. Full names pass through."""
    if "/" not in shorthand:
        return shorthand
    proj, layer = shorthand.split("/", 1)
    if proj not in _PROJ_MAP:
        raise ValueError(f"Unknown shorthand '{proj}'. Known: {', '.join(_PROJ_MAP)}")
    return f"model.layers.{layer}.{_PROJ_MAP[proj]}"


# ─────────────────────────────────────────────────────────────────────────────
# Gradient extraction (one backward, all targets)
# ─────────────────────────────────────────────────────────────────────────────

def extract_gradient(
    model:          PreTrainedModel,
    input_ids:      Tensor,
    attention_mask: Tensor | None,
    ctx_len:        int,
    target_params:  list[str],
) -> dict[str, Tensor]:
    """One forward+backward; returns flattened float16 gradients for all target params.

    All non-target gradients are zeroed immediately via hooks to keep VRAM low.

    Returns
    -------
    dict mapping full param name → Tensor(d,) float16 on CPU.
    """
    was_training = model.training
    model.train()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    target_set = set(target_params)
    captured: dict[str, Tensor] = {}
    handles = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in target_set:
            def _capture(param, _name=name):
                if param.grad is not None:
                    captured[_name] = param.grad.float().flatten().half().cpu()
                    param.grad = None
            handles.append(p.register_post_accumulate_grad_hook(_capture))
        else:
            def clear_hook(param):
                param.grad = None
            handles.append(p.register_post_accumulate_grad_hook(clear_hook))

    model.zero_grad(set_to_none=True)
    logits = model(input_ids, attention_mask, use_cache=False).logits
    _ntp_loss(logits, input_ids, ctx_len).backward()

    for h in handles:
        h.remove()
    model.gradient_checkpointing_disable()
    if not was_training:
        model.eval()
    model.zero_grad(set_to_none=True)

    return captured


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory loop
# ─────────────────────────────────────────────────────────────────────────────

def extract_trajectory(
    traj:          Trajectory,
    model:         PreTrainedModel,
    tokenizer,
    max_tokens:    int,
    device:        str,
    target_params: list[str],
    pbar=None,
) -> dict[str, dict[int, Tensor]]:
    """Returns {full_param_name: {step_idx: grad_tensor}} for all scoreable steps."""
    result: dict[str, dict[int, Tensor]] = {p: {} for p in target_params}

    for step_idx in iter_scoreable_steps(traj):
        encoded        = build_context(traj.history, step_idx, tokenizer, max_tokens=max_tokens)
        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        ctx_len = encoded["ctx_len"]

        if pbar is not None:
            pbar.set_postfix(
                OrderedDict(
                    file    = traj.filename, 
                    step    = step_idx,
                    seq_len = input_ids.shape[1], 
                    ctx_len = ctx_len)
                )

        if input_ids.shape[1] <= ctx_len:   # degenerate: no step tokens
            continue

        grads = extract_gradient(model, input_ids, attention_mask, ctx_len, target_params)
        for full_name, g in grads.items():
            result[full_name][step_idx] = g

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract NTP-loss gradients for target parameters.")
    p.add_argument("--model",         required=True)
    p.add_argument("--input",         required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--target_params", required=True, nargs="+",
                   help="Shorthands like 'v/35' 'q/35', or full dotted param names.")
    p.add_argument("--max_tokens",    type=int, default=8192)
    p.add_argument("--start_idx",     type=int, default=0)
    p.add_argument("--end_idx",       type=int, default=None)
    p.add_argument("--device",        default=None)
    p.add_argument("--dtype",         choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--subset",        default=None)
    return p.parse_args()


def main():
    args = parse_args()

    device      = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {
        "float32": torch.float32, 
        "bfloat16": torch.bfloat16, 
        "float16": torch.float16
    }[args.dtype]

    # Resolve shorthands
    try:
        param_pairs = [(sh, expand_param_name(sh)) for sh in args.target_params]
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

    # Load model
    print(f"Loading model: {args.model} → {device} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch_dtype, 
        device_map={"": device}
    )
    model.eval()

    # Verify params exist
    named = dict(model.named_parameters())
    for sh, full in param_pairs:
        if full not in named:
            print(f"ERROR: '{full}' not found in model.", file=sys.stderr); sys.exit(1)
        print(f"  {sh!r:12s} → {full}  shape={tuple(named[full].shape)}  d={named[full].numel():,}")

    full_params = [full for _, full in param_pairs]

    # Load dataset
    input_path = Path(args.input)
    if args.subset: base_path, subset = str(input_path), args.subset
    else:           base_path, subset = str(input_path.parent), input_path.name
    trajectories = load_dataset(base_path, subset=subset)
    end_idx = args.end_idx if args.end_idx is not None else len(trajectories)
    trajectories = trajectories[args.start_idx : end_idx]
    print(f"  {len(trajectories)} trajectories")

    # Output dirs + config.json per param
    base_out = Path(args.output)
    out_dirs: dict[str, Path] = {}
    for sh, full in param_pairs:
        d = base_out / sh                      # e.g. outputs/.../v/35
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[full] = d
        cfg = dict(
            model        = args.model, 
            shorthand    = sh, 
            target_param = full,
            shape        = list(named[full].shape), 
            d            = named[full].numel(),
            max_tokens   = args.max_tokens, 
            dtype        = args.dtype, 
            subset       = subset
        )
        (d / "config.json").write_text(json.dumps(cfg, indent=2))

    # Extract
    t0   = time.perf_counter()
    pbar = tqdm(trajectories)
    for traj in pbar:
        stem = traj.filename.replace(".json", ".pt")

        # Skip if all outputs already exist
        if all((out_dirs[f] / stem).exists() for f in full_params):
            pbar.write(f"  skip: {traj.filename}"); continue

        result = extract_trajectory(
            traj, model, tokenizer, args.max_tokens, device, full_params, pbar
        )

        metadata = dict(
            filename      = traj.filename, 
            question_id   = traj.question_id,
            mistake_agent = traj.mistake_agent, 
            mistake_step  = traj.mistake_step,
            level         = traj.level, 
            subset        = traj.subset, 
            question      = traj.question
        )
        for full, grads in result.items():
            torch.save({"metadata": metadata, "gradients": grads}, out_dirs[full] / stem)

    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
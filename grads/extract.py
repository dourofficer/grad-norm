"""
grads/extract.py — Extract and save reduced gradient vectors for trajectories.

For each scoreable step in each trajectory, runs one forward+backward pass and
saves a reduced gradient vector (mean over the larger dimension) for every
requested weight parameter.

Output layout
-------------
One .pt file per trajectory under --output:
    {
        "metadata":  { filename, question_id, mistake_agent, ... },
        "gradients": {
            step_idx: {
                "v/35":   Tensor(4096,) float CPU,
                "gate/35": Tensor(4096,) float CPU,
                ...
            },
            ...
        }
    }

Usage
-----
# Single parameter
python -m grads.extract \
    --model  "/data/hoang/resources/models/Qwen/Qwen3-8B" \
    --input  ww/hand-crafted \
    --output outputs/grads/qwen3-8b/hand-crafted \
    --target_params v/35 \
    --max_tokens 8192

# Multiple parameters
python -m grads.extract \
    --model  "/data/hoang/resources/models/Qwen/Qwen3-8B" \
    --input  ww/hand-crafted \
    --output outputs/grads/qwen3-8b/hand-crafted \
    --target_params v/35 k/21 gate/35 \
    --max_tokens 8192

# All parameters
python -m grads.extract \
    --model  "/data/hoang/resources/models/Qwen/Qwen3-8B" \
    --input  ww/hand-crafted \
    --output outputs/grads/qwen3-8b/hand-crafted \
    --target_params all \
    --max_tokens 8192 \
    --start_idx 0 --end_idx 1

TODO: change device to auto, make the code run on multiple GPUs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable
from safetensors.torch import save_file

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from gradnorm.data import (
    Trajectory,
    _serialize_turns,
    iter_scoreable_steps,
    load_dataset,
)
from gradnorm.data import build_context
from .core import extract_gradient_hooked, shorthand_to_param
from .core import LOSSES

def extract_trajectory(
    traj:          Trajectory,
    model:         PreTrainedModel,
    tokenizer,
    max_tokens:    int,
    target_params: list[str] | str,   # list of shorthands, or "all"
    loss_func:     Callable,
    device:        str,
    context_strategy:   str = "dependency",
    pbar=None,
) -> dict[int, dict[str, Tensor]]:
    """Extract reduced gradient vectors for all scoreable steps in a trajectory.

    Parameters
    ----------
    target_params : list of shorthand strings (e.g. ["v/35", "gate/35"]) or "all".
    context_strategy : the context selection strategy to use.

    Returns
    -------
    dict mapping step_idx → {shorthand: reduced Tensor}
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Resolve shorthands to full param names (or keep "all")
    if target_params == "all":
        full_params: list[str] | str = "all"
    else:
        full_params = [shorthand_to_param(sh) for sh in target_params]

    gradients: dict[int, dict[str, Tensor]] = {}

    for step_idx in iter_scoreable_steps(traj):
        encoded = build_context(
            traj.history, 
            step_idx, 
            tokenizer, 
            max_tokens=max_tokens, 
            strategy=context_strategy
        )

        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        ctx_len = encoded["ctx_len"]
        seq_len = input_ids.shape[1]

        if pbar is not None:
            pbar.set_postfix(OrderedDict([
                ("file",     traj.filename),
                ("seq_len",  seq_len),
                ("ctx_len",  ctx_len),
                ("step_idx", step_idx),
                ("n_steps",  len(traj.history)),
            ]))

        if seq_len <= ctx_len:
            continue

        step_grads = extract_gradient_hooked(
            model, input_ids, attention_mask, ctx_len, full_params, loss_func,
        )
        gradients[step_idx] = step_grads

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
        description="Extract reduced NTP-loss gradients for trajectory datasets."
    )
    p.add_argument("--model",         required=True,  help="HF model name or path.")
    p.add_argument("--input",         required=True,  help="Dataset directory.")
    p.add_argument("--output",        required=True,  help="Output directory for .pt files.")
    p.add_argument(
        "--target_params", required=True, nargs="+",
        help=(
            "Shorthands like 'v/35' 'gate/35', or 'all' to extract every parameter. "
            "Full dotted names are also accepted."
        ),
    )
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--start_idx",  type=int, default=0)
    p.add_argument("--end_idx",    type=int, default=None)
    p.add_argument("--device",     default=None)
    p.add_argument("--dtype",      choices=["float32", "bfloat16", "float16"],
                   default="bfloat16")
    p.add_argument("--subset",     default=None)
    p.add_argument("--loss",       choices=LOSSES.keys(), default="ntp")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature for KL divergence loss (if applicable).")
    p.add_argument("--context", choices=["dependency", "all"], default="dependency",
                   help="Context selection strategy for handcrafted trajectories. ")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]
    context_strategy = args.context

    # ── Resolve target_params ────────────────────────────────────────────────
    loss_func = LOSSES[args.loss]
    if args.loss == "kl_temp":
        from functools import partial
        temp = args.temperature
        print(f"Computing KL divergence with the temperatured-scaled ({temp}) distribution.")
        loss_func = partial(LOSSES[args.loss], temperature=temp)

    if len(args.target_params) == 1 and args.target_params[0] == "all":
        target_params = "all"
    else:
        target_params = args.target_params  # list of shorthands

    # ── Load model ───────────────────────────────────────────────────────────
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

    # ── Validate requested params ────────────────────────────────────────────
    if target_params != "all":
        all_param_names = {n for n, _ in model.named_parameters()}
        for sh in target_params:
            full = shorthand_to_param(sh)
            if full not in all_param_names:
                print(f"ERROR: '{sh}' → '{full}' not found in model.", file=sys.stderr)
                sys.exit(1)
        print(f"  Target params: {target_params}")
    else:
        n = sum(1 for _ in model.named_parameters())
        print(f"  Target params: all ({n} parameters)")

    # ── Load data ────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if args.subset:
        base_path, subset = str(input_path), args.subset
    else:
        base_path, subset = str(input_path.parent), input_path.name

    trajectories = load_dataset(base_path, subset=subset)
    end_idx      = args.end_idx if args.end_idx is not None else len(trajectories)
    trajectories = trajectories[args.start_idx:end_idx]
    print(f"  {len(trajectories)} trajectories [{args.start_idx}:{end_idx}]")

    # ── Output dir ───────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model":         args.model,
        "target_params": target_params if target_params != "all" else "all",
        "max_tokens":    args.max_tokens,
        "dtype":         args.dtype,
        "subset":        subset,
        "loss":          args.loss,
        "temperature":   args.temperature,
        "context":       args.context,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ── Extract ──────────────────────────────────────────────────────────────
    t0   = time.perf_counter()
    pbar = tqdm(trajectories)

    for traj in pbar:
        # out_path = out_dir / traj.filename.replace(".json", ".pt")
        out_path = out_dir / traj.filename.replace(".json", ".safetensors")
        if out_path.exists():
            pbar.write(f"  skip: {traj.filename}")
            continue

        pbar.set_postfix(file=traj.filename, n_steps=len(traj.history))

        gradients = extract_trajectory(
            traj, model, tokenizer, args.max_tokens,
            target_params, loss_func, device, context_strategy, 
            pbar,
        )

        # payload = {
        #     "metadata":  _extract_metadata(traj),
        #     "gradients": gradients,
        # }
        # torch.save(payload, out_path)
        flat_dict = {
            f"{step_idx}.{layer_name}": tensor.contiguous()
            for step_idx, layer_dict in gradients.items()
            for layer_name, tensor in layer_dict.items()
        }
        header_metadata = {
            "payload_metadata": json.dumps(_extract_metadata(traj))
        }
        save_file(flat_dict, out_path, metadata=header_metadata)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s  "
          f"({elapsed / max(len(trajectories), 1):.1f}s/traj)")


if __name__ == "__main__":
    main()
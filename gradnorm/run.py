"""
gradnorm.run — Multi-layer GradNorm inference for Who&When.

Usage
-----
MODEL_LLAMA = "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct"
MODEL_QWEN  = "/data/hoang/resources/models/Qwen/Qwen3-8B" 

# Sweep all layers (for ablation):
CUDA_VISIBLE_DEVICES=1 python -m gradnorm.run \
    --model  "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct" \
    --input  ww/algorithm-generated \
    --max_tokens 8192 \
    --output outputs/llama-3.1-8b/grad-norm/algorithm-generated \
    --start_idx 0 --end_idx 5

CUDA_VISIBLE_DEVICES=0 python -m gradnorm.run \
    --model  "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct" \
    --input  ww/hand-crafted \
    --max_tokens 8192 \
    --loss kl_uniform \
    --output outputs/gradnorm-v2/llama-3.1-8b-kl/hand-crafted \
    --start_idx 0 --end_idx 5


CUDA_VISIBLE_DEVICES=3 python -m gradnorm.run \
    --model  "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct" \
    --input  ww/hand-crafted \
    --max_tokens 8192 \
    --loss kl_temp --temperature 0.3 \
    --output outputs/gradnorm-v2/llama-3.1-8b-kl_temp/hand-crafted \
    --start_idx 0 --end_idx 5

Output schema (one JSON per trajectory)
---------------------------------------
{
    "metadata": { "mistake_agent": ..., "mistake_step": ..., ... },
    "steps":    [ { "step_idx": 0, "role": ..., "content": ... }, ... ],
    "logs":     [
        {
            "step_idx": 1,
            "l1_norm": { "layer_0": ..., "lm_head": ... },
            "l2_norm": { "layer_0": ..., "lm_head": ... }
        },
        ...
    ]
}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable
from collections import OrderedDict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from .data import (
    Trajectory,
    load_dataset,
    build_context,
    iter_scoreable_steps
)
from .core import gradnorm_hooked, gradnorm_hooked_all
from .core import LOSSES

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pad_encoded(encoded: dict[str, Any], tokenizer: PreTrainedTokenizer, max_tokens: int = 4096):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    padded = tokenizer.pad(
        [{"input_ids": ids} for ids in encoded["input_ids"]],
        return_tensors="pt",
        padding_side="left",
        padding="max_length",
        max_length=max_tokens
    )
    padded_ids, attention_mask = padded["input_ids"], padded["attention_mask"]
    num_padded = int(attention_mask.shape[1] - attention_mask.sum(dim=1))
    padded_ctx_len = encoded["ctx_len"] + num_padded
    return {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "ctx_len": padded_ctx_len
    }

def memory_accounting():
    device = "cuda"
    before_mb = torch.cuda.memory_reserved(device) / 1e6
    torch.cuda.empty_cache()
    after_mb = torch.cuda.memory_reserved(device) / 1e6
    print(f"[{device}] reserved: {before_mb:.1f} MB → {after_mb:.1f} MB "
          f"(freed {before_mb - after_mb:.1f} MB)")
    # allocated = torch.cuda.memory_allocated(device)
    # print(f"[{device}] allocated: {allocated / 1e6:.1f} MB")

def _build_output(traj: Trajectory, logs: list[dict]) -> dict:
    """Assemble the output dict for one trajectory."""
    metadata = _extract_metadata_from_traj(traj)
    steps = [
        {
            "step_idx": i,
            "role":     step.get("role", ""),
            "content":  step.get("content", ""),
        }
        for i, step in enumerate(traj.history)
    ]
    return {"metadata": metadata, "steps": steps, "logs": logs}


def _extract_metadata_from_traj(traj: Trajectory) -> dict:
    """Extract metadata dict from a Trajectory dataclass."""
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
# Core scoring loop
# ─────────────────────────────────────────────────────────────────────────────

def score_trajectory(
    traj:       Trajectory,
    model,
    tokenizer,
    max_tokens:  int,
    loss_func:   Callable,
    device:      str,
    pbar: None = None,
) -> list[dict]:
    """Score every scoreable step and return the logs list."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    logs = []
    for step_idx in iter_scoreable_steps(traj):
        encoded = build_context(
            traj.history, step_idx, tokenizer, max_tokens=max_tokens,
        )
        # encoded = pad_encoded(encoded, tokenizer, max_tokens)

        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        seq_len = input_ids.shape[1]
        ctx_len = encoded["ctx_len"]
        if pbar is not None: 
            postfix = OrderedDict([
                ('file', traj.filename),
                ('seq_len', seq_len),
                ('ctx_len', ctx_len),
                ('step_idx', step_idx),
                ('n_steps', len(traj.history))
            ])
            pbar.set_postfix(postfix)

        # Skip degenerate cases (step has 0 tokens)
        if input_ids.shape[1] <= ctx_len:
            logs.append({"step_idx": step_idx, "l1_norm": {}, "l2_norm": {}})
            continue

        statistics = gradnorm_hooked_all(
            model, input_ids, attention_mask, ctx_len, loss_func
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logs.append({
            "step_idx": step_idx,
            "statistics": statistics,
        })

    return logs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-layer GradNorm scoring.")
    p.add_argument("--model",      required=True, help="HF model name or local path.")
    p.add_argument("--input",      required=True, help="Dataset directory (e.g. ww/hand-crafted).")
    p.add_argument("--output",     required=True, help="Output directory for per-trajectory JSONs.")
    p.add_argument("--loss", choices=["ntp", "kl_uniform", "kl_temp"], default="ntp", 
                   help="Loss function for gradient computation.")
    p.add_argument("--temperature", type=float, default=None, help="scaled temperature, specifically for kl_temp.")
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--start_idx",  type=int, default=0)
    p.add_argument("--end_idx",    type=int, default=None)
    p.add_argument("--device",     default=None)
    p.add_argument("--dtype",      choices=["float32", "bfloat16", "float16"],
                   default="bfloat16")
    p.add_argument("--subset",     default=None,
                   help="Subset filter passed to load_dataset (e.g. hand-crafted).")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device / dtype ──────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = getattr(torch, args.dtype)
    loss_func = LOSSES[args.loss]
    if args.loss == "kl_temp":
        from functools import partial
        temp = args.temperature
        print(f"Computing KL divergence with the temperatured-scaled ({temp}) distribution.")
        loss_func = partial(LOSSES[args.loss], temperature=temp)

    # ── Load model ──────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model → {device}  (dtype={args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, 
        # device_map={"": device},
        device_map="auto",
    )
    input_device = next(model.parameters()).device
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f}B parameters loaded.")

    # ── Load data ───────────────────────────────────────────────────────
    # input path is like "ww/hand-crafted" → split into base path + subset
    input_path = Path(args.input)
    # If subset is given, use it; otherwise infer from path
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

    # ── Score ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    pbar = tqdm(trajectories)
    for traj in pbar:
        out_path = out_dir / traj.filename
        if out_path.exists(): 
            print(out_path)
            continue                   

        pbar.set_postfix(file=traj.filename, n_steps=len(traj.history))
        logs = score_trajectory(
            traj, model, tokenizer, args.max_tokens, loss_func, 
            # device, 
            input_device,
            pbar
        )

        result = _build_output(traj, logs)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s  ({elapsed / max(len(trajectories), 1):.1f}s/traj)")


if __name__ == "__main__":
    main()
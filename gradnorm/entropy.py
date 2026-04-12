"""
gradnorm.entropy — Token-entropy uncertainty scoring for Who&When.

Scores each step by the mean Shannon entropy of the model's per-token
predictive distribution.  Unlike ``gradnorm.run``, no backpropagation is
performed — the computation is a single forward pass per step, making this
scorer substantially cheaper.

Intuition
---------
A decisive error step is *surprising* to the model: its next-token
distributions are more peaked (low entropy) because the model is confidently
predicting tokens that conform to its learned patterns but diverge from what
a correct agent would write.  Alternatively, genuine uncertainty manifests as
high entropy.  Either regime can be useful; the raw per-step entropy scalar is
stored so downstream code can negate it as needed.

Score definition
----------------
For a step spanning token positions ctx_len … seq_len-1, after the standard
autoregressive shift (logit i predicts token i+1), the score is:

    H(x_t) = mean_{i in step_positions}  -sum_c  p_c(i) * log p_c(i)

where p_c(i) = softmax(logits[i])_c.  A single float per step is stored.

Output schema (one JSON per trajectory)
---------------------------------------
{
    "metadata": { "mistake_agent": ..., "mistake_step": ..., ... },
    "steps":    [ { "step_idx": 0, "role": ..., "content": ... }, ... ],
    "logs":     [
        { "step_idx": 1, "entropy": 3.217 },
        { "step_idx": 3, "entropy": 2.984 },
        ...
    ]
}

Usage
-----
CUDA_VISIBLE_DEVICES=0 python -m gradnorm.entropy \
    --model  "/data/hoang/resources/models/meta-llama/Llama-3.1-8B-Instruct" \
    --input  ww/algorithm-generated \
    --output outputs/entropy/llama-3.1-8b/algorithm-generated \
    --start_idx 0 --end_idx 25

CUDA_VISIBLE_DEVICES=1 python -m gradnorm.entropy \
    --model  "/data/hoang/resources/models/Qwen/Qwen3-8B" \
    --input  ww/hand-crafted \
    --output outputs/entropy/qwen3-8b/hand-crafted \
    --temperature 0.5 \
    --max_tokens 8192 \
    --start_idx 0 --end_idx 20
"""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import (
    Trajectory,
    load_dataset,
    build_context,
    iter_scoreable_steps,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core: per-step entropy from a single forward pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def entropy_score_step(
    model,
    input_ids:      torch.Tensor,   # (1, seq_len)
    attention_mask: torch.Tensor | None,
    ctx_len:        int,
    temperature:    float = 1.0,
) -> float:
    """Return the mean Shannon entropy (nats) over step tokens for one step.

    Parameters
    ----------
    model         : HuggingFace causal LM in eval mode.
    input_ids     : (1, seq_len) token ids on the correct device.
    attention_mask: (1, seq_len) or None.
    ctx_len       : number of context tokens; step tokens begin at ctx_len.
    temperature   : temperature applied to logits before softmax.  Values < 1
                    sharpen the distribution (lower entropy); values > 1 flatten
                    it (higher entropy).  Default 1.0 leaves logits unchanged.

    Returns
    -------
    Mean entropy in nats as a Python float.
    """
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
    logits  = outputs.logits  # (1, seq_len, vocab)

    # Autoregressive shift: logit at position i predicts token i+1.
    # Step tokens occupy positions ctx_len … seq_len-1 in input_ids.
    # In the shifted view the corresponding logit slice is ctx_len-1 … seq_len-2,
    # i.e. starting at index (ctx_len - 1) (same boundary convention as losses.py).
    shift_logits = logits[:, :-1, :].float()         # (1, seq_len-1, vocab)
    step_logits  = shift_logits[:, ctx_len - 1:, :]  # (1, n_step,   vocab)

    if temperature != 1.0:
        step_logits = step_logits / temperature

    # p log p entropy: -sum_c p_c log p_c, shape (1, n_step)
    log_probs = F.log_softmax(step_logits, dim=-1)   # (1, n_step, vocab)
    probs     = log_probs.exp()
    entropy   = -(probs * log_probs).sum(dim=-1)     # (1, n_step)  in nats

    return entropy.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory-level scoring loop
# ─────────────────────────────────────────────────────────────────────────────

def score_trajectory(
    traj:        Trajectory,
    model,
    tokenizer,
    max_tokens:  int,
    temperature: float,
    device:      str,
    pbar=None,
) -> list[dict]:
    """Score every scoreable step in *traj* and return the logs list."""
    logs: list[dict] = []

    for step_idx in iter_scoreable_steps(traj):
        encoded = build_context(
            traj.history, step_idx, tokenizer, max_tokens=max_tokens,
        )

        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        seq_len = input_ids.shape[1]
        ctx_len = encoded["ctx_len"]

        if pbar is not None:
            pbar.set_postfix(OrderedDict([
                ("file",     traj.filename),
                ("seq_len",  seq_len),
                ("ctx_len",  ctx_len),
                ("step_idx", step_idx),
                ("n_steps",  len(traj.history)),
            ]))

        # Degenerate case: step has zero tokens
        if seq_len <= ctx_len:
            logs.append({"step_idx": step_idx, "entropy": None})
            continue

        h = entropy_score_step(
            model, input_ids, attention_mask, ctx_len, temperature,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logs.append({"step_idx": step_idx, "entropy": h})

    return logs


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers  (mirror gradnorm/run.py)
# ─────────────────────────────────────────────────────────────────────────────

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


def _build_output(traj: Trajectory, logs: list[dict]) -> dict:
    metadata = _extract_metadata(traj)
    steps = [
        {"step_idx": i, "role": s.get("role", ""), "content": s.get("content", "")}
        for i, s in enumerate(traj.history)
    ]
    return {"metadata": metadata, "steps": steps, "logs": logs}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entropy-based uncertainty scoring (forward-pass only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── identical surface as gradnorm.run ──────────────────────────────────
    p.add_argument("--model",      required=True,
                   help="HF model name or local path.")
    p.add_argument("--input",      required=True,
                   help="Dataset directory (e.g. ww/hand-crafted).")
    p.add_argument("--output",     required=True,
                   help="Output directory for per-trajectory JSONs.")
    p.add_argument("--max_tokens", type=int,   default=8192)
    p.add_argument("--start_idx",  type=int,   default=0)
    p.add_argument("--end_idx",    type=int,   default=None)
    p.add_argument("--device",     default=None)
    p.add_argument("--dtype",      choices=["float32", "bfloat16", "float16"],
                   default="bfloat16")
    p.add_argument("--subset",     default=None,
                   help="Subset filter passed to load_dataset (e.g. hand-crafted).")
    # ── entropy-specific ───────────────────────────────────────────────────
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Logit temperature before softmax.  < 1 sharpens, "
                        "> 1 flattens.  Default 1.0 (no scaling).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device / dtype ──────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = getattr(torch, args.dtype)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model → {device}  (dtype={args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map={"": device},
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f}B parameters loaded.")
    if args.temperature != 1.0:
        print(f"  Logit temperature: {args.temperature}")

    # ── Load data ───────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if args.subset:
        base_path, subset = str(input_path), args.subset
    else:
        base_path, subset = str(input_path.parent), input_path.name

    trajectories = load_dataset(base_path, subset=subset)
    end_idx      = args.end_idx if args.end_idx is not None else len(trajectories)
    trajectories = trajectories[args.start_idx:end_idx]
    print(f"  {len(trajectories)} trajectories [{args.start_idx}:{end_idx}]")

    # ── Output dir ──────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Score ───────────────────────────────────────────────────────────────
    t0   = time.perf_counter()
    pbar = tqdm(trajectories)

    for traj in pbar:
        out_path = out_dir / traj.filename
        if out_path.exists():
            print(out_path)
            continue

        pbar.set_postfix(file=traj.filename, n_steps=len(traj.history))
        logs = score_trajectory(
            traj, model, tokenizer,
            args.max_tokens, args.temperature, device, pbar,
        )

        result = _build_output(traj, logs)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s  ({elapsed / max(len(trajectories), 1):.1f}s/traj)")


if __name__ == "__main__":
    main()
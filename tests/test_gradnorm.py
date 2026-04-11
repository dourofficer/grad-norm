from __future__ import annotations

from typing import Any, Callable, Literal
import math
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from contextlib import contextmanager

from gradnorm.data import load_dataset
from gradnorm.losses import _kl_uniform_loss, _kl_temp_loss, _ntp_loss
from gradnorm.core import gradnorm_standard, gradnorm_hooked_all

# qwen3-8b memory, seq_len: 8192, ctx_len: 8085
# kl_temp:       [hooked] peak allocated: 41550.2 MB  (delta: 25168.6 MB)
# kl_temp slice: [hooked] peak allocated: 29103.6 MB  (delta: 12722.0 MB)
# kl_uniform:    [hooked] peak allocated: 36571.5 MB  (delta: 20189.9 MB)
# ntp_loss:      [hooked] peak allocated: 36571.5 MB  (delta: 20189.9 MB)

# ── Configuration (edit these) ───────────────────────────────────
MODEL_NAME    = "/data/hoang/resources/models/Qwen/Qwen3-8B"          
DEVICE        = 0                        # CUDA device index
DATASET_DIR   = "ww"                     # path to dataset
SUBSET        = "hand-crafted"           # or a string subset name
MAX_TOKENS    = 8192 # 12000 is the limit for qwen3-8b
LOSSES        = dict(
    ntp        =_ntp_loss,
    kl_uniform =_kl_uniform_loss,
    kl_temp    =_kl_temp_loss
)

# ── Clean up memory ─────────────────────────────────────────────
def memory_accounting():
    device = "cuda"
    before_mb = torch.cuda.memory_reserved(device) / 1e6
    torch.cuda.empty_cache()
    after_mb = torch.cuda.memory_reserved(device) / 1e6
    print(f"[{device}] reserved: {before_mb:.1f} MB → {after_mb:.1f} MB "
          f"(freed {before_mb - after_mb:.1f} MB)")
    allocated = torch.cuda.memory_allocated(device)
    print(f"[{device}] allocated: {allocated / 1e6:.1f} MB")


def pad_encoded(
    encoded: dict[str, Any], 
    tokenizer: PreTrainedTokenizer, 
    max_tokens: int = 4096
):
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


# ── Peak memory tracker ──────────────────────────────────────────
@contextmanager
def track_peak_memory(device: int = 0, label: str = ""):
    """Context manager that yields peak GPU memory (MB) used inside the block."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    result = {"peak_mb": 0.0, "delta_mb": 0.0}
    yield result
    torch.cuda.synchronize(device)
    peak  = torch.cuda.max_memory_allocated(device)
    after = torch.cuda.memory_allocated(device)
    result["peak_mb"]  = peak / 1e6
    result["delta_mb"] = (peak - before) / 1e6
    if label:
        print(f"[{label}] peak allocated: {result['peak_mb']:.1f} MB  "
              f"(delta: {result['delta_mb']:.1f} MB)")


# ── Comparison helpers ───────────────────────────────────────────
def compare_statistics(stats_std: dict, stats_hook: dict) -> None:
    """Print per-module relative differences between two statistics dicts."""
    print(f"\n{'module':<12} {'metric':<8} {'standard':>14} {'hooked':>14} {'rel_diff':>12}")
    print("─" * 62)
    for name in stats_std:
        for metric in ("l1_norm", "l2_norm"):
            v_std  = stats_std[name][metric]
            v_hook = stats_hook[name][metric]
            denom  = max(abs(v_std), abs(v_hook), 1e-30)
            rel    = abs(v_std - v_hook) / denom
            flag   = " ✗" if rel > 1e-4 else ""
            print(f"{name:<12} {metric:<8} {v_std:>14.8e} {v_hook:>14.8e} {rel:>11.2e}{flag}")


def compare_rank_order(stats_std: dict, stats_hook: dict) -> None:
    """Compare ranked module ordering by l1_norm between the two methods."""
    from scipy.stats import spearmanr, kendalltau

    for metric in ("l1_norm", "l2_norm"):
        rank_std  = sorted(stats_std,  key=lambda n: stats_std[n][metric],  reverse=True)
        rank_hook = sorted(stats_hook, key=lambda n: stats_hook[n][metric], reverse=True)

        # Build rank vectors (0-indexed) for correlation
        names = list(stats_std.keys())
        pos_std  = {n: i for i, n in enumerate(rank_std)}
        pos_hook = {n: i for i, n in enumerate(rank_hook)}
        vec_std  = [pos_std[n]  for n in names]
        vec_hook = [pos_hook[n] for n in names]

        rho, p_spearman = spearmanr(vec_std, vec_hook)
        tau, p_kendall  = kendalltau(vec_std, vec_hook)

        print(f"\n── Rank comparison ({metric}) ──")
        print(f"Spearman ρ = {rho:.6f}  (p = {p_spearman:.2e})")
        print(f"Kendall  τ = {tau:.6f}  (p = {p_kendall:.2e})")

        # Show side-by-side top/bottom 5
        n_show = min(5, len(rank_std))
        print(f"\n  {'rank':<6} {'standard':<14} {'hooked':<14} {'match'}")
        print(f"  {'─'*42}")
        for i in range(len(names)):
            match = "✓" if rank_std[i] == rank_hook[i] else "✗"
            print(f"  {i+1:<6} {rank_std[i]:<14} {rank_hook[i]:<14} {match}")


def test_memory():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .data import build_context        # adjust import as needed

    loss_func = LOSSES['kl_temp']
    print(f"\nLoading tokeniser: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model ({MODEL_NAME}) → cuda:{DEVICE}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype  = torch.bfloat16,
        device_map   = {"": DEVICE},
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f}B parameters loaded.\n")

    trajectories = load_dataset(DATASET_DIR, subset=SUBSET)
    traj_idx, step_idx = 11, 15
    traj = trajectories[traj_idx]

    # ── Tokenise ────────────────────────────────────────────────────
    encoded = build_context(traj.history, step_idx, tokenizer, max_tokens=MAX_TOKENS)
    encoded = pad_encoded(encoded, tokenizer, max_tokens=MAX_TOKENS)

    input_ids      = encoded["input_ids"].to(f"cuda:{DEVICE}")
    attention_mask = encoded["attention_mask"].to(f"cuda:{DEVICE}")
    ctx_len        = encoded["ctx_len"]
    print(f"seq_len: {input_ids.shape[1]}, ctx_len: {ctx_len}")

    # ── Run hooked ──────────────────────────────────────────────────
    print("\n=== gradnorm_hooked ===")
    with track_peak_memory(DEVICE, "hooked") as mem_hook:
        gradnorm_hooked_all(model, input_ids, attention_mask, ctx_len, loss_func)


def test_correctness():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .data import build_context        # adjust import as needed

    print(f"\nLoading tokeniser: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model ({MODEL_NAME}) → cuda:{DEVICE}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype  = torch.bfloat16,
        device_map   = {"": DEVICE},
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f}B parameters loaded.\n")

    trajectories = load_dataset(DATASET_DIR, subset=SUBSET)
    traj_idx, step_idx = 11, 15
    traj = trajectories[traj_idx]

    # ── Tokenise ────────────────────────────────────────────────────
    encoded = build_context(traj.history, step_idx, tokenizer, max_tokens=MAX_TOKENS)
    encoded = pad_encoded(encoded, tokenizer, max_tokens=MAX_TOKENS)

    input_ids      = encoded["input_ids"].to(f"cuda:{DEVICE}")
    attention_mask = encoded["attention_mask"].to(f"cuda:{DEVICE}")
    ctx_len        = encoded["ctx_len"]
    print(f"seq_len: {input_ids.shape[1]}, ctx_len: {ctx_len}")

    # ── Run standard (zero grads first to avoid stale grad corruption) ──
    print("\n=== gradnorm_standard ===")
    model.zero_grad(set_to_none=True)
    with track_peak_memory(DEVICE, "standard") as mem_std:
        stats_std = gradnorm_standard(model, input_ids, attention_mask, ctx_len)

    # ── Run hooked ──────────────────────────────────────────────────
    print("\n=== gradnorm_hooked ===")
    with track_peak_memory(DEVICE, "hooked") as mem_hook:
        stats_hook = gradnorm_hooked(model, input_ids, attention_mask, ctx_len)

    # ── Compare ─────────────────────────────────────────────────────
    compare_statistics(stats_std, stats_hook)
    compare_rank_order(stats_std, stats_hook)

    print(f"\n── Peak memory ──")
    print(f"  standard: {mem_std['peak_mb']:.1f} MB  (delta {mem_std['delta_mb']:.1f} MB)")
    print(f"  hooked:   {mem_hook['peak_mb']:.1f} MB  (delta {mem_hook['delta_mb']:.1f} MB)")
    ratio = mem_std['delta_mb'] / max(mem_hook['delta_mb'], 1e-6)
    print(f"  ratio:    {ratio:.2f}x")

if __name__ == "__main__":
    test_memory()
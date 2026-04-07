from __future__ import annotations

from typing import Any, Callable, Literal
import math
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from contextlib import contextmanager

from .data import load_dataset
from .losses import _kl_uniform_loss, _kl_temp_loss, _ntp_loss

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


def gradnorm_standard(
    model:          PreTrainedModel,
    input_ids:      Tensor,
    attention_mask: Tensor,
    ctx_len:        int,
    loss_func:      Callable = _ntp_loss,
    normalize:      bool = True,
) -> dict:
    # ── Compute gradients ────────────────────────────────────────────
    logits = model(
        input_ids, 
        attention_mask, 
        use_cache=False
    ).logits
    # loss   = _ntp_loss(logits, input_ids, ctx_len)
    loss   = loss_func(logits, input_ids, ctx_len)
    loss.backward()

    # ── Compute score ────────────────────────────────────────────────
    target_weights = list(model.model.layers) + [model.lm_head]
    n_layers = len(model.model.layers)
    module_names = [f"layer_{i}" for i in range(n_layers)] + ["lm_head"]

    statistics = {}
    for name, module in zip(module_names, target_weights):
        n_params = 0
        l1_norm_total = 0.0
        l2_norm_sq_total = 0.0
        
        for p in module.parameters():
            if p.grad is not None:
                n_params += p.numel()
                
                # ✨ Upcast to float64 before doing any math ✨
                grad_f64 = p.grad.detach().double()
                
                l1_norm_total += grad_f64.abs().sum().item()
                l2_norm_sq_total += grad_f64.square().sum().item()
        
        # Calculate final L2 norm from the accumulated double-precision squares
        l2_norm = math.sqrt(l2_norm_sq_total)
        
        if normalize and n_params > 0:
            l1_norm_total /= n_params
            l2_norm /= n_params

        statistics[name] = {
            "l1_norm": l1_norm_total,
            "l2_norm": l2_norm,
        }

    # ── Cleanup ──────────────────────────────────────────────────────
    for p in model.parameters():
        p.grad = None
    memory_accounting()

    return statistics

def gradnorm_hooked(
    model:          PreTrainedModel,
    input_ids:      Tensor,
    attention_mask: Tensor,
    ctx_len:        int,
    loss_func:      Callable = _ntp_loss,
    normalize:      bool = True,
) -> dict:
    # ── 1. Fix ACTIVATION memory ─────────────────────────────────────
    # HF silently ignores gradient checkpointing if model is in eval mode!
    was_training = model.training
    model.train()
    
    # Ensure inputs require grad so checkpointing triggers correctly
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        
    model.gradient_checkpointing_enable()

    # ── Build module → name mapping ──────────────────────────────────
    target_modules = {module: name for name, module in zip(
        [f"layer_{i}" for i in range(len(model.model.layers))] + ["lm_head"],
        list(model.model.layers) + [model.lm_head],
    )}

    # ── 2. Fix GRADIENT memory via Post-Accumulate Hooks ─────────────
    statistics = {}
    handles = []
    hooked_params = set() # Prevents double-counting tied weights (e.g. Qwen embeddings)
    
    # Pre-allocate stats on GPU to prevent CPU-GPU sync bottlenecks!
    device = next(model.parameters()).device

    for module, name in target_modules.items():
        statistics[name] = {
            "l1_norm": torch.tensor(0.0, device=device, dtype=torch.float64), 
            "l2_norm_sq": torch.tensor(0.0, device=device, dtype=torch.float64), 
            "n_params": 0
        }
        
        for p in module.parameters():
            if not p.requires_grad or p in hooked_params:
                continue
            hooked_params.add(p)
            
            entry = statistics[name]
            entry["n_params"] += p.numel()

            def make_stat_hook(entry_dict):
                # PyTorch 2.1+ post-accumulate hook receives the parameter itself
                def hook(param):
                    if param.grad is not None:
                        with torch.no_grad():
                            # Cast to float32/64 to prevent bf16 overflow when squaring
                            grad_f32 = param.grad.float()
                            entry_dict["l1_norm"]    += grad_f32.abs().sum().double()
                            entry_dict["l2_norm_sq"] += grad_f32.square().sum().double()
                        
                        # IMMEDIATELY free the gradient memory!
                        param.grad = None
                return hook

            h = p.register_post_accumulate_grad_hook(make_stat_hook(entry))
            handles.append(h)

    # ── 3. Catch untracked parameters (embeddings, layernorm) ────────
    # So they don't silently leak VRAM
    def clear_hook(param):
        param.grad = None

    for p in model.parameters():
        if p.requires_grad and p not in hooked_params:
            h = p.register_post_accumulate_grad_hook(clear_hook)
            handles.append(h)
            hooked_params.add(p)

    # ── Forward + backward ───────────────────────────────────────────
    model.zero_grad(set_to_none=True)
    
    logits = model(
        input_ids, attention_mask, use_cache=False,
    ).logits
    
    # loss = _ntp_loss(logits, input_ids, ctx_len)
    loss = loss_func(logits, input_ids, ctx_len)
    
    # As backward runs, gradients are instantiated, recorded, and instantly destroyed!
    loss.backward()

    # ── Cleanup ──────────────────────────────────────────────────────
    for h in handles:
        h.remove()
        
    model.gradient_checkpointing_disable()
    if not was_training:
        model.eval()

    # ── 4. Pull metrics to CPU exactly ONCE ──────────────────────────
    for name, stats in statistics.items():
        n_params = stats.pop("n_params")
        
        l1_val = stats.pop("l1_norm").item()
        l2_val = math.sqrt(stats.pop("l2_norm_sq").item())
        
        if normalize and n_params > 0:
            l1_val /= n_params
            l2_val /= n_params
            
        stats["l1_norm"] = l1_val
        stats["l2_norm"] = l2_val

    # Double check no stray gradients remain
    model.zero_grad(set_to_none=True)

    return statistics

# ---------------------------------------------------------------------
# hook all layers
# ---------------------------------------------------------------------
def gradnorm_hooked_all(
    model:          PreTrainedModel,
    input_ids:      Tensor,
    attention_mask: Tensor,
    ctx_len:        int,
    loss_func:      Callable = _ntp_loss,
    normalize:      bool = True,
) -> dict:
    # ── 1. Fix ACTIVATION memory ─────────────────────────────────────
    # HF silently ignores gradient checkpointing if model is in eval mode!
    was_training = model.training
    model.train()
    
    # Ensure inputs require grad so checkpointing triggers correctly
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # ── Build per-parameter stats + hooks ───────────────────────────
    statistics = {}
    handles = []
    hooked_params = set()
    device = next(model.parameters()).device

    for param_name, p in model.named_parameters():
        if not p.requires_grad or p in hooked_params:
            continue
        hooked_params.add(p)

        statistics[param_name] = {
            "l1_norm":    torch.tensor(0.0, device=p.device, dtype=torch.float64),
            "l2_norm_sq": torch.tensor(0.0, device=p.device, dtype=torch.float64),
            "n_params":   p.numel(),
        }
        entry = statistics[param_name]

        def make_stat_hook(entry_dict):
            def hook(param):
                if param.grad is not None:
                    with torch.no_grad():
                        grad_f32 = param.grad.float()
                        entry_dict["l1_norm"]    += grad_f32.abs().sum().double()
                        entry_dict["l2_norm_sq"] += grad_f32.square().sum().double()
                    param.grad = None
            return hook

        h = p.register_post_accumulate_grad_hook(make_stat_hook(entry))
        handles.append(h)

    # ── Forward + backward ───────────────────────────────────────────
    model.zero_grad(set_to_none=True)
    
    logits = model(
        input_ids, attention_mask, use_cache=False,
    ).logits
    
    # loss = _ntp_loss(logits, input_ids, ctx_len)
    loss = loss_func(logits, input_ids, ctx_len)
    
    # import pdb; pdb.set_trace()
    # As backward runs, gradients are instantiated, recorded, and instantly destroyed!
    loss.backward()

    # ── Cleanup ──────────────────────────────────────────────────────
    for name, stats in statistics.items():
        for k, v in stats.items():
            if isinstance(v, Tensor): stats[k] = v.item()
        statistics[name] = stats

    for h in handles:
        h.remove()
        
    model.gradient_checkpointing_disable()
    if not was_training: model.eval()

    # Double check no stray gradients remain
    model.zero_grad(set_to_none=True)

    return statistics


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
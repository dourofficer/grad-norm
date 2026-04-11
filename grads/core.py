"""
grads/core.py — Low-level gradient extraction primitives.

Public API
----------
param_to_shorthand(name)          full param name → shorthand (e.g. "v/35")
shorthand_to_param(sh)            inverse mapping
reduce_gradient(grad, shape)      2-D mean-reduction to the smaller dimension
extract_gradient_hooked(...)      forward+backward with hooks; memory-efficient
extract_gradient_standard(...)    plain forward+backward; ground-truth reference
"""
from __future__ import annotations

import re
from typing import Literal, Callable

import torch
from torch import Tensor
from transformers import PreTrainedModel

from gradnorm.losses import _kl_uniform_loss, _kl_temp_loss, _ntp_loss
LOSSES        = dict(
    ntp        =_ntp_loss,
    kl_uniform =_kl_uniform_loss,
    kl_temp    =_kl_temp_loss
)

# ─────────────────────────────────────────────────────────────────────────────
# Shorthand ↔ full parameter name
# ─────────────────────────────────────────────────────────────────────────────

# Patterns are matched in order; first match wins.
# Each tuple: (regex_on_full_name, shorthand_template)
# Use {i} as a placeholder for the captured layer index.

# NOTE: qwen3-4b has tied lm_head and embed_tokens, no lm_head in the weight list
# overall, hooked implementation is consistent with standard implementation
#   for example, ntp loss on qwen3-8b:
#   [✓ PASS] q/35  shape=(4096,)
#       cos_sim=1.000000  max_diff=0.000e+00  rel_diff=0.000e+00
#       L1  std=0.0154  hook=0.0154  L2  std=0.0004  hook=0.0004

_FULL_TO_SHORT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"),    "q/{i}"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$"),    "k/{i}"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$"),    "v/{i}"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"),    "o/{i}"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_norm\.weight$"),    "q_norm/{i}"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_norm\.weight$"),    "k_norm/{i}"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"),       "gate/{i}"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$"),         "up/{i}"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$"),       "down/{i}"),
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.weight$"),      "in_norm/{i}"),
    (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$"), "post_norm/{i}"),
    (re.compile(r"^model\.embed_tokens\.weight$"),                        "embed"),
    (re.compile(r"^model\.norm\.weight$"),                                "norm"),
    (re.compile(r"^lm_head\.weight$"),                                    "lm_head"),
]

# Inverse: shorthand prefix → full name template (with {i} for layer index)
_SHORT_PREFIX_TO_FULL: dict[str, str] = {
    "q":         "model.layers.{i}.self_attn.q_proj.weight",
    "k":         "model.layers.{i}.self_attn.k_proj.weight",
    "v":         "model.layers.{i}.self_attn.v_proj.weight",
    "o":         "model.layers.{i}.self_attn.o_proj.weight",
    "q_norm":    "model.layers.{i}.self_attn.q_norm.weight",
    "k_norm":    "model.layers.{i}.self_attn.k_norm.weight",
    "gate":      "model.layers.{i}.mlp.gate_proj.weight",
    "up":        "model.layers.{i}.mlp.up_proj.weight",
    "down":      "model.layers.{i}.mlp.down_proj.weight",
    "in_norm":   "model.layers.{i}.input_layernorm.weight",
    "post_norm": "model.layers.{i}.post_attention_layernorm.weight",
}
_SHORT_FLAT_TO_FULL: dict[str, str] = {
    "embed":    "model.embed_tokens.weight",
    "norm":     "model.norm.weight",
    "lm_head":  "lm_head.weight",
}


def param_to_shorthand(full_name: str) -> str:
    """Convert a full dotted parameter name to its shorthand.

    Examples
    --------
    >>> param_to_shorthand("model.layers.35.self_attn.v_proj.weight")
    'v/35'
    >>> param_to_shorthand("model.norm.weight")
    'norm'
    """
    for pattern, template in _FULL_TO_SHORT_PATTERNS:
        m = pattern.match(full_name)
        if m:
            groups = m.groups()
            return template.replace("{i}", groups[0]) if groups else template
    # Fall back to the full name if no pattern matches (e.g. bias terms)
    return full_name


def shorthand_to_param(shorthand: str) -> str:
    """Convert a shorthand back to its full dotted parameter name.

    Examples
    --------
    >>> shorthand_to_param("v/35")
    'model.layers.35.self_attn.v_proj.weight'
    >>> shorthand_to_param("norm")
    'model.norm.weight'
    """
    if "/" in shorthand:
        prefix, layer = shorthand.split("/", 1)
        if prefix not in _SHORT_PREFIX_TO_FULL:
            raise ValueError(f"Unknown shorthand prefix '{prefix}'.")
        return _SHORT_PREFIX_TO_FULL[prefix].replace("{i}", layer)
    if shorthand in _SHORT_FLAT_TO_FULL:
        return _SHORT_FLAT_TO_FULL[shorthand]
    # Assume it's already a full name
    return shorthand


# ─────────────────────────────────────────────────────────────────────────────
# Gradient reduction
# ─────────────────────────────────────────────────────────────────────────────

def reduce_gradient(grad: Tensor, original_shape: tuple[int, ...]) -> Tensor:
    """Reduce a flattened gradient to a compact 1-D vector.

    For a 2-D weight of shape (R, C):
      - mean over dim-0 (rows) if R >= C  →  result shape (C,)
      - mean over dim-1 (cols) if C >  R  →  result shape (R,)
    This always keeps the smaller dimension, consistently producing
    the hidden-size axis (e.g. 4096) for transformer weights.

    For 1-D weights (layer-norm, etc.) the gradient is returned as-is.
    For higher-rank weights the gradient is flattened and returned as-is.

    NOTE: somehow, mean reduction produces a vector with smaller magnitude.
    Parameters
    ----------
    grad           : float32 gradient tensor (shape: prod(original_shape),)
    original_shape : the shape of the weight parameter before flattening

    Returns
    -------
    Reduced float32 tensor on CPU.
    """
    if len(original_shape) == 2:
        R, C = original_shape
        assert grad.shape == (R, C)
        # g2d = grad.reshape(R, C)
        if R >= C:
            reduced = grad.sum(dim=0)   # (C,)
        else:
            reduced = grad.sum(dim=1)   # (R,)
        return reduced.half().cpu()

    # 1-D or unusual shapes: keep as-is
    return grad.half().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Hooked extraction  (memory-efficient)
# ─────────────────────────────────────────────────────────────────────────────

def _capture_hook(name: str, shape: tuple, captured: dict):
    def hook(param):
        if param.grad is not None:
            # breakpoint()
            with torch.no_grad():
                captured[name] = reduce_gradient(param.grad.float(), shape)
            param.grad = None
    return hook


def _clear_hook(param):
    param.grad = None


def extract_gradient_hooked(
    model:          PreTrainedModel,
    input_ids:      Tensor,
    attention_mask: Tensor | None,
    ctx_len:        int,
    target_params:  list[str] | Literal["all"],
    loss_func:      Callable,
) -> dict[str, Tensor]:
    """One forward+backward pass; returns reduced gradients for requested params.

    Uses gradient checkpointing and post-accumulate hooks to capture only the
    target gradients, discarding all others immediately to minimise VRAM.

    Parameters
    ----------
    model          : HuggingFace causal LM.
    input_ids      : (1, seq_len) token ids.
    attention_mask : (1, seq_len) or None.
    ctx_len        : context token count (NTP loss ignores these positions).
    target_params  : list of full dotted param names, or "all" for every param.

    Returns
    -------
    dict mapping shorthand → reduced float32 Tensor on CPU.
    """
    was_training = model.training
    model.train()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Determine which param names we want
    all_named: dict[str, Tensor] = dict(model.named_parameters())
    if target_params == "all":
        wanted: set[str] = set(all_named.keys())
    else:
        wanted = set(target_params)
        missing = wanted - set(all_named.keys())
        if missing:
            raise ValueError(f"Parameters not found in model: {missing}")

    # Collect shapes before the pass (needed for reduction)
    param_shapes: dict[str, tuple[int, ...]] = {
        name: tuple(p.shape) for name, p in all_named.items() if name in wanted
    }

    # ── Hooks ────────────────────────────────────────────────────────────────
    captured: dict[str, Tensor] = {}
    handles = []

    for name, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        if name in wanted: hook = _capture_hook(name, param_shapes[name], captured)
        else:              hook = _clear_hook
        handles.append(p.register_post_accumulate_grad_hook(hook))

    # ── Forward + backward ───────────────────────────────────────────────────
    model.zero_grad(set_to_none=True)
    logits = model(input_ids, attention_mask, use_cache=False).logits
    loss = loss_func(logits, input_ids, ctx_len)
    loss.backward()

    # ── Cleanup ──────────────────────────────────────────────────────────────
    for h in handles:
        h.remove()
    model.gradient_checkpointing_disable()
    if not was_training:
        model.eval()
    model.zero_grad(set_to_none=True)

    # Re-key by shorthand
    return {param_to_shorthand(n): g for n, g in captured.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Standard extraction  (ground-truth reference, no hooks)
# ─────────────────────────────────────────────────────────────────────────────

def extract_gradient_standard(
    model:          PreTrainedModel,
    input_ids:      Tensor,
    attention_mask: Tensor | None,
    ctx_len:        int,
    target_params:  list[str] | Literal["all"],
    loss_func:      Callable,
) -> dict[str, Tensor]:
    """Plain forward+backward; reads .grad directly after loss.backward().

    No gradient checkpointing, no hooks. Intended as a correctness reference.
    Returns the same dict[shorthand → reduced Tensor] schema as the hooked version.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    all_named = dict(model.named_parameters())
    if target_params == "all":
        wanted: set[str] = set(all_named.keys())
    else:
        wanted = set(target_params)

    logits = model(input_ids, attention_mask, use_cache=False).logits
    loss   = loss_func(logits, input_ids, ctx_len)
    loss.backward()

    result: dict[str, Tensor] = {}
    for name, p in model.named_parameters():
        if name not in wanted:
            continue
        if p.grad is None:
            raise RuntimeError(f"No gradient computed for '{name}'.")
        with torch.no_grad():
            g = p.grad.float()
            result[param_to_shorthand(name)] = reduce_gradient(g, tuple(p.shape))

    model.zero_grad(set_to_none=True)
    return result
"""
gradnorm.py ― GradNorm anomaly scoring for Who&When evaluation.

Core computation
----------------
For each step v_t in a trajectory:

    S(v_t) = |∂ L_NTP(v_t) / ∂W|₁

    L_NTP(v_t) = -(1/L) Σ log p_θ(x_ℓ | x_{<ℓ}, ctx_t)

where ctx_t is all turns preceding t, x_{1..L} are the step's tokens,
and W is a target weight matrix.

Layer variants (W choices)
--------------------------
    "lm_head"     — model.lm_head.weight only
    "out_proj"    — self_attn.o_proj.weight in the final transformer layer
    "final_layer" — all weight matrices in the final transformer layer + lm_head

Gradient strategies
-------------------
    "standard"  — full forward pass; only target params have requires_grad=True.
                  Simple and correct.  Stores the full computation graph.

    "split"     — memory-efficient variant.
                  Phase 1 (no_grad): run the model up to the layer boundary,
                  capturing exact inputs to the final layer via a forward hook.
                  Phase 2 (grad):    re-run only the target component from the
                  captured state; backward propagates only through it.
                  Avoids storing activations for the prefix layers.

Public API
----------
score_trajectory()   — score every step in a Trajectory; primary entry point
gradnorm_standard()  — single-step, strategy="standard"
gradnorm_split()     — single-step, strategy="split"
get_target_params()  — helper: list of Parameter objects for a layer variant
"""
from __future__ import annotations

from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

from .data import Trajectory, build_context, iter_scoreable_steps

LayerVariant  = Literal["lm_head", "out_proj", "final_layer"]
GradStrategy  = Literal["standard", "split"]


# ─────────────────────────────────────────────────────────────────────────────
# Target parameter selection
# ─────────────────────────────────────────────────────────────────────────────

def get_target_params(
    model:         PreTrainedModel,
    layer_variant: LayerVariant,
) -> list[torch.nn.Parameter]:
    """Return the parameter tensors that define W for the chosen layer variant.

    Supports Llama-style and Qwen-style architectures, which share the
    attribute path ``model.model.layers[-1].self_attn.o_proj``.

    Parameters
    ----------
    model         : the language model (must have a .model.layers attribute).
    layer_variant : "lm_head" | "out_proj" | "final_layer".

    Returns
    -------
    list[torch.nn.Parameter]
        Each element has a .grad attribute populated after .backward().
    """
    transformer = model.model          # LlamaModel / Qwen2Model / ...
    final_layer = transformer.layers[-1]

    if layer_variant == "lm_head":
        return [model.lm_head.weight]

    elif layer_variant == "out_proj":
        # Output projection of the final attention block.
        # Named "o_proj" in both Llama and Qwen.
        return [final_layer.self_attn.o_proj.weight]

    elif layer_variant == "final_layer":
        # All weight matrices in the last transformer layer + lm_head.
        # Uses list() to materialise the generator so callers get a stable list.
        params = list(final_layer.parameters())
        params.append(model.lm_head.weight)
        return params

    else:
        raise ValueError(
            f"Unknown layer_variant {layer_variant!r}. "
            "Choose from: 'lm_head', 'out_proj', 'final_layer'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# NTP loss
# ─────────────────────────────────────────────────────────────────────────────

def _ntp_loss(
    logits:    Tensor,   # (1, seq_len, vocab_size)
    input_ids: Tensor,   # (1, seq_len)
    ctx_len:   int,
) -> Tensor:
    """Mean NTP loss over the step tokens (positions ctx_len … seq_len-1).

    Uses the standard autoregressive shift: logits[i] predicts token i+1.

    Positions belonging to the context (indices 0 … ctx_len-1 in the shifted
    representation) are masked with ignore_index=-100 so they do not
    contribute to the loss.

    Parameters
    ----------
    logits    : raw logits from the language model head.
    input_ids : token IDs of the full sequence.
    ctx_len   : number of tokens before the first step-content token.

    Returns
    -------
    Scalar Tensor (the mean NTP loss).

    Derivation of the mask boundary
    --------------------------------
    Shifted positions:  0, 1, ..., N-2   (each predicts the next token)
    Step tokens are at positions ctx_len … N-1 in input_ids.
    In the shifted view, predicting step token ctx_len requires logit at
    position ctx_len-1.  So we mask positions 0 … ctx_len-2, i.e. the first
    (ctx_len - 1) positions of shift_labels.
    """
    # Autoregressive shift
    shift_logits = logits[:, :-1, :].contiguous().float()   # (1, N-1, vocab)
    shift_labels = input_ids[:, 1:].clone()          # (1, N-1)

    # Mask context positions: first (ctx_len - 1) positions do not predict
    # step tokens.
    mask_end = ctx_len - 1   # exclusive upper bound of masked region
    if mask_end > 0:
        shift_labels[:, :mask_end] = -100

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index = -100,
        reduction    = "mean",
    )
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Standard
# ─────────────────────────────────────────────────────────────────────────────

def gradnorm_standard(
    model:         PreTrainedModel,
    input_ids:     Tensor,
    ctx_len:       int,
    normalize:     bool,
    layer_variant: LayerVariant,
) -> float:
    """Compute GradNorm via a full forward + backward pass.

    All model parameters are frozen except the target weight(s), so PyTorch
    only back-propagates through the minimal graph segment that connects the
    loss to those weights.  Nevertheless, intermediate activations for the
    entire forward pass are held in memory until .backward() completes.

    Parameters
    ----------
    model         : language model in eval mode, on the correct device.
    input_ids     : (1, seq_len) on the same device.
    ctx_len       : context token count (from build_context).
    layer_variant : which weight(s) to differentiate.

    Returns
    -------
    float — L1 norm of ∂L/∂W (summed over all target parameters).
    """
    # ── Freeze everything ────────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad_(False)

    target_params = get_target_params(model, layer_variant)
    for p in target_params:
        p.requires_grad_(True)

    # ── Compute score ────────────────────────────────────────────────
    score = 0.0
    logits = model(input_ids, use_cache=False).logits  # (1, seq_len, vocab)
    loss   = _ntp_loss(logits, input_ids, ctx_len)
    loss.backward()

    score = sum(
        p.grad.abs().sum().item()
        for p in target_params
        if p.grad is not None
    )
    if normalize: score = score / sum(p.numel() for p in target_params)

    # ── Cleanup: zero grads, re-freeze ───────────────────────────────
    for p in target_params:
        if p.grad is not None:
            p.grad = None
        p.requires_grad_(False)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Split forward
# ─────────────────────────────────────────────────────────────────────────────

def gradnorm_split(
    model:         PreTrainedModel,
    input_ids:     Tensor,
    ctx_len:       int,
    layer_variant: LayerVariant,
) -> float:
    """Compute GradNorm via a split forward pass (memory-efficient).

    Two-phase execution:

    Phase 1 — no_grad:
        Run the full model forward with ``torch.no_grad()``.
        For the "lm_head" variant, capture the transformer body's output
        (last_hidden_state).
        For "out_proj" / "final_layer" variants, register a forward pre-hook
        on ``transformer.layers[-1]`` to capture *exactly* the (args, kwargs)
        passed to that layer during the normal forward — including any
        model-specific tensors such as ``position_embeddings`` for Llama 3.
        The hook is removed immediately after the pass.

    Phase 2 — grad:
        Re-run only the target component from the captured boundary state,
        with ``requires_grad=True`` on the target parameters.
        Backward propagates only through this lightweight sub-graph, so
        activations for the prefix layers never need to be retained.

    Requires PyTorch ≥ 2.0 (for ``register_forward_pre_hook(with_kwargs=True)``).

    Parameters
    ----------
    model         : language model in eval mode, on the correct device.
    input_ids     : (1, seq_len) on the same device.
    ctx_len       : context token count (from build_context).
    layer_variant : which weight(s) to differentiate.

    Returns
    -------
    float — L1 norm of ∂L/∂W (summed over all target parameters).
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    transformer = model.model   # the transformer body (LlamaModel / Qwen2Model)
    score = 0.0

    # ── lm_head: clean split at the transformer-body boundary ────────────
    if layer_variant == "lm_head":
        # Phase 1: full transformer body, no grad
        with torch.no_grad():
            last_hidden = transformer(
                input_ids = input_ids,
                use_cache = False,
            ).last_hidden_state.detach()   # (1, seq_len, d_model)

        # Phase 2: lm_head only, with grad
        target = model.lm_head.weight
        target.requires_grad_(True)
        try:
            logits = model.lm_head(last_hidden)    # (1, seq_len, vocab)
            loss   = _ntp_loss(logits, input_ids, ctx_len)
            loss.backward()
            if target.grad is not None:
                score = target.grad.abs().sum().item()
        finally:
            if target.grad is not None:
                target.grad = None
            target.requires_grad_(False)

    # ── out_proj / final_layer: hook-based boundary capture ──────────────
    elif layer_variant in ("out_proj", "final_layer"):
        # Phase 1: full forward with no_grad; hook captures the exact inputs
        # to the final transformer layer (args + all kwargs, e.g.
        # position_embeddings for Llama 3), then is immediately removed.
        _captured: dict[str, Any] = {}

        def _pre_hook(module, args, kwargs):  # noqa: ANN001
            # Detach all tensor arguments so they become leaf constants in the
            # phase-2 graph; non-tensor arguments are passed through as-is.
            _captured["args"] = tuple(
                a.detach() if isinstance(a, Tensor) else a for a in args
            )
            _captured["kwargs"] = {
                k: v.detach() if isinstance(v, Tensor) else v
                for k, v in kwargs.items()
            }

        handle = transformer.layers[-1].register_forward_pre_hook(
            _pre_hook, with_kwargs=True   # requires PyTorch ≥ 2.0
        )
        with torch.no_grad():
            model(input_ids, use_cache=False)
        handle.remove()

        # Phase 2: re-run final_layer → norm → lm_head with grad
        if layer_variant == "out_proj":
            target_params = [transformer.layers[-1].self_attn.o_proj.weight]
        else:  # "final_layer"
            target_params = list(transformer.layers[-1].parameters())
            target_params.append(model.lm_head.weight)

        for p in target_params:
            p.requires_grad_(True)

        try:
            # Re-run the final layer from the captured boundary state.
            # position_embeddings, attention_mask, etc. are already in kwargs.
            layer_out  = transformer.layers[-1](
                *_captured["args"], **_captured["kwargs"]
            )
            hidden     = layer_out[0]               # hidden_states output
            hidden     = transformer.norm(hidden)   # final layer-norm
            logits     = model.lm_head(hidden)      # (1, seq_len, vocab)
            loss       = _ntp_loss(logits, input_ids, ctx_len)
            loss.backward()

            score = sum(
                p.grad.abs().sum().item()
                for p in target_params
                if p.grad is not None
            )
        finally:
            for p in target_params:
                if p.grad is not None:
                    p.grad = None
                p.requires_grad_(False)

    else:
        raise ValueError(
            f"Unknown layer_variant {layer_variant!r}. "
            "Choose from: 'lm_head', 'out_proj', 'final_layer'."
        )

    return score


# ─────────────────────────────────────────────────────────────────────────────
# High-level trajectory scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_trajectory(
    trajectory:      Trajectory,
    model:           PreTrainedModel,
    tokenizer:       PreTrainedTokenizer,
    layer_variant:   LayerVariant,
    strategy:        GradStrategy           = "standard",
    context_builder: Callable               = build_context,
    normalize:       bool                   = True,
    device:          str | torch.device     = "cuda",
    verbose:         bool                   = False,
) -> dict[str, Any]:
    """Score every scoreable step in a trajectory with GradNorm.

    Iterates over steps [1, …, T-1] (step 0 = human question is skipped),
    tokenises each (context, step) pair via ``context_builder``, and computes
    the GradNorm score with the chosen strategy and layer variant.

    Steps where the step content has zero tokens after tokenisation are
    assigned a score of 0.0 and flagged.

    Parameters
    ----------
    trajectory      : a Trajectory instance.
    model           : language model in eval mode on ``device``.
    tokenizer       : corresponding tokeniser.
    layer_variant   : "lm_head" | "out_proj" | "final_layer".
    strategy        : "standard" | "split".
    context_builder : callable with signature
                      ``(history, step_idx, tokenizer) → {"input_ids", "ctx_len"}``.
                      Swap in ``custom_build_context`` for custom behaviour.
    device          : torch device string or object.
    verbose         : if True, print per-step progress.

    Returns
    -------
    dict with keys:
        "question_id"  : str
        "scores"       : dict[int, float]   step_idx → GradNorm score
        "true_step"    : int                ground-truth mistake step
        "true_agent"   : str                ground-truth mistake agent
        "step_agents"  : dict[int, str]     step_idx → role string
        "skipped"      : list[int]          steps with zero scoreable tokens
    """
    model.eval()
    grad_fn = gradnorm_standard if strategy == "standard" else gradnorm_split

    scores:      dict[int, float] = {}
    step_agents: dict[int, str]   = {}
    skipped:     list[int]        = []

    for step_idx in range(len(trajectory.history)):
        role = trajectory.history[step_idx].get("role", f"step_{step_idx}")
        step_agents[step_idx] = role

        # ── Tokenise ────────────────────────────────────────────────────
        encoded   = context_builder(trajectory.history, step_idx, tokenizer)
        input_ids = encoded["input_ids"].to(device)   # (1, seq_len)
        ctx_len   = encoded["ctx_len"]
        seq_len   = input_ids.shape[1]

        # ── Guard: skip if step has no tokens ───────────────────────────
        if seq_len <= ctx_len:
            if verbose:
                print(f"  step {step_idx:3d} [{role}]: SKIP (no step tokens)")
            scores[step_idx] = 0.0
            skipped.append(step_idx)
            continue

        # ── Score ────────────────────────────────────────────────────────
        score = grad_fn(model, input_ids, ctx_len, normalize, layer_variant)
        scores[step_idx] = score

        if verbose:
            print(f"  step {step_idx:3d} [{role}]: {score:.6f}")

        # ── Free CUDA memory between steps ───────────────────────────────
        del input_ids
        if device != "cpu" and str(device) != "cpu":
            torch.cuda.empty_cache()

    return {
        "question_id": trajectory.question_id,
        "scores":      scores,
        "true_step":   trajectory.mistake_step,
        "true_agent":  trajectory.mistake_agent,
        "step_agents": step_agents,
        "skipped":     skipped,
    }

"""
data.py ― Dataset loading, step parsing, and context construction
           for Who&When GradNorm evaluation.

Public API
----------
Trajectory          dataclass holding one failure instance
load_dataset()      load a JSON file → list[Trajectory]
select_context()    ← PLACEHOLDER: return which history indices serve as context
build_context()     tokenise one (context, step) pair via apply_chat_template
custom_build_context()  ← PLACEHOLDER: drop-in replacement for build_context
iter_scoreable_steps()  steps that should receive a GradNorm score
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from transformers import PreTrainedTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """One Who&When failure instance.

    Attributes
    ----------
    question_id   : unique ID string from the dataset.
    history       : raw history list; step t == history[t] (0-indexed).
    mistake_agent : ground-truth agent name (matches a history[t]["role"]).
    mistake_step  : ground-truth step index (0-indexed into history).
    level         : difficulty level.
    subset        : "algo" | "handcrafted" (or "unknown" if absent in JSON).
    question      : original user question string.
    """
    question_id:   str
    history:       list[dict]
    mistake_agent: str
    mistake_step:  int           # 0-indexed
    level:         int
    subset:        str
    question:      str


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    path: str | Path,
    subset: str | None = None,
) -> list[Trajectory]:
    """Load the Who&When dataset from a JSON file.

    Parameters
    ----------
    path   : path to the JSON file.
             Accepts a list of dicts (full dataset) or a single dict.
    subset : optional filter — "algo" | "handcrafted".
             Pass None to return everything.

    Returns
    -------
    list[Trajectory]

    Expected JSON schema per item
    ------------------------------
    {
        "history":       [{"role": str, "content": str}, ...],
        "mistake_agent": str,
        "mistake_step":  str | int,   # parsed to int; 0-indexed
        "question_ID":   str,
        "level":         int,          # optional
        "subset":        str,          # optional; "algo" | "handcrafted"
        "question":      str           # optional
    }

    Notes
    -----
    If the JSON file does not contain a "subset" key (e.g., separate files per
    subset), supply the subset label via the `subset` argument *as a filter*
    only, or pre-tag the items before calling this function.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]

    trajectories: list[Trajectory] = []
    for item in raw:
        traj = Trajectory(
            question_id   = item["question_ID"],
            history       = item["history"],
            mistake_agent = item["mistake_agent"],
            mistake_step  = int(item["mistake_step"]),
            level         = item.get("level", -1),
            subset        = item.get("subset", "unknown"),
            question      = item.get("question", ""),
        )
        if subset is None or traj.subset == subset:
            trajectories.append(traj)

    return trajectories


# ─────────────────────────────────────────────────────────────────────────────
# Context selection  ←  PLACEHOLDER
# ─────────────────────────────────────────────────────────────────────────────

def select_context(history: list[dict], step_idx: int) -> list[int]:
    """Return the indices of history turns to use as context for step `step_idx`.

    **Default**: every turn strictly before step_idx, i.e. range(step_idx).

    This function is called inside :func:`build_context` and is the
    **primary hook for truncation strategies**.  Replace or monkey-patch it
    to implement e.g.:

    * Last-K turns::

        def select_context(history, step_idx):
            K = 20
            start = max(0, step_idx - K)
            return list(range(start, step_idx))

    * Token-budget truncation (compute token counts externally, then slice)::

        def select_context(history, step_idx):
            # build from the right until budget is exhausted
            ...

    * Agent-role filtering (drop "Orchestrator (thought)" turns, etc.)::

        def select_context(history, step_idx):
            keep = {"human", "WebSurfer", "Assistant", "ComputerTerminal"}
            return [i for i in range(step_idx) if history[i]["role"] in keep]

    Parameters
    ----------
    history  : full trajectory history list.
    step_idx : the step being scored (0-indexed; never 0 itself).

    Returns
    -------
    list[int]
        Ordered indices into `history` to include as context.
        All indices must satisfy idx < step_idx.
    """
    return list(range(step_idx))


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_turns(history: list[dict], indices: list[int]) -> str:
    """Flatten selected turns into a single plain-text string.

    Format per turn:
        [<role>]: <content>

    Turns are separated by a blank line.  Roles are kept verbatim (e.g.
    "Orchestrator (thought)", "WebSurfer") so the model sees the full
    multi-agent structure.
    """
    parts: list[str] = []
    for i in indices:
        turn    = history[i]
        role    = turn.get("role", f"turn_{i}")
        content = turn.get("content", "").strip()
        parts.append(f"[{role}] - Step {i}: {content}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Context builders
# ─────────────────────────────────────────────────────────────────────────────

def build_context(
    history:   list[dict],
    step_idx:  int,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, Any]:
    """Tokenise one (context, step) pair for GradNorm scoring.

    Layout fed into apply_chat_template
    ------------------------------------

        <user>
          [role_0]: content_0

          [role_1]: content_1
          ...                         ← context turns from select_context()
        </user>
        <assistant>
          content of history[step_idx] ← NTP loss is computed over these tokens
        </assistant>

    The context turns are serialised as plain text and placed in the user
    slot; the step content is placed verbatim in the assistant slot.
    apply_chat_template wraps both with model-specific special tokens.

    Parameters
    ----------
    history   : full trajectory history.
    step_idx  : step to score.  Must be ≥ 1 (step 0 is the human question).
    tokenizer : HuggingFace tokeniser with a chat template.

    Returns
    -------
    dict with:
        "input_ids" : LongTensor shape (1, seq_len)
        "ctx_len"   : int
            Number of tokens *before* the first step-content token.
            Used in :func:`gradnorm._ntp_loss` to mask context positions.

    Notes
    -----
    ctx_len is computed as the length of the user-turn prefix with
    ``add_generation_prompt=True``, which appends the assistant header tokens
    (e.g. ``<|start_header_id|>assistant<|end_header_id|>\\n\\n`` for Llama 3).
    This correctly accounts for any template-injected tokens surrounding the
    assistant response.

    Qwen3 note: Qwen3's chat template may prepend <think> tokens by default.
    Disable this by calling
        tokenizer.apply_chat_template(..., enable_thinking=False)
    or by patching the template variable before calling build_context.
    """
    ctx_indices  = select_context(history, step_idx)
    ctx_text     = _serialize_turns(history, ctx_indices)
    step_content = history[step_idx].get("content", "").strip()

    user_msg      = {"role": "user",      "content": ctx_text}
    assistant_msg = {"role": "assistant", "content": step_content}

    # ── Full token sequence ───────────────────────────────────────────────
    full_ids = tokenizer.apply_chat_template(
        [user_msg, assistant_msg],
        tokenize              = True,
        add_generation_prompt = False,
        return_tensors        = "pt",
    )  # shape (1, seq_len)

    # ── Context length ────────────────────────────────────────────────────
    # Tokenise only the user turn with add_generation_prompt=True to obtain
    # the exact prefix up to (but not including) the first step-content token.
    prefix_ids = tokenizer.apply_chat_template(
        [user_msg],
        tokenize              = True,
        add_generation_prompt = True,
        return_tensors        = "pt",
    )  # shape (1, ctx_len)

    # return {"full_ids": full_ids, "prefix_ids": prefix_ids}
    ctx_len = prefix_ids["input_ids"].shape[1]

    return {"input_ids": full_ids["input_ids"], "ctx_len": ctx_len}


def custom_build_context(
    history:   list[dict],
    step_idx:  int,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, Any]:
    """PLACEHOLDER — fully custom context builder.

    Drop this in as the ``context_builder`` argument to
    :func:`gradnorm.score_trajectory` when you need a completely different
    tokenisation strategy (e.g. injecting special separator tokens between
    agent turns, per-role system prompts, or ignoring the chat template
    altogether).

    Must return the same dict schema as :func:`build_context`:
        {
            "input_ids": LongTensor  shape (1, seq_len),
            "ctx_len":   int          # tokens before the step content starts
        }
    """
    raise NotImplementedError(
        "custom_build_context is a placeholder. "
        "Implement your context-building strategy here."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def iter_scoreable_steps(trajectory: Trajectory) -> list[int]:
    """Return step indices that should receive a GradNorm score.

    Step 0 is the human question and is never a mistake step, so it is
    excluded.  Returns [1, 2, ..., T-1].
    """
    return list(range(1, len(trajectory.history)))

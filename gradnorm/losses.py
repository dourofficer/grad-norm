import math
import torch
import torch.nn.functional as F
from torch import Tensor

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

def _kl_loss(
    logits:    Tensor,   # (1, seq_len, vocab_size)
    input_ids: Tensor,   # (1, seq_len)
    ctx_len:   int,
) -> Tensor:
    """Mean KL divergence from a uniform distribution over the step tokens.

    KL(uniform || p) = log(C) - (1/C) * sum_c log p_c
                     = log(C) + cross_entropy(logits, uniform_target)

    Since log(C) is a constant that doesn't affect gradients, we drop it and
    compute only the cross-entropy term with a uniform target, which is:

        loss = -mean_over_step_tokens( mean_over_vocab( log_softmax(logits) ) )

    Same autoregressive shift and context masking as _ntp_loss.
    """
    vocab_size = logits.shape[-1]

    # Autoregressive shift
    shift_logits = logits[:, :-1, :].contiguous().float()   # (1, N-1, vocab)

    # Build a boolean mask: True = step token (should contribute to loss)
    mask_end = ctx_len - 1   # same boundary as _ntp_loss
    seq_len_shifted = shift_logits.shape[1]
    mask = torch.zeros(1, seq_len_shifted, dtype=torch.bool, device=logits.device)
    mask[:, mask_end:] = True   # (1, N-1)

    # Log-softmax over vocab, then mean over vocab gives mean log p_c
    log_probs = F.log_softmax(shift_logits, dim=-1)          # (1, N-1, vocab)
    mean_log_p = log_probs.mean(dim=-1)                      # (1, N-1)

    # Average only over unmasked (step) positions
    loss = -mean_log_p[mask].mean()
    breakpoint()
    return loss
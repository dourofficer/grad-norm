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

def _kl_uniform_loss(
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


def _kl_temp_loss(
    logits:      Tensor,   # (1, seq_len, vocab_size)
    input_ids:   Tensor,   # (1, seq_len)  — unused; kept for uniform signature
    ctx_len:     int,
    temperature: float = 2.0,
) -> Tensor:
    """KL divergence KL(p_T ‖ p) where p_T = softmax(logits / T) is a fixed target.

    Derivation
    ----------
    KL(p_T ‖ p)  =  Σ_c p_T_c · log(p_T_c / p_c)
                 =  −H(p_T)  +  CE(p_T, p)

    p_T is detached so −H(p_T) is a constant w.r.t. model parameters and
    drops out of the gradient.  The effective loss is the soft cross-entropy:

        loss = −mean_over_step_tokens( Σ_c p_T_c · log_softmax(logits)_c )

    Intuition
    ---------
    A high temperature flattens p_T toward uniform.  A step whose output
    distribution is *already* close to p_T (i.e. already uncertain) produces
    a small loss and therefore a small gradient norm — the model is not
    surprised.  A peaked, confident step diverges sharply from the flattened
    target, yielding a large loss and a large gradient norm.  Compared with
    _kl_loss (which measures distance from a perfectly uniform target),
    _kl_temp_loss provides a *relative* measure: how much sharper is the
    model than its own temperature-smoothed self.

    """

    shift_logits = logits[:, :-1, :].contiguous().float()  # (1, N-1, vocab)

    with torch.no_grad():
        p_temp = F.softmax(shift_logits / temperature, dim=-1)  # (1, N-1, vocab)

    log_p = F.log_softmax(shift_logits, dim=-1)            # (1, N-1, vocab)

    kl_per_pos = -(p_temp * log_p).sum(dim=-1)             # (1, N-1)

    mask_end = ctx_len - 1
    loss = kl_per_pos[:, mask_end:].mean()
    # breakpoint()
    return loss
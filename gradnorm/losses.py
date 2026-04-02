import math
import torch
import torch.nn.functional as F
from torch import Tensor

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
    return loss
from gradnorm.losses import _kl_loss
import torch
import torch.nn.functional as F

# ── helpers ──────────────────────────────────────────────────────────────────

def make_input(seq_len, vocab_size, logit_val=0.0):
    """Uniform logits by default; input_ids are arbitrary (not used by kl_loss)."""
    logits    = torch.full((1, seq_len, vocab_size), logit_val)
    input_ids = torch.zeros((1, seq_len), dtype=torch.long)
    return logits, input_ids


# ── Test 1: uniform logits → loss == log(vocab_size) ─────────────────────────
# When logits are all equal, softmax is uniform, so log p_c = -log(C) for all c.
# loss = -mean_c log p_c = log(C).  KL itself is 0; our loss is KL + log(C).

def test_uniform_logits():
    vocab_size = 128
    logits, input_ids = make_input(seq_len=10, vocab_size=vocab_size)
    loss = _kl_loss(logits, input_ids, ctx_len=3)
    expected = torch.log(torch.tensor(float(vocab_size)))
    assert torch.isclose(loss, expected, atol=1e-5), \
        f"Expected {expected:.4f}, got {loss:.4f}"
    print("PASS test_uniform_logits")


# ── Test 2: manual reference on a tiny example ────────────────────────────────
# With known logits we can compute the expected value by hand.

def test_manual():
    vocab_size = 4
    seq_len    = 4   # positions 0,1,2,3
    ctx_len    = 2   # mask_end = 1 → only shifted positions 1,2 contribute

    # Two distinct logit vectors for the two contributing positions
    logit_A = torch.tensor([1.0, 2.0, 3.0, 4.0])
    logit_B = torch.tensor([0.5, 0.5, 0.5, 0.5])

    logits = torch.zeros(1, seq_len, vocab_size)
    # Shifted position 1 comes from original position 1 → logits[:, 1, :]
    logits[0, 1, :] = logit_A
    # Shifted position 2 comes from original position 2 → logits[:, 2, :]
    logits[0, 2, :] = logit_B

    input_ids = torch.zeros(1, seq_len, dtype=torch.long)

    loss = _kl_loss(logits, input_ids, ctx_len=ctx_len)

    # Manual: -mean_c log_softmax(logit)[c], averaged over the two step positions
    lp_A = F.log_softmax(logit_A, dim=-1).mean()
    lp_B = F.log_softmax(logit_B, dim=-1).mean()
    expected = -(lp_A + lp_B) / 2

    assert torch.isclose(loss, expected, atol=1e-5), \
        f"Expected {expected:.4f}, got {loss:.4f}"
    print("PASS test_manual")


# ── Test 3: context positions don't affect the loss ───────────────────────────
# Mutating context logits should leave the loss unchanged.

def test_masking():
    vocab_size = 64
    seq_len    = 12
    ctx_len    = 5

    logits, input_ids = make_input(seq_len, vocab_size)
    loss_before = _kl_loss(logits, input_ids, ctx_len=ctx_len)

    # Corrupt context positions (original positions 0..ctx_len-1)
    logits[0, :ctx_len, :] = 999.0
    loss_after = _kl_loss(logits, input_ids, ctx_len=ctx_len)

    assert torch.isclose(loss_before, loss_after, atol=1e-5), \
        f"Context mutation changed loss: {loss_before:.4f} → {loss_after:.4f}"
    print("PASS test_masking")


# ── Test 4: cross-check against torch KLDivLoss ───────────────────────────────
# F.kl_div(log_p, uniform) computes KL(uniform‖p).
# Our loss = KL(uniform‖p) + log(C), so:  kl_loss == torch_kl + log(C)

def test_vs_kldivloss():
    vocab_size = 50
    seq_len    = 8
    ctx_len    = 3

    torch.manual_seed(0)
    logits    = torch.randn(1, seq_len, vocab_size)
    input_ids = torch.zeros(1, seq_len, dtype=torch.long)

    our_loss = _kl_loss(logits, input_ids, ctx_len=ctx_len)

    # Extract only the step positions from the shifted view
    mask_end     = ctx_len - 1
    shift_logits = logits[:, :-1, :].float()          # (1, N-1, vocab)
    step_logits  = shift_logits[:, mask_end:, :]      # (1, n_step, vocab)

    log_probs = F.log_softmax(step_logits, dim=-1)    # (1, n_step, vocab)
    uniform   = torch.full_like(log_probs, 1.0 / vocab_size)

    # F.kl_div with batchmean averages over batch × step tokens
    torch_kl = F.kl_div(log_probs, uniform, reduction="mean") * vocab_size
    # ↑ 'mean' divides by batch*step*vocab; multiply back by vocab to get
    #   the per-token KL (averaged over tokens, summed over vocab classes).

    expected = torch_kl + torch.log(torch.tensor(float(vocab_size)))

    assert torch.isclose(our_loss, expected, atol=1e-4), \
        f"Expected {expected:.4f}, got {our_loss:.4f}"
    print("PASS test_vs_kldivloss")


if __name__ == "__main__":
    # ── run all ───────────────────────────────────────────────────────────────────
    test_uniform_logits()
    test_manual()
    test_masking()
    test_vs_kldivloss()
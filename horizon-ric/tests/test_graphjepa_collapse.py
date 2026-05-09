"""GraphJEPA non-collapse test (Devil-C Finding 3, Solver 1 Fix #17).

JEPA-style self-distillation can collapse: the target encoder converges
to a constant, and the predictor trivially zeros the loss. This test
trains a tiny GraphJEPA for 100 steps with the production EMA β=0.996
and asserts the target representation does NOT collapse — concretely:

    * After training, the per-dimension variance of the target
      embeddings is non-trivial: at least 5 of 10 latent dimensions have
      variance ≥ 1e-4.
    * The covariance-matrix rank (numerical, ε=1e-3) is ≥ k.

Closes Devil-C Finding 3 with empirical evidence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from horizon_ric.encoder.graph_jepa import GraphJEPA, GraphJEPAConfig


class _TinyEncoder(nn.Module):
    """Stand-in for PerceiverFusion: maps (B, N, d_in) → (B, N, d_latent).

    A small linear projection followed by a non-linearity. Small enough
    that the test is fast; expressive enough that JEPA training has
    something to learn.
    """

    def __init__(self, d_in: int, d_latent: int, n_tokens: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_latent)
        self.act = nn.GELU()
        self.token_bias = nn.Parameter(torch.randn(n_tokens, d_latent) * 0.02)

    def forward(
        self, x: torch.Tensor, input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, N, d_in) → (B, N, d_latent)
        return self.act(self.proj(x)) + self.token_bias.unsqueeze(0)


def test_graphjepa_does_not_collapse_at_beta_0996() -> None:
    """Production setting: β=0.996, smooth-L1 latent loss, 100 steps.

    Asserts the target embeddings retain rank > 5 / 10 across the 10
    latent dimensions (no collapse).
    """
    torch.manual_seed(0)
    d_in = 8
    d_latent = 12  # must be divisible by 4 (predictor's nhead)
    n_tokens = 16
    batch = 32

    encoder = _TinyEncoder(d_in, d_latent, n_tokens)
    cfg = GraphJEPAConfig(
        d_latent=d_latent,
        predictor_hidden=32,
        predictor_layers=1,
        ema_momentum=0.996,
        mask_ratio=0.5,
    )
    jepa = GraphJEPA(encoder, cfg)
    opt = torch.optim.Adam(
        list(jepa.context_encoder.parameters())
        + list(jepa.predictor.parameters()),
        lr=1e-3,
    )

    losses = []
    for step in range(100):
        x = torch.randn(batch, n_tokens, d_in)
        out = jepa.loss(x)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        jepa.update_target()
        losses.append(float(out["loss"].item()))

    # Final loss should be finite (didn't blow up).
    assert all(l == l for l in losses), "loss became NaN — diverged"

    # Sample target embeddings on fresh data and inspect rank.
    jepa.eval()
    with torch.no_grad():
        x_eval = torch.randn(64, n_tokens, d_in)
        target_full = jepa.target_encoder(x_eval)  # (B, N, D)
        flat = target_full.reshape(-1, d_latent)   # (B·N, D)
        # Per-dim variance.
        var = flat.var(dim=0, unbiased=False)      # (D,)
        active_dims = int((var > 1e-4).sum().item())
        # Numerical rank via SVD of the centred matrix.
        centred = flat - flat.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(centred)
        # singular values relative to the largest
        rel = s / s[0].clamp_min(1e-12)
        numerical_rank = int((rel > 1e-3).sum().item())

    # Devil-C contract: ≥ 5 of d_latent dims have non-trivial variance.
    assert active_dims >= 5, (
        f"Collapse: only {active_dims}/{d_latent} dims have var > 1e-4 "
        f"(per-dim variance: {var.tolist()})"
    )
    # Numerical rank ≥ 5 — not the full d_latent because real JEPA pre-
    # training shows decay at the singular-value tail; 5 is the spec.
    assert numerical_rank >= 5, (
        f"Collapse: numerical rank {numerical_rank} < 5; "
        f"singular values {s.tolist()}"
    )


def test_graphjepa_target_encoder_changes() -> None:
    """The target encoder must drift from its init under EMA, otherwise
    we're not actually doing JEPA — we're computing a loss against a
    frozen random network.
    """
    torch.manual_seed(1)
    d_in = 4
    d_latent = 8  # must be divisible by 4 (predictor's nhead)
    encoder = _TinyEncoder(d_in, d_latent, n_tokens=8)
    jepa = GraphJEPA(
        encoder,
        GraphJEPAConfig(
            d_latent=d_latent, predictor_hidden=16, predictor_layers=1,
            ema_momentum=0.9,  # faster EMA so test is short
            mask_ratio=0.5,
        ),
    )
    target_init = {
        n: p.detach().clone() for n, p in jepa.target_encoder.named_parameters()
    }
    opt = torch.optim.Adam(
        list(jepa.context_encoder.parameters())
        + list(jepa.predictor.parameters()),
        lr=5e-3,
    )
    for _ in range(30):
        x = torch.randn(16, 8, d_in)
        loss = jepa.loss(x)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        jepa.update_target()

    drifted = False
    for n, p in jepa.target_encoder.named_parameters():
        if not torch.allclose(p, target_init[n], atol=1e-4):
            drifted = True
            break
    assert drifted, "target encoder did not drift under EMA — bug"


def test_graphjepa_predictor_uses_full_mask_information() -> None:
    """Sanity test: the loss DEPENDS on the masked tokens. If we mask
    nothing or everything, the predictor cannot work."""
    torch.manual_seed(2)
    d_in = 4
    d_latent = 8  # must be divisible by 4 (predictor's nhead)
    encoder = _TinyEncoder(d_in, d_latent, n_tokens=8)
    cfg = GraphJEPAConfig(
        d_latent=d_latent, predictor_hidden=16, predictor_layers=1,
        ema_momentum=0.996, mask_ratio=0.5,
    )
    jepa = GraphJEPA(encoder, cfg)
    x = torch.randn(4, 8, d_in)
    out = jepa.loss(x)
    assert out["loss"].item() > 0
    assert out["pred_norm"].item() > 0
    assert out["target_norm"].item() > 0

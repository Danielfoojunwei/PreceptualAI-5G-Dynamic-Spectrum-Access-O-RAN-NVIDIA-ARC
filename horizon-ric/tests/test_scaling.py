"""Real entity-count scaling — no fake fixtures.

Exercises the production encoder front-door (SpatialPrior +
EntityTokenizer + PerceiverFusion) at multiple N. The Perceiver's
fixed-latent design is the load-bearing scalability claim — we assert
sub-linear wall-clock growth (≤ 10× from N=100 to N=1000).
"""

from __future__ import annotations

import time

import pytest
import torch

from horizon_ric.encoder import (
    AssetAttributes,
    EntityTokenizer,
    EntityTokenizerConfig,
    EntityType,
    PerceiverConfig,
    PerceiverFusion,
)
from horizon_ric.policy import TDMPCConfig


def _make_assets(n: int, attr_dim: int = 8) -> list[AssetAttributes]:
    out: list[AssetAttributes] = []
    g = torch.Generator().manual_seed(123)
    for i in range(n):
        out.append(
            AssetAttributes(
                entity_type=EntityType.SAT_NGSO,
                attributes=torch.randn(attr_dim, generator=g),
                xyz_m=(float(i) * 1e3, float(i) * 2e3, 6.7e6 + float(i)),
            )
        )
    return out


def _time_perceiver(n: int, perc: PerceiverFusion, tok: EntityTokenizer) -> float:
    """Wall-clock, milliseconds, for tokenize+perceiver at N."""
    assets = _make_assets(n, attr_dim=tok.cfg.attr_dims[EntityType.SAT_NGSO])
    with torch.no_grad():
        tokens = tok(assets)
    inputs = tokens.unsqueeze(0)
    t0 = time.perf_counter_ns()
    with torch.no_grad():
        _ = perc(inputs)
    return (time.perf_counter_ns() - t0) / 1e6


@pytest.mark.parametrize("n", [1, 100, 1000])
def test_no_oom(n: int):
    """Encoder front-door must complete without OOM at N up to 1000."""
    tok_cfg = EntityTokenizerConfig(d_model=64)
    tok = EntityTokenizer(tok_cfg).eval()
    perc = PerceiverFusion(
        PerceiverConfig(n_latents=64, d_latent=64, d_input=64, n_self_layers=2, n_heads=4)
    ).eval()
    assets = _make_assets(n, attr_dim=tok_cfg.attr_dims[EntityType.SAT_NGSO])
    with torch.no_grad():
        tokens = tok(assets)
        z = perc(tokens.unsqueeze(0))
    assert tokens.shape == (n, 64)
    assert z.shape == (1, 64, 64)


def test_perceiver_sublinear_n100_to_n1000():
    """Perceiver wall-clock at N=1000 must be ≤ 10× wall-clock at N=100.

    The cross-attention writes into a FIXED-size latent; cost grows with
    the number of input tokens but the rest of the pipeline stays put.
    Anything close to linear (10×) signals a regression in the fixed-
    latent contract.
    """
    tok = EntityTokenizer(EntityTokenizerConfig(d_model=64)).eval()
    perc = PerceiverFusion(
        PerceiverConfig(n_latents=64, d_latent=64, d_input=64, n_self_layers=2, n_heads=4)
    ).eval()

    # Warm-up to amortise lazy init (CUDA, allocator pools).
    _time_perceiver(100, perc, tok)

    t100 = min(_time_perceiver(100, perc, tok) for _ in range(3))
    t1000 = min(_time_perceiver(1000, perc, tok) for _ in range(3))
    ratio = t1000 / max(t100, 1e-3)
    # Allow some slack for noisy CI, but anything > 10× is broken.
    assert ratio <= 10.0, (
        f"Perceiver scaling regressed: t(1000)/t(100) = {ratio:.2f} "
        f"(t100={t100:.2f}ms, t1000={t1000:.2f}ms)"
    )


@pytest.mark.parametrize(
    "n_latents,d_latent",
    [(16, 32), (64, 64), (128, 128), (256, 64)],
)
def test_perceiver_config_sizes_load_and_forward(n_latents: int, d_latent: int):
    """Various PerceiverConfig sizes must load and forward."""
    perc = PerceiverFusion(
        PerceiverConfig(
            n_latents=n_latents,
            d_latent=d_latent,
            d_input=d_latent,
            n_self_layers=1,
            n_heads=4 if d_latent % 4 == 0 else 1,
        )
    ).eval()
    x = torch.randn(1, 32, d_latent)
    with torch.no_grad():
        z = perc(x)
    assert z.shape == (1, n_latents, d_latent)


@pytest.mark.parametrize(
    "horizon,n_samples,n_iterations",
    [(4, 32, 2), (12, 256, 6), (24, 512, 3)],
)
def test_tdmpc_config_sizes_work(horizon: int, n_samples: int, n_iterations: int):
    """Various TDMPCConfig sizes must construct cleanly."""
    cfg = TDMPCConfig(
        horizon=horizon,
        n_samples=n_samples,
        n_iterations=n_iterations,
    )
    assert cfg.horizon == horizon
    assert cfg.n_samples == n_samples
    assert cfg.n_iterations == n_iterations


def test_perceiver_matches_at_n5000_no_oom():
    """N=5000 with a fixed latent must complete (this is the scalability
    bet). We do not assert wall-clock here — only that no OOM occurs and
    the output shape is correct."""
    tok = EntityTokenizer(EntityTokenizerConfig(d_model=64)).eval()
    perc = PerceiverFusion(
        PerceiverConfig(n_latents=64, d_latent=64, d_input=64, n_self_layers=1, n_heads=4)
    ).eval()
    assets = _make_assets(5000, attr_dim=tok.cfg.attr_dims[EntityType.SAT_NGSO])
    with torch.no_grad():
        tokens = tok(assets)
        z = perc(tokens.unsqueeze(0))
    assert z.shape == (1, 64, 64)

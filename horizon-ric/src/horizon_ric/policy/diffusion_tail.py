"""Conditional diffusion sampler for tail-risk latent rollouts.

Used by the constraint layer when the policy decision needs the *worst-
case* rollout, not the median. Most decisions are well served by the
TD-MPC2 plan; SLA / EPFD-tail decisions need a sampler over the future
distribution, and a small DDPM is the standard way to get one.

Implementation:
  * Schedule:   linear β ∈ [β_min, β_max] over T diffusion steps.
  * Network:    SiLU MLP + (z_0_cond, action, t_embed) condition.
  * Train obj:  MSE on noise prediction (Ho et al. 2020 "ε-prediction").
  * Sample:     DDPM ancestral sampling at T steps; cheap on Orin Nano
                because the denoiser is a small MLP, not a UNet.

We deliberately operate over the **latent** rollout (B, H, Z) rather
than raw observations — keeps the model tiny and conditions naturally
on the action sequence.

Reference:
    Ho, Jain, Abbeel *Denoising Diffusion Probabilistic Models*,
    NeurIPS 2020 — arXiv:2006.11239.
    Yang et al. *Diffusion World Model* (DWM), ICLR 2024 —
    arXiv:2402.03570 (latent-space DDPM for world models).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    d_latent: int = 128
    d_action: int = 16
    horizon: int = 12
    n_steps: int = 50
    beta_min: float = 1e-4
    beta_max: float = 0.02
    hidden_dim: int = 256


def _make_betas(n_steps: int, beta_min: float, beta_max: float) -> torch.Tensor:
    return torch.linspace(beta_min, beta_max, n_steps)


def _sinusoidal_step_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time-step embedding (Vaswani / NeRF / Ho 2020)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device) / max(half - 1, 1)
    )
    emb = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class _Denoiser(nn.Module):
    """Per-step noise-prediction MLP conditioned on z_0, actions, t."""

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.t_embed_dim = 64
        in_dim = (
            cfg.d_latent * cfg.horizon       # noisy trajectory flattened
            + cfg.d_latent                   # z_0 condition
            + cfg.d_action * cfg.horizon     # action sequence
            + self.t_embed_dim
        )
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.d_latent * cfg.horizon),
        )

    def forward(
        self,
        z_t: torch.Tensor,         # (B, H, d_latent)
        z_cond: torch.Tensor,      # (B, d_latent)
        action_seq: torch.Tensor,  # (B, H, d_action)
        step: torch.Tensor,        # (B,) integer timestep
    ) -> torch.Tensor:
        B = z_t.shape[0]
        flat_z = z_t.reshape(B, -1)
        flat_a = action_seq.reshape(B, -1)
        t_emb = _sinusoidal_step_embedding(step, self.t_embed_dim)
        x = torch.cat([flat_z, z_cond, flat_a, t_emb], dim=-1)
        out = self.net(x)
        return out.reshape(B, self.cfg.horizon, self.cfg.d_latent)


class DiffusionTailSampler(nn.Module):
    """Latent-space DDPM for tail-risk rollouts.

    Two public entry points:
        loss(z0, traj, actions)            ε-prediction MSE for training.
        sample(z0, actions, n_samples)     ancestral sampling, returns
                                            (n_samples, H, d_latent).
    """

    def __init__(self, config: DiffusionConfig | None = None):
        super().__init__()
        self.cfg = config or DiffusionConfig()
        betas = _make_betas(
            self.cfg.n_steps, self.cfg.beta_min, self.cfg.beta_max
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.denoiser = _Denoiser(self.cfg)

    def _q_sample(
        self, x0: torch.Tensor, step: torch.Tensor, noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward (noising) process: q(x_t | x_0)."""
        ab = self.alpha_bars[step].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def loss(
        self,
        z0: torch.Tensor,                # (B, d_latent)
        rollout: torch.Tensor,           # (B, H, d_latent) — clean target
        action_seq: torch.Tensor,        # (B, H, d_action)
    ) -> torch.Tensor:
        """ε-prediction MSE."""
        if rollout.shape[1] != self.cfg.horizon:
            raise ValueError(
                f"rollout horizon {rollout.shape[1]} != cfg.horizon "
                f"{self.cfg.horizon}"
            )
        B = z0.shape[0]
        step = torch.randint(0, self.cfg.n_steps, (B,), device=z0.device)
        noise = torch.randn_like(rollout)
        z_noisy = self._q_sample(rollout, step, noise)
        eps_pred = self.denoiser(z_noisy, z0, action_seq, step)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        z0: torch.Tensor,                # (d_latent,) or (B, d_latent)
        action_seq: torch.Tensor,        # (H, d_action) or (B, H, d_action)
        n_samples: int = 32,
    ) -> torch.Tensor:
        """Ancestral DDPM sampling. Returns (n_samples, H, d_latent)."""
        if z0.ndim == 1:
            z0 = z0.unsqueeze(0)
        if action_seq.ndim == 2:
            action_seq = action_seq.unsqueeze(0)
        if z0.shape[0] != 1 or action_seq.shape[0] != 1:
            raise ValueError(
                "DiffusionTailSampler.sample currently expects a single "
                "(z0, action_seq); broadcast happens internally over n_samples."
            )

        device = z0.device
        z0_b = z0.expand(n_samples, -1).contiguous()
        a_b = action_seq.expand(n_samples, -1, -1).contiguous()
        x = torch.randn(
            n_samples, self.cfg.horizon, self.cfg.d_latent, device=device,
        )
        for t in reversed(range(self.cfg.n_steps)):
            step_t = torch.full(
                (n_samples,), t, device=device, dtype=torch.long,
            )
            eps = self.denoiser(x, z0_b, a_b, step_t)
            beta = self.betas[t]
            alpha = self.alphas[t]
            ab = self.alpha_bars[t]
            mean = (1.0 / alpha.sqrt()) * (
                x - beta / (1.0 - ab).sqrt() * eps
            )
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + beta.sqrt() * noise
            else:
                x = mean
        return x

    def percentile_metric(
        self,
        samples: torch.Tensor,
        score_fn,
        p_pct: float = 95.0,
    ) -> float:
        """Worst-case (high-percentile) tail summary of a per-rollout score."""
        if not 0 < p_pct < 100:
            raise ValueError(f"p_pct must be in (0, 100), got {p_pct}")
        scores = score_fn(samples)
        if scores.ndim != 1:
            raise ValueError("score_fn must return a 1-D per-sample score")
        sorted_scores = scores.sort().values
        idx = int(round((p_pct / 100.0) * (len(sorted_scores) - 1)))
        return float(sorted_scores[idx].item())


__all__ = ["DiffusionConfig", "DiffusionTailSampler"]

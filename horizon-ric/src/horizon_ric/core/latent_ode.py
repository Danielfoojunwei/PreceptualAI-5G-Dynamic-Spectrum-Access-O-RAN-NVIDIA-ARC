"""Latent-ODE for irregularly sampled telemetry (Rubanova 2019).

Two pieces:

    LatentODEPosterior     ODE-RNN encoder over (t_i, x_i) sequences →
                           Gaussian posterior over the initial latent z_0.
    ODESolverHead          Given z_0 and arbitrary query times, integrate
                           the latent dynamics and decode each query.

The ODE solver is a fixed-step explicit RK4 — small but standard,
runs in pure PyTorch (no torchdiffeq dep), and is stable enough for
slot-coarse rollouts at the rApp's 100 ms cadence. For tight-tolerance
research use replace with `torchdiffeq.odeint`.

Why this matters for PreceptualAI: telemetry from cells / UEs is gappy
(handovers, RLF, sleep cycles, satellite-pass blackouts). A vanilla RNN
sees the gaps as zeroes; a Latent-ODE has a posterior at any wall-clock
t even when the observation grid is shredded. Crucially, "query at
arbitrary t" is exactly what the rApp's 5–30 s ahead horizon needs.

References:
    Chen et al. *Neural ODEs*, NeurIPS 2018 — arXiv:1806.07366
    Rubanova, Chen, Duvenaud *Latent ODEs for Irregularly-Sampled Time
        Series*, NeurIPS 2019 — arXiv:1907.03907.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LatentODEConfig:
    obs_dim: int
    latent_dim: int = 32
    hidden_dim: int = 64
    rk4_max_step_s: float = 1.0
    """Cap the per-RK4-step interval; larger gaps are subdivided."""


class _ODEFunc(nn.Module):
    """f(z, t) — the learned latent dynamics."""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, _t: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(z)


def _rk4_step(
    f: _ODEFunc, z: torch.Tensor, dt: float
) -> torch.Tensor:
    """One classical RK4 step of dz/dt = f(z)."""
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate_to(
    f: _ODEFunc, z: torch.Tensor, dt_total: float, max_step_s: float
) -> torch.Tensor:
    """Integrate from z over interval `dt_total`, subdividing if it
    exceeds `max_step_s`. Returns the final state."""
    if dt_total == 0:
        return z
    sign = 1.0 if dt_total > 0 else -1.0
    remaining = abs(dt_total)
    while remaining > 0:
        step = min(remaining, max_step_s)
        z = _rk4_step(f, z, sign * step)
        remaining -= step
    return z


class ODESolverHead(nn.Module):
    """Given z0 and a list of query times, return per-query latents.

    Useful as the "decoder" side of a Latent-ODE: callers pass the posterior
    mean/sample and receive the latent trajectory.
    """

    def __init__(self, config: LatentODEConfig):
        super().__init__()
        self.cfg = config
        self.func = _ODEFunc(config.latent_dim, config.hidden_dim)

    def forward(
        self, z0: torch.Tensor, query_times_s: torch.Tensor
    ) -> torch.Tensor:
        """z0:    (B, latent_dim) initial latent.
        query_times_s: (K,) — query times relative to t=0 of z0.
            Must be monotonic ascending.

        Returns:
            (B, K, latent_dim) trajectory.
        """
        if z0.ndim != 2:
            raise ValueError(f"z0 must be (B, latent_dim), got {tuple(z0.shape)}")
        if query_times_s.ndim != 1:
            raise ValueError(
                f"query_times_s must be 1-D, got {tuple(query_times_s.shape)}"
            )
        if query_times_s.numel() == 0:
            return torch.empty(z0.shape[0], 0, z0.shape[1], device=z0.device)
        # Check monotonic ascending (within numerical tolerance).
        diffs = query_times_s[1:] - query_times_s[:-1]
        if torch.any(diffs < -1e-9):
            raise ValueError("query_times_s must be monotonic ascending")

        out = []
        z = z0
        prev_t = 0.0
        for t in query_times_s.tolist():
            dt = float(t - prev_t)
            z = _integrate_to(self.func, z, dt, self.cfg.rk4_max_step_s)
            out.append(z)
            prev_t = t
        return torch.stack(out, dim=1)


class LatentODEPosterior(nn.Module):
    """ODE-RNN encoder → Gaussian posterior over z_0.

    Iterates the irregular observations (t_i, x_i) backwards in time:
    starts from a zero hidden state, integrates the ODE backward to t_i,
    applies a GRU-like update with the observation x_i, repeats.
    Yields a posterior approximated by a diagonal Gaussian (μ, log σ²).
    """

    def __init__(self, config: LatentODEConfig):
        super().__init__()
        self.cfg = config
        self.func = _ODEFunc(config.latent_dim, config.hidden_dim)
        self.gru = nn.GRUCell(config.obs_dim, config.latent_dim)
        self.mu_head = nn.Linear(config.latent_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.latent_dim, config.latent_dim)

    def forward(
        self, observations: list[tuple[float, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an irregular sequence into (mu, logvar) at the LATEST time.

        observations: list of (t_seconds, x_tensor) where x_tensor has
            shape (B, obs_dim). All x must share batch B; t must be
            ascending.

        Returns:
            (mu, logvar) each of shape (B, latent_dim) at t=t_last.
        """
        if not observations:
            raise ValueError("at least one observation is required")
        first_t, first_x = observations[0]
        if first_x.ndim != 2 or first_x.shape[1] != self.cfg.obs_dim:
            raise ValueError(
                f"x must be (B, obs_dim={self.cfg.obs_dim}); got {tuple(first_x.shape)}"
            )
        B = first_x.shape[0]
        h = torch.zeros(B, self.cfg.latent_dim, device=first_x.device)

        prev_t = first_t
        for t, x in observations:
            if t < prev_t:
                raise ValueError("observations must be in ascending time order")
            dt = float(t - prev_t)
            if dt > 0:
                h = _integrate_to(self.func, h, dt, self.cfg.rk4_max_step_s)
            h = self.gru(x, h)
            prev_t = t

        return self.mu_head(h), self.logvar_head(h)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard VAE reparam — sample z ~ N(μ, σ²) with the gradient trick."""
    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn_like(sigma)
    return mu + sigma * eps


__all__ = [
    "LatentODEConfig",
    "LatentODEPosterior",
    "ODESolverHead",
    "reparameterize",
]

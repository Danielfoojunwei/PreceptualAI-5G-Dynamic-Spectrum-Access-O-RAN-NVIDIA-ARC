"""
Multi-Scale Liquid Time-Constant Cell for 5G Spectrum.

Extends LTCCellGPU with physically-motivated multi-scale dt per layer,
designed for 5G channels with 4 distinct timescales:
  - Layer 0: dt=0.001 (ms)  — Fast fading / small-scale multipath
  - Layer 1: dt=0.1   (s)   — PU arrival/departure (Markov transitions)
  - Layer 2: dt=1.0   (s)   — Mobility-driven handover
  - Layer 3: dt=60.0  (min) — Traffic pattern shifts (busy hour cycles)

Also supports physics-aware tau initialization from known
channel decorrelation times.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from preceptualai.core.ltc_cell_gpu import LTCCellGPU


class MultiScaleLTCCell(LTCCellGPU):
    """
    LTC cell with physics-aware tau initialization for 5G channels.

    Extends LTCCellGPU with:
      - Configurable tau_base initialization from physical decorrelation times
      - Optional learnable dt (instead of fixed)
      - Tau clamping to prevent collapse or explosion
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dt: float = 1.0,
        solver: str = "heun",
        sub_steps: int = 1,
        tau_init_mean: float = 1.0,
        tau_init_std: float = 0.5,
        tau_min: float = 0.01,
        tau_max: float = 1000.0,
        learnable_dt: bool = False,
    ):
        super().__init__(input_dim, hidden_dim, dt, solver, sub_steps)
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Physics-aware tau initialization
        with torch.no_grad():
            self.tau_base.data = torch.abs(
                torch.randn(hidden_dim) * tau_init_std + tau_init_mean
            ).clamp(min=tau_min)

        # Optional learnable dt
        if learnable_dt:
            self._dt_param = nn.Parameter(torch.tensor(float(dt)).log())
        else:
            self._dt_param = None

    @property
    def effective_dt(self) -> float:
        if self._dt_param is not None:
            return self._dt_param.exp().item()
        return self.dt

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Use effective dt (possibly learned)
        if self._dt_param is not None:
            old_dt = self.dt
            self.dt = self._dt_param.exp().item()
            self.sub_dt = self.dt / max(1, int(self.dt / old_dt * self.sub_steps)) if hasattr(self, 'sub_steps') else self.dt

        result = super().forward(x_t, h)

        if self._dt_param is not None:
            self.dt = old_dt

        return result


class MultiScaleLTCEncoder(nn.Module):
    """
    Multi-scale LTC encoder with per-layer temporal resolution.

    Each layer operates at a different timescale, creating a hierarchical
    decomposition that matches the physics of 5G channels.

    Default timescales (5G-specific):
      Layer 0: dt=0.001  — fast dynamics (fast fading, multipath)
      Layer 1: dt=0.1    — medium dynamics (PU Markov transitions)
      Layer 2: dt=1.0    — slow dynamics (mobility, handover)
      Layer 3: dt=60.0   — very slow dynamics (traffic pattern shifts)
    """

    # 5G-motivated default timescales
    DEFAULT_5G_DT = [0.001, 0.1, 1.0, 60.0]
    DEFAULT_5G_TAU = [0.01, 0.5, 5.0, 30.0]  # decorrelation times

    # Satellite-motivated default timescales
    DEFAULT_SATELLITE_DT = [0.01, 1.0, 30.0, 300.0]
    DEFAULT_SATELLITE_TAU = [0.1, 1.0, 30.0, 120.0]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_layers: int = 4,
        dt_per_layer: list = None,
        tau_init_per_layer: list = None,
        solver: str = "heun",
        sub_steps: int = 1,
        domain: str = "5g",  # "5g" or "satellite"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Use domain defaults if not specified
        if dt_per_layer is None:
            if domain == "satellite":
                dt_per_layer = self.DEFAULT_SATELLITE_DT[:num_layers]
            else:
                dt_per_layer = self.DEFAULT_5G_DT[:num_layers]
        if tau_init_per_layer is None:
            if domain == "satellite":
                tau_init_per_layer = self.DEFAULT_SATELLITE_TAU[:num_layers]
            else:
                tau_init_per_layer = self.DEFAULT_5G_TAU[:num_layers]

        # Pad if fewer values than layers
        while len(dt_per_layer) < num_layers:
            dt_per_layer.append(dt_per_layer[-1])
        while len(tau_init_per_layer) < num_layers:
            tau_init_per_layer.append(tau_init_per_layer[-1])

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(MultiScaleLTCCell(
                in_dim, hidden_dim,
                dt=dt_per_layer[i],
                solver=solver,
                sub_steps=sub_steps,
                tau_init_mean=tau_init_per_layer[i],
                tau_init_std=tau_init_per_layer[i] * 0.3,
            ))

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.proj = None
        if hidden_dim != latent_dim:
            self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) input sequence
        Returns:
            z: (B, latent_dim)
        """
        B, T, F = x.shape
        device = x.device

        h = [torch.zeros(B, self.hidden_dim, device=device, dtype=x.dtype)
             for _ in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t, :]
            for i in range(self.num_layers):
                h[i] = self.cells[i](inp, h[i])
                h[i] = self.layer_norms[i](h[i])
                inp = h[i]

        z = h[-1]
        if self.proj is not None:
            z = self.proj(z)
        return z

    def get_tau_stats(self) -> dict:
        """Return tau_base statistics per layer for analysis."""
        stats = {}
        for i, cell in enumerate(self.cells):
            tau = cell.tau_base.detach().cpu()
            stats[f"layer_{i}"] = {
                "dt": cell.dt,
                "tau_mean": tau.mean().item(),
                "tau_std": tau.std().item(),
                "tau_min": tau.min().item(),
                "tau_max": tau.max().item(),
            }
        return stats

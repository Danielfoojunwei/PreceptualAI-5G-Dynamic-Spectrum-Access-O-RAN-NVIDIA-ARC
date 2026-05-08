"""Compositional World-Model substrate (PARADIGMS.md §H2).

The contract:

    ŷ(t+Δ) = f_phys(state, action, t+Δ) + g_θ(latent, action, t+Δ)

DIMENSIONAL CONTRACT (closes Devil-C Finding 1, Finding 7):
    ``f_phys`` returns a quantity in **dB** (e.g. path loss in dB,
    EPFD in dBW/m², RSRP in dBm). The learned residual ``g_θ`` is
    contractually a **dB residual** — i.e. a small additive correction
    *also expressed in dB* with magnitude bounded by ±5 dB at construction.
    Therefore ``ŷ = f_phys + g_θ`` is dimensionally consistent: dB + dB → dB.

    The 5 dB bound is the OPERATIONAL choice that closes the H2 paradigm:
        * 5 dB is < 50 % of typical path-loss-model RMS error (8-10 dB) per
          ITU-R P.1546 / 3GPP TR 38.901, so the residual cannot replace the
          physics — only correct it.
        * 5 dB is > 1 σ of typical small-scale fading (4 dB log-normal),
          so the residual head retains enough capacity to fit measured
          telemetry shifts.

    For LINEAR-power composition (sum of incoherent contributions, e.g.
    summing two interferer PFDs in linear units before going back to dB),
    callers must use ``compose_linear_power(f_dB, g_dB)`` which performs
    ``10·log10(10^(f/10) + 10^(g/10))``. That is the correct combinator
    for SUM-of-powers; ``f + g`` in dB is a *correction*, not a sum.

Three guarantees:

  1. **Sample efficiency** — the residual only fits ‖y − f_phys‖, which is
     O(dB) where the physics is O(100 dB). Orders of magnitude less data
     needed than a from-scratch black-box.
  2. **Distribution-shift safety** — when the residual norm exceeds a
     trained envelope, the planner should fall back to physics-only.
     `should_fallback()` returns the binary signal; the gate-aware
     planner consumes it.
  3. **Auditability** — the rApp emits `(physics_pred, residual_pred,
     uncertainty)` per decision; operators see exactly what the learned
     part is contributing.

References:
    Compositional World Model: PARADIGMS.md §H2 (this repo).
    Universal Differential Equations (Rackauckas et al. 2020,
        arXiv:2001.04385) — analogous formulation in PDE land.
    PI-DeepONet (Lu et al., Nature MI 2021) — physics + neural operator
        composition.
    THEOREMS.md §1 — Compositional dB-correctness theorem (closes Devil-C #1).
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from horizon_ric.core.cfc_core import CfCCell, CfCConfig

# Operational bound on the dB residual. Closes the dimensional contract
# stated in the module docstring + THEOREMS.md §1.
DEFAULT_RESIDUAL_DB_BOUND: float = 5.0


def compose_linear_power(f_dB: torch.Tensor, g_dB: torch.Tensor) -> torch.Tensor:
    """Combine two dB quantities as a sum of LINEAR powers.

    For aggregate interference / sum-of-incoherent-emitters scenarios, the
    correct combinator in dB-space is::

        result_dB = 10 · log10( 10^(f_dB/10) + 10^(g_dB/10) )

    which is **NOT** ``f_dB + g_dB`` (that would multiply linear powers).

    This is the same combinator EPFD aggregation already uses (see
    `planner/physics/epfd.py:175`); we surface it here so callers wishing
    for sum-of-powers behaviour have one canonical implementation.

    Args:
        f_dB, g_dB: tensors of the same shape, both in dB.

    Returns:
        Tensor in dB representing 10·log10(linear sum of the two inputs).
    """
    if f_dB.shape != g_dB.shape:
        raise ValueError(
            f"compose_linear_power: shape mismatch "
            f"{tuple(f_dB.shape)} vs {tuple(g_dB.shape)}"
        )
    # logaddexp gives log(exp(a) + exp(b)) numerically stably; we scale by
    # ln(10)/10 so the inputs/outputs are in dB units.
    scale = math.log(10.0) / 10.0
    return torch.logaddexp(f_dB * scale, g_dB * scale) / scale


class PhysicsResidualHead(nn.Module):
    """Wraps an analytic physics function with a learned dB residual.

    Dimensional contract (see module docstring + THEOREMS.md §1):
        ``physics_fn(state)`` returns a tensor in **dB**.
        The learned residual is also in **dB**, bounded to
        ``±residual_db_bound`` (default ±5 dB) by a tanh saturation at
        construction time. Composition is ``total_dB = physics_dB +
        residual_dB`` — both legs share the same dB unit, so the addition
        is dimensionally consistent.

    Args:
        physics_fn: callable accepting a (B, state_dim) tensor and
            returning a (B, output_dim) tensor in dB. NO grad through
            this; it is the analytic prior.
        state_dim, latent_dim: input dims for the residual.
        output_dim: shared with physics_fn output.
        residual_db_bound: hard cap on |delta_dB| (in dB). Default 5 dB.
            See module docstring for justification.
        ood_threshold_sigma: gate threshold in standard deviations of
            the running residual-norm distribution.
    """

    def __init__(
        self,
        physics_fn: Callable[[torch.Tensor], torch.Tensor],
        state_dim: int,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        residual_db_bound: float = DEFAULT_RESIDUAL_DB_BOUND,
        ood_threshold_sigma: float = 3.0,
    ):
        super().__init__()
        if residual_db_bound <= 0:
            raise ValueError(
                f"residual_db_bound must be > 0 dB, got {residual_db_bound}"
            )
        self._physics_fn = physics_fn
        self.output_dim = output_dim
        self.residual_db_bound = float(residual_db_bound)
        self.ood_threshold_sigma = ood_threshold_sigma

        self.cell = CfCCell(
            CfCConfig(
                input_dim=state_dim + latent_dim,
                hidden_dim=hidden_dim,
            )
        )
        self.delta_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

        # Running mean/std of the residual norm — used for OOD gating.
        self.register_buffer("residual_norm_mean", torch.tensor(0.0))
        self.register_buffer("residual_norm_std", torch.tensor(1.0))
        self.register_buffer("residual_norm_count", torch.tensor(0.0))

    def forward(
        self,
        state: torch.Tensor,
        latent: torch.Tensor,
        h: torch.Tensor | None = None,
        dt: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Run the compositional prediction.

        state:  (B, state_dim) — physics inputs (e.g. range, frequency).
        latent: (B, latent_dim) — encoder context.
        h:      (B, hidden_dim) prior CfC hidden state, or None to start fresh.
        dt:     time step seconds.

        Returns dict with:
            physics_pred  (B, output_dim)  — pure analytic output (dB)
            residual_pred (B, output_dim)  — learned correction (dB), |δ|≤bound
            total_pred    (B, output_dim)  — physics + residual (dB)
            logvar        (B, output_dim)  — uncertainty (log-variance)
            hidden        (B, hidden_dim)  — next CfC hidden state
            should_fallback  bool tensor (B,) — True when residual is OOD
        """
        with torch.no_grad():
            physics_pred = self._physics_fn(state)
        if physics_pred.shape != (state.shape[0], self.output_dim):
            raise ValueError(
                f"physics_fn returned {tuple(physics_pred.shape)}, expected "
                f"({state.shape[0]}, {self.output_dim})"
            )

        if h is None:
            h = self.cell.init_hidden(state.shape[0], device=state.device)
        x = torch.cat([state, latent], dim=-1)
        h_next = self.cell(x, h, dt=dt)

        delta_raw = self.delta_head(h_next)
        # Bound the residual to ±residual_db_bound (in dB) — ensures the
        # learned head can never *replace* physics, only correct it. The
        # tanh saturation gives a smooth, differentiable bound; the
        # multiplier IS the dB cap (no further scaling by std() of the
        # physics — that conflated bounded units with the relative scale of
        # the physics output, which is the dimensional bug Devil-C flagged).
        delta = self.residual_db_bound * torch.tanh(delta_raw)

        logvar = self.logvar_head(h_next)

        # dB + dB → dB. Both legs share units (see THEOREMS.md §1).
        total = physics_pred + delta

        residual_norm = delta.detach().norm(dim=-1)  # (B,)
        if self.training:
            self._update_running_stats(residual_norm)
        z_score = (residual_norm - self.residual_norm_mean) / (
            self.residual_norm_std + 1e-6
        )
        should_fallback = z_score > self.ood_threshold_sigma

        return {
            "physics_pred": physics_pred,
            "residual_pred": delta,
            "total_pred": total,
            "logvar": logvar,
            "hidden": h_next,
            "should_fallback": should_fallback,
        }

    def compose_linear_power(
        self, physics_pred: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Linear-power combinator for sum-of-emitters (NOT sum-of-corrections).

        Use this when ``delta`` represents another *independent emitter*
        contributing power on the same channel (e.g. an aggregate
        interference contribution learned by the residual head trained
        with that target). Most callers should NOT use this — the
        default contract is dB + dB (correction).

        See module-level :func:`compose_linear_power` for the closed
        form. This method is just the bound-method convenience wrapper.
        """
        return compose_linear_power(physics_pred, delta)

    def _update_running_stats(self, residual_norm: torch.Tensor) -> None:
        """Welford running mean/std for the residual norm — used in OOD gate."""
        with torch.no_grad():
            batch_n = float(residual_norm.numel())
            new_count = self.residual_norm_count + batch_n
            batch_mean = residual_norm.mean()
            delta = batch_mean - self.residual_norm_mean
            new_mean = self.residual_norm_mean + delta * (batch_n / new_count)
            batch_var = residual_norm.var(unbiased=False)
            new_std = torch.sqrt(
                (
                    self.residual_norm_count * (self.residual_norm_std ** 2)
                    + batch_n * batch_var
                    + (delta ** 2) * (self.residual_norm_count * batch_n / new_count)
                )
                / new_count.clamp_min(1.0)
            )
            self.residual_norm_mean = new_mean
            self.residual_norm_std = new_std
            self.residual_norm_count = new_count


__all__ = [
    "PhysicsResidualHead",
    "compose_linear_power",
    "DEFAULT_RESIDUAL_DB_BOUND",
]

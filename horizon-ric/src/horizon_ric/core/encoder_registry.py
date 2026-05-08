"""Encoder registry — pluggable temporal encoders.

The world model needs a `(B, T, D)` → `(B, T, D)` temporal encoder. Several
families are appropriate (CfC, Liquid-S4, Latent-ODE, identity, …) and the
right choice depends on edge-vs-cloud constraints and the cadence of the
data stream. This registry lets the rApp swap encoders by config name only.

Mirror of :mod:`horizon_ric.io.registry`'s connector pattern. Two ways
to register an encoder:

1. **In-process** — call :func:`register_encoder("cfc", CfCEncoder)` at
   import time. Built-in encoders use this path.
2. **Plugin entry-point** — declare in ``pyproject.toml``::

       [project.entry-points."horizon_ric.encoders"]
       my_encoder = "my_pkg.module:MyEncoder"

Every registered encoder MUST be a ``torch.nn.Module`` whose ``forward``
takes a tensor of shape ``(B, T, D_in)`` and returns ``(B, T, D_out)``.
``D_out`` defaults to ``D_in`` (this is the contract verified by tests).

The factory ``get_encoder(name, **kwargs)`` returns a fresh instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, ClassVar

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─── Encoder contract ────────────────────────────────────────────────────


class TemporalEncoder(nn.Module):
    """ABC: any registered encoder must be a TemporalEncoder.

    The contract is shape preservation: input ``(B, T, D)`` → output
    ``(B, T, D)``. Implementations may use any internal mechanism (RNN,
    SSM, Transformer, ODE, …).
    """

    d_model: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover — overridden
        raise NotImplementedError


# ─── Built-in encoders (thin wrappers around existing modules) ───────────


class IdentityEncoder(TemporalEncoder):
    """Pass-through. Useful as a baseline and as a sanity test."""

    def __init__(self, d_model: int = 64, **_: Any):
        super().__init__()
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"input must be (B,T,D); got {tuple(x.shape)}")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"identity encoder configured for D={self.d_model}, "
                f"got input D={x.shape[-1]}"
            )
        return x


class CfCEncoder(TemporalEncoder):
    """Wraps :class:`horizon_ric.core.cfc_core.CfCCell` into the
    ``(B,T,D)`` temporal encoder contract."""

    def __init__(self, d_model: int = 64, hidden_dim: int | None = None, **_: Any):
        super().__init__()
        from horizon_ric.core.cfc_core import CfCCell, CfCConfig

        self.d_model = int(d_model)
        hd = int(hidden_dim or d_model)
        self.cell = CfCCell(CfCConfig(input_dim=d_model, hidden_dim=hd))
        # Project hidden state back to d_model to preserve shape.
        self.out = nn.Linear(hd, d_model) if hd != d_model else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"input must be (B,T,D); got {tuple(x.shape)}")
        B, T, D = x.shape
        h = self.cell.init_hidden(B, device=x.device)
        outs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outs.append(self.out(h))
        return torch.stack(outs, dim=1)


class LiquidS4Encoder(TemporalEncoder):
    """Wraps :class:`horizon_ric.core.liquid_s4.LiquidS4` into the
    ``(B,T,D)`` temporal encoder contract."""

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 2,
        **_: Any,
    ):
        super().__init__()
        from horizon_ric.core.liquid_s4 import LiquidS4, LiquidS4Config

        self.d_model = int(d_model)
        self.net = LiquidS4(
            LiquidS4Config(
                d_model=int(d_model),
                d_state=int(d_state),
                n_layers=int(n_layers),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"input must be (B,T,D); got {tuple(x.shape)}")
        return self.net(x)


class LatentODEEncoder(TemporalEncoder):
    """Wraps the ODE solver head as a ``(B,T,D)`` encoder.

    We treat each timestep as a query at integer seconds; the ODE rolls
    forward from the initial latent. Useful when the rApp needs the
    irregular-time semantics of Latent-ODE.
    """

    def __init__(
        self,
        d_model: int = 64,
        hidden_dim: int = 64,
        rk4_max_step_s: float = 0.5,
        **_: Any,
    ):
        super().__init__()
        from horizon_ric.core.latent_ode import LatentODEConfig, ODESolverHead

        self.d_model = int(d_model)
        self.solver = ODESolverHead(
            LatentODEConfig(
                latent_dim=int(d_model),
                hidden_dim=int(hidden_dim),
                obs_dim=int(d_model),
                rk4_max_step_s=float(rk4_max_step_s),
            )
        )
        # Take the first timestep as the initial latent z0.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"input must be (B,T,D); got {tuple(x.shape)}")
        B, T, D = x.shape
        z0 = x[:, 0, :]  # (B, D)
        # Query at t = 1, 2, ..., T (skip t=0 which is z0 itself).
        if T <= 1:
            return z0.unsqueeze(1)
        query_t = torch.arange(1, T, dtype=z0.dtype, device=z0.device).float()
        traj = self.solver(z0, query_t)  # (B, T-1, D)
        return torch.cat([z0.unsqueeze(1), traj], dim=1)


# ─── Registry implementation ─────────────────────────────────────────────


@dataclass(frozen=True)
class EncoderEntry:
    """A registered encoder factory."""

    name: str
    factory: Callable[..., TemporalEncoder]


class EncoderRegistry:
    """name → factory lookup for temporal encoders."""

    _ENTRY_POINT_GROUP: ClassVar[str] = "horizon_ric.encoders"

    def __init__(self):
        self._entries: dict[str, EncoderEntry] = {}
        self._loaded_entry_points = False

    def register(self, name: str, factory: Callable[..., TemporalEncoder]) -> None:
        if not callable(factory):
            raise TypeError(f"factory for {name!r} must be callable, got {factory!r}")
        if name in self._entries and self._entries[name].factory is not factory:
            raise ValueError(
                f"encoder {name!r} already registered to {self._entries[name].factory!r}"
            )
        self._entries[name] = EncoderEntry(name=name, factory=factory)
        logger.debug("registered encoder %s -> %s", name, factory)

    def get(self, name: str, **kwargs: Any) -> TemporalEncoder:
        self._maybe_load_entry_points()
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(
                f"unknown encoder {name!r}; registered: {sorted(self._entries)}"
            )
        instance = entry.factory(**kwargs)
        if not isinstance(instance, nn.Module):
            raise TypeError(
                f"encoder factory for {name!r} returned non-Module {type(instance)}"
            )
        return instance

    def list_encoders(self) -> list[str]:
        self._maybe_load_entry_points()
        return sorted(self._entries)

    def _maybe_load_entry_points(self) -> None:
        if self._loaded_entry_points:
            return
        self._loaded_entry_points = True
        try:
            eps = metadata.entry_points(group=self._ENTRY_POINT_GROUP)
        except Exception:  # pragma: no cover
            return
        for ep in eps:
            try:
                obj = ep.load()
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to load encoder entry point %s: %s", ep.name, exc)
                continue
            if callable(obj):
                self.register(ep.name, obj)


registry = EncoderRegistry()


def register_encoder(name: str, factory: Callable[..., TemporalEncoder]) -> None:
    registry.register(name, factory)


def get_encoder(name: str, **kwargs: Any) -> TemporalEncoder:
    return registry.get(name, **kwargs)


def list_encoders() -> list[str]:
    return registry.list_encoders()


# ─── Built-ins registration ─────────────────────────────────────────────

register_encoder("identity", IdentityEncoder)
register_encoder("cfc", CfCEncoder)
register_encoder("liquid_s4", LiquidS4Encoder)
register_encoder("latent_ode", LatentODEEncoder)


__all__ = [
    "CfCEncoder",
    "EncoderEntry",
    "EncoderRegistry",
    "IdentityEncoder",
    "LatentODEEncoder",
    "LiquidS4Encoder",
    "TemporalEncoder",
    "get_encoder",
    "list_encoders",
    "register_encoder",
    "registry",
]

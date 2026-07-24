"""Differential privacy for federated DSA — a real (ε, δ) accountant, no torch.

Robust aggregation and secure aggregation bound *poisoning* and hide individual
updates, but neither *bounds privacy leakage*: a curious server (or a membership-
inference adversary) can still learn whether a particular client's data shaped the
global model (O-RAN WG11 ML03 model-inversion / ML04 membership-inference; the
"no differential-privacy accountant" GAP in ``docs/THREAT_MODEL.md`` §5).

This module closes that gap with **DP-FedAvg** and a real **Rényi-DP accountant**,
in the lineage of the NTU/DTC work that pairs unlearning with differential privacy
(Liu, Jiang, Lam et al., *Efficient Federated Unlearning with Adaptive Differential
Privacy Preservation*, IEEE BigData 2024). The mechanism and accounting are the
standard, peer-reviewed constructions:

* **Per-client L2 clipping** to a public bound ``C`` caps every client update.
  Under the conservative replace-one client adjacency used here, changing one
  client can move the clipped sum by at most ``2C``.
* **Gaussian mechanism**: add ``N(0, (z·2C)²)`` to the summed updates; the noise
  multiplier ``z`` is defined relative to that replace-one sensitivity.
* **Rényi-DP accounting** (Mironov, CSF 2017): the Gaussian mechanism is
  ``(α, α / (2 z²))``-RDP; RDP composes *additively* over rounds; convert to
  ``(ε, δ)``-DP via ``ε = ε_RDP(α) + log(1/δ)/(α−1)`` minimised over ``α``.

Everything is numpy / pure Python and exact for the full-participation (no
subsampling) regime DP-FedAvg uses here; privacy amplification by subsampling is a
documented tightening, not implemented (we report the conservative bound).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# A standard grid of Rényi orders for the accountant (Abadi/Opacus use a similar set).
DEFAULT_ORDERS: tuple[float, ...] = (
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
)


def l2_clip(update: np.ndarray, clip_norm: float) -> np.ndarray:
    """Scale ``update`` so its L2 norm is at most ``clip_norm`` (DP sensitivity bound)."""
    if clip_norm <= 0:
        raise ValueError("clip_norm must be > 0")
    norm = float(np.linalg.norm(update.ravel()))
    if norm <= clip_norm or norm == 0.0:
        return update.astype(np.float64, copy=True)
    return (update.astype(np.float64) * (clip_norm / norm))


def _gaussian_rdp(orders: Sequence[float], noise_multiplier: float) -> dict[float, float]:
    """RDP of one Gaussian-mechanism step: (α, α/(2 z²)) for each order α."""
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be > 0")
    z2 = noise_multiplier * noise_multiplier
    return {a: a / (2.0 * z2) for a in orders}


def rdp_to_dp_epsilon(rdp: dict[float, float], delta: float) -> tuple[float, float]:
    """Convert accumulated RDP to (ε, δ)-DP (Mironov 2017, Prop. 3).

    Returns ``(epsilon, best_order)`` minimising ``rdp(α) + log(1/δ)/(α−1)``.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    best_eps = math.inf
    best_alpha = math.nan
    for a, r in rdp.items():
        if a <= 1.0:
            continue
        eps = r + math.log(1.0 / delta) / (a - 1.0)
        if eps < best_eps:
            best_eps, best_alpha = eps, a
    return best_eps, best_alpha


@dataclass
class RDPAccountant:
    """Accumulates Rényi-DP across composed Gaussian-mechanism steps.

    Each :meth:`step` records one DP-FedAvg round at noise multiplier ``z``; RDP
    composes additively, so the running privacy cost is exact under composition.
    """

    orders: tuple[float, ...] = DEFAULT_ORDERS
    _rdp: dict[float, float] = field(default_factory=dict)
    steps: int = 0

    def __post_init__(self) -> None:
        if not self._rdp:
            self._rdp = {a: 0.0 for a in self.orders}

    def step(self, noise_multiplier: float, *, count: int = 1) -> None:
        """Account ``count`` Gaussian rounds at the given noise multiplier."""
        per = _gaussian_rdp(self.orders, noise_multiplier)
        for a in self.orders:
            self._rdp[a] += count * per[a]
        self.steps += count

    def get_epsilon(self, delta: float) -> float:
        """Current (ε, δ)-DP guarantee for the composed mechanism."""
        eps, _ = rdp_to_dp_epsilon(self._rdp, delta)
        return eps

    def get_epsilon_and_order(self, delta: float) -> tuple[float, float]:
        return rdp_to_dp_epsilon(self._rdp, delta)


@dataclass(frozen=True)
class DPConfig:
    """DP-FedAvg knobs for conservative replace-one client adjacency.

    ``clip_norm`` must be selected independently of the private cohort (for
    example from a public calibration set or a pre-registered deployment
    policy). The Gaussian noise standard deviation on the *sum* is
    ``noise_multiplier * 2 * clip_norm``.
    """

    clip_norm: float = 1.0
    noise_multiplier: float = 1.0


def dp_fedavg(
    updates: Sequence[np.ndarray],
    cfg: DPConfig,
    *,
    rng: np.random.Generator,
    accountant: Optional[RDPAccountant] = None,
) -> np.ndarray:
    """One DP-FedAvg aggregation: clip each client update, sum, add Gaussian noise,
    average. If ``accountant`` is given, charge one Gaussian round to it.

    Under replace-one client adjacency, the sensitivity of the *sum* is at most
    ``2 * clip_norm``: one clipped vector of norm at most C can be replaced by
    another clipped vector of norm at most C. Noise
    ``N(0, (z·2C)²)`` per coordinate therefore gives the
    ``(α, α/2z²)``-RDP step the accountant records.
    """
    flats = [l2_clip(np.asarray(u, dtype=np.float64).ravel(), cfg.clip_norm) for u in updates]
    n = len(flats)
    if n == 0:
        raise ValueError("dp_fedavg needs at least one update")
    summed = np.sum(flats, axis=0)
    replace_one_sensitivity = 2.0 * cfg.clip_norm
    sigma = cfg.noise_multiplier * replace_one_sensitivity
    noise = rng.normal(0.0, sigma, size=summed.shape)
    if accountant is not None:
        accountant.step(cfg.noise_multiplier)
    return (summed + noise) / n


__all__ = [
    "DEFAULT_ORDERS",
    "l2_clip",
    "rdp_to_dp_epsilon",
    "RDPAccountant",
    "DPConfig",
    "dp_fedavg",
]

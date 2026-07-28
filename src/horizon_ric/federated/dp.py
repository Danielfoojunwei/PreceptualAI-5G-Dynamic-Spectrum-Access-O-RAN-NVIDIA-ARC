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
  The *adjacency convention* fixes the sensitivity of the clipped sum:
  ``replace_one`` (swap one client's data for another's) gives ``2C``;
  ``add_remove`` (add or drop one client) gives ``C``. See :class:`Adjacency`.
* **Gaussian mechanism**: add ``N(0, (z·Δ)²)`` to the summed updates, where
  ``Δ`` is that sensitivity; the noise multiplier ``z`` is defined relative to it.
* **Rényi-DP accounting** (Mironov, CSF 2017): the Gaussian mechanism is
  ``(α, α / (2 z²))``-RDP; RDP composes *additively* over rounds; convert to
  ``(ε, δ)``-DP via ``ε = ε_RDP(α) + log(1/δ)/(α−1)`` minimised over ``α``.

Everything is numpy / pure Python and exact for the full-participation (no
subsampling) regime DP-FedAvg uses here; privacy amplification by subsampling is a
documented tightening, not implemented (we report the conservative bound).

Reading the utility numbers honestly
------------------------------------
``epsilon`` is a property of ``(z, rounds, delta)`` **only**. The clip norm and
the round count are free utility knobs *at fixed ε*, and getting them wrong is a
configuration defect, not a privacy/utility trade-off. Two facts drive the
tuning, both measured on the real DeepMIMO federation:

1. **A clip that never binds is pure loss.** If ``C`` is far above the true
   update norms, the mechanism still injects ``z·Δ(C)`` noise — calibrated to a
   sensitivity that the data never attains. Shrinking ``C`` to the actual scale
   of the updates cuts σ proportionally at *identical* ε.
2. **Rounds are not free.** For a fixed ε over ``R`` composed Gaussian rounds,
   ``z ∝ √R``, while the averaging only divides the accumulated noise by ``R``
   — so total injected noise grows like ``√R``. Running a model that converges
   in ~10 rounds for 40 rounds pays ~2× the noise for nothing.

The strongest form of (2) is :func:`dp_sufficient_statistic`: when the learning
problem is a least-squares fit whose design matrix is *publicly* whitened, the
entire task is one 6-dimensional sum, releasable **once** — ``R = 1``, the
smallest ``z`` any ε permits.
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


def solve_noise_multiplier(
    rounds: int,
    target_epsilon: float,
    delta: float,
    *,
    orders: tuple[float, ...] = DEFAULT_ORDERS,
    tol: float = 1e-12,
) -> float:
    """Smallest ``z`` whose ``rounds``-fold composition costs at most ``target_epsilon``.

    ``epsilon`` is monotonically decreasing in ``z``, so a bisection is exact to
    machine precision. This is the honest way to compare round counts: it holds
    the *certified guarantee* fixed and lets utility move, instead of holding a
    hard-coded ``z`` fixed and letting the guarantee move.

    Because one Gaussian round is ``(α, α/2z²)``-RDP and RDP composes additively,
    ``ε(z, R) ≈ R·α/(2z²) + log(1/δ)/(α−1)`` minimised over ``α``, giving the
    familiar ``z ∝ √R`` at fixed ``ε``.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be > 0")

    def eps_at(z: float) -> float:
        acct = RDPAccountant(orders=orders)
        acct.step(z, count=rounds)
        return acct.get_epsilon(delta)

    lo, hi = 1e-9, 1.0
    while eps_at(hi) > target_epsilon:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError("target_epsilon unreachable")
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if eps_at(mid) > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


class Adjacency:
    """Client-level adjacency conventions and their sum-sensitivity multipliers.

    The multiplier is what turns a clip norm ``C`` into the Gaussian
    mechanism's sensitivity ``Δ``. It is *not* a tuning knob: it names which
    neighbouring-dataset relation the reported ``ε`` is a guarantee about.

    * ``REPLACE_ONE`` (``Δ = 2C``) — one client's contribution is swapped for a
      different one. Strictly the stronger statement, and the default here.
    * ``ADD_REMOVE`` (``Δ = C``) — one client joins or leaves. The convention
      most DP-FedAvg papers report against. Halving σ at the same numeric ``ε``
      is **not** a free win: it is a genuinely weaker guarantee.
    """

    REPLACE_ONE = "replace_one"
    ADD_REMOVE = "add_remove"
    MULTIPLIER = {REPLACE_ONE: 2.0, ADD_REMOVE: 1.0}


@dataclass(frozen=True)
class DPConfig:
    """DP-FedAvg knobs.

    ``clip_norm`` must be selected independently of the private cohort (for
    example from a public calibration set or a pre-registered deployment
    policy); picking it from the private updates would itself leak. The Gaussian
    noise standard deviation on the *sum* is
    ``noise_multiplier * Adjacency.MULTIPLIER[adjacency] * clip_norm``.
    """

    clip_norm: float = 1.0
    noise_multiplier: float = 1.0
    adjacency: str = Adjacency.REPLACE_ONE

    @property
    def sensitivity(self) -> float:
        """Sum-sensitivity ``Δ`` implied by the clip norm and adjacency."""
        try:
            mult = Adjacency.MULTIPLIER[self.adjacency]
        except KeyError:
            raise ValueError(f"unknown adjacency {self.adjacency!r}") from None
        return mult * self.clip_norm

    @property
    def sigma(self) -> float:
        """Gaussian standard deviation added to the *sum* of clipped updates."""
        return self.noise_multiplier * self.sensitivity


def dp_fedavg(
    updates: Sequence[np.ndarray],
    cfg: DPConfig,
    *,
    rng: np.random.Generator,
    accountant: Optional[RDPAccountant] = None,
) -> np.ndarray:
    """One DP-FedAvg aggregation: clip each client update, sum, add Gaussian noise,
    average. If ``accountant`` is given, charge one Gaussian round to it.

    The sensitivity of the *sum* is ``cfg.sensitivity``: under replace-one
    adjacency a clipped vector of norm at most ``C`` can be swapped for another
    of norm at most ``C`` (``Δ = 2C``); under add/remove-one a single clipped
    vector appears or disappears (``Δ = C``). Noise ``N(0, (z·Δ)²)`` per
    coordinate gives the ``(α, α/2z²)``-RDP step the accountant records — and
    note that the accountant's number depends on ``z`` alone, so the clip norm
    is a pure utility knob at fixed ``ε``.
    """
    flats = [l2_clip(np.asarray(u, dtype=np.float64).ravel(), cfg.clip_norm) for u in updates]
    n = len(flats)
    if n == 0:
        raise ValueError("dp_fedavg needs at least one update")
    summed = np.sum(flats, axis=0)
    noise = rng.normal(0.0, cfg.sigma, size=summed.shape)
    if accountant is not None:
        accountant.step(cfg.noise_multiplier)
    return (summed + noise) / n


def dp_sufficient_statistic(
    contributions: Sequence[np.ndarray],
    cfg: DPConfig,
    *,
    rng: np.random.Generator,
    accountant: Optional[RDPAccountant] = None,
) -> np.ndarray:
    """One-shot Gaussian release of a summed per-client sufficient statistic.

    When the federated task is a least-squares fit ``min_w ‖Aw − b‖²`` whose
    design matrix has been whitened by a **public** preprocessing — so that
    ``AᵀA / N = I`` — the normal equations collapse to ``w* = Aᵀb / N``. That is
    a single additive sum over clients, ``Σ_c A_cᵀ b_c / N``. It can therefore be
    released **once** under the Gaussian mechanism rather than re-released every
    round, which is the ``R = 1`` corner of the ``z ∝ √R`` composition law: the
    smallest noise multiplier the target ``ε`` allows.

    This is not a weaker guarantee — it is the same Gaussian mechanism with the
    same accountant, charged once instead of ``R`` times. It is the right thing
    to do whenever the sufficient statistic is small and the whitening is public.

    Args:
        contributions: each client's ``A_cᵀ b_c / N`` vector.
        cfg: clip norm (a public bound on ‖contribution‖), noise multiplier,
            adjacency.

    Returns the noisy summed statistic, i.e. the DP estimate of ``w*``.
    """
    flats = [l2_clip(np.asarray(c, dtype=np.float64).ravel(), cfg.clip_norm)
             for c in contributions]
    if not flats:
        raise ValueError("dp_sufficient_statistic needs at least one contribution")
    summed = np.sum(flats, axis=0)
    noise = rng.normal(0.0, cfg.sigma, size=summed.shape)
    if accountant is not None:
        accountant.step(cfg.noise_multiplier)
    return summed + noise


__all__ = [
    "DEFAULT_ORDERS",
    "l2_clip",
    "rdp_to_dp_epsilon",
    "solve_noise_multiplier",
    "Adjacency",
    "RDPAccountant",
    "DPConfig",
    "dp_fedavg",
    "dp_sufficient_statistic",
]

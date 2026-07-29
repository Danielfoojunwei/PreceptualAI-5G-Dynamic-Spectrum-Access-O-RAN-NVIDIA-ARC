"""Realistic federated poisoning attacks (NOT a strawman).

The previous benchmark used a strawman Byzantine update (``np.full(dim, 50.0)``)
that any robust aggregator trivially rejects — a tautology. This module
implements two *real* attacks from the literature that are specifically designed
to defeat robust aggregation, so the defences are graded honestly (and shown to
FAIL where they actually fail):

* :func:`alie_attack` — **A Little Is Enough** (Baruch, Gilad-Baruch, Goldberg,
  NeurIPS 2019). The malicious clients move the aggregate by a *small* amount
  per coordinate — just inside the variance envelope of the benign updates — so
  that coordinate-wise median / trimmed-mean / Krum cannot distinguish the
  malicious updates from benign noise. The shift is ``z * std`` where ``z`` is
  the largest deviation that still keeps a malicious worker within the benign
  population's order statistics, derived from the standard normal CDF.

* :func:`fang_attack` — the **Fang et al.** optimized local-model poisoning
  attack (USENIX Security 2020). It computes the benign mean and the sign of
  each coordinate, then pushes all malicious updates in the ``-sign`` direction
  scaled by a factor ``lambda`` chosen (by line search) as large as possible
  while a target robust aggregator still *accepts* a malicious update. We provide
  the Krum-targeted variant (maximise ``lambda`` such that a malicious point is
  Krum-selected) and the median/trimmed-mean directed-deviation variant.

All numpy. The math is the real attack math; nothing here is faked.
"""

from __future__ import annotations

import math
from typing import Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

from horizon_ric.federated import robust

# A proper type alias (not a bare variable) so mypy accepts `Vector` in
# annotations and resolves element types through `list[Vector]` returns —
# otherwise `Vector = np.ndarray` reads as a runtime value and downstream
# `.reshape(...)` calls fail with "Vector? has no attribute".
Vector: TypeAlias = npt.NDArray[np.float64]


def _phi_inv(p: float) -> float:
    """Inverse standard-normal CDF (probit) via the Acklam rational approx."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    # Coefficients (Peter Acklam's algorithm) — accurate to ~1e-9.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def alie_z(n_total: int, n_malicious: int) -> float:
    """The ALIE deviation factor ``z`` (Baruch et al. 2019, eq. for z_max).

    ``s = floor(n/2 + 1) - m`` benign workers must be "supported" for the
    malicious workers to control the median; ``z = Phi^{-1}((n - m - s)/(n - m))``.
    """
    n, m = n_total, n_malicious
    s = math.floor(n / 2 + 1) - m
    arg = (n - m - s) / max(n - m, 1)
    arg = min(max(arg, 1e-6), 1 - 1e-6)
    return float(_phi_inv(arg))


def alie_attack(
    benign_updates: Sequence[Vector],
    n_malicious: int,
    *,
    z: float | None = None,
) -> list[Vector]:
    """Craft ``n_malicious`` ALIE malicious updates from the benign population.

    Each malicious update is ``mean - z * std`` (coordinate-wise) — a small,
    coordinated shift that sits *inside* the benign variance envelope, so robust
    aggregators that rely on the benign order statistics cannot screen it out.
    Returns the list of malicious update vectors.
    """
    mat = np.asarray([np.asarray(u, dtype=np.float64).ravel() for u in benign_updates])
    mu = mat.mean(axis=0)
    sigma = mat.std(axis=0)
    n_total = mat.shape[0] + n_malicious
    if z is None:
        z = alie_z(n_total, n_malicious)
    # Move against the mean by z*std (the classic ALIE direction).
    mal = mu - z * sigma
    return [mal.copy() for _ in range(n_malicious)]


def fang_attack_krum(
    benign_updates: Sequence[Vector],
    n_malicious: int,
    *,
    lambda_init: float = 10.0,
    n_steps: int = 25,
) -> list[Vector]:
    """Fang directed-deviation attack tuned against Krum (USENIX-Sec 2020).

    Direction = ``-sign(mean(benign))``; magnitude ``lambda`` is halved by line
    search until a crafted malicious point would be *selected* by Krum over the
    benign points (i.e. the attack is admitted). All malicious workers send a
    tight cluster around ``mean + lambda * direction`` so they vote for each
    other under Krum's nearest-neighbour score.
    """
    mat = np.asarray([np.asarray(u, dtype=np.float64).ravel() for u in benign_updates])
    mu = mat.mean(axis=0)
    direction = -np.sign(mu)
    direction[direction == 0] = -1.0

    n_total = mat.shape[0] + n_malicious
    f = n_malicious

    lam = lambda_init
    chosen = None
    for _ in range(n_steps):
        mal_point = mu + lam * direction
        # Malicious cluster: identical points so they are each other's neighbours.
        mal = [mal_point.copy() for _ in range(n_malicious)]
        updates = list(mat) + mal
        if n_total > 2 * f + 2:
            res = robust.krum(updates, f=f)
            if res.selected_index >= mat.shape[0]:
                chosen = mal  # Krum picked a malicious point — attack succeeds
                break
        lam /= 2.0
    if chosen is None:
        # Even tiny lambda not selected — return the closest-to-benign cluster.
        mal_point = mu + lam * direction
        chosen = [mal_point.copy() for _ in range(n_malicious)]
    return chosen


def fang_attack_median(
    benign_updates: Sequence[Vector],
    n_malicious: int,
    *,
    lam: float = 3.0,
) -> list[Vector]:
    """Fang directed-deviation attack tuned against coordinate median / trimmed-mean.

    Pushes the malicious updates ``lam`` benign-standard-deviations in the
    ``-sign(mean)`` direction. With enough malicious workers this drags the
    per-coordinate median/trimmed-mean off the benign value; with too few it is
    bounded (which is exactly what we want to *measure* honestly).
    """
    mat = np.asarray([np.asarray(u, dtype=np.float64).ravel() for u in benign_updates])
    mu = mat.mean(axis=0)
    sigma = mat.std(axis=0) + 1e-9
    direction = -np.sign(mu)
    direction[direction == 0] = -1.0
    mal = mu + lam * sigma * direction
    return [mal.copy() for _ in range(n_malicious)]


# Breakdown points (the fraction of Byzantine clients each aggregator tolerates
# before it can be made arbitrarily wrong). Stated honestly for the report.
BREAKDOWN_POINTS = {
    "krum": "f Byzantine tolerated iff n > 2f + 2 (so < ~50%); single-point "
            "selection makes it brittle and high-variance under ALIE.",
    "median": "breakdown point ~50% (needs < n/2 Byzantine per coordinate); "
              "ALIE stays *inside* the envelope so small bias leaks through.",
    "trimmed_mean": "needs beta >= number of Byzantine and n > 2*beta; tolerates "
                    "< ~50% but the leaked bias grows with the trim slack.",
    "fedavg": "breakdown point 0% — a single Byzantine client moves it arbitrarily.",
}


__all__ = [
    "alie_z",
    "alie_attack",
    "fang_attack_krum",
    "fang_attack_median",
    "BREAKDOWN_POINTS",
]

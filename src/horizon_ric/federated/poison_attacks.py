"""Model-poisoning attack battery for federated learning (REAL, numpy-only).

This module completes the poisoning *attack battery* used to grade the shipped
robust aggregators in :mod:`horizon_ric.federated.robust`. It is deliberately
HONEST: several of these attacks are designed to *defeat* Krum / coordinate
median / trimmed-mean near their breakdown points, demonstrating that a robust
aggregator BOUNDS poisoning damage but is not a silver bullet.

We do NOT reimplement the two attacks that already live in
:mod:`horizon_ric.spectrum.attacks` — we IMPORT them:

* ``alie_attack`` / ``alie_z`` — "A Little Is Enough" (Baruch, Gilad-Baruch,
  Goldberg, NeurIPS 2019).
* ``fang_attack_krum`` / ``fang_attack_median`` — Fang et al. optimized local
  model poisoning (USENIX Security 2020).

This file adds the remaining canonical attacks:

1. :func:`sign_flip_attack` — sign-flipping / gradient inversion. Negate the
   honest-mean direction and scale it. (Bernstein et al., "signSGD with
   Majority Vote", ICLR 2019, motivates sign-based adversaries; Blanchard et
   al., NeurIPS 2017, analyse the inverted-gradient Byzantine update.)
2. :func:`scaling_attack` — scaling / boosting, a.k.a. model replacement.
   Multiply a malicious update by ~ ``n / f`` so that after averaging it
   replaces the global model. (Bagdasaryan et al., "How To Backdoor Federated
   Learning", AISTATS 2020.)
3. :func:`gaussian_attack` — random Byzantine. Large-variance Gaussian noise
   updates. (Classic Byzantine baseline; e.g. Blanchard et al., NeurIPS 2017.)
4. :func:`min_max_attack` / :func:`min_sum_attack` — the Min-Max and Min-Sum
   optimized attacks (Shejwalkar & Houmansadr, "Manipulating the Byzantine",
   NDSS 2021). They search for the largest perturbation along a chosen
   direction (inverse unit-vector / inverse-std) such that the malicious update
   stays *inside* the benign distribution — bounded by the maximum
   benign-to-benign distance (Min-Max) or the maximum benign-to-benign distance
   *sum* (Min-Sum) — so robust aggregators still select / average it.

All math is the real attack math. Pure numpy; no torch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Import (do NOT reimplement) ALIE + Fang from the spectrum attack module.
from horizon_ric.spectrum.attacks import (
    BREAKDOWN_POINTS,
    alie_attack,
    alie_z,
    fang_attack_krum,
    fang_attack_median,
)

Vector = np.ndarray


def _stack(updates: Sequence[Vector]) -> Vector:
    """Stack client update vectors into an ``(n, dim)`` float64 matrix."""
    mat = np.asarray([np.asarray(u, dtype=np.float64).ravel() for u in updates])
    if mat.ndim != 2:
        raise ValueError("all client updates must have the same length")
    return mat


# ---------------------------------------------------------------------------
# 1. Sign-flip / gradient-inversion attack
# ---------------------------------------------------------------------------
def sign_flip_attack(
    benign_updates: Sequence[Vector],
    n_malicious: int,
    *,
    scale: float = 1.0,
) -> list[Vector]:
    """Sign-flip / gradient-inversion attack (Bernstein et al. 2019; Blanchard 2017).

    Each malicious client submits ``-scale * mean(benign)`` — the *inverted*
    honest gradient direction, scaled by ``scale``. Inverting the gradient pushes
    the model uphill on the loss. With ``scale = 1`` the malicious update is a
    mirror image of the honest mean; with ``scale > 1`` it both inverts and
    boosts, which is what makes plain FedAvg unbounded.

    Returns ``n_malicious`` identical malicious vectors (a tight cluster, so they
    are each other's nearest neighbours under Krum).
    """
    mu = _stack(benign_updates).mean(axis=0)
    mal = -scale * mu
    return [mal.copy() for _ in range(n_malicious)]


# ---------------------------------------------------------------------------
# 2. Scaling / boosting (model-replacement) attack
# ---------------------------------------------------------------------------
def scaling_attack(
    benign_updates: Sequence[Vector],
    n_malicious: int,
    *,
    target: Vector | None = None,
    boost: float | None = None,
) -> list[Vector]:
    """Scaling / boosting "model-replacement" attack (Bagdasaryan et al. 2020).

    The adversary wants the averaged global update to equal an attacker-chosen
    ``target``. Because FedAvg divides by ``n = n_honest + n_malicious``, a
    malicious client multiplies its contribution by the boost factor
    ``gamma ~ n / n_malicious`` so that, after averaging, the malicious mass
    dominates and the aggregate is dragged onto ``target`` (model replacement).

    Args:
        benign_updates: the observed honest updates (used to size ``n`` and to
            default ``target`` to the inverted honest direction).
        n_malicious: number of malicious clients.
        target: the desired post-aggregation direction. Defaults to
            ``-mean(benign)`` (push the model the wrong way). The boosted update
            is ``boost * (target - mean(benign)) + mean(benign)`` per the
            model-replacement construction.
        boost: override the ``n / n_malicious`` boost factor.

    Returns ``n_malicious`` identical boosted malicious vectors.
    """
    mat = _stack(benign_updates)
    mu = mat.mean(axis=0)
    n_honest = mat.shape[0]
    n_total = n_honest + n_malicious
    if target is None:
        target = -mu  # default malicious objective: invert the model direction
    target = np.asarray(target, dtype=np.float64).ravel()
    gamma = float(boost) if boost is not None else (n_total / max(n_malicious, 1))
    # Model replacement: X_mal = gamma * (target - mu) + mu so that the post-FedAvg
    # global update lands on `target` when n_malicious clients are boosted by gamma.
    mal = gamma * (target - mu) + mu
    return [mal.copy() for _ in range(n_malicious)]


# ---------------------------------------------------------------------------
# 3. Gaussian / random Byzantine attack
# ---------------------------------------------------------------------------
def gaussian_attack(
    benign_updates: Sequence[Vector],
    n_malicious: int,
    *,
    sigma_mult: float = 30.0,
    seed: int | None = None,
) -> list[Vector]:
    """Random Byzantine attack: large-variance Gaussian noise updates.

    Each malicious client samples ``N(mean(benign), (sigma_mult * std)^2)`` — a
    noise update with a far larger spread than the benign population. This is the
    *naive* Byzantine baseline: it has no structure, so any working robust
    aggregator bounds it. We keep it precisely to show the contrast against the
    optimized Min-Max / Min-Sum / Fang attacks, which do far more damage.

    Args:
        sigma_mult: how many benign standard deviations wide the noise is.
        seed: optional RNG seed for reproducibility.
    """
    mat = _stack(benign_updates)
    mu = mat.mean(axis=0)
    sigma = mat.std(axis=0) + 1e-12
    rng = np.random.default_rng(seed)
    return [rng.normal(mu, sigma_mult * sigma) for _ in range(n_malicious)]


# ---------------------------------------------------------------------------
# 4. Min-Max and Min-Sum optimized attacks (Shejwalkar & Houmansadr, NDSS 2021)
# ---------------------------------------------------------------------------
def _perturbation_direction(mat: Vector, kind: str) -> Vector:
    """Perturbation direction used by the NDSS'21 attacks.

    The paper studies three directions; we support the two most effective ones:

    * ``"std"``    — ``-std(benign)`` (inverse standard deviation). Most damaging
      in practice and the paper's default for the "unit-vector" experiments.
    * ``"unit"``   — ``-mean(benign) / ||mean(benign)||`` (inverse unit vector of
      the benign mean).
    """
    mu = mat.mean(axis=0)
    if kind == "unit":
        norm = np.linalg.norm(mu)
        direction = -mu / norm if norm > 0 else -np.ones_like(mu)
    elif kind == "std":
        direction = -mat.std(axis=0)
    else:
        raise ValueError(f"unknown perturbation direction {kind!r}")
    return direction


def min_max_attack(
    benign_updates: Sequence[Vector],
    n_malicious: int,
    *,
    direction: str = "std",
    gamma_init: float = 10.0,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> list[Vector]:
    r"""Min-Max optimized attack (Shejwalkar & Houmansadr, NDSS 2021, §4 / Alg.).

    Craft a single malicious update ``m = mu + gamma * d`` (``d`` = perturbation
    direction, ``mu`` = benign mean) with the *largest* ``gamma`` such that the
    maximum distance from ``m`` to any benign update is no larger than the
    maximum distance *between any two benign* updates:

    .. math::

        \max_{i} \lVert m - b_i \rVert \;\le\; \max_{i,j} \lVert b_i - b_j \rVert .

    This keeps the malicious update inside the benign "ball", so distance-based
    robust aggregators (Krum, and effectively trimmed-mean/median order
    statistics) cannot screen it out, yet ``gamma`` is maximal so the injected
    bias is as large as the benign spread allows. ``gamma`` is found by binary
    search on the feasibility constraint.

    Returns ``n_malicious`` identical malicious vectors.
    """
    mat = _stack(benign_updates)
    mu = mat.mean(axis=0)
    d = _perturbation_direction(mat, direction)

    # Maximum benign-to-benign distance (the feasibility budget).
    diff = mat[:, None, :] - mat[None, :, :]
    benign_pairwise = np.sqrt(np.sum(diff**2, axis=2))
    max_benign_dist = float(np.max(benign_pairwise))

    def max_dist_to_benign(gamma: float) -> float:
        m = mu + gamma * d
        return float(np.max(np.linalg.norm(mat - m, axis=1)))

    # Binary search for the largest feasible gamma in [0, gamma_init],
    # expanding gamma_init until it becomes infeasible.
    lo, hi = 0.0, gamma_init
    while max_dist_to_benign(hi) <= max_benign_dist and hi < 1e9:
        hi *= 2.0
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if max_dist_to_benign(mid) <= max_benign_dist:
            lo = mid
        else:
            hi = mid
    gamma = lo
    mal = mu + gamma * d
    return [mal.copy() for _ in range(n_malicious)]


def min_sum_attack(
    benign_updates: Sequence[Vector],
    n_malicious: int,
    *,
    direction: str = "std",
    gamma_init: float = 10.0,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> list[Vector]:
    r"""Min-Sum optimized attack (Shejwalkar & Houmansadr, NDSS 2021, §4 / Alg.).

    Same construction as :func:`min_max_attack` but the feasibility constraint
    bounds the *sum* of squared distances from ``m`` to the benign updates by the
    maximum such sum over benign points:

    .. math::

        \sum_{i} \lVert m - b_i \rVert^2 \;\le\;
        \max_{j} \sum_{i} \lVert b_j - b_i \rVert^2 .

    The sum-of-distances constraint is the one used by Krum's neighbour score, so
    Min-Sum is tuned to slip a maximally-biased point past Krum's selection. As
    with Min-Max, ``gamma`` is found by binary search.

    Returns ``n_malicious`` identical malicious vectors.
    """
    mat = _stack(benign_updates)
    mu = mat.mean(axis=0)
    d = _perturbation_direction(mat, direction)

    # Maximum (over benign points) of the sum of squared distances to all benign.
    diff = mat[:, None, :] - mat[None, :, :]
    sq = np.sum(diff**2, axis=2)
    max_benign_sumsq = float(np.max(np.sum(sq, axis=1)))

    def sumsq_to_benign(gamma: float) -> float:
        m = mu + gamma * d
        return float(np.sum(np.sum((mat - m) ** 2, axis=1)))

    lo, hi = 0.0, gamma_init
    while sumsq_to_benign(hi) <= max_benign_sumsq and hi < 1e9:
        hi *= 2.0
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if sumsq_to_benign(mid) <= max_benign_sumsq:
            lo = mid
        else:
            hi = mid
    gamma = lo
    mal = mu + gamma * d
    return [mal.copy() for _ in range(n_malicious)]


# ---------------------------------------------------------------------------
# Battery registry — name -> (callable, citation). ALIE/Fang are IMPORTED.
# ---------------------------------------------------------------------------
ATTACK_BATTERY = {
    "sign_flip": (sign_flip_attack, "Bernstein et al. ICLR'19 / Blanchard et al. NeurIPS'17"),
    "scaling": (scaling_attack, "Bagdasaryan et al. AISTATS'20 (model replacement)"),
    "gaussian": (gaussian_attack, "Blanchard et al. NeurIPS'17 (random Byzantine baseline)"),
    "min_max": (min_max_attack, "Shejwalkar & Houmansadr NDSS'21 (Min-Max)"),
    "min_sum": (min_sum_attack, "Shejwalkar & Houmansadr NDSS'21 (Min-Sum)"),
    "alie": (alie_attack, "Baruch et al. NeurIPS'19 (A Little Is Enough) [imported]"),
    "fang_krum": (fang_attack_krum, "Fang et al. USENIX-Sec'20 (Krum-targeted) [imported]"),
    "fang_median": (fang_attack_median, "Fang et al. USENIX-Sec'20 (median-targeted) [imported]"),
}


__all__ = [
    "Vector",
    "sign_flip_attack",
    "scaling_attack",
    "gaussian_attack",
    "min_max_attack",
    "min_sum_attack",
    "ATTACK_BATTERY",
    # re-exported (imported, not reimplemented)
    "alie_attack",
    "alie_z",
    "fang_attack_krum",
    "fang_attack_median",
    "BREAKDOWN_POINTS",
]

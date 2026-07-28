"""Robust federated aggregation — bound the influence of poisoning clients.

Plain FedAvg (the coordinate-wise mean) has unbounded sensitivity: a single
Byzantine client can drag the global model anywhere by sending a large update.
In an AI-RAN setting where per-cell clients fine-tune a neural-PHY model, that
is the O-RAN WG11 *model-poisoning* threat. This module ships three classical
robust aggregators that cap that influence:

* **Krum** (Blanchard et al., NeurIPS 2017) — selects the single client update
  closest to its ``n - f - 2`` nearest neighbours, rejecting the ``f`` furthest
  outliers. Byzantine-robust for ``n > 2f + 2``.
* **Coordinate-wise median** — per-parameter median; breakdown point ~50 %.
* **Trimmed mean** (Yin et al., ICML 2018) — drop the ``beta`` highest and
  lowest values per coordinate, average the rest.
* **FLTrust** (Cao, Fang, Liu, Gong, NDSS 2021) — the server holds a small,
  clean *root* dataset, computes its own update ``g0`` from it each round, and
  scores every client update by ``ReLU(cos(g_i, g0))``. Updates pointing away
  from the server direction get trust score 0 and are dropped entirely; the
  survivors are rescaled to ``‖g0‖`` (so a boosted update cannot buy influence
  with magnitude) and averaged with trust-score weights.

Why FLTrust matters here, measured on the real DeepMIMO federation
(``benchmarks/federated_coverage_loop.py``): the three distance/order-statistic
aggregators above all pay a *clean* accuracy tax — they discard honest
information under a non-IID split — and Krum is hard-limited to ``f < (K-2)/2``,
so at ``K = 10`` it cannot even be *run* with four or more adversaries. FLTrust
has no such combinatorial limit: its breakdown is governed by whether the server
root direction remains meaningful, not by a counting bound.

Updates are plain ``numpy`` vectors (one flattened client update each), so this
runs anywhere — no torch, no accelerator. Use :func:`flatten_state` /
:func:`unflatten_state` to move between a parameter dict and a flat vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

Vector = np.ndarray


# ---------------------------------------------------------------------------
# state_dict <-> flat vector helpers
# ---------------------------------------------------------------------------
def flatten_state(state: dict[str, np.ndarray]) -> tuple[Vector, list[tuple[str, tuple[int, ...]]]]:
    """Flatten a parameter dict into one vector + a shape manifest."""
    parts = []
    manifest: list[tuple[str, tuple[int, ...]]] = []
    for key in sorted(state):
        arr = np.asarray(state[key], dtype=np.float64)
        manifest.append((key, arr.shape))
        parts.append(arr.ravel())
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
    return flat, manifest


def unflatten_state(
    flat: Vector, manifest: list[tuple[str, tuple[int, ...]]]
) -> dict[str, np.ndarray]:
    """Inverse of :func:`flatten_state`."""
    out: dict[str, np.ndarray] = {}
    i = 0
    for key, shape in manifest:
        n = int(np.prod(shape)) if shape else 1
        out[key] = flat[i : i + n].reshape(shape)
        i += n
    return out


def _stack(updates: Iterable[Vector]) -> Vector:
    mat = np.asarray([np.asarray(u, dtype=np.float64).ravel() for u in updates])
    if mat.ndim != 2:
        raise ValueError("all client updates must have the same length")
    return mat


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------
def fedavg(updates: Sequence[Vector]) -> Vector:
    """Plain coordinate-wise mean (the non-robust baseline)."""
    return _stack(updates).mean(axis=0)


def coordinate_median(updates: Sequence[Vector]) -> Vector:
    """Per-coordinate median. Breakdown point ~50 %."""
    return np.median(_stack(updates), axis=0)


def trimmed_mean(updates: Sequence[Vector], beta: int = 1) -> Vector:
    """Per-coordinate trimmed mean: drop ``beta`` highest and lowest, average rest.

    Requires ``n > 2 * beta``.
    """
    mat = _stack(updates)
    n = mat.shape[0]
    if n <= 2 * beta:
        raise ValueError(f"trimmed_mean needs n > 2*beta; got n={n}, beta={beta}")
    srt = np.sort(mat, axis=0)
    return srt[beta : n - beta].mean(axis=0)


@dataclass(frozen=True)
class KrumResult:
    aggregate: Vector
    selected_index: int
    scores: list[float]


def krum(updates: Sequence[Vector], f: int) -> KrumResult:
    """Krum: pick the update closest to its ``n - f - 2`` nearest neighbours.

    Args:
        updates: client update vectors.
        f: assumed number of Byzantine clients. Requires ``n > 2f + 2``.

    Returns the selected update plus the per-client scores.
    """
    mat = _stack(updates)
    n = mat.shape[0]
    if n <= 2 * f + 2:
        raise ValueError(f"krum needs n > 2f+2; got n={n}, f={f}")
    # Pairwise squared distances.
    sq = np.sum(
        (mat[:, None, :] - mat[None, :, :]) ** 2, axis=2
    )  # (n, n)
    m = n - f - 2
    scores: list[float] = []
    for i in range(n):
        dists = np.sort(sq[i])  # includes self-distance 0 at front
        # Sum the m smallest *excluding* self (index 0).
        scores.append(float(np.sum(dists[1 : m + 1])))
    selected = int(np.argmin(scores))
    return KrumResult(aggregate=mat[selected].copy(), selected_index=selected, scores=scores)


@dataclass(frozen=True)
class FLTrustResult:
    aggregate: Vector
    trust_scores: list[float]
    root_norm: float


def fltrust(updates: Sequence[Vector], root_update: Vector) -> FLTrustResult:
    r"""FLTrust (Cao et al., NDSS 2021): server-root cosine trust + norm clipping.

    The server holds a small clean *root* dataset and runs the same local solver
    on it to obtain a reference update ``g0``. For each client update ``g_i``:

    .. math::

        TS_i = \mathrm{ReLU}\!\left(\frac{\langle g_i, g_0\rangle}
                                         {\lVert g_i\rVert\,\lVert g_0\rVert}\right),
        \qquad
        \bar g_i = \frac{\lVert g_0 \rVert}{\lVert g_i \rVert}\, g_i ,
        \qquad
        g = \frac{\sum_i TS_i\,\bar g_i}{\sum_i TS_i}.

    Two mechanisms do the work, and they are complementary:

    * **Direction.** A client whose update points away from the server's own
      direction gets ``TS_i = 0`` and is *dropped*, not merely down-weighted.
      Sign-flip and model-replacement adversaries are removed outright.
    * **Magnitude.** Every surviving update is renormalised to ``‖g0‖`` before
      averaging, so scaling/boosting buys an adversary exactly nothing — which
      is the failure mode that makes plain FedAvg unbounded.

    Unlike Krum, this needs no ``n > 2f + 2`` counting bound: robustness comes
    from the server's own clean data, so it degrades smoothly as ``f`` grows
    rather than becoming undefined.

    Args:
        updates: client update vectors.
        root_update: the server's update computed on its root dataset.

    Returns the aggregate plus the per-client trust scores (an auditable record
    of which clients were admitted) and ``‖g0‖``.
    """
    g0 = np.asarray(root_update, dtype=np.float64).ravel()
    root_norm = float(np.linalg.norm(g0))
    mat = _stack(updates)
    if mat.shape[1] != g0.size:
        raise ValueError("root_update length does not match the client updates")
    if root_norm == 0.0:
        # A zero server direction carries no information; trust nobody.
        return FLTrustResult(
            aggregate=np.zeros_like(g0), trust_scores=[0.0] * mat.shape[0], root_norm=0.0
        )

    norms = np.linalg.norm(mat, axis=1)
    safe = np.where(norms > 0.0, norms, 1.0)
    cos = (mat @ g0) / (safe * root_norm)
    ts = np.where(norms > 0.0, np.maximum(cos, 0.0), 0.0)
    total = float(ts.sum())
    if total <= 0.0:
        # Every client disagrees with the server: fall back to the server's own
        # update rather than emitting an adversary-chosen direction.
        return FLTrustResult(
            aggregate=g0.copy(), trust_scores=[float(v) for v in ts], root_norm=root_norm
        )
    scaled = mat * (root_norm / safe)[:, None]
    agg = (ts[:, None] * scaled).sum(axis=0) / total
    return FLTrustResult(
        aggregate=agg, trust_scores=[float(v) for v in ts], root_norm=root_norm
    )


# Strategy registry for ergonomic config-driven selection.
def aggregate(
    updates: Sequence[Vector],
    method: str = "krum",
    *,
    f: int = 1,
    beta: int = 1,
    root_update: Vector | None = None,
) -> Vector:
    """Dispatch to a named robust aggregator. ``method`` ∈ {krum, median,
    trimmed_mean, fltrust, fedavg}.

    ``fltrust`` additionally requires ``root_update`` — the server's own update
    computed on its clean root dataset.
    """
    if method == "fedavg":
        return fedavg(updates)
    if method == "median":
        return coordinate_median(updates)
    if method == "trimmed_mean":
        return trimmed_mean(updates, beta=beta)
    if method == "krum":
        return krum(updates, f=f).aggregate
    if method == "fltrust":
        if root_update is None:
            raise ValueError("fltrust requires root_update (the server root-set update)")
        return fltrust(updates, root_update).aggregate
    raise ValueError(f"unknown aggregation method {method!r}")


__all__ = [
    "Vector",
    "flatten_state",
    "unflatten_state",
    "fedavg",
    "coordinate_median",
    "trimmed_mean",
    "krum",
    "KrumResult",
    "fltrust",
    "FLTrustResult",
    "aggregate",
]

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


# Strategy registry for ergonomic config-driven selection.
def aggregate(
    updates: Sequence[Vector],
    method: str = "krum",
    *,
    f: int = 1,
    beta: int = 1,
) -> Vector:
    """Dispatch to a named robust aggregator. ``method`` ∈ {krum, median,
    trimmed_mean, fedavg}."""
    if method == "fedavg":
        return fedavg(updates)
    if method == "median":
        return coordinate_median(updates)
    if method == "trimmed_mean":
        return trimmed_mean(updates, beta=beta)
    if method == "krum":
        return krum(updates, f=f).aggregate
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
    "aggregate",
]

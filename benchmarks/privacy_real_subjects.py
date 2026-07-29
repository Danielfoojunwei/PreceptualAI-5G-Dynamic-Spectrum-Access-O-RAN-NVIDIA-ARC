#!/usr/bin/env python3
"""Real data subjects for the privacy / unlearning suites — shared substrate.

The four privacy suites (``dp_privacy_suite``, ``subject_erasure_suite``,
``federated_unlearning_suite``, ``verifiable_secagg_suite``) used to run on
``numpy.random`` draws and on synthetic tabular Q-tables rolled out in a toy
``DSAEnv``. A privacy claim made on ``rng.normal`` is a claim about nothing: the
records it protects were never anybody's.

This module gives all four the same **real** substrate, with no download — the
licence-gated DeepMIMO ASU Campus 3.5 GHz build is already in the tree:

* ``datasets/deepmimo_asu_3p5/generated/channel_features.jsonl`` — 4096 rows,
  each a **real ray-traced receiver**: its real 3D position (``position_m``,
  Wireless InSite site-specific ray tracing on the ASU campus) and its real
  measured gain in each of 6 subbands across 100 MHz (``subband_gain_dbw``).

**A subject is one real receiver.** Its private record is its real position and
its six real measured gains. A client (an O-RAN cell site) owns a geographically
contiguous set of those receivers. The learning task is the same real coverage /
per-subband path-loss regression the already-real ``federated_coverage_loop.py``
uses, and this module deliberately reuses that benchmark's proven recipe:

  whitened quadratic position design  ->  per-site FedProx proximal ridge steps
  ->  warm-started federated rounds   ->  a global per-subband gain predictor.

What that buys the privacy suites, concretely:

* **DP** — the records the Gaussian mechanism protects are real measurements of
  real positions, so the clip norm can be calibrated on a *public* server-held
  root slice of the same real population, and the achieved epsilon can be
  *audited* against the real adjacent-dataset pair instead of asserted.
* **Erasure** — a FedProx step is an exact function of the per-subject
  sufficient statistics ``Dᵀ D`` and ``Dᵀ T``. Erasing a named real receiver is
  an exact rank-1 downdate, so "as if this subject's data was never used" is
  *computable* and checkable to machine precision, not approximated.
* **Membership inference** — member and non-member clients are drawn from the
  same real campus, so an MIA AUC above 0.5 is a real signal, not an artefact of
  two different random number generators.
* **Secure aggregation** — the vectors being secret-shared are real model
  updates with real magnitudes, which is what exposes the fixed-point
  quantisation cost honestly.

Scope note, stated once here and repeated in every result JSON: these are
**ray-traced receivers, not subscribers**. A receiver is a grid point in a
site-specific propagation simulation. Every epsilon, AUC and erasure bound below
is a statement about that population under the stated mechanism. It is *not* a
measurement of privacy risk to real mobile subscribers, and no real personal
data was used anywhere in this repository.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_FEATURES = Path("datasets/deepmimo_asu_3p5/generated/channel_features.jsonl")
DEFAULT_MANIFEST = Path("datasets/deepmimo_asu_3p5/manifest.json")

N_SUBBANDS = 6
N_FEATURES = 6  # whitened [x, y, x², y², xy] + intercept
MODEL_DIM = N_FEATURES * N_SUBBANDS  # 36 - the federated update vector length

# FedProx knobs, identical to benchmarks/federated_coverage_loop.py so the
# privacy suites and the coverage suite describe the same learning problem.
PROX_LAMBDA = 2.0
PROX_ETA = 0.6

ROOT_STRIDE = 8  # every 8th real receiver is server-held and PUBLIC (512 of 4096)


# ─────────────────────────────────────────────────────────────────────────────
# Loading the real receiver population
# ─────────────────────────────────────────────────────────────────────────────
def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no feature rows in {path}")
    return rows


def whiten_design(pos: np.ndarray) -> np.ndarray:
    """Whitened quadratic position design ``[whitened(x, y, x², y², xy), 1]``.

    Centering + SVD whitening makes the global least-squares problem well
    conditioned so that local proximal solves stay stable on geographically
    concentrated, non-IID clients. Identical preprocessing to
    ``federated_coverage_loop._whiten_design``.
    """
    x, y = pos[:, 0], pos[:, 1]
    feats = np.column_stack([x, y, x * x, y * y, x * y])
    feats = feats - feats.mean(axis=0)
    u, _s, _vt = np.linalg.svd(feats, full_matrices=False)
    whitened = u * math.sqrt(len(feats))
    return np.column_stack([whitened, np.ones(len(feats))])


@dataclass
class SubjectPopulation:
    """The real receiver population, split into a public root and private subjects.

    Every array is indexed by the same row order as ``channel_features.jsonl``.
    """

    receiver_index: np.ndarray  # real DeepMIMO receiver ids (the subject ids)
    pos: np.ndarray  # real 2D positions (m)
    gains_dbw: np.ndarray  # (N, 6) real measured subband gains
    design: np.ndarray  # (N, 6) whitened quadratic position features
    target: np.ndarray  # (N, 6) publicly standardised gains
    center_dbw: float  # PUBLIC standardisation centre (from the manifest)
    scale_db: float  # PUBLIC standardisation scale (from the manifest)
    root_idx: np.ndarray  # server-held public receivers (never a subject)
    private_idx: np.ndarray  # the private subject pool

    @property
    def n(self) -> int:
        return int(self.design.shape[0])

    def subject_id(self, row: int) -> str:
        """Stable subject identifier for a real receiver row."""
        return f"deepmimo-rx-{int(self.receiver_index[row])}"

    def predict_dbw(self, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return (self.design[idx] @ w.reshape(N_FEATURES, N_SUBBANDS)) * self.scale_db + (
            self.center_dbw
        )

    def rmse_db(self, w: np.ndarray, idx: np.ndarray) -> float:
        """Per-subband RMSE (dB) of the model against the real measured gains."""
        err = self.predict_dbw(w, idx) - self.gains_dbw[idx]
        return float(np.sqrt(np.mean(err**2)))

    def baseline_rmse_db(self, idx: np.ndarray) -> float:
        """Predict the public centre everywhere — the do-nothing model."""
        return float(np.sqrt(np.mean((self.gains_dbw[idx] - self.center_dbw) ** 2)))

    def per_subject_squared_error(self, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Mean squared dB error per receiver — the loss-threshold MIA statistic."""
        err = self.predict_dbw(w, idx) - self.gains_dbw[idx]
        return np.mean(err**2, axis=1)


def build_population(
    features_path: Path = DEFAULT_FEATURES,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    root_stride: int = ROOT_STRIDE,
) -> tuple[SubjectPopulation, dict[str, Any]]:
    """Load the real receivers and the manifest that pins their provenance.

    The standardisation constants are taken from ``subband_gain_dbw_min`` /
    ``subband_gain_dbw_max`` in the **committed manifest**. Those are already
    published aggregates of this build, so using them as the centre and scale
    leaks nothing that is not already public — unlike standardising on the
    private cohort's own mean and standard deviation.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_rows(features_path)
    gains = np.asarray([r["subband_gain_dbw"] for r in rows], dtype=np.float64)
    pos = np.asarray([r["position_m"][:2] for r in rows], dtype=np.float64)
    rx = np.asarray([r["receiver_index"] for r in rows], dtype=np.int64)
    if not (np.all(np.isfinite(gains)) and np.all(np.isfinite(pos))):
        raise ValueError("feature rows have non-finite gains or positions")
    if gains.shape[1] != N_SUBBANDS:
        raise ValueError(f"expected {N_SUBBANDS} subbands, got {gains.shape[1]}")

    lo = float(manifest["subband_gain_dbw_min"])
    hi = float(manifest["subband_gain_dbw_max"])
    center = 0.5 * (lo + hi)
    scale = 0.25 * (hi - lo)
    design = whiten_design(pos)
    target = (gains - center) / scale

    root_idx = np.arange(0, len(rows), root_stride)
    mask = np.ones(len(rows), dtype=bool)
    mask[root_idx] = False
    pop = SubjectPopulation(
        receiver_index=rx,
        pos=pos,
        gains_dbw=gains,
        design=design,
        target=target,
        center_dbw=center,
        scale_db=scale,
        root_idx=root_idx,
        private_idx=np.flatnonzero(mask),
    )
    return pop, manifest


# ─────────────────────────────────────────────────────────────────────────────
# Real, geographically non-IID client partition
# ─────────────────────────────────────────────────────────────────────────────
def partition_geographic(pos: np.ndarray, n_sites: int, *, seed: int) -> list[np.ndarray]:
    """Lloyd partition of real 2D positions into ``n_sites`` contiguous clients.

    Ray tracing is site-specific, so clustering on the real positions produces
    genuinely non-IID clients: each holds one region of the campus with its own
    slice of the propagation function. Same construction as
    ``federated_coverage_loop._partition_geographic``.
    """
    if n_sites < 1 or n_sites > len(pos):
        raise ValueError("n_sites must be between 1 and the receiver count")
    rng = np.random.default_rng(seed)
    centers = [pos[int(rng.integers(len(pos)))]]
    for _ in range(1, n_sites):
        d = np.min([np.linalg.norm(pos - c, axis=1) for c in centers], axis=0)
        centers.append(pos[int(np.argmax(d))])
    centers = np.asarray(centers)
    assign = np.zeros(len(pos), dtype=np.int64)
    for _ in range(50):
        assign = np.argmin(
            np.stack([np.linalg.norm(pos - c, axis=1) for c in centers], axis=1), axis=1
        )
        new = np.asarray(
            [
                pos[assign == k].mean(axis=0) if np.any(assign == k) else centers[k]
                for k in range(n_sites)
            ]
        )
        if np.allclose(new, centers):
            centers = new
            break
        centers = new
    return [np.flatnonzero(assign == k) for k in range(n_sites)]


# ─────────────────────────────────────────────────────────────────────────────
# The local learner: a FedProx step, exactly expressible in sufficient statistics
# ─────────────────────────────────────────────────────────────────────────────
def suff_stats(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-client sufficient statistics ``(DᵀD, DᵀT, n)``.

    Both are plain sums over the client's subjects, so a single subject's
    contribution is an exact rank-1 term that can be added or removed without
    touching any other subject's data. That is what makes erasure exact.
    """
    return design.T @ design, design.T @ target, int(design.shape[0])


def prox_update_from_stats(
    w: np.ndarray, gram: np.ndarray, cross: np.ndarray, n: int
) -> np.ndarray:
    """FedProx proximal ridge update computed from sufficient statistics only.

    Solves ``argmin_v ½·(1/n)‖D v − T‖² + ½λ‖v − w‖²`` in closed form and returns
    the damped step ``η·(v* − w)``, flattened. Identical arithmetic to
    ``federated_coverage_loop._local_prox`` but expressed so that subject-level
    erasure is a rank-1 downdate of ``gram``/``cross``.
    """
    if n <= 0:
        raise ValueError("a client must hold at least one subject")
    wm = w.reshape(N_FEATURES, N_SUBBANDS)
    hessian = gram / n + PROX_LAMBDA * np.eye(N_FEATURES)
    grad = (gram @ wm - cross) / n
    return (-PROX_ETA * np.linalg.solve(hessian, grad)).ravel()


def local_prox_update(w: np.ndarray, design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """FedProx update for one client, from its raw real rows."""
    gram, cross, n = suff_stats(design, target)
    return prox_update_from_stats(w, gram, cross, n)


def federated_rounds(
    pop: SubjectPopulation,
    clients: Sequence[np.ndarray],
    *,
    rounds: int,
    aggregate=None,
) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """Warm-started federated training over real client partitions.

    Returns the final global weights and, per round, the list of client updates
    that were aggregated (the cached uploads an unlearning service replays).
    """
    if aggregate is None:
        def aggregate(ups: Sequence[np.ndarray]) -> np.ndarray:
            return np.mean(np.stack(ups), axis=0)

    w = np.zeros(MODEL_DIM, dtype=np.float64)
    trace: list[list[np.ndarray]] = []
    for _ in range(rounds):
        ups = [
            local_prox_update(w, pop.design[idx], pop.target[idx])
            for idx in clients
        ]
        trace.append(ups)
        w = w + aggregate(ups)
    return w, trace


# ─────────────────────────────────────────────────────────────────────────────
# Membership-inference machinery (real members vs real non-members)
# ─────────────────────────────────────────────────────────────────────────────
def auc_mann_whitney(member: Sequence[float], nonmember: Sequence[float]) -> float:
    """``P(score_member > score_nonmember)`` over all pairs; ties count 0.5."""
    m = np.asarray(member, dtype=np.float64)
    k = np.asarray(nonmember, dtype=np.float64)
    if m.size == 0 or k.size == 0:
        return 0.5
    gt = float(np.sum(m[:, None] > k[None, :]))
    eq = float(np.sum(m[:, None] == k[None, :]))
    return (gt + 0.5 * eq) / float(m.size * k.size)


# ─────────────────────────────────────────────────────────────────────────────
# Numerically VERIFYING the privacy guarantee (rather than asserting it)
# ─────────────────────────────────────────────────────────────────────────────
def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def gaussian_delta(epsilon: float, sensitivity: float, sigma: float) -> float:
    """Exact ``δ(ε)`` of the Gaussian mechanism (Balle & Wang, ICML 2018, Thm 8).

    ``δ = Φ(Δ/(2σ) − εσ/Δ) − e^ε · Φ(−Δ/(2σ) − εσ/Δ)``. This is tight — not an
    RDP relaxation — so it is the right yardstick for checking that the
    accountant's number is a valid *upper* bound.
    """
    if sigma <= 0 or sensitivity <= 0:
        raise ValueError("sensitivity and sigma must be positive")
    a = sensitivity / (2.0 * sigma)
    b = epsilon * sigma / sensitivity
    return _phi(a - b) - math.exp(epsilon) * _phi(-a - b)


def analytic_gaussian_epsilon(sensitivity: float, sigma: float, delta: float) -> float:
    """Smallest ``ε`` for which the Gaussian mechanism is ``(ε, δ)``-DP (exact)."""
    lo, hi = 0.0, 1.0
    while gaussian_delta(hi, sensitivity, sigma) > delta:
        hi *= 2.0
        if hi > 1e6:
            return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gaussian_delta(mid, sensitivity, sigma) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def _clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Two-sided Clopper–Pearson interval for a binomial proportion.

    Implemented from the incomplete-beta / F relation via bisection on the
    regularised incomplete beta computed by a continued fraction, so the suite
    keeps its "pure numpy, no scipy" property.
    """
    if n <= 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return lo, hi


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)`` (Lentz continued fraction)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log1p(-x) * b - lbeta) / a
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Inverse regularised incomplete beta by bisection (60 halvings ≈ 1e-18)."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def empirical_epsilon_lower_bound(
    scores_d: np.ndarray,
    scores_dprime: np.ndarray,
    delta: float,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Audit an achieved ``ε`` from real mechanism outputs on adjacent datasets.

    ``scores_d`` / ``scores_dprime`` are the attacker's scalar test statistic on
    independent runs of the mechanism over ``D`` and its replace-one neighbour
    ``D'`` (both built from real receivers). For every candidate threshold the
    attacker's ``(TPR, FPR)`` gives the DP lower bound

        ``ε̂ = log((TPR − δ) / FPR)``   and   ``log((1 − FPR − δ) / (1 − TPR))``,

    with Clopper–Pearson 95% bounds applied in the conservative direction so the
    reported number is a statistically valid lower bound on the mechanism's true
    ``ε`` (Jagielski, Ullman & Oprea, NeurIPS 2020; Nasr et al., S&P 2021).

    A valid implementation must satisfy ``ε̂ ≤ ε_analytic ≤ ε_RDP``. Reporting
    ε̂ is the difference between claiming a guarantee and measuring one.
    """
    n = min(len(scores_d), len(scores_dprime))
    if n < 100:
        raise ValueError("audit needs at least 100 trials per dataset")
    thresholds = np.unique(np.concatenate([scores_d, scores_dprime]))
    if thresholds.size > 257:
        thresholds = np.quantile(thresholds, np.linspace(0.0, 1.0, 257))
    scores_d = np.sort(scores_d[:n])
    scores_dprime = np.sort(scores_dprime[:n])
    best = 0.0
    best_at: dict[str, Any] = {}
    for t in thresholds:
        k_tp = n - int(np.searchsorted(scores_d, t, side="left"))
        k_fp = n - int(np.searchsorted(scores_dprime, t, side="left"))
        tpr_lo, _ = _clopper_pearson(k_tp, n, alpha)
        _, fpr_hi = _clopper_pearson(k_fp, n, alpha)
        for num, den in (
            (tpr_lo - delta, fpr_hi),
            (1.0 - fpr_hi - delta, 1.0 - tpr_lo + 1e-300),
        ):
            if num > 0.0 and den > 0.0:
                eps = math.log(num / den)
                if eps > best:
                    best = eps
                    best_at = {
                        "threshold": float(t),
                        "tpr_point": k_tp / n,
                        "fpr_point": k_fp / n,
                        "tpr_lower_95": tpr_lo,
                        "fpr_upper_95": fpr_hi,
                    }
    return {
        "epsilon_empirical_lower_95": round(float(best), 4),
        "trials_per_dataset": int(n),
        "attained_at": {k: round(float(v), 6) for k, v in best_at.items()},
        "method": (
            "threshold sweep over the optimal likelihood-ratio statistic with "
            "one-sided Clopper-Pearson 95% bounds (Jagielski et al., NeurIPS 2020)"
        ),
    }


def provenance_block(
    manifest: dict[str, Any],
    pop: SubjectPopulation,
    *,
    extra_scope: str = "",
) -> dict[str, Any]:
    """The provenance header every privacy result JSON carries."""
    scope = (
        "These are ray-traced DeepMIMO receivers, not mobile subscribers. A "
        "'data subject' here is one simulated receiver grid point whose real "
        "measured subband gains and real position form its private record. Every "
        "epsilon, membership-inference AUC and erasure bound below is a property "
        "of the stated mechanism on that population under the stated adjacency "
        "convention. It is NOT a measurement of privacy risk to real subscribers, "
        "and no personal data was used. The DeepMIMO scenario archive carries no "
        "separate licence file, so no raw or row-level derived data is "
        "redistributed - only the manifest hashes and these aggregate results."
    )
    return {
        "dataset": manifest["dataset"],
        "scenario": manifest["scenario"],
        "data_kind": manifest["data_kind"],
        "source_archive_sha256": manifest["source_archive_sha256"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "features_sha256": manifest["features_sha256"],
        "deepmimo_version": manifest["deepmimo_version"],
        "carrier_frequency_hz": manifest["transform"]["carrier_frequency_hz"],
        "receivers_total": pop.n,
        "public_root_receivers": int(pop.root_idx.size),
        "private_subject_receivers": int(pop.private_idx.size),
        "subject_definition": (
            "one real DeepMIMO receiver (receiver_index); its private record is "
            "its real position_m and its 6 real subband_gain_dbw values"
        ),
        "public_standardisation": {
            "center_dbw": round(pop.center_dbw, 6),
            "scale_db": round(pop.scale_db, 6),
            "source": (
                "midpoint and quarter-range of the manifest-published "
                "subband_gain_dbw_min / subband_gain_dbw_max, which are already "
                "committed public aggregates of this build"
            ),
        },
        "scope_note": scope + (f" {extra_scope}" if extra_scope else ""),
    }


__all__ = [
    "DEFAULT_FEATURES",
    "DEFAULT_MANIFEST",
    "MODEL_DIM",
    "N_FEATURES",
    "N_SUBBANDS",
    "PROX_ETA",
    "PROX_LAMBDA",
    "SubjectPopulation",
    "analytic_gaussian_epsilon",
    "auc_mann_whitney",
    "build_population",
    "empirical_epsilon_lower_bound",
    "federated_rounds",
    "gaussian_delta",
    "load_rows",
    "local_prox_update",
    "partition_geographic",
    "prox_update_from_stats",
    "provenance_block",
    "suff_stats",
    "whiten_design",
]

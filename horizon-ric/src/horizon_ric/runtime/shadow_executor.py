"""Shadow executor — run candidate model side-by-side with the active.

The candidate's predictions are LOGGED, never emitted as A1 policies.
After enough shadow steps, the recommendation logic compares predicted
vs actual outcomes and emits a verdict (promote / reject / insufficient).

This is the validation gate that sits between Validate and Promote in
the TS 28.567 LoopState machine — see `runtime/loop_state.py`.
"""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal, Optional, Protocol, Sequence


class _Predictor(Protocol):
    """Anything that maps a state to a numeric prediction in [0, 1]."""

    def predict(self, state) -> float:  # noqa: D401
        ...


@dataclass(frozen=True)
class ShadowDecision:
    """One shadow step. Active is what got emitted; candidate is logged.

    ``divergence_score`` = abs(active_pred − candidate_pred) ∈ [0, 1].
    Higher = candidate disagrees more with the production model.
    """

    timestamp: datetime
    active_pred: float
    candidate_pred: float
    divergence_score: float
    actual_outcome: Optional[float] = None  # filled in retroactively

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "active_pred": self.active_pred,
            "candidate_pred": self.candidate_pred,
            "divergence_score": self.divergence_score,
            "actual_outcome": self.actual_outcome,
        }


def _ece(preds: Sequence[float], outcomes: Sequence[float], n_bins: int = 10) -> float:
    """Expected Calibration Error.

    Bins predictions in [0,1] into n_bins equal-width buckets, computes
    |mean_pred − mean_outcome| per bucket, weights by bucket population.
    """
    if not preds:
        return 0.0
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p, o in zip(preds, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, o))
    total = len(preds)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        mp = statistics.mean(p for p, _ in bucket)
        mo = statistics.mean(o for _, o in bucket)
        ece += (len(bucket) / total) * abs(mp - mo)
    return ece


class ShadowExecutor:
    """Runs candidate side-by-side with active; logs only.

    Every ``step()`` emits a ``ShadowDecision`` to the optional sink and
    appends to the in-memory log. The candidate's prediction is NEVER
    used for actual A1 policy emission — that's the safety guarantee.
    """

    def __init__(
        self,
        active: _Predictor,
        candidate: _Predictor,
        *,
        min_samples: int = 100,
        divergence_warn: float = 0.10,
        divergence_reject: float = 0.30,
        sink: Optional[Callable[[ShadowDecision], None]] = None,
    ):
        self._active = active
        self._candidate = candidate
        self._min_samples = min_samples
        self._divergence_warn = divergence_warn
        self._divergence_reject = divergence_reject
        self._sink = sink
        self._log: list[ShadowDecision] = []
        self._lock = threading.RLock()

    def step(self, state) -> ShadowDecision:
        """Run both models on ``state``; log the candidate; emit no policy."""
        a = float(self._active.predict(state))
        c = float(self._candidate.predict(state))
        d = abs(a - c)
        decision = ShadowDecision(
            timestamp=datetime.now(timezone.utc),
            active_pred=a,
            candidate_pred=c,
            divergence_score=d,
        )
        with self._lock:
            self._log.append(decision)
        if self._sink is not None:
            try:
                self._sink(decision)
            except Exception:
                # Sink failure must not block shadow execution.
                pass
        return decision

    def log(self) -> list[ShadowDecision]:
        with self._lock:
            return list(self._log)

    def divergence_histogram(self, n_bins: int = 10) -> list[int]:
        """Return a bin-count histogram of divergence scores in [0, 1]."""
        h = [0] * n_bins
        with self._lock:
            entries = list(self._log)
        for e in entries:
            idx = min(int(e.divergence_score * n_bins), n_bins - 1)
            h[idx] += 1
        return h

    def annotate_outcomes(self, outcomes: Sequence[float]) -> None:
        """Attach actual outcomes to logged decisions, in order."""
        with self._lock:
            n = min(len(outcomes), len(self._log))
            for i in range(n):
                old = self._log[i]
                self._log[i] = ShadowDecision(
                    timestamp=old.timestamp,
                    active_pred=old.active_pred,
                    candidate_pred=old.candidate_pred,
                    divergence_score=old.divergence_score,
                    actual_outcome=float(outcomes[i]),
                )

    def ece_delta(self) -> Optional[float]:
        """ECE(candidate) − ECE(active). Negative → candidate better.

        Returns None if no outcomes have been annotated yet.
        """
        with self._lock:
            entries = [e for e in self._log if e.actual_outcome is not None]
        if not entries:
            return None
        active_preds = [e.active_pred for e in entries]
        cand_preds = [e.candidate_pred for e in entries]
        outs = [float(e.actual_outcome) for e in entries]  # type: ignore[arg-type]
        return _ece(cand_preds, outs) - _ece(active_preds, outs)

    def recommendation(self) -> Literal["promote", "reject", "insufficient-data"]:
        with self._lock:
            entries = list(self._log)
        if len(entries) < self._min_samples:
            return "insufficient-data"
        mean_div = statistics.mean(e.divergence_score for e in entries)
        if mean_div >= self._divergence_reject:
            return "reject"
        # If we have outcomes, also gate on ECE delta.
        ed = self.ece_delta()
        if ed is not None and ed > 0.05:
            # Candidate is materially worse calibrated.
            return "reject"
        if mean_div < self._divergence_warn:
            return "promote"
        # Borderline: divergence high but not extreme. Default to reject
        # — the failure mode of promoting a flaky candidate is worse
        # than waiting for more evidence.
        return "reject"


__all__ = [
    "ShadowDecision",
    "ShadowExecutor",
]

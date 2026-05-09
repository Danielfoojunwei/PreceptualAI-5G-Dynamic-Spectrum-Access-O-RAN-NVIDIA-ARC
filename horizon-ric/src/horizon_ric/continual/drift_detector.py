"""Streaming distribution drift detectors + Prometheus exporters.

Two detectors are exposed:

* :class:`KSdriftDetector` — Kolmogorov-Smirnov 2-sample test over a
  rolling reference window vs. a rolling current window. Fires
  ``drift.fired`` when the K-S p-value drops below ``threshold``.
* :class:`PageHinkleyDetector` — classical Page-Hinkley change-point
  test on the running mean. Cheaper per-sample (O(1)) than K-S
  (O(n log n)) and typically fires sooner on a step-mean shift, which
  matches the auto-retrain trigger story in PLAN-v1 Phase 2.

Both detectors register with :class:`DriftDetectorRegistry`, which
owns the Prometheus counters/gauges:

* ``horizon_drift_fired_total`` (Counter, label ``detector``)
* ``horizon_drift_pvalue`` (Gauge, label ``detector``) — last K-S
  p-value, or ``1 - PH_score / threshold`` for Page-Hinkley.

The registry uses a private ``CollectorRegistry`` so import order can't
cause duplicate-metric errors when the test suite runs alongside the
rApp's main metrics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from prometheus_client import CollectorRegistry, Counter, Gauge
from scipy.stats import ks_2samp


@dataclass
class DriftEvent:
    """Emitted when a detector fires; consumed by auto-retrain hook."""

    detector: str
    sample_index: int
    statistic: float
    pvalue: float | None = None
    payload: dict = field(default_factory=dict)


class _BaseDetector:
    name: str = "base"

    def update(self, x: float) -> DriftEvent | None:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError


class KSdriftDetector(_BaseDetector):
    """Rolling-window K-S 2-sample drift test.

    Maintains a fixed reference window (the first ``window_size``
    samples) and a sliding current window of the most recent
    ``window_size`` samples. Once both windows are full, every new
    sample triggers a K-S test; if ``p < threshold`` for at least
    ``min_consec`` consecutive evaluations, the detector fires.

    The reference window can be re-baselined explicitly via
    :meth:`reset_reference` after a retrain.
    """

    def __init__(
        self,
        window_size: int = 200,
        threshold: float = 1e-3,
        min_consec: int = 1,
        name: str = "ks",
        registry: "DriftDetectorRegistry | None" = None,
    ):
        if window_size < 10:
            raise ValueError("window_size must be >= 10 for K-S to be meaningful")
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be in (0, 1)")
        self.window_size = int(window_size)
        self.threshold = float(threshold)
        self.min_consec = int(min_consec)
        self.name = name
        self._ref: list[float] = []
        self._cur: Deque[float] = deque(maxlen=self.window_size)
        self._n_seen = 0
        self._consec_below = 0
        self.last_pvalue: float | None = None
        self.last_statistic: float | None = None
        self._registry = registry
        if registry is not None:
            registry._register(self)

    def reset(self) -> None:
        self._ref = []
        self._cur.clear()
        self._n_seen = 0
        self._consec_below = 0
        self.last_pvalue = None
        self.last_statistic = None

    def reset_reference(self) -> None:
        """Drop reference; next ``window_size`` samples become new reference."""
        self._ref = []
        self._cur.clear()
        self._consec_below = 0

    def update(self, x: float) -> DriftEvent | None:
        self._n_seen += 1
        if len(self._ref) < self.window_size:
            self._ref.append(float(x))
            return None
        self._cur.append(float(x))
        if len(self._cur) < self.window_size:
            return None

        stat, pvalue = ks_2samp(self._ref, list(self._cur))
        self.last_statistic = float(stat)
        self.last_pvalue = float(pvalue)
        if self._registry is not None:
            self._registry.observe_pvalue(self.name, float(pvalue))

        if pvalue < self.threshold:
            self._consec_below += 1
        else:
            self._consec_below = 0

        if self._consec_below >= self.min_consec:
            self._consec_below = 0
            ev = DriftEvent(
                detector=self.name,
                sample_index=self._n_seen,
                statistic=float(stat),
                pvalue=float(pvalue),
                payload={"window_size": self.window_size, "threshold": self.threshold},
            )
            if self._registry is not None:
                self._registry.fire(self.name, ev)
            return ev
        return None


class PageHinkleyDetector(_BaseDetector):
    """Page-Hinkley test on the running mean.

    Tracks ``m_t = sum_{i<=t} (x_i - mean_running - delta)`` and the
    running min ``M_t``. Fires when ``m_t - M_t > lambda_``.

    Parameters
    ----------
    delta
        Tolerance on benign drift (the test ignores changes smaller
        than ``delta``).
    lambda_
        Detection threshold; larger -> fewer false positives, slower
        detection.
    alpha
        EMA factor for the running mean (``0 < alpha <= 1``).
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        alpha: float = 1.0,
        name: str = "page_hinkley",
        registry: "DriftDetectorRegistry | None" = None,
    ):
        if delta < 0.0:
            raise ValueError("delta must be >= 0")
        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.delta = float(delta)
        self.lambda_ = float(lambda_)
        self.alpha = float(alpha)
        self.name = name
        self._mean = 0.0
        self._n = 0
        self._mt = 0.0
        self._min_mt = 0.0
        self._registry = registry
        if registry is not None:
            registry._register(self)

    def reset(self) -> None:
        self._mean = 0.0
        self._n = 0
        self._mt = 0.0
        self._min_mt = 0.0

    def update(self, x: float) -> DriftEvent | None:
        x = float(x)
        self._n += 1
        if self._n == 1:
            self._mean = x
            self._mt = 0.0
            self._min_mt = 0.0
            return None

        # Running mean (EMA when alpha < 1, plain mean when alpha == 1).
        if self.alpha >= 1.0:
            self._mean += (x - self._mean) / self._n
        else:
            self._mean = (1.0 - self.alpha) * self._mean + self.alpha * x

        self._mt += x - self._mean - self.delta
        if self._mt < self._min_mt:
            self._min_mt = self._mt

        score = self._mt - self._min_mt
        if self._registry is not None:
            # Map "score" into a pseudo-pvalue gauge in [0, 1] for
            # Grafana: 1 = far below threshold, 0 = at/above threshold.
            pseudo_p = max(0.0, 1.0 - score / self.lambda_)
            self._registry.observe_pvalue(self.name, pseudo_p)

        if score > self.lambda_:
            ev = DriftEvent(
                detector=self.name,
                sample_index=self._n,
                statistic=score,
                pvalue=None,
                payload={"lambda": self.lambda_, "delta": self.delta},
            )
            # Re-baseline so we can detect the next change without
            # latching forever.
            self._mt = 0.0
            self._min_mt = 0.0
            if self._registry is not None:
                self._registry.fire(self.name, ev)
            return ev
        return None


class DriftDetectorRegistry:
    """Owns Prometheus counters/gauges shared by every detector.

    Use a private ``CollectorRegistry`` so instantiating multiple
    registries (e.g. in tests) does not raise the global Prometheus
    duplicate-metric error.
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry if registry is not None else CollectorRegistry()
        self.counter = Counter(
            "horizon_drift_fired_total",
            "Total number of drift events fired, by detector.",
            labelnames=("detector",),
            registry=self.registry,
        )
        self.pvalue_gauge = Gauge(
            "horizon_drift_pvalue",
            "Last drift p-value (or pseudo-p for Page-Hinkley).",
            labelnames=("detector",),
            registry=self.registry,
        )
        self._detectors: dict[str, _BaseDetector] = {}
        self._events: list[DriftEvent] = []

    def _register(self, det: _BaseDetector) -> None:
        self._detectors[det.name] = det
        # Initialise child labels so /metrics shows zero-valued series
        # before the first event/observation.
        self.counter.labels(detector=det.name)
        self.pvalue_gauge.labels(detector=det.name).set(1.0)

    def observe_pvalue(self, name: str, p: float) -> None:
        self.pvalue_gauge.labels(detector=name).set(float(p))

    def fire(self, name: str, ev: DriftEvent) -> None:
        self.counter.labels(detector=name).inc()
        self._events.append(ev)

    @property
    def events(self) -> list[DriftEvent]:
        return list(self._events)

    def fired_count(self, name: str) -> int:
        # prometheus_client exposes ._value.get() for tests; safer to
        # iterate the registry directly.
        for metric in self.registry.collect():
            if metric.name == "horizon_drift_fired":
                for sample in metric.samples:
                    if (
                        sample.name == "horizon_drift_fired_total"
                        and sample.labels.get("detector") == name
                    ):
                        return int(sample.value)
        return 0


__all__ = [
    "DriftEvent",
    "KSdriftDetector",
    "PageHinkleyDetector",
    "DriftDetectorRegistry",
]

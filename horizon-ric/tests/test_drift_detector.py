"""Tests for streaming drift detectors (`horizon_ric.continual.drift_detector`)."""

from __future__ import annotations

import numpy as np
import pytest

from horizon_ric.continual.drift_detector import (
    DriftDetectorRegistry,
    KSdriftDetector,
    PageHinkleyDetector,
)


def _stream(detector, samples) -> int | None:
    """Feed samples to a detector; return index of first fired event, or None."""
    for i, x in enumerate(samples):
        ev = detector.update(float(x))
        if ev is not None:
            return i
    return None


def test_ks_fires_on_synthetic_shift_within_500_samples():
    """K-S over a 200-sample window must catch a mean+variance shift within
    the next 500 samples after the change."""
    rng = np.random.default_rng(0)
    reg = DriftDetectorRegistry()
    det = KSdriftDetector(window_size=200, threshold=1e-3, name="ks_test", registry=reg)

    # Reference: 200 samples N(0, 1).
    ref = rng.normal(0.0, 1.0, size=200)
    # Stable middle: 200 samples N(0, 1) so the rolling current window
    # is fully overwritten with stable data.
    stable = rng.normal(0.0, 1.0, size=200)
    # Shift: N(2.0, 1.5).
    shifted = rng.normal(2.0, 1.5, size=500)

    fire_idx = _stream(det, np.concatenate([ref, stable, shifted]))
    assert fire_idx is not None, "K-S must fire on synthetic shift"
    # Must fire within 500 samples after the shift starts.
    shift_start = len(ref) + len(stable)
    assert fire_idx - shift_start < 500
    # And the registry counter must reflect at least one event.
    assert reg.counter.labels(detector="ks_test")._value.get() >= 1.0


def test_ks_stable_distribution_low_false_positive_rate_over_5000_samples():
    """Under a stable distribution the FPR (number of fires / number of
    K-S evaluations) must stay below 5%."""
    rng = np.random.default_rng(123)
    reg = DriftDetectorRegistry()
    det = KSdriftDetector(window_size=200, threshold=1e-3, name="ks_stable", registry=reg)

    n = 5000
    samples = rng.normal(0.5, 2.0, size=n)
    n_fires = 0
    for x in samples:
        if det.update(float(x)) is not None:
            n_fires += 1
    # Number of K-S evaluations = n - 2*window_size (after both windows full).
    n_evals = max(1, n - 2 * 200)
    fpr = n_fires / n_evals
    assert fpr < 0.05, f"FPR {fpr:.3f} exceeds 5% on stable data"


def test_page_hinkley_fires_faster_than_ks_on_step_shift():
    """On a clean mean step Page-Hinkley should fire sooner than K-S
    (it is O(1) per sample and tracks the running mean directly)."""
    rng = np.random.default_rng(7)
    pre = rng.normal(0.0, 1.0, size=500)
    post = rng.normal(2.0, 1.0, size=1500)
    stream = np.concatenate([pre, post])

    ks = KSdriftDetector(window_size=200, threshold=1e-3, name="ks_cmp")
    ph = PageHinkleyDetector(delta=0.005, lambda_=50.0, name="ph_cmp")

    ks_fire = _stream(ks, stream)
    ph_fire = _stream(ph, stream)
    assert ks_fire is not None and ph_fire is not None
    # Page-Hinkley should fire faster (smaller index) than K-S.
    assert ph_fire < ks_fire, (
        f"expected Page-Hinkley to beat K-S; got ph={ph_fire}, ks={ks_fire}"
    )


def test_registry_exposes_prometheus_counters_and_gauges():
    """horizon_drift_fired_total / horizon_drift_pvalue must be exposed
    via the registry's collectors with the per-detector label."""
    reg = DriftDetectorRegistry()
    det = KSdriftDetector(window_size=50, threshold=1e-3, name="reg_test", registry=reg)

    rng = np.random.default_rng(2)
    # Drive to ensure a pvalue gets observed.
    for _ in range(60):
        det.update(float(rng.normal(0, 1)))
    # Inspect Prometheus collectors directly.
    metric_names = {m.name for m in reg.registry.collect()}
    assert "horizon_drift_fired" in metric_names
    assert "horizon_drift_pvalue" in metric_names

    # Force a fire, then the counter must increment.
    rng2 = np.random.default_rng(3)
    for _ in range(500):
        det.update(float(rng2.normal(5.0, 0.1)))
    val = reg.counter.labels(detector="reg_test")._value.get()
    assert val >= 1.0


def test_page_hinkley_reset_clears_state():
    """After reset() the detector must behave like a fresh instance."""
    ph = PageHinkleyDetector(delta=0.0, lambda_=10.0)
    rng = np.random.default_rng(5)
    for _ in range(100):
        ph.update(float(rng.normal(3.0, 0.1)))
    assert ph._n == 100
    ph.reset()
    assert ph._n == 0 and ph._mt == 0.0 and ph._min_mt == 0.0

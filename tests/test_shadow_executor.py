"""Shadow executor tests."""

from __future__ import annotations

import random
from dataclasses import dataclass

from horizon_ric.runtime.shadow_executor import (
    ShadowDecision,
    ShadowExecutor,
)

# ─── Toy predictors ───────────────────────────────────────────────────


@dataclass
class _Linear:
    slope: float
    bias: float

    def predict(self, state: float) -> float:
        v = self.slope * state + self.bias
        return min(max(v, 0.0), 1.0)


# ─── Tests ────────────────────────────────────────────────────────────


class TestShadowEmitsNoPolicy:
    def test_step_returns_decision_only(self):
        active = _Linear(0.5, 0.2)
        candidate = _Linear(0.5, 0.3)
        ex = ShadowExecutor(active, candidate)
        d = ex.step(0.5)
        assert isinstance(d, ShadowDecision)
        # Critical safety: ShadowExecutor exposes no `emit_policy()` method.
        assert not hasattr(ex, "emit_policy")
        assert not hasattr(ex, "send_a1")


class TestDivergenceHistogram:
    def test_aggregates_across_steps(self):
        active = _Linear(1.0, 0.0)
        candidate = _Linear(1.0, 0.05)  # slight constant offset
        ex = ShadowExecutor(active, candidate)
        rng = random.Random(0)
        for _ in range(1000):
            ex.step(rng.random())
        hist = ex.divergence_histogram(n_bins=10)
        assert sum(hist) == 1000
        # Most divergence should be in bin 0 (0.0–0.1)
        assert hist[0] > 800

    def test_divergent_candidate_widens_histogram(self):
        active = _Linear(1.0, 0.0)
        candidate = _Linear(0.0, 0.5)  # constant 0.5 — wildly divergent
        ex = ShadowExecutor(active, candidate)
        rng = random.Random(1)
        for _ in range(500):
            ex.step(rng.random())
        hist = ex.divergence_histogram(n_bins=10)
        # Mass should be spread across multiple bins
        nonzero = sum(1 for c in hist if c > 0)
        assert nonzero >= 3


class TestRecommendation:
    def test_insufficient_data_below_min_samples(self):
        ex = ShadowExecutor(_Linear(1, 0), _Linear(1, 0), min_samples=100)
        for _ in range(20):
            ex.step(0.5)
        assert ex.recommendation() == "insufficient-data"

    def test_promote_when_low_divergence(self):
        ex = ShadowExecutor(
            _Linear(1, 0),
            _Linear(1, 0.001),  # near-identical
            min_samples=50,
            divergence_warn=0.05,
            divergence_reject=0.20,
        )
        rng = random.Random(0)
        for _ in range(200):
            ex.step(rng.random())
        assert ex.recommendation() == "promote"

    def test_reject_when_high_divergence(self):
        ex = ShadowExecutor(
            _Linear(1, 0),
            _Linear(0, 0.5),  # candidate is constant 0.5; very divergent
            min_samples=50,
        )
        rng = random.Random(0)
        for _ in range(200):
            ex.step(rng.random())
        assert ex.recommendation() == "reject"


class TestECEDelta:
    def test_ece_delta_none_without_outcomes(self):
        ex = ShadowExecutor(_Linear(1, 0), _Linear(1, 0))
        for _ in range(10):
            ex.step(0.5)
        assert ex.ece_delta() is None

    def test_ece_delta_negative_when_candidate_better(self):
        # Active is biased toward 0.5; candidate is well-calibrated.
        # Outcomes match candidate's predictions.
        ex = ShadowExecutor(_Linear(0.0, 0.5), _Linear(1.0, 0.0))
        rng = random.Random(0)
        states_outcomes = []
        for _ in range(200):
            s = rng.random()
            ex.step(s)
            # Outcome equals candidate's prediction.
            states_outcomes.append(s)
        ex.annotate_outcomes(states_outcomes)
        delta = ex.ece_delta()
        assert delta is not None
        assert delta < 0  # candidate is better calibrated


class TestSinkAndLog:
    def test_sink_invoked_on_each_step(self):
        captured: list[ShadowDecision] = []
        ex = ShadowExecutor(_Linear(1, 0), _Linear(1, 0.1), sink=captured.append)
        for s in [0.1, 0.5, 0.9]:
            ex.step(s)
        assert len(captured) == 3

    def test_sink_failure_does_not_block_step(self):
        def boom(_d):
            raise RuntimeError("audit chain unreachable")

        ex = ShadowExecutor(_Linear(1, 0), _Linear(1, 0.1), sink=boom)
        d = ex.step(0.5)
        # Step still returned a decision; log still appended.
        assert isinstance(d, ShadowDecision)
        assert len(ex.log()) == 1


class TestSerialisation:
    def test_to_dict_shape(self):
        ex = ShadowExecutor(_Linear(1, 0), _Linear(1, 0.1))
        d = ex.step(0.5)
        out = d.to_dict()
        for k in ("timestamp", "active_pred", "candidate_pred", "divergence_score"):
            assert k in out

"""Tests for the VIAVI Pipeline 2 digital-twin counterfactual driver.

Covers:

  * synthetic backend rollout, no real bench needed
  * divergence-from-actual arithmetic
  * confidence monotonicity & cap
  * backend-name discrimination
  * integration with `policy/counterfactual.py` via the new
    ``external_twin`` hook (existing test patterns must still pass —
    that is asserted by re-running the historical assertions here in
    the no-twin path)
  * synthetic determinism
  * n_steps validation
  * mock external backend integration
  * 2 additional edge cases: actual_kpis with key the twin didn't emit,
    confidence_floor honoured at very small n_steps.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from horizon_ric.integrations.viavi_digital_twin import (
    CounterfactualRollout,
    DigitalTwinBackend,
    ViaviDigitalTwin,
)
from horizon_ric.policy.counterfactual import (
    AlternativeCandidate,
    build_rejected_alternatives,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingBackend:
    """Mock DigitalTwinBackend that records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict, int]] = []

    async def evaluate(
        self, state: dict, action: dict, n_steps: int,
    ) -> dict:
        self.calls.append((dict(state), dict(action), int(n_steps)))
        # Pretend KPI: throughput proportional to prb_count, latency 10 ms.
        return {
            "throughput_mbps": float(action.get("prb_count", 0)) * 2.0,
            "latency_ms": 10.0,
            "sla_risk": 0.05,
        }


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. synthetic backend rollout
# ---------------------------------------------------------------------------


def test_synthetic_backend_rollout_basic() -> None:
    twin = ViaviDigitalTwin()  # backend=None → synthetic
    out = _run(
        twin.rollout(
            state={"snr_db": 10.0, "n_ues": 2.0},
            action_chosen={"prb_count": 5},
            action_alternative={"prb_count": 10},
            n_steps=10,
        )
    )
    assert isinstance(out, CounterfactualRollout)
    assert out.predicted_kpis_chosen, "chosen KPIs must be non-empty"
    assert out.predicted_kpis_alternative, "alt KPIs must be non-empty"
    assert "throughput_mbps" in out.predicted_kpis_chosen
    # More PRBs ⇒ higher throughput in the synthetic rules.
    assert (
        out.predicted_kpis_alternative["throughput_mbps"]
        > out.predicted_kpis_chosen["throughput_mbps"]
    )
    assert out.n_steps == 10
    assert out.backend_name == "synthetic"


# ---------------------------------------------------------------------------
# 2. divergence computed when actual_kpis provided
# ---------------------------------------------------------------------------


def test_divergence_computed_when_actual_provided() -> None:
    twin = ViaviDigitalTwin()
    state = {"snr_db": 5.0}
    action = {"prb_count": 4}
    # Run once with no actual to grab the prediction.
    base = _run(twin.rollout(state, action, action, n_steps=10))
    pred = base.predicted_kpis_chosen
    # Build an "actual" that intentionally drifts by +1.0 on throughput,
    # +2.0 on latency, ignore others.
    actual = {
        "throughput_mbps": pred["throughput_mbps"] + 1.0,
        "latency_ms": pred["latency_ms"] + 2.0,
    }
    out = _run(twin.rollout(state, action, action, n_steps=10, actual_kpis=actual))
    assert out.divergence_from_actual is not None
    assert math.isclose(out.divergence_from_actual, 3.0, rel_tol=1e-6)


def test_divergence_is_none_when_no_actual() -> None:
    twin = ViaviDigitalTwin()
    out = _run(
        twin.rollout(
            {"snr_db": 1.0}, {"prb_count": 1}, {"prb_count": 2}, n_steps=5,
        )
    )
    assert out.divergence_from_actual is None


# ---------------------------------------------------------------------------
# 3. confidence increases with n_steps and is capped
# ---------------------------------------------------------------------------


def test_confidence_increases_with_n_steps() -> None:
    twin = ViaviDigitalTwin()
    state = {"snr_db": 1.0}
    a = {"prb_count": 1}
    out_small = _run(twin.rollout(state, a, a, n_steps=10))
    out_med = _run(twin.rollout(state, a, a, n_steps=100))
    out_large = _run(twin.rollout(state, a, a, n_steps=1000))
    assert out_small.confidence < out_med.confidence < out_large.confidence + 1e-9
    # Cap at 0.95.
    assert out_large.confidence <= 0.95
    assert out_large.confidence == pytest.approx(0.95, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. backend_name reflects backend
# ---------------------------------------------------------------------------


def test_backend_name_synthetic_when_no_backend() -> None:
    twin = ViaviDigitalTwin()
    assert twin.backend_name == "synthetic"
    out = _run(twin.rollout({}, {}, {}, n_steps=1))
    assert out.backend_name == "synthetic"


def test_backend_name_pipeline2_when_backend_provided() -> None:
    backend = _RecordingBackend()
    # Protocol check is duck-typed; ensure runtime_checkable accepts it.
    assert isinstance(backend, DigitalTwinBackend)
    twin = ViaviDigitalTwin(backend=backend)
    assert twin.backend_name == "viavi_pipeline_2"
    out = _run(twin.rollout({}, {"prb_count": 3}, {"prb_count": 7}, n_steps=8))
    assert out.backend_name == "viavi_pipeline_2"
    # Backend was called twice (chosen + alternative).
    assert len(backend.calls) == 2
    chosen_call, alt_call = backend.calls
    assert chosen_call[1] == {"prb_count": 3}
    assert alt_call[1] == {"prb_count": 7}
    assert chosen_call[2] == alt_call[2] == 8


# ---------------------------------------------------------------------------
# 5. integration with policy/counterfactual.py
# ---------------------------------------------------------------------------


def test_external_twin_hook_in_build_rejected_alternatives() -> None:
    """The new external_twin hook routes evaluation through the twin
    without breaking the historical no-twin path."""
    cands = [
        AlternativeCandidate(
            action={"prb_count": 1.0},
            reward_sum=-5.0,
            constraint_violation_sum=2.0,
            primary_metric="epfd",
            primary_value=2.0,
            threshold=0.0,
            horizon="60s",
        ),
        AlternativeCandidate(
            action={"prb_count": 8.0},
            reward_sum=-1.0,
            constraint_violation_sum=0.0,
            sla_risk_30s=0.30,
            primary_metric="sla",
            primary_value=0.30,
            threshold=0.20,
            horizon="30s",
        ),
    ]
    # No twin → existing behaviour, byte-identical to legacy.
    legacy = build_rejected_alternatives(cands, chosen_score=0.0)
    assert len(legacy) == 2
    causes = {a.rejection_reason_machine.primary_cause for a in legacy}
    assert "constraint_violation_hard" in causes
    assert "sla_breach_predicted" in causes

    # With twin → routed through ViaviDigitalTwin, sla_risk overridden
    # by the synthetic backend output (single value mirrored to all 3
    # horizons).
    twin = ViaviDigitalTwin()
    routed = build_rejected_alternatives(
        cands,
        chosen_score=0.0,
        external_twin=twin,
        state={"snr_db": 10.0},
        chosen_action={"prb_count": 4.0},
        twin_n_steps=20,
    )
    assert len(routed) == 2
    # Twin overrode sla_risk: all three horizons must now be equal.
    for r in routed:
        po = r.predicted_outcome
        assert po.sla_risk_30s == po.sla_risk_1min == po.sla_risk_5min
        assert 0.0 <= po.sla_risk_30s <= 1.0


def test_external_twin_requires_state_and_chosen_action() -> None:
    twin = ViaviDigitalTwin()
    with pytest.raises(ValueError):
        build_rejected_alternatives(
            [
                AlternativeCandidate(
                    action={"prb_count": 1.0},
                    reward_sum=-1.0,
                    constraint_violation_sum=0.0,
                ),
            ],
            chosen_score=0.0,
            external_twin=twin,
        )


# ---------------------------------------------------------------------------
# 6. determinism
# ---------------------------------------------------------------------------


def test_synthetic_rollout_is_deterministic() -> None:
    twin = ViaviDigitalTwin()
    state = {"snr_db": 7.0, "n_ues": 3.0}
    a = {"prb_count": 5}
    b = {"prb_count": 9}
    out1 = _run(twin.rollout(state, a, b, n_steps=42))
    out2 = _run(twin.rollout(state, a, b, n_steps=42))
    assert out1.predicted_kpis_chosen == out2.predicted_kpis_chosen
    assert out1.predicted_kpis_alternative == out2.predicted_kpis_alternative
    assert out1.confidence == out2.confidence


# ---------------------------------------------------------------------------
# 7. n_steps validation
# ---------------------------------------------------------------------------


def test_n_steps_zero_raises() -> None:
    twin = ViaviDigitalTwin()
    with pytest.raises(ValueError):
        _run(twin.rollout({}, {}, {}, n_steps=0))


def test_n_steps_negative_raises() -> None:
    twin = ViaviDigitalTwin()
    with pytest.raises(ValueError):
        _run(twin.rollout({}, {}, {}, n_steps=-3))


# ---------------------------------------------------------------------------
# 8. mock external backend integration (Protocol)
# ---------------------------------------------------------------------------


def test_mock_backend_is_called_for_each_action() -> None:
    backend = _RecordingBackend()
    twin = ViaviDigitalTwin(backend=backend)
    out = _run(
        twin.rollout(
            state={"snr_db": 10.0},
            action_chosen={"prb_count": 2},
            action_alternative={"prb_count": 6},
            n_steps=15,
        )
    )
    # Exactly 2 backend calls per rollout (chosen + alternative).
    assert len(backend.calls) == 2
    assert out.predicted_kpis_chosen["throughput_mbps"] == 4.0
    assert out.predicted_kpis_alternative["throughput_mbps"] == 12.0


# ---------------------------------------------------------------------------
# 9. edge case: actual_kpis with a key the prediction omits
# ---------------------------------------------------------------------------


def test_divergence_handles_missing_predicted_key() -> None:
    """If actual contains a KPI the prediction did not emit, divergence
    is still well-defined (treated as full-actual gap)."""
    backend = _RecordingBackend()  # emits throughput_mbps, latency_ms, sla_risk
    twin = ViaviDigitalTwin(backend=backend)
    actual = {
        "throughput_mbps": 4.0,        # backend will return 4.0 for prb_count=2
        "latency_ms": 10.0,             # exact match
        "energy_w": 7.5,                # backend doesn't emit this — full gap
    }
    out = _run(
        twin.rollout(
            state={"snr_db": 10.0},
            action_chosen={"prb_count": 2},
            action_alternative={"prb_count": 2},
            n_steps=5,
            actual_kpis=actual,
        )
    )
    assert out.divergence_from_actual is not None
    # 0 (throughput) + 0 (latency) + 7.5 (missing energy_w treated as gap).
    assert math.isclose(out.divergence_from_actual, 7.5, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 10. edge case: confidence_floor is honoured at small n_steps
# ---------------------------------------------------------------------------


def test_confidence_floor_honoured() -> None:
    twin_low_floor = ViaviDigitalTwin(confidence_floor=0.0)
    twin_high_floor = ViaviDigitalTwin(confidence_floor=0.5)
    state, a = {"snr_db": 1.0}, {"prb_count": 1}
    out_low = _run(twin_low_floor.rollout(state, a, a, n_steps=1))
    out_high = _run(twin_high_floor.rollout(state, a, a, n_steps=1))
    # n_steps=1 → raw confidence ≈ 0.0099. Floor must rescue the high case.
    assert out_low.confidence < 0.05
    assert out_high.confidence == pytest.approx(0.5, abs=1e-9)


def test_confidence_floor_validation() -> None:
    with pytest.raises(ValueError):
        ViaviDigitalTwin(confidence_floor=0.95)
    with pytest.raises(ValueError):
        ViaviDigitalTwin(confidence_floor=-0.1)

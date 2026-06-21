"""Tests for the graceful-degradation FSM.

We simulate each failure-mode by directly invoking the controller's
`enter()` (which is what the rApp's main loop does when it observes a
real failure such as a CircuitBreakerError on R1). The test then asserts
the documented degraded behaviour, and that recovery flips the state
back and surfaces the buffered work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from horizon_ric.runtime.graceful_degradation import (
    DegradationController,
    DegradedState,
    FailureMode,
)


def test_initial_state_is_normal_and_serving():
    ctl = DegradationController()
    assert ctl.state == DegradedState.NORMAL
    assert ctl.is_serving() is True


def test_smo_unreachable_enters_keep_last_good():
    ctl = DegradationController()
    ctl.enter(FailureMode.SMO_UNREACHABLE, reason="cb_open")
    assert ctl.state == DegradedState.KEEP_LAST_GOOD
    assert ctl.failure_mode == FailureMode.SMO_UNREACHABLE
    assert ctl.is_serving() is False  # /readyz must return 503

    # Simulate caching last-good policies.
    ctl.record_last_good_policy(
        "horizon.qos.priority", "policy-1", {"scope": {"slice_id": "x"}}
    )
    cached = ctl.get_last_good_policy("horizon.qos.priority")
    assert cached is not None
    assert cached["policy_id"] == "policy-1"


def test_evidence_store_down_buffers_to_tmp(tmp_path):
    buf = tmp_path / "evidence-buffer.jsonl"
    ctl = DegradationController(tmp_buffer_path=buf)
    ctl.enter(FailureMode.EVIDENCE_STORE_DOWN, reason="disk_full")
    assert ctl.state == DegradedState.BUFFER_TO_TMP
    assert ctl.is_serving() is False

    rec = {"decision_id": "d1", "ts": 1.0}
    ctl.buffer_evidence(rec)
    ctl.buffer_evidence({"decision_id": "d2", "ts": 2.0})
    assert buf.exists()

    lines = buf.read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["decision_id"] == "d1"
    assert parsed[1]["decision_id"] == "d2"


def test_telemetry_dropped_holds_output():
    ctl = DegradationController()
    ctl.enter(FailureMode.TELEMETRY_DROPPED, reason="encoder_stalled")
    assert ctl.state == DegradedState.HOLD_OUTPUT
    assert ctl.is_serving() is False

    ctl.hold_decision({"decision_id": "d1"})
    ctl.hold_decision({"decision_id": "d2"})
    snap = ctl.snapshot()
    assert snap.held_decisions == 2


def test_recover_resets_state_and_returns_buffered_work(tmp_path):
    ctl = DegradationController(tmp_buffer_path=tmp_path / "buf.jsonl")
    ctl.enter(FailureMode.EVIDENCE_STORE_DOWN)
    ctl.buffer_evidence({"d": 1})
    ctl.buffer_evidence({"d": 2})

    ctl.recover(reason="store_back")
    assert ctl.state == DegradedState.NORMAL
    assert ctl.is_serving() is True
    assert ctl.failure_mode is None

    drained = ctl.drain_buffered_evidence()
    assert len(drained) == 2

    # Drain is now empty
    assert ctl.drain_buffered_evidence() == []


def test_recover_when_already_normal_is_idempotent():
    ctl = DegradationController()
    assert ctl.recover() == DegradedState.NORMAL
    assert ctl.state == DegradedState.NORMAL


def test_enter_is_idempotent_for_same_mode():
    ctl = DegradationController()
    ctl.enter(FailureMode.SMO_UNREACHABLE)
    first_entered_at = ctl.snapshot().entered_at
    ctl.enter(FailureMode.SMO_UNREACHABLE)  # again
    second = ctl.snapshot().entered_at
    assert first_entered_at == second  # didn't reset clock


def test_transition_between_failure_modes_updates_state():
    ctl = DegradationController()
    ctl.enter(FailureMode.SMO_UNREACHABLE)
    assert ctl.state == DegradedState.KEEP_LAST_GOOD
    ctl.enter(FailureMode.TELEMETRY_DROPPED)
    assert ctl.state == DegradedState.HOLD_OUTPUT


def test_snapshot_is_consistent():
    ctl = DegradationController()
    ctl.enter(FailureMode.SMO_UNREACHABLE)
    ctl.record_last_good_policy("t", "id", {"x": 1})
    snap = ctl.snapshot()
    assert snap.state == "keep_last_good"
    assert snap.failure_mode == "smo_unreachable"
    assert snap.last_good_policies == 1
    assert snap.entered_at is not None


def test_recovery_cycle_full_round_trip():
    """SMO down → recover → telemetry drops → recover."""
    ctl = DegradationController()
    ctl.enter(FailureMode.SMO_UNREACHABLE)
    ctl.recover()
    assert ctl.state == DegradedState.NORMAL

    ctl.enter(FailureMode.TELEMETRY_DROPPED)
    assert ctl.state == DegradedState.HOLD_OUTPUT
    ctl.hold_decision({"d": 1})
    ctl.recover()
    assert ctl.state == DegradedState.NORMAL
    held = ctl.drain_held_decisions()
    assert held == [{"d": 1}]

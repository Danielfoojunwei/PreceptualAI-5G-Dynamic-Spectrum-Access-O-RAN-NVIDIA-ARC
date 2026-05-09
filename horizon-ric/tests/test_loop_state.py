"""TS 28.567 LoopState machine tests."""

from __future__ import annotations

import threading

import pytest

from horizon_ric.runtime.loop_state import (
    InvalidTransitionError,
    LoopState,
    LoopStateMachine,
    StateTransition,
)


class TestValidTransitions:
    def test_full_happy_path_idle_to_monitor(self):
        m = LoopStateMachine()
        m.transition(LoopState.RETRAIN, "drift detected")
        m.transition(LoopState.VALIDATE, "training complete")
        m.transition(LoopState.PROMOTE, "validation pass")
        m.transition(LoopState.MONITOR, "deployed")
        assert m.current_state() is LoopState.MONITOR
        assert len(m.history()) == 4

    def test_validate_to_idle_rejection_path(self):
        m = LoopStateMachine()
        m.transition(LoopState.RETRAIN, "drift")
        m.transition(LoopState.VALIDATE, "trained")
        m.transition(LoopState.IDLE, "rejected at validation")
        assert m.current_state() is LoopState.IDLE

    def test_monitor_to_rollback_path(self):
        m = LoopStateMachine()
        for to_state, reason in [
            (LoopState.RETRAIN, "x"),
            (LoopState.VALIDATE, "x"),
            (LoopState.PROMOTE, "x"),
            (LoopState.MONITOR, "x"),
            (LoopState.ROLLBACK, "p99 spike"),
            (LoopState.IDLE, "rolled back"),
        ]:
            m.transition(to_state, reason)
        assert m.current_state() is LoopState.IDLE

    def test_monitor_to_retrain_drift_path(self):
        m = LoopStateMachine()
        for to_state in (
            LoopState.RETRAIN,
            LoopState.VALIDATE,
            LoopState.PROMOTE,
            LoopState.MONITOR,
            LoopState.RETRAIN,  # drift triggers retune
        ):
            m.transition(to_state, "x")
        assert m.current_state() is LoopState.RETRAIN


class TestInvalidTransitions:
    def test_idle_to_promote_rejected(self):
        m = LoopStateMachine()
        with pytest.raises(InvalidTransitionError):
            m.transition(LoopState.PROMOTE, "skip")

    def test_idle_to_monitor_rejected(self):
        m = LoopStateMachine()
        with pytest.raises(InvalidTransitionError):
            m.transition(LoopState.MONITOR, "skip")

    def test_retrain_to_promote_rejected(self):
        m = LoopStateMachine()
        m.transition(LoopState.RETRAIN, "x")
        with pytest.raises(InvalidTransitionError):
            m.transition(LoopState.PROMOTE, "skip")

    def test_promote_to_idle_rejected(self):
        m = LoopStateMachine()
        for s in (LoopState.RETRAIN, LoopState.VALIDATE, LoopState.PROMOTE):
            m.transition(s, "x")
        with pytest.raises(InvalidTransitionError):
            m.transition(LoopState.IDLE, "skip")

    def test_rollback_to_monitor_rejected(self):
        m = LoopStateMachine()
        for s in (
            LoopState.RETRAIN,
            LoopState.VALIDATE,
            LoopState.PROMOTE,
            LoopState.MONITOR,
            LoopState.ROLLBACK,
        ):
            m.transition(s, "x")
        with pytest.raises(InvalidTransitionError):
            m.transition(LoopState.MONITOR, "skip")


class TestHistoryAndReplay:
    def test_history_is_append_only(self):
        m = LoopStateMachine()
        m.transition(LoopState.RETRAIN, "x")
        h1 = m.history()
        m.transition(LoopState.VALIDATE, "x")
        h2 = m.history()
        assert len(h1) == 1
        assert len(h2) == 2
        assert h2[0] == h1[0]

    def test_replay_reconstructs_current_state(self):
        m = LoopStateMachine()
        for s in (LoopState.RETRAIN, LoopState.VALIDATE, LoopState.PROMOTE):
            m.transition(s, "x")
        assert m.replay() == m.current_state()

    def test_history_returns_copy(self):
        m = LoopStateMachine()
        m.transition(LoopState.RETRAIN, "x")
        h = m.history()
        h.clear()  # mutate caller's copy
        assert len(m.history()) == 1  # internal unchanged


class TestSinkAndAuditChain:
    def test_sink_called_on_every_transition(self):
        captured: list[StateTransition] = []
        m = LoopStateMachine(sink=captured.append)
        m.transition(LoopState.RETRAIN, "x")
        m.transition(LoopState.VALIDATE, "x")
        assert len(captured) == 2
        assert captured[0].to_state is LoopState.RETRAIN
        assert captured[1].from_state is LoopState.RETRAIN

    def test_sink_raise_rolls_back_transition(self):
        def raising_sink(_t):
            raise RuntimeError("audit chain unreachable")

        m = LoopStateMachine(sink=raising_sink)
        with pytest.raises(RuntimeError):
            m.transition(LoopState.RETRAIN, "x")
        # State unchanged, history empty
        assert m.current_state() is LoopState.IDLE
        assert m.history() == []


class TestConcurrency:
    def test_concurrent_transitions_serialised(self):
        m = LoopStateMachine()
        # Run a long happy-path under thread contention
        errors: list[Exception] = []

        def worker():
            try:
                # Each worker tries the full path; only one can succeed
                m.transition(LoopState.RETRAIN, "t")
            except InvalidTransitionError:
                pass  # expected for losers
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Exactly one IDLE→RETRAIN succeeded; current state is RETRAIN
        assert m.current_state() is LoopState.RETRAIN
        assert len(m.history()) == 1


class TestSerialisation:
    def test_state_transition_to_dict_roundtrip(self):
        m = LoopStateMachine()
        t = m.transition(LoopState.RETRAIN, "drift", actor="auto", model_sha="ab" * 32)
        d = t.to_dict()
        assert d["from"] == "Idle"
        assert d["to"] == "Retrain"
        assert d["reason"] == "drift"
        assert d["actor"] == "auto"
        assert d["model_sha"] == "ab" * 32
        assert "T" in d["timestamp"]

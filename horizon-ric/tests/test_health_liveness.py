"""Tests for the real /healthz and /readyz liveness checks.

Closes Devil-B findings #3 (hard-coded healthz=200) and #13 (no proper
liveness assertion). The signals come from `LivenessRegistry`; the
FastAPI route in `horizon_ric.rapp.health.build_health_app` reads from
that registry.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from horizon_ric.rapp.health import build_health_app
from horizon_ric.rapp.lifecycle import RAppState
from horizon_ric.runtime.liveness import LivenessRegistry


class _FakeLifecycle:
    """Stand-in lifecycle that exposes the bits health.py reads."""

    def __init__(
        self,
        state: RAppState = RAppState.RUNNING,
        liveness: LivenessRegistry | None = None,
    ) -> None:
        self.state = state
        self.liveness = liveness or LivenessRegistry()
        self.degradation = None  # No DegradationController in these tests.


def _client(lifecycle: _FakeLifecycle) -> TestClient:
    return TestClient(build_health_app(lifecycle))


# A controllable monotonic clock for time-based tests.
class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---------- /healthz ----------

def test_healthz_all_green_returns_200() -> None:
    """Boot → first probe → all signals fresh → 200."""
    reg = LivenessRegistry()
    lc = _FakeLifecycle(RAppState.RUNNING, reg)
    resp = _client(lc).get("/healthz")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["failed"] == []
    assert "watchdog_age_s" in body["checks"]


def test_healthz_503_when_watchdog_stale() -> None:
    """Watchdog hasn't ticked in > 30s → /healthz 503 with watchdog_stale."""
    clk = _FakeClock()
    reg = LivenessRegistry(time_source=clk, watchdog_max_age_s=30.0)
    reg.mark_watchdog_tick()  # record tick at t=1000
    clk.advance(31.0)  # 31s later — exceeds threshold
    lc = _FakeLifecycle(RAppState.RUNNING, reg)
    resp = _client(lc).get("/healthz")
    assert resp.status_code == 503
    body = resp.json()
    assert "watchdog_stale" in body["failed"]
    assert body["status"] == "unhealthy"


def test_healthz_503_when_loop_lag_high() -> None:
    """Recorded loop lag > threshold → /healthz 503 with loop_lag_high."""
    reg = LivenessRegistry(loop_lag_max_s=0.1)
    reg.record_loop_lag(0.5)  # 500 ms — five times the threshold
    lc = _FakeLifecycle(RAppState.RUNNING, reg)
    resp = _client(lc).get("/healthz")
    assert resp.status_code == 503
    assert "loop_lag_high" in resp.json()["failed"]


def test_healthz_503_when_planner_stalled() -> None:
    """Planner has emitted heartbeats but nothing in 5×SLA → 503."""
    clk = _FakeClock()
    reg = LivenessRegistry(
        time_source=clk,
        planner_sla_s=5.0,
        planner_max_missed=3,  # silence_max = 15s
    )
    reg.mark_planner_heartbeat()
    clk.advance(20.0)  # 20s > 15s
    lc = _FakeLifecycle(RAppState.RUNNING, reg)
    resp = _client(lc).get("/healthz")
    assert resp.status_code == 503
    assert "planner_stalled" in resp.json()["failed"]


def test_healthz_planner_silence_pre_first_heartbeat_is_ok() -> None:
    """Boot phase before first planner decision must NOT 503 the pod.

    Otherwise k8s livenessProbe restarts the pod every boot before it
    gets a chance to receive its first telemetry.
    """
    clk = _FakeClock()
    reg = LivenessRegistry(
        time_source=clk, planner_sla_s=1.0, planner_max_missed=3
    )
    # Never call mark_planner_heartbeat — we are still booting.
    reg.mark_watchdog_tick()
    clk.advance(60.0)  # An hour with no planner heartbeat
    reg.mark_watchdog_tick()  # Watchdog kept ticking
    lc = _FakeLifecycle(RAppState.RUNNING, reg)
    resp = _client(lc).get("/healthz")
    assert resp.status_code == 200, resp.json()


def test_healthz_recovery_after_stale_watchdog() -> None:
    """A wedged loop that recovers (next watchdog tick) → /healthz back to 200."""
    clk = _FakeClock()
    reg = LivenessRegistry(time_source=clk, watchdog_max_age_s=30.0)
    reg.mark_watchdog_tick()
    clk.advance(60.0)  # stale
    lc = _FakeLifecycle(RAppState.RUNNING, reg)
    assert _client(lc).get("/healthz").status_code == 503
    reg.mark_watchdog_tick()  # recovery tick
    assert _client(lc).get("/healthz").status_code == 200


# ---------- /readyz ----------

def test_readyz_503_when_lifecycle_not_running() -> None:
    """RAppState.INIT must produce 503 even with all liveness green."""
    reg = LivenessRegistry()
    lc = _FakeLifecycle(RAppState.INIT, reg)
    resp = _client(lc).get("/readyz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"


def test_readyz_503_when_connector_not_running() -> None:
    """Connector state != RUNNING → 503 connector_not_running."""
    reg = LivenessRegistry()
    reg.set_connector_state_provider(lambda: "STARTING")
    reg.set_evidence_writable_probe(lambda: True)
    reg.set_r1_registered_provider(lambda: True)
    lc = _FakeLifecycle(RAppState.RUNNING, reg)
    resp = _client(lc).get("/readyz")
    assert resp.status_code == 503
    assert "connector_not_running" in resp.json()["failed"]


def test_readyz_503_when_evidence_store_unwritable() -> None:
    """Evidence-store probe returns False → 503 evidence_not_writable."""
    reg = LivenessRegistry()
    reg.set_connector_state_provider(lambda: "RUNNING")
    reg.set_evidence_writable_probe(lambda: False)
    reg.set_r1_registered_provider(lambda: True)
    lc = _FakeLifecycle(RAppState.RUNNING, reg)
    resp = _client(lc).get("/readyz")
    assert resp.status_code == 503
    assert "evidence_not_writable" in resp.json()["failed"]


def test_readyz_503_when_r1_not_registered() -> None:
    """R1 unregistered (e.g. SMO outage at boot) → 503 r1_not_registered."""
    reg = LivenessRegistry()
    reg.set_connector_state_provider(lambda: "RUNNING")
    reg.set_evidence_writable_probe(lambda: True)
    reg.set_r1_registered_provider(lambda: False)
    lc = _FakeLifecycle(RAppState.RUNNING, reg)
    resp = _client(lc).get("/readyz")
    assert resp.status_code == 503
    assert "r1_not_registered" in resp.json()["failed"]


def test_readyz_200_when_all_green() -> None:
    """Lifecycle RUNNING + connector RUNNING + writable + R1 registered → 200."""
    reg = LivenessRegistry()
    reg.set_connector_state_provider(lambda: "RUNNING")
    reg.set_evidence_writable_probe(lambda: True)
    reg.set_r1_registered_provider(lambda: True)
    lc = _FakeLifecycle(RAppState.RUNNING, reg)
    resp = _client(lc).get("/readyz")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"]["connector_state"] == "RUNNING"
    assert body["checks"]["evidence_writable"] is True
    assert body["checks"]["r1_registered"] is True


def test_readyz_evidence_probe_exception_treated_as_unwritable() -> None:
    """If the evidence probe raises, treat as not-writable (fail-closed)."""
    reg = LivenessRegistry()
    reg.set_connector_state_provider(lambda: "RUNNING")

    def explode() -> bool:
        raise RuntimeError("disk-full")

    reg.set_evidence_writable_probe(explode)
    reg.set_r1_registered_provider(lambda: True)
    lc = _FakeLifecycle(RAppState.RUNNING, reg)
    resp = _client(lc).get("/readyz")
    assert resp.status_code == 503
    assert "evidence_not_writable" in resp.json()["failed"]

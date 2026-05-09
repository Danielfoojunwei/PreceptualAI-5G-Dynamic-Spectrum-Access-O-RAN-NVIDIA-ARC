"""Lifecycle health + Prometheus endpoint tests."""

from fastapi.testclient import TestClient

from horizon_ric.rapp.health import build_health_app
from horizon_ric.rapp.lifecycle import HorizonRAppLifecycle, RAppState


class _FakeLifecycle:
    """Stand-in for HorizonRAppLifecycle that exposes only `.state`."""

    def __init__(self, state: RAppState):
        self.state = state


class TestHealthEndpoints:
    def test_healthz_always_ok(self):
        # /healthz no longer hardcodes 200 — it consults LivenessRegistry.
        # A fresh _FakeLifecycle has the default registry whose initial
        # timestamps are now() so all signals are healthy at t=0.
        app = build_health_app(_FakeLifecycle(RAppState.INIT))
        client = TestClient(app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["failed"] == []

    def test_readyz_503_when_not_running(self):
        app = build_health_app(_FakeLifecycle(RAppState.INIT))
        client = TestClient(app)
        resp = client.get("/readyz")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"
        assert resp.json()["rapp_state"] == "init"

    def test_readyz_200_when_running(self):
        app = build_health_app(_FakeLifecycle(RAppState.RUNNING))
        client = TestClient(app)
        resp = client.get("/readyz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["rapp_state"] == "running"

    def test_metrics_returns_prometheus_format(self):
        app = build_health_app(_FakeLifecycle(RAppState.RUNNING))
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        # Prometheus exposition format markers
        assert "# HELP" in body
        assert "# TYPE" in body
        # Our state gauge is updated on each scrape
        assert "horizon_rapp_state" in body


class TestLifecycleAccessors:
    def test_a1_and_r1_exposed(self):
        lc = HorizonRAppLifecycle()
        assert lc.a1 is not None
        assert lc.r1 is not None
        # State accessor returns current state
        assert lc.state == RAppState.INIT

    def test_mark_degraded_transitions(self):
        lc = HorizonRAppLifecycle()
        assert lc.state == RAppState.INIT
        lc.mark_degraded("synthetic test")
        assert lc.state == RAppState.DEGRADED

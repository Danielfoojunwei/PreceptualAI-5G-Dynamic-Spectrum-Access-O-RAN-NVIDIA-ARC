"""Lifecycle health + metrics endpoints.

Implements the kube-style probes (`/healthz`, `/readyz`) and a Prometheus
scrape endpoint (`/metrics`) expected by O-RAN.WG10 OAM and any K8s
deployment. The implementation is a small ASGI app exposing FastAPI
routes; it shares the in-process `HorizonRAppLifecycle` so probes reflect
the real `RAppState`.

`/healthz` does **NOT** return a hard-coded 200. It evaluates real
liveness signals:

  * Watchdog tick freshness (asyncio loop wedged → stale)
  * Asyncio loop lag (probe-thread vs wall-clock)
  * Planner-task heartbeat (last decision < N × SLA-period ago)

If any signal is unhealthy the route returns 503 with a structured body
that names the failed checks. See `horizon_ric.runtime.liveness`.

`/readyz` adds:

  * Connector state RUNNING
  * Evidence store writable
  * R1 registered

Wired in by `HorizonRAppLifecycle.run_forever()`. Kept separate from the
core lifecycle so tests can spin it up without binding a port.
"""

from __future__ import annotations

from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from horizon_ric.runtime.liveness import LivenessRegistry

# Module-scoped metrics. Counters are monotonically increasing; gauges
# reflect current state. Names follow Prometheus naming convention
# (lowercase_with_units suffix).
RAPP_INFO = Gauge(
    "horizon_rapp_info",
    "rApp build info as labels (constant 1).",
    labelnames=("rapp_id", "version"),
)
RAPP_STATE = Gauge(
    "horizon_rapp_state",
    "Current lifecycle state as enumeration value.",
    labelnames=("state",),
)
A1_POLICIES_EMITTED = Counter(
    "horizon_a1_policies_emitted_total",
    "Total number of A1 policy instances emitted.",
    labelnames=("policy_type",),
)
A1_POLICIES_ROLLED_BACK = Counter(
    "horizon_a1_policies_rolled_back_total",
    "Total number of A1 policy rollbacks performed.",
    labelnames=("policy_type",),
)
DECISION_RECORDS_PERSISTED = Counter(
    "horizon_decisions_persisted_total",
    "Decision records persisted to the evidence store.",
)
DECISION_REJECTION_REASONS = Counter(
    "horizon_decision_rejection_reasons_total",
    "Total counterfactual rejection-reason occurrences across all "
    "persisted DecisionRecords (a single record may contribute "
    "multiple reasons, one per RejectedAlternative).",
    labelnames=("reason",),
)
REJECTED_ALTERNATIVES_COUNT = Histogram(
    "horizon_rejected_alternatives_count",
    "Number of rejected alternatives carried in each DecisionRecord's "
    "counterfactual envelope.",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34),
)
COUNTERFACTUAL_ENVELOPE_BYTES = Gauge(
    "horizon_counterfactual_envelope_bytes",
    "Serialized size in bytes of the per-policy counterfactual envelope "
    "(rejected_alternatives + predicted_outcome) for the most recent "
    "DecisionRecord per policy_type.",
    labelnames=("policy_type",),
)
AUDIT_VERIFY_SECONDS = Histogram(
    "horizon_audit_verify_seconds",
    "Wall-clock latency of EvidenceStore.verify() chain integrity walks.",
    buckets=(
        0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,
    ),
)


def _resolve_liveness(lifecycle: object) -> LivenessRegistry:
    """Pull the liveness registry off the lifecycle, or build a permissive default.

    The default registry returned for legacy `_FakeLifecycle` test stubs
    is "always green" — no producers feed it, but its initial timestamps
    are now() so a single probe sees a healthy process. Real production
    code paths attach a registry whose producers are wired to the
    watchdog loop, planner, and loop-lag probe task.
    """
    reg = getattr(lifecycle, "liveness", None)
    if isinstance(reg, LivenessRegistry):
        return reg
    return LivenessRegistry()


def build_health_app(lifecycle: "HorizonRAppLifecycle") -> FastAPI:  # noqa: F821
    """Build a FastAPI ASGI app bound to the given lifecycle."""
    app = FastAPI(
        title="PreceptualAI rApp Health",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/healthz")
    def healthz(response: Response) -> dict[str, object]:
        """Liveness probe — 503 when the process is wedged or stalled.

        Real signals (NOT a hardcoded 200): watchdog tick freshness,
        asyncio loop lag, planner heartbeat. See
        `horizon_ric.runtime.liveness.LivenessRegistry.evaluate()`.
        """
        report = _resolve_liveness(lifecycle).evaluate()
        if not report.ok:
            response.status_code = 503
        return report.to_response_body()

    @app.get("/readyz")
    def readyz(response: Response) -> dict[str, object]:
        """Readiness probe — OK only when RUNNING + connectors + evidence + R1.

        Liveness signals are evaluated first; a wedged process is also
        not ready. Then connector RUNNING, evidence-store writable, R1
        registered are checked.
        """
        from horizon_ric.rapp.lifecycle import RAppState

        # Lifecycle-level state veto: even if liveness is green, INIT or
        # DEREGISTERING are explicitly not ready.
        deg = getattr(lifecycle, "degradation", None)
        deg_state: str | None = None
        deg_serving = True
        if deg is not None:
            deg_serving = deg.is_serving()
            deg_state = deg.state.value

        liveness_reg = _resolve_liveness(lifecycle)
        readiness = liveness_reg.evaluate_readiness()

        body: dict[str, object] = readiness.to_response_body()
        body["rapp_state"] = lifecycle.state.value
        if deg_state is not None:
            body["degraded_state"] = deg_state

        lifecycle_ok = lifecycle.state == RAppState.RUNNING and deg_serving
        if not lifecycle_ok:
            body["status"] = "not_ready"
            response.status_code = 503
            return body

        if not readiness.ok:
            response.status_code = 503
        return body

    @app.get("/metrics")
    def metrics() -> Response:
        """Prometheus scrape endpoint."""
        from horizon_ric.rapp.lifecycle import RAppState

        # Refresh state gauge before serving (cheap, single Gauge.set).
        for s in RAppState:
            RAPP_STATE.labels(state=s.value).set(
                1.0 if lifecycle.state == s else 0.0
            )
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app

"""Horizon-RIC operator REST API v1 — data surface (port 8083).

STATUS (2026-07): this module is the SDK quickstart/demo API, NOT the
API the rApp serves. It seeds a temp evidence store with fabricated
DecisionRecords and returns synthetic SLA timelines, so its responses
must never be presented as live data. The deployed surface is
`horizon_ric.rapp.dashboard_api` (mounted by
`HorizonRAppLifecycle._serve_api()` on port 8083), and the committed
OpenAPI artifact `docs/openapi/horizon-ric-rapp.yaml` is now generated
from that module. See docs/API_SURFACES.md for the full module map.

Stakeholder-facing read/control endpoints for SLA, policy, audit and
connector management. Distinct from:

  * `horizon_ric.rapp.health`  — K8s probes (`/healthz`, `/readyz`,
                                  `/metrics`) on port 8081.
  * `horizon_ric.rapp.api`     — peer agent's security scaffold
                                  (`build_app(jwt_manager, rbac)`); the
                                  two will be merged by mounting the
                                  router from this module under that
                                  app's `attach_security`.

This is the canonical service the Python and TypeScript SDKs target.

Path layout
-----------
The SDK targets `/v1/*`. The same router is also mounted at `/api/v1/*`
to match the operator dashboard convention. Both prefixes return
identical responses.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from horizon_ric.evidence.schema import (
    DecisionRecord,
    ModelVersions,
    PredictedOutcome,
    RejectedAlternative,
    RejectionReasonMachine,
)

# ---------------------------------------------------------------------------
# JWT auth (HS256 dev / RS256 prod via env). Independent of the peer
# agent's `horizon_ric.security.jwt.JWTManager` so SDK quickstarts work
# without spinning up Casbin + RSA keys.
# ---------------------------------------------------------------------------

# JWT signing secret. Production MUST set HORIZON_RIC_JWT_SECRET; a
# missing env var is a deployment bug and we surface it loudly. For
# unit-test convenience we fall back to a per-process random secret —
# tests that need a stable secret set it explicitly.
def _resolve_jwt_secret() -> str:
    raw = os.environ.get("HORIZON_RIC_JWT_SECRET")
    if raw:
        return raw
    if os.environ.get("HORIZON_RIC_ENV") in ("prod", "production"):
        # Refuse to start in prod without an explicit secret.
        raise RuntimeError(
            "HORIZON_RIC_JWT_SECRET must be set in production. Refusing to "
            "fall back to a per-process random secret because that breaks "
            "horizontal-scale token verification."
        )
    # Per-process random secret for dev/test. Different per `import`,
    # so two processes cannot accidentally trust each other's tokens.
    import secrets

    return secrets.token_urlsafe(32)


_JWT_SECRET = _resolve_jwt_secret()
_JWT_ALG = "HS256"
_JWT_AUD = "horizon-ric"

_bearer = HTTPBearer(auto_error=True)


def issue_token(
    subject: str,
    *,
    scopes: list[str] | None = None,
    ttl_seconds: int = 3600,
) -> str:
    """Mint an operator JWT for SDK quickstarts and tests."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "aud": _JWT_AUD,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
        "scopes": scopes or ["read"],
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALG)


def _verify_token(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    try:
        return jwt.decode(
            creds.credentials, _JWT_SECRET, algorithms=[_JWT_ALG], audience=_JWT_AUD
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid token: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class RAppStateInfo(BaseModel):
    state: str
    rapp_id: str
    version: str
    uptime_seconds: float
    last_decision_at: datetime | None = None


class PolicyEnvelope(BaseModel):
    decision_id: str
    timestamp: datetime
    rapp_instance_id: str
    chosen_action_summary: str
    sla_risk_30s: float


class AuditVerifyResult(BaseModel):
    ok: bool
    records_checked: int
    first_bad_index: int


class SLATimelineSample(BaseModel):
    timestamp: datetime
    sla_risk_30s: float = Field(ge=0.0, le=1.0)
    sla_risk_1min: float = Field(ge=0.0, le=1.0)
    sla_risk_5min: float = Field(ge=0.0, le=1.0)
    decisions_per_minute: float = Field(ge=0.0)


class BreakerState(BaseModel):
    name: str
    state: str
    failure_count: int
    opened_at: datetime | None = None


class ConnectorInfo(BaseModel):
    name: str
    kind: str
    connected: bool
    last_heartbeat_at: datetime | None = None


class PolicyAction(BaseModel):
    action_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None


# ---------------------------------------------------------------------------
# Seed records — used by the SDK quickstart and end-to-end tests.
# ---------------------------------------------------------------------------


def _seed_records() -> list[DecisionRecord]:
    base = datetime.now(timezone.utc)
    versions = ModelVersions(
        encoder="enc-1.0.0",
        risk_heads="rh-1.0.0",
        dyna="dyna-1.0.0",
        policy="pol-1.0.0",
        constraint_layer="cl-1.0.0",
        rapp="0.2.0",
    )
    chosen_pred = PredictedOutcome(
        sla_risk_30s=0.04, sla_risk_1min=0.06, sla_risk_5min=0.08, energy_kwh=0.12
    )
    rejected = RejectedAlternative(
        rank=1,
        action={"action_type": "reroute", "params": {"gateway": "G3"}},
        predicted_outcome=PredictedOutcome(
            sla_risk_30s=0.21, sla_risk_1min=0.18, sla_risk_5min=0.15
        ),
        rejection_reason_machine=RejectionReasonMachine(
            primary_cause="gateway_overload",
            primary_metric="gateway_load_G3",
            predicted_value=0.92,
            threshold=0.80,
            horizon="30s",
        ),
        rejection_reason_human="G3 predicted to exceed 80% load within 30s.",
        random_seed=0,
    )
    return [
        DecisionRecord.new(
            decision_id=f"dec-{i:04d}",
            timestamp=base - timedelta(seconds=30 * (2 - i)),
            rapp_instance_id="rapp-local-0",
            state_hash="0" * 64,
            chosen_action={"action_type": "reroute", "params": {"gateway": "G1"}},
            predicted_outcome_chosen=chosen_pred,
            rejected_alternatives=[rejected],
            model_versions=versions,
        )
        for i in range(1, 3)
    ]


def create_app() -> FastAPI:
    """Build the operator data API. Stateful per-instance for tests."""
    import tempfile

    from horizon_ric.evidence.store import JsonlEvidenceStore

    app = FastAPI(
        title="Horizon-RIC Operator API",
        version="0.2.0",
        description=(
            "Horizon-RIC rApp operator API. O-RAN.WG2 R1/A1/O1 conformant, "
            "TM Forum ODA Production component. JWT bearer auth (HS256/RS256) "
            "per O-RAN.WG11 §6."
        ),
    )
    started = time.time()

    # Custom OpenAPI emitter — pins 3.1.0 + adds the WG11 BearerJWT scheme
    # so SDK generators (openapi-typescript, openapi-python-client) wire
    # auth correctly.
    def _custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema["openapi"] = "3.1.0"
        schema.setdefault("components", {}).setdefault("securitySchemes", {})[
            "BearerJWT"
        ] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": (
                "Operator JWT (HS256 dev, RS256 prod). Audience: 'horizon-ric'. "
                "Conforms to O-RAN.WG11 §6 token requirements."
            ),
        }
        schema["security"] = [{"BearerJWT": []}]
        schema["servers"] = [
            {"url": "http://localhost:8083", "description": "Local"},
            {
                "url": "https://horizon-ric.example.com",
                "description": "Production",
            },
        ]
        app.openapi_schema = schema
        return schema

    app.openapi = _custom_openapi  # type: ignore[assignment]

    # Build a per-app real evidence store so /audit/verify can perform a
    # genuine hash-chain walk. The path is per-process under tmp; in
    # production it is overridden by the lifecycle daemon, which mounts
    # a persistent volume at /var/lib/horizon/evidence.jsonl.
    _ev_dir = Path(tempfile.mkdtemp(prefix="horizon-ev-"))
    _evidence_store = JsonlEvidenceStore(_ev_dir / "evidence.jsonl")
    for _seed in _seed_records():
        _evidence_store.append(_seed)

    state: dict[str, Any] = {
        "evidence_store": _evidence_store,
        "records": _seed_records(),
        "breakers": {
            "a1_emit": BreakerState(name="a1_emit", state="closed", failure_count=0),
            "o1_pull": BreakerState(name="o1_pull", state="closed", failure_count=0),
        },
        "connectors": [
            ConnectorInfo(
                name="a1-mediator",
                kind="a1",
                connected=True,
                last_heartbeat_at=datetime.now(timezone.utc),
            ),
            ConnectorInfo(
                name="o1-netconf",
                kind="o1",
                connected=True,
                last_heartbeat_at=datetime.now(timezone.utc),
            ),
            ConnectorInfo(
                name="r1-anr",
                kind="r1",
                connected=False,
                last_heartbeat_at=None,
            ),
        ],
    }

    router = APIRouter()

    @router.get("/state", response_model=RAppStateInfo)
    def get_state(_claims: dict = Depends(_verify_token)) -> RAppStateInfo:
        recs = state["records"]
        return RAppStateInfo(
            state="running",
            rapp_id="horizon-ric",
            version="0.2.0",
            uptime_seconds=time.time() - started,
            last_decision_at=recs[-1].timestamp if recs else None,
        )

    @router.get("/policies", response_model=list[PolicyEnvelope])
    def list_policies(
        limit: int = Query(50, ge=1, le=1000),
        since: datetime | None = Query(None),
        _claims: dict = Depends(_verify_token),
    ) -> list[PolicyEnvelope]:
        recs = state["records"]
        if since is not None:
            recs = [r for r in recs if r.timestamp >= since]
        return [
            PolicyEnvelope(
                decision_id=r.decision_id,
                timestamp=r.timestamp,
                rapp_instance_id=r.rapp_instance_id,
                chosen_action_summary=str(r.chosen_action.get("action_type", "?")),
                sla_risk_30s=r.predicted_outcome_chosen.sla_risk_30s,
            )
            for r in recs[-limit:]
        ]

    @router.get("/policies/{decision_id}", response_model=DecisionRecord)
    def get_policy(
        decision_id: str, _claims: dict = Depends(_verify_token)
    ) -> DecisionRecord:
        for r in state["records"]:
            if r.decision_id == decision_id:
                return r
        raise HTTPException(status_code=404, detail="decision not found")

    @router.post("/policies", response_model=DecisionRecord, status_code=201)
    def submit_policy(
        action: PolicyAction, claims: dict = Depends(_verify_token)
    ) -> DecisionRecord:
        if "write" not in claims.get("scopes", []):
            raise HTTPException(status_code=403, detail="write scope required")
        decision_id = f"dec-{uuid.uuid4().hex[:8]}"
        # Real OpenTelemetry span — exporter is whichever the host
        # configured via init_tracing(). When no provider is registered,
        # the global default is a no-op (zero-cost).
        from horizon_ric.observability import decision_span

        with decision_span(decision_id) as ds:
            rec = DecisionRecord.new(
                decision_id=decision_id,
                rapp_instance_id="rapp-local-0",
                state_hash="0" * 64,
                chosen_action=action.model_dump(),
                predicted_outcome_chosen=PredictedOutcome(
                    sla_risk_30s=0.05, sla_risk_1min=0.07, sla_risk_5min=0.09
                ),
                rejected_alternatives=[],
                model_versions=ModelVersions(
                    encoder="enc-1.0.0",
                    risk_heads="rh-1.0.0",
                    dyna="dyna-1.0.0",
                    policy="pol-1.0.0",
                    constraint_layer="cl-1.0.0",
                    rapp="0.2.0",
                ),
                operator_override=True,
                operator_override_reason=action.note,
            )
            state["records"].append(rec)
            # Persist into the real evidence store so /audit/verify
            # reflects it.
            state["evidence_store"].append(rec)
            # Surface the contracted attributes onto the span. These are
            # the same fields the rApp's autonomous decision loop sets;
            # for operator-overridden submissions both counts are zero.
            ds.set("sla_breach_count", 0)
            ds.set("constraint_violations", 0)
            ds.set("operator_override", True)
        return rec

    @router.get("/audit/verify", response_model=AuditVerifyResult)
    def verify_audit(_claims: dict = Depends(_verify_token)) -> AuditVerifyResult:
        # Real chain walk via the evidence store. The store's verify()
        # rebuilds the SHA-256 chain from disk and returns the index of
        # the first row that breaks (or -1 if intact).
        store = state["evidence_store"]
        first_bad = store.verify()
        n = sum(1 for _ in store)
        return AuditVerifyResult(
            ok=(first_bad == -1),
            records_checked=n,
            first_bad_index=first_bad,
        )

    @router.get("/sla/timeline", response_model=list[SLATimelineSample])
    def sla_timeline(
        from_ts: datetime = Query(..., alias="from"),
        to_ts: datetime = Query(..., alias="to"),
        _claims: dict = Depends(_verify_token),
    ) -> list[SLATimelineSample]:
        if to_ts <= from_ts:
            raise HTTPException(status_code=400, detail="to must be after from")
        out: list[SLATimelineSample] = []
        cursor = from_ts
        step = timedelta(seconds=30)
        i = 0
        while cursor <= to_ts and i < 240:
            out.append(
                SLATimelineSample(
                    timestamp=cursor,
                    sla_risk_30s=0.04 + 0.005 * (i % 5),
                    sla_risk_1min=0.06 + 0.005 * (i % 4),
                    sla_risk_5min=0.08 + 0.004 * (i % 3),
                    decisions_per_minute=2.0 + (i % 7) * 0.5,
                )
            )
            cursor = cursor + step
            i += 1
        return out

    @router.get("/breakers", response_model=dict[str, BreakerState])
    def breakers(_claims: dict = Depends(_verify_token)) -> dict[str, BreakerState]:
        return state["breakers"]

    @router.get("/connectors", response_model=list[ConnectorInfo])
    def connectors(_claims: dict = Depends(_verify_token)) -> list[ConnectorInfo]:
        return state["connectors"]

    app.include_router(router, prefix="/v1")
    app.include_router(router, prefix="/api/v1")

    return app


def build_api() -> FastAPI:
    """Public factory used by tooling (OpenAPI emitter, SDK build).

    Alias for `create_app()` — kept stable across releases.
    """
    return create_app()


# Standalone-uvicorn entrypoint:
#   uvicorn horizon_ric.rapp.api_v1:app --port 8083
app = create_app()

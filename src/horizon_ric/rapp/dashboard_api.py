"""PreceptualAI operator dashboard REST API (port 8083).

Stakeholder-facing read/control surface for SLA, policy, audit, and
connector management. Distinct from:

  * `health.py`  — kube probes + Prometheus metrics on port 8081.
  * `api.py`     — RBAC-protected `/policies`, `/audit/*`, `/state` for
                   the production multi-tenant operator API
                   (RS256 + Casbin, owned by the security module).

This module is the *dashboard* API consumed by the Next.js frontend in
`frontend/`. It exposes a minimal, OpenAPI-typed surface at
`/api/v1/*` and authenticates with HS256 JWTs (``python-jose``).

Endpoints:

  POST /api/v1/auth/token        — exchange username+password for a JWT
  GET  /api/v1/state             — lifecycle state, degraded state, watchdog ts
  GET  /api/v1/policies          — recent A1 emissions from the evidence store
  GET  /api/v1/policies/{id}     — full DecisionRecord with counterfactuals
  POST /api/v1/audit/verify      — runs ``EvidenceStore.verify()``
  GET  /api/v1/sla/timeline      — per-horizon SLA risk samples over a window
  GET  /api/v1/circuit-breakers  — state of horizon.r1, .a1, .o1 breakers
  GET  /api/v1/connectors        — registered Source/Sink connectors

Mounted into ``HorizonRAppLifecycle.run_forever()`` as a separate
``_serve_api()`` task on port 8083.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from horizon_ric.evidence.schema import DecisionRecord
from horizon_ric.evidence.store import EvidenceStore, JsonlEvidenceStore
from horizon_ric.io.registry import list_connectors as _list_connectors

logger = structlog.get_logger(__name__)


# ── JWT settings ────────────────────────────────────────────────────────────

DEFAULT_JWT_ALGORITHM = "HS256"
DEFAULT_JWT_ISSUER = "horizon-ric"
DEFAULT_JWT_AUDIENCE = "horizon-dashboard"
DEFAULT_TOKEN_TTL_S = 3600


def _jwt_secret() -> str:
    """Read the HS256 secret. Refuses an empty string."""
    secret = os.environ.get("HORIZON_API_JWT_SECRET")
    if not secret:
        raise RuntimeError(
            "HORIZON_API_JWT_SECRET is not set. The dashboard API refuses to "
            "start without a JWT signing secret. Generate one with "
            "`openssl rand -hex 32` and export it before launching the rApp."
        )
    return secret


def issue_token(
    sub: str,
    role: str = "viewer",
    ttl_seconds: int = DEFAULT_TOKEN_TTL_S,
    secret: str | None = None,
) -> str:
    """Mint a JWT for a given subject + role. Used by /auth/token + tests."""
    now = int(time.time())
    payload = {
        "iss": DEFAULT_JWT_ISSUER,
        "aud": DEFAULT_JWT_AUDIENCE,
        "sub": sub,
        "role": role,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    return jwt.encode(payload, secret or _jwt_secret(), algorithm=DEFAULT_JWT_ALGORITHM)


_bearer = HTTPBearer(auto_error=False)


def _decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(
            token,
            _jwt_secret(),
            algorithms=[DEFAULT_JWT_ALGORITHM],
            audience=DEFAULT_JWT_AUDIENCE,
            issuer=DEFAULT_JWT_ISSUER,
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid token: {exc}",
        ) from exc


def require_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    """FastAPI dependency: enforces a valid bearer token on every request."""
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _decode_token(creds.credentials)


# ── Pydantic response models ────────────────────────────────────────────────


class StateResponse(BaseModel):
    rapp_state: str
    degraded_state: str
    watchdog_last_tick: float
    watchdog_interval_s: float
    a1_policies_emitted: int
    last_decision_id: str | None
    evidence_chain_head: str | None
    rapp_id: str
    rapp_version: str
    timestamp: float
    uptime_seconds: float


class PolicySummary(BaseModel):
    decision_id: str
    timestamp: datetime
    chosen_action: dict[str, Any]
    sla_risk_30s: float
    sla_risk_1min: float
    sla_risk_5min: float
    rejected_count: int
    chain_hash: str


class PolicyListResponse(BaseModel):
    items: list[PolicySummary]
    total: int


class PolicyDetailResponse(BaseModel):
    record: DecisionRecord
    chain_hash: str


class AuditVerifyResponse(BaseModel):
    intact: bool
    broken_at_index: int  # -1 when intact
    chain_length: int
    head_hash: str | None


class SlaSample(BaseModel):
    timestamp: datetime
    horizon: str
    sla_risk: float = Field(ge=0.0, le=1.0)
    decision_id: str


class SlaTimelineResponse(BaseModel):
    items: list[SlaSample]
    horizons: list[str]
    from_ts: datetime
    to_ts: datetime


class CircuitBreakerState(BaseModel):
    name: str
    state: str
    fail_counter: int


class CircuitBreakerListResponse(BaseModel):
    breakers: list[CircuitBreakerState]


class ConnectorListResponse(BaseModel):
    sources: list[str]
    sinks: list[str]


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = DEFAULT_TOKEN_TTL_S
    role: str


# ── helpers ─────────────────────────────────────────────────────────────────


def _users_from_env() -> dict[str, tuple[str, str]]:
    """Parse ``HORIZON_API_USERS`` (``user:pw:role,user2:pw:role``)."""
    raw = os.environ.get("HORIZON_API_USERS", "")
    out: dict[str, tuple[str, str]] = {}
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) != 3:
            continue
        user, pw, role = parts
        if user and pw and role:
            out[user] = (pw, role)
    return out


def _read_chain(store: EvidenceStore) -> list[tuple[DecisionRecord, str]]:
    """Materialise the chain into memory. Audit chains stay small (~10K)."""
    return list(store)


def _summarise(record: DecisionRecord, chain_hash: str) -> PolicySummary:
    return PolicySummary(
        decision_id=record.decision_id,
        timestamp=record.timestamp,
        chosen_action=record.chosen_action,
        sla_risk_30s=record.predicted_outcome_chosen.sla_risk_30s,
        sla_risk_1min=record.predicted_outcome_chosen.sla_risk_1min,
        sla_risk_5min=record.predicted_outcome_chosen.sla_risk_5min,
        rejected_count=len(record.rejected_alternatives),
        chain_hash=chain_hash,
    )


# ── App factory ─────────────────────────────────────────────────────────────


def build_dashboard_api(
    lifecycle: Any | None = None,
    *,
    evidence_store: EvidenceStore | None = None,
    breakers: dict[str, Any] | None = None,
) -> FastAPI:
    """Build the operator-facing FastAPI app.

    ``lifecycle`` is the live ``HorizonRAppLifecycle`` instance (may be None
    when the API is started in standalone read-only mode).
    ``evidence_store`` defaults to a JSONL store at
    ``$HORIZON_EVIDENCE_PATH`` (or ``./var/evidence/horizon.jsonl``).
    ``breakers`` is the dict ``{"horizon.r1": AsyncCircuitBreaker, ...}``.
    """

    if evidence_store is None:
        path = Path(
            os.environ.get(
                "HORIZON_EVIDENCE_PATH", "./var/evidence/horizon.jsonl"
            )
        )
        evidence_store = JsonlEvidenceStore(path)

    app = FastAPI(
        title="PreceptualAI Operator Dashboard API",
        version="0.2.0",
        description=(
            "Operator-facing REST surface for the PreceptualAI dashboard. "
            "Every endpoint requires a JWT bearer token (HS256). "
            "See README at frontend/ for usage."
        ),
    )

    cors_env = os.environ.get(
        "HORIZON_API_CORS_ORIGINS", "http://localhost:3000"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_env.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.state.lifecycle = lifecycle
    app.state.evidence_store = evidence_store
    app.state.breakers = breakers
    app.state.boot_ts = time.time()
    app.state.last_watchdog_tick = time.time()

    # ── auth/token ──────────────────────────────────────────────────────
    @app.post(
        "/api/v1/auth/token",
        response_model=TokenResponse,
        tags=["auth"],
    )
    def auth_token(req: TokenRequest) -> TokenResponse:
        users = _users_from_env()
        if req.username not in users:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="unknown user",
            )
        pw, role = users[req.username]
        if pw != req.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="bad credentials",
            )
        token = issue_token(sub=req.username, role=role)
        return TokenResponse(access_token=token, role=role)

    # ── 1. /state ───────────────────────────────────────────────────────
    @app.get(
        "/api/v1/state",
        response_model=StateResponse,
        tags=["lifecycle"],
    )
    def state_endpoint(
        claims: dict[str, Any] = Depends(require_auth),
    ) -> StateResponse:
        lc = app.state.lifecycle
        rapp_state = "stopped"
        deg_state = "normal"
        emitted = 0
        last_decision_id: str | None = None
        rapp_id = "horizon-ric"
        rapp_version = os.environ.get("HORIZON_VERSION", "0.2.0")
        watchdog_interval = 15.0

        if lc is not None:
            cur = getattr(lc, "state", "stopped")
            rapp_state = str(getattr(cur, "value", cur))
            deg = getattr(lc, "degradation", None)
            if deg is not None:
                deg_state = deg.state.value
            a1 = getattr(lc, "a1", None)
            if a1 is not None and hasattr(a1, "policies_emitted_count"):
                try:
                    emitted = int(a1.policies_emitted_count())
                except Exception:
                    emitted = 0
            last_decision_id = getattr(lc, "_last_decision_id", None)
            r1 = getattr(lc, "r1", None)
            if r1 is not None:
                cfg = getattr(r1, "cfg", None)
                if cfg is not None:
                    rapp_id = getattr(cfg, "rapp_id", rapp_id)
                    rapp_version = getattr(cfg, "rapp_version", rapp_version)
            watchdog_interval = float(
                getattr(lc, "_watchdog_interval_s", 15.0)
            )

        head: str | None = None
        try:
            chain = _read_chain(app.state.evidence_store)
            if chain:
                head = chain[-1][1]
                if last_decision_id is None:
                    last_decision_id = chain[-1][0].decision_id
        except Exception:  # pragma: no cover — best-effort
            head = None

        return StateResponse(
            rapp_state=rapp_state,
            degraded_state=deg_state,
            watchdog_last_tick=app.state.last_watchdog_tick,
            watchdog_interval_s=watchdog_interval,
            a1_policies_emitted=emitted,
            last_decision_id=last_decision_id,
            evidence_chain_head=head,
            rapp_id=rapp_id,
            rapp_version=rapp_version,
            timestamp=time.time(),
            uptime_seconds=time.time() - app.state.boot_ts,
        )

    # ── 2. /policies ────────────────────────────────────────────────────
    @app.get(
        "/api/v1/policies",
        response_model=PolicyListResponse,
        tags=["policies"],
    )
    def policies(
        limit: int = Query(50, ge=1, le=500),
        claims: dict[str, Any] = Depends(require_auth),
    ) -> PolicyListResponse:
        chain = _read_chain(app.state.evidence_store)
        items = [_summarise(r, h) for r, h in chain[-limit:][::-1]]
        return PolicyListResponse(items=items, total=len(chain))

    # ── 3. /policies/{decision_id} ──────────────────────────────────────
    @app.get(
        "/api/v1/policies/{decision_id}",
        response_model=PolicyDetailResponse,
        tags=["policies"],
    )
    def policy_detail(
        decision_id: str,
        claims: dict[str, Any] = Depends(require_auth),
    ) -> PolicyDetailResponse:
        for rec, chain_hash in app.state.evidence_store:
            if rec.decision_id == decision_id:
                return PolicyDetailResponse(record=rec, chain_hash=chain_hash)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"decision {decision_id} not found",
        )

    # ── 4. /audit/verify ────────────────────────────────────────────────
    @app.post(
        "/api/v1/audit/verify",
        response_model=AuditVerifyResponse,
        tags=["audit"],
    )
    def audit_verify(
        claims: dict[str, Any] = Depends(require_auth),
    ) -> AuditVerifyResponse:
        store: EvidenceStore = app.state.evidence_store
        # Time the verify() walk for the `horizon_audit_verify_seconds`
        # histogram surfaced in the counterfactual Grafana dashboard.
        import time as _time
        _t0 = _time.perf_counter()
        broken = store.verify()
        try:
            from horizon_ric.rapp.health import AUDIT_VERIFY_SECONDS

            AUDIT_VERIFY_SECONDS.observe(_time.perf_counter() - _t0)
        except Exception:  # pragma: no cover
            pass
        chain = _read_chain(store)
        head = chain[-1][1] if chain else None
        return AuditVerifyResponse(
            intact=(broken == -1),
            broken_at_index=broken,
            chain_length=len(chain),
            head_hash=head,
        )

    # ── 5. /sla/timeline ────────────────────────────────────────────────
    @app.get(
        "/api/v1/sla/timeline",
        response_model=SlaTimelineResponse,
        tags=["sla"],
    )
    def sla_timeline(
        from_ts: datetime | None = Query(None, alias="from"),
        to_ts: datetime | None = Query(None, alias="to"),
        claims: dict[str, Any] = Depends(require_auth),
    ) -> SlaTimelineResponse:
        if to_ts is None:
            to_ts = datetime.now(timezone.utc)
        if from_ts is None:
            from_ts = to_ts - timedelta(hours=1)
        if from_ts.tzinfo is None:
            from_ts = from_ts.replace(tzinfo=timezone.utc)
        if to_ts.tzinfo is None:
            to_ts = to_ts.replace(tzinfo=timezone.utc)

        chain = _read_chain(app.state.evidence_store)
        items: list[SlaSample] = []
        for rec, _h in chain:
            ts = rec.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < from_ts or ts > to_ts:
                continue
            po = rec.predicted_outcome_chosen
            for horizon, value in (
                ("30s", po.sla_risk_30s),
                ("60s", po.sla_risk_1min),
                ("300s", po.sla_risk_5min),
            ):
                items.append(SlaSample(
                    timestamp=ts, horizon=horizon,
                    sla_risk=float(value), decision_id=rec.decision_id,
                ))
        return SlaTimelineResponse(
            items=items,
            horizons=["30s", "60s", "300s"],
            from_ts=from_ts,
            to_ts=to_ts,
        )

    # ── 6. /circuit-breakers ────────────────────────────────────────────
    @app.get(
        "/api/v1/circuit-breakers",
        response_model=CircuitBreakerListResponse,
        tags=["lifecycle"],
    )
    def circuit_breakers(
        claims: dict[str, Any] = Depends(require_auth),
    ) -> CircuitBreakerListResponse:
        breakers_map: dict[str, Any] = dict(app.state.breakers or {})
        if not breakers_map and app.state.lifecycle is not None:
            for slot, name in (
                ("r1", "horizon.r1"),
                ("a1", "horizon.a1"),
                ("o1", "horizon.o1"),
            ):
                adapter = getattr(app.state.lifecycle, slot, None)
                if adapter is None:
                    continue
                cb = getattr(adapter, "circuit_breaker", None)
                if callable(cb):
                    try:
                        cb_obj = cb()
                    except Exception:
                        cb_obj = None
                else:
                    cb_obj = cb
                if cb_obj is not None:
                    breakers_map[name] = cb_obj
        out: list[CircuitBreakerState] = []
        for name, cb in breakers_map.items():
            try:
                state_str = str(getattr(cb, "state", "unknown"))
                fail_counter = int(getattr(cb, "fail_counter", 0))
            except Exception:
                state_str = "unknown"
                fail_counter = 0
            out.append(CircuitBreakerState(
                name=name, state=state_str, fail_counter=fail_counter,
            ))
        return CircuitBreakerListResponse(breakers=out)

    # ── 7. /connectors ──────────────────────────────────────────────────
    @app.get(
        "/api/v1/connectors",
        response_model=ConnectorListResponse,
        tags=["lifecycle"],
    )
    def connectors(
        claims: dict[str, Any] = Depends(require_auth),
    ) -> ConnectorListResponse:
        info = _list_connectors()
        return ConnectorListResponse(
            sources=info.get("sources", []),
            sinks=info.get("sinks", []),
        )

    @app.middleware("http")
    async def _record_watchdog(request: Request, call_next):
        # Updates the in-memory watchdog timestamp on every request so
        # the /state endpoint reflects API responsiveness for the UI.
        app.state.last_watchdog_tick = time.time()
        return await call_next(request)

    return app


# Backwards-compatible aliases ----------------------------------------------
build_api_app = build_dashboard_api


def create_app() -> FastAPI:
    """Standalone factory for ``uvicorn horizon_ric.rapp.dashboard_api:create_app``."""
    return build_dashboard_api(lifecycle=None)


__all__ = [
    "build_dashboard_api",
    "build_api_app",
    "create_app",
    "issue_token",
    "require_auth",
    "DEFAULT_JWT_AUDIENCE",
    "DEFAULT_JWT_ISSUER",
    "DEFAULT_TOKEN_TTL_S",
    "StateResponse",
    "PolicyListResponse",
    "PolicyDetailResponse",
    "PolicySummary",
    "AuditVerifyResponse",
    "SlaSample",
    "SlaTimelineResponse",
    "CircuitBreakerState",
    "CircuitBreakerListResponse",
    "ConnectorListResponse",
    "TokenRequest",
    "TokenResponse",
]

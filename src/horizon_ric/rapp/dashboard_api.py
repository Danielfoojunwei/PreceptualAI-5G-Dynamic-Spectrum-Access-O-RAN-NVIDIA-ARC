"""Horizon-RIC operator dashboard REST API (port 8083).

Stakeholder-facing read/control surface for SLA, policy, audit, and
connector management. Distinct from:

  * `health.py`  — kube probes + Prometheus metrics on port 8081.
  * `api.py`     — RBAC-protected `/policies`, `/audit/*`, `/state` for
                   the production multi-tenant operator API
                   (RS256 + Casbin, owned by the security module).

This module is the *dashboard* API for operator dashboard clients. It
exposes a minimal, OpenAPI-typed surface at `/api/v1/*` and
authenticates with HS256 JWTs (``python-jose``). A bundled web UI
consuming this surface is roadmap — this repository ships the API only.

Endpoints (role required in brackets):

  POST /api/v1/auth/token        — exchange username+password for a JWT [public]
  GET  /api/v1/state             — lifecycle/degraded state, watchdog ts [any role]
  GET  /api/v1/policies          — recent A1 emissions from evidence     [any role]
  GET  /api/v1/policies/{id}     — full DecisionRecord + counterfactuals [any role]
  POST /api/v1/audit/verify      — runs ``EvidenceStore.verify()``       [admin|operator]
  GET  /api/v1/sla/timeline      — per-horizon SLA risk samples          [any role]
  GET  /api/v1/circuit-breakers  — horizon.r1/.a1/.o1 breaker state      [any role]
  GET  /api/v1/connectors        — registered Source/Sink connectors     [any role]

Credential store (``HORIZON_API_USERS``): comma-separated entries of
``username:secret:role``. ``secret`` SHOULD be a PBKDF2 hash in the form
``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>`` (generate with
``python -m horizon_ric.rapp.dashboard_api --hash-password``). Plaintext
secrets are accepted only when ``HORIZON_PRODUCTION_MODE`` is explicitly
falsy ("0"/"false"/"no") — the default is production, where plaintext
entries are refused at login.

Mounted into ``HorizonRAppLifecycle.run_forever()`` as a separate
``_serve_api()`` task on port 8083.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
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

# Runner-aligned default: scripts/run_horizon_rapp.py writes the audit chain
# to /var/lib/horizon/audit.jsonl, so the standalone dashboard API reads the
# same file unless HORIZON_EVIDENCE_PATH overrides it.
DEFAULT_EVIDENCE_PATH = "/var/lib/horizon/audit.jsonl"

# Password-hash entry format for HORIZON_API_USERS.
PBKDF2_SCHEME = "pbkdf2_sha256"
DEFAULT_PBKDF2_ITERATIONS = 600_000

# Role → endpoint matrix. Read-only endpoints accept any authenticated role;
# the privileged full-chain verify walk is restricted. "operator" is kept on
# the privileged list for backward compatibility with previously issued
# operator tokens (pre-hardening the endpoint accepted every role).
PRIVILEGED_ROLES: tuple[str, ...] = ("admin", "operator")


def _production_mode() -> bool:
    """``HORIZON_PRODUCTION_MODE`` flag; unset defaults to True (production).

    Same truthiness rules as ``lifecycle._flag``: only an explicit
    "0"/"false"/"no" (case-insensitive) disables production mode.
    """
    raw = os.environ.get("HORIZON_PRODUCTION_MODE")
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in {"0", "false", "no"}


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


def require_role(*allowed: str):
    """FastAPI dependency factory: valid bearer token AND an allowed role.

    Reads the ``role`` claim minted by :func:`issue_token` (JWT shape is
    unchanged — enforcement only). 401 without a valid token, 403 when the
    token's role is not in ``allowed``.
    """

    def _dep(
        claims: dict[str, Any] = Depends(require_auth),
    ) -> dict[str, Any]:
        role = str(claims.get("role") or "")
        if role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"role '{role or 'none'}' is not permitted; "
                    f"requires one of: {', '.join(sorted(allowed))}"
                ),
            )
        return claims

    return _dep


# ── Password hashing (stdlib PBKDF2-HMAC-SHA256) ────────────────────────────


def hash_password(
    password: str,
    *,
    iterations: int = DEFAULT_PBKDF2_ITERATIONS,
    salt: bytes | None = None,
) -> str:
    """Return a ``pbkdf2_sha256$<iter>$<salt_hex>$<hash_hex>`` entry.

    Suitable for the secret field of ``HORIZON_API_USERS``. Uses only the
    standard library (``hashlib.pbkdf2_hmac``); ``$`` separators keep the
    entry safe inside the colon-delimited ``user:secret:role`` format.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{PBKDF2_SCHEME}${iterations}${salt.hex()}${digest.hex()}"


def _verify_hashed(stored: str, password: str) -> bool:
    """Constant-time verify of a candidate password against a PBKDF2 entry."""
    try:
        scheme, iter_s, salt_hex, hash_hex = stored.split("$")
        if scheme != PBKDF2_SCHEME:
            return False
        iterations = int(iter_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        logger.error("dashboard_api.auth.malformed_hash_entry")
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(candidate, expected)


def _verify_secret(stored: str, password: str, *, username: str) -> bool:
    """Check ``password`` against a stored secret (hashed or plaintext).

    Hashed (``pbkdf2_sha256$...``) entries always work. Plaintext entries
    are a dev-only convenience: they trigger a loud warning, and when
    ``HORIZON_PRODUCTION_MODE`` is truthy (the default) they are refused
    outright — production deployments must store PBKDF2 hashes.
    """
    if stored.startswith(PBKDF2_SCHEME + "$"):
        return _verify_hashed(stored, password)
    if _production_mode():
        logger.error(
            "dashboard_api.auth.plaintext_refused",
            username=username,
            reason=(
                "HORIZON_API_USERS holds a plaintext secret but "
                "HORIZON_PRODUCTION_MODE is enabled (the default). Generate "
                "a hash with `python -m horizon_ric.rapp.dashboard_api "
                "--hash-password` or explicitly set "
                "HORIZON_PRODUCTION_MODE=false for dev."
            ),
        )
        return False
    logger.warning(
        "dashboard_api.auth.plaintext_credentials",
        username=username,
        warning="plaintext credentials — dev only",
    )
    return hmac.compare_digest(stored.encode("utf-8"), password.encode("utf-8"))


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
    """Parse ``HORIZON_API_USERS`` into ``{user: (secret, role)}``.

    Entry format: ``user:secret:role`` (or ``user:secret``, which defaults
    the role to ``viewer``). ``secret`` is either a
    ``pbkdf2_sha256$<iter>$<salt_hex>$<hash_hex>`` hash (recommended; the
    ``$`` separators never collide with the ``:`` delimiters) or a
    dev-only plaintext password — see :func:`_verify_secret`.
    """
    raw = os.environ.get("HORIZON_API_USERS", "")
    out: dict[str, tuple[str, str]] = {}
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) == 2:
            parts = [*parts, "viewer"]
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
    ``$HORIZON_EVIDENCE_PATH`` (or ``/var/lib/horizon/audit.jsonl``, the
    same audit chain ``scripts/run_horizon_rapp.py`` writes).
    ``breakers`` is the dict ``{"horizon.r1": AsyncCircuitBreaker, ...}``.
    """

    if evidence_store is None:
        path = Path(
            os.environ.get("HORIZON_EVIDENCE_PATH", DEFAULT_EVIDENCE_PATH)
        )
        evidence_store = JsonlEvidenceStore(path)

    app = FastAPI(
        title="Horizon-RIC Operator Dashboard API",
        version="0.2.0",
        description=(
            "Operator-facing REST surface for the Horizon-RIC dashboard. "
            "Every endpoint (except /api/v1/auth/token) requires a JWT "
            "bearer token (HS256) with a `role` claim. Read endpoints "
            "accept any authenticated role; POST /api/v1/audit/verify "
            "requires role `admin` or `operator` (403 otherwise). "
            "Interactive OpenAPI docs are served at /docs."
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
        stored, role = users[req.username]
        if not _verify_secret(stored, req.password, username=req.username):
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
        claims: dict[str, Any] = Depends(require_role(*PRIVILEGED_ROLES)),
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


def _cli(argv: list[str] | None = None) -> int:
    """``python -m horizon_ric.rapp.dashboard_api --hash-password``.

    Prints a ``pbkdf2_sha256$...`` secret (or, with ``--user``, a complete
    ``user:hash:role`` entry) for ``HORIZON_API_USERS``. With no password
    argument the CLI prompts interactively so the secret never lands in
    shell history or ``ps`` output.
    """
    import argparse
    import getpass

    parser = argparse.ArgumentParser(
        prog="python -m horizon_ric.rapp.dashboard_api",
        description="Generate HORIZON_API_USERS password-hash entries.",
    )
    parser.add_argument(
        "--hash-password",
        nargs="?",
        const="",
        default=None,
        metavar="PASSWORD",
        help=(
            "Emit a pbkdf2_sha256 hash for PASSWORD (omit the value to be "
            "prompted interactively — preferred, keeps it out of ps/history)."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_PBKDF2_ITERATIONS,
        help=f"PBKDF2 iteration count (default {DEFAULT_PBKDF2_ITERATIONS}).",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="If given, print a full 'user:hash:role' entry instead of the bare hash.",
    )
    parser.add_argument(
        "--role",
        default="viewer",
        help="Role for the --user entry (default: viewer).",
    )
    args = parser.parse_args(argv)

    if args.hash_password is None:
        parser.error("--hash-password is required")
    password = args.hash_password or getpass.getpass("Password: ")
    if not password:
        parser.error("empty password")
    entry = hash_password(password, iterations=args.iterations)
    if args.user:
        print(f"{args.user}:{entry}:{args.role}")
    else:
        print(entry)
    return 0


__all__ = [
    "build_dashboard_api",
    "build_api_app",
    "create_app",
    "issue_token",
    "hash_password",
    "require_auth",
    "require_role",
    "DEFAULT_EVIDENCE_PATH",
    "DEFAULT_PBKDF2_ITERATIONS",
    "PRIVILEGED_ROLES",
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


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess test
    raise SystemExit(_cli())

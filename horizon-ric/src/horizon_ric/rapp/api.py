"""Minimal FastAPI scaffolding for the rApp REST surface.

This file exists to host the `JWTAuthMiddleware` while the parallel
frontend agent builds out the real route handlers. The routes here are
intentionally tiny — they're the smallest set the security tests need
to exercise the middleware end-to-end.

When the frontend agent fleshes out the rApp REST API they should
import `build_app` and add their routes to the returned `FastAPI`
instance, or use `attach_security` on an existing app.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, Request

# Re-export the production rApp data API builder. Kept here so external
# tooling (OpenAPI spec emitter, SDK generators) can import the canonical
# `build_api` from a stable path while the security-only scaffold below
# remains for middleware tests.
from horizon_ric.rapp.api_v1 import build_api  # noqa: E402,F401
from horizon_ric.security.jwt import JWTManager
from horizon_ric.security.middleware import (
    JWTAuthMiddleware,
    Principal,
    require_role,
)
from horizon_ric.security.rbac import Casbin


def attach_security(
    app: FastAPI,
    jwt_manager: JWTManager,
    rbac: Casbin,
) -> None:
    """Wire the JWT + RBAC middleware into an existing FastAPI app."""
    app.add_middleware(
        JWTAuthMiddleware, jwt_manager=jwt_manager, rbac=rbac
    )


def build_app(
    jwt_manager: JWTManager,
    rbac: Casbin,
) -> FastAPI:
    """Construct a small FastAPI app with security wired in.

    Routes:
        GET  /healthz             — public liveness
        GET  /policies            — listing (api-user, operator, auditor, admin, regulator)
        GET  /policies/{policy_id}— detail
        POST /policies/{policy_id}— emit (operator, admin)
        GET  /audit/recent        — auditor / regulator / admin
        GET  /state               — api-user and above
    """
    app = FastAPI(title="PreceptualAI rApp", version="0.1.0")
    attach_security(app, jwt_manager, rbac)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.get("/policies")
    async def list_policies(request: Request) -> dict:
        principal: Principal = request.state.principal
        return {
            "tenant": principal.tenant,
            "subject": principal.subject,
            "roles": list(principal.roles),
            "policies": [],
        }

    @app.get("/policies/{policy_id}")
    async def get_policy(policy_id: str, request: Request) -> dict:
        principal: Principal = request.state.principal
        return {
            "tenant": principal.tenant,
            "policy_id": policy_id,
            "viewer": principal.subject,
        }

    @app.post("/policies/{policy_id}")
    async def emit_policy(
        policy_id: str,
        request: Request,
        principal: Principal = Depends(require_role("operator", "admin")),
    ) -> dict:
        return {
            "tenant": principal.tenant,
            "policy_id": policy_id,
            "emitted_by": principal.subject,
        }

    @app.get("/audit/recent")
    async def audit_recent(
        request: Request,
        principal: Principal = Depends(
            require_role("auditor", "regulator", "admin")
        ),
    ) -> dict:
        return {"tenant": principal.tenant, "records": []}

    @app.get("/state")
    async def get_state(request: Request) -> dict:
        principal: Principal = request.state.principal
        return {"tenant": principal.tenant, "ok": True}

    return app


def build_app_from_env(
    signing_key_path: Path | str,
    issuer: str = "horizon-ric",
    audience: str = "horizon-ric-api",
) -> FastAPI:
    """Convenience: build a real app with real Casbin + JWTManager.

    Used by `scripts/horizon_security.py` and by smoke tests."""
    rbac = Casbin()
    jwt_manager = JWTManager(
        signing_key_path=signing_key_path, issuer=issuer, audience=audience
    )
    return build_app(jwt_manager, rbac)


# ---------------------------------------------------------------------------
# SLA management routes — wired in by `attach_sla_routes(app, engine)`
# ---------------------------------------------------------------------------


def attach_sla_routes(app: FastAPI, engine=None) -> FastAPI:
    """Attach the SLA-management REST surface to `app`.

    Endpoints:
        GET  /api/v1/slas                       — list configured SLAs
        POST /api/v1/slas                       — create
        GET  /api/v1/slas/{id}/breaches         — historical breaches
        POST /api/v1/sla/breaches/{id}/ack      — acknowledge
        GET  /api/v1/sla/timeline               — single-SLO time series
    """
    from datetime import datetime

    from fastapi import HTTPException, Query

    from horizon_ric.sla.policy import (
        SLA,
        SLOTarget,
        ack_breach,
        default_engine,
        list_breaches,
        list_slas,
        persist_sla,
    )

    if engine is None:
        engine = default_engine()

    @app.get("/api/v1/slas")
    def api_list_slas() -> list[dict]:
        return [s.model_dump() for s in list_slas(engine)]

    @app.post("/api/v1/slas", status_code=201)
    def api_create_sla(payload: dict) -> dict:
        try:
            sla = SLA(
                id=payload.get("id"),
                name=payload["name"],
                description=payload.get("description", ""),
                scope=payload.get("scope", {}),
                targets=[SLOTarget(**t) for t in payload["targets"]],
                severity_levels=payload.get("severity_levels", {}),
            ) if payload.get("id") else SLA(
                name=payload["name"],
                description=payload.get("description", ""),
                scope=payload.get("scope", {}),
                targets=[SLOTarget(**t) for t in payload["targets"]],
                severity_levels=payload.get("severity_levels", {}),
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        persist_sla(engine, sla)
        return sla.model_dump()

    @app.get("/api/v1/slas/{sla_id}/breaches")
    def api_list_breaches(sla_id: str, since: str | None = Query(None)) -> list[dict]:
        cutoff = None
        if since:
            cutoff = datetime.fromisoformat(since.replace("Z", "+00:00"))
        rows = list_breaches(engine, sla_id=sla_id, since=cutoff)
        return [b.model_dump(mode="json") for b in rows]

    @app.post("/api/v1/sla/breaches/{breach_id}/ack")
    def api_ack_breach(breach_id: str, body: dict | None = None) -> dict:
        by = (body or {}).get("by", "api")
        ok = ack_breach(engine, breach_id, by=by)
        if not ok:
            raise HTTPException(status_code=404, detail="breach not found")
        return {"ok": True, "breach_id": breach_id, "acknowledged_by": by}

    @app.get("/api/v1/sla/timeline")
    def api_sla_timeline(
        metric: str = Query(...),
        from_ts: str = Query(..., alias="from"),
        to_ts: str = Query(..., alias="to"),
    ) -> dict:
        f = datetime.fromisoformat(from_ts.replace("Z", "+00:00"))
        t = datetime.fromisoformat(to_ts.replace("Z", "+00:00"))
        if t <= f:
            raise HTTPException(status_code=400, detail="`to` must be after `from`")
        # Pull breaches for context; the actual time-series of observations
        # is owned by Prometheus. We return the breach overlay here.
        rows = [
            b
            for b in list_breaches(engine, since=f)
            if b.source_metric == metric and b.ts_utc <= t
        ]
        return {
            "metric": metric,
            "from": f.isoformat(),
            "to": t.isoformat(),
            "breaches": [b.model_dump(mode="json") for b in rows],
        }

    return app


__all__ = [
    "build_app",
    "build_app_from_env",
    "attach_security",
    "attach_sla_routes",
]

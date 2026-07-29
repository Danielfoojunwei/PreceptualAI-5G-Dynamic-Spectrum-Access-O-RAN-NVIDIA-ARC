"""Minimal FastAPI scaffolding for the rApp REST surface.

STATUS (2026-07): this module is a security scaffold, NOT the API the
rApp serves. `HorizonRAppLifecycle._serve_api()` mounts
`horizon_ric.rapp.dashboard_api` (HS256 + per-endpoint role checks) on
port 8083; this module's RS256 `JWTManager` + Casbin `build_app` wraps
mostly-empty demo handlers and is exercised only by the security
middleware tests. It also re-exports `build_api` from the seeded
SDK/demo module `horizon_ric.rapp.api_v1`. See docs/API_SURFACES.md for
the full module map before adding endpoints here.

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
    app = FastAPI(title="Horizon-RIC rApp", version="0.1.0")
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


__all__ = [
    "build_app",
    "build_app_from_env",
    "attach_security",
]

"""FastAPI middleware that authenticates JWTs and enforces RBAC.

Pipeline per request:

    1. Extract `Authorization: Bearer <jwt>` header.
       Missing / malformed  → 401 + `auth.token_invalid`.
    2. Verify the JWT via `JWTManager.verify_token`.
       Bad signature / expired / wrong aud / missing claims  → 401.
    3. Resolve the (sub, tenant) from the claims and call
       `Casbin.enforce(sub, tenant, path, method)`.
       Deny  → 403 + `auth.denied`.
    4. If `X-Tenant` header is present it must equal `claims["tenant"]`
       (resource tenant must match token tenant).
    5. Attach `request.state.principal` for downstream handlers and
       emit `auth.granted`.

`require_role(*roles)` and `require_tenant(tenant_id)` are FastAPI
`Depends`-compatible helpers for fine-grained per-route enforcement
on top of the global middleware.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable

import structlog
from fastapi import HTTPException, Request, status
from jose.exceptions import JWTError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from horizon_ric.security.jwt import JWTManager
from horizon_ric.security.rbac import Casbin

# Audit logger — stable event names used by the SOC tooling.
_log = structlog.get_logger("horizon_ric.security")

# Paths that should bypass auth entirely. Liveness/readiness must always
# be reachable for k8s probes.
_PUBLIC_PATHS: tuple[str, ...] = (
    "/healthz",
    "/livez",
    "/readyz",
    "/openapi.json",
    "/docs",
    "/redoc",
)

_BEARER_RE = re.compile(r"^\s*Bearer\s+(?P<token>[A-Za-z0-9._\-]+)\s*$")


@dataclass(frozen=True)
class Principal:
    """Authenticated identity attached to `request.state.principal`."""

    subject: str
    tenant: str
    roles: tuple[str, ...]
    jti: str | None = None


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Verify JWTs and enforce Casbin RBAC on every protected route."""

    def __init__(
        self,
        app,
        jwt_manager: JWTManager,
        rbac: Casbin,
        public_paths: Iterable[str] = _PUBLIC_PATHS,
    ):
        super().__init__(app)
        self._jwt = jwt_manager
        self._rbac = rbac
        self._public = tuple(public_paths)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in self._public):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            _log.warning("auth.token_invalid", reason="missing_authorization", path=path)
            return _json_error(401, "missing Authorization header")

        m = _BEARER_RE.match(auth_header)
        if not m:
            _log.warning("auth.token_invalid", reason="malformed_header", path=path)
            return _json_error(401, "malformed Authorization header")

        token = m.group("token")
        try:
            claims = self._jwt.verify_token(token)
        except JWTError as exc:
            _log.warning(
                "auth.token_invalid",
                reason="jwt_decode_failed",
                error=str(exc),
                path=path,
            )
            return _json_error(401, "invalid token")

        sub = claims.get("sub", "")
        tok_tenant = claims.get("tenant", "")
        roles = tuple(claims.get("roles", []) or [])
        jti = claims.get("jti")

        # Resource-tenant enforcement: if the caller specifies the tenant
        # they're operating on, it MUST match the token's tenant claim.
        resource_tenant = request.headers.get("X-Tenant")
        if resource_tenant and resource_tenant != tok_tenant:
            _log.warning(
                "auth.denied",
                reason="tenant_mismatch",
                subject=sub,
                token_tenant=tok_tenant,
                resource_tenant=resource_tenant,
                path=path,
            )
            return _json_error(403, "tenant mismatch")

        # Casbin object is the URL path; action is the HTTP method, lowercased
        # except where Casbin policies use semantic verbs (read/emit/etc).
        # We derive a semantic action from method as a default and let
        # routes override via `require_role(...)` if they need finer
        # control.
        action = _http_method_to_action(request.method)
        # Strip leading slash so policies match `policies/*` form.
        obj = path.lstrip("/")
        allowed = self._rbac.enforce(sub, tok_tenant, obj, action)
        if not allowed:
            _log.warning(
                "auth.denied",
                subject=sub,
                tenant=tok_tenant,
                roles=list(roles),
                obj=obj,
                act=action,
                path=path,
            )
            return _json_error(403, "forbidden")

        request.state.principal = Principal(
            subject=sub, tenant=tok_tenant, roles=roles, jti=jti
        )
        _log.info(
            "auth.granted",
            subject=sub,
            tenant=tok_tenant,
            roles=list(roles),
            obj=obj,
            act=action,
            path=path,
        )
        return await call_next(request)


def _http_method_to_action(method: str) -> str:
    """Map HTTP verb → Casbin action string used in our policy seed."""
    m = method.upper()
    if m == "GET" or m == "HEAD" or m == "OPTIONS":
        return "read"
    if m == "POST":
        return "emit"
    if m == "PUT" or m == "PATCH":
        return "emit"
    if m == "DELETE":
        return "rollback"
    return m.lower()


def _json_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


# ---------------------------------------------------------------------------
# Per-route Depends helpers
# ---------------------------------------------------------------------------
def require_role(*allowed_roles: str):
    """Return a FastAPI dependency that ensures the caller holds at least
    one of `allowed_roles` (in their token's tenant)."""

    async def _dep(request: Request) -> Principal:
        principal = getattr(request.state, "principal", None)
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="not authenticated",
            )
        if not any(r in principal.roles for r in allowed_roles):
            _log.warning(
                "auth.denied",
                reason="role_required",
                subject=principal.subject,
                tenant=principal.tenant,
                required=list(allowed_roles),
                held=list(principal.roles),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="role required",
            )
        return principal

    return _dep


def require_tenant(tenant_id: str):
    """Return a FastAPI dependency that ensures the caller's tenant
    matches `tenant_id` exactly (used to lock a route to a single tenant)."""

    async def _dep(request: Request) -> Principal:
        principal = getattr(request.state, "principal", None)
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="not authenticated",
            )
        if principal.tenant != tenant_id:
            _log.warning(
                "auth.denied",
                reason="tenant_required",
                subject=principal.subject,
                tenant=principal.tenant,
                required_tenant=tenant_id,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="tenant mismatch",
            )
        return principal

    return _dep


__all__ = [
    "JWTAuthMiddleware",
    "Principal",
    "require_role",
    "require_tenant",
]

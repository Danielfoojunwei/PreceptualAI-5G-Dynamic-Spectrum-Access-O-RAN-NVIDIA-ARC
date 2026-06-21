"""Tenant-scoped context for the evidence store.

`TenantScope(tenant_id)` is a context manager that pins all
`EvidenceStore.append` / iteration calls inside its `with` block to a
specific tenant. Each tenant maintains its own independent SHA-256 hash
chain — appends from tenant A never affect tenant B's chain, and an
auditor authenticated for tenant A cannot read tenant B's records.

Implementation:
    A `contextvars.ContextVar` holds the current tenant id. The
    `EvidenceStore.append` path consults this var to stamp the record
    and pick the correct per-tenant chain. The var is async-safe — a
    coroutine for tenant A and one for tenant B running concurrently
    on the same event loop never see each other's tenant id.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

# The current tenant. None outside of any TenantScope. We deliberately
# do NOT default to "default" at module level — the evidence store will
# refuse to write without an active scope so misuse fails loudly.
_CURRENT_TENANT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "horizon_ric_current_tenant", default=None
)


class TenantScope:
    """Scope a block of code to a specific tenant.

    Both sync and async usage are supported because `contextvars` is
    safe across both. Nested scopes restore the outer tenant on exit.
    """

    def __init__(self, tenant_id: str):
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a non-empty string")
        self._tenant_id = tenant_id
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "TenantScope":
        self._token = _CURRENT_TENANT.set(self._tenant_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _CURRENT_TENANT.reset(self._token)
            self._token = None

    async def __aenter__(self) -> "TenantScope":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.__exit__(exc_type, exc, tb)

    @property
    def tenant_id(self) -> str:
        return self._tenant_id


def current_tenant() -> str | None:
    """Return the currently-active tenant, or None if no scope is active."""
    return _CURRENT_TENANT.get()


def require_current_tenant() -> str:
    """Like `current_tenant` but raises if no scope is active."""
    t = _CURRENT_TENANT.get()
    if t is None:
        raise RuntimeError(
            "no active TenantScope; wrap the operation in "
            "`with TenantScope('<tenant_id>'): ...`"
        )
    return t


@contextmanager
def override_tenant(tenant_id: str | None) -> Iterator[None]:
    """Temporarily override the tenant. Used by tests; production code
    should prefer `TenantScope`."""
    token = _CURRENT_TENANT.set(tenant_id)
    try:
        yield
    finally:
        _CURRENT_TENANT.reset(token)


__all__ = [
    "TenantScope",
    "current_tenant",
    "require_current_tenant",
    "override_tenant",
]

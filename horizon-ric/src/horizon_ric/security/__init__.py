"""PreceptualAI security: RBAC + multi-tenancy + JWT-bearer auth.

Public re-exports — every consumer of the security layer should import
from this package, not from the submodules directly.
"""

from horizon_ric.security.hsm import (
    HSMBackend,
    InMemoryHSMBackend,
    SoftHSM2Backend,
)
from horizon_ric.security.jwt import JWTManager
from horizon_ric.security.middleware import (
    JWTAuthMiddleware,
    Principal,
    require_role,
    require_tenant,
)
from horizon_ric.security.rbac import Casbin, Role, Tenant, User
from horizon_ric.security.tenant import TenantScope, current_tenant

__all__ = [
    "Casbin",
    "Role",
    "Tenant",
    "User",
    "JWTManager",
    "JWTAuthMiddleware",
    "Principal",
    "require_role",
    "require_tenant",
    "TenantScope",
    "current_tenant",
    "HSMBackend",
    "InMemoryHSMBackend",
    "SoftHSM2Backend",
]

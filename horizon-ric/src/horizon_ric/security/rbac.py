"""Casbin-backed RBAC with domains for PreceptualAI multi-tenant access control.

Production-grade RBAC built on Casbin (Apache 2.0). Five default roles per
tenant (`admin`, `operator`, `auditor`, `regulator`, `api-user`) — see
`rbac_policy.csv` for the seed permissions and `docs/RBAC.md` for the
full table.

The model file (`rbac_model.conf`) uses the canonical Casbin "RBAC with
domains" template. The matcher resolves a user's roles in the request
domain (tenant) and then checks each role's permission against the
requested object/action. `keyMatch` enables wildcard objects like
`policies/*`.

Usage:
    >>> rbac = Casbin()
    >>> rbac.add_role("alice", "operator", "default")
    >>> rbac.enforce("alice", "default", "policies/abc", "emit")
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import casbin

# ---------------------------------------------------------------------------
# Default file locations — kept beside this module so tests and the CLI can
# locate them without env vars.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH: Path = _HERE / "rbac_model.conf"
DEFAULT_POLICY_PATH: Path = _HERE / "rbac_policy.csv"


# ---------------------------------------------------------------------------
# Lightweight value objects for type hints / external consumers.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Role:
    """A named RBAC role (e.g. admin, operator)."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Tenant:
    """A tenant identity (Casbin domain)."""

    id: str

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True)
class User:
    """A user identity (Casbin subject)."""

    name: str

    def __str__(self) -> str:
        return self.name


# Canonical role set — kept in lock-step with rbac_policy.csv.
DEFAULT_ROLES: tuple[str, ...] = (
    "admin",
    "operator",
    "auditor",
    "regulator",
    "api-user",
)

# Tenant domains shipped in the seed CSV. ``default`` is the legacy
# back-compat domain (kept for existing tests + single-tenant lab
# deployments). The named tenants are the real per-tenant domains
# introduced in Devil-A Finding #16 closure.
SEED_TENANTS: tuple[str, ...] = (
    "default",
    "tenant_alpha",
    "tenant_bravo",
    "tenant_charlie",
)


class Casbin:
    """Wrapper around `casbin.Enforcer` providing a small, stable API.

    Loads the model + policy from CSV at construction. The policy is
    persisted back to the same CSV on `add_role` / `remove_role` so that
    changes survive a process restart and are visible to any operator
    inspecting the file.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        policy_path: Path | str | None = None,
    ):
        self._model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._policy_path = Path(policy_path or DEFAULT_POLICY_PATH)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Casbin model not found: {self._model_path}")
        if not self._policy_path.exists():
            raise FileNotFoundError(f"Casbin policy not found: {self._policy_path}")
        self._enforcer = casbin.Enforcer(
            str(self._model_path), str(self._policy_path)
        )
        # auto_save flushes adapter changes back to the CSV.
        self._enforcer.enable_auto_save(True)

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------
    def enforce(self, sub: str, dom: str, obj: str, act: str) -> bool:
        """Return True iff `sub` may perform `act` on `obj` in domain `dom`."""
        return bool(self._enforcer.enforce(sub, dom, obj, act))

    # ------------------------------------------------------------------
    # Role assignment (Casbin grouping policy `g`)
    # ------------------------------------------------------------------
    def add_role(self, user: str, role: str, tenant: str) -> bool:
        """Bind `user` to `role` within `tenant`. No-op if already bound.

        Returns True if a new binding was added. Persists to the policy
        CSV immediately so out-of-process readers see the change.
        """
        added = bool(self._enforcer.add_grouping_policy(user, role, tenant))
        if added:
            self._enforcer.save_policy()
        return added

    def remove_role(self, user: str, role: str, tenant: str) -> bool:
        """Unbind `user` from `role` in `tenant`. Returns True if removed."""
        removed = bool(self._enforcer.remove_grouping_policy(user, role, tenant))
        if removed:
            self._enforcer.save_policy()
        return removed

    def list_users_for_role(self, role: str, tenant: str) -> list[str]:
        """Return the users bound to `role` in `tenant`."""
        # Casbin's `get_users_for_role_in_domain` returns the inverse of what
        # we need? No — it does what its name says: users with this role in
        # the domain.
        return list(self._enforcer.get_users_for_role_in_domain(role, tenant))

    def list_roles_for_user(self, user: str, tenant: str) -> list[str]:
        """Return the roles bound to `user` in `tenant`."""
        return list(self._enforcer.get_roles_for_user_in_domain(user, tenant))

    def list_grouping_policies(self) -> list[list[str]]:
        """Return every (user, role, tenant) binding currently loaded."""
        return [list(p) for p in self._enforcer.get_grouping_policy()]

    def list_policies(self) -> list[list[str]]:
        """Return every (role, tenant, object, action) policy currently loaded."""
        return [list(p) for p in self._enforcer.get_policy()]

    def list_tenants(self) -> list[str]:
        """Return the set of tenant domains present in the loaded policy.

        Devil-A Finding #16 closure: per-tenant domains. This lets the
        deployment helm chart enumerate which tenants the seed knows
        about and decide whether ``default`` is acceptable in this env.
        """
        seen: set[str] = set()
        for row in self._enforcer.get_policy():
            if len(row) >= 2:
                seen.add(row[1])
        return sorted(seen)

    def assert_no_default_domain_in_production(self) -> None:
        """Refuse to operate if ``default`` is the only domain present.

        A regulator-grade deployment MUST have at least one named tenant
        domain; ``default`` exists for legacy/lab use only. This helper
        is intended to be called from the rApp lifecycle pre-flight.
        """
        tenants = self.list_tenants()
        non_default = [t for t in tenants if t != "default"]
        if not non_default:
            raise RuntimeError(
                "RBAC seed has only the 'default' domain; production "
                "deployment requires at least one named tenant domain "
                "(see Devil-A Finding #16 closure / rbac_policy.csv)."
            )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def reload(self) -> None:
        """Reload the policy CSV (e.g. after an out-of-band edit)."""
        self._enforcer.load_policy()

    def save(self) -> None:
        """Persist the in-memory policy to disk."""
        self._enforcer.save_policy()


def known_roles() -> Iterable[str]:
    """Return the canonical role names shipped with the policy seed."""
    return DEFAULT_ROLES


__all__ = [
    "Casbin",
    "Role",
    "Tenant",
    "User",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_POLICY_PATH",
    "DEFAULT_ROLES",
    "known_roles",
]

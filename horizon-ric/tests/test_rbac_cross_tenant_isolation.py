"""Cross-tenant isolation tests for the per-tenant Casbin domains.

Devil-A Finding #16 closure: tenant_alpha's roles must NOT enforce on
tenant_bravo's resources, and vice versa. The seed CSV ships three
named tenants — ``tenant_alpha``, ``tenant_bravo``, ``tenant_charlie``
— each with the same role catalogue but separate domain rows.

The matcher rule is `r.dom == p.dom`: if Alice has only the role binding
``g, alice, operator, tenant_alpha`` then a request with
``dom=tenant_bravo`` MUST evaluate to False for *every* object/action.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from horizon_ric.security.rbac import (
    Casbin,
    DEFAULT_MODEL_PATH,
    DEFAULT_POLICY_PATH,
)


@pytest.fixture
def rbac(tmp_path) -> Casbin:
    """Fresh enforcer with a writable copy of the seed CSV.

    Uses a tmp_path-scoped copy so add_role's auto_save doesn't pollute
    the committed seed (Devil-A Finding #16: per-tenant seed must remain
    deterministic).
    """
    policy_copy = tmp_path / "rbac_policy.csv"
    shutil.copy(DEFAULT_POLICY_PATH, policy_copy)
    return Casbin(model_path=DEFAULT_MODEL_PATH, policy_path=policy_copy)


def test_seed_csv_ships_three_named_tenants(rbac: Casbin) -> None:
    tenants = rbac.list_tenants()
    for t in ("tenant_alpha", "tenant_bravo", "tenant_charlie"):
        assert t in tenants, (
            f"per-tenant seed missing {t!r}; "
            f"Devil-A Finding #16 closure requires three named tenants. "
            f"Got: {tenants}"
        )


def test_alpha_operator_cannot_emit_in_bravo(rbac: Casbin) -> None:
    """Operator scoped to tenant_alpha must NOT emit policies in tenant_bravo."""
    rbac.add_role("alice", "operator", "tenant_alpha")
    # Allowed in own tenant.
    assert rbac.enforce("alice", "tenant_alpha", "policies/p-1", "emit")
    # Refused in another tenant.
    assert not rbac.enforce("alice", "tenant_bravo", "policies/p-1", "emit")
    assert not rbac.enforce("alice", "tenant_charlie", "policies/p-1", "emit")


def test_alpha_auditor_cannot_read_bravo_audit(rbac: Casbin) -> None:
    """Auditor in tenant_alpha must NOT read tenant_bravo audit log."""
    rbac.add_role("audrey", "auditor", "tenant_alpha")
    assert rbac.enforce("audrey", "tenant_alpha", "audit/recent", "read")
    assert not rbac.enforce("audrey", "tenant_bravo", "audit/recent", "read")


def test_alpha_admin_does_not_become_bravo_admin(rbac: Casbin) -> None:
    """Admin role is per-tenant: admin@alpha must not be admin@bravo."""
    rbac.add_role("root_alpha", "admin", "tenant_alpha")
    assert rbac.enforce("root_alpha", "tenant_alpha", "anything", "delete")
    assert not rbac.enforce("root_alpha", "tenant_bravo", "anything", "delete")
    assert not rbac.enforce("root_alpha", "tenant_charlie", "policies/x", "emit")


def test_user_with_no_role_in_tenant_is_refused_everywhere_in_that_tenant(
    rbac: Casbin,
) -> None:
    """A user with role only in tenant_alpha must hit zero hits in tenant_bravo."""
    rbac.add_role("alice", "regulator", "tenant_alpha")
    assert rbac.enforce("alice", "tenant_alpha", "policies/p-1", "explain")
    for obj, act in [
        ("policies/p-1", "explain"),
        ("policies/p-1", "read"),
        ("audit/recent", "read"),
        ("audit/x", "verify"),
        ("anything", "anything"),
    ]:
        assert not rbac.enforce("alice", "tenant_bravo", obj, act), (
            f"cross-tenant leak: alice@tenant_alpha enforced "
            f"{obj!r}/{act!r} on tenant_bravo"
        )


def test_default_domain_does_not_leak_to_named_tenants(rbac: Casbin) -> None:
    """A user with role only in ``default`` must not enforce against named tenants."""
    rbac.add_role("legacy", "admin", "default")
    assert rbac.enforce("legacy", "default", "anything", "anything")
    for t in ("tenant_alpha", "tenant_bravo", "tenant_charlie"):
        assert not rbac.enforce("legacy", t, "policies/x", "emit"), (
            f"admin@default leaked into {t}"
        )


def test_assert_no_default_domain_in_production_passes_with_named_tenants(
    rbac: Casbin,
) -> None:
    rbac.assert_no_default_domain_in_production()

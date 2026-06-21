"""Casbin RBAC tests — exercise every role × resource × action and the
tenant-isolation invariants."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from horizon_ric.security.rbac import (
    DEFAULT_MODEL_PATH,
    DEFAULT_POLICY_PATH,
    Casbin,
)


@pytest.fixture()
def rbac(tmp_path: Path) -> Casbin:
    """Fresh Casbin enforcer with a writable copy of the seed CSV.

    Tests that mutate the policy must not bleed into other tests; we
    copy the canonical CSV into the per-test tmpdir."""
    pol = tmp_path / "rbac_policy.csv"
    shutil.copyfile(DEFAULT_POLICY_PATH, pol)
    return Casbin(model_path=DEFAULT_MODEL_PATH, policy_path=pol)


# ---------------------------------------------------------------------------
# 1. admin can do anything
# ---------------------------------------------------------------------------
def test_admin_full_access(rbac: Casbin) -> None:
    rbac.add_role("alice", "admin", "default")
    assert rbac.enforce("alice", "default", "policies/abc", "emit")
    assert rbac.enforce("alice", "default", "audit/recent", "verify")
    assert rbac.enforce("alice", "default", "anything/at/all", "delete")
    assert rbac.enforce("alice", "default", "lifecycle", "degrade")


# ---------------------------------------------------------------------------
# 2. operator: read/emit/rollback policies + lifecycle read/degrade
# ---------------------------------------------------------------------------
def test_operator_policies_emit(rbac: Casbin) -> None:
    rbac.add_role("op", "operator", "default")
    assert rbac.enforce("op", "default", "policies/foo", "read")
    assert rbac.enforce("op", "default", "policies/foo", "emit")
    assert rbac.enforce("op", "default", "policies/foo", "rollback")
    assert rbac.enforce("op", "default", "lifecycle", "degrade")
    # operator may NOT touch audit
    assert not rbac.enforce("op", "default", "audit/recent", "verify")


# ---------------------------------------------------------------------------
# 3. auditor: read/verify audit, read policies, no emit
# ---------------------------------------------------------------------------
def test_auditor_read_only(rbac: Casbin) -> None:
    rbac.add_role("aud", "auditor", "default")
    assert rbac.enforce("aud", "default", "audit/x", "read")
    assert rbac.enforce("aud", "default", "audit/x", "verify")
    assert rbac.enforce("aud", "default", "policies/p1", "read")
    assert not rbac.enforce("aud", "default", "policies/p1", "emit")


# ---------------------------------------------------------------------------
# 4. regulator: audit + policies/explain, but no policy emission
# ---------------------------------------------------------------------------
def test_regulator_audit_and_explain(rbac: Casbin) -> None:
    rbac.add_role("reg", "regulator", "default")
    assert rbac.enforce("reg", "default", "audit/recent", "read")
    assert rbac.enforce("reg", "default", "audit/recent", "verify")
    assert rbac.enforce("reg", "default", "policies/p1", "explain")
    assert rbac.enforce("reg", "default", "policies/p1", "read")
    assert not rbac.enforce("reg", "default", "policies/p1", "emit")


# ---------------------------------------------------------------------------
# 5. api-user: only state read + policies read
# ---------------------------------------------------------------------------
def test_api_user_minimal(rbac: Casbin) -> None:
    rbac.add_role("bot", "api-user", "default")
    assert rbac.enforce("bot", "default", "state", "read")
    assert rbac.enforce("bot", "default", "policies", "read")
    assert not rbac.enforce("bot", "default", "policies", "emit")
    assert not rbac.enforce("bot", "default", "audit/recent", "read")


# ---------------------------------------------------------------------------
# 6. unbound user gets nothing
# ---------------------------------------------------------------------------
def test_no_role_denied(rbac: Casbin) -> None:
    assert not rbac.enforce("ghost", "default", "policies", "read")
    assert not rbac.enforce("ghost", "default", "audit/x", "read")
    assert not rbac.enforce("ghost", "default", "anything", "read")


# ---------------------------------------------------------------------------
# 7. Tenant cross-access denied
# ---------------------------------------------------------------------------
def test_cross_tenant_denied(rbac: Casbin) -> None:
    rbac.add_role("alice", "admin", "tenant_a")
    # Need policy seed for tenant_a to give admin power; without it
    # admin in tenant_a still has no permissions because the seed only
    # covers `default`. So replicate the admin grant for tenant_a:
    # we add a policy directly via the enforcer.
    rbac._enforcer.add_policy("admin", "tenant_a", "*", "*")
    rbac._enforcer.add_policy("admin", "tenant_b", "*", "*")
    assert rbac.enforce("alice", "tenant_a", "anything", "anything")
    # alice has admin in tenant_a only; tenant_b should refuse her.
    assert not rbac.enforce("alice", "tenant_b", "anything", "anything")


# ---------------------------------------------------------------------------
# 8. Role addition takes effect immediately
# ---------------------------------------------------------------------------
def test_role_add_takes_effect(rbac: Casbin) -> None:
    assert not rbac.enforce("dora", "default", "policies/x", "emit")
    rbac.add_role("dora", "operator", "default")
    assert rbac.enforce("dora", "default", "policies/x", "emit")


# ---------------------------------------------------------------------------
# 9. Role removal takes effect immediately
# ---------------------------------------------------------------------------
def test_role_remove_takes_effect(rbac: Casbin) -> None:
    rbac.add_role("dora", "operator", "default")
    assert rbac.enforce("dora", "default", "policies/x", "emit")
    rbac.remove_role("dora", "operator", "default")
    assert not rbac.enforce("dora", "default", "policies/x", "emit")


# ---------------------------------------------------------------------------
# 10. CSV persistence: changes survive a Casbin reload
# ---------------------------------------------------------------------------
def test_csv_persistence(tmp_path: Path) -> None:
    pol = tmp_path / "rbac_policy.csv"
    shutil.copyfile(DEFAULT_POLICY_PATH, pol)

    r1 = Casbin(model_path=DEFAULT_MODEL_PATH, policy_path=pol)
    r1.add_role("persistent_user", "operator", "default")
    r1.save()

    # Re-load from disk in a fresh enforcer.
    r2 = Casbin(model_path=DEFAULT_MODEL_PATH, policy_path=pol)
    assert r2.enforce("persistent_user", "default", "policies/foo", "emit")


# ---------------------------------------------------------------------------
# 11. list_users_for_role round-trip
# ---------------------------------------------------------------------------
def test_list_users_for_role(rbac: Casbin) -> None:
    rbac.add_role("alice", "operator", "default")
    rbac.add_role("bob", "operator", "default")
    rbac.add_role("carol", "auditor", "default")
    ops = rbac.list_users_for_role("operator", "default")
    assert "alice" in ops
    assert "bob" in ops
    assert "carol" not in ops


# ---------------------------------------------------------------------------
# 12. operator denied admin-only paths
# ---------------------------------------------------------------------------
def test_operator_denied_audit(rbac: Casbin) -> None:
    rbac.add_role("op", "operator", "default")
    assert not rbac.enforce("op", "default", "audit/recent", "read")
    assert not rbac.enforce("op", "default", "audit/recent", "verify")


# ---------------------------------------------------------------------------
# 13. Multiple roles for one user — union of permissions
# ---------------------------------------------------------------------------
def test_user_with_multiple_roles(rbac: Casbin) -> None:
    rbac.add_role("multi", "operator", "default")
    rbac.add_role("multi", "auditor", "default")
    # operator gives policy emit
    assert rbac.enforce("multi", "default", "policies/x", "emit")
    # auditor gives audit verify
    assert rbac.enforce("multi", "default", "audit/x", "verify")


# ---------------------------------------------------------------------------
# 14. Wildcard object matching (keyMatch) works as expected
# ---------------------------------------------------------------------------
def test_keymatch_wildcard(rbac: Casbin) -> None:
    rbac.add_role("op", "operator", "default")
    # `policies/*` should match nested paths
    assert rbac.enforce("op", "default", "policies/abc/123", "emit")
    # but not the literal "policies" (without a slash) — semantic check
    # against the seed (operator only has emit on `policies/*`).
    assert not rbac.enforce("op", "default", "policies", "emit")

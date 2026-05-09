"""Tests for LIConstraint — Lawful Intercept jurisdiction hook.

Covers WG11 / ETSI TS 103 221 compliance surface:

    * advertised hard-constraint IDs
    * protected-UE refusal
    * non-protected-UE pass-through
    * protected-slice + wrong-jurisdiction refusal
    * protected-slice + allowed-jurisdiction pass
    * projection strips protected UEs
    * empty rule list yields zero violations
    * lagrangian returns zeros (LI must never be soft)
"""

from __future__ import annotations

import torch

from horizon_ric.policy import LIConstraint, LIJurisdictionRule


def _rule_de() -> LIJurisdictionRule:
    return LIJurisdictionRule(
        rule_id="warrant-de-001",
        protected_ue_ids={"ue-target-A", "ue-target-B"},
        protected_slice_ids={"slice-warrant-de"},
        allowed_jurisdictions={"DE", "EU"},
        description="DE-issued warrant; bearer must remain in DE/EU.",
    )


def _rule_uk() -> LIJurisdictionRule:
    return LIJurisdictionRule(
        rule_id="warrant-uk-002",
        protected_ue_ids={"ue-target-C"},
        protected_slice_ids={"slice-warrant-uk"},
        allowed_jurisdictions={"UK"},
        description="UK-issued warrant; UK-only anchor.",
    )


# ─── advertised constraint IDs ─────────────────────────────────────────


class TestConstraintIDs:
    def test_advertises_hard_ids(self):
        cl = LIConstraint(rules=[_rule_de()])
        ids = cl.hard_constraint_ids()
        # Three hard ids: protected-UE, jurisdiction, and the synthetic
        # fail-closed gate added in Devil-A Finding #1 closure.
        assert "li_protected_ue" in ids
        assert "li_jurisdiction" in ids
        assert "li_fail_closed" in ids


# ─── protected UE behaviour ────────────────────────────────────────────


class TestProtectedUE:
    def test_protected_ue_is_blocked(self):
        cl = LIConstraint(rules=[_rule_de()])
        action = {
            "affected_ue_ids": {"ue-target-A", "ue-normal-1"},
            "affected_slice_ids": set(),
            "target_jurisdiction": "DE",
        }
        viols = cl.check_feasibility(action, context={})
        ids = [v.constraint_id for v in viols]
        assert "li_protected_ue" in ids
        # the message must name the offending UE for the audit log
        msg = next(v.message for v in viols if v.constraint_id == "li_protected_ue")
        assert "ue-target-A" in msg

    def test_non_protected_ue_passes(self):
        cl = LIConstraint(rules=[_rule_de(), _rule_uk()])
        action = {
            "affected_ue_ids": {"ue-normal-1", "ue-normal-2"},
            "affected_slice_ids": set(),
            "target_jurisdiction": "DE",
        }
        viols = cl.check_feasibility(action, context={})
        assert viols == []

    def test_protected_ue_under_second_rule_is_blocked(self):
        # Make sure rule iteration covers all rules, not just the first.
        cl = LIConstraint(rules=[_rule_de(), _rule_uk()])
        action = {
            "affected_ue_ids": {"ue-target-C"},
        }
        viols = cl.check_feasibility(action, context={})
        assert any(v.constraint_id == "li_protected_ue" for v in viols)


# ─── protected slice + jurisdiction behaviour ──────────────────────────


class TestSliceJurisdiction:
    def test_protected_slice_wrong_jurisdiction_blocked(self):
        cl = LIConstraint(rules=[_rule_de()])
        action = {
            "affected_ue_ids": set(),
            "affected_slice_ids": {"slice-warrant-de"},
            "target_jurisdiction": "US",  # not in {DE, EU}
        }
        viols = cl.check_feasibility(action, context={})
        ids = [v.constraint_id for v in viols]
        assert "li_jurisdiction" in ids

    def test_protected_slice_allowed_jurisdiction_passes(self):
        cl = LIConstraint(rules=[_rule_de()])
        action = {
            "affected_ue_ids": set(),
            "affected_slice_ids": {"slice-warrant-de"},
            "target_jurisdiction": "EU",  # in {DE, EU}
        }
        viols = cl.check_feasibility(action, context={})
        assert viols == []

    def test_non_protected_slice_any_jurisdiction_passes(self):
        cl = LIConstraint(rules=[_rule_de()])
        action = {
            "affected_ue_ids": set(),
            "affected_slice_ids": {"slice-public-iot"},
            "target_jurisdiction": "ZW",  # nowhere on the allow-list,
                                          # but slice isn't protected
        }
        viols = cl.check_feasibility(action, context={})
        assert viols == []


# ─── projection ────────────────────────────────────────────────────────


class TestProjection:
    def test_projection_removes_protected_ues(self):
        cl = LIConstraint(rules=[_rule_de()])
        action = {
            "affected_ue_ids": ["ue-target-A", "ue-target-B", "ue-normal-1"],
            "affected_slice_ids": set(),
            "target_jurisdiction": "DE",
        }
        feasible, corrections = cl.project(action, context={})
        remaining = set(feasible["affected_ue_ids"])
        assert remaining == {"ue-normal-1"}
        assert any(c.constraint_id == "li_protected_ue" for c in corrections)

    def test_projection_does_not_rewrite_jurisdiction(self):
        # Conservative-by-design: project does NOT silently move the
        # jurisdiction. The slice violation must surface in `corrections`.
        cl = LIConstraint(rules=[_rule_de()])
        action = {
            "affected_ue_ids": set(),
            "affected_slice_ids": {"slice-warrant-de"},
            "target_jurisdiction": "US",
        }
        feasible, corrections = cl.project(action, context={})
        assert feasible["target_jurisdiction"] == "US"  # unchanged
        assert any(c.constraint_id == "li_jurisdiction" for c in corrections)


# ─── empty-rules / fail-closed behaviour ──────────────────────────────


class TestEmptyRulesFailClosed:
    """Devil-A Finding #1 closure: empty rule catalogue must NOT silently pass.

    Three behaviours are required:
      1. Empty rules + fail_closed=True (default) ⇒ every action rejected
         with synthetic constraint id ``li_fail_closed``.
      2. Empty rules + fail_closed=False + non-empty deployment_audit_record
         ⇒ no rejection (legitimate no-LI lab deployment, audit-recorded).
      3. The fail-closed default is the safe default — calling
         ``LIConstraint(rules=[])`` without explicit kwargs MUST fail closed.
    """

    def test_empty_rules_default_rejects_every_action(self):
        # Default is fail_closed=True (the safe default).
        cl = LIConstraint(rules=[])
        assert cl.fail_closed is True

        action = {
            "affected_ue_ids": {"anyone"},
            "affected_slice_ids": {"any-slice"},
            "target_jurisdiction": "ZZ",
        }
        viols = cl.check_feasibility(action, context={})
        assert len(viols) == 1
        assert viols[0].constraint_id == "li_fail_closed"
        assert viols[0].severity == "hard"

        # project() also surfaces the fail-closed violation.
        feasible, corrections = cl.project(action, context={})
        assert any(c.constraint_id == "li_fail_closed" for c in corrections)
        # No-op projection: action body untouched (the emit-guard refuses it).
        assert feasible["affected_ue_ids"] == {"anyone"}

    def test_empty_rules_explicit_opt_out_with_audit_record_passes(self):
        # The escape hatch for legitimate testbed/lab deployments: the
        # operator MUST explicitly opt out AND record that election in the
        # audit chain (deployment_audit_record).
        cl = LIConstraint(
            rules=[],
            fail_closed=False,
            deployment_audit_record="audit-evidence-id-12345",
        )
        assert cl.fail_closed is False
        assert cl.deployment_audit_record == "audit-evidence-id-12345"

        action = {
            "affected_ue_ids": {"anyone"},
            "affected_slice_ids": {"any-slice"},
            "target_jurisdiction": "ZZ",
        }
        assert cl.check_feasibility(action, context={}) == []
        feasible, corrections = cl.project(action, context={})
        assert corrections == []
        assert feasible["affected_ue_ids"] == {"anyone"}

    def test_empty_rules_opt_out_without_audit_record_raises(self):
        # An operator who tries to opt out of fail-closed without an audit
        # entry MUST be refused at construction time.
        import pytest as _pytest

        with _pytest.raises(ValueError, match="audit"):
            LIConstraint(rules=[], fail_closed=False)

        with _pytest.raises(ValueError, match="audit"):
            LIConstraint(
                rules=[], fail_closed=False, deployment_audit_record=""
            )

    def test_default_is_fail_closed_safe_default(self):
        # The whole point of Devil-A Finding #1 closure: a developer who
        # forgets the kwargs gets the SAFE default, not the bypass.
        cl = LIConstraint(rules=[])
        # Synthesise a benign action — everything must still be rejected.
        action = {
            "affected_ue_ids": ["benign-ue-1"],
            "affected_slice_ids": ["public-iot"],
            "target_jurisdiction": "DE",
        }
        viols = cl.check_feasibility(action, context={})
        assert viols, "default must fail closed"
        assert viols[0].constraint_id == "li_fail_closed"
        # And a non-empty rules list with default fail_closed=True must
        # behave normally (no synthetic fail_closed violation).
        cl2 = LIConstraint(rules=[_rule_de()])
        ok_action = {
            "affected_ue_ids": ["ue-normal-1"],
            "affected_slice_ids": [],
            "target_jurisdiction": "DE",
        }
        viols2 = cl2.check_feasibility(ok_action, context={})
        assert viols2 == []


# ─── lagrangian — must always be zero ──────────────────────────────────


class TestLagrangianHardOnly:
    def test_lagrangian_returns_zeros_for_any_input(self):
        cl = LIConstraint(rules=[_rule_de(), _rule_uk()])
        for shape in [(1, 4), (8, 4), (16, 7), (3, 2)]:
            a = torch.randn(*shape)
            out = cl.lagrangian_violation(a, context={})
            assert out.shape == (shape[0],)
            assert torch.all(out == 0.0)
            # gradient must be zero too — LI must not enter the policy gradient
            assert not out.requires_grad or out.grad_fn is None or torch.all(out == 0)

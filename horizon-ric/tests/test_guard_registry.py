"""Guard registry — register / retrieve / chain.

Mirrors ``test_emit_guards_counterfactual.py`` but exercises the
registry-driven path used by the rApp lifecycle. Adding a new guard at
runtime must compose into the chain without code changes elsewhere.
"""

from __future__ import annotations

from horizon_ric.contracts.constraint_layer import ConstraintViolation
from horizon_ric.policy.constraints import (
    PreceptualAIConstraintConfig,
    PreceptualAIConstraintLayer,
)
from horizon_ric.policy.emit_guards import GuardFailure
from horizon_ric.policy.guard_registry import (
    GuardContext,
    GuardRegistry,
    list_guards,
    register_guard,
    run_guard_chain,
)


def test_builtin_guards_registered():
    """The four production guards must be present after import."""
    names = list_guards()
    for required in (
        "head_pretrained",
        "constraint_context_complete",
        "decision_within_a1_budget",
        "corrections_recorded",
    ):
        assert required in names, f"missing built-in guard {required!r}"


def test_clean_path_no_failures():
    """A correctly-shaped context must produce no failures."""
    layer = PreceptualAIConstraintLayer(PreceptualAIConstraintConfig())
    ctx = GuardContext(
        head_is_pretrained=True,
        constraint_context={},
        constraint_layer=layer,
        elapsed_ms=20.0,
        policy_period_ms=100.0,
        corrections=None,
        audit_corrections_field=None,
    )
    failures = run_guard_chain(ctx)
    assert failures == []


def test_each_guard_can_fail_independently():
    """Each built-in must surface its own GuardFailure when violated."""
    layer = PreceptualAIConstraintLayer(PreceptualAIConstraintConfig())

    # 1. untrained head
    ctx_a = GuardContext(head_is_pretrained=False, constraint_layer=layer)
    fa = run_guard_chain(ctx_a, only=["head_pretrained"])
    assert len(fa) == 1
    assert fa[0].guard_id == "sla_head_not_pretrained"

    # 2. budget exceeded
    ctx_b = GuardContext(elapsed_ms=200.0, policy_period_ms=100.0, constraint_layer=layer)
    fb = run_guard_chain(ctx_b, only=["decision_within_a1_budget"])
    assert len(fb) == 1
    assert fb[0].guard_id == "decision_over_budget"

    # 3. corrections applied but not audited
    corr = [ConstraintViolation("gso_pfd_floor", "hard", -1.0, "test")]
    ctx_c = GuardContext(
        corrections=corr,
        audit_corrections_field=None,
        constraint_layer=layer,
    )
    fc = run_guard_chain(ctx_c, only=["corrections_recorded"])
    assert len(fc) == 1
    assert fc[0].guard_id == "corrections_not_audited"


def test_register_custom_guard_invoked_in_chain():
    """A user-registered guard must run alongside the built-ins."""
    custom_called = {"count": 0}

    def my_guard(ctx: GuardContext) -> GuardFailure | None:
        custom_called["count"] += 1
        if ctx.extras and ctx.extras.get("must_fail"):
            return GuardFailure(
                guard_id="__test_custom__",
                message="custom guard refused",
            )
        return None

    register_guard("__test_custom__", my_guard)
    assert "__test_custom__" in list_guards()

    layer = PreceptualAIConstraintLayer(PreceptualAIConstraintConfig())
    ctx = GuardContext(
        head_is_pretrained=True,
        constraint_layer=layer,
        elapsed_ms=10.0,
        extras={"must_fail": True},
    )
    failures = run_guard_chain(ctx)
    assert custom_called["count"] >= 1
    assert any(f.guard_id == "__test_custom__" for f in failures)


def test_guard_exception_is_isolated():
    """A misbehaving guard must not crash the whole chain."""

    def boom(ctx: GuardContext) -> GuardFailure | None:
        raise RuntimeError("bug in plugin")

    reg = GuardRegistry()
    reg.register("explodes", boom)
    # Use module-level registration for this run.
    register_guard("__test_explodes__", boom)
    layer = PreceptualAIConstraintLayer(PreceptualAIConstraintConfig())
    ctx = GuardContext(constraint_layer=layer)
    failures = run_guard_chain(ctx, only=["__test_explodes__"])
    assert len(failures) == 1
    assert failures[0].guard_id.startswith("_guard_internal_error::")


def test_separate_registries_isolated():
    """Two registries should not share state."""
    r1 = GuardRegistry()
    r2 = GuardRegistry()
    r1.register("only_r1", lambda ctx: None)
    assert "only_r1" in r1.list_guards()
    assert "only_r1" not in r2.list_guards()
